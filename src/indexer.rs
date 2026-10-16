mod diagnostics;

pub use diagnostics::FileParseDiagnostics;
pub(crate) use diagnostics::load_parse_diagnostics;
use diagnostics::{cleanup_parse_diagnostics, save_parse_diagnostics, save_read_error};

use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs::{self, DirEntry};
use std::io;
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, Transaction, TransactionBehavior};
use tracing::{debug, info, warn};

use crate::chunker;

use crate::classify::classify_session;
use crate::index_observer::Observer;
use crate::parser::{Message, ParseResult, Role, Source, parse_session_including_empty};

const INDEX_RULES_VERSION: i64 = 2;

struct SessionEntry {
    session_id: String,
    source: Option<Source>,
    mtime: Option<f64>,
    file_size: Option<u64>,
    rules_version: Option<i64>,
}

pub struct IndexStats {
    pub indexed: usize,
    pub parse_diagnostics: Vec<FileParseDiagnostics>,
    pub total_sessions: usize,
    pub elapsed_secs: f64,
    pub skipped_roots: Vec<SkippedRoot>,
    pub preserved_embedded: usize,
}

#[derive(Debug, Clone)]
pub struct SkippedRoot {
    pub source: Source,
    pub preserved_sessions: usize,
    pub reason: SkippedReason,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkippedReason {
    MissingRoot,
    IncompleteEnumeration,
}

impl SkippedReason {
    pub(crate) fn note(self, source: &str, preserved: usize) -> String {
        match self {
            SkippedReason::MissingRoot => format!(
                "{source} root unavailable — preserved {preserved} existing session(s), none deleted this run; re-run `recall index` once the root is back so a full scan can reconcile any genuinely deleted sessions"
            ),
            SkippedReason::IncompleteEnumeration if preserved == 0 => format!(
                "{source} root could not be fully read; some sessions may be missing from the index. check the directory's permissions and access, then re-run `recall index`"
            ),
            SkippedReason::IncompleteEnumeration => format!(
                "{source} root could not be fully read — preserved {preserved} existing session(s), none deleted this run; check the directory's permissions and access, then re-run `recall index` so the next full scan can reconcile any genuinely deleted sessions"
            ),
        }
    }

    pub(crate) fn as_str(self) -> &'static str {
        match self {
            SkippedReason::MissingRoot => "missing_root",
            SkippedReason::IncompleteEnumeration => "incomplete_enumeration",
        }
    }
}

#[derive(Default)]
struct ReuseStats {
    matched: usize,
    embedded: usize,
    created: usize,
    removed: usize,
    seconds: f64,
}

enum IndexOutcome {
    Indexed(ReuseStats),
    Unchanged,
    Empty,
    Failed,
    Preserved(String),
}

struct IndexContext<'a> {
    tx: &'a Transaction<'a>,
    existing: &'a HashMap<String, SessionEntry>,
    force: bool,
    embedded_sessions: Option<HashSet<String>>,
}

pub(crate) struct IndexOptions<'a> {
    pub force: bool,
    pub claude_dir: &'a Path,
    pub codex_dir: &'a Path,
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct FileStamp {
    mtime: Option<SystemTime>,
    size: u64,
}

impl FileStamp {
    fn mtime_secs(self) -> Option<f64> {
        self.mtime?
            .duration_since(UNIX_EPOCH)
            .ok()
            .map(|d| d.as_secs_f64())
    }
}

fn resolve_file_stamp(fpath: &Path) -> Option<FileStamp> {
    let meta = fs::metadata(fpath).ok()?;
    Some(FileStamp {
        mtime: meta.modified().ok(),
        size: meta.len(),
    })
}

fn delete_session_dependents(tx: &Transaction, session_id: &str) -> Result<()> {
    tx.execute("DELETE FROM messages WHERE session_id = ?", [session_id])?;
    tx.execute(
        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ?)",
        [session_id],
    )?;
    tx.execute("DELETE FROM qa_chunks WHERE session_id = ?", [session_id])?;
    tx.execute(
        "DELETE FROM session_files WHERE session_id = ?",
        [session_id],
    )?;
    Ok(())
}

fn insert_session_files(tx: &Transaction, session_id: &str, paths: &[String]) -> Result<()> {
    let mut stmt =
        tx.prepare_cached("INSERT INTO session_files (session_id, path) VALUES (?1, ?2)")?;
    for path in paths {
        stmt.execute(rusqlite::params![session_id, path])?;
    }
    Ok(())
}

fn upsert_session(
    ctx: &IndexContext,
    fpath_str: &str,
    stamp: Option<FileStamp>,
    parsed: ParseResult,
) -> Result<ReuseStats> {
    let cached = ctx.existing.get(fpath_str);
    let session_id = &parsed.metadata.session_id;
    let reuse = !ctx.force
        && ctx.tx.query_row(
            "SELECT EXISTS(SELECT 1 FROM sessions WHERE session_id = ?1 AND chunks_indexed = ?2 AND EXISTS(SELECT 1 FROM qa_chunks WHERE session_id = ?1))",
            rusqlite::params![session_id, INDEX_RULES_VERSION],
            |r| r.get::<_, bool>(0),
        )?;
    if let Some(entry) = cached
        && entry.session_id != *session_id
    {
        ctx.tx.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            [&entry.session_id],
        )?;
        delete_session_dependents(ctx.tx, &entry.session_id)?;
    }
    if reuse {
        ctx.tx
            .execute("DELETE FROM messages WHERE session_id = ?", [session_id])?;
        ctx.tx.execute(
            "DELETE FROM session_files WHERE session_id = ?",
            [session_id],
        )?;
    } else {
        delete_session_dependents(ctx.tx, session_id)?;
    }

    let session_type = classify_session(
        parsed
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| m.text.as_str())
            .unwrap_or(""),
        parsed.origin.automated,
    )
    .as_str();

    ctx.tx.execute(
        "INSERT OR REPLACE INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type, files_scanned, file_size, chunks_indexed) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
        rusqlite::params![
            parsed.metadata.session_id,
            parsed.metadata.source.as_str(),
            parsed.metadata.file_path,
            parsed.metadata.project,
            parsed.metadata.slug,
            parsed.metadata.timestamp,
            stamp.and_then(FileStamp::mtime_secs),
            session_type,
            1,
            stamp.and_then(|s| i64::try_from(s.size).ok()),
            -INDEX_RULES_VERSION,
        ],
    )?;

    let mut msg_stmt = ctx
        .tx
        .prepare_cached("INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)")?;
    let mut messages = Vec::new();
    for msg in parsed.messages {
        msg_stmt.execute(rusqlite::params![
            parsed.metadata.session_id,
            msg.role.as_str(),
            msg.text
        ])?;
        if reuse {
            messages.push((ctx.tx.last_insert_rowid(), msg));
        }
    }
    let stats = if reuse {
        reconcile_chunks(
            ctx.tx,
            session_id,
            &messages,
            parsed.metadata.timestamp,
            ctx.embedded_sessions.is_some(),
        )?
    } else {
        ReuseStats::default()
    };

    insert_session_files(ctx.tx, &parsed.metadata.session_id, &parsed.scanned_files)?;

    Ok(stats)
}

fn reconcile_chunks(
    tx: &Transaction,
    session_id: &str,
    messages: &[(i64, Message)],
    timestamp: Option<i64>,
    known_unembedded: bool,
) -> Result<ReuseStats> {
    let start = Instant::now();
    let embedded: HashSet<i64> = if known_unembedded {
        HashSet::new()
    } else {
        let mut stmt = tx.prepare_cached("SELECT DISTINCT chunk_id FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ?)")?;
        stmt.query_map([session_id], |r| r.get(0))?
            .collect::<StdResult<_, _>>()?
    };
    let mut stored: HashMap<String, Vec<i64>> = HashMap::new();
    {
        let mut stmt = tx.prepare_cached(
            "SELECT id, content FROM qa_chunks WHERE session_id = ? ORDER BY id DESC",
        )?;
        let rows = stmt.query_map([session_id], |r| {
            Ok((r.get::<_, i64>(0)?, r.get::<_, String>(1)?))
        })?;
        for row in rows {
            let (id, content) = row?;
            stored.entry(content).or_default().push(id);
        }
    }
    for ids in stored.values_mut() {
        ids.sort_by_key(|id| embedded.contains(id));
    }
    tx.execute(
        "UPDATE qa_chunks SET src_rowid_lo = NULL, src_rowid_hi = NULL WHERE session_id = ?",
        [session_id],
    )?;
    let mut stats = ReuseStats::default();
    for chunk in chunker::chunk_messages(session_id, messages, timestamp) {
        if let Some(id) = stored.get_mut(&chunk.content).and_then(Vec::pop) {
            tx.execute("UPDATE qa_chunks SET timestamp = ?1, src_rowid_lo = ?2, src_rowid_hi = ?3 WHERE id = ?4",
                rusqlite::params![timestamp, chunk.src_rowid_lo, chunk.src_rowid_hi, id])?;
            stats.matched += 1;
            stats.embedded += usize::from(embedded.contains(&id));
        } else {
            tx.execute("INSERT INTO qa_chunks (session_id, content, timestamp, src_rowid_lo, src_rowid_hi) VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![session_id, chunk.content, timestamp, chunk.src_rowid_lo, chunk.src_rowid_hi])?;
            stats.created += 1;
        }
    }
    if stored.values().any(|ids| !ids.is_empty()) {
        if stored.values().flatten().any(|id| embedded.contains(id)) {
            tx.execute("DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ? AND src_rowid_lo IS NULL)", [session_id])?;
        }
        stats.removed = tx.execute(
            "DELETE FROM qa_chunks WHERE session_id = ? AND src_rowid_lo IS NULL",
            [session_id],
        )?;
    }

    tx.execute(
        "UPDATE sessions SET chunks_indexed = ?1 WHERE session_id = ?2",
        rusqlite::params![INDEX_RULES_VERSION, session_id],
    )?;
    stats.seconds = start.elapsed().as_secs_f64();
    Ok(stats)
}

fn is_unchanged(ctx: &IndexContext, fpath_str: &str, stamp: FileStamp) -> Result<bool> {
    if !ctx.force
        && let Some(new_mt) = stamp.mtime_secs()
        && let Some(entry) = ctx.existing.get(fpath_str)
        && let Some(old_mt) = entry.mtime
        && entry.rules_version == Some(INDEX_RULES_VERSION)
        && entry.file_size == Some(stamp.size)
        && (old_mt - new_mt).abs() < 0.001
        && !ctx.tx.prepare_cached(
            "SELECT EXISTS(SELECT 1 FROM parse_diagnostics WHERE file_path = ? AND read_error = 1)",
        )?.query_row([fpath_str], |row| row.get::<_, bool>(0))?
    {
        return Ok(true);
    }

    Ok(false)
}

fn index_file_with_parser(
    ctx: &IndexContext,
    fpath: &Path,
    source: &Source,
    parse: impl FnOnce(&Path, Source) -> Result<Option<ParseResult>>,
) -> Result<IndexOutcome> {
    let (Some(fpath_str), Some(before)) = (fpath.to_str(), resolve_file_stamp(fpath)) else {
        save_read_error(ctx.tx, fpath, *source)?;
        return Ok(IndexOutcome::Failed);
    };
    if is_unchanged(ctx, fpath_str, before)? {
        return Ok(IndexOutcome::Unchanged);
    }

    if let Some(embedded) = &ctx.embedded_sessions
        && let Some(entry) = ctx.existing.get(fpath_str)
        && embedded.contains(&entry.session_id)
    {
        return Ok(IndexOutcome::Preserved(entry.session_id.clone()));
    }

    let mut parsed = match parse(fpath, *source) {
        Ok(Some(p)) => p,
        Ok(None) => return Ok(IndexOutcome::Empty),
        Err(_) => {
            save_read_error(ctx.tx, fpath, *source)?;
            return Ok(IndexOutcome::Failed);
        }
    };

    save_parse_diagnostics(ctx.tx, fpath_str, *source, parsed.diagnostics)?;

    if parsed.messages.is_empty() {
        let Some(entry) = ctx.existing.get(fpath_str) else {
            return Ok(if parsed.diagnostics.is_empty() {
                IndexOutcome::Empty
            } else {
                IndexOutcome::Failed
            });
        };
        if !parsed.diagnostics.is_empty() {
            return Ok(IndexOutcome::Failed);
        }
        if parsed.origin.has_records && parsed.metadata.session_id != entry.session_id {
            return Ok(IndexOutcome::Failed);
        }
        parsed.metadata.session_id.clone_from(&entry.session_id);
    }

    if let Some(embedded) = &ctx.embedded_sessions
        && embedded.contains(&parsed.metadata.session_id)
    {
        return Ok(IndexOutcome::Preserved(parsed.metadata.session_id));
    }

    let stable_stamp = (resolve_file_stamp(fpath) == Some(before)).then_some(before);
    Ok(IndexOutcome::Indexed(upsert_session(
        ctx,
        fpath_str,
        stable_stamp,
        parsed,
    )?))
}

pub(crate) fn resolve_claude_dir_with<F>(home: &Path, get: F) -> PathBuf
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(dir) = get("RECALL_CLAUDE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = get("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir).join("projects");
    }
    home.join(".claude").join("projects")
}

pub(crate) fn resolve_claude_dir(home: &Path) -> PathBuf {
    resolve_claude_dir_with(home, |key| env::var_os(key))
}

pub(crate) fn resolve_codex_dir_with<F>(home: &Path, get: F) -> PathBuf
where
    F: Fn(&str) -> Option<OsString>,
{
    get("RECALL_CODEX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".codex").join("sessions"))
}

pub(crate) fn resolve_codex_dir(home: &Path) -> PathBuf {
    resolve_codex_dir_with(home, |key| env::var_os(key))
}

struct ScanOutcome {
    sources: Vec<(PathBuf, Source)>,
    scanned: HashSet<Source>,
    skipped: HashMap<Source, SkippedReason>,
}

fn collect_sources(opts: &IndexOptions) -> ScanOutcome {
    let mut out = ScanOutcome {
        sources: Vec::new(),
        scanned: HashSet::new(),
        skipped: HashMap::new(),
    };
    classify_root(opts.claude_dir, Source::Claude, &mut out);
    classify_root(opts.codex_dir, Source::Codex, &mut out);
    out
}

fn classify_root(dir: &Path, source: Source, out: &mut ScanOutcome) {
    if !dir.is_dir() {
        out.skipped.insert(source, SkippedReason::MissingRoot);
    } else if collect_jsonl_files(dir, &mut out.sources, source) {
        out.scanned.insert(source);
    } else {
        out.skipped
            .insert(source, SkippedReason::IncompleteEnumeration);
    }
}

struct IndexTotals {
    indexed: usize,
    reuse: ReuseStats,
    preserved_embedded: usize,
    preserved_sessions: HashSet<String>,
}

fn index_all(
    ctx: &IndexContext,
    sources: &[(PathBuf, Source)],
    observer: &Observer,
) -> Result<IndexTotals> {
    let stage = observer.stage("parse_fts");
    let mut unchanged = 0;
    let mut empty = 0;
    let mut failed = 0;
    let mut indexed = 0;
    let mut reuse = ReuseStats::default();
    let mut preserved_embedded = 0;
    let mut preserved_sessions = HashSet::new();

    for (position, (fpath, source)) in sources.iter().enumerate() {
        match index_file_with_parser(ctx, fpath, source, |path, source| {
            let start = Instant::now();
            let parsed = parse_session_including_empty(path, source);
            observer.add_seconds("parse", start.elapsed().as_secs_f64());
            parsed
        })? {
            IndexOutcome::Indexed(stats) => {
                indexed += 1;
                reuse.matched += stats.matched;
                reuse.embedded += stats.embedded;
                reuse.created += stats.created;
                reuse.removed += stats.removed;
                reuse.seconds += stats.seconds;
            }
            IndexOutcome::Unchanged => unchanged += 1,
            IndexOutcome::Empty => empty += 1,
            IndexOutcome::Failed => failed += 1,
            IndexOutcome::Preserved(session_id) => {
                preserved_embedded += 1;
                preserved_sessions.insert(session_id);
            }
        }
        observer.count("files_remaining", sources.len() - position - 1);
        observer.count("files_updated", indexed);
        observer.count("files_unchanged", unchanged);
        observer.count("files_empty", empty);
        observer.count("files_failed", failed);
        observer.count("files_deferred", preserved_embedded);
        stage.progress(position + 1, sources.len(), 0);
    }

    stage.finish("processed; awaiting FTS transaction commit");
    if sources.is_empty() {
        observer.count("files_updated", 0);
        observer.count("files_unchanged", 0);
        observer.count("files_empty", 0);
        observer.count("files_failed", 0);
        observer.count("files_deferred", 0);
    }
    Ok(IndexTotals {
        indexed,
        reuse,
        preserved_embedded,
        preserved_sessions,
    })
}

fn finalize_fts(conn: &mut Connection, indexed: usize, force: bool) -> Result<()> {
    if force || indexed >= 500 {
        conn.execute("INSERT INTO messages(messages) VALUES('optimize')", [])?;
    }
    conn.execute(
        "INSERT INTO messages(messages, rank) VALUES('automerge', 4)",
        [],
    )?;
    Ok(())
}

#[cfg(test)]
pub(crate) fn index_from_dirs(
    conn: &mut Connection,
    opts: &IndexOptions,
    embed_capable: bool,
) -> Result<IndexStats> {
    index_from_dirs_observed(conn, opts, embed_capable, &Observer::default())
}

pub(crate) fn index_from_dirs_observed(
    conn: &mut Connection,
    opts: &IndexOptions,
    embed_capable: bool,
    observer: &Observer,
) -> Result<IndexStats> {
    let start = Instant::now();

    let fts = observer.stage("fts_transaction");
    let tx = conn
        .transaction_with_behavior(TransactionBehavior::Immediate)
        .context("Failed to begin transaction")?;

    let enumeration = observer.stage("enumeration");
    let scan = collect_sources(opts);
    observer.count("files_discovered", scan.sources.len());
    observer.count("files_remaining", scan.sources.len());
    observer.count("files_updated_committed", 0);
    enumeration.finish("complete; unavailable roots reported separately");
    let sources = &scan.sources;
    let source_paths: HashSet<&Path> = sources.iter().map(|(p, _)| p.as_path()).collect();

    info!(count = sources.len(), "Found source files");

    let preparation = observer.stage("index_prepare");
    let existing = load_existing_sessions(&tx)?;
    let embedded_sessions = if embed_capable {
        None
    } else {
        Some(embedded_session_ids(&tx)?)
    };
    preparation.finish("complete");

    tx.execute(
        "INSERT INTO messages(messages, rank) VALUES('automerge', 0)",
        [],
    )?;

    let ctx = IndexContext {
        tx: &tx,
        existing: &existing,
        force: opts.force,
        embedded_sessions,
    };
    let totals = index_all(&ctx, sources, observer)?;
    cleanup_orphans(
        &tx,
        &existing,
        &source_paths,
        &scan.scanned,
        &totals.preserved_sessions,
    )?;
    let parse_diagnostics = cleanup_parse_diagnostics(&tx, &source_paths, &scan.scanned)?;
    tx.commit().context("Failed to commit transaction")?;

    observer.count("files_updated_committed", totals.indexed);
    observer.count("chunks_matched_committed", totals.reuse.matched);
    observer.count("embeddings_reused_committed", totals.reuse.embedded);
    observer.count("chunks_reconciled_created_committed", totals.reuse.created);
    observer.count("chunks_removed_committed", totals.reuse.removed);
    observer.add_seconds("chunk_reconciliation", totals.reuse.seconds);
    fts.finish("committed");
    observer.add_seconds(
        "fts_excluding_parse",
        (observer.seconds("fts_transaction") - observer.seconds("parse")).max(0.0),
    );
    let optimize = observer.stage("fts_finalize");
    finalize_fts(conn, totals.indexed, opts.force)?;
    optimize.finish("complete");

    let total_sessions = {
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        usize::try_from(n.max(0)).expect("non-negative session count fits in usize")
    };

    Ok(IndexStats {
        indexed: totals.indexed,
        parse_diagnostics,
        total_sessions,
        elapsed_secs: start.elapsed().as_secs_f64(),
        skipped_roots: summarize_skipped_roots(&existing, &source_paths, &scan.skipped),
        preserved_embedded: totals.preserved_embedded,
    })
}

fn embedded_session_ids(conn: &Connection) -> Result<HashSet<String>> {
    let embedded: HashSet<i64> = {
        let mut stmt = conn.prepare("SELECT DISTINCT chunk_id FROM vec_chunks")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        rows.collect::<StdResult<_, _>>()?
    };

    let mut stmt = conn.prepare("SELECT id, session_id FROM qa_chunks")?;
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        if embedded.contains(&id) {
            return Ok(Some(row.get::<_, String>(1)?));
        }
        Ok(None)
    })?;
    rows.filter_map(StdResult::transpose)
        .collect::<StdResult<_, _>>()
        .map_err(Into::into)
}

fn summarize_skipped_roots(
    existing: &HashMap<String, SessionEntry>,
    source_paths: &HashSet<&Path>,
    skipped: &HashMap<Source, SkippedReason>,
) -> Vec<SkippedRoot> {
    [Source::Claude, Source::Codex]
        .into_iter()
        .filter_map(|source| {
            let reason = *skipped.get(&source)?;
            let preserved_sessions = existing
                .iter()
                .filter(|(fp, e)| {
                    e.source == Some(source) && !source_paths.contains(Path::new(fp.as_str()))
                })
                .count();
            let emit = match reason {
                SkippedReason::MissingRoot => preserved_sessions > 0,
                SkippedReason::IncompleteEnumeration => true,
            };
            emit.then_some(SkippedRoot {
                source,
                preserved_sessions,
                reason,
            })
        })
        .collect()
}

fn load_existing_sessions(conn: &Connection) -> Result<HashMap<String, SessionEntry>> {
    let mut stmt = conn.prepare(
        "SELECT file_path, session_id, source, mtime, file_size, chunks_indexed FROM sessions",
    )?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<f64>>(3)?,
            row.get::<_, Option<i64>>(4)?,
            row.get::<_, Option<i64>>(5)?,
        ))
    })?;
    let mut map = HashMap::new();
    for row in rows {
        let (fp, sid, src, mt, size, rules_version) = row?;
        map.insert(
            fp,
            SessionEntry {
                session_id: sid,
                source: Source::from_db(&src),
                mtime: mt,
                file_size: size.and_then(|size| u64::try_from(size).ok()),
                rules_version,
            },
        );
    }
    Ok(map)
}

fn cleanup_orphans(
    tx: &Transaction,
    existing: &HashMap<String, SessionEntry>,
    source_paths: &HashSet<&Path>,
    scanned: &HashSet<Source>,
    preserved_sessions: &HashSet<String>,
) -> Result<()> {
    if existing.is_empty() {
        return Ok(());
    }
    for (fp, entry) in existing {
        // A missing/unreadable source root is not evidence that its sessions were deleted.
        let scanned_here = entry.source.is_some_and(|s| scanned.contains(&s));
        if !scanned_here || preserved_sessions.contains(&entry.session_id) {
            continue;
        }
        if !source_paths.contains(Path::new(fp.as_str())) {
            let deleted = tx.execute(
                "DELETE FROM sessions WHERE session_id = ? AND file_path = ?",
                rusqlite::params![entry.session_id, fp],
            )?;
            if deleted > 0 {
                delete_session_dependents(tx, &entry.session_id)?;
            }
        }
    }
    Ok(())
}

const MAX_DIR_DEPTH: usize = 10;

fn collect_jsonl_files(dir: &Path, out: &mut Vec<(PathBuf, Source)>, source: Source) -> bool {
    collect_jsonl_files_inner(dir, out, source, 0)
}

fn collect_jsonl_files_inner(
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    depth: usize,
) -> bool {
    if depth >= MAX_DIR_DEPTH {
        warn!(
            max_depth = MAX_DIR_DEPTH,
            path = %dir.display(),
            "depth limit reached"
        );
        return false;
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "cannot read dir");
            return false;
        }
    };
    collect_from_entries(entries, dir, out, source, depth)
}

fn collect_from_entries(
    entries: impl Iterator<Item = io::Result<DirEntry>>,
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    depth: usize,
) -> bool {
    let mut fully_read = true;
    for entry in entries {
        let (entry, ft) = match entry.and_then(|e| e.file_type().map(|ft| (e, ft))) {
            Ok(pair) => pair,
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "cannot read entry");
                fully_read = false;
                continue;
            }
        };
        if ft.is_symlink() {
            debug!(path = %entry.path().display(), "skipping symlink");
            continue;
        }
        let path = entry.path();
        if ft.is_dir() {
            fully_read &= collect_jsonl_files_inner(&path, out, source, depth + 1);
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push((path, source));
        }
    }
    fully_read
}

#[derive(Default)]
pub(crate) struct ChunkStats {
    pub chunks_created: usize,
    pub sessions_chunked: usize,
    pub message_batches: usize,
    pub message_scans: usize,
    pub messages_read: usize,
}

const CHUNK_BATCH_SESSIONS: usize = 64;
const CHUNK_BATCH_BYTES: i64 = 8 * 1024 * 1024;

const STAGE_CHUNK_MESSAGES: &str =
    "INSERT INTO chunk_message_rows (session_id, message_rowid, text_bytes)
     SELECT m.session_id, m.rowid, length(CAST(m.text AS BLOB))
     FROM messages m CROSS JOIN sessions s ON s.session_id = m.session_id
     WHERE s.chunks_indexed IS NULL OR s.chunks_indexed < 0";
const READ_CHUNK_BATCH: &str = "SELECT r.session_id, m.rowid, m.role, m.text
     FROM chunk_message_rows r CROSS JOIN messages m ON m.rowid = r.message_rowid
     WHERE r.session_id >= ?1 AND r.session_id <= ?2
     ORDER BY r.session_id, r.message_rowid";

type PendingChunkSession = (String, Option<i64>, i64);

fn chunk_batch_len(sessions: &[PendingChunkSession]) -> usize {
    let mut bytes = 0_i64;
    let mut count = 0;
    for (_, _, size) in sessions.iter().take(CHUNK_BATCH_SESSIONS) {
        if count > 0 && size.saturating_add(bytes) > CHUNK_BATCH_BYTES {
            break;
        }
        bytes = bytes.saturating_add(*size);
        count += 1;
    }
    count
}

#[cfg(test)]
pub(crate) fn index_chunks(
    conn: &mut Connection,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<ChunkStats> {
    index_chunks_observed(conn, on_progress, &Observer::default())
}

pub(crate) fn index_chunks_observed(
    conn: &mut Connection,
    on_progress: Option<&dyn Fn(usize, usize)>,
    observer: &Observer,
) -> Result<ChunkStats> {
    let stage = observer.stage("chunking");
    observer.count("sessions_chunked_committed", 0);
    observer.count("chunks_created_committed", 0);
    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
    let pending: bool = tx.query_row(
        "SELECT EXISTS(SELECT 1 FROM sessions WHERE chunks_indexed IS NULL OR chunks_indexed < 0)",
        [],
        |row| row.get(0),
    )?;
    if !pending {
        observer.count("sessions_chunk_pending", 0);
        observer.count("sessions_chunk_remaining", 0);
        observer.count("sessions_chunked_empty", 0);
        stage.finish("no pending sessions");
        return Ok(ChunkStats::default());
    }

    tx.execute_batch(
        "CREATE TEMP TABLE chunk_message_rows (
            session_id TEXT NOT NULL,
            message_rowid INTEGER NOT NULL,
            text_bytes INTEGER NOT NULL,
            PRIMARY KEY (session_id, message_rowid)
        ) WITHOUT ROWID;",
    )?;
    tx.execute(STAGE_CHUNK_MESSAGES, [])?;
    let sessions: Vec<PendingChunkSession> = {
        let mut stmt = tx.prepare(
            "SELECT s.session_id, s.timestamp, COALESCE(SUM(r.text_bytes), 0)
             FROM sessions s LEFT JOIN chunk_message_rows r ON r.session_id = s.session_id
             WHERE s.chunks_indexed IS NULL OR s.chunks_indexed < 0
             GROUP BY s.session_id ORDER BY s.session_id",
        )?;
        let rows = stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?)))?;
        rows.collect::<StdResult<_, _>>()?
    };
    let total = sessions.len();
    observer.count("sessions_chunk_pending", total);
    observer.count("sessions_chunk_remaining", total);
    let mut empty = 0;
    info!(count = total, "Chunking sessions");

    let mut stats = ChunkStats {
        message_scans: 1,
        ..ChunkStats::default()
    };
    let mut remaining = sessions.as_slice();
    while !remaining.is_empty() {
        let (batch, rest) = remaining.split_at(chunk_batch_len(remaining));
        let mut messages: HashMap<String, Vec<(i64, Message)>> = HashMap::new();
        {
            let mut stmt = tx.prepare_cached(READ_CHUNK_BATCH)?;
            let mut rows = stmt.query(rusqlite::params![batch[0].0, batch[batch.len() - 1].0])?;
            while let Some(row) = rows.next()? {
                let session_id: String = row.get(0)?;
                let rowid = row.get(1)?;
                let role_str: String = row.get(2)?;
                let text = row.get(3)?;
                stats.messages_read += 1;
                let Some(role) = Role::from_db(&role_str) else {
                    debug!(role = %role_str, session_id, "unknown role");
                    continue;
                };
                messages
                    .entry(session_id)
                    .or_default()
                    .push((rowid, Message { role, text }));
            }
        }
        stats.message_batches += 1;
        for (session_id, timestamp, _) in batch {
            let body = messages.remove(session_id).unwrap_or_default();
            let chunks = chunker::chunk_messages(session_id, &body, *timestamp);
            if chunks.is_empty() {
                empty += 1;
            }
            for chunk in &chunks {
                tx.execute(
                    "INSERT INTO qa_chunks (session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
                     VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![chunk.session_id, chunk.content, chunk.timestamp,
                                      chunk.src_rowid_lo, chunk.src_rowid_hi],
                )?;
                stats.chunks_created += 1;
            }
            tx.execute(
                "UPDATE sessions SET chunks_indexed = CASE WHEN chunks_indexed = -?1 THEN ?1 ELSE 0 END WHERE session_id = ?2",
                rusqlite::params![INDEX_RULES_VERSION, session_id],
            )?;
            stats.sessions_chunked += 1;
            stage.progress(stats.sessions_chunked, total, 0);
            if let Some(cb) = on_progress {
                cb(stats.sessions_chunked, total);
            }
        }
        remaining = rest;
    }
    tx.execute_batch("DROP TABLE chunk_message_rows;")?;
    tx.commit()?;
    observer.count("sessions_chunk_remaining", 0);
    observer.count("sessions_chunked_committed", stats.sessions_chunked);
    observer.count("sessions_chunked_empty", empty);
    observer.count("chunks_created_committed", stats.chunks_created);
    stage.finish("committed");
    debug!(
        sessions = stats.sessions_chunked,
        batches = stats.message_batches,
        scans = stats.message_scans,
        messages = stats.messages_read,
        "Chunking complete"
    );
    Ok(stats)
}

fn read_session_messages(tx: &Transaction, session_id: &str) -> Result<Vec<(i64, Message)>> {
    let mut stmt = tx.prepare_cached(
        "SELECT rowid, role, text FROM messages WHERE session_id = ? ORDER BY rowid",
    )?;
    let rows = stmt.query_map([session_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let mut msgs = Vec::new();
    for r in rows {
        let (rowid, role_str, text) = r?;
        let Some(role) = Role::from_db(&role_str) else {
            debug!(role = %role_str, session_id, "unknown role");
            continue;
        };
        msgs.push((rowid, Message { role, text }));
    }
    Ok(msgs)
}

pub(crate) fn backfill_rowid_ranges(conn: &mut Connection) -> Result<usize> {
    let sessions: Vec<(String, Option<i64>)> = {
        let null_row_sessions_sql = "SELECT DISTINCT s.session_id, s.timestamp FROM sessions s \
             JOIN qa_chunks c ON c.session_id = s.session_id \
             WHERE c.src_rowid_lo IS NULL";
        let mut stmt = conn.prepare(null_row_sessions_sql)?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
        })?;
        rows.collect::<StdResult<Vec<_>, _>>()?
    };

    if sessions.is_empty() {
        return Ok(0);
    }

    let tx = conn.transaction()?;
    let mut backfilled = 0;
    for (session_id, timestamp) in &sessions {
        let messages = read_session_messages(&tx, session_id)?;
        let chunks = chunker::chunk_messages(session_id, &messages, *timestamp);

        let stored_chunks: Vec<(i64, String)> = {
            let mut stmt = tx.prepare_cached(
                "SELECT id, content FROM qa_chunks WHERE session_id = ? ORDER BY id",
            )?;
            let rows = stmt.query_map([session_id], |row| {
                Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
            })?;
            rows.collect::<StdResult<Vec<_>, _>>()?
        };

        let (stored, rederived) = (stored_chunks.len(), chunks.len());
        if stored != rederived {
            debug!(%session_id, stored, rederived, "rowid backfill skipped: count mismatch");
            continue;
        }
        if stored_chunks
            .iter()
            .zip(chunks.iter())
            .any(|((_, stored_content), chunk)| *stored_content != chunk.content)
        {
            debug!(%session_id, "rowid backfill skipped: content mismatch");
            continue;
        }

        for ((id, _), chunk) in stored_chunks.iter().zip(chunks.iter()) {
            tx.execute(
                "UPDATE qa_chunks SET src_rowid_lo = ?1, src_rowid_hi = ?2 WHERE id = ?3",
                rusqlite::params![chunk.src_rowid_lo, chunk.src_rowid_hi, id],
            )?;
        }
        backfilled += 1;
    }

    tx.commit()?;

    Ok(backfilled)
}

enum RereadOutcome {
    Paths(Vec<String>),
    Unavailable,
}

fn reread_scanned_files(tx: &Transaction, fpath: &Path, source: &str) -> Result<RereadOutcome> {
    let Some(source) = Source::from_db(source) else {
        return Ok(RereadOutcome::Paths(Vec::new()));
    };
    match parse_session_including_empty(fpath, source) {
        Ok(Some(parsed)) => {
            if !parsed.diagnostics.is_empty() {
                save_parse_diagnostics(tx, &fpath.to_string_lossy(), source, parsed.diagnostics)?;
            }
            Ok(RereadOutcome::Paths(parsed.scanned_files))
        }
        Ok(None) => Ok(RereadOutcome::Paths(Vec::new())),
        Err(_) => {
            save_read_error(tx, fpath, source)?;
            Ok(RereadOutcome::Unavailable)
        }
    }
}

pub(crate) fn backfill_session_files(
    conn: &mut Connection,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<usize> {
    let sessions: Vec<(String, String, String)> = {
        let unmarked_sessions_sql =
            "SELECT session_id, file_path, source FROM sessions WHERE files_scanned IS NULL";
        let mut stmt = conn.prepare(unmarked_sessions_sql)?;
        let rows = stmt.query_map([], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
            ))
        })?;
        rows.collect::<StdResult<Vec<_>, _>>()?
    };

    if sessions.is_empty() {
        return Ok(0);
    }

    let total = sessions.len();
    let mut marked = 0;
    let tx = conn.transaction()?;
    for (done, (session_id, file_path, source)) in sessions.iter().enumerate() {
        if let Some(cb) = on_progress {
            cb(done + 1, total);
        }
        let scanned = match reread_scanned_files(&tx, Path::new(file_path), source)? {
            RereadOutcome::Paths(paths) => paths,
            RereadOutcome::Unavailable => continue,
        };
        insert_session_files(&tx, session_id, &scanned)?;
        tx.prepare_cached("UPDATE sessions SET files_scanned = 1 WHERE session_id = ?")?
            .execute([session_id])?;
        marked += 1;
    }
    tx.commit()?;

    Ok(marked)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod benchmarks {
    use super::*;
    use crate::db::setup_test_db;

    fn legacy_chunk_pass(conn: &mut Connection) -> ChunkStats {
        let sessions: Vec<(String, Option<i64>)> = conn
            .prepare("SELECT s.session_id, s.timestamp FROM sessions s WHERE NOT EXISTS (SELECT 1 FROM qa_chunks c WHERE c.session_id = s.session_id)")
            .unwrap().query_map([], |r| Ok((r.get(0)?, r.get(1)?))).unwrap()
            .map(Result::unwrap).collect();
        let tx = conn.transaction().unwrap();
        let mut stats = ChunkStats::default();
        for (id, timestamp) in sessions {
            let messages = read_session_messages(&tx, &id).unwrap();
            stats.sessions_chunked += 1;
            stats.message_scans += 1;
            stats.message_batches += 1;
            stats.messages_read += messages.len();
            for chunk in chunker::chunk_messages(&id, &messages, timestamp) {
                tx.execute("INSERT INTO qa_chunks (session_id, content, timestamp, src_rowid_lo, src_rowid_hi) VALUES (?1, ?2, ?3, ?4, ?5)",
                    rusqlite::params![chunk.session_id, chunk.content, chunk.timestamp, chunk.src_rowid_lo, chunk.src_rowid_hi]).unwrap();
                stats.chunks_created += 1;
            }
        }
        tx.commit().unwrap();
        stats
    }

    type StoredChunk = (String, String, Option<i64>, i64, i64);
    fn stored_chunks(conn: &Connection) -> Vec<StoredChunk> {
        conn.prepare("SELECT session_id, content, timestamp, src_rowid_lo, src_rowid_hi FROM qa_chunks ORDER BY session_id, id").unwrap()
            .query_map([], |r| Ok((r.get(0)?, r.get(1)?, r.get(2)?, r.get(3)?, r.get(4)?)))
            .unwrap().map(Result::unwrap).collect()
    }

    fn timed_pass(conn: &mut Connection, legacy: bool, scenario: &str, run: usize) -> ChunkStats {
        let start = Instant::now();
        let stats = if legacy {
            legacy_chunk_pass(conn)
        } else {
            index_chunks(conn, None).unwrap()
        };
        let elapsed = start.elapsed().as_secs_f64();
        println!(
            "{scenario} run={run} legacy={legacy} seconds={elapsed:.6} sessions={} scans={} batches={} messages={} chunks={}",
            stats.sessions_chunked,
            stats.message_scans,
            stats.message_batches,
            stats.messages_read,
            stats.chunks_created
        );
        stats
    }

    #[test]
    #[ignore = "manual 20,580-session benchmark; no timing assertion"]
    fn chunk_index_scale() {
        let (_dir, mut conn) = setup_test_db();
        let tx = conn.transaction().unwrap();
        for i in 0..20_580 {
            let id = format!("s{i:05}");
            tx.execute(
                "INSERT INTO sessions (session_id, timestamp) VALUES (?1, 123)",
                [&id],
            )
            .unwrap();
            let count = if i < 61 {
                9 + i32::from(i < 40)
            } else {
                11 + i32::from(i - 61 < 19_459)
            };
            for j in 0..count {
                let role = if i >= 61 && j == 0 {
                    "user"
                } else {
                    "assistant"
                };
                tx.execute(
                    "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
                    rusqlite::params![
                        id,
                        role,
                        format!("synthetic conversation {i} message {j}: example indexing text")
                    ],
                )
                .unwrap();
            }
        }
        tx.commit().unwrap();
        let (messages, bytes): (i64, i64) = conn
            .query_row(
                "SELECT count(*), sum(length(CAST(text AS BLOB))) FROM messages",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .unwrap();
        assert_eq!(messages, 245_757);
        println!(
            "seed sessions=20580 messages={messages} text_bytes={bytes} sqlite={}",
            rusqlite::version()
        );
        let initial = index_chunks(&mut conn, None).unwrap();
        assert_eq!(
            (
                initial.sessions_chunked,
                initial.chunks_created,
                initial.messages_read,
                initial.message_scans
            ),
            (20_580, 20_519, 245_757, 1)
        );
        let expected = stored_chunks(&conn);
        for run in 0..3 {
            for legacy in if run % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                let stats = timed_pass(&mut conn, legacy, "unchanged", run);
                assert_eq!(
                    (
                        stats.sessions_chunked,
                        stats.message_scans,
                        stats.messages_read
                    ),
                    if legacy { (61, 61, 589) } else { (0, 0, 0) }
                );
                assert_eq!(stats.chunks_created, 0);
                assert_eq!(stored_chunks(&conn), expected);
            }
        }
        conn.execute("INSERT INTO messages (session_id, role, text) VALUES ('s00000', 'user', 'appended question')", []).unwrap();
        let appended_rowid = conn.last_insert_rowid();
        let mut appended_expected = expected;
        appended_expected.insert(
            0,
            (
                "s00000".to_owned(),
                "appended question".to_owned(),
                Some(123),
                appended_rowid,
                appended_rowid,
            ),
        );
        for run in 0..3 {
            for legacy in if run % 2 == 0 {
                [true, false]
            } else {
                [false, true]
            } {
                conn.execute_batch("DELETE FROM qa_chunks WHERE session_id = 's00000'; UPDATE sessions SET chunks_indexed = NULL WHERE session_id = 's00000';").unwrap();
                let stats = timed_pass(&mut conn, legacy, "append", run);
                assert_eq!(
                    (
                        stats.sessions_chunked,
                        stats.message_scans,
                        stats.messages_read,
                        stats.chunks_created
                    ),
                    if legacy {
                        (61, 61, 590, 1)
                    } else {
                        (1, 1, 11, 1)
                    }
                );
                assert_eq!(stored_chunks(&conn), appended_expected);
            }
        }
    }
}
