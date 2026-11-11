mod backfill;
mod chunks;
mod diagnostics;
mod scan;

pub(crate) use backfill::{backfill_rowid_ranges, backfill_session_files};
#[cfg(test)]
pub(crate) use chunks::index_chunks;
pub(crate) use chunks::index_chunks_observed;
#[cfg(test)]
use chunks::{
    CHUNK_BATCH_BYTES, CHUNK_BATCH_SESSIONS, READ_CHUNK_BATCH, STAGE_CHUNK_MESSAGES,
    chunk_batch_len,
};
pub use diagnostics::FileParseDiagnostics;
pub(crate) use diagnostics::load_parse_diagnostics;
use diagnostics::{cleanup_parse_diagnostics, save_parse_diagnostics, save_read_error};
use scan::{RemovalGate, collect_sources};
#[cfg(test)]
use scan::{
    collect_from_entries, resolve_claude_dir_with, resolve_codex_archived_dir_with,
    resolve_codex_dir_with,
};
pub(crate) use scan::{resolve_claude_dir, resolve_codex_archived_dir, resolve_codex_dir};

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::time::{Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, Transaction, TransactionBehavior};
use tracing::info;

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
    /// Another root of the same source was read in full, so its deleted sessions were removed.
    pub other_roots_reconciled: bool,
}

impl SkippedRoot {
    pub(crate) fn note(&self) -> String {
        let source = self.source.as_str();
        if self.other_roots_reconciled && self.reason == SkippedReason::MissingRoot {
            return format!(
                "a {source} root is unavailable — preserved {preserved} existing session(s) under it and reconciled the rest; re-run `recall index` once the root is back so a full scan can reconcile them",
                preserved = self.preserved_sessions,
            );
        }
        self.reason.note(source, self.preserved_sessions)
    }
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
                "{source} root could not be fully read; some sessions may be missing from the index. check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index`"
            ),
            SkippedReason::IncompleteEnumeration => format!(
                "{source} root could not be fully read — preserved {preserved} existing session(s), none deleted this run; check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index` so the next full scan can reconcile any genuinely deleted sessions"
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
    pub codex_archived_dir: Option<&'a Path>,
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
    let source_paths: HashSet<&Path> = sources
        .iter()
        .map(|(p, _)| p.as_path())
        .chain(scan.set_aside.iter().map(PathBuf::as_path))
        .collect();

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
        &scan.removal,
        &totals.preserved_sessions,
    )?;
    let parse_diagnostics = cleanup_parse_diagnostics(&tx, &source_paths, &scan.removal)?;
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
        skipped_roots: summarize_skipped_roots(
            &existing,
            &source_paths,
            &scan.skipped,
            &scan.removal,
        ),
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
    removal: &RemovalGate,
) -> Vec<SkippedRoot> {
    [Source::Claude, Source::Codex]
        .into_iter()
        .filter_map(|source| {
            let reason = *skipped.get(&source)?;
            let preserved_sessions = existing
                .iter()
                .filter(|(fp, e)| {
                    let fp = Path::new(fp.as_str());
                    e.source == Some(source)
                        && !source_paths.contains(fp)
                        && !removal.allows(e.source, fp)
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
                other_roots_reconciled: removal.is_fully_read(source),
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
    removal: &RemovalGate,
    preserved_sessions: &HashSet<String>,
) -> Result<()> {
    if existing.is_empty() {
        return Ok(());
    }
    for (fp, entry) in existing {
        // A missing/unreadable source root is not evidence that its sessions were deleted.
        if !removal.allows(entry.source, Path::new(fp.as_str()))
            || preserved_sessions.contains(&entry.session_id)
        {
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

#[cfg(test)]
mod tests;

#[cfg(test)]
mod benchmarks;
