use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs::{self, DirEntry};
use std::io;
use std::path::{Path, PathBuf};
use std::result::Result as StdResult;
use std::time::{Instant, UNIX_EPOCH};

use anyhow::{Context, Result};
use rusqlite::{Connection, Transaction};
use tracing::{debug, info, warn};

use crate::chunker;
use crate::classify::classify_first_turn;
use crate::parser::{
    Message, ParseResult, Role, Source, parse_claude_session, parse_codex_session,
};

struct SessionEntry {
    session_id: String,
    /// `None` when the stored source string is unrecognized — such a row is
    /// preserved by orphan cleanup since its origin root can't be confirmed scanned.
    source: Option<Source>,
    mtime: Option<f64>,
}

pub struct IndexStats {
    pub indexed: usize,
    pub parse_errors: usize,
    pub first_error: Option<String>,
    pub total_sessions: usize,
    pub elapsed_secs: f64,
    /// Sources whose root was not scanned this run yet still hold sessions in
    /// the DB. Those sessions were preserved (not cleaned) because the missing
    /// root can't prove their files were deleted (#165). Surfaced so a silent
    /// preservation — a root outage masking real deletions — stays visible.
    pub skipped_roots: Vec<SkippedRoot>,
    /// Embedded sessions left untouched because the model was absent at index
    /// time (#215). Re-indexing them would delete their chunks/embeddings before
    /// the embed gate could rebuild them, dropping semantic search to degraded;
    /// preserving them keeps search intact and lets a later model-present run
    /// re-embed any content that changed meanwhile. Surfaced so the skip is
    /// visible, not silent.
    pub preserved_embedded: usize,
}

#[derive(Debug, Clone)]
pub struct SkippedRoot {
    pub source: Source,
    pub preserved_sessions: usize,
    pub reason: SkippedReason,
}

/// Why a source root was skipped this run, so the operator-facing surface can
/// give the right remediation instead of one fixed message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkippedReason {
    /// Root path does not exist as a directory (#165/#177). Remedy: restore the root.
    MissingRoot,
    /// Root exists but the tree could not be fully enumerated — a permission/access
    /// failure on the root or a child, a read_dir error, or a depth-limit truncation
    /// (#181). Remedy: fix the directory's permissions/access, not wait for the root.
    IncompleteEnumeration,
}

impl SkippedReason {
    // pub(crate): main.rs (a different module) calls both methods to render the surface.
    pub(crate) fn note(self, source: &str, preserved: usize) -> String {
        match self {
            // Like IncompleteEnumeration, a missing root never enters `scanned`,
            // so cleanup_orphans skips it and nothing is deleted this run; the
            // reconcile is forward-looking, deferred to a later full scan (#187).
            SkippedReason::MissingRoot => format!(
                "{source} root unavailable — preserved {preserved} existing session(s), none deleted this run; re-run `recall index` once the root is back so a full scan can reconcile any genuinely deleted sessions"
            ),
            // With nothing preserved (e.g. a first index where the partial read
            // means some files were never seen), the "preserved N / reconcile
            // deletions" framing is self-contradictory, so drop it (#185).
            SkippedReason::IncompleteEnumeration if preserved == 0 => format!(
                "{source} root could not be fully read; some sessions may be missing from the index. check the directory's permissions and access, then re-run `recall index`"
            ),
            // preserved>0: the reconcile is forward-looking. Nothing is deleted
            // this run (the source never enters `scanned`, so cleanup_orphans
            // skips it); the wording points at the next full scan so it does not
            // read as deletions that already happened (#187).
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

impl IndexStats {
    pub fn parse_error_detail(&self) -> Option<String> {
        self.first_error
            .as_ref()
            .map(|err| format!("Failed to parse {} files — {err}", self.parse_errors))
    }
}

enum IndexOutcome {
    Indexed,
    Unchanged,
    /// A re-index was due, but the session already has embeddings and the model
    /// is absent this run, so its rows were left untouched to protect them (#215).
    Preserved,
    ParseError(String),
}

struct IndexContext<'a> {
    tx: &'a Transaction<'a>,
    existing: &'a HashMap<String, SessionEntry>,
    /// Force re-index: `check_freshness` must not skip an mtime-unchanged file,
    /// so every present file is re-parsed and its chunks/embeddings rebuilt.
    force: bool,
    /// Session ids that already hold embeddings, populated only when the model is
    /// absent this run (#215). `Some` activates preserve-in-place: a re-index of
    /// a session in this set is skipped so its chunks/embeddings survive instead
    /// of being deleted before an embed gate that cannot rebuild them. `None`
    /// (model present) leaves the normal delete-and-reindex path unchanged.
    embedded_sessions: Option<HashSet<String>>,
}

pub(crate) struct IndexOptions<'a> {
    pub force: bool,
    pub claude_dir: &'a Path,
    pub codex_dir: &'a Path,
}

enum Mtime {
    Value(f64),
    /// File exists but mtime unavailable — index without freshness check.
    Unknown,
    /// File cannot be stat'd — skip entirely.
    Inaccessible,
}

fn resolve_mtime(fpath: &Path) -> Mtime {
    let meta = match fs::metadata(fpath) {
        Ok(m) => m,
        Err(_) => {
            debug!(path = %fpath.display(), "cannot stat");
            return Mtime::Inaccessible;
        }
    };
    let mtime = match meta.modified() {
        Ok(t) => t,
        Err(e) => {
            debug!(path = %fpath.display(), error = %e, "mtime unavailable");
            return Mtime::Unknown;
        }
    };
    match mtime.duration_since(UNIX_EPOCH) {
        Ok(d) => Mtime::Value(d.as_secs_f64()),
        Err(e) => {
            debug!(path = %fpath.display(), error = %e, "mtime before epoch");
            Mtime::Unknown
        }
    }
}

/// Delete all dependent data for a session (messages, chunks, embeddings, and
/// scanned-file markers). Removing the session_files rows here keeps a session
/// deletion / re-parse from leaving orphaned scanned-file markers behind.
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

/// Persist a session's write-target paths to `session_files`. Callers are
/// responsible for clearing any pre-existing rows under `session_id` first (the
/// `(session_id, path)` PK would otherwise collide on a re-parse / re-backfill),
/// so a plain INSERT suffices here. Shared by `upsert_session` (fresh parse) and
/// `backfill_session_files` (path-only re-read of a legacy session).
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
    mtime: Option<f64>,
    parsed: &ParseResult,
) -> Result<()> {
    let cached = ctx.existing.get(fpath_str);

    if let Some(entry) = cached {
        ctx.tx.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            [&entry.session_id],
        )?;
        delete_session_dependents(ctx.tx, &entry.session_id)?;
    }

    // New session_id differs from cached — clean up any orphaned messages under the new ID
    // (e.g. left over from a renamed file that previously used it).
    let same_session_id =
        matches!(cached, Some(entry) if entry.session_id == parsed.metadata.session_id);
    if !same_session_id {
        ctx.tx.execute(
            "DELETE FROM messages WHERE session_id = ?",
            [&parsed.metadata.session_id],
        )?;
        // Clear any scanned-file markers a prior file under this id left behind, so
        // the fresh inserts below cannot collide on the (session_id, path) PK.
        ctx.tx.execute(
            "DELETE FROM session_files WHERE session_id = ?",
            [&parsed.metadata.session_id],
        )?;
    }

    // Classify from the first user turn so `recall search` can exclude automated
    // sessions by default (#24). Done at ingest from the already-parsed messages —
    // no extra read. No user turn → interactive (never hidden by default).
    let session_type = classify_first_turn(
        parsed
            .messages
            .iter()
            .find(|m| matches!(m.role, Role::User))
            .map(|m| m.text.as_str())
            .unwrap_or(""),
    )
    .as_str();

    // files_scanned = 1 marks that scanned-file extraction ran for this session,
    // distinct from NULL (never recorded). Set on every upsert, so the re-parse
    // path — where the DELETE FROM sessions above drops the marker — re-raises it.
    ctx.tx.execute(
        "INSERT OR REPLACE INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type, files_scanned) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        rusqlite::params![
            parsed.metadata.session_id,
            parsed.metadata.source.as_str(),
            parsed.metadata.file_path,
            parsed.metadata.project,
            parsed.metadata.slug,
            parsed.metadata.timestamp,
            mtime,
            session_type,
            1,
        ],
    )?;

    if parsed.skipped_lines > 0 {
        debug!(
            skipped_lines = parsed.skipped_lines,
            path = fpath_str,
            "skipped lines during parse"
        );
    }

    let mut msg_stmt = ctx
        .tx
        .prepare_cached("INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)")?;
    for msg in &parsed.messages {
        msg_stmt.execute(rusqlite::params![
            parsed.metadata.session_id,
            msg.role.as_str(),
            msg.text
        ])?;
    }

    // Persist the session's write-target paths (precondition upheld by the
    // session_files DELETE above).
    insert_session_files(ctx.tx, &parsed.metadata.session_id, &parsed.scanned_files)?;

    Ok(())
}

fn check_freshness(ctx: &IndexContext, fpath: &Path) -> Option<(String, Option<f64>)> {
    let fpath_str = match fpath.to_str() {
        Some(s) => s.to_owned(),
        None => {
            debug!(path = %fpath.display(), "skipping non-UTF-8 path");
            return None;
        }
    };

    let mtime = match resolve_mtime(fpath) {
        Mtime::Value(v) => Some(v),
        Mtime::Unknown => None,
        Mtime::Inaccessible => return None,
    };

    // Epsilon 0.001s tolerates filesystem mtime rounding (HFS+ 1s, ext4 nano→f64).
    // Force bypasses the skip so a `rebuild` re-parses every present file even
    // when its mtime is unchanged.
    if !ctx.force
        && let Some(new_mt) = mtime
        && let Some(entry) = ctx.existing.get(&fpath_str)
        && let Some(old_mt) = entry.mtime
        && (old_mt - new_mt).abs() < 0.001
    {
        return None;
    }

    Some((fpath_str, mtime))
}

fn index_file(ctx: &IndexContext, fpath: &Path, source: &Source) -> Result<IndexOutcome> {
    let Some((fpath_str, mtime)) = check_freshness(ctx, fpath) else {
        return Ok(IndexOutcome::Unchanged);
    };

    // Preserve-in-place (#215): when the model is absent, a re-index of a session
    // that already holds embeddings must not run — `upsert_session` would delete
    // its chunks/embeddings before the embed gate, which cannot rebuild them this
    // run, leaving search degraded. Skip before parsing so its rows (and mtime)
    // stay untouched; a later model-present run re-indexes it (mtime drives the
    // self-heal for changed files, the embed gate for unchanged ones). A cached
    // session without embeddings, or a new session, is not in the set and falls
    // through to the normal FTS-only index.
    if let Some(embedded) = &ctx.embedded_sessions
        && let Some(entry) = ctx.existing.get(&fpath_str)
        && embedded.contains(&entry.session_id)
    {
        return Ok(IndexOutcome::Preserved);
    }

    let parsed = match parse_session(fpath, *source) {
        Ok(Some(p)) => p,
        Ok(None) => return Ok(IndexOutcome::Unchanged),
        Err(e) => {
            warn!(path = %fpath.display(), error = %e, "parse failed");
            return Ok(IndexOutcome::ParseError(format!(
                "{}: {e}",
                fpath.display()
            )));
        }
    };

    upsert_session(ctx, &fpath_str, mtime, &parsed)?;
    Ok(IndexOutcome::Indexed)
}

/// Priority: `RECALL_CLAUDE_DIR` > `CLAUDE_CONFIG_DIR/projects` > `~/.claude/projects`
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

/// The outcome of scanning the source roots: the discovered files, the sources
/// whose root was fully enumerated (`scanned` — only these are eligible for
/// orphan cleanup so a missing/unreadable root never has its sessions deleted,
/// #165), and the sources that were skipped with why (`skipped`).
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

/// Classifies one root: missing (no dir), scanned (fully enumerated), or
/// incompletely enumerated. A skipped root records why so the surface can give
/// the matching remedy. `classify_root` dedups the two call sites (claude/codex).
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
    parse_errors: usize,
    first_error: Option<String>,
    preserved_embedded: usize,
}

fn index_all(ctx: &IndexContext, sources: &[(PathBuf, Source)]) -> Result<IndexTotals> {
    let mut indexed = 0;
    let mut parse_errors = 0;
    let mut first_error = None;
    let mut preserved_embedded = 0;

    for (fpath, source) in sources {
        match index_file(ctx, fpath, source)? {
            IndexOutcome::Indexed => indexed += 1,
            IndexOutcome::Unchanged => {}
            IndexOutcome::Preserved => preserved_embedded += 1,
            IndexOutcome::ParseError(msg) => {
                parse_errors += 1;
                if first_error.is_none() {
                    first_error = Some(msg);
                }
            }
        }
    }

    Ok(IndexTotals {
        indexed,
        parse_errors,
        first_error,
        preserved_embedded,
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

/// `embed_capable` is whether the embedder loaded this run. When false (model
/// absent), sessions that already hold embeddings are preserved in place rather
/// than re-indexed (#215), since the embed gate downstream cannot rebuild what a
/// re-index would delete. The caller loads the embedder once and passes the
/// result here so the model is not probed twice.
pub(crate) fn index_from_dirs(
    conn: &mut Connection,
    opts: &IndexOptions,
    embed_capable: bool,
) -> Result<IndexStats> {
    let start = Instant::now();

    // Always full-scan: file-level mtime checks in check_freshness keep indexing
    // incremental. A directory-mtime skip optimization here used to miss new files
    // added to existing deep dirs (Codex Y/M/D, Claude subagents) — #52 / #70.
    //
    // `existing` is loaded even under force so deletion flows through the
    // source-keyed `cleanup_orphans` (#165): a blanket `DELETE FROM ...` before
    // re-scan would wipe a missing root's sessions before the scan can prove they
    // were deleted, destroying the index on a transient root outage (#177). Force
    // instead re-indexes every present file (mtime skip bypassed via `ctx.force`)
    // while cleanup removes only rows whose scanned root confirms their absence.
    let existing = load_existing_sessions(conn)?;
    // Compute the embedded-session set before the transaction borrows conn, only
    // when the model is absent — when present the normal re-index path applies and
    // the (potentially large) set is never built (#215).
    let embedded_sessions = if embed_capable {
        None
    } else {
        Some(embedded_session_ids(conn)?)
    };
    let scan = collect_sources(opts);
    let sources = &scan.sources;

    info!(count = sources.len(), "Found source files");

    let tx = conn.transaction().context("Failed to begin transaction")?;
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
    let totals = index_all(&ctx, sources)?;
    cleanup_orphans(&tx, &existing, sources, &scan.scanned, totals.indexed)?;
    tx.commit().context("Failed to commit transaction")?;

    finalize_fts(conn, totals.indexed, opts.force)?;

    let total_sessions = {
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        usize::try_from(n.max(0)).expect("non-negative session count fits in usize")
    };

    Ok(IndexStats {
        indexed: totals.indexed,
        parse_errors: totals.parse_errors,
        first_error: totals.first_error,
        total_sessions,
        elapsed_secs: start.elapsed().as_secs_f64(),
        skipped_roots: summarize_skipped_roots(&existing, sources, &scan.skipped),
        preserved_embedded: totals.preserved_embedded,
    })
}

/// Session ids that already hold at least one embedding. One pass over each table
/// (#138): load the embedded chunk ids into a set, then scan `qa_chunks` once,
/// collecting the session of every chunk in that set. A correlated subquery
/// against `vec_chunks.chunk_id` would re-scan it per row — that column is an
/// unindexed auxiliary column (see `embedder::pending_chunks`), so the one-pass
/// form keeps this O(N+M), not O(N*M).
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
            // Decode session_id only on a hit; sqlite materializes only the
            // columns a row actually reads (mirrors `pending_chunks`, #138).
            return Ok(Some(row.get::<_, String>(1)?));
        }
        Ok(None)
    })?;
    rows.filter_map(StdResult::transpose)
        .collect::<StdResult<_, _>>()
        .map_err(Into::into)
}

/// Counts, per source whose root was not scanned this run, how many existing
/// sessions were preserved from orphan cleanup. Only rows whose file is absent
/// from `sources` are counted — mirroring the deletion condition in
/// `cleanup_orphans` — so a partially-failed scan that re-indexed some files of
/// the source does not over-report the at-risk magnitude. A `MissingRoot` with
/// zero preserved rows is omitted so a user who simply never used one tool sees
/// no noise; an `IncompleteEnumeration` always surfaces regardless of preserved
/// count, since the partial read itself is the silent-under-index signal (#185).
/// `None`-source rows are excluded by construction (they match neither variant),
/// consistent with cleanup_orphans preserving them unconditionally.
fn summarize_skipped_roots(
    existing: &HashMap<String, SessionEntry>,
    sources: &[(PathBuf, Source)],
    skipped: &HashMap<Source, SkippedReason>,
) -> Vec<SkippedRoot> {
    let source_paths: HashSet<&Path> = sources.iter().map(|(p, _)| p.as_path()).collect();
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
            // MissingRoot + 0 preserved = a tool the user never ran; staying
            // silent suppresses noise (#165). IncompleteEnumeration is a present
            // root that failed to read, so surface it even with nothing preserved
            // — that case (#185) is exactly the silent under-index this exists to
            // catch. The asymmetry is intentional.
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
    let mut stmt = conn.prepare("SELECT file_path, session_id, source, mtime FROM sessions")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
            row.get::<_, Option<f64>>(3)?,
        ))
    })?;
    let mut map = HashMap::new();
    for row in rows {
        let (fp, sid, src, mt) = row?;
        map.insert(
            fp,
            SessionEntry {
                session_id: sid,
                source: Source::from_db(&src),
                mtime: mt,
            },
        );
    }
    Ok(map)
}

fn cleanup_orphans(
    tx: &Transaction,
    existing: &HashMap<String, SessionEntry>,
    sources: &[(PathBuf, Source)],
    scanned: &HashSet<Source>,
    indexed: usize,
) -> Result<()> {
    if existing.is_empty() || (indexed == 0 && sources.len() == existing.len()) {
        return Ok(());
    }
    let source_paths: HashSet<&Path> = sources.iter().map(|(p, _)| p.as_path()).collect();
    for (fp, entry) in existing {
        // Only a successfully scanned root proves its files were deleted. A row
        // whose source root was missing or unreadable this run is preserved, not
        // treated as orphaned — otherwise a transient root outage wipes the index
        // (#165). A `None` source (unrecognized DB value) is likewise preserved.
        let scanned_here = entry.source.is_some_and(|s| scanned.contains(&s));
        if !scanned_here {
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

/// Returns whether the whole tree was successfully enumerated. `collect_sources`
/// uses this to decide if the root counts as scanned: only a fully-read tree
/// proves which files are absent, so orphan cleanup may delete its missing rows.
/// Any failure — the root itself, a nested subdirectory, or a depth-limit
/// truncation — yields `false`, preserving the source's rows this run (#165).
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
        // Truncation means files below this depth went unseen, so the tree is
        // not fully enumerated — report incomplete to keep cleanup off this source.
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

/// Classifies each directory entry, returning whether every entry was read.
/// Taking the entry iterator as an argument lets tests inject an `Err` item to
/// exercise the per-entry failure path, which a real APFS filesystem never
/// triggers (readdir returns d_type, so `file_type()` makes no syscall). A
/// `DirEntry` or `file_type()` failure leaves the entry unseen, so it flips
/// `fully_read` to keep `cleanup_orphans` off this source's rows (#181).
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

pub(crate) struct ChunkStats {
    pub chunks_created: usize,
}

/// Chunks every un-chunked session inside a single transaction. `on_progress`
/// receives `(done_sessions, total_sessions)` after each session — session
/// units, unlike `embed_chunks` whose callback counts chunks. Progress counts
/// staged work, not durable state: the callback fires before the final commit,
/// whereas `embed_chunks` commits each batch before firing.
///
/// # Panics
///
/// Propagates a panic from `on_progress`; the unwind drops the open
/// transaction and rusqlite rolls back every staged chunk.
pub(crate) fn index_chunks(
    conn: &mut Connection,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<ChunkStats> {
    let sessions: Vec<(String, Option<i64>)> = {
        let mut stmt = conn.prepare(
            "SELECT s.session_id, s.timestamp FROM sessions s \
             WHERE NOT EXISTS (SELECT 1 FROM qa_chunks c WHERE c.session_id = s.session_id)",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
        })?;
        rows.collect::<StdResult<Vec<_>, _>>()?
    };

    if sessions.is_empty() {
        return Ok(ChunkStats { chunks_created: 0 });
    }

    info!(count = sessions.len(), "Chunking sessions");

    let total = sessions.len();
    let tx = conn.transaction()?;
    let mut chunks_created = 0;
    for (done, (session_id, timestamp)) in sessions.iter().enumerate() {
        let messages = read_session_messages(&tx, session_id)?;

        let chunks = chunker::chunk_messages(session_id, &messages, *timestamp);

        for chunk in &chunks {
            tx.execute(
                "INSERT INTO qa_chunks (session_id, content, timestamp, src_rowid_lo, src_rowid_hi) \
                 VALUES (?1, ?2, ?3, ?4, ?5)",
                rusqlite::params![
                    chunk.session_id,
                    chunk.content,
                    chunk.timestamp,
                    chunk.src_rowid_lo,
                    chunk.src_rowid_hi
                ],
            )?;
            chunks_created += 1;
        }

        if let Some(cb) = &on_progress {
            cb(done + 1, total);
        }
    }

    tx.commit()?;

    Ok(ChunkStats { chunks_created })
}

/// Read a session's messages in rowid order, paired with their fts5 rowid. The
/// rowid feeds the chunker's source range (#192) and is the same rowid the search
/// JOIN resolves an FTS hit against. Unknown roles are skipped (recall only writes
/// user/assistant, so this is defensive), and their rowids never enter a range.
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

/// Link pre-#192 chunks (NULL src_rowid_lo/hi) to their source message rowid range
/// without a re-embed. For each session holding an unlinked chunk, re-run the
/// deterministic chunker over its messages and zip the re-derived ranges onto the
/// stored chunks by qa_chunks.id ascending (= original insertion order). A session
/// is updated only when the re-derived chunks match the stored ones both in count
/// and in content, position by position. A count or content mismatch means the
/// chunker re-grouped differently than the stored chunks were built (chunker logic
/// drift since the session was indexed), so the rows stay NULL and keep resolving
/// via the instr fallback rather than inherit a mis-aligned range over stale
/// content. Content equality is the load-bearing guard: a count-preserving
/// re-grouping (e.g. #229 merging consecutive assistants into one chunk) keeps the
/// count but rewrites the content, so stamping the widened range onto the old
/// content would make an FTS hit on the newly covered text resolve to a chunk that
/// never held it. `rebuild` re-chunks and re-embeds those drifted sessions.
/// Returns the number of sessions backfilled. A no-op once every row is linked,
/// since the NULL-row scan then finds no session.
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

        // The id-order zip is only safe when the re-derived chunks correspond to
        // the stored ones position by position. A count mismatch breaks that
        // outright; a content mismatch means the chunker re-grouped the same
        // number of chunks differently (e.g. #229 merging consecutive assistants
        // widens a range while rewriting its content), so stamping the new range
        // onto stale content would misroute an FTS hit on the newly covered text.
        // Either way the grouping drifted since the session was indexed, so skip
        // it and keep NULL; log it so the permanent instr-fallback degradation is
        // observable rather than silent. `rebuild` re-chunks and re-embeds.
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

/// The single source→parser dispatch, shared by `index_file` and the backfill
/// re-read so a new `Source` variant cannot update one site and miss the other.
fn parse_session(fpath: &Path, source: Source) -> Result<Option<ParseResult>> {
    match source {
        Source::Claude => parse_claude_session(fpath),
        Source::Codex => parse_codex_session(fpath),
    }
}

/// Outcome of re-reading one session's JSONL for path extraction.
enum RereadOutcome {
    /// A terminal read: parsed paths, a clean-but-empty (zero-touch) session,
    /// or an unrecognized source. Re-reading can never yield more, so the
    /// caller marks the session.
    Paths(Vec<String>),
    /// The JSONL could not be read (open / I/O error) — see
    /// `backfill_session_files` for the retry contract.
    Unavailable,
}

/// Re-read the write-target paths from one session's JSONL for path extraction
/// only. Malformed lines are skipped by the parsers themselves, so an `Err`
/// from the parse is an I/O failure, not a content problem.
fn reread_scanned_files(fpath: &Path, source: &str) -> RereadOutcome {
    let result = match Source::from_db(source) {
        Some(source) => parse_session(fpath, source),
        None => return RereadOutcome::Paths(Vec::new()),
    };
    match result {
        Ok(Some(parsed)) => RereadOutcome::Paths(parsed.scanned_files),
        Ok(None) => RereadOutcome::Paths(Vec::new()),
        Err(e) => {
            warn!(
                error = %e,
                path = %fpath.display(),
                "backfill: session JSONL unreadable; left unmarked for retry on the next index"
            );
            RereadOutcome::Unavailable
        }
    }
}

/// Fill `session_files` for legacy sessions indexed before the feature existed.
/// Such a session has `files_scanned` NULL and no `session_files` rows; its JSONL
/// mtime is unchanged, so `index_file`'s freshness check skips a re-parse and
/// `upsert_session` never runs — only this pass can record its write-target paths.
/// Each unmarked session's JSONL is re-read for path extraction only, so messages
/// / qa_chunks / vec_chunks are untouched. The `session_files` inserts and the
/// `files_scanned` raise happen in ONE transaction, so an interruption never
/// leaves the marker up with paths missing. A session with no write-target
/// tool_use, or an unrecognized source, still gets the marker so it is not
/// re-read on a later index; a session whose JSONL cannot be READ stays unmarked
/// so a transient failure is retried on the next index instead of being locked
/// out permanently (the retry cost is one failed open per index run). A no-op
/// once every session is marked, since the NULL-marker scan then finds no
/// session. Reports per-session progress through `on_progress(done, total)`;
/// propagates a panic from it like `index_chunks`. Returns the number of
/// sessions marked.
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
        let scanned = match reread_scanned_files(Path::new(file_path), source) {
            RereadOutcome::Paths(paths) => paths,
            RereadOutcome::Unavailable => continue,
        };
        // Precondition upheld by the `files_scanned IS NULL` scan above: no
        // pre-existing session_files rows under this id.
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
