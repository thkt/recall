use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs;
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
}

#[derive(Debug, Clone)]
pub struct SkippedRoot {
    pub source: Source,
    pub preserved_sessions: usize,
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
    ParseError(String),
}

struct IndexContext<'a> {
    tx: &'a Transaction<'a>,
    existing: &'a HashMap<String, SessionEntry>,
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

/// Delete all dependent data for a session (messages, chunks, embeddings).
fn delete_session_dependents(tx: &Transaction, session_id: &str) -> Result<()> {
    tx.execute("DELETE FROM messages WHERE session_id = ?", [session_id])?;
    tx.execute(
        "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ?)",
        [session_id],
    )?;
    tx.execute("DELETE FROM qa_chunks WHERE session_id = ?", [session_id])?;
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

    ctx.tx.execute(
        "INSERT OR REPLACE INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime, session_type) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
        rusqlite::params![
            parsed.metadata.session_id,
            parsed.metadata.source.as_str(),
            parsed.metadata.file_path,
            parsed.metadata.project,
            parsed.metadata.slug,
            parsed.metadata.timestamp,
            mtime,
            session_type,
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
    if let Some(new_mt) = mtime
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

    let result = match source {
        Source::Claude => parse_claude_session(fpath),
        Source::Codex => parse_codex_session(fpath),
    };
    let parsed = match result {
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

/// Returns the discovered files and the set of sources whose root was
/// successfully enumerated this run. A missing or unreadable root is absent
/// from the set so orphan cleanup never treats its sessions as deleted (#165).
fn collect_sources(opts: &IndexOptions) -> (Vec<(PathBuf, Source)>, HashSet<Source>) {
    let mut sources = Vec::new();
    let mut scanned = HashSet::new();
    if opts.claude_dir.is_dir()
        && collect_jsonl_files(opts.claude_dir, &mut sources, Source::Claude)
    {
        scanned.insert(Source::Claude);
    }
    if opts.codex_dir.is_dir() && collect_jsonl_files(opts.codex_dir, &mut sources, Source::Codex) {
        scanned.insert(Source::Codex);
    }
    (sources, scanned)
}

fn index_all(
    ctx: &IndexContext,
    sources: &[(PathBuf, Source)],
) -> Result<(usize, usize, Option<String>)> {
    let mut indexed = 0;
    let mut parse_errors = 0;
    let mut first_error = None;

    for (fpath, source) in sources {
        match index_file(ctx, fpath, source)? {
            IndexOutcome::Indexed => indexed += 1,
            IndexOutcome::Unchanged => {}
            IndexOutcome::ParseError(msg) => {
                parse_errors += 1;
                if first_error.is_none() {
                    first_error = Some(msg);
                }
            }
        }
    }

    Ok((indexed, parse_errors, first_error))
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

pub(crate) fn index_from_dirs(conn: &mut Connection, opts: &IndexOptions) -> Result<IndexStats> {
    let start = Instant::now();

    // Always full-scan: file-level mtime checks in check_freshness keep indexing
    // incremental. A directory-mtime skip optimization here used to miss new files
    // added to existing deep dirs (Codex Y/M/D, Claude subagents) — #52 / #70.
    let existing = if opts.force {
        HashMap::new()
    } else {
        load_existing_sessions(conn)?
    };
    let (sources, scanned) = collect_sources(opts);

    info!(count = sources.len(), "Found source files");

    let tx = conn.transaction().context("Failed to begin transaction")?;
    if opts.force {
        tx.execute_batch(
            "DELETE FROM vec_chunks; DELETE FROM qa_chunks; \
             DELETE FROM sessions; DELETE FROM messages;",
        )?;
    }
    tx.execute(
        "INSERT INTO messages(messages, rank) VALUES('automerge', 0)",
        [],
    )?;

    let ctx = IndexContext {
        tx: &tx,
        existing: &existing,
    };
    let (indexed, parse_errors, first_error) = index_all(&ctx, &sources)?;
    cleanup_orphans(&tx, &existing, &sources, &scanned, indexed)?;
    tx.commit().context("Failed to commit transaction")?;

    finalize_fts(conn, indexed, opts.force)?;

    let total_sessions = {
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        usize::try_from(n.max(0)).expect("non-negative session count fits in usize")
    };

    Ok(IndexStats {
        indexed,
        parse_errors,
        first_error,
        total_sessions,
        elapsed_secs: start.elapsed().as_secs_f64(),
        skipped_roots: summarize_skipped_roots(&existing, &sources, &scanned),
    })
}

/// Counts, per source whose root was not scanned this run, how many existing
/// sessions were preserved from orphan cleanup. Only rows whose file is absent
/// from `sources` are counted — mirroring the deletion condition in
/// `cleanup_orphans` — so a partially-failed scan that re-indexed some files of
/// the source does not over-report the at-risk magnitude. Sources with zero
/// preserved rows are omitted so a user who simply never used one tool sees no
/// noise. `None`-source rows are excluded by construction (they match neither
/// variant), consistent with cleanup_orphans preserving them unconditionally.
fn summarize_skipped_roots(
    existing: &HashMap<String, SessionEntry>,
    sources: &[(PathBuf, Source)],
    scanned: &HashSet<Source>,
) -> Vec<SkippedRoot> {
    let source_paths: HashSet<&Path> = sources.iter().map(|(p, _)| p.as_path()).collect();
    [Source::Claude, Source::Codex]
        .into_iter()
        .filter(|s| !scanned.contains(s))
        .filter_map(|source| {
            let preserved_sessions = existing
                .iter()
                .filter(|(fp, e)| {
                    e.source == Some(source) && !source_paths.contains(Path::new(fp.as_str()))
                })
                .count();
            (preserved_sessions > 0).then_some(SkippedRoot {
                source,
                preserved_sessions,
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
        debug!(
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
    let mut fully_read = true;
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "cannot read entry");
                continue;
            }
        };
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                warn!(path = %entry.path().display(), error = %e, "cannot get file type");
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
        let messages = {
            let mut stmt = tx.prepare_cached(
                "SELECT role, text FROM messages WHERE session_id = ? ORDER BY rowid",
            )?;
            let rows = stmt.query_map([session_id], |row| {
                Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
            })?;
            let mut msgs = Vec::new();
            for r in rows {
                let (role_str, text) = r?;
                let Some(role) = Role::from_db(&role_str) else {
                    debug!(role = %role_str, session_id, "unknown role");
                    continue;
                };
                msgs.push(Message { role, text });
            }
            msgs
        };

        let chunks = chunker::chunk_messages(session_id, &messages, *timestamp);

        for chunk in &chunks {
            tx.execute(
                "INSERT INTO qa_chunks (session_id, content, timestamp) VALUES (?1, ?2, ?3)",
                rusqlite::params![chunk.session_id, chunk.content, chunk.timestamp],
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

#[cfg(test)]
mod tests;
