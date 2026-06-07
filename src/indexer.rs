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

type SessionEntry = (String, Option<f64>);

pub struct IndexStats {
    pub indexed: usize,
    pub parse_errors: usize,
    pub first_error: Option<String>,
    pub total_sessions: usize,
    pub elapsed_secs: f64,
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

    if let Some((old_sid, _)) = cached {
        ctx.tx
            .execute("DELETE FROM sessions WHERE session_id = ?", [old_sid])?;
        delete_session_dependents(ctx.tx, old_sid)?;
    }

    // New session_id differs from cached — clean up any orphaned messages under the new ID
    // (e.g. left over from a renamed file that previously used it).
    let same_session_id =
        matches!(cached, Some((old_sid, _)) if *old_sid == parsed.metadata.session_id);
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
        && let Some((_, Some(old_mt))) = ctx.existing.get(&fpath_str)
        && (*old_mt - new_mt).abs() < 0.001
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

fn collect_sources(opts: &IndexOptions) -> Vec<(PathBuf, Source)> {
    let mut sources = Vec::new();
    if opts.claude_dir.is_dir() {
        collect_jsonl_files(opts.claude_dir, &mut sources, Source::Claude);
    }
    if opts.codex_dir.is_dir() {
        collect_jsonl_files(opts.codex_dir, &mut sources, Source::Codex);
    }
    sources
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
    let sources = collect_sources(opts);

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
    cleanup_orphans(&tx, &existing, &sources, indexed)?;
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
    })
}

fn load_existing_sessions(conn: &Connection) -> Result<HashMap<String, SessionEntry>> {
    let mut stmt = conn.prepare("SELECT file_path, session_id, mtime FROM sessions")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, Option<f64>>(2)?,
        ))
    })?;
    let mut map = HashMap::new();
    for row in rows {
        let (fp, sid, mt) = row?;
        map.insert(fp, (sid, mt));
    }
    Ok(map)
}

fn cleanup_orphans(
    tx: &Transaction,
    existing: &HashMap<String, SessionEntry>,
    sources: &[(PathBuf, Source)],
    indexed: usize,
) -> Result<()> {
    if existing.is_empty() || (indexed == 0 && sources.len() == existing.len()) {
        return Ok(());
    }
    let source_paths: HashSet<&Path> = sources.iter().map(|(p, _)| p.as_path()).collect();
    for (fp, (sid, _)) in existing {
        if !source_paths.contains(Path::new(fp.as_str())) {
            let deleted = tx.execute(
                "DELETE FROM sessions WHERE session_id = ? AND file_path = ?",
                rusqlite::params![sid, fp],
            )?;
            if deleted > 0 {
                delete_session_dependents(tx, sid)?;
            }
        }
    }
    Ok(())
}

const MAX_DIR_DEPTH: usize = 10;

fn collect_jsonl_files(dir: &Path, out: &mut Vec<(PathBuf, Source)>, source: Source) {
    collect_jsonl_files_inner(dir, out, source, 0);
}

fn collect_jsonl_files_inner(
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    depth: usize,
) {
    if depth >= MAX_DIR_DEPTH {
        debug!(
            max_depth = MAX_DIR_DEPTH,
            path = %dir.display(),
            "depth limit reached"
        );
        return;
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "cannot read dir");
            return;
        }
    };
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
            collect_jsonl_files_inner(&path, out, source, depth + 1);
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push((path, source));
        }
    }
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
mod tests {
    use std::sync::Mutex;
    use std::time::{Duration, SystemTime};

    use super::*;
    use crate::db::{seed_session, setup_test_db};
    use crate::embedder::{EMBED_BATCH_SIZE, MockEmbedder, embed_recent_chunks};
    use tempfile::TempDir;

    #[test]
    fn test_index_from_dirs_indexes_and_skips() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 1);
        assert_eq!(stats.total_sessions, 1);

        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.indexed, 0);
        assert_eq!(stats2.total_sessions, 1);
    }

    // TC-003 (#130): the production steady state is a hook re-indexing an
    // already-indexed tree. Beyond `indexed == 0` (parse skip, pinned above),
    // the DB content must be byte-stable: no duplicate chunks, no FTS growth,
    // no re-created sessions.
    #[test]
    fn test_index_from_dirs_steady_state_keeps_db_stable() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("s.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"steady question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"steady answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
            ),
        )
        .unwrap();
        let codex_dir = tmp.path().join("codex_sessions");
        let opts = IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };

        index_from_dirs(&mut conn, &opts).unwrap();
        index_chunks(&mut conn, None).unwrap();
        let counts = |conn: &Connection| -> (i64, i64, i64, i64) {
            let m = conn
                .query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))
                .unwrap();
            let c = conn
                .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
                .unwrap();
            let s = conn
                .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
                .unwrap();
            let v = conn
                .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
                .unwrap();
            (m, c, s, v)
        };
        let before = counts(&conn);

        let stats = index_from_dirs(&mut conn, &opts).unwrap();
        index_chunks(&mut conn, None).unwrap();

        assert_eq!(stats.indexed, 0, "an unchanged tree re-parses nothing");
        assert_eq!(
            counts(&conn),
            before,
            "re-index of an unchanged tree must not grow or rewrite the DB"
        );
    }

    // TC-004 (#130): mtime-forward re-index is the self-heal core — a session
    // file appended after indexing (the tail of a live session) must be
    // re-parsed, surface the new message, and replace its chunks without
    // duplicating the session.
    #[test]
    fn test_index_from_dirs_reindexes_appended_file() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        let file = claude_dir.join("s.jsonl");
        fs::write(
            &file,
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"first question"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
            ),
        )
        .unwrap();
        let codex_dir = tmp.path().join("codex_sessions");
        let opts = IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        index_from_dirs(&mut conn, &opts).unwrap();

        // Append a turn and push mtime clearly past the 0.001s freshness epsilon.
        let mut all = fs::read_to_string(&file).unwrap();
        all.push_str(
            r#"{"type":"assistant","cwd":"/proj","message":{"role":"assistant","content":"appended answer"},"timestamp":"2026-03-01T00:00:01Z"}"#,
        );
        fs::write(&file, all).unwrap();
        let f = fs::OpenOptions::new().append(true).open(&file).unwrap();
        f.set_modified(SystemTime::now() + Duration::from_secs(10))
            .unwrap();
        drop(f);

        let stats = index_from_dirs(&mut conn, &opts).unwrap();

        assert_eq!(stats.indexed, 1, "the appended file must be re-parsed");
        let sessions: i64 = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            sessions, 1,
            "re-index upserts the session, not duplicates it"
        );
        let appended: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM messages WHERE messages MATCH 'appended answer'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(appended, 1, "the appended message must be searchable");
        let original: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM messages WHERE messages MATCH 'first question'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            original, 1,
            "re-parsing the appended file must keep the original message searchable"
        );
    }

    fn index_one_claude_session(content: &str) -> Option<String> {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        let line = format!(
            r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":{content:?}}},"timestamp":"2026-03-01T00:00:00Z"}}"#
        );
        fs::write(claude_dir.join("s.jsonl"), line).unwrap();
        let codex_dir = tmp.path().join("codex_sessions");
        index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        conn.query_row("SELECT session_type FROM sessions", [], |r| r.get(0))
            .unwrap()
    }

    // T-004 (#24/FR-003): ingest tags session_type from the first user turn — an
    // automated marker yields 'automated', a human turn yields 'interactive'.
    #[test]
    fn test_ingest_tags_session_type_from_first_user_turn() {
        assert_eq!(
            index_one_claude_session("<command-message>clear</command-message>").as_deref(),
            Some("automated"),
            "a slash-command first turn must be tagged automated"
        );
        assert_eq!(
            index_one_claude_session("how do I implement authentication").as_deref(),
            Some("interactive"),
            "a human first turn must be tagged interactive"
        );
    }

    #[test]
    fn test_incremental_scan_picks_up_new_file_in_existing_codex_day_dir() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        let codex_dir = tmp.path().join("codex_sessions");
        let day_dir = codex_dir.join("2026/04/27");
        fs::create_dir_all(&day_dir).unwrap();

        let s1 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s1","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"first codex session"}]}}"#;
        fs::write(day_dir.join("s1.jsonl"), s1).unwrap();

        let stats1 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats1.indexed, 1);

        // New session added into the SAME existing day dir. The parent `2026/` mtime
        // does not change, so the old dirs_changed_since optimization skipped the scan
        // and permanently missed the new session (#52 / #70).
        let s2 = r#"{"timestamp":"2026-04-27T00:00:00Z","type":"session_meta","payload":{"id":"codex-s2","cwd":"/proj"}}
{"timestamp":"2026-04-27T00:00:01Z","type":"response_item","payload":{"role":"user","content":[{"type":"input_text","text":"second codex session"}]}}"#;
        fs::write(day_dir.join("s2.jsonl"), s2).unwrap();

        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(
            stats2.indexed, 1,
            "a new session in an existing deep dir must be indexed (parent mtime unchanged)"
        );
        assert_eq!(stats2.total_sessions, 2);
    }

    #[test]
    fn test_index_from_dirs_force_reindexes() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 1);

        // Populate qa_chunks + vec_chunks before force reindex
        index_chunks(&mut conn, None).unwrap();
        let embedder = MockEmbedder::new();
        embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
        let qa_before: i64 = conn
            .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap();
        let vec_before: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert!(qa_before > 0, "should have qa_chunks before force reindex");
        assert!(
            vec_before > 0,
            "should have vec_chunks before force reindex"
        );

        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: true,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.indexed, 1);

        let qa_after: i64 = conn
            .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap();
        let vec_after: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(qa_after, 0, "force reindex should clear qa_chunks");
        assert_eq!(vec_after, 0, "force reindex should clear vec_chunks");
    }

    #[test]
    fn test_orphan_cleanup_removes_deleted_files() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        let f1 = claude_dir.join("session1.jsonl");
        let f2 = claude_dir.join("session2.jsonl");
        fs::write(&f1, r#"{"type":"user","message":{"role":"user","content":"one"},"timestamp":"2026-03-01T00:00:00Z"}"#).unwrap();
        fs::write(&f2, r#"{"type":"user","message":{"role":"user","content":"two"},"timestamp":"2026-03-01T00:00:00Z"}"#).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 2);
        assert_eq!(stats.total_sessions, 2);

        // Create chunks + embeddings for session2 before deleting the file
        index_chunks(&mut conn, None).unwrap();
        let embedder = MockEmbedder::new();
        embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
        let chunks_before: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 'session2'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert!(chunks_before > 0, "session2 should have chunks");
        let vecs_before: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert!(
            vecs_before > 0,
            "should have vec_chunks before orphan cleanup"
        );

        fs::remove_file(&f2).unwrap();
        // force: false exercises cleanup_orphans (force: true bulk-deletes everything,
        // which would make the cascade assertion a false pass)
        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.total_sessions, 1);

        let chunks_after: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM qa_chunks WHERE session_id = 'session2'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(
            chunks_after, 0,
            "orphan cleanup should cascade to qa_chunks"
        );

        let vecs_for_session1: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        // session1 still exists, so its vec_chunks should remain; session2's should be gone
        assert!(
            vecs_for_session1 < vecs_before,
            "orphan cleanup should remove session2 vec_chunks: before={vecs_before}, after={vecs_for_session1}"
        );
    }

    #[test]
    fn test_session_id_collision_cleans_old_messages() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let dir_a = tmp.path().join("claude_a");
        let dir_b = tmp.path().join("claude_b");
        fs::create_dir_all(&dir_a).unwrap();
        fs::create_dir_all(&dir_b).unwrap();
        fs::write(
            dir_a.join("collision.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"from dir a"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();
        fs::write(
            dir_b.join("collision.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"from dir b"},"timestamp":"2026-03-02T00:00:00Z"}"#,
        ).unwrap();

        let empty_dir = tmp.path().join("empty");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &dir_a,
                codex_dir: &empty_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 1);

        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &dir_b,
                codex_dir: &empty_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.indexed, 1);

        let msg_count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM messages WHERE session_id = 'collision'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(msg_count, 1);
    }

    #[test]
    fn test_collect_depth_limit() {
        let tmp = TempDir::new().unwrap();

        let mut deep = tmp.path().to_path_buf();
        for i in 0..MAX_DIR_DEPTH {
            deep = deep.join(format!("d{i}"));
        }
        fs::create_dir_all(&deep).unwrap();
        fs::write(deep.join("deep.jsonl"), "{}").unwrap();

        fs::write(tmp.path().join("shallow.jsonl"), "{}").unwrap();

        let mut files = Vec::new();
        collect_jsonl_files(tmp.path(), &mut files, Source::Claude);

        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("shallow.jsonl"));
    }

    #[cfg(unix)]
    #[test]
    fn test_collect_skips_symlinks() {
        use std::os::unix::fs::symlink;

        let tmp = TempDir::new().unwrap();

        let real_file = tmp.path().join("real.jsonl");
        fs::write(&real_file, "{}").unwrap();

        let link = tmp.path().join("link.jsonl");
        symlink(&real_file, &link).unwrap();

        let mut files = Vec::new();
        collect_jsonl_files(tmp.path(), &mut files, Source::Claude);

        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("real.jsonl"));
    }

    #[test]
    fn test_011_index_chunks_incremental() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("s1.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"hi there"}}"#,
            ),
        )
        .unwrap();
        let codex_dir = tmp.path().join("codex_sessions");

        index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();

        let stats1 = index_chunks(&mut conn, None).unwrap();
        assert_eq!(stats1.chunks_created, 1);

        let stats2 = index_chunks(&mut conn, None).unwrap();
        assert_eq!(stats2.chunks_created, 0);
    }

    #[test]
    fn test_index_chunks_progress_callback_counts_sessions() {
        let (_dir, mut conn) = setup_test_db();
        seed_session(&conn, "s1");
        seed_session(&conn, "s2");
        for (role, text) in [("user", "hello"), ("assistant", "hi there")] {
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES ('s1', ?1, ?2)",
                rusqlite::params![role, text],
            )
            .unwrap();
        }

        let calls = Mutex::new(Vec::new());
        let stats = index_chunks(
            &mut conn,
            Some(&|done, total| calls.lock().unwrap().push((done, total))),
        )
        .unwrap();

        // Progress advances per session processed, not per chunk created:
        // s2 has no messages (0 chunks) yet still counts toward done/total.
        assert_eq!(calls.into_inner().unwrap(), vec![(1, 2), (2, 2)]);
        assert_eq!(stats.chunks_created, 1);
    }

    #[test]
    fn test_index_chunks_progress_callback_silent_when_all_chunked() {
        let (_dir, mut conn) = setup_test_db();
        seed_session(&conn, "s1");
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello')",
            [],
        )
        .unwrap();
        index_chunks(&mut conn, None).unwrap();

        // Every session is chunked, so the empty early-return must not
        // invoke the callback even when one is supplied.
        let calls = Mutex::new(Vec::new());
        let stats = index_chunks(
            &mut conn,
            Some(&|done, total| calls.lock().unwrap().push((done, total))),
        )
        .unwrap();

        assert_eq!(stats.chunks_created, 0);
        assert_eq!(calls.into_inner().unwrap(), Vec::<(usize, usize)>::new());
    }

    #[test]
    fn test_009_recall_claude_dir_overrides_claude_config_dir() {
        let home = Path::new("/fake/home");
        let result = resolve_claude_dir_with(home, |key| match key {
            "RECALL_CLAUDE_DIR" => Some(OsString::from("/custom/recall/dir")),
            "CLAUDE_CONFIG_DIR" => Some(OsString::from("/custom/config/dir")),
            _ => None,
        });
        assert_eq!(
            result,
            PathBuf::from("/custom/recall/dir"),
            "RECALL_CLAUDE_DIR should take priority over CLAUDE_CONFIG_DIR"
        );
    }

    #[test]
    fn test_010_claude_config_dir_used_as_fallback() {
        let home = Path::new("/fake/home");
        let result = resolve_claude_dir_with(home, |key| match key {
            "CLAUDE_CONFIG_DIR" => Some(OsString::from("/custom/config")),
            _ => None,
        });
        assert_eq!(
            result,
            PathBuf::from("/custom/config/projects"),
            "CLAUDE_CONFIG_DIR/projects should be used when RECALL_CLAUDE_DIR is unset"
        );
    }

    #[test]
    fn test_codex_dir_env_override() {
        let home = Path::new("/fake/home");
        let result = resolve_codex_dir_with(home, |key| match key {
            "RECALL_CODEX_DIR" => Some(OsString::from("/custom/codex/sessions")),
            _ => None,
        });
        assert_eq!(result, PathBuf::from("/custom/codex/sessions"));
    }

    #[test]
    fn test_codex_dir_default() {
        let home = Path::new("/fake/home");
        let result = resolve_codex_dir_with(home, |_| None);
        assert_eq!(result, PathBuf::from("/fake/home/.codex/sessions"));
    }

    #[test]
    fn test_embed_recent_chunks_populates_vec_chunks() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();
        let claude_dir = tmp.path().join("claude_projects");
        fs::create_dir_all(&claude_dir).unwrap();
        fs::write(
            claude_dir.join("s1.jsonl"),
            concat!(
                r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
                "\n",
                r#"{"type":"assistant","message":{"role":"assistant","content":"hi there"}}"#,
            ),
        )
        .unwrap();
        let codex_dir = tmp.path().join("codex_sessions");

        index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        index_chunks(&mut conn, None).unwrap();

        let chunk_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap();
        assert!(chunk_count > 0, "should have chunks to embed");

        let embedder = MockEmbedder::new();
        let result = embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();

        assert_eq!(result.embedded, usize::try_from(chunk_count).unwrap());
        assert_eq!(result.failed_count, 0);

        let vec_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(vec_count, chunk_count, "all chunks should be embedded");

        let result2 = embed_recent_chunks(&mut conn, &embedder, 100, None).unwrap();
        assert_eq!(result2.embedded, 0);
    }

    // T-007 (FR-001, FR-002): a mid-run batch failure is non-blocking — the run
    // continues past it and counts the rest as failed. With failing_after(128) the
    // first batch (128 chunks) succeeds and commits; the MockEmbedder counter is not
    // reset, so every later batch fails (call_count stays >= 128). Under continue the
    // run does not stop at batch 2: it reports embedded == 128 and failed_count ==
    // the remaining chunks, and the committed batch survives.
    // (Updated from test_embed_chunks_stops_on_error_preserves_progress, whose
    // stopped_at_error.is_some() assertion predates the break->continue change.)
    // Perspective: branch (the Err arm continues) + boundary (split at EMBED_BATCH_SIZE).
    #[test]
    fn test_embed_chunks_continues_past_failed_batch_and_counts_remainder() {
        let (_dir, mut conn) = setup_test_db();

        seed_session(&conn, "s1");
        let message_count = EMBED_BATCH_SIZE + 10;
        for i in 0..message_count {
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', ?1)",
                [format!("question number {i}")],
            )
            .unwrap();
            conn.execute(
                "INSERT INTO messages (session_id, role, text) VALUES ('s1', 'assistant', ?1)",
                [format!("answer number {i}")],
            )
            .unwrap();
        }
        index_chunks(&mut conn, None).unwrap();

        let chunk_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap();
        assert!(
            chunk_count > EMBED_BATCH_SIZE as i64,
            "need enough chunks for multiple batches, got {chunk_count}"
        );

        let embedder = MockEmbedder::failing_after(EMBED_BATCH_SIZE);
        let result = embed_recent_chunks(
            &mut conn,
            &embedder,
            usize::try_from(chunk_count).unwrap(),
            None,
        )
        .unwrap();

        assert_eq!(
            result.embedded, EMBED_BATCH_SIZE,
            "the first batch embeds and commits before the failures start"
        );
        let remaining = usize::try_from(chunk_count).unwrap() - EMBED_BATCH_SIZE;
        assert_eq!(
            result.failed_count, remaining,
            "every batch after the first fails (counter not reset), so all remaining chunks are failed"
        );

        let vec_count: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            vec_count, EMBED_BATCH_SIZE as i64,
            "the committed first batch survives the later failures"
        );
    }
}
