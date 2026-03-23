use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::chunker;
use crate::embedder::Embed;
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

enum IndexOutcome {
    Indexed,
    Unchanged,
    ParseError(String),
}

struct IndexContext<'a> {
    tx: &'a rusqlite::Transaction<'a>,
    existing: &'a HashMap<String, SessionEntry>,
    verbose: bool,
}

pub(crate) struct IndexOptions<'a> {
    pub force: bool,
    pub verbose: bool,
    pub claude_dir: &'a Path,
    pub codex_dir: &'a Path,
}

enum Mtime {
    /// File mtime resolved successfully.
    Value(f64),
    /// File exists but mtime unavailable — index without freshness check.
    Unknown,
    /// File cannot be stat'd — skip entirely.
    Inaccessible,
}

fn resolve_mtime(fpath: &Path, verbose: bool) -> Mtime {
    let meta = match std::fs::metadata(fpath) {
        Ok(m) => m,
        Err(_) => {
            if verbose {
                eprintln!("Warning: cannot stat {}", fpath.display());
            }
            return Mtime::Inaccessible;
        }
    };
    let mtime = match meta.modified() {
        Ok(t) => t,
        Err(e) => {
            if verbose {
                eprintln!("Warning: mtime unavailable for {}: {e}", fpath.display());
            }
            return Mtime::Unknown;
        }
    };
    match mtime.duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => Mtime::Value(d.as_secs_f64()),
        Err(e) => {
            if verbose {
                eprintln!("Warning: mtime before epoch for {}: {e}", fpath.display());
            }
            Mtime::Unknown
        }
    }
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
        ctx.tx
            .execute("DELETE FROM messages WHERE session_id = ?", [old_sid])?;
        // Clean up chunks so they get regenerated on next `recall index`
        ctx.tx.execute(
            "DELETE FROM vec_chunks WHERE chunk_id IN (SELECT id FROM qa_chunks WHERE session_id = ?)",
            [old_sid],
        )?;
        ctx.tx
            .execute("DELETE FROM qa_chunks WHERE session_id = ?", [old_sid])?;
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

    ctx.tx.execute(
        "INSERT OR REPLACE INTO sessions (session_id, source, file_path, project, slug, timestamp, mtime) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
        rusqlite::params![
            parsed.metadata.session_id,
            parsed.metadata.source.as_str(),
            parsed.metadata.file_path,
            parsed.metadata.project,
            parsed.metadata.slug,
            parsed.metadata.timestamp,
            mtime,
        ],
    )?;

    if ctx.verbose && parsed.skipped_lines > 0 {
        eprintln!(
            "Warning: {} skipped lines in {}",
            parsed.skipped_lines, fpath_str
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
        Some(s) => s.to_string(),
        None => {
            if ctx.verbose {
                eprintln!("Warning: skipping non-UTF-8 path: {}", fpath.display());
            }
            return None;
        }
    };

    let mtime = match resolve_mtime(fpath, ctx.verbose) {
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
            if ctx.verbose {
                eprintln!("Warning: {}: {e}", fpath.display());
            }
            return Ok(IndexOutcome::ParseError(format!(
                "{}: {e}",
                fpath.display()
            )));
        }
    };

    upsert_session(ctx, &fpath_str, mtime, &parsed)?;
    Ok(IndexOutcome::Indexed)
}

/// Resolve the Claude projects directory from env vars.
///
/// Priority: `RECALL_CLAUDE_DIR` > `CLAUDE_CONFIG_DIR/projects` > `~/.claude/projects`
pub(crate) fn resolve_claude_dir(home: &Path) -> PathBuf {
    if let Some(dir) = std::env::var_os("RECALL_CLAUDE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = std::env::var_os("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir).join("projects");
    }
    home.join(".claude").join("projects")
}

pub fn index_sessions(conn: &mut Connection, force: bool, verbose: bool) -> Result<IndexStats> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let claude_dir = resolve_claude_dir(&home);
    let codex_dir = std::env::var_os("RECALL_CODEX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".codex").join("sessions"));
    index_from_dirs(
        conn,
        &IndexOptions {
            force,
            verbose,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        },
    )
}

fn collect_sources(opts: &IndexOptions) -> Vec<(PathBuf, Source)> {
    let mut sources = Vec::new();
    if opts.claude_dir.is_dir() {
        collect_jsonl_files(opts.claude_dir, &mut sources, Source::Claude, opts.verbose);
    }
    if opts.codex_dir.is_dir() {
        collect_jsonl_files(opts.codex_dir, &mut sources, Source::Codex, opts.verbose);
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

fn dir_mtime_secs(path: &Path) -> Option<f64> {
    std::fs::metadata(path)
        .and_then(|m| m.modified())
        .ok()
        .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
        .map(|d| d.as_secs_f64())
}

fn dirs_changed_since(opts: &IndexOptions, last_scan: f64) -> bool {
    for dir in [opts.claude_dir, opts.codex_dir] {
        if !dir.is_dir() {
            continue;
        }
        match dir_mtime_secs(dir) {
            Some(mt) if mt >= last_scan => return true,
            None => return true,
            _ => {}
        }
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                if let Some(mt) = dir_mtime_secs(&entry.path())
                    && mt >= last_scan
                {
                    return true;
                }
            }
        }
    }
    false
}

fn scan_key(opts: &IndexOptions) -> String {
    format!(
        "last_scan:{}:{}",
        opts.claude_dir.display(),
        opts.codex_dir.display()
    )
}

fn get_last_scan(conn: &Connection, key: &str) -> Option<f64> {
    conn.query_row("SELECT value FROM recall_meta WHERE key = ?", [key], |r| {
        r.get::<_, f64>(0)
    })
    .ok()
}

fn set_last_scan(conn: &Connection, key: &str, ts: f64) -> Result<()> {
    conn.execute(
        "INSERT OR REPLACE INTO recall_meta (key, value) VALUES (?, ?)",
        rusqlite::params![key, ts],
    )?;
    Ok(())
}

pub(crate) fn index_from_dirs(conn: &mut Connection, opts: &IndexOptions) -> Result<IndexStats> {
    let start = Instant::now();

    let key = scan_key(opts);
    if !opts.force
        && let Some(last_scan) = get_last_scan(conn, &key)
        && !dirs_changed_since(opts, last_scan)
    {
        let total_sessions: usize = conn
            .query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get::<_, i64>(0))?
            .max(0) as usize;
        return Ok(IndexStats {
            indexed: 0,
            parse_errors: 0,
            first_error: None,
            total_sessions,
            elapsed_secs: start.elapsed().as_secs_f64(),
        });
    }

    let existing = if opts.force {
        HashMap::new()
    } else {
        load_existing_sessions(conn)?
    };
    let sources = collect_sources(opts);

    if opts.verbose {
        eprintln!("Found {} source files", sources.len());
    }

    let tx = conn.transaction().context("Failed to begin transaction")?;
    if opts.force {
        tx.execute_batch("DELETE FROM sessions; DELETE FROM messages;")?;
    }
    tx.execute(
        "INSERT INTO messages(messages, rank) VALUES('automerge', 0)",
        [],
    )?;

    let ctx = IndexContext {
        tx: &tx,
        existing: &existing,
        verbose: opts.verbose,
    };
    let (indexed, parse_errors, first_error) = index_all(&ctx, &sources)?;
    cleanup_orphans(&tx, &existing, &sources, indexed)?;
    tx.commit().context("Failed to commit transaction")?;

    finalize_fts(conn, indexed, opts.force)?;

    let total_sessions = {
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        n.max(0) as usize
    };

    let now_secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    if let Err(e) = set_last_scan(conn, &key, now_secs) {
        eprintln!("Warning: failed to save scan timestamp: {e}");
    }

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
    tx: &rusqlite::Transaction,
    existing: &HashMap<String, SessionEntry>,
    sources: &[(PathBuf, Source)],
    indexed: usize,
) -> Result<()> {
    if existing.is_empty() || (indexed == 0 && sources.len() == existing.len()) {
        return Ok(());
    }
    let source_paths: std::collections::HashSet<&Path> =
        sources.iter().map(|(p, _)| p.as_path()).collect();
    for (fp, (sid, _)) in existing {
        if !source_paths.contains(Path::new(fp.as_str())) {
            let deleted = tx.execute(
                "DELETE FROM sessions WHERE session_id = ? AND file_path = ?",
                rusqlite::params![sid, fp],
            )?;
            if deleted > 0 {
                tx.execute("DELETE FROM messages WHERE session_id = ?", [sid])?;
            }
        }
    }
    Ok(())
}

const MAX_DIR_DEPTH: usize = 10;

fn collect_jsonl_files(
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    verbose: bool,
) {
    collect_jsonl_files_inner(dir, out, source, 0, verbose);
}

fn collect_jsonl_files_inner(
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    depth: usize,
    verbose: bool,
) {
    if depth >= MAX_DIR_DEPTH {
        if verbose {
            eprintln!(
                "Warning: depth limit ({MAX_DIR_DEPTH}) reached at {}",
                dir.display()
            );
        }
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Warning: cannot read {}: {e}", dir.display());
            return;
        }
    };
    for entry in entries {
        let entry = match entry {
            Ok(e) => e,
            Err(e) => {
                eprintln!("Warning: cannot read entry in {}: {e}", dir.display());
                continue;
            }
        };
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(e) => {
                eprintln!(
                    "Warning: cannot get file type for {}: {e}",
                    entry.path().display()
                );
                continue;
            }
        };
        if ft.is_symlink() {
            if verbose {
                eprintln!("Warning: skipping symlink {}", entry.path().display());
            }
            continue;
        }
        let path = entry.path();
        if ft.is_dir() {
            collect_jsonl_files_inner(&path, out, source, depth + 1, verbose);
        } else if path.extension().is_some_and(|ext| ext == "jsonl") {
            out.push((path, source));
        }
    }
}

pub(crate) struct ChunkStats {
    pub chunks_created: usize,
}

pub(crate) fn index_chunks(conn: &mut Connection, verbose: bool) -> Result<ChunkStats> {
    let sessions: Vec<(String, Option<i64>)> = {
        let mut stmt = conn.prepare(
            "SELECT s.session_id, s.timestamp FROM sessions s \
             WHERE NOT EXISTS (SELECT 1 FROM qa_chunks c WHERE c.session_id = s.session_id)",
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };

    if sessions.is_empty() {
        return Ok(ChunkStats { chunks_created: 0 });
    }

    if verbose {
        eprintln!("Chunking {} sessions", sessions.len());
    }

    let tx = conn.transaction()?;
    let mut chunks_created = 0;
    for (session_id, timestamp) in &sessions {
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
                let role = match role_str.as_str() {
                    "user" => Role::User,
                    "assistant" => Role::Assistant,
                    _ => continue,
                };
                msgs.push(Message { role, text });
            }
            msgs
        };

        let chunks = chunker::chunk_messages(session_id, &messages, *timestamp);

        for chunk in &chunks {
            tx.execute(
                "INSERT INTO qa_chunks (session_id, user_text, assistant_text, content, timestamp, chunk_hash) \
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
                rusqlite::params![
                    chunk.session_id,
                    chunk.user_text,
                    chunk.assistant_text,
                    chunk.content,
                    chunk.timestamp,
                    chunk.chunk_hash,
                ],
            )?;
            chunks_created += 1;
        }
    }

    tx.commit()?;

    Ok(ChunkStats { chunks_created })
}

pub(crate) struct EmbedResult {
    pub embedded: usize,
    pub stopped_at_error: Option<String>,
}

impl EmbedResult {
    pub(crate) fn warn_if_stopped(&self) {
        if let Some(ref err) = self.stopped_at_error {
            eprintln!("Warning: embedding stopped early: {err}");
        }
    }
}

const EMBED_BATCH_SIZE: usize = 32;

/// Embeds chunks in batches, committing each batch separately.
/// Stops on first batch failure; previously committed batches are preserved.
fn embed_chunks(
    conn: &mut Connection,
    embedder: &mut dyn Embed,
    chunks: &[(i64, String)],
) -> Result<EmbedResult> {
    if chunks.is_empty() {
        return Ok(EmbedResult {
            embedded: 0,
            stopped_at_error: None,
        });
    }

    // Sort by content length to minimize padding waste within each batch
    let mut sorted: Vec<usize> = (0..chunks.len()).collect();
    sorted.sort_by_key(|&i| chunks[i].1.len());

    let total = chunks.len();
    let show_progress = total >= 100;
    let mut embedded = 0;
    let mut stopped_at_error = None;

    for batch_idx in sorted.chunks(EMBED_BATCH_SIZE) {
        let texts: Vec<&str> = batch_idx.iter().map(|&i| chunks[i].1.as_str()).collect();
        match embedder.embed_documents_batch(&texts) {
            Ok(embeddings) => {
                let tx = conn.transaction()?;
                for (embedding, &i) in embeddings.iter().zip(batch_idx) {
                    let embedding_bytes: &[u8] = bytemuck::cast_slice(embedding);
                    tx.execute(
                        "INSERT INTO vec_chunks (chunk_id, embedding) VALUES (?1, ?2)",
                        rusqlite::params![chunks[i].0, embedding_bytes],
                    )?;
                }
                tx.commit()?;
                embedded += embeddings.len();
                if show_progress {
                    eprint!("\r  {embedded}/{total} chunks embedded");
                }
            }
            Err(e) => {
                stopped_at_error = Some(format!("batch: {e}"));
                break;
            }
        }
    }
    if show_progress {
        eprintln!();
    }

    Ok(EmbedResult {
        embedded,
        stopped_at_error,
    })
}

pub(crate) fn embed_near_sessions(
    conn: &mut Connection,
    embedder: &mut dyn Embed,
    session_ids: &[String],
    budget: usize,
) -> Result<EmbedResult> {
    if session_ids.is_empty() || budget == 0 {
        return Ok(EmbedResult {
            embedded: 0,
            stopped_at_error: None,
        });
    }

    let placeholders = session_ids
        .iter()
        .map(|_| "?")
        .collect::<Vec<_>>()
        .join(", ");
    let sql = format!(
        "SELECT c.id, c.content FROM qa_chunks c \
         WHERE c.session_id IN ({placeholders}) \
         AND NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id) \
         ORDER BY c.id \
         LIMIT {budget}"
    );

    let missing: Vec<(i64, String)> = {
        let mut stmt = conn.prepare(&sql)?;
        let refs: Vec<&dyn rusqlite::types::ToSql> = session_ids
            .iter()
            .map(|s| s as &dyn rusqlite::types::ToSql)
            .collect();
        let rows = stmt.query_map(refs.as_slice(), |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };

    embed_chunks(conn, embedder, &missing)
}

pub(crate) fn embed_recent_chunks(
    conn: &mut Connection,
    embedder: &mut dyn Embed,
    budget: usize,
) -> Result<EmbedResult> {
    if budget == 0 {
        return Ok(EmbedResult {
            embedded: 0,
            stopped_at_error: None,
        });
    }

    let missing: Vec<(i64, String)> = {
        let mut stmt = conn.prepare(
            "SELECT c.id, c.content FROM qa_chunks c \
             WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id) \
             ORDER BY c.timestamp DESC NULLS LAST \
             LIMIT ?",
        )?;
        let rows = stmt.query_map([budget as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<std::result::Result<Vec<_>, _>>()?
    };

    embed_chunks(conn, embedder, &missing)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::setup_test_db;
    use tempfile::TempDir;

    #[test]
    fn test_index_from_dirs_indexes_and_skips() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                verbose: false,
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
                verbose: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.indexed, 0);
        assert_eq!(stats2.total_sessions, 1);
    }

    #[test]
    fn test_index_from_dirs_force_reindexes() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                verbose: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 1);

        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: true,
                verbose: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.indexed, 1);
    }

    #[test]
    fn test_orphan_cleanup_removes_deleted_files() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        let f1 = claude_dir.join("session1.jsonl");
        let f2 = claude_dir.join("session2.jsonl");
        std::fs::write(&f1, r#"{"type":"user","message":{"role":"user","content":"one"},"timestamp":"2026-03-01T00:00:00Z"}"#).unwrap();
        std::fs::write(&f2, r#"{"type":"user","message":{"role":"user","content":"two"},"timestamp":"2026-03-01T00:00:00Z"}"#).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                verbose: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats.indexed, 2);
        assert_eq!(stats.total_sessions, 2);

        std::fs::remove_file(&f2).unwrap();
        let stats2 = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: true,
                verbose: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();
        assert_eq!(stats2.total_sessions, 1);
    }

    #[test]
    fn test_session_id_collision_cleans_old_messages() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let dir_a = tmp.path().join("claude_a");
        let dir_b = tmp.path().join("claude_b");
        std::fs::create_dir_all(&dir_a).unwrap();
        std::fs::create_dir_all(&dir_b).unwrap();
        std::fs::write(
            dir_a.join("collision.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"from dir a"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();
        std::fs::write(
            dir_b.join("collision.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"from dir b"},"timestamp":"2026-03-02T00:00:00Z"}"#,
        ).unwrap();

        let empty_dir = tmp.path().join("empty");

        let stats = index_from_dirs(
            &mut conn,
            &IndexOptions {
                force: false,
                verbose: false,
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
                verbose: false,
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
        std::fs::create_dir_all(&deep).unwrap();
        std::fs::write(deep.join("deep.jsonl"), "{}").unwrap();

        std::fs::write(tmp.path().join("shallow.jsonl"), "{}").unwrap();

        let mut files = Vec::new();
        collect_jsonl_files(tmp.path(), &mut files, Source::Claude, false);

        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("shallow.jsonl"));
    }

    #[cfg(unix)]
    #[test]
    fn test_collect_skips_symlinks() {
        let tmp = TempDir::new().unwrap();

        let real_file = tmp.path().join("real.jsonl");
        std::fs::write(&real_file, "{}").unwrap();

        let link = tmp.path().join("link.jsonl");
        std::os::unix::fs::symlink(&real_file, &link).unwrap();

        let mut files = Vec::new();
        collect_jsonl_files(tmp.path(), &mut files, Source::Claude, false);

        assert_eq!(files.len(), 1);
        assert!(files[0].0.ends_with("real.jsonl"));
    }

    // T-011: incremental chunk index — unchanged sessions skip re-chunking (FR-004)
    #[test]
    fn test_011_index_chunks_incremental() {
        let (_dir, mut conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
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
                verbose: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            },
        )
        .unwrap();

        let stats1 = index_chunks(&mut conn, false).unwrap();
        assert_eq!(stats1.chunks_created, 1);

        let stats2 = index_chunks(&mut conn, false).unwrap();
        assert_eq!(stats2.chunks_created, 0);
    }

    // -- T-009, T-010: resolve_claude_dir env var priority -------------------------

    static ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

    /// SAFETY: caller must hold ENV_LOCK.
    unsafe fn apply_env(key: &str, val: Option<&str>) {
        match val {
            Some(v) => std::env::set_var(key, v),
            None => std::env::remove_var(key),
        }
    }

    fn with_env_vars<F: FnOnce()>(vars: &[(&str, Option<&str>)], f: F) {
        let _lock = ENV_LOCK.lock().unwrap();
        let saved: Vec<(&str, Option<String>)> = vars
            .iter()
            .map(|(k, _)| (*k, std::env::var(*k).ok()))
            .collect();
        for (k, v) in vars {
            // SAFETY: serialized by ENV_LOCK; no other threads access these env vars.
            unsafe { apply_env(k, *v) };
        }
        f();
        for (k, old) in &saved {
            // SAFETY: restoring original values under the same lock.
            unsafe { apply_env(k, old.as_deref()) };
        }
    }

    // T-009: RECALL_CLAUDE_DIR overrides CLAUDE_CONFIG_DIR (FR-003)
    #[test]
    fn test_009_recall_claude_dir_overrides_claude_config_dir() {
        let home = Path::new("/fake/home");
        with_env_vars(
            &[
                ("RECALL_CLAUDE_DIR", Some("/custom/recall/dir")),
                ("CLAUDE_CONFIG_DIR", Some("/custom/config/dir")),
            ],
            || {
                let result = resolve_claude_dir(home);
                assert_eq!(
                    result,
                    PathBuf::from("/custom/recall/dir"),
                    "T-009: RECALL_CLAUDE_DIR should take priority over CLAUDE_CONFIG_DIR"
                );
            },
        );
    }

    // T-010: CLAUDE_CONFIG_DIR used as fallback (FR-003)
    #[test]
    fn test_010_claude_config_dir_used_as_fallback() {
        let home = Path::new("/fake/home");
        with_env_vars(
            &[
                ("RECALL_CLAUDE_DIR", None),
                ("CLAUDE_CONFIG_DIR", Some("/custom/config")),
            ],
            || {
                let result = resolve_claude_dir(home);
                assert_eq!(
                    result,
                    PathBuf::from("/custom/config/projects"),
                    "T-010: CLAUDE_CONFIG_DIR/projects should be used when RECALL_CLAUDE_DIR is unset"
                );
            },
        );
    }
}
