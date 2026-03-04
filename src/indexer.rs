use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use rusqlite::Connection;

use crate::parser::{parse_claude_session, parse_codex_session, Source};

pub struct IndexStats {
    pub indexed: usize,
    pub skipped: usize,
    pub total_sessions: i64,
    pub total_messages: i64,
    pub elapsed_secs: f64,
}

pub fn index_sessions(conn: &Connection, force: bool, verbose: bool) -> Result<IndexStats> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let claude_dir = home.join(".claude").join("projects");
    let codex_dir = home.join(".codex").join("sessions");
    index_from_dirs(conn, force, verbose, &claude_dir, &codex_dir)
}

pub(crate) fn index_from_dirs(
    conn: &Connection,
    force: bool,
    verbose: bool,
    claude_dir: &Path,
    codex_dir: &Path,
) -> Result<IndexStats> {
    let start = Instant::now();

    if force {
        conn.execute_batch("DELETE FROM sessions; DELETE FROM messages;")?;
    }

    let mut existing: HashMap<String, (String, f64)> = HashMap::new();
    let mut stmt = conn.prepare("SELECT file_path, session_id, mtime FROM sessions")?;
    let rows = stmt.query_map([], |row| {
        Ok((
            row.get::<_, String>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, f64>(2)?,
        ))
    })?;
    for row in rows {
        let (fp, sid, mt) = row?;
        existing.insert(fp, (sid, mt));
    }

    let mut sources: Vec<(PathBuf, Source)> = Vec::new();

    if claude_dir.is_dir() {
        collect_jsonl_files(claude_dir, &mut sources, Source::Claude);
    }
    if codex_dir.is_dir() {
        collect_jsonl_files(codex_dir, &mut sources, Source::Codex);
    }

    if verbose {
        eprintln!("Found {} source files", sources.len());
    }

    let tx = conn.unchecked_transaction()
        .context("Failed to begin transaction")?;

    // Disable FTS5 auto-merge during bulk insert for performance.
    // Inside tx so it rolls back if the transaction fails.
    tx.execute(
        "INSERT INTO messages(messages, rank) VALUES('automerge', 0)",
        [],
    )?;

    let mut indexed = 0usize;
    let mut skipped = 0usize;

    for (fpath, source) in &sources {
        let fpath_str = fpath.to_string_lossy().to_string();
        let mtime = match std::fs::metadata(fpath) {
            Ok(m) => m
                .modified()
                .ok()
                .and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok())
                .map(|d| d.as_secs_f64())
                .unwrap_or(0.0),
            Err(_) => continue,
        };

        let cached = existing.get(&fpath_str);

        if !force
            && let Some((_, old_mtime)) = cached
            && (*old_mtime - mtime).abs() < 0.001
        {
            continue;
        }

        if let Some((old_sid, _)) = cached {
            tx.execute("DELETE FROM sessions WHERE session_id = ?", [old_sid])?;
            tx.execute("DELETE FROM messages WHERE session_id = ?", [old_sid])?;
        }

        let result = match source {
            Source::Claude => parse_claude_session(fpath),
            Source::Codex => parse_codex_session(fpath),
        };

        let parsed = match result {
            Ok(Some(p)) => p,
            Ok(None) => {
                skipped += 1;
                continue;
            }
            Err(e) => {
                eprintln!("Warning: {}: {e}", fpath.display());
                skipped += 1;
                continue;
            }
        };

        tx.execute(
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

        if verbose && parsed.skipped_lines > 0 {
            eprintln!(
                "Warning: {} skipped lines in {}",
                parsed.skipped_lines,
                fpath.display()
            );
        }

        for msg in &parsed.messages {
            tx.execute(
                "INSERT INTO messages (session_id, role, text) VALUES (?1, ?2, ?3)",
                rusqlite::params![parsed.metadata.session_id, msg.role.as_str(), msg.text],
            )?;
        }

        indexed += 1;
    }

    tx.commit().context("Failed to commit transaction")?;

    if indexed > 0 {
        conn.execute("INSERT INTO messages(messages) VALUES('optimize')", [])?;
    }
    conn.execute(
        "INSERT INTO messages(messages, rank) VALUES('automerge', 4)",
        [],
    )?;

    let total_sessions: i64 =
        conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    let total_messages: i64 =
        conn.query_row("SELECT COUNT(*) FROM messages", [], |r| r.get(0))?;

    Ok(IndexStats {
        indexed,
        skipped,
        total_sessions,
        total_messages,
        elapsed_secs: start.elapsed().as_secs_f64(),
    })
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
        return;
    }
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Warning: cannot read {}: {e}", dir.display());
            return;
        }
    };
    for entry in entries.flatten() {
        let ft = match entry.file_type() {
            Ok(ft) => ft,
            Err(_) => continue,
        };
        if ft.is_symlink() {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::open_db;
    use tempfile::TempDir;

    fn setup_test_db() -> (TempDir, Connection) {
        let dir = TempDir::new().unwrap();
        let db_path = dir.path().join("test.db");
        let conn = open_db(&db_path).unwrap();
        (dir, conn)
    }

    // T-003: Incremental Indexing (integration tests via index_from_dirs)

    #[test]
    fn test_index_from_dirs_indexes_and_skips() {
        let (_dir, conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(&conn, false, false, &claude_dir, &codex_dir).unwrap();
        assert_eq!(stats.indexed, 1);
        assert_eq!(stats.total_sessions, 1);
        assert!(stats.total_messages >= 1);

        // Second call should skip (mtime unchanged)
        let stats2 = index_from_dirs(&conn, false, false, &claude_dir, &codex_dir).unwrap();
        assert_eq!(stats2.indexed, 0);
        assert_eq!(stats2.total_sessions, 1);
    }

    #[test]
    fn test_index_from_dirs_force_reindexes() {
        let (_dir, conn) = setup_test_db();
        let tmp = TempDir::new().unwrap();

        let claude_dir = tmp.path().join("claude_projects");
        std::fs::create_dir_all(&claude_dir).unwrap();
        std::fs::write(
            claude_dir.join("session1.jsonl"),
            r#"{"type":"user","message":{"role":"user","content":"hello"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        ).unwrap();

        let codex_dir = tmp.path().join("codex_sessions");

        let stats = index_from_dirs(&conn, false, false, &claude_dir, &codex_dir).unwrap();
        assert_eq!(stats.indexed, 1);

        // Force reindex should re-process
        let stats2 = index_from_dirs(&conn, true, false, &claude_dir, &codex_dir).unwrap();
        assert_eq!(stats2.indexed, 1);
    }
}
