use std::path::Path;
use std::result::Result as StdResult;

use anyhow::Result;
use rusqlite::{Connection, Transaction};
use tracing::debug;

use super::diagnostics::{save_parse_diagnostics, save_read_error};
use super::insert_session_files;
use crate::chunker;
use crate::parser::{Message, Role, Source, parse_session_including_empty};

pub(super) fn read_session_messages(
    tx: &Transaction,
    session_id: &str,
) -> Result<Vec<(i64, Message)>> {
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
