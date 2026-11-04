use std::collections::HashMap;
use std::result::Result as StdResult;

use anyhow::Result;
use rusqlite::{Connection, Transaction, TransactionBehavior};
use tracing::{debug, info};

use super::INDEX_RULES_VERSION;
use crate::chunker;
use crate::index_observer::Observer;
use crate::parser::{Message, Role};

#[derive(Default)]
pub(crate) struct ChunkStats {
    pub chunks_created: usize,
    pub sessions_chunked: usize,
    pub message_batches: usize,
    pub message_scans: usize,
    pub messages_read: usize,
}

pub(super) const CHUNK_BATCH_SESSIONS: usize = 64;
pub(super) const CHUNK_BATCH_BYTES: i64 = 8 * 1024 * 1024;

pub(super) const STAGE_CHUNK_MESSAGES: &str =
    "INSERT INTO chunk_message_rows (session_id, message_rowid, text_bytes)
     SELECT m.session_id, m.rowid, length(CAST(m.text AS BLOB))
     FROM messages m CROSS JOIN sessions s ON s.session_id = m.session_id
     WHERE s.chunks_indexed IS NULL OR s.chunks_indexed < 0";
pub(super) const READ_CHUNK_BATCH: &str = "SELECT r.session_id, m.rowid, m.role, m.text
     FROM chunk_message_rows r CROSS JOIN messages m ON m.rowid = r.message_rowid
     WHERE r.session_id >= ?1 AND r.session_id <= ?2
     ORDER BY r.session_id, r.message_rowid";

type PendingChunkSession = (String, Option<i64>, i64);

pub(super) fn chunk_batch_len(sessions: &[PendingChunkSession]) -> usize {
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

    let sessions = stage_pending_sessions(&tx)?;
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
        let mut messages = read_chunk_batch(&tx, batch, &mut stats)?;
        stats.message_batches += 1;
        for (session_id, timestamp, _) in batch {
            let body = messages.remove(session_id).unwrap_or_default();
            let chunks = chunker::chunk_messages(session_id, &body, *timestamp);
            if chunks.is_empty() {
                empty += 1;
            }
            store_session_chunks(&tx, session_id, &chunks, &mut stats)?;
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

fn stage_pending_sessions(tx: &Transaction) -> Result<Vec<PendingChunkSession>> {
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
    Ok(sessions)
}

fn read_chunk_batch(
    tx: &Transaction,
    batch: &[PendingChunkSession],
    stats: &mut ChunkStats,
) -> Result<HashMap<String, Vec<(i64, Message)>>> {
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
    Ok(messages)
}

fn store_session_chunks(
    tx: &Transaction,
    session_id: &str,
    chunks: &[chunker::QAChunk],
    stats: &mut ChunkStats,
) -> Result<()> {
    for chunk in chunks {
        tx.execute(
            "INSERT INTO qa_chunks (session_id, content, timestamp, src_rowid_lo, src_rowid_hi)
             VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![
                chunk.session_id,
                chunk.content,
                chunk.timestamp,
                chunk.src_rowid_lo,
                chunk.src_rowid_hi
            ],
        )?;
        stats.chunks_created += 1;
    }
    tx.execute(
        "UPDATE sessions SET chunks_indexed = CASE WHEN chunks_indexed = -?1 THEN ?1 ELSE 0 END WHERE session_id = ?2",
        rusqlite::params![INDEX_RULES_VERSION, session_id],
    )?;
    Ok(())
}
