//! Manual scale evidence for #320. Kept out of routine CI latency; run with
//! cargo test --locked --bin recall chunk_index_scale -- --ignored --nocapture --test-threads=1
use super::*;
use crate::db::setup_test_db;

// The pre-#320 chunk pass from b4e4eb8, using the unchanged legacy reader and
// chunker. No timing assertions: cache, SQLite builds and hardware vary.
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
        // 61 assistant-only conversations, 589 messages. The remaining 20,519
        // conversations contain one Q&A each, with 245,168 messages in total.
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
        // Alternate order within each pair, using the same WAL DB and warm cache.
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
    // Isolate chunking cost; lifecycle tests cover JSONL re-parse/invalidation.
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
