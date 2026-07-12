use std::sync::Mutex;

use super::*;
use crate::db::{seed_chunk, seed_session, setup_test_db};

#[test]
fn test_embed_recent_chunks_budget_zero() {
    let (_dir, mut conn) = setup_test_db();
    let embedder = MockEmbedder::new();
    let result = embed_recent_chunks(&mut conn, &embedder, 0, None).unwrap();
    assert_eq!(result.embedded, 0);
    assert_eq!(result.failed_count, 0);
}

#[test]
fn test_embed_chunks_progress_callback() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    for i in 0..3 {
        seed_chunk(&conn, i + 1, &format!("content {i}"));
    }

    let embedder = MockEmbedder::new();
    let chunks: Vec<(i64, String)> = vec![
        (1, "content 0".into()),
        (2, "content 1".into()),
        (3, "content 2".into()),
    ];

    let calls = Mutex::new(Vec::new());
    let result = embed_chunks(
        &mut conn,
        &embedder,
        &chunks,
        Some(&|done, total| {
            calls.lock().unwrap().push((done, total));
        }),
        &EmbedOptions::default(),
    )
    .unwrap();

    assert_eq!(result.embedded, 3);
    let calls = calls.into_inner().unwrap();
    assert!(!calls.is_empty());
    assert_eq!(calls.last().unwrap(), &(3, 3));
}

#[test]
fn test_embed_chunks_replaces_existing_vectors_for_stale_pending_work() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_chunk(&conn, 1, "content 0");

    let embedder = MockEmbedder::new();
    let chunks: Vec<(i64, String)> = vec![(1, "content 0".into())];

    let first = embed_chunks(
        &mut conn,
        &embedder,
        &chunks,
        None,
        &EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(first.embedded, 1);
    let second = embed_chunks(
        &mut conn,
        &embedder,
        &chunks,
        None,
        &EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(second.embedded, 1);

    let vec_count: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = 1",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(vec_count, 1, "re-embedding must not duplicate vec rows");
}

/// Seed `count` qa_chunks whose content is the given closure of the row id, so
/// a test controls each chunk's text (and thus its length-sort batch). Returns
/// nothing; the rows live in `conn`.
fn seed_chunks(conn: &Connection, count: usize, content: impl Fn(i64) -> String) {
    seed_session(conn, "s1");
    for id in 1..=count as i64 {
        seed_chunk(conn, id, &content(id));
    }
}

// Decision table for embed_chunks under a one-batch poison (EMBED_BATCH_SIZE=128):
// | batch | contains poison "x" | embed_documents_batch | committed | counted as |
// | ----- | ------------------- | --------------------- | --------- | ---------- |
// | 1     | yes (sorted first)  | Err (all-or-nothing)  | no        | failed     |
// | 2     | no                  | Ok                    | yes       | embedded   |
//
// T-001 (FR-001, FR-002): a poison batch does not stop the next batch. Given 129
// chunks — poison text "x" (shortest, so it sorts into batch 1) plus 128 longer
// "content_NNNN" texts — and failing_on_text("x"), embed_chunks continues past
// the failed batch 1: batch 2 (1 chunk) embeds, batch 1 (128 chunks) is counted
// failed, and the first batch error is reported.
// Perspective: branch (the Err arm now continues, not breaks) + boundary (the
// batch-1/batch-2 split at EMBED_BATCH_SIZE).
#[test]
fn test_embed_chunks_poison_batch_does_not_block_next_batch() {
    let (_dir, mut conn) = setup_test_db();
    // 128 healthy chunks (uniform length 12, all longer than "x") + 1 poison.
    // Ascending length-sort puts "x" at index 0 → batch 1 (with 127 healthy);
    // the single remaining healthy chunk is batch 2.
    seed_chunks(&conn, 128, |id| format!("content_{id:04}"));
    seed_chunk(&conn, 129, "x");

    let chunks: Vec<(i64, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, content FROM qa_chunks ORDER BY id")
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(StdResult::unwrap)
            .collect()
    };
    assert_eq!(chunks.len(), 129, "129 chunks span exactly two batches");

    let embedder = MockEmbedder::failing_on_text("x");
    let result = embed_chunks(
        &mut conn,
        &embedder,
        &chunks,
        None,
        &EmbedOptions::default(),
    )
    .unwrap();

    assert_eq!(
        result.embedded, 1,
        "batch 2 (the single non-poison chunk) embeds despite batch 1 failing"
    );
    assert_eq!(
        result.failed_count, 128,
        "the whole poison batch (128 chunks) is counted failed, not just the poison chunk"
    );
    let err = result.first_error.as_deref().unwrap_or_default();
    assert!(
        err.contains("poison text: x"),
        "first_error carries the batch error content, not a fallback, got: {err:?}"
    );
}

// T-002 (FR-003): chunks in a failed batch keep no embedding so the next index
// retries them. Given the same poison batch as T-001, after embed_chunks the
// poison batch's 128 chunks have no vec_chunks row (the failed batch's tx never
// commits), while batch 2's single chunk does. The pending set (NOT EXISTS) is
// exactly the failed batch.
// Perspective: hazard (a half-applied batch would leave a chunk embedded with no
// record, or strand a poison chunk as permanently skipped).
#[test]
fn test_embed_chunks_failed_batch_leaves_chunks_pending() {
    let (_dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 128, |id| format!("content_{id:04}"));
    seed_chunk(&conn, 129, "x");

    let chunks: Vec<(i64, String)> = {
        let mut stmt = conn
            .prepare("SELECT id, content FROM qa_chunks ORDER BY id")
            .unwrap();
        stmt.query_map([], |r| Ok((r.get(0)?, r.get(1)?)))
            .unwrap()
            .map(StdResult::unwrap)
            .collect()
    };

    let embedder = MockEmbedder::failing_on_text("x");
    embed_chunks(
        &mut conn,
        &embedder,
        &chunks,
        None,
        &EmbedOptions::default(),
    )
    .unwrap();

    let vec_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vec_count, 1,
        "only batch 2's chunk is embedded; the failed poison batch commits nothing"
    );
    let pending: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks c \
                 WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        pending, 128,
        "the failed batch's 128 chunks stay pending for the next index to retry"
    );
}

// T-007 (FR-002, FR-004): when every batch fails, the embed run is non-fatal and
// fully retryable. Given 1 pending chunk and failing_after(0) (every batch
// fails), embed_recent_chunks returns Ok (the index does not abort), counts the
// whole batch as failed, and leaves no vec row so the next index retries it.
// (Updated from the old T-009: stopped_at_error.is_some() → failed_count > 0.)
// Perspective: error (the all-fail path) + boundary (failed_count == every chunk).
#[test]
fn test_embed_recent_chunks_all_batches_fail_is_non_fatal_and_retryable() {
    let (_dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 1, |_| "pending content".to_owned());

    let embedder = MockEmbedder::failing_after(0);
    let result = embed_recent_chunks(&mut conn, &embedder, 10, None).unwrap();

    assert_eq!(result.embedded, 0, "every batch failed, so nothing embeds");
    assert_eq!(
        result.failed_count, 1,
        "the sole batch's chunk count is reported as failed"
    );
    let err = result.first_error.as_deref().unwrap_or_default();
    assert!(
        err.contains("mock failure"),
        "the all-fail run still records the first batch error, got: {err:?}"
    );
    let vec_count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        vec_count, 0,
        "the chunk stays pending for the next index to retry"
    );
}

// T-008 (FR-003): the failed batch is actually picked up by the next run — the
// cross-invocation half of T-002's "stays pending" promise (OUTCOME Behavior 2:
// repeated `recall index` fills the un-embedded chunks). Given the T-001 poison
// state (batch 1 failed, 128 pending), a second embed_recent_chunks with a
// healthy embedder embeds exactly those 128 and leaves nothing pending.
// Perspective: state (a two-invocation invariant that single-run C0/C1 misses).
#[test]
fn test_embed_recent_chunks_second_run_embeds_previously_failed_batch() {
    let (_dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 128, |id| format!("content_{id:04}"));
    seed_chunk(&conn, 129, "x");

    // Run 1: the poison strands batch 1 (its accounting is T-001/T-002's job).
    let poisoned = MockEmbedder::failing_on_text("x");
    let first = embed_recent_chunks(&mut conn, &poisoned, 129, None).unwrap();
    assert_eq!(
        first.embedded, 1,
        "precondition: only batch 2 embeds on run 1"
    );

    // Run 2: a healthy embedder picks the 128 up via the NOT EXISTS gate.
    let healthy = MockEmbedder::new();
    let second = embed_recent_chunks(&mut conn, &healthy, 129, None).unwrap();
    assert_eq!(
        second.embedded, 128,
        "the previously failed batch embeds on the retry run"
    );
    assert_eq!(second.failed_count, 0, "nothing fails on the healthy retry");
    let pending: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM qa_chunks c \
                 WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(pending, 0, "no chunk is left behind after the retry");
}

// #138: budget truncation must keep the newest chunks. With 3 pending chunks
// at distinct timestamps and budget 1, the embedded row is the one with the
// newest timestamp (ORDER BY timestamp DESC NULLS LAST). Pins the selection
// order across the one-pass rewrite of the pending query.
// Perspective: boundary (budget < pending).
#[test]
fn test_embed_recent_chunks_budget_prefers_newest_chunk() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    for (id, ts) in [(1_i64, 100_i64), (2, 300), (3, 200)] {
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
                 VALUES (?1, 's1', ?2, ?3)",
            rusqlite::params![id, format!("content_{id}"), ts],
        )
        .unwrap();
    }

    let embedder = MockEmbedder::new();
    let result = embed_recent_chunks(&mut conn, &embedder, 1, None).unwrap();

    assert_eq!(result.embedded, 1, "budget 1 embeds exactly one chunk");
    let embedded_id: i64 = conn
        .query_row("SELECT chunk_id FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(
        embedded_id, 2,
        "budget 1 must pick the chunk with the newest timestamp (ts=300)"
    );
}
