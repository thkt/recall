use std::result::Result as StdResult;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use amici::storage::anon_placeholders;
use anyhow::Result;
use rurico::embed::Embed;
#[cfg(test)]
use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS, EmbedError};
use rusqlite::Connection;
use rusqlite::types::ToSql;
use tracing::warn;

#[derive(Default)]
pub(crate) struct EmbedResult {
    pub embedded: usize,
    pub failed_count: usize,
    pub first_error: Option<String>,
}

impl EmbedResult {
    pub(crate) fn warn_if_batches_failed(&self) {
        if self.failed_count > 0 {
            // first_error is always Some when failed_count > 0 (both are set in
            // embed_chunks' Err arm together); the fallback is defensive only.
            warn!(
                failed_count = self.failed_count,
                first_error = self.first_error.as_deref().unwrap_or("unknown"),
                "embedding skipped chunks after batch failures"
            );
        }
    }
}

pub(crate) const EMBED_BATCH_SIZE: usize = 128;

/// Reinterprets an f32 slice as its raw byte view for sqlite-vec storage.
/// Replaces `rurico::storage::f32_as_bytes`, removed in rurico a573655 (#78).
pub(crate) fn f32_as_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

fn embed_chunks(
    conn: &mut Connection,
    embedder: &dyn Embed,
    chunks: &[(i64, String)],
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<EmbedResult> {
    if chunks.is_empty() {
        return Ok(EmbedResult::default());
    }

    let mut sorted: Vec<usize> = (0..chunks.len()).collect();
    sorted.sort_by_key(|&i| chunks[i].1.len());

    let total = chunks.len();
    let mut embedded = 0;
    let mut failed_count = 0;
    let mut first_error = None;

    for batch_idx in sorted.chunks(EMBED_BATCH_SIZE) {
        let texts: Vec<&str> = batch_idx.iter().map(|&i| chunks[i].1.as_str()).collect();
        match embedder.embed_documents_batch(&texts) {
            Ok(embeddings) => {
                let tx = conn.transaction()?;
                // Idempotent replay guard for stale concurrent work that reached
                // this tx after the NOT EXISTS pending query. Batched into one
                // IN-delete: vec0's +chunk_id is an unindexed auxiliary column, so
                // each `DELETE WHERE chunk_id = ?` is a full O(N) scan (EXPLAIN
                // reports "SCAN vec_chunks VIRTUAL TABLE") — per-chunk deletes were
                // O(batch × N). Keyed on batch_idx (non-empty by the early return);
                // on the NOT EXISTS path nothing matches, so it is a no-op there.
                let batch_ids: Vec<i64> = batch_idx.iter().map(|&i| chunks[i].0).collect();
                let placeholders = anon_placeholders(batch_ids.len());
                tx.execute(
                    &format!("DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})"),
                    rusqlite::params_from_iter(batch_ids.iter()),
                )?;
                for (chunked, &i) in embeddings.iter().zip(batch_idx) {
                    for (sub_idx, sub_emb) in chunked.chunks.iter().enumerate() {
                        let embedding_bytes = f32_as_bytes(sub_emb);
                        tx.execute(
                            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) \
                             VALUES (?1, ?2, ?3)",
                            rusqlite::params![embedding_bytes, chunks[i].0, sub_idx as i64],
                        )?;
                    }
                }
                tx.commit()?;
                embedded += embeddings.len();
                if let Some(cb) = &on_progress {
                    cb(embedded, total);
                }
            }
            Err(e) => {
                // Skip the failed batch and keep going: a poison chunk fails its
                // whole ≤128-batch (all-or-nothing) but must not block the rest of
                // the backlog. The chunks stay pending (no tx committed here) for
                // the next index to retry via the NOT EXISTS gate.
                failed_count += batch_idx.len();
                if first_error.is_none() {
                    first_error = Some(format!("batch: {e}"));
                }
            }
        }
    }

    Ok(EmbedResult {
        embedded,
        failed_count,
        first_error,
    })
}

fn query_and_embed(
    conn: &mut Connection,
    embedder: &dyn Embed,
    sql: &str,
    params: &[&dyn ToSql],
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<EmbedResult> {
    let missing: Vec<(i64, String)> = {
        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(params, |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
        })?;
        rows.collect::<StdResult<Vec<_>, _>>()?
    };

    embed_chunks(conn, embedder, &missing, on_progress)
}

pub(crate) fn embed_recent_chunks(
    conn: &mut Connection,
    embedder: &dyn Embed,
    budget: usize,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<EmbedResult> {
    if budget == 0 {
        return Ok(EmbedResult::default());
    }

    let budget_i64 = budget as i64;
    query_and_embed(
        conn,
        embedder,
        "SELECT c.id, c.content FROM qa_chunks c \
         WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id) \
         ORDER BY c.timestamp DESC NULLS LAST \
         LIMIT ?",
        &[&budget_i64 as &dyn ToSql],
        on_progress,
    )
}

/// Returns deterministic 768-dim vectors derived from text bytes.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MockEmbedder {
    call_count: AtomicUsize,
    fail_after: Option<usize>,
    fail_on_text: Option<String>,
}

#[cfg(test)]
impl MockEmbedder {
    pub(crate) fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: None,
            fail_on_text: None,
        }
    }

    pub(crate) fn failing_after(n: usize) -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: Some(n),
            fail_on_text: None,
        }
    }

    /// A mock that fails any batch containing `text` (all-or-nothing, matching
    /// `embed_chunks`'s batch semantics). T-001 uses it to poison one batch.
    pub(crate) fn failing_on_text(text: &str) -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: None,
            fail_on_text: Some(text.to_owned()),
        }
    }

    pub(crate) fn deterministic_vector(text: &str) -> Vec<f32> {
        let dims = EMBEDDING_DIMS;
        let mut v = vec![0.0f32; dims];
        for (i, b) in text.bytes().enumerate() {
            v[i % dims] += b as f32;
        }
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for x in &mut v {
                *x /= norm;
            }
        }
        v
    }
}

#[cfg(test)]
impl Embed for MockEmbedder {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if let Some(limit) = self.fail_after {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count >= limit {
                return Err(EmbedError::Inference("mock failure".to_owned()));
            }
        }
        Ok(Self::deterministic_vector(text))
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        if let Some(limit) = self.fail_after {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count >= limit {
                return Err(EmbedError::Inference("mock failure".to_owned()));
            }
        }
        Ok(ChunkedEmbedding::new(vec![Self::deterministic_vector(
            text,
        )]))
    }

    /// Explicit impl of the production dispatch target: `embed_chunks` calls this
    /// method, so the poison check lives here (all-or-nothing per batch), not in
    /// the per-item delegate. Being explicit also survives rurico revisions where
    /// the trait's default body is removed (required at rurico HEAD).
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        if let Some(poison) = self.fail_on_text.as_deref()
            && texts.contains(&poison)
        {
            return Err(EmbedError::Inference(format!("poison text: {poison}")));
        }
        texts.iter().map(|t| self.embed_document(t)).collect()
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::deterministic_vector(text))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::db::setup_test_db;

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
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        for i in 0..3 {
            conn.execute(
                "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
                 VALUES (?1, 's1', ?2, 0)",
                rusqlite::params![i + 1, format!("content {i}")],
            )
            .unwrap();
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
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (1, 's1', 'content 0', 0)",
            [],
        )
        .unwrap();

        let embedder = MockEmbedder::new();
        let chunks: Vec<(i64, String)> = vec![(1, "content 0".into())];

        let first = embed_chunks(&mut conn, &embedder, &chunks, None).unwrap();
        assert_eq!(first.embedded, 1);
        let second = embed_chunks(&mut conn, &embedder, &chunks, None).unwrap();
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
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        for id in 1..=count as i64 {
            conn.execute(
                "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
                 VALUES (?1, 's1', ?2, 0)",
                rusqlite::params![id, content(id)],
            )
            .unwrap();
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
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (129, 's1', 'x', 0)",
            [],
        )
        .unwrap();

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
        let result = embed_chunks(&mut conn, &embedder, &chunks, None).unwrap();

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
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (129, 's1', 'x', 0)",
            [],
        )
        .unwrap();

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
        embed_chunks(&mut conn, &embedder, &chunks, None).unwrap();

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
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (129, 's1', 'x', 0)",
            [],
        )
        .unwrap();

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
}
