use std::result::Result as StdResult;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use amici::storage::anon_placeholders;
use anyhow::Result;
use rurico::embed::Embed;
#[cfg(test)]
use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS, EmbedError};
use rurico::storage::f32_as_bytes;
use rusqlite::Connection;
use rusqlite::types::ToSql;

#[derive(Default)]
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

pub(crate) const EMBED_BATCH_SIZE: usize = 32;

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
    let mut stopped_at_error = None;

    for batch_idx in sorted.chunks(EMBED_BATCH_SIZE) {
        let texts: Vec<&str> = batch_idx.iter().map(|&i| chunks[i].1.as_str()).collect();
        match embedder.embed_documents_batch(&texts) {
            Ok(embeddings) => {
                let tx = conn.transaction()?;
                for (chunked, &i) in embeddings.iter().zip(batch_idx) {
                    for (sub_idx, sub_emb) in chunked.chunks.iter().enumerate() {
                        let embedding_bytes = f32_as_bytes(sub_emb);
                        tx.execute(
                            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) \
                             VALUES (?1, ?2, ?3)",
                            rusqlite::params![embedding_bytes, chunks[i].0, sub_idx as i64],
                        )?;
                        let vec_rowid = tx.last_insert_rowid();
                        tx.execute(
                            "INSERT INTO embedded_chunk_ids (chunk_id, sub_idx, vec_rowid) \
                             VALUES (?1, ?2, ?3)",
                            rusqlite::params![chunks[i].0, sub_idx as i64, vec_rowid],
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
                stopped_at_error = Some(format!("batch: {e}"));
                break;
            }
        }
    }

    Ok(EmbedResult {
        embedded,
        stopped_at_error,
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

pub(crate) fn embed_near_sessions(
    conn: &mut Connection,
    embedder: &dyn Embed,
    session_ids: &[String],
    budget: usize,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<EmbedResult> {
    if session_ids.is_empty() || budget == 0 {
        return Ok(EmbedResult::default());
    }

    let placeholders = anon_placeholders(session_ids.len());
    let sql = format!(
        "SELECT c.id, c.content FROM qa_chunks c \
         WHERE c.session_id IN ({placeholders}) \
         AND NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id) \
         ORDER BY c.id \
         LIMIT ?"
    );

    let budget_i64 = budget as i64;
    let mut refs: Vec<&dyn ToSql> = session_ids.iter().map(|s| s as &dyn ToSql).collect();
    refs.push(&budget_i64);

    query_and_embed(conn, embedder, &sql, &refs, on_progress)
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
}

#[cfg(test)]
impl MockEmbedder {
    pub(crate) fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: None,
        }
    }

    pub(crate) fn failing_after(n: usize) -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: Some(n),
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
        Ok(ChunkedEmbedding {
            chunks: vec![Self::deterministic_vector(text)],
        })
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
        assert!(result.stopped_at_error.is_none());
    }

    #[test]
    fn test_embed_chunks_progress_callback() {
        let (_dir, mut conn) = setup_test_db();
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0)",
            [],
        )
        .unwrap();
        for i in 0..3 {
            conn.execute(
                "INSERT INTO qa_chunks (id, session_id, user_text, assistant_text, content, timestamp, chunk_hash) \
                 VALUES (?1, 's1', 'q', 'a', ?2, 0, ?3)",
                rusqlite::params![i + 1, format!("content {i}"), format!("hash{i}")],
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
}
