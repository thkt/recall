use std::collections::HashSet;
use std::result::Result as StdResult;
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use amici::storage::anon_placeholders;
use anyhow::Result;
use rurico::embed::Embed;
#[cfg(test)]
use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS, EmbedError};
use rusqlite::{Connection, Result as SqlResult, Row, TransactionBehavior};
use tracing::warn;

use crate::index_observer::Observer;

/// Per-call knobs for the index/rebuild embed pass, threaded from the
/// `--token-budget` / `--forward-pause-ms` CLI flags down to rurico's
/// `Embed::embed_documents_batch_with_options`. `None`/`None` (`Default`)
/// preserves the pre-flag behavior. rurico's `Embedder` applies `token_budget`
/// to forward-pass sub-batch sizing and sleeps `forward_pause` after each
/// forward pass (per-forward, not per-batch), so a small budget plus a pause
/// yields the GPU to interactive processes during large runs.
pub(crate) use rurico::embed::EmbedOptions;

/// Ceiling for `--token-budget` (index/rebuild CLI flag), matching rurico's
/// `TOKEN_BUDGET` forward-pass ceiling (`docs/decisions/0009-...`). Values above
/// this clamp down (never up) at clap parse time so an over-large flag cannot
/// push a forward pass past the Metal OOM ceiling the const was chosen to avoid
/// (rurico does not clamp; the caller owns this bound).
pub(crate) const TOKEN_BUDGET_CEILING: usize = 256_000;

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
            let first_error = self.first_error.as_deref().unwrap_or("unknown");
            warn!(
                failed_count = self.failed_count,
                first_error, "embedding skipped chunks after batch failures"
            );
        }
    }
}

pub(crate) const EMBED_BATCH_SIZE: usize = 128;

/// Identity and input captured together before inference. Generation survives
/// deletion of other rows and changes even when an ID is reused with equal text.
pub(crate) struct PendingChunk {
    id: i64,
    content: String,
    generation: i64,
}

/// Reinterprets an f32 slice as its raw byte view for sqlite-vec storage.
/// Replaces `rurico::storage::f32_as_bytes`, removed in rurico a573655 (#78).
pub(crate) fn f32_as_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

#[cfg(test)]
pub(crate) fn embed_chunks(
    conn: &mut Connection,
    embedder: &dyn Embed,
    chunks: &[PendingChunk],
    on_progress: Option<&dyn Fn(usize, usize)>,
    options: &EmbedOptions,
) -> Result<EmbedResult> {
    embed_chunks_observed(
        conn,
        embedder,
        chunks,
        on_progress,
        options,
        &Observer::default(),
    )
}

pub(crate) fn embed_chunks_observed(
    conn: &mut Connection,
    embedder: &dyn Embed,
    chunks: &[PendingChunk],
    on_progress: Option<&dyn Fn(usize, usize)>,
    options: &EmbedOptions,
    observer: &Observer,
) -> Result<EmbedResult> {
    observer.start_embedding(chunks.len());

    let mut sorted: Vec<usize> = (0..chunks.len()).collect();
    sorted.sort_by_key(|&i| chunks[i].content.len());

    let total = chunks.len();
    let mut embedded = 0;
    let mut failed_count = 0;
    let mut first_error = None;
    let mut stale = 0;
    let mut attempted = 0;

    for batch_idx in sorted.chunks(EMBED_BATCH_SIZE) {
        let texts: Vec<&str> = batch_idx
            .iter()
            .map(|&i| chunks[i].content.as_str())
            .collect();
        let inference = observer.stage("inference");
        let result = embedder.embed_documents_batch_with_options(&texts, options);
        inference.finish(if result.is_ok() {
            "complete; not saved"
        } else {
            "failed; not saved"
        });
        attempted += batch_idx.len();
        observer.count("chunks_unattempted", total - attempted);
        match result {
            Ok(embeddings) => {
                // Acquire the writer lock BEFORE reading the current generations:
                // no writer can replace a checked row before this batch commits.
                // Inference itself never holds the lock.
                let save = observer.stage("embedding_db_save");
                let saved: Result<usize> = (|| {
                    let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
                    let mut current = Vec::new();
                    {
                        let mut check = tx.prepare_cached(
                            "SELECT EXISTS(SELECT 1 FROM qa_chunks \
                         WHERE id = ?1 AND content = ?2 AND generation = ?3)",
                        )?;
                        for (chunked, &i) in embeddings.iter().zip(batch_idx) {
                            let chunk = &chunks[i];
                            if check.query_row(
                                rusqlite::params![chunk.id, chunk.content, chunk.generation],
                                |row| row.get::<_, bool>(0),
                            )? {
                                current.push((chunked, chunk.id));
                            }
                        }
                    }
                    if current.is_empty() {
                        return Ok(0);
                    }
                    // Delete only validated IDs, preserving another worker's current
                    // vectors when our result is stale. Keep one IN-delete per batch:
                    // vec0's +chunk_id is unindexed, so per-chunk deletes are O(batch × N).
                    let placeholders = anon_placeholders(current.len());
                    tx.execute(
                        &format!("DELETE FROM vec_chunks WHERE chunk_id IN ({placeholders})"),
                        rusqlite::params_from_iter(current.iter().map(|(_, id)| id)),
                    )?;
                    for (chunked, id) in &current {
                        for (sub_idx, sub_emb) in chunked.chunks().iter().enumerate() {
                            let embedding_bytes = f32_as_bytes(sub_emb);
                            tx.execute(
                                "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) \
                             VALUES (?1, ?2, ?3)",
                                rusqlite::params![embedding_bytes, id, sub_idx as i64],
                            )?;
                        }
                    }
                    tx.commit()?;
                    Ok(current.len())
                })();
                let saved = match saved {
                    Ok(saved) => saved,
                    Err(error) => {
                        observer.count("chunks_save_failed", batch_idx.len());
                        observer.embedding_progress();
                        return Err(error);
                    }
                };
                stale += batch_idx.len() - saved;
                embedded += saved;
                observer.count("chunks_stale", stale);
                observer.count("chunks_saved", embedded);
                observer.count("chunks_remaining_snapshot", total - embedded);
                save.finish(if saved == 0 {
                    "no current results to save"
                } else {
                    "committed"
                });
                if saved > 0
                    && let Some(cb) = &on_progress
                {
                    cb(embedded, total);
                }
            }
            Err(_) => {
                // Skip the failed batch and keep going: a poison chunk fails its
                // whole ≤128-batch (all-or-nothing) but must not block the rest of
                // the backlog. The chunks stay pending (no tx committed here) for
                // the next index to retry via the pending gate.
                failed_count += batch_idx.len();
                observer.count("chunks_failed", failed_count);
                if first_error.is_none() {
                    first_error = Some("batch inference failed".to_owned());
                }
            }
        }
        observer.embedding_progress();
    }

    Ok(EmbedResult {
        embedded,
        failed_count,
        first_error,
    })
}

/// Collect up to `budget` chunks that have no `vec_chunks` row, newest first.
///
/// One pass over each table (#138): the previous correlated `NOT EXISTS` form
/// re-scanned vec_chunks per qa_chunks row — its `+chunk_id` is an unindexed
/// auxiliary column (see the DELETE note in `embed_chunks`), so at 38k chunks
/// that was O(N×M) ≈ 15-20 minutes of silence before the first batch.
pub(crate) fn pending_chunks(conn: &Connection, budget: usize) -> Result<Vec<PendingChunk>> {
    if budget == 0 {
        return Ok(Vec::new());
    }

    let mut missing = Vec::new();
    visit_pending(conn, budget, true, |row| {
        missing.push(PendingChunk {
            id: row.get(0)?,
            content: row.get(1)?,
            generation: row.get(2)?,
        });
        Ok(())
    })?;
    Ok(missing)
}

/// Same single-pass selection without loading bodies when inference is unavailable.
pub(crate) fn pending_count(conn: &Connection) -> Result<usize> {
    visit_pending(conn, usize::MAX, false, |_| Ok(()))
}

fn visit_pending(
    conn: &Connection,
    budget: usize,
    with_content: bool,
    mut visit: impl FnMut(&Row<'_>) -> SqlResult<()>,
) -> Result<usize> {
    let embedded: HashSet<i64> = {
        let mut stmt = conn.prepare("SELECT DISTINCT chunk_id FROM vec_chunks")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        rows.collect::<StdResult<_, _>>()?
    };
    let sql = if with_content {
        "SELECT id, content, generation FROM qa_chunks ORDER BY timestamp DESC NULLS LAST"
    } else {
        "SELECT id FROM qa_chunks"
    };
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([])?;
    let mut count = 0;
    while count < budget {
        let Some(row) = rows.next()? else {
            break;
        };
        if !embedded.contains(&row.get::<_, i64>(0)?) {
            visit(row)?;
            count += 1;
        }
    }
    Ok(count)
}

/// Test-only composition of pending selection and embedding, using the same
/// observed implementation as production with a silent reporter.
#[cfg(test)]
pub(crate) fn embed_recent_chunks(
    conn: &mut Connection,
    embedder: &dyn Embed,
    budget: usize,
    on_progress: Option<&dyn Fn(usize, usize)>,
) -> Result<EmbedResult> {
    let missing = pending_chunks(conn, budget)?;
    embed_chunks(
        conn,
        embedder,
        &missing,
        on_progress,
        &EmbedOptions::default(),
    )
}

/// Returns deterministic 768-dim vectors derived from text bytes.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MockEmbedder {
    call_count: AtomicUsize,
    fail_after: Option<usize>,
    fail_on_text: Option<String>,
    /// Last `EmbedOptions` seen by `embed_documents_batch_with_options`, for
    /// tests asserting the CLI flags → `EmbedOptions` forwarding path
    /// (T-005/T-006). `None` until a with-options call happens.
    captured_options: Mutex<Option<EmbedOptions>>,
}

#[cfg(test)]
impl MockEmbedder {
    pub(crate) fn new() -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: None,
            fail_on_text: None,
            captured_options: Mutex::new(None),
        }
    }

    pub(crate) fn failing_after(n: usize) -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: Some(n),
            fail_on_text: None,
            captured_options: Mutex::new(None),
        }
    }

    /// A mock that fails any batch containing `text` (all-or-nothing, matching
    /// `embed_chunks`'s batch semantics). T-001 uses it to poison one batch.
    pub(crate) fn failing_on_text(text: &str) -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: None,
            fail_on_text: Some(text.to_owned()),
            captured_options: Mutex::new(None),
        }
    }

    /// A mock identical to [`Self::new`], named for tests (T-005/T-006) that
    /// assert on [`Self::captured_options`] rather than embed behavior.
    pub(crate) fn capturing_options() -> Self {
        Self::new()
    }

    /// The `EmbedOptions` forwarded by the most recent
    /// `embed_documents_batch_with_options` call, or `None` if that entry point
    /// was never invoked.
    pub(crate) fn captured_options(&self) -> Option<EmbedOptions> {
        *self.captured_options.lock().unwrap()
    }

    /// Constructs the error inline: rurico's `EmbedError::inference` helpers are
    /// `pub(crate)`, and the enum's `#[non_exhaustive]` does not block downstream
    /// construction of existing variants.
    fn inference_error(message: String) -> EmbedError {
        EmbedError::Inference {
            message,
            source: None,
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
                return Err(Self::inference_error("mock failure".to_owned()));
            }
        }
        Ok(Self::deterministic_vector(text))
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        if let Some(limit) = self.fail_after {
            let count = self.call_count.fetch_add(1, Ordering::SeqCst);
            if count >= limit {
                return Err(Self::inference_error("mock failure".to_owned()));
            }
        }
        Ok(ChunkedEmbedding::try_new(vec![
            Self::deterministic_vector(text),
        ])?)
    }

    /// Explicit impl of the production dispatch target: `embed_chunks` calls this
    /// method, so the poison check lives here (all-or-nothing per batch), not in
    /// the per-item delegate. Being explicit also survives rurico revisions where
    /// the trait's default body is removed (required at rurico HEAD).
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        if let Some(poison) = self.fail_on_text.as_deref()
            && texts.contains(&poison)
        {
            return Err(Self::inference_error(format!("poison text: {poison}")));
        }
        texts.iter().map(|t| self.embed_document(t)).collect()
    }

    fn embed_text(&self, text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(Self::deterministic_vector(text))
    }

    /// Records `options` into `captured_options` before delegating
    /// (T-005/T-006), overriding rurico's ignore-and-delegate default so tests
    /// can assert the CLI flags → `EmbedOptions` forwarding path. Does not
    /// sleep: pause application is rurico `Embedder`'s responsibility.
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        *self.captured_options.lock().unwrap() = Some(*options);
        self.embed_documents_batch(texts)
    }
}

#[cfg(test)]
mod tests;
