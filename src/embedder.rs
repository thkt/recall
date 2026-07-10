use std::collections::HashSet;
use std::result::Result as StdResult;
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use amici::storage::anon_placeholders;
use anyhow::Result;
use rurico::embed::Embed;
#[cfg(test)]
use rurico::embed::{ChunkedEmbedding, EMBEDDING_DIMS, EmbedError};
#[cfg(not(test))]
use rurico::embed::{ChunkedEmbedding, EmbedError};
use rusqlite::Connection;
use tracing::warn;

/// Ceiling for `--token-budget` (index/rebuild CLI flag), matching rurico's
/// `TOKEN_BUDGET` forward-pass ceiling (`docs/decisions/0009-...`). Values above
/// this clamp down (never up) at clap parse time so an over-large flag cannot
/// push a forward pass past the Metal OOM ceiling the const was chosen to avoid.
pub(crate) const TOKEN_BUDGET_CEILING: usize = 256_000;

/// Per-call knobs for the index/rebuild embed pass, threaded from the
/// `--token-budget` / `--forward-pause-ms` CLI flags down to
/// `embed_documents_batch_with_options`. `None`/`None` (`Default`) preserves the
/// pre-flag behavior: no forward-pass budget override, no inter-batch pause.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub(crate) struct EmbedOptions {
    /// Forward-pass token budget override (clamped to [`TOKEN_BUDGET_CEILING`]
    /// at clap parse time). Reserved for a future rurico revision to consume;
    /// `rurico::embed::Embed` has no method that accepts a budget override
    /// today, so this field is not read anywhere in the embed path and has no
    /// effect on the embedder's actual forward-pass token budget or on
    /// OOM/swap behavior. `--forward-pause-ms` / `forward_pause` is the only
    /// currently-effective knob for host responsiveness during large runs.
    pub token_budget: Option<usize>,
    /// Pause applied before each `embed_documents_batch` call, trading
    /// throughput for host responsiveness during large embed runs.
    pub forward_pause: Option<Duration>,
}

/// Extends [`Embed`] with an options-aware batch entry point. Default body
/// applies `forward_pause` (real effect, needs no embedder cooperation) and
/// otherwise delegates to [`Embed::embed_documents_batch`], so any existing
/// `Embed` implementor is usable through this trait with unchanged behavior
/// under `EmbedOptions::default()`. Implemented for the trait-object type `dyn
/// Embed` itself (not via a blanket `impl<T: Embed>`, which would conflict with
/// `MockEmbedder`'s capturing override below); production reaches this impl via
/// the [`AsEmbedWithOptions`] adapter, since `&dyn Embed` cannot itself coerce
/// to `&dyn EmbedWithOptions` (see that type's doc comment).
///
/// `options.token_budget` is intentionally not read here: [`Embed`] has no
/// method to carry a forward-pass token budget override into the embedder, so
/// there is nothing for this default body to apply it to (see
/// [`EmbedOptions::token_budget`]).
pub(crate) trait EmbedWithOptions: Embed {
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        if let Some(pause) = options.forward_pause {
            thread::sleep(pause);
        }
        self.embed_documents_batch(texts)
    }
}

impl EmbedWithOptions for dyn Embed + '_ {}

/// Adapts a `&dyn Embed` into `&dyn EmbedWithOptions` for the production embed
/// path, where the loaded embedder is only known as `Arc<dyn Embed>`. Rust's
/// dyn-upcasting coercion covers only supertrait relationships, and `dyn Embed`
/// itself is unsized so it cannot use the standard `T: Sized + Trait -> &dyn
/// Trait` unsizing rule either — so `&dyn Embed` cannot coerce directly to
/// `&dyn EmbedWithOptions` even though `impl EmbedWithOptions for dyn Embed`
/// exists above. This newtype is `Sized` (it just wraps the fat pointer), so it
/// picks up the standard coercion, delegating `Embed` verbatim and taking
/// `EmbedWithOptions`'s default (pause-then-delegate) body.
pub(crate) struct AsEmbedWithOptions<'a>(pub &'a dyn Embed);

impl Embed for AsEmbedWithOptions<'_> {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.0.embed_query(text)
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        self.0.embed_document(text)
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.0.embed_documents_batch(texts)
    }

    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        self.0.embed_text(text, prefix)
    }
}

impl EmbedWithOptions for AsEmbedWithOptions<'_> {}

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

/// Reinterprets an f32 slice as its raw byte view for sqlite-vec storage.
/// Replaces `rurico::storage::f32_as_bytes`, removed in rurico a573655 (#78).
pub(crate) fn f32_as_bytes(v: &[f32]) -> &[u8] {
    bytemuck::cast_slice(v)
}

pub(crate) fn embed_chunks(
    conn: &mut Connection,
    embedder: &dyn EmbedWithOptions,
    chunks: &[(i64, String)],
    on_progress: Option<&dyn Fn(usize, usize)>,
    options: &EmbedOptions,
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
        match embedder.embed_documents_batch_with_options(&texts, options) {
            Ok(embeddings) => {
                let tx = conn.transaction()?;
                // Idempotent replay guard for stale concurrent work that reached
                // this tx after the pending query. Batched into one
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
                    for (sub_idx, sub_emb) in chunked.chunks().iter().enumerate() {
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
                // the next index to retry via the pending gate.
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

/// Collect up to `budget` chunks that have no `vec_chunks` row, newest first.
///
/// One pass over each table (#138): the previous correlated `NOT EXISTS` form
/// re-scanned vec_chunks per qa_chunks row — its `+chunk_id` is an unindexed
/// auxiliary column (see the DELETE note in `embed_chunks`), so at 38k chunks
/// that was O(N×M) ≈ 15-20 minutes of silence before the first batch.
pub(crate) fn pending_chunks(conn: &Connection, budget: usize) -> Result<Vec<(i64, String)>> {
    if budget == 0 {
        return Ok(Vec::new());
    }

    let embedded: HashSet<i64> = {
        let mut stmt = conn.prepare("SELECT DISTINCT chunk_id FROM vec_chunks")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        rows.collect::<StdResult<_, _>>()?
    };

    let mut stmt = conn.prepare(
        "SELECT c.id, c.content FROM qa_chunks c \
         ORDER BY c.timestamp DESC NULLS LAST",
    )?;
    let rows = stmt.query_map([], |row| {
        let id: i64 = row.get(0)?;
        if embedded.contains(&id) {
            // Skip without decoding content; sqlite materializes only the
            // columns a row actually reads.
            return Ok(None);
        }
        Ok(Some((id, row.get::<_, String>(1)?)))
    })?;
    let missing: Vec<(i64, String)> = rows
        .filter_map(StdResult::transpose)
        .take(budget)
        .collect::<StdResult<_, _>>()?;
    Ok(missing)
}

/// Test-only composition of the production pair (`pending_chunks` →
/// `embed_chunks`); `embed_all_pending` (main.rs) calls the pair directly so
/// the empty-pending case can skip the Spinner.
#[cfg(test)]
pub(crate) fn embed_recent_chunks(
    conn: &mut Connection,
    embedder: &dyn EmbedWithOptions,
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
}

/// Records `options` into `captured_options` before delegating (T-005/T-006),
/// distinct from the trait-object `dyn Embed` default so a `MockEmbedder`
/// passed directly (not behind `&dyn Embed`) observes what was forwarded.
#[cfg(test)]
impl EmbedWithOptions for MockEmbedder {
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        *self.captured_options.lock().unwrap() = Some(*options);
        if let Some(pause) = options.forward_pause {
            thread::sleep(pause);
        }
        self.embed_documents_batch(texts)
    }
}

#[cfg(test)]
mod tests;
