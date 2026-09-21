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
use rusqlite::{Connection, OptionalExtension, Result as SqlResult, Row, TransactionBehavior};
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
            // embed_page's inference error arm together); the fallback is defensive only.
            let first_error = self.first_error.as_deref().unwrap_or("unknown");
            warn!(
                failed_count = self.failed_count,
                first_error, "embedding skipped chunks after batch failures"
            );
        }
    }
}

pub(crate) const EMBED_BATCH_SIZE: usize = 128;
// Host measurements compare these bounds with the former all-body selection.
const PENDING_PAGE_CHUNKS: usize = 1024;
const PENDING_PAGE_BYTES: usize = 8 * 1024 * 1024;

struct PendingIdentity {
    id: i64,
    generation: i64,
    bytes: usize,
}

#[derive(Default)]
struct EmbeddingProgress {
    total: usize,
    attempted: usize,
    stale: usize,
    batches: usize,
    inferred: usize,
    save_failed: usize,
    first_save_error: Option<anyhow::Error>,
    result: EmbedResult,
}

impl EmbeddingProgress {
    fn finish(self) -> Result<EmbedResult> {
        match self.first_save_error {
            Some(error) => Err(error),
            None => Ok(self.result),
        }
    }
}

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

#[cfg(test)]
pub(crate) fn embed_chunks_observed(
    conn: &mut Connection,
    embedder: &dyn Embed,
    chunks: &[PendingChunk],
    on_progress: Option<&dyn Fn(usize, usize)>,
    options: &EmbedOptions,
    observer: &Observer,
) -> Result<EmbedResult> {
    observer.start_embedding(chunks.len());
    let mut progress = EmbeddingProgress {
        total: chunks.len(),
        ..Default::default()
    };
    embed_page(
        conn,
        embedder,
        chunks,
        on_progress,
        options,
        observer,
        &mut progress,
    );
    progress.finish()
}

fn embed_page(
    conn: &mut Connection,
    embedder: &dyn Embed,
    chunks: &[PendingChunk],
    on_progress: Option<&dyn Fn(usize, usize)>,
    options: &EmbedOptions,
    observer: &Observer,
    progress: &mut EmbeddingProgress,
) {
    // load_page preserves the snapshot's (byte length, ID) order, including
    // when changed or deleted generations are omitted.
    for batch in chunks.chunks(EMBED_BATCH_SIZE) {
        let texts: Vec<&str> = batch.iter().map(|chunk| chunk.content.as_str()).collect();
        progress.batches += 1;
        progress.inferred += batch.len();
        observer.count("inference_batches", progress.batches);
        observer.count("inference_chunks", progress.inferred);
        let inference = observer.stage("inference");
        let result = embedder.embed_documents_batch_with_options(&texts, options);
        inference.finish(if result.is_ok() {
            "complete; not saved"
        } else {
            "failed; not saved"
        });
        progress.attempted += batch.len();
        observer.count("chunks_unattempted", progress.total - progress.attempted);
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
                        for (chunked, chunk) in embeddings.iter().zip(batch) {
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
                        progress.save_failed += batch.len();
                        observer.count("chunks_save_failed", progress.save_failed);
                        if progress.first_save_error.is_none() {
                            progress.first_save_error = Some(error);
                        }
                        save.finish("failed; batch rolled back");
                        observer.embedding_progress();
                        continue;
                    }
                };
                progress.stale += batch.len() - saved;
                progress.result.embedded += saved;
                observer.count("chunks_stale", progress.stale);
                observer.count("chunks_saved", progress.result.embedded);
                observer.count(
                    "chunks_remaining_snapshot",
                    progress.total - progress.result.embedded,
                );
                save.finish(if saved == 0 {
                    "no current results to save"
                } else {
                    "committed"
                });
                if saved > 0
                    && let Some(cb) = &on_progress
                {
                    cb(progress.result.embedded, progress.total);
                }
            }
            Err(_) => {
                // Skip the failed batch and keep going: a poison chunk fails its
                // whole ≤128-batch (all-or-nothing) but must not block the rest of
                // the backlog. The chunks stay pending (no tx committed here) for
                // the next index to retry via the pending gate.
                progress.result.failed_count += batch.len();
                observer.count("chunks_failed", progress.result.failed_count);
                if progress.result.first_error.is_none() {
                    progress.result.first_error = Some("batch inference failed".to_owned());
                }
            }
        }
        observer.embedding_progress();
    }
}

/// Capture only identities and byte lengths in one read snapshot. No body sort
/// in SQLite, no per-page anti-join against vec0's unindexed auxiliary column.
fn pending_snapshot(conn: &Connection) -> Result<Vec<PendingIdentity>> {
    let tx = conn.unchecked_transaction()?;
    let mut pending = Vec::new();
    visit_pending(&tx, true, |row| {
        pending.push(PendingIdentity {
            id: row.get(0)?,
            generation: row.get(1)?,
            // SQLite limits a single value to at most 2^31-1 bytes.
            bytes: row.get::<_, u32>(2)? as usize,
        });
        Ok(())
    })?;
    tx.commit()?;
    // Preserve global length bucketing with O(pending IDs) metadata, avoiding
    // padding inflation from timestamp-local pages. No sort scratch allocation.
    pending.sort_unstable_by_key(|chunk| (chunk.bytes, chunk.id));
    Ok(pending)
}

fn load_page(conn: &Connection, identities: &[PendingIdentity]) -> Result<Vec<PendingChunk>> {
    let mut query =
        conn.prepare_cached("SELECT content FROM qa_chunks WHERE id = ?1 AND generation = ?2")?;
    let mut chunks = Vec::with_capacity(identities.len());
    for identity in identities {
        if let Some(content) = query
            .query_row(rusqlite::params![identity.id, identity.generation], |row| {
                row.get(0)
            })
            .optional()?
        {
            chunks.push(PendingChunk {
                id: identity.id,
                generation: identity.generation,
                content,
            });
        }
    }
    Ok(chunks)
}

/// A finite, immutable worklist ensures failed pages are tried only once per
/// invocation. New or replaced generations are selected on the next invocation.
pub(crate) fn embed_pending_observed(
    conn: &mut Connection,
    embedder: &dyn Embed,
    on_progress: Option<&dyn Fn(usize, usize)>,
    options: &EmbedOptions,
    observer: &Observer,
) -> Result<EmbedResult> {
    let extraction = observer.stage("pending_extraction");
    let pending = pending_snapshot(conn)?;
    extraction.finish("complete; identity snapshot");
    observer.start_embedding(pending.len());
    observer.count(
        "pending_metadata_bytes",
        pending.capacity() * size_of::<PendingIdentity>(),
    );
    observer.count("pending_body_bytes_peak", 0);
    observer.count("pending_page_chunks_peak", 0);
    observer.count("pending_pages", 0);
    let mut progress = EmbeddingProgress {
        total: pending.len(),
        ..Default::default()
    };
    let mut offset = 0;
    let mut pages = 0;
    let mut peak_bytes = 0;
    let mut peak_chunks = 0;
    while offset < pending.len() {
        let extraction = observer.stage("pending_extraction");
        let mut end = offset;
        let mut bytes = 0;
        while end < pending.len() && end - offset < PENDING_PAGE_CHUNKS {
            let next = pending[end].bytes;
            if end > offset && next > PENDING_PAGE_BYTES.saturating_sub(bytes) {
                break;
            }
            bytes += next;
            end += 1;
        }
        // A single oversized chunk travels alone, unchanged; the model's own
        // tokenizer/subchunking still applies. Never truncate or loop on it.
        let chunks = load_page(conn, &pending[offset..end])?;
        let disappeared = end - offset - chunks.len();
        progress.stale += disappeared;
        progress.attempted += disappeared;
        observer.count("chunks_stale", progress.stale);
        observer.count("chunks_unattempted", progress.total - progress.attempted);
        peak_bytes = peak_bytes.max(chunks.iter().map(|chunk| chunk.content.len()).sum());
        peak_chunks = peak_chunks.max(chunks.len());
        pages += 1;
        observer.count("pending_body_bytes_peak", peak_bytes);
        observer.count("pending_page_chunks_peak", peak_chunks);
        observer.count("pending_pages", pages);
        extraction.finish("page loaded");
        embed_page(
            conn,
            embedder,
            &chunks,
            on_progress,
            options,
            observer,
            &mut progress,
        );
        offset = end;
        if chunks.is_empty() {
            observer.embedding_progress();
        }
        // Drop this page before loading the next; no body is retained in pending.
    }
    progress.finish()
}

/// Test adapter for scheduling mutations between selection and inference.
#[cfg(test)]
pub(crate) fn pending_chunks(conn: &Connection, budget: usize) -> Result<Vec<PendingChunk>> {
    let pending = pending_snapshot(conn)?;
    load_page(conn, &pending[..pending.len().min(budget)])
}

/// Same single-pass selection without loading bodies when inference is unavailable.
pub(crate) fn pending_count(conn: &Connection) -> Result<usize> {
    visit_pending(conn, false, |_| Ok(()))
}

fn visit_pending(
    conn: &Connection,
    with_lengths: bool,
    mut visit: impl FnMut(&Row<'_>) -> SqlResult<()>,
) -> Result<usize> {
    let embedded: HashSet<i64> = {
        let mut stmt = conn.prepare("SELECT chunk_id FROM vec_chunks")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        rows.collect::<StdResult<_, _>>()?
    };
    let sql = if with_lengths {
        "SELECT id, generation, length(CAST(content AS BLOB)) FROM qa_chunks"
    } else {
        "SELECT id FROM qa_chunks"
    };
    let mut stmt = conn.prepare(sql)?;
    let mut rows = stmt.query([])?;
    let mut count = 0;
    while let Some(row) = rows.next()? {
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
    /// the pipeline's batch semantics). T-001 uses it to poison one batch.
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

    /// Query/document inference attempts, including failures.
    pub(crate) fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    /// The options from the most recent batch call, or None before any call.
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
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_after.is_some_and(|limit| count >= limit) {
            return Err(Self::inference_error("mock failure".to_owned()));
        }
        Ok(Self::deterministic_vector(text))
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        let count = self.call_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_after.is_some_and(|limit| count >= limit) {
            return Err(Self::inference_error("mock failure".to_owned()));
        }
        Ok(ChunkedEmbedding::try_new(vec![
            Self::deterministic_vector(text),
        ])?)
    }

    /// Explicit impl of the production dispatch target: `embed_page` calls this
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
