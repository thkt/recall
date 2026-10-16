use std::collections::HashSet;
use std::result::Result as StdResult;
#[cfg(test)]
use std::sync::Mutex;
#[cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::compat::anon_placeholders;
use crate::embedding::Embed;
#[cfg(test)]
use crate::embedding::{ChunkedEmbedding, EMBEDDING_DIMS, EmbedError};
use anyhow::Result;
use rusqlite::{Connection, OptionalExtension, Result as SqlResult, Row, TransactionBehavior};
use tracing::warn;

use crate::index_observer::Observer;

pub(crate) use crate::embedding::EmbedOptions;

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
            let first_error = self.first_error.as_deref().unwrap_or("unknown");
            warn!(
                failed_count = self.failed_count,
                first_error, "embedding skipped chunks after batch failures"
            );
        }
    }
}

pub(crate) const EMBED_BATCH_SIZE: usize = 128;
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

pub(crate) struct PendingChunk {
    id: i64,
    content: String,
    generation: i64,
}

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
                // Inference ran without a write lock; validate generations before saving.
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

fn pending_snapshot(conn: &Connection) -> Result<Vec<PendingIdentity>> {
    let tx = conn.unchecked_transaction()?;
    let mut pending = Vec::new();
    visit_pending(&tx, true, |row| {
        pending.push(PendingIdentity {
            id: row.get(0)?,
            generation: row.get(1)?,
            bytes: row.get::<_, u32>(2)? as usize,
        });
        Ok(())
    })?;
    tx.commit()?;
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
    }
    progress.finish()
}

#[cfg(test)]
pub(crate) fn pending_chunks(conn: &Connection, budget: usize) -> Result<Vec<PendingChunk>> {
    let pending = pending_snapshot(conn)?;
    load_page(conn, &pending[..pending.len().min(budget)])
}

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

#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MockEmbedder {
    call_count: AtomicUsize,
    fail_after: Option<usize>,
    fail_on_text: Option<String>,
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

    pub(crate) fn failing_on_text(text: &str) -> Self {
        Self {
            call_count: AtomicUsize::new(0),
            fail_after: None,
            fail_on_text: Some(text.to_owned()),
            captured_options: Mutex::new(None),
        }
    }

    pub(crate) fn capturing_options() -> Self {
        Self::new()
    }

    pub(crate) fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }

    pub(crate) fn captured_options(&self) -> Option<EmbedOptions> {
        *self.captured_options.lock().unwrap()
    }

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
        ChunkedEmbedding::try_new(vec![Self::deterministic_vector(text)])
    }

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
