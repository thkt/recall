use std::sync::Mutex;

use super::*;
use crate::db::{open_db, seed_chunk, seed_session, setup_test_db};

#[test]
fn test_embed_recent_chunks_budget_zero() {
    let (_dir, mut conn) = setup_test_db();
    let embedder = MockEmbedder::new();
    let result = embed_recent_chunks(&mut conn, &embedder, 0, None).unwrap();
    assert_eq!(result.embedded, 0);
    assert_eq!(result.failed_count, 0);
}

#[test]
fn embed_chunks_reports_committed_progress() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    for i in 0..3 {
        seed_chunk(&conn, i + 1, &format!("content {i}"));
    }

    let embedder = MockEmbedder::new();
    let chunks = pending_chunks(&conn, usize::MAX).unwrap();

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
fn replaying_the_same_generation_replaces_vectors_without_duplicates() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    seed_chunk(&conn, 1, "content 0");

    let embedder = MockEmbedder::new();
    let chunks = pending_chunks(&conn, usize::MAX).unwrap();

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

// One poison batch exercises continuation, atomic batch failure, and retry
// together; checking only the retry would miss wrong first-run accounting.
#[test]
fn successful_batches_survive_a_poison_batch_and_failed_chunks_retry() {
    let (_dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 128, |id| format!("content_{id:04}"));
    seed_chunk(&conn, 129, "x");

    // Shortest text sorts into the first batch; the second batch still commits.
    let poisoned = MockEmbedder::failing_on_text("x");
    let first = embed_recent_chunks(&mut conn, &poisoned, 129, None).unwrap();
    assert_eq!(
        first.embedded, 1,
        "precondition: only batch 2 embeds on run 1"
    );

    assert_eq!(first.failed_count, 128);
    assert!(
        first
            .first_error
            .as_deref()
            .unwrap()
            .contains("poison text: x")
    );
    let stored: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(stored, 1, "a failed batch must not partially commit");
    assert_eq!(pending_chunks(&conn, 129).unwrap().len(), 128);

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

/// Interrupt after vectors have been computed but before embed_chunks can save.
/// The production Embed boundary gives deterministic scheduling without a
/// product flag, model download, or timing-dependent inference delay.
struct InterruptedEmbedder<F>(F);

impl<F: Fn() + Send + Sync> Embed for InterruptedEmbedder<F> {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
        MockEmbedder::new().embed_query(text)
    }

    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        MockEmbedder::new().embed_document(text)
    }

    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        let result = MockEmbedder::new().embed_documents_batch(texts)?;
        (self.0)();
        Ok(result)
    }

    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        MockEmbedder::new().embed_text(text, prefix)
    }
}

fn assert_vectors_match_current_content(conn: &Connection) {
    let mut stmt = conn
        .prepare(
            "SELECT v.embedding, c.content, v.sub_idx FROM vec_chunks v \
             LEFT JOIN qa_chunks c ON c.id = v.chunk_id",
        )
        .unwrap();
    let rows = stmt
        .query_map([], |r| {
            Ok((
                r.get::<_, Vec<u8>>(0)?,
                r.get::<_, String>(1)?,
                r.get::<_, i64>(2)?,
            ))
        })
        .unwrap();
    for row in rows {
        let (bytes, content, sub_idx) = row.unwrap(); // NULL content fails for orphans.
        assert_eq!(
            bytes,
            f32_as_bytes(&MockEmbedder::deterministic_vector(&content))
        );
        assert_eq!(sub_idx, 0);
    }
}

#[test]
fn interrupted_inference_saves_only_unchanged_generations_and_preserves_newer_vectors() {
    let (dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 7, |id| {
        if id == 1 {
            // This valid result sorts AFTER the discarded ones, exposing any
            // accidental re-pairing of filtered IDs with unfiltered vectors.
            "unchanged input that sorts last".to_owned()
        } else {
            format!("old {id}")
        }
    });
    let other_saved = Mutex::new(None);
    let embedder = InterruptedEmbedder(|| {
        let mut other = open_db(&dir.path().join("test.db")).unwrap();
        let tx = other.transaction().unwrap();
        tx.execute_batch(
            "UPDATE qa_chunks SET content = 'updated content' WHERE id = 2;
             DELETE FROM qa_chunks WHERE id IN (3, 4, 5, 7);
             INSERT INTO qa_chunks (id, session_id, content) VALUES
                (4, 's1', 'replacement content'), (5, 's1', 'old 5'),
                (7, 's1', 'another worker saved this');
             UPDATE qa_chunks SET content = 'intermediate' WHERE id = 6;
             UPDATE qa_chunks SET content = 'old 6' WHERE id = 6;",
        )
        .unwrap();
        tx.commit().unwrap();
        let current: Vec<_> = pending_chunks(&other, 10)
            .unwrap()
            .into_iter()
            .filter(|chunk| chunk.id == 7)
            .collect();
        let result = embed_chunks(
            &mut other,
            &MockEmbedder::new(),
            &current,
            None,
            &EmbedOptions::default(),
        )
        .unwrap();
        assert_eq!(result.embedded, 1);
        *other_saved.lock().unwrap() = Some(
            other
                .query_row(
                    "SELECT rowid, embedding FROM vec_chunks WHERE chunk_id = 7",
                    [],
                    |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)),
                )
                .unwrap(),
        );
    });
    let progress = Mutex::new(Vec::new());
    let chunks = pending_chunks(&conn, 10).unwrap();
    let result = embed_chunks(
        &mut conn,
        &embedder,
        &chunks,
        Some(&|done, total| progress.lock().unwrap().push((done, total))),
        &EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(result.embedded, 1, "only the unchanged chunk is a success");
    assert_eq!(
        result.failed_count, 0,
        "stale work is not an inference failure"
    );
    assert_eq!(*progress.lock().unwrap(), vec![(1, 7)]);
    let preserved = conn
        .query_row(
            "SELECT rowid, embedding FROM vec_chunks WHERE chunk_id = 7",
            [],
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, Vec<u8>>(1)?)),
        )
        .unwrap();
    assert_eq!(Some(preserved), other_saved.into_inner().unwrap());
    assert_vectors_match_current_content(&conn);
    let mut pending: Vec<_> = pending_chunks(&conn, 10)
        .unwrap()
        .iter()
        .map(|c| c.id)
        .collect();
    pending.sort_unstable();
    assert_eq!(pending, vec![2, 4, 5, 6]);
    let retry = embed_recent_chunks(&mut conn, &MockEmbedder::new(), 10, None).unwrap();
    assert_eq!(retry.embedded, 4);
    assert!(pending_chunks(&conn, 10).unwrap().is_empty());
    assert_vectors_match_current_content(&conn);
}

// The same test executable is launched twice, each invoking the index command's
// real orchestration with a deterministic embedder. Files are barriers, not
// sleeps intended to guess inference duration. Keep all inputs synthetic.
#[test]
fn overlapping_index_processes_keep_vectors_for_the_replacement_text() {
    use std::env;
    use std::fs;
    use std::path::Path;
    use std::process::{Child, Command};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    use crate::index_and_report_with;
    use crate::indexer::IndexOptions;

    fn wait_until(mut ready: impl FnMut() -> bool) {
        let deadline = Instant::now() + Duration::from_secs(30);
        while !ready() {
            assert!(Instant::now() < deadline, "index process barrier timed out");
            thread::sleep(Duration::from_millis(10));
        }
    }

    struct Worker(Child);
    impl Worker {
        fn finish(&mut self) {
            wait_until(|| match self.0.try_wait().unwrap() {
                Some(status) => {
                    assert!(status.success(), "index worker failed: {status}");
                    true
                }
                None => false,
            });
        }
    }
    impl Drop for Worker {
        fn drop(&mut self) {
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }

    fn write_session(root: &Path, text: &str) {
        fs::write(root.join("claude/session.jsonl"), format!(
            "{{\"type\":\"user\",\"cwd\":\"/synthetic\",\"message\":{{\"role\":\"user\",\"content\":\"{text}\"}},\"timestamp\":\"2026-09-20T00:00:00Z\"}}\n"
        )).unwrap();
    }

    if let Some(root) = env::var_os("RECALL_EMBED_RACE_ROOT") {
        let root = Path::new(&root);
        let pause = env::var("RECALL_EMBED_RACE_ROLE").unwrap() == "old";
        let owned_root = root.to_path_buf();
        let embedder = InterruptedEmbedder(move || {
            if pause {
                fs::write(owned_root.join("inference-ready"), b"ready").unwrap();
                wait_until(|| owned_root.join("resume").exists());
            }
        });
        let result = index_and_report_with(
            &Some(root.join("race.db")),
            &IndexOptions {
                force: true,
                claude_dir: &root.join("claude"),
                codex_dir: &root.join("codex"),
            },
            || Ok(Arc::new(embedder) as Arc<dyn Embed>),
        )
        .unwrap();
        assert_eq!(result.embedded, if pause { 0 } else { 1 });
        assert_eq!(result.failed_count, 0);
        assert!(result.degraded_note.is_none());
        return;
    }

    let dir = tempfile::TempDir::new().unwrap();
    let root = dir.path();
    fs::create_dir(root.join("claude")).unwrap();
    fs::create_dir(root.join("codex")).unwrap();
    write_session(root, "old synthetic question");
    let spawn = |role| {
        Worker(Command::new(env::current_exe().unwrap())
        .args(["--exact", "embedder::tests::overlapping_index_processes_keep_vectors_for_the_replacement_text", "--nocapture"])
        .env("RECALL_EMBED_RACE_ROOT", root)
        .env("RECALL_EMBED_RACE_ROLE", role)
        .spawn().unwrap())
    };
    let mut old = spawn("old");
    wait_until(|| {
        assert!(
            old.0.try_wait().unwrap().is_none(),
            "old worker exited before inference"
        );
        root.join("inference-ready").exists()
    });
    let conn = open_db(&root.join("race.db")).unwrap();
    let original: (i64, i64) = conn
        .query_row("SELECT id, generation FROM qa_chunks", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    write_session(root, "new synthetic question");
    let mut new = spawn("new");
    new.finish();
    let replacement: (i64, i64, String) = conn
        .query_row("SELECT id, generation, content FROM qa_chunks", [], |r| {
            Ok((r.get(0)?, r.get(1)?, r.get(2)?))
        })
        .unwrap();
    assert_eq!(replacement.0, original.0, "exercise actual tail-ID reuse");
    assert_ne!(replacement.1, original.1);
    assert!(replacement.2.contains("new synthetic question"));
    let saved: (i64, Vec<u8>) = conn
        .query_row("SELECT rowid, embedding FROM vec_chunks", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    fs::write(root.join("resume"), b"resume").unwrap();
    old.finish();
    let after: (i64, Vec<u8>) = conn
        .query_row("SELECT rowid, embedding FROM vec_chunks", [], |r| {
            Ok((r.get(0)?, r.get(1)?))
        })
        .unwrap();
    assert_eq!(
        after, saved,
        "discarding stale work must not touch current vectors"
    );
    let count: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(count, 1);
    assert!(pending_chunks(&conn, 10).unwrap().is_empty());
    assert_vectors_match_current_content(&conn);
}
