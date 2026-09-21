use std::sync::Mutex;

use super::*;
use crate::db::{open_db, seed_chunk, seed_session, setup_test_db};
use crate::index_observer::recording;

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
// fails), the production pending pipeline returns Ok (the index does not abort), counts the
// whole batch as failed, and leaves no vec row so the next index retries it.
// (Updated from the old T-009: stopped_at_error.is_some() → failed_count > 0.)
// Perspective: error (the all-fail path) + boundary (failed_count == every chunk).
#[test]
fn all_batches_failing_is_non_fatal_and_retryable() {
    let (_dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 1, |_| "pending content".to_owned());

    let embedder = MockEmbedder::failing_after(0);
    let result = embed_pending_observed(
        &mut conn,
        &embedder,
        None,
        &EmbedOptions::default(),
        &Observer::default(),
    )
    .unwrap();

    assert_eq!(result.embedded, 0, "every batch failed, so nothing embeds");
    assert_eq!(
        result.failed_count, 1,
        "the sole batch's chunk count is reported as failed"
    );
    let err = result.first_error.as_deref().unwrap_or_default();
    assert!(
        err == "batch inference failed",
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
    seed_chunks(&conn, 1024, |id| format!("content_{id:04}"));
    seed_chunk(&conn, 1025, "x");

    // Shortest text sorts into the first batch; the second batch still commits.
    let poisoned = MockEmbedder::failing_on_text("x");
    let (observer, lines) = recording();
    let first = embed_pending_observed(
        &mut conn,
        &poisoned,
        None,
        &EmbedOptions::default(),
        &observer,
    )
    .unwrap();
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_saved"], 897);
    assert_eq!(counts["chunks_failed"], 128);
    assert_eq!(counts["chunks_remaining_snapshot"], 128);
    assert_eq!(counts["chunks_unattempted"], 0);
    assert_eq!(counts["chunks_stale"], 0);
    assert!(
        !lines
            .borrow()
            .iter()
            .any(|line| line.contains("poison text"))
    );
    assert_eq!(
        lines
            .borrow()
            .iter()
            .filter(|line| line.contains("embedding_db_save: committed"))
            .count(),
        8
    );
    assert_eq!(
        first.embedded, 897,
        "later batches and the next page still commit"
    );

    assert_eq!(first.failed_count, 128);
    assert!(
        first
            .first_error
            .as_deref()
            .unwrap()
            .eq("batch inference failed")
    );
    let stored: i64 = conn
        .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
        .unwrap();
    assert_eq!(stored, 897, "a failed batch must not partially commit");
    assert_eq!(pending_chunks(&conn, usize::MAX).unwrap().len(), 128);

    assert_eq!(counts["pending_pages"], 2);
    assert_eq!(counts["inference_chunks"], 1025);
    assert_eq!(counts["inference_batches"], 9);
    let healthy = MockEmbedder::new();
    let second = embed_pending_observed(
        &mut conn,
        &healthy,
        None,
        &EmbedOptions::default(),
        &Observer::default(),
    )
    .unwrap();
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
    let observer = Observer::default();
    let result = embed_chunks_observed(
        &mut conn,
        &embedder,
        &chunks,
        Some(&|done, total| progress.lock().unwrap().push((done, total))),
        &EmbedOptions::default(),
        &observer,
    )
    .unwrap();
    assert_eq!(result.embedded, 1, "only the unchanged chunk is a success");
    assert_eq!(
        result.failed_count, 0,
        "stale work is not an inference failure"
    );
    assert_eq!(*progress.lock().unwrap(), vec![(1, 7)]);
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_pending_snapshot"], 7);
    assert_eq!(counts["chunks_saved"], 1);
    assert_eq!(counts["chunks_stale"], 6);
    assert_eq!(counts["chunks_failed"], 0);
    assert_eq!(counts["chunks_save_failed"], 0);
    assert_eq!(counts["chunks_unattempted"], 0);
    assert_eq!(counts["chunks_remaining_snapshot"], 6);
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
    assert_eq!(pending_count(&conn).unwrap(), 4);
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

#[test]
fn later_page_database_save_failure_preserves_other_batches_and_retries_only_failed_chunks() {
    // Fail the second INSERT of page 2, after all eight page-1 batches committed and one
    // vector in that failing batch was staged. This exercises rollback, not only BEGIN.
    struct InvalidLaterBatch(AtomicUsize);
    impl Embed for InvalidLaterBatch {
        fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError> {
            MockEmbedder::new().embed_query(text)
        }

        fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError> {
            MockEmbedder::new().embed_document(text)
        }

        fn embed_documents_batch(
            &self,
            texts: &[&str],
        ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
            let mut vectors = MockEmbedder::new().embed_documents_batch(texts)?;
            if self.0.fetch_add(1, Ordering::SeqCst) == 8 {
                vectors[1] = ChunkedEmbedding::try_new(vec![vec![0.0; 1]])?;
            }
            Ok(vectors)
        }

        fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
            MockEmbedder::new().embed_text(text, prefix)
        }
    }

    let (dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 1153, |id| format!("private input {id:04}"));
    let reader = open_db(&dir.path().join("test.db")).unwrap();
    let saved_ids = || {
        reader
            .prepare("SELECT chunk_id FROM vec_chunks ORDER BY chunk_id")
            .unwrap()
            .query_map([], |row| row.get::<_, i64>(0))
            .unwrap()
            .collect::<SqlResult<Vec<_>>>()
            .unwrap()
    };
    let embedder = InvalidLaterBatch(AtomicUsize::new(0));
    let (observer, lines) = recording();
    let commits = Mutex::new(Vec::new());
    let error = embed_pending_observed(
        &mut conn,
        &embedder,
        Some(&|done, total| {
            let mut expected: Vec<i64> = (1..=i64::try_from(done.min(1024)).unwrap()).collect();
            if done == 1025 {
                expected.push(1153);
            }
            assert_eq!(saved_ids(), expected);
            commits.lock().unwrap().push((done, total));
        }),
        &EmbedOptions::default(),
        &observer,
    )
    .err()
    .expect("the second page must fail to save");
    assert!(
        error.to_string().contains("Dimension mismatch"),
        "must reach the failing vector INSERT: {error}"
    );
    assert_eq!(embedder.0.load(Ordering::SeqCst), 10);
    assert_eq!(
        *commits.lock().unwrap(),
        (1..=8)
            .map(|batch| (batch * 128, 1153))
            .chain([(1025, 1153)])
            .collect::<Vec<_>>()
    );
    assert_eq!(saved_ids(), (1..=1024).chain([1153]).collect::<Vec<_>>());
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_pending_snapshot"], 1153);
    assert_eq!(counts["chunks_saved"], 1025);
    assert_eq!(counts["chunks_save_failed"], 128);
    assert_eq!(counts["chunks_failed"], 0);
    assert_eq!(counts["chunks_stale"], 0);
    assert_eq!(counts["chunks_unattempted"], 0);
    assert_eq!(counts["chunks_remaining_snapshot"], 128);
    assert_eq!(
        lines
            .borrow()
            .iter()
            .filter(|line| line.contains("embedding_db_save: committed"))
            .count(),
        9
    );
    assert!(
        lines
            .borrow()
            .iter()
            .any(|line| line.contains("embedding_db_save: failed; batch rolled back"))
    );
    let first_counts: serde_json::Value = serde_json::from_str(
        lines
            .borrow()
            .iter()
            .find_map(|line| line.strip_prefix("index: embedding counts: "))
            .unwrap(),
    )
    .unwrap();
    assert_eq!(first_counts["chunks_unattempted"], 1153);
    assert_eq!(first_counts["chunks_saved"], 0);

    let remaining = pending_chunks(&reader, usize::MAX).unwrap();
    let mut remaining_ids: Vec<_> = remaining.iter().map(|chunk| chunk.id).collect();
    remaining_ids.sort_unstable();
    assert_eq!(remaining_ids, (1025..=1152).collect::<Vec<_>>());
    // Reuse the real pending gate on restart; saved chunks must not be retried.
    let (resumed, _) = recording();
    let result = embed_pending_observed(
        &mut conn,
        &MockEmbedder::new(),
        None,
        &EmbedOptions::default(),
        &resumed,
    )
    .unwrap();
    assert_eq!(result.embedded, 128);
    assert_eq!(resumed.snapshot()["counts"]["chunks_pending_snapshot"], 128);
    assert_eq!(resumed.snapshot()["counts"]["chunks_remaining_snapshot"], 0);
    assert_eq!(saved_ids(), (1..=1153).collect::<Vec<_>>());
    assert_eq!(pending_count(&reader).unwrap(), 0);
    assert_vectors_match_current_content(&reader);
}

#[test]
fn failed_database_save_is_counted_without_claiming_a_commit() {
    let (_dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 1, |_| "private input".to_owned());
    let pending = pending_chunks(&conn, 1).unwrap();
    // Keep BEGIN failure coverage as well as the later INSERT/rollback case.
    conn.execute_batch("PRAGMA query_only = ON;").unwrap();
    let (observer, lines) = recording();
    assert!(
        embed_chunks_observed(
            &mut conn,
            &MockEmbedder::new(),
            &pending,
            None,
            &EmbedOptions::default(),
            &observer
        )
        .is_err()
    );
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_saved"], 0);
    assert_eq!(counts["chunks_save_failed"], 1);
    assert_eq!(counts["chunks_failed"], 0);
    assert_eq!(counts["chunks_unattempted"], 0);
    assert_eq!(counts["chunks_remaining_snapshot"], 1);
    assert!(
        !lines
            .borrow()
            .iter()
            .any(|line| line.contains("embedding_db_save: committed"))
    );
    assert!(
        lines
            .borrow()
            .iter()
            .any(|line| line.contains("embedding_db_save: failed; batch rolled back"))
    );
    conn.execute_batch("PRAGMA query_only = OFF;").unwrap();
    assert_eq!(
        embed_recent_chunks(&mut conn, &MockEmbedder::new(), 1, None)
            .unwrap()
            .embedded,
        1
    );
    assert_eq!(pending_count(&conn).unwrap(), 0);
}

#[test]
fn byte_bounded_pages_preserve_unicode_nuls_and_an_oversized_chunk() {
    let (_dir, mut conn) = setup_test_db();
    seed_session(&conn, "s1");
    // Descending lengths force the real length bucketing to change ID order.
    seed_chunk(&conn, 1, &"c".repeat(PENDING_PAGE_BYTES + 1));
    seed_chunk(&conn, 2, &"b".repeat(PENDING_PAGE_BYTES / 2));
    seed_chunk(&conn, 3, &"a".repeat(PENDING_PAGE_BYTES / 2 - 3));
    // Four UTF-8 bytes: counting characters (or stopping at NUL) would wrongly
    // admit chunk 2 into the first page, whose bodies would exceed 8 MiB by one.
    seed_chunk(&conn, 4, "é\0x");
    let observer = Observer::default();
    let pages = Mutex::new(Vec::new());
    let result = embed_pending_observed(
        &mut conn,
        &MockEmbedder::new(),
        Some(&|done, total| {
            let counts = observer.snapshot()["counts"].clone();
            pages.lock().unwrap().push((
                done,
                total,
                counts["pending_body_bytes_peak"].as_u64().unwrap(),
            ));
        }),
        &EmbedOptions::default(),
        &observer,
    )
    .unwrap();
    assert_eq!(result.embedded, 4);
    assert_eq!(
        *pages.lock().unwrap(),
        vec![(2, 4, 4_194_305), (3, 4, 4_194_305), (4, 4, 8_388_609)]
    );
    assert_eq!(observer.snapshot()["counts"]["pending_pages"], 3);
    assert_eq!(observer.snapshot()["counts"]["inference_batches"], 3);
    assert_eq!(pending_count(&conn).unwrap(), 0);
    assert_vectors_match_current_content(&conn);
}

#[test]
fn later_pages_reject_replaced_and_deleted_generations_without_loading_new_bodies() {
    let (dir, mut conn) = setup_test_db();
    seed_chunks(&conn, 1027, |id| format!("input {id:04}"));
    let mut other = open_db(&dir.path().join("test.db")).unwrap();
    let observer = Observer::default();
    let other = Mutex::new(&mut other);
    let result = embed_pending_observed(
        &mut conn,
        &MockEmbedder::new(),
        Some(&|done, total| {
            assert_eq!(total, 1027);
            if done != 128 {
                return;
            }
            let mut writer = other.lock().unwrap();
            writer
                .execute("DELETE FROM qa_chunks WHERE id IN (1025, 1027)", [])
                .unwrap();
            writer
                .execute(
                    "UPDATE qa_chunks SET content = ?1 WHERE id = 1026",
                    [&"x".repeat(PENDING_PAGE_BYTES + 1)],
                )
                .unwrap();
            seed_chunk(&writer, 1027, "input 1027");
            seed_chunk(&writer, 1028, "new input");
            let current: Vec<_> = pending_chunks(&writer, usize::MAX)
                .unwrap()
                .into_iter()
                .filter(|chunk| chunk.id == 1027)
                .collect();
            embed_chunks(
                &mut writer,
                &MockEmbedder::new(),
                &current,
                None,
                &EmbedOptions::default(),
            )
            .unwrap();
        }),
        &EmbedOptions::default(),
        &observer,
    )
    .unwrap();
    assert_eq!(result.embedded, 1024);
    let counts = observer.snapshot()["counts"].clone();
    assert_eq!(counts["chunks_pending_snapshot"], 1027);
    assert_eq!(counts["chunks_stale"], 3);
    assert_eq!(counts["chunks_unattempted"], 0);
    assert_eq!(counts["chunks_remaining_snapshot"], 3);
    assert_eq!(counts["inference_chunks"], 1024);
    assert_eq!(counts["pending_body_bytes_peak"], 10_240);
    assert_eq!(pending_count(&conn).unwrap(), 2);
    let resumed = embed_pending_observed(
        &mut conn,
        &MockEmbedder::new(),
        None,
        &EmbedOptions::default(),
        &Observer::default(),
    )
    .unwrap();
    assert_eq!(resumed.embedded, 2);
    assert_eq!(pending_count(&conn).unwrap(), 0);
    let (rows, distinct): (i64, i64) = conn
        .query_row(
            "SELECT COUNT(*), COUNT(DISTINCT chunk_id) FROM vec_chunks",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )
        .unwrap();
    assert_eq!((rows, distinct), (1027, 1027));
    assert_vectors_match_current_content(&conn);
}
