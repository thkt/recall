use std::fs;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::*;
use crate::commands::index::index_and_report_observed;
use crate::db::{open_db, open_db_readonly};
use crate::embedder::{EmbedOptions, MockEmbedder};
use crate::embedding::Embed;
use crate::indexer::IndexOptions;
use crate::model::DegradedReason;
use tempfile::TempDir;

#[test]
fn index_runs_keep_their_stage_order_for_ready_unavailable_legacy_and_failed_backfill_runs() {
    let dir = TempDir::new().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    fs::write(
        source.join("session.jsonl"),
        "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"question\"},\"timestamp\":\"2026-03-01T00:00:00Z\"}\n",
    )
    .unwrap();
    let absent = dir.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: &source,
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let db = dir.path().join("index.db");
    let run = |loader: Result<Arc<dyn Embed>, DegradedReason>| {
        let (observer, lines) = recording();
        let outcome = index_and_report_observed(
            &Some(db.clone()),
            &opts,
            || loader,
            EmbedOptions::default(),
            &observer,
        );
        let counts = observer.snapshot()["counts"].clone();
        let stable: Vec<String> = stable_lines(&lines.borrow())
            .into_iter()
            .map(|line| match line.split_once("embedding counts:") {
                Some((prefix, _)) => format!("{prefix}embedding counts"),
                None => line,
            })
            .collect();
        (outcome, stable, counts)
    };
    let expected = |probe_state: &str, from_chunking: &[&str]| -> Vec<String> {
        let probe_line = format!("model_load_probe: {probe_state}");
        [
            "total: started (not yet complete)",
            "database_open: started (not yet complete)",
            "database_open: complete",
            "model_load_probe: started (not yet complete)",
            probe_line.as_str(),
            "fts_transaction: started (not yet complete)",
            "enumeration: started (not yet complete)",
            "enumeration: complete; unavailable roots reported separately",
            "index_prepare: started (not yet complete)",
            "index_prepare: complete",
            "parse_fts: started (not yet complete)",
            "parse_fts: processed=1/1 unprocessed=0 committed=0",
            "parse_fts: processed; awaiting FTS transaction commit",
            "fts_transaction: committed",
            "fts_finalize: started (not yet complete)",
            "fts_finalize: complete",
            "chunking: started (not yet complete)",
        ]
        .iter()
        .chain(from_chunking)
        .map(|line| format!("index: {line}"))
        .collect()
    };
    let ready = || -> Result<Arc<dyn Embed>, DegradedReason> { Ok(Arc::new(MockEmbedder::new())) };

    let (outcome, first_lines, counts) = run(ready());
    let outcome = outcome.unwrap();
    assert_eq!(
        (
            outcome.embedded,
            outcome.failed_count,
            outcome.degraded_note
        ),
        (1, 0, None)
    );
    assert_eq!(counts["sessions_stored"], 1);
    assert_eq!(
        first_lines,
        expected(
            "ready",
            &[
                "chunking: processed=1/1 unprocessed=0 committed=0",
                "chunking: committed",
                "legacy_backfill: started (not yet complete)",
                "legacy_backfill: committed",
                "pending_extraction: started (not yet complete)",
                "pending_extraction: complete; identity snapshot",
                "embedding counts",
                "pending_extraction: started (not yet complete)",
                "pending_extraction: page loaded",
                "inference: started (not yet complete)",
                "inference: complete; not saved",
                "embedding_db_save: started (not yet complete)",
                "embedding_db_save: committed",
                "embedding counts",
                "total: complete; see counts for deferred or failed work",
            ],
        )
    );

    let (outcome, unavailable_lines, _) = run(Err(DegradedReason::NotInstalled));
    let note = outcome.unwrap().degraded_note.unwrap();
    assert!(note.contains("recall model download"), "{note}");
    assert_eq!(
        unavailable_lines,
        expected(
            "unavailable; embedding deferred",
            &[
                "chunking: no pending sessions",
                "legacy_backfill: started (not yet complete)",
                "legacy_backfill: committed",
                "pending_extraction: started (not yet complete)",
                "pending_extraction: complete",
                "embedding counts",
                "total: complete; see counts for deferred or failed work",
            ],
        )
    );

    let (outcome, probe_failed_lines, _) = run(Ok(Arc::new(MockEmbedder::failing_after(0))));
    let note = outcome.unwrap().degraded_note.unwrap();
    assert!(note.contains("unavailable"), "{note}");
    assert!(!note.contains("download"), "{note}");
    assert_eq!(probe_failed_lines, unavailable_lines);

    let conn = open_db(&db).unwrap();
    conn.execute_batch(
        "UPDATE sessions SET files_scanned = NULL;
         UPDATE qa_chunks SET src_rowid_lo = NULL, src_rowid_hi = NULL;",
    )
    .unwrap();
    let (outcome, legacy_lines, _) = run(ready());
    outcome.unwrap();
    assert_eq!(
        legacy_lines,
        expected(
            "ready",
            &[
                "chunking: no pending sessions",
                "legacy_backfill: started (not yet complete)",
                "legacy_backfill: committed",
                "pending_extraction: started (not yet complete)",
                "pending_extraction: complete; identity snapshot",
                "embedding counts",
                "total: complete; see counts for deferred or failed work",
            ],
        )
    );
    let unlinked: i64 = conn
        .query_row(
            "SELECT (SELECT count(*) FROM sessions WHERE files_scanned IS NULL)
                  + (SELECT count(*) FROM qa_chunks WHERE src_rowid_lo IS NULL)",
            [],
            |row| row.get(0),
        )
        .unwrap();
    assert_eq!(unlinked, 0, "both legacy backfills must complete");

    conn.execute_batch(
        "UPDATE sessions SET files_scanned = NULL;
         CREATE TRIGGER fail_backfill BEFORE UPDATE OF files_scanned ON sessions
         BEGIN SELECT RAISE(ABORT, 'injected backfill failure'); END;",
    )
    .unwrap();
    let (outcome, failed_lines, _) = run(ready());
    let error = outcome.err().expect("the backfill must fail");
    assert!(error.to_string().contains("injected backfill failure"));
    assert_eq!(
        failed_lines,
        expected(
            "ready",
            &[
                "chunking: no pending sessions",
                "legacy_backfill: started (not yet complete)",
                "legacy_backfill: unfinished; only earlier commit notifications confirm saves",
                "total: unfinished; only earlier commit notifications confirm saves",
            ],
        )
    );
}

#[test]
fn slow_loader_is_announced_before_work_and_run_counts_distinguish_changes() {
    let dir = TempDir::new().unwrap();
    let source = dir.path().join("source");
    fs::create_dir(&source).unwrap();
    let file = source.join("session.jsonl");
    let line = "{\"type\":\"user\",\"message\":{\"role\":\"user\",\"content\":\"PRIVATE_BODY_SENTINEL\"}}\n";
    fs::write(&file, line).unwrap();
    let absent = dir.path().join("absent");
    let opts = IndexOptions {
        force: false,
        claude_dir: &source,
        codex_dir: &absent,
        codex_archived_dir: None,
    };
    let db = Some(dir.path().join("index.db"));
    for (run, updated, unchanged) in [(0, 1, 0), (1, 0, 1), (2, 1, 0)] {
        if run == 2 {
            fs::write(&file, line.repeat(2)).unwrap();
        }
        let lines = Rc::new(RefCell::new(Vec::new()));
        let output = Rc::clone(&lines);
        let db_file = db.as_ref().unwrap().clone();
        let observer = Observer::with_reporter(move |line| {
            output.borrow_mut().push(line.to_owned());
            if line.contains(": committed (") {
                let (reader, _) = open_db_readonly(&db_file).unwrap();
                let count = |sql| {
                    reader
                        .query_row(sql, [], |row| row.get::<_, i64>(0))
                        .unwrap()
                };
                if line.contains("fts_transaction:") {
                    assert_eq!(
                        count("SELECT count(*) FROM messages"),
                        if run == 2 { 2 } else { 1 },
                        "FTS must be visible to another reader when saved is announced"
                    );
                } else if line.contains("chunking:") {
                    assert_eq!(
                        count("SELECT count(*) FROM sessions WHERE chunks_indexed = 2"),
                        1
                    );
                    assert!(count("SELECT count(*) FROM qa_chunks") > 0);
                } else if line.contains("embedding_db_save:") {
                    assert!(
                        count("SELECT count(*) FROM vec_chunks") > 0,
                        "vectors must be visible before save notification"
                    );
                }
            }
        });
        let outcome = index_and_report_observed(
            &db,
            &opts,
            || {
                assert!(
                    lines
                        .borrow()
                        .iter()
                        .any(|line| line.contains("model_load_probe: started"))
                );
                assert!(
                    !lines
                        .borrow()
                        .iter()
                        .any(|line| line.contains("enumeration: started"))
                );
                thread::sleep(Duration::from_millis(20));
                Ok(Arc::new(MockEmbedder::new()))
            },
            EmbedOptions::default(),
            &observer,
        )
        .unwrap();
        let observations = &outcome.observations;
        assert!(
            observations["seconds"]["model_load_probe"]
                .as_f64()
                .unwrap()
                >= 0.020
        );
        let counts = &observations["counts"];
        assert_eq!(counts["files_discovered"], 1);
        assert_eq!(counts["files_updated"], updated);
        assert_eq!(counts["files_updated_committed"], updated);
        assert_eq!(counts["files_unchanged"], unchanged);
        assert_eq!(counts["files_remaining"], 0);
        assert_eq!(counts["chunks_remaining_snapshot"], 0);
        assert_eq!(counts["inference_chunks"], updated);
        assert_eq!(counts["inference_batches"], updated);
        assert_eq!(counts["embeddings_reused_committed"], usize::from(run == 2));
        let lines = lines.borrow();
        let position = |text: &str| lines.iter().position(|line| line.contains(text)).unwrap();
        assert!(position("fts_transaction: committed") < position("chunking: started"));
        assert!(position("pending_extraction: started") < position("pending_extraction: complete"));
        if updated > 0 {
            assert!(
                position("embedding counts:") < position("inference: started"),
                "known pending totals must be visible during the first inference"
            );
        } else {
            assert_eq!(counts["chunks_pending_snapshot"], 0);
            assert_eq!(counts["chunks_unattempted"], 0);
            assert_eq!(counts["chunks_saved"], 0);
            assert_eq!(
                lines
                    .iter()
                    .filter(|line| line.contains("embedding counts:"))
                    .count(),
                1,
                "an empty pending set still needs its initial counts"
            );
            assert!(!lines.iter().any(|line| line.contains("inference: started")));
        }
        assert!(
            !lines
                .iter()
                .any(|line| line.contains("PRIVATE_BODY_SENTINEL"))
        );
    }
}
