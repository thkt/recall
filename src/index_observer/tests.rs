use std::fs;
use std::rc::Rc;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use super::*;
use crate::db::open_db_readonly;
use crate::embedder::{EmbedOptions, MockEmbedder};
use crate::index_and_report_observed;
use crate::indexer::IndexOptions;
use tempfile::TempDir;

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
                        count("SELECT count(*) FROM sessions WHERE chunks_indexed = 1"),
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
