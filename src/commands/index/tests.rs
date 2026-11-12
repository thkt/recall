use std::fs;
use std::path::Path;

use crate::envelope::render_json_success;
use crate::model::tests::FakeHost;
use crate::tests::{mock_loader, payload_keys, vec_chunk_count};
use crate::{parser, search};

use super::*;

fn absent_source_dirs(root: &Path) -> (PathBuf, PathBuf) {
    (root.join("no_claude"), root.join("no_codex"))
}

fn seed_claude_source(claude_dir: &Path) {
    fs::create_dir_all(claude_dir).unwrap();
    fs::write(
            claude_dir.join("s1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"quokka rebuild probe"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
}

/// The `--json` envelope `recall index` prints for an outcome.
fn printed_envelope(outcome: &IndexOutcome) -> serde_json::Value {
    serde_json::from_str(&render_json_success(&index_command_output(outcome))).unwrap()
}

/// The `--json` envelope `recall index` prints for a run, without `data.observations.seconds`,
/// whose stage timings change from run to run.
fn printed_json(outcome: &IndexOutcome) -> serde_json::Value {
    let mut envelope = printed_envelope(outcome);
    envelope["data"]["observations"]
        .as_object_mut()
        .expect("a run prints data.observations as an object")
        .remove("seconds")
        .expect("a run prints data.observations.seconds");
    envelope
}

#[test]
fn index_hands_the_token_budget_and_forward_pause_flags_to_the_embedder() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let claude_dir = db_dir.path().join("claude");
    let codex_dir = db_dir.path().join("codex");
    seed_claude_source(&claude_dir);
    fs::create_dir(&codex_dir).unwrap();
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    let embedder = Arc::new(embedder::MockEmbedder::capturing_options());

    let outcome = index_and_report_observed(
        &Some(db_path),
        &opts,
        || Ok(embedder.clone()),
        embed_options_from_flags(Some(2048), Some(700)),
        &index_observer::Observer::stderr(),
    )
    .unwrap();

    assert_eq!(
        embedder.captured_options(),
        Some(embedder::EmbedOptions {
            token_budget: Some(2048),
            forward_pause: Some(Duration::from_millis(700)),
        }),
        "--token-budget 2048 --forward-pause-ms 700 reach the embedder as 2048 tokens and a 700 ms pause"
    );
    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 1,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 1,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 1,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 1,
                        "inference_chunks": 1,
                        "pending_body_bytes_peak": 20,
                        "pending_metadata_bytes": 96,
                        "pending_page_chunks_peak": 1,
                        "pending_pages": 1,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn index_without_the_token_budget_and_forward_pause_flags_hands_the_embedder_the_default_options() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let claude_dir = db_dir.path().join("claude");
    let codex_dir = db_dir.path().join("codex");
    seed_claude_source(&claude_dir);
    fs::create_dir(&codex_dir).unwrap();
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let embedder = Arc::new(embedder::MockEmbedder::capturing_options());
    let no_flags = embed_options_from_flags(None, None);

    index_and_report_observed(
        &Some(db_path),
        &opts,
        || Ok(embedder.clone()),
        no_flags,
        &index_observer::Observer::stderr(),
    )
    .unwrap();

    assert_eq!(
        no_flags,
        embedder::EmbedOptions::default(),
        "no --token-budget and no --forward-pause-ms mean no token budget and no pause"
    );
    assert_eq!(
        embedder.captured_options(),
        Some(embedder::EmbedOptions::default()),
        "a run without the flags hands the embedder the default options"
    );
}

#[test]
fn rebuild_with_the_model_embeds_the_session_again_and_keeps_it_searchable() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            1,
            "precondition: the first index embeds the session's one chunk"
        );
    }

    let rebuild_opts = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let rebuilt =
        index_and_report_with(&Some(db_path.clone()), &rebuild_opts, mock_loader).unwrap();

    assert_eq!(
        printed_json(&rebuilt),
        serde_json::json!({
            "data": {
                "embedded": 1,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 1,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 1,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 1,
                        "inference_chunks": 1,
                        "pending_body_bytes_peak": 20,
                        "pending_metadata_bytes": 96,
                        "pending_page_chunks_peak": 1,
                        "pending_pages": 1,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": false,
            "notes": [],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(vec_chunk_count(&conn), 1, "the session keeps one vector");
    let hits = search::search(&conn, "quokka", &search::SearchOptions::default()).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "rebuild must reconstruct FTS: the seeded term must return its session"
    );
}

/// Writes a session log whose user turn is followed by an assistant `Edit` of `edit_path`.
fn write_backfill_session(dir: &Path, file: &str, user_text: &str, edit_path: &str) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let user = serde_json::json!({
        "type": "user",
        "cwd": "/proj",
        "message": {"role": "user", "content": user_text},
        "timestamp": "2026-03-01T00:00:00Z"
    });
    let assistant = serde_json::json!({
        "type": "assistant",
        "cwd": "/proj",
        "message": {"role": "assistant", "content": [
            {"type": "text", "text": "editing"},
            {"type": "tool_use", "id": "t1", "name": "Edit",
             "input": {"file_path": edit_path, "old_string": "x", "new_string": "y"}}
        ]},
        "timestamp": "2026-03-01T00:00:01Z"
    });
    let full = dir.join(file);
    fs::write(
        &full,
        format!(
            "{}\n{}",
            serde_json::to_string(&user).unwrap(),
            serde_json::to_string(&assistant).unwrap()
        ),
    )
    .unwrap();
    full
}

fn collect_i64(conn: &Connection, sql: &str) -> Vec<i64> {
    let mut stmt = conn.prepare(sql).unwrap();
    let rows = stmt.query_map([], |r| r.get::<_, i64>(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

fn body_snapshot(conn: &Connection) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    (
        collect_i64(conn, "SELECT rowid FROM messages ORDER BY rowid"),
        collect_i64(conn, "SELECT id FROM qa_chunks ORDER BY id"),
        collect_i64(conn, "SELECT chunk_id FROM vec_chunks ORDER BY chunk_id"),
    )
}

fn session_file_paths(conn: &Connection) -> Vec<String> {
    let mut stmt = conn
        .prepare("SELECT path FROM session_files ORDER BY path")
        .unwrap();
    let rows = stmt.query_map([], |r| r.get::<_, String>(0)).unwrap();
    rows.map(Result::unwrap).collect()
}

fn reset_to_legacy_state(conn: &Connection) {
    conn.execute("UPDATE sessions SET files_scanned = NULL", [])
        .unwrap();
    conn.execute("DELETE FROM session_files", []).unwrap();
}

#[test]
fn legacy_file_backfill_records_touched_files_and_parse_loss_without_changing_messages_chunks_or_embeddings()
 {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    let partial = write_backfill_session(
        &claude_dir,
        "a.jsonl",
        "session a probe",
        "/proj/marker_a.rs",
    );
    write_backfill_session(
        &claude_dir,
        "b.jsonl",
        "session b probe",
        "/proj/marker_b.rs",
    );

    let body = fs::read_to_string(&partial).unwrap();
    fs::write(&partial, format!("{body}\ninvalid JSON\n")).unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        reset_to_legacy_state(&conn);
        conn.execute("DELETE FROM parse_diagnostics", []).unwrap();
        assert_eq!(
            body_snapshot(&conn),
            (vec![1, 2, 3, 4], vec![1, 2], vec![1, 2]),
            "precondition: the legacy DB holds two messages, one chunk and one vector per session"
        );
    }

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let partial = partial.to_str().unwrap();
    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 2,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 2,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "pending_body_bytes_peak": 0,
                        "pending_metadata_bytes": 0,
                        "pending_page_chunks_peak": 0,
                        "pending_pages": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 2,
                    },
                },
                "parse_diagnostics": [{
                    "file_path": partial,
                    "incomplete_tail_lines": 0,
                    "invalid_json_lines": 1,
                    "invalid_utf8_lines": 0,
                    "read_error": false,
                    "source": "claude",
                }],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [format!(
                "{partial}: skipped 1 invalid JSON line(s) and 0 invalid UTF-8 line(s); repair the source file, then rerun `recall index` (use `recall rebuild` if size and mtime are unchanged)"
            )],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        body_snapshot(&conn),
        (vec![1, 2, 3, 4], vec![1, 2], vec![1, 2]),
        "backfill must not change messages / qa_chunks / vec_chunks row counts or id sets"
    );
    assert_eq!(
        session_file_paths(&conn),
        vec![
            "/proj/marker_a.rs".to_owned(),
            "/proj/marker_b.rs".to_owned(),
        ],
        "the backfill must fill session_files from the write-target tool_use paths"
    );
}

#[test]
fn index_with_its_claude_root_missing_keeps_the_session_and_reports_it_skipped_in_json() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let gone_claude = src.path().join("claude_gone");
    let missing_opts = indexer::IndexOptions {
        force: false,
        claude_dir: &gone_claude,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome =
        index_and_report_with(&Some(db_path.clone()), &missing_opts, mock_loader).unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 0,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "pending_body_bytes_peak": 0,
                        "pending_metadata_bytes": 0,
                        "pending_page_chunks_peak": 0,
                        "pending_pages": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [{
                    "preserved_sessions": 1,
                    "reason": "missing_root",
                    "source": "claude",
                }],
            },
            "degraded": true,
            "notes": [
                "claude root unavailable — preserved 1 existing session(s), none deleted this run; re-run `recall index` once the root is back so a full scan can reconcile any genuinely deleted sessions",
            ],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    let hits = search::search(&conn, "quokka", &search::SearchOptions::default()).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the session under the missing root stays searchable"
    );
}

#[test]
fn index_without_the_model_stores_the_session_for_text_search_and_reports_its_chunk_unattempted() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 1,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 1,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 1,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "note: embedding model not installed; run `recall model download` to enable semantic search.",
            ],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    let search_opts = search::SearchOptions::default();
    let hits = search::search(&conn, "quokka", &search_opts).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the FTS index must be complete after a degraded index: the seeded term returns its session"
    );
}

#[test]
fn index_without_the_model_keeps_the_embeddings_of_a_changed_session_and_reports_them_preserved() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            1,
            "precondition: the first index embeds the session's one chunk"
        );
    }

    fs::write(
            claude_dir.join("s1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"quokka rebuild probe, edited after it was embedded"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 1,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 1,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "note: embedding model not installed; run `recall model download` to enable semantic search.",
                "preserved embeddings for 1 session(s) the model could not rebuild this run — semantic search for them stays intact; re-run `recall index` once the model can embed again to refresh any that changed",
            ],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        1,
        "the changed session keeps its stored embedding"
    );
}

#[test]
fn rebuild_whose_model_fails_its_probe_keeps_the_stored_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            1,
            "precondition: the initial index embeds the session's one chunk"
        );
    }

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Ok(Arc::new(embedder::MockEmbedder::failing_after(0)) as Arc<dyn Embed>)
    })
    .unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 1,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 1,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "note: embedding model unavailable; results from text search only.",
                "preserved embeddings for 1 session(s) the model could not rebuild this run — semantic search for them stays intact; re-run `recall index` once the model can embed again to refresh any that changed",
            ],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        1,
        "a rebuild whose model fails its probe keeps the stored embedding"
    );
}

#[test]
fn model_present_rebuild_heals_orphan_vec_rows() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    const ORPHAN_ID: i64 = 9_999_999;
    {
        let conn = open_or_create_db(&db_path).unwrap();
        let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, 0)",
            rusqlite::params![embedder::f32_as_bytes(&orphan), ORPHAN_ID],
        )
        .unwrap();
    }

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let orphan_remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = ?1",
            [ORPHAN_ID],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_remaining, 0,
        "a model-present rebuild must delete the pre-existing orphan vec row"
    );
}

#[test]
fn model_absent_rebuild_heals_orphan_and_preserves_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            1,
            "precondition: the initial index embeds the session's one chunk"
        );
    }

    const ORPHAN_ID: i64 = 9_999_999;
    {
        let conn = open_or_create_db(&db_path).unwrap();
        let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, ?2, 0)",
            rusqlite::params![embedder::f32_as_bytes(&orphan), ORPHAN_ID],
        )
        .unwrap();
    }

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let orphan_remaining: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id = ?1",
            [ORPHAN_ID],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        orphan_remaining, 0,
        "the orphan vec row must be deleted on the model-absent rebuild arm"
    );
    assert_eq!(
        vec_chunk_count(&conn),
        1,
        "preserved embeddings (stable qa ids) survive; only the orphan is removed"
    );
}

#[test]
fn rebuild_without_the_model_keeps_the_embedded_session_and_indexes_a_new_one_for_text_search() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            1,
            "precondition: the first session's one chunk is embedded"
        );
    }

    fs::write(
            claude_dir.join("s2.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"platypus newcomer"},"timestamp":"2026-03-02T00:00:00Z"}"#,
        )
        .unwrap();
    let rebuild_opts = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild_opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 1,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 1,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 1,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 1,
                        "files_discovered": 2,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 2,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 1,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "note: embedding model not installed; run `recall model download` to enable semantic search.",
                "preserved embeddings for 1 session(s) the model could not rebuild this run — semantic search for them stays intact; re-run `recall index` once the model can embed again to refresh any that changed",
            ],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        1,
        "the original session's embeddings must be preserved, not destroyed"
    );
    let search_opts = search::SearchOptions::default();
    let hits = search::search(&conn, "platypus", &search_opts).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the new session must be FTS-indexed even with the model absent"
    );
}

#[test]
fn model_absent_rebuild_reindexes_cached_session_without_embeddings() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    fs::create_dir_all(&claude_dir).unwrap();
    let session = claude_dir.join("s1.jsonl");
    fs::write(
            &session,
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"alphaword original"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            0,
            "precondition: a model-absent index embeds nothing"
        );
    }

    fs::write(
            &session,
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"betaword replaced"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 1,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 1,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 1,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "note: embedding model not installed; run `recall model download` to enable semantic search.",
            ],
        }),
        "a cached session without embeddings is not preserved"
    );
    let conn = open_or_create_db(&db_path).unwrap();
    let search_opts = search::SearchOptions::default();
    assert_eq!(
        search::search(&conn, "betaword", &search_opts)
            .unwrap()
            .len(),
        1,
        "the rebuild must re-index the new content, not freeze the old"
    );
    assert_eq!(
        search::search(&conn, "alphaword", &search_opts)
            .unwrap()
            .len(),
        0,
        "the stale content must be gone, proving a real re-index occurred"
    );
}

#[test]
fn index_whose_embedding_batch_fails_reports_the_failed_chunks_and_embeds_them_on_the_next_run() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    // The load probe is the embedder's first call and the only one that succeeds; every batch
    // after it fails.
    let failed = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Ok(Arc::new(embedder::MockEmbedder::failing_after(1)) as Arc<dyn Embed>)
    })
    .unwrap();

    assert_eq!(
        printed_json(&failed),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 1,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 1,
                        "chunks_failed": 1,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 1,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 1,
                        "inference_chunks": 1,
                        "pending_body_bytes_peak": 20,
                        "pending_metadata_bytes": 96,
                        "pending_page_chunks_peak": 1,
                        "pending_pages": 1,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "1 chunk(s) failed to embed (batch inference failed); rerun `recall index` to retry",
            ],
        })
    );

    let retried = index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    assert_eq!(
        printed_json(&retried),
        serde_json::json!({
            "data": {
                "embedded": 1,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 1,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 1,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 1,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 1,
                        "inference_chunks": 1,
                        "pending_body_bytes_peak": 20,
                        "pending_metadata_bytes": 96,
                        "pending_page_chunks_peak": 1,
                        "pending_pages": 1,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": false,
            "notes": [],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(vec_chunk_count(&conn), 1, "the retried chunk is stored");
}

/// One session-log line holding a user turn whose text is `text`.
fn user_turn_line(text: &str) -> String {
    let turn = serde_json::json!({
        "type": "user",
        "cwd": "/proj",
        "message": {"role": "user", "content": text},
        "timestamp": "2026-03-01T00:00:00Z"
    });
    format!("{turn}\n")
}

#[test]
fn index_whose_second_embedding_batch_fails_keeps_the_first_batch_and_hands_the_flags_to_both_batches()
 {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    fs::create_dir_all(&claude_dir).unwrap();
    // Pending chunks are embedded shortest first, so the 128 short turns fill the first batch
    // and the one longer turn is the whole second batch.
    let long_text = "wombat batch, the longest turn";
    fs::write(
        claude_dir.join("s1.jsonl"),
        format!(
            "{}{}",
            user_turn_line("wombat batch").repeat(128),
            user_turn_line(long_text)
        ),
    )
    .unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let embedder = Arc::new(embedder::MockEmbedder::failing_on_text(long_text));

    let outcome = index_and_report_observed(
        &Some(db_path),
        &opts,
        || Ok(embedder.clone()),
        embed_options_from_flags(Some(2048), Some(700)),
        &index_observer::Observer::stderr(),
    )
    .unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 128,
                "failed_count": 1,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 129,
                        "chunks_failed": 1,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 129,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 1,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 128,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 1,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 1,
                        "files_updated_committed": 1,
                        "inference_batches": 2,
                        "inference_chunks": 129,
                        "pending_body_bytes_peak": 1566,
                        "pending_metadata_bytes": 6144,
                        "pending_page_chunks_peak": 129,
                        "pending_pages": 1,
                        "sessions_chunk_pending": 1,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 1,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                "1 chunk(s) failed to embed (batch inference failed); rerun `recall index` to retry",
            ],
        })
    );
    let flags = embedder::EmbedOptions {
        token_budget: Some(2048),
        forward_pause: Some(Duration::from_millis(700)),
    };
    assert_eq!(
        embedder.captured_batch_options(),
        vec![flags, flags],
        "--token-budget 2048 --forward-pause-ms 700 reach both the saved and the failed batch"
    );
}

#[test]
fn lone_parse_diagnostic_degrades_and_prints_a_240_character_path_without_control_characters() {
    let outcome = IndexOutcome {
        parse_diagnostics: vec![indexer::FileParseDiagnostics {
            file_path: format!("/logs/\x1b[31mfile\n\t{}", "界".repeat(300)).into(),
            source: parser::Source::Claude,
            lines: parser::LineDiagnostics {
                invalid_json_lines: 2,
                invalid_utf8_lines: 1,
                incomplete_tail_lines: 1,
            },
            read_error: true,
        }],
        ..IndexOutcome::default()
    };

    // The color escape and the newline are dropped and the tab becomes a space.
    let path = format!("/logs/file {}", "界".repeat(229));
    assert_eq!(
        printed_envelope(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": null,
                "parse_diagnostics": [{
                    "file_path": path,
                    "incomplete_tail_lines": 1,
                    "invalid_json_lines": 2,
                    "invalid_utf8_lines": 1,
                    "read_error": true,
                    "source": "claude",
                }],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [
                format!("{path}: skipped 2 invalid JSON line(s) and 1 invalid UTF-8 line(s); repair the source file, then rerun `recall index` (use `recall rebuild` if size and mtime are unchanged)"),
                format!("{path}: 1 incomplete final line(s), possibly still being written; wait for the next append, then rerun `recall index`; if the writer has stopped, repair the source file"),
                format!("{path}: 1 file could not be read; check file access and UTF-8 path encoding, then rerun `recall index`; existing indexed data was retained"),
            ],
        })
    );
}

#[test]
fn parse_loss_notes_come_after_the_embedding_root_and_preserved_notes_of_the_same_run() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    seed_claude_source(&claude_dir);
    let broken = claude_dir.join("broken.jsonl");
    fs::write(
        &broken,
        "{\"type\":\"user\",\"cwd\":\"/proj\",\"message\":{\"role\":\"user\",\"content\":\"wombat broken\"},\"timestamp\":\"2026-03-01T00:00:00Z\"}\ninvalid JSON\n",
    )
    .unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    // An empty root is not an absolute path, so index does not read it and reports it as
    // incompletely enumerated.
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: Path::new(""),
        codex_archived_dir: None,
    };
    let rebuild_opts = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: Path::new(""),
        codex_archived_dir: None,
    };
    let broken = broken.to_str().unwrap();
    let parse_diagnostic = serde_json::json!({
        "file_path": broken,
        "incomplete_tail_lines": 0,
        "invalid_json_lines": 1,
        "invalid_utf8_lines": 0,
        "read_error": false,
        "source": "claude",
    });
    let unread_codex_root = serde_json::json!({
        "preserved_sessions": 0,
        "reason": "incomplete_enumeration",
        "source": "codex",
    });
    let unread_codex_root_note = "codex root could not be fully read; some sessions may be missing from the index. check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index`";
    let parse_loss_note = format!(
        "{broken}: skipped 1 invalid JSON line(s) and 0 invalid UTF-8 line(s); repair the source file, then rerun `recall index` (use `recall rebuild` if size and mtime are unchanged)"
    );

    let failed = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Ok(Arc::new(embedder::MockEmbedder::failing_after(1)) as Arc<dyn Embed>)
    })
    .unwrap();

    assert_eq!(
        printed_json(&failed),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 2,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 2,
                        "chunks_failed": 2,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 2,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 2,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 2,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 2,
                        "files_updated_committed": 2,
                        "inference_batches": 1,
                        "inference_chunks": 2,
                        "pending_body_bytes_peak": 33,
                        "pending_metadata_bytes": 96,
                        "pending_page_chunks_peak": 2,
                        "pending_pages": 1,
                        "sessions_chunk_pending": 2,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 2,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 2,
                    },
                },
                "parse_diagnostics": [parse_diagnostic],
                "preserved_embedded": 0,
                "skipped_roots": [unread_codex_root],
            },
            "degraded": true,
            "notes": [
                "2 chunk(s) failed to embed (batch inference failed); rerun `recall index` to retry",
                unread_codex_root_note,
                parse_loss_note,
            ],
        })
    );

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let rebuilt = index_and_report_with(&Some(db_path.clone()), &rebuild_opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        printed_json(&rebuilt),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 2,
                        "files_discovered": 2,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 2,
                    },
                },
                "parse_diagnostics": [parse_diagnostic],
                "preserved_embedded": 2,
                "skipped_roots": [unread_codex_root],
            },
            "degraded": true,
            "notes": [
                "note: embedding model not installed; run `recall model download` to enable semantic search.",
                unread_codex_root_note,
                "preserved embeddings for 2 session(s) the model could not rebuild this run — semantic search for them stays intact; re-run `recall index` once the model can embed again to refresh any that changed",
                parse_loss_note,
            ],
        })
    );
}

#[test]
fn embed_failure_note_strips_control_characters_and_keeps_the_first_120_characters_of_the_error() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 0,
        failed_count: 2,
        first_error: Some(format!(
            "batch: \x1b[31mboom\x1b[0m {}{}",
            "e".repeat(100),
            "界".repeat(200)
        )),
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    assert_eq!(
        printed_envelope(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 2,
                "observations": null,
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [],
            },
            "degraded": true,
            "notes": [format!(
                "2 chunk(s) failed to embed (batch: boom {}{}); rerun `recall index` to retry",
                "e".repeat(100),
                "界".repeat(8)
            )],
        })
    );
}

#[test]
fn index_with_a_claude_root_that_cannot_be_read_reports_its_preserved_session_as_incomplete_enumeration()
 {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    seed_claude_source(&claude_dir);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    // An empty root is not an absolute path, so index does not read it.
    let unread_opts = indexer::IndexOptions {
        force: false,
        claude_dir: Path::new(""),
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &unread_opts, mock_loader).unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 0,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "pending_body_bytes_peak": 0,
                        "pending_metadata_bytes": 0,
                        "pending_page_chunks_peak": 0,
                        "pending_pages": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 1,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [{
                    "preserved_sessions": 1,
                    "reason": "incomplete_enumeration",
                    "source": "claude",
                }],
            },
            "degraded": true,
            "notes": [
                "claude root could not be fully read — preserved 1 existing session(s), none deleted this run; check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index` so the next full scan can reconcile any genuinely deleted sessions",
            ],
        })
    );
    let conn = open_or_create_db(&db_path).unwrap();
    let hits = search::search(&conn, "quokka", &search::SearchOptions::default()).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "the session under the unread root stays searchable"
    );
}

#[test]
fn index_with_a_claude_root_that_cannot_be_read_and_no_stored_sessions_warns_that_sessions_may_be_missing()
 {
    let src = tempfile::TempDir::new().unwrap();
    let (_, codex_dir) = absent_source_dirs(src.path());

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");

    // An empty root is not an absolute path, so index does not read it.
    let unread_opts = indexer::IndexOptions {
        force: false,
        claude_dir: Path::new(""),
        codex_dir: &codex_dir,
        codex_archived_dir: None,
    };
    let outcome = index_and_report_with(&Some(db_path), &unread_opts, mock_loader).unwrap();

    assert_eq!(
        printed_json(&outcome),
        serde_json::json!({
            "data": {
                "embedded": 0,
                "failed_count": 0,
                "observations": {
                    "counts": {
                        "chunks_created_committed": 0,
                        "chunks_failed": 0,
                        "chunks_matched_committed": 0,
                        "chunks_pending_snapshot": 0,
                        "chunks_reconciled_created_committed": 0,
                        "chunks_remaining_snapshot": 0,
                        "chunks_removed_committed": 0,
                        "chunks_save_failed": 0,
                        "chunks_saved": 0,
                        "chunks_stale": 0,
                        "chunks_unattempted": 0,
                        "embeddings_reused_committed": 0,
                        "files_deferred": 0,
                        "files_discovered": 0,
                        "files_empty": 0,
                        "files_failed": 0,
                        "files_remaining": 0,
                        "files_unchanged": 0,
                        "files_updated": 0,
                        "files_updated_committed": 0,
                        "inference_batches": 0,
                        "inference_chunks": 0,
                        "pending_body_bytes_peak": 0,
                        "pending_metadata_bytes": 0,
                        "pending_page_chunks_peak": 0,
                        "pending_pages": 0,
                        "sessions_chunk_pending": 0,
                        "sessions_chunk_remaining": 0,
                        "sessions_chunked_committed": 0,
                        "sessions_chunked_empty": 0,
                        "sessions_stored": 0,
                    },
                },
                "parse_diagnostics": [],
                "preserved_embedded": 0,
                "skipped_roots": [{
                    "preserved_sessions": 0,
                    "reason": "incomplete_enumeration",
                    "source": "claude",
                }],
            },
            "degraded": true,
            "notes": [
                "claude root could not be fully read; some sessions may be missing from the index. check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index`",
            ],
        })
    );
}

#[test]
fn freeze_index_payload_key_set() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 4,
        failed_count: 0,
        first_error: None,
        skipped_roots: vec![indexer::SkippedRoot {
            source: parser::Source::Codex,
            preserved_sessions: 2,
            reason: indexer::SkippedReason::IncompleteEnumeration,
            other_roots_reconciled: false,
        }],
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert_eq!(
        payload_keys(&out.data),
        [
            "embedded",
            "failed_count",
            "observations",
            "parse_diagnostics",
            "preserved_embedded",
            "skipped_roots"
        ],
        "index --json top-level keys are frozen"
    );
    assert_eq!(
        payload_keys(&out.data["skipped_roots"][0]),
        ["preserved_sessions", "reason", "source"],
        "index --json skipped_roots element keys are frozen"
    );
}

#[test]
fn index_still_starts_the_probe_in_an_environment_search_has_recorded() {
    let dir = tempfile::TempDir::new().unwrap();
    let records_dir = dir.path().join("probe");
    let host = FakeHost::healthy();
    let first_search = host.serve(dir.path(), &records_dir, |model_host| {
        model_host.load_for_search()
    });
    let second_search = host.serve(dir.path(), &records_dir, |model_host| {
        model_host.load_for_search()
    });
    assert!(first_search.is_ok());
    assert!(second_search.is_ok());
    assert_eq!(host.probes.get(), 1, "the second search skipped the probe");

    let loaded = host.serve(dir.path(), &records_dir, index_embedder);

    assert!(loaded.is_ok());
    assert_eq!(
        host.probes.get(),
        2,
        "index and rebuild run the probe every time"
    );
}
