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

/// The `--json` envelope `recall index` prints for a run, without `data.observations`, whose
/// stage seconds change from run to run.
fn printed_json(outcome: &IndexOutcome) -> serde_json::Value {
    let printed = render_json_success(&index_command_output(outcome));
    let mut envelope: serde_json::Value = serde_json::from_str(&printed).unwrap();
    envelope["data"]
        .as_object_mut()
        .unwrap()
        .remove("observations");
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
fn index_and_report_forwards_default_embed_options_when_flags_unset() {
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

    let result = index_and_report_with(&Some(db_path), &opts, || Ok(embedder.clone())).unwrap();

    assert_eq!(
        result.embedded, 1,
        "default EmbedOptions must preserve current embed behavior"
    );
    assert_eq!(
        embedder.captured_options(),
        Some(embedder::EmbedOptions::default()),
        "default EmbedOptions (None/None) must forward when no flags are set"
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
    let outcome = index_and_report_with(&Some(db_path), &missing_opts, mock_loader).unwrap();

    assert_eq!(
        outcome.skipped_roots.len(),
        1,
        "the missing claude root must be threaded into the outcome"
    );
    assert_eq!(outcome.skipped_roots[0].source, parser::Source::Claude);
    assert_eq!(
        outcome.skipped_roots[0].preserved_sessions, 1,
        "the one preserved claude session must be counted"
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
    });
    assert!(
        outcome.is_ok(),
        "a model-less index must complete FTS and exit 0, not abort"
    );
    let outcome = outcome.unwrap();
    let counts = &outcome.observations["counts"];
    assert_eq!(counts["chunks_pending_snapshot"], 1);
    assert_eq!(counts["chunks_unattempted"], 1);
    assert_eq!(counts["chunks_remaining_snapshot"], 1);
    for key in [
        "chunks_saved",
        "chunks_failed",
        "chunks_stale",
        "chunks_save_failed",
    ] {
        assert_eq!(counts[key], 0);
    }

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
fn index_and_report_with_unavailable_loader_carries_unavailability_note() {
    for reason in [
        DegradedReason::BackendUnavailable,
        DegradedReason::ProbeFailed,
    ] {
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
        let outcome = index_and_report_with(&Some(db_path), &opts, || Err(reason))
            .expect("a degraded index must succeed");

        let note = outcome
            .degraded_note
            .unwrap_or_else(|| panic!("{reason:?} must carry a degraded note"));
        assert!(
            note.contains("unavailable"),
            "{reason:?} must report unavailability (not a download hint), got: {note}"
        );
        assert!(
            !note.to_lowercase().contains("download"),
            "{reason:?} is not repaired by re-downloading, so the note must not \
                 suggest it (that hint belongs to NotInstalled), got: {note}"
        );
        assert_eq!(outcome.embedded, 0, "{reason:?} skips embedding");
        assert_eq!(outcome.failed_count, 0, "{reason:?} fails no batches");
    }
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
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(
        before > 0,
        "precondition: the initial index must embed at least one chunk"
    );

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

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "a load-OK-but-inference-fails rebuild must preserve existing embeddings, not destroy them"
    );
    assert!(
        outcome.preserved_embedded > 0,
        "the preserved-embedding count must ride the degraded arm"
    );
    assert!(
        outcome.degraded_note.is_some(),
        "an inference-incapable model must report a degraded note"
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
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(
        before > 0,
        "precondition: the initial index embedded chunks"
    );

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
        before,
        "preserved embeddings (stable qa ids) survive; only the orphan is removed"
    );
}

#[test]
fn model_absent_rebuild_surfaces_preserved_count_and_note() {
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

    assert!(
        outcome.preserved_embedded > 0,
        "the outcome must report the preserved session count, got {}",
        outcome.preserved_embedded
    );

    let out = index_command_output(&outcome);
    assert!(
        out.degraded,
        "a model-absent run with preservation degrades"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains(&outcome.preserved_embedded.to_string())),
        "the preserve note must name the preserved count, got: {:?}",
        out.notes
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "the degraded note must carry the download remedy for a not-installed model, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["preserved_embedded"], outcome.preserved_embedded,
        "the --json payload carries the preserved magnitude for an agent"
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
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(before > 0, "precondition: the first session was embedded");

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

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "the original session's embeddings must be preserved, not destroyed"
    );
    assert_eq!(
        outcome.preserved_embedded, 1,
        "only the one embedded session is preserved; the new one is indexed"
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
        outcome.preserved_embedded, 0,
        "a cached session without embeddings must not be preserved"
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

fn embed_stop_outcome() -> IndexOutcome {
    IndexOutcome {
        degraded_note: None,
        embedded: 5,
        failed_count: 3,
        first_error: Some("batch: mock failure".to_owned()),
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    }
}

fn model_absent_outcome() -> IndexOutcome {
    IndexOutcome {
        degraded_note: Some("note: run `recall model download` to enable.".to_owned()),
        embedded: 0,
        failed_count: 0,
        first_error: None,
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    }
}

fn success_outcome() -> IndexOutcome {
    IndexOutcome {
        degraded_note: None,
        embedded: 8,
        failed_count: 0,
        first_error: None,
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    }
}

#[test]
fn index_command_output_surfaces_embed_stop_with_failed_count_and_retry() {
    let out = index_command_output(&embed_stop_outcome());

    assert!(
        out.degraded,
        "an embed-stop index is degraded: --json must not report it as a clean success"
    );
    assert!(
        out.notes[0].contains("3 chunk"),
        "the note must name how many chunks failed, got: {:?}",
        out.notes
    );
    assert!(
        out.notes[0].contains("recall index"),
        "the note must guide the agent to retry via `recall index`, got: {:?}",
        out.notes
    );
    assert!(
        out.notes[0].contains("mock failure"),
        "the note carries the first batch error, not the unknown-error fallback, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["failed_count"], 3,
        "the payload carries the failure magnitude, not just a bool"
    );
    assert_eq!(
        out.data["embedded"], 5,
        "the payload reports how many chunks did embed despite the stop"
    );
}

#[test]
fn index_command_output_clean_success_is_not_degraded() {
    let out = index_command_output(&success_outcome());

    assert!(!out.degraded, "a fully embedded index is not degraded");
    assert!(
        out.notes.is_empty(),
        "a clean success carries no notes, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["failed_count"], 0,
        "the payload reports zero failures on success"
    );
    assert_eq!(
        out.data["embedded"], 8,
        "the payload reports how many chunks were embedded on success"
    );
}

#[test]
fn index_command_output_degraded_covaries_with_notes() {
    for outcome in [
        embed_stop_outcome(),
        model_absent_outcome(),
        success_outcome(),
    ] {
        let out = index_command_output(&outcome);
        assert_eq!(
            out.degraded,
            !out.notes.is_empty(),
            "degraded must equal notes-non-empty (degraded={}, notes={:?})",
            out.degraded,
            out.notes
        );
    }
}

#[test]
fn parse_diagnostics_degrade_alone_and_coexist_with_embedding_and_root_reasons() {
    for mut outcome in [
        success_outcome(),
        model_absent_outcome(),
        embed_stop_outcome(),
    ] {
        let existing_notes = index_command_output(&outcome).notes;
        outcome
            .parse_diagnostics
            .push(indexer::FileParseDiagnostics {
                file_path: format!("/logs/\x1b[31mfile\n\t{}", "界".repeat(300)).into(),
                source: parser::Source::Claude,
                lines: parser::LineDiagnostics {
                    invalid_json_lines: 2,
                    invalid_utf8_lines: 1,
                    incomplete_tail_lines: 1,
                },
                read_error: true,
            });
        let without_root = index_command_output(&outcome);
        assert!(without_root.degraded);
        assert_eq!(without_root.notes.len(), existing_notes.len() + 3);
        outcome.skipped_roots.push(indexer::SkippedRoot {
            source: parser::Source::Codex,
            preserved_sessions: 1,
            reason: indexer::SkippedReason::MissingRoot,
            other_roots_reconciled: false,
        });
        let out = index_command_output(&outcome);
        assert!(out.degraded);
        for note in existing_notes {
            assert!(out.notes.contains(&note));
        }
        assert!(out.notes.iter().any(|n| n.contains("root unavailable")));
        let d = &out.data["parse_diagnostics"][0];
        let path = d["file_path"].as_str().unwrap();
        assert_eq!(path.chars().count(), 240);
        assert!(path.starts_with("/logs/file"));
        assert!(!path.chars().any(char::is_control));
        assert_eq!(d["invalid_json_lines"], 2);
        assert_eq!(d["invalid_utf8_lines"], 1);
        assert_eq!(d["incomplete_tail_lines"], 1);
        assert_eq!(d["read_error"], true);
        for note in &out.notes {
            assert!(!note.chars().any(char::is_control));
            assert!(note.chars().count() < 500);
        }
    }
}

#[test]
fn embed_failure_note_strips_control_characters_and_keeps_the_first_120_characters_of_the_error() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 0,
        failed_count: 2,
        first_error: Some(format!("batch: \x1b[31mboom\x1b[0m {}", "e".repeat(300))),
        skipped_roots: Vec::new(),
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert!(
        !out.notes[0].contains('\x1b'),
        "control chars are stripped from the note, got: {:?}",
        out.notes
    );
    assert!(
        out.notes[0].len() < 200,
        "a verbose error is capped, got len {}",
        out.notes[0].len()
    );
    assert!(
        out.notes[0].contains("rerun `recall index`"),
        "the retry guidance survives the capping, got: {:?}",
        out.notes
    );
}

#[test]
fn index_command_output_surfaces_skipped_root_in_json() {
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

    assert!(
        out.degraded,
        "a skipped root is a degradation: --json must not report it as a clean success"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("codex") && n.contains("2") && n.contains("recall index")),
        "the note must name the source, the preserved count, and the reconcile command, got: {:?}",
        out.notes
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("permissions and access")),
        "an incomplete-enumeration root must surface the access remedy, not the missing-root wording, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["skipped_roots"][0]["source"], "codex",
        "the payload carries the skipped source, not just prose"
    );
    assert_eq!(
        out.data["skipped_roots"][0]["preserved_sessions"], 2,
        "the payload carries the preserved magnitude an agent can act on"
    );
    assert_eq!(
        out.data["skipped_roots"][0]["reason"], "incomplete_enumeration",
        "the payload carries the machine-readable skip cause so an agent picks the right remedy"
    );
}

#[test]
fn index_command_note_for_zero_preserved_incomplete_root_omits_reconcile() {
    let outcome = IndexOutcome {
        degraded_note: None,
        embedded: 0,
        failed_count: 0,
        first_error: None,
        skipped_roots: vec![indexer::SkippedRoot {
            source: parser::Source::Claude,
            preserved_sessions: 0,
            reason: indexer::SkippedReason::IncompleteEnumeration,
            other_roots_reconciled: false,
        }],
        preserved_embedded: 0,
        parse_diagnostics: Vec::new(),
        observations: serde_json::Value::Null,
    };

    let out = index_command_output(&outcome);

    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("claude") && n.contains("some sessions may be missing")),
        "a zero-preserved incomplete root must warn that sessions may be missing, got: {:?}",
        out.notes
    );
    assert!(
        !out.notes
            .iter()
            .any(|n| n.contains("reconcile any real deletions")),
        "with nothing preserved the note must not promise to reconcile deletions, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["skipped_roots"][0]["reason"], "incomplete_enumeration",
        "the payload still carries the machine-readable cause at zero preserved"
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
    for _ in 0..2 {
        let searched = host.serve(dir.path(), &records_dir, |model_host| {
            model_host.load_for_search()
        });
        assert!(searched.is_ok());
    }
    assert_eq!(host.probes.get(), 1, "the second search skipped the probe");

    let loaded = host.serve(dir.path(), &records_dir, index_embedder);

    assert!(loaded.is_ok());
    assert_eq!(
        host.probes.get(),
        2,
        "index and rebuild run the probe every time"
    );
}
