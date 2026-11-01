use std::fs::{self, OpenOptions};
use std::path::Path;

use crate::tests::{mock_loader, payload_keys, vec_chunk_count};
use crate::{db, parser, search};

use super::*;

fn pending_embed_count(conn: &Connection) -> i64 {
    conn.query_row(
        "SELECT COUNT(*) FROM qa_chunks c \
             WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
        [],
        |r| r.get(0),
    )
    .unwrap()
}

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

#[test]
fn embed_all_pending_embeds_every_pending_chunk() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let mut conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    for i in 1..=3 {
        db::seed_chunk(&conn, i, &format!("pending content {i}"));
    }

    let result = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::new(),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();

    assert_eq!(result.embedded, 3, "every pending chunk embeds");
    assert_eq!(result.failed_count, 0, "a working embedder fails nothing");
    assert_eq!(pending_embed_count(&conn), 0, "the pending set empties");
    assert_eq!(vec_chunk_count(&conn), 3, "each chunk gains a vec row");
}

#[test]
fn embed_all_pending_partial_failure_reports_failed_count() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let mut conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 1, "x");
    for i in 2..=3 {
        db::seed_chunk(&conn, i, &format!("pending content {i}"));
    }

    let result = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::failing_on_text("x"),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();

    assert_eq!(
        result.embedded, 0,
        "the poisoned single batch embeds nothing"
    );
    assert_eq!(
        result.failed_count, 3,
        "the whole 3-chunk batch is reported failed"
    );
    let err = result.first_error.as_deref().unwrap_or_default();
    assert!(
        err == "batch inference failed",
        "the first batch error is carried up for the --json note, got: {err:?}"
    );
    assert_eq!(
        pending_embed_count(&conn),
        3,
        "the failed batch stays pending for the next run"
    );
}

#[test]
fn embed_all_pending_reembeds_nothing_on_second_run() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let mut conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    for (id, content) in [(1, "already embedded"), (2, "new pending")] {
        db::seed_chunk(&conn, id, content);
    }
    let emb = embedder::MockEmbedder::deterministic_vector("already embedded");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
        [embedder::f32_as_bytes(&emb)],
    )
    .unwrap();

    let first = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::new(),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(
        first.embedded, 1,
        "the first run embeds only the pending chunk"
    );
    assert_eq!(vec_chunk_count(&conn), 2, "vec gains exactly the new chunk");

    let second = embed_all_pending(
        &mut conn,
        &embedder::MockEmbedder::new(),
        &embedder::EmbedOptions::default(),
    )
    .unwrap();
    assert_eq!(second.embedded, 0, "the second run re-embeds nothing");
    assert_eq!(
        vec_chunk_count(&conn),
        2,
        "a second run must re-embed nothing (NOT EXISTS vec_chunks gate)"
    );
}

#[test]
fn index_and_report_forwards_index_flags_as_embed_options() {
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
    };

    let embedder = Arc::new(embedder::MockEmbedder::capturing_options());
    let options = embedder::EmbedOptions {
        token_budget: Some(2048),
        forward_pause: Some(Duration::from_millis(700)),
    };

    let result = index_and_report_observed(
        &Some(db_path),
        &opts,
        || Ok(embedder.clone()),
        options,
        &index_observer::Observer::stderr(),
    )
    .unwrap();

    assert_eq!(result.embedded, 1);
    assert_eq!(
        embedder.captured_options(),
        Some(options),
        "--token-budget 2048 --forward-pause-ms 700 must forward as \
             EmbedOptions{{token_budget:Some(2048), forward_pause:Some(700ms)}}"
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
fn rebuild_reembeds_all_and_rebuilds_fts_index() {
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
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            pending_embed_count(&conn),
            0,
            "precondition: initial index must leave nothing pending"
        );
    }

    let rebuild_opts = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild_opts, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        pending_embed_count(&conn),
        0,
        "rebuild must re-embed every chunk (pending=0 after full reset)"
    );

    let opts = search::SearchOptions::default();
    let hits = search::search(&conn, "quokka", &opts).unwrap();
    assert_eq!(
        hits.len(),
        1,
        "rebuild must reconstruct FTS: the seeded term must return its session"
    );
}

#[test]
fn index_and_report_with_backfills_legacy_null_rowid_ranges() {
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
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    {
        let conn = open_or_create_db(&db_path).unwrap();
        let cleared = conn
            .execute(
                "UPDATE qa_chunks SET src_rowid_lo = NULL, src_rowid_hi = NULL",
                [],
            )
            .unwrap();
        assert_eq!(cleared, 1, "precondition: one chunk to un-link");
    }

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let (lo, hi): (Option<i64>, Option<i64>) = conn
        .query_row(
            "SELECT src_rowid_lo, src_rowid_hi FROM qa_chunks",
            [],
            |r| Ok((r.get(0)?, r.get(1)?)),
        )
        .unwrap();
    assert!(
        lo.is_some() && hi.is_some(),
        "the index-path backfill must repopulate the legacy NULL rowid range, got lo={lo:?} hi={hi:?}"
    );
}

fn write_backfill_session(
    dir: &Path,
    file: &str,
    user_text: &str,
    edit_path: Option<&str>,
) -> PathBuf {
    fs::create_dir_all(dir).unwrap();
    let user = serde_json::json!({
        "type": "user",
        "cwd": "/proj",
        "message": {"role": "user", "content": user_text},
        "timestamp": "2026-03-01T00:00:00Z"
    });
    let mut content = serde_json::to_string(&user).unwrap();
    if let Some(path) = edit_path {
        let assistant = serde_json::json!({
            "type": "assistant",
            "cwd": "/proj",
            "message": {"role": "assistant", "content": [
                {"type": "text", "text": "editing"},
                {"type": "tool_use", "id": "t1", "name": "Edit",
                 "input": {"file_path": path, "old_string": "x", "new_string": "y"}}
            ]},
            "timestamp": "2026-03-01T00:00:01Z"
        });
        content.push('\n');
        content.push_str(&serde_json::to_string(&assistant).unwrap());
    }
    let full = dir.join(file);
    fs::write(&full, content).unwrap();
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
fn backfill_が本文_index_と_embedding_を変えない() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    let partial = write_backfill_session(
        &claude_dir,
        "a.jsonl",
        "session a probe",
        Some("/proj/marker_a.rs"),
    );
    write_backfill_session(
        &claude_dir,
        "b.jsonl",
        "session b probe",
        Some("/proj/marker_b.rs"),
    );

    let body = fs::read_to_string(&partial).unwrap();
    fs::write(&partial, format!("{body}\ninvalid JSON\n")).unwrap();

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    assert_eq!(
        index_command_output(&outcome).data["parse_diagnostics"][0]["invalid_json_lines"],
        1
    );
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        reset_to_legacy_state(&conn);
        conn.execute("DELETE FROM parse_diagnostics", []).unwrap();
        assert!(
            !collect_i64(&conn, "SELECT chunk_id FROM vec_chunks").is_empty(),
            "precondition: the legacy DB must hold embeddings, or the invariant is vacuous"
        );
        body_snapshot(&conn)
    };

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    assert_eq!(
        index_command_output(&outcome).data["parse_diagnostics"][0]["invalid_json_lines"],
        1
    );

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        body_snapshot(&conn),
        before,
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
fn 二回目の_index_で_backfill_が再実行されない() {
    let src = tempfile::TempDir::new().unwrap();
    let claude_dir = src.path().join("claude");
    let (_, codex_dir) = absent_source_dirs(src.path());
    let write_path = write_backfill_session(
        &claude_dir,
        "write.jsonl",
        "write session probe",
        Some("/proj/orig_marker.rs"),
    );
    write_backfill_session(&claude_dir, "zero.jsonl", "zero touch probe", None);

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let opts = indexer::IndexOptions {
        force: false,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    {
        let conn = open_or_create_db(&db_path).unwrap();
        reset_to_legacy_state(&conn);
    }
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let original = fs::metadata(&write_path).unwrap();
    let pinned = original.modified().unwrap();
    write_backfill_session(
        &claude_dir,
        "write.jsonl",
        "write session probe",
        Some("/proj/next_marker.rs"),
    );
    let f = OpenOptions::new().write(true).open(&write_path).unwrap();
    f.set_modified(pinned).unwrap();
    drop(f);
    let replaced = fs::metadata(&write_path).unwrap();
    assert_eq!(replaced.len(), original.len(), "canary must preserve size");
    assert_eq!(replaced.modified().unwrap(), pinned);

    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    let unmarked: i64 = conn
        .query_row(
            "SELECT COUNT(*) FROM sessions WHERE files_scanned IS NULL",
            [],
            |r| r.get(0),
        )
        .unwrap();
    assert_eq!(
        unmarked, 0,
        "every session (write-target and zero-touch) must carry files_scanned after the backfill"
    );
    assert!(
        !session_file_paths(&conn)
            .iter()
            .any(|p| p == "/proj/next_marker.rs"),
        "a marked session must not be re-read: the canary path must not appear in session_files"
    );

    conn.execute("DELETE FROM session_files WHERE session_id = 'write'", [])
        .unwrap();
    conn.execute(
        "UPDATE sessions SET files_scanned = NULL WHERE session_id = 'write'",
        [],
    )
    .unwrap();
    drop(conn);
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let conn = open_or_create_db(&db_path).unwrap();
    assert!(session_file_paths(&conn).contains(&"/proj/next_marker.rs".to_owned()));
}

#[test]
fn index_and_report_with_missing_root_threads_skipped_root_into_outcome() {
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
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let gone_claude = src.path().join("claude_gone");
    let missing_opts = indexer::IndexOptions {
        force: false,
        claude_dir: &gone_claude,
        codex_dir: &codex_dir,
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
fn index_and_report_with_degraded_loader_completes_fts_and_succeeds() {
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
fn index_and_report_with_degraded_loader_emits_model_download_note() {
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
    };
    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .expect("a degraded index must succeed");

    let note = outcome
        .degraded_note
        .expect("a model-less index must carry a degraded note");
    assert!(
        note.contains("recall model download"),
        "the degraded note must name `recall model download` to guide the user, got: {note}"
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
fn index_and_report_with_disabled_loader_is_indistinguishable_from_success() {
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
    };
    let outcome = index_and_report_with(&Some(db_path), &opts, || Err(DegradedReason::Disabled))
        .expect("a disabled-embedder index must succeed");

    assert_eq!(
        outcome.degraded_note, None,
        "opt-out is not a degradation: no note is carried"
    );
    assert_eq!(outcome.embedded, 0, "a disabled embedder embeds nothing");
    assert_eq!(
        outcome.failed_count, 0,
        "a disabled embedder fails no batches"
    );
    let envelope = index_command_output(&outcome);
    assert!(
        !envelope.degraded,
        "the --json envelope reports a disabled embedder as non-degraded"
    );
}

#[test]
fn model_absent_rebuild_preserves_existing_embeddings() {
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
    };
    index_and_report_with(&Some(db_path.clone()), &rebuild, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "a model-absent rebuild must preserve existing embeddings, not destroy them"
    );
}

#[test]
fn load_ok_inference_fail_rebuild_preserves_existing_embeddings() {
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
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

    let rebuild = indexer::IndexOptions {
        force: true,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
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
fn model_absent_index_adds_new_session_and_preserves_embeddings() {
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

#[test]
fn model_absent_incremental_index_preserves_edited_embedded_session() {
    use std::time::{Duration, SystemTime};

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
    };
    index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
    let before = {
        let conn = open_or_create_db(&db_path).unwrap();
        vec_chunk_count(&conn)
    };
    assert!(before > 0, "precondition: the session was embedded");

    let session = claude_dir.join("s1.jsonl");
    let mut all = fs::read_to_string(&session).unwrap();
    all.push('\n');
    all.push_str(
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"echidna appended"},"timestamp":"2026-03-01T00:00:01Z"}"#,
        );
    fs::write(&session, all).unwrap();
    let f = fs::OpenOptions::new().append(true).open(&session).unwrap();
    f.set_modified(SystemTime::now() + Duration::from_secs(10))
        .unwrap();
    drop(f);

    let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
        Err(DegradedReason::NotInstalled)
    })
    .unwrap();

    assert_eq!(
        outcome.preserved_embedded, 1,
        "the edited embedded session is preserved, not destroyed, on `recall index`"
    );
    let conn = open_or_create_db(&db_path).unwrap();
    assert_eq!(
        vec_chunk_count(&conn),
        before,
        "the embeddings survive the incremental re-index"
    );
    let search_opts = search::SearchOptions::default();
    assert_eq!(
        search::search(&conn, "quokka", &search_opts).unwrap().len(),
        1,
        "the preserved session's original FTS rows remain searchable"
    );
    assert_eq!(
        search::search(&conn, "echidna", &search_opts)
            .unwrap()
            .len(),
        0,
        "the appended turn stays out of FTS until the model returns (FR-2 trade-off)"
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
fn index_command_output_surfaces_model_absent_download_note() {
    let out = index_command_output(&model_absent_outcome());

    assert!(
        out.degraded,
        "a model-absent index is degraded: embedding was skipped"
    );
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("recall model download")),
        "the model-absent note must reach --json (not stderr only), got: {:?}",
        out.notes
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
fn index_command_output_sanitizes_and_bounds_first_error_in_note() {
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
