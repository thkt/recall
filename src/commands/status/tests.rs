use crate::envelope::render_json_success;
use crate::model::tests::FakeHost;
use crate::tests::{bare_empty_db, payload_keys};
use crate::{embedder, open_or_create_db};

use super::*;

/// The `--json` envelope `recall status` prints for an output.
fn printed_envelope(out: &CommandOutput) -> serde_json::Value {
    serde_json::from_str(&render_json_success(out)).unwrap()
}

#[test]
fn status_counts_a_chunk_with_two_sub_embeddings_as_one_embedded_chunk() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 1, "one chunk, two sub-embeddings");
    let embedding = embedder::MockEmbedder::deterministic_vector("one chunk, two sub-embeddings");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0), (?1, 1, 1)",
        [embedder::f32_as_bytes(&embedding)],
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_status_in(false, &Some(db_path), host)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        "Sessions: 1\nQA chunks: 1\nEmbedded: 1/1\nModel: ready\n"
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "embedded": 1, "model_ready": true, "qa_chunks": 1, "sessions": 1 },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn status_without_a_database_reports_zero_counts_and_does_not_create_it() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("missing.db");

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_status_in(false, &Some(db_path.clone()), host)
        })
        .unwrap();

    assert_eq!(out.markdown, "");
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "embedded": 0, "model_ready": true, "qa_chunks": 0, "sessions": 0 },
            "degraded": false,
            "notes": [],
        })
    );
    assert!(
        !db_path.exists(),
        "status must not create the database while reporting"
    );
}

#[test]
fn status_on_a_database_file_without_tables_reports_zero_counts() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let not_installed = || Err(model::DegradedReason::NotInstalled);
    let out = FakeHost::healthy()
        .run(|steps| {
            run_status_in(
                false,
                &Some(db_path),
                &model::ModelHost {
                    resolve: &not_installed,
                    steps,
                },
            )
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        "Sessions: 0\nQA chunks: 0\nEmbedded: 0/0\nModel: not installed (run `recall model download`)\n"
    );
    // DR 0001: status reports a missing model on stdout without degrading or adding a note.
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": { "embedded": 0, "model_ready": false, "qa_chunks": 0, "sessions": 0 },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn freeze_status_payload_key_set() {
    let frozen = ["embedded", "model_ready", "qa_chunks", "sessions"];

    let missing = tempfile::TempDir::new().unwrap();
    let no_db = run_status(false, &Some(missing.path().join("absent.db"))).unwrap();
    assert_eq!(
        payload_keys(&no_db.data),
        frozen,
        "status --json keys are frozen on the no-database branch"
    );

    let dir = tempfile::TempDir::new().unwrap();
    let db_path = dir.path().join("recall.db");
    {
        let conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
    }
    let with_db = run_status(false, &Some(db_path)).unwrap();
    assert_eq!(
        payload_keys(&with_db.data),
        frozen,
        "status --json keys are frozen on the with-database branch"
    );
}
