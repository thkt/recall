use crate::tests::{bare_empty_db, payload_keys};
use crate::{embedder, open_or_create_db};

use super::*;

#[test]
fn run_status_counts_distinct_embedded_chunks_not_subembedding_rows() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    db::seed_session(&conn, "s1");
    db::seed_chunk(&conn, 1, "one chunk, two sub-embeddings");
    let emb = embedder::MockEmbedder::deterministic_vector("one chunk, two sub-embeddings");
    for sub_idx in 0..2 {
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, ?2)",
            rusqlite::params![embedder::f32_as_bytes(&emb), sub_idx],
        )
        .unwrap();
    }
    drop(conn);

    let out = run_status(false, &Some(db_path)).unwrap();

    assert_eq!(
        out.data["embedded"], 1,
        "one qa_chunk with two sub-embedding rows is one embedded chunk, not two"
    );
    assert_eq!(out.data["qa_chunks"], 1);
    assert!(
        out.markdown.contains("Embedded: 1/1"),
        "human output must not overstate coverage past the qa_chunks total, got: {}",
        out.markdown
    );
}

#[test]
fn run_status_does_not_create_db_when_absent() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("missing.db");

    let out = run_status(false, &Some(db_path.clone())).unwrap();

    assert_eq!(out.data["sessions"], 0);
    assert_eq!(out.data["qa_chunks"], 0);
    assert_eq!(out.data["embedded"], 0);
    assert!(
        !db_path.exists(),
        "status must not create the database while reporting"
    );
}

#[test]
fn run_status_on_existing_empty_db_reports_zero_counts() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = run_status(false, &Some(db_path)).unwrap();

    assert_eq!(out.data["sessions"], 0);
    assert_eq!(out.data["qa_chunks"], 0);
    assert_eq!(out.data["embedded"], 0);
}

#[test]
fn freeze_status_payload_key_set_both_branches() {
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
