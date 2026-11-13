use std::cell::Cell;
use std::fs;
use std::sync::Arc;

use crate::commands::search::run_search_in;
use crate::embedder;
use crate::embedding::{ChunkedEmbedding, EmbedError};
use crate::model::tests::{CachedArtifacts, FakeHost};
use crate::open_or_create_db;
use crate::tests::{
    bare_empty_db, mock_loader, printed_envelope, search_command, seed_healthy_index,
};

use super::*;

fn no_recover() -> Result<(), RecallError> {
    panic!("model recovery must not run")
}

#[test]
fn doctor_fix_model_mismatch_preserves_index_and_never_suggests_deletion() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("index.db");
    {
        let conn = open_or_create_db(&path).unwrap();
        conn.execute(
            "UPDATE embedding_identity SET fingerprint='previous-model'",
            [],
        )
        .unwrap();
    }
    let out = run_doctor_with(false, true, &Some(path.clone()), mock_loader, || {
        panic!("model artifacts must not be replaced for an index identity mismatch")
    })
    .unwrap();
    assert_eq!(out.data["healthy"], false);
    assert!(out.degraded);
    assert!(
        out.notes
            .iter()
            .any(|n| n.contains("matching --model-manifest"))
    );
    assert!(out.notes.iter().all(|n| !n.contains("remove the database")));
    let (conn, _) = db::open_db_readonly(&path).unwrap();
    assert_eq!(
        conn.query_row("SELECT fingerprint FROM embedding_identity", [], |r| r
            .get::<_, String>(
            0
        ))
        .unwrap(),
        "previous-model"
    );
}

struct WidthEmbedder(usize);

impl Embed for WidthEmbedder {
    fn embed_query(&self, _text: &str) -> Result<Vec<f32>, EmbedError> {
        Ok(vec![0.0; self.0])
    }
    fn embed_document(&self, _text: &str) -> Result<ChunkedEmbedding, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
    fn embed_documents_batch(&self, _texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
    fn embed_text(&self, _text: &str, _prefix: &str) -> Result<Vec<f32>, EmbedError> {
        unreachable!("doctor probe only calls embed_query")
    }
}

#[test]
fn doctor_reports_healthy_on_intact_index_and_loadable_model() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let out = run_doctor_with(false, false, &Some(db_path), mock_loader, no_recover).unwrap();

    assert_eq!(out.data["healthy"], true);
    assert!(!out.degraded, "an intact index must not be degraded");
    assert!(
        out.notes.is_empty(),
        "no failing check means no remedy notes"
    );
    assert!(
        out.markdown.contains("Health: healthy"),
        "got: {}",
        out.markdown
    );
}

#[test]
fn doctor_flags_orphan_embedding_referencing_missing_chunk() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    let emb = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&emb)],
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, false, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [FAIL] 1 embedding(s) reference a missing chunk -> recall doctor --fix\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    {
                        "detail": "1 embedding(s) reference a missing chunk",
                        "name": "orphan_embeddings",
                        "ok": false,
                        "remedy": "recall doctor --fix",
                    },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["orphan_embeddings: recall doctor --fix"],
        })
    );
}

#[test]
fn doctor_fix_repairs_orphan_and_rechecks_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&orphan)],
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "Repaired orphan_embeddings: 1 row(s)\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": { "orphan_embeddings": 1 },
            },
            "degraded": false,
            "notes": [],
        })
    );
    assert_eq!(
        stored_rows(
            &db_path,
            "SELECT chunk_id, sub_idx FROM vec_chunks",
            |r| Ok((r.get::<_, i64>(0)?, r.get::<_, i64>(1)?))
        ),
        vec![(1, 0)]
    );
}

#[test]
fn render_doctor_report_surfaces_repair_error() {
    let path = PathBuf::from("/tmp/recall-test.db");
    let checks = vec![DoctorCheck::fail(
        "orphan_embeddings",
        "1 embedding(s) reference a missing chunk".to_owned(),
        "recall doctor --fix",
    )];
    let repairs = Some(vec![RepairOutcome {
        name: "orphan_embeddings",
        result: Err(anyhow::anyhow!("disk I/O error")),
    }]);

    let out = render_doctor_report(false, &path, &checks, &repairs, &ModelRecovery::Skipped);

    assert!(
        out.markdown
            .contains("Repair failed orphan_embeddings: disk I/O error"),
        "the repair cause must appear in human output, got: {}",
        out.markdown
    );
    assert!(
        out.notes.iter().any(|n| n.contains("repair failed")),
        "the repair cause must appear in notes, got: {:?}",
        out.notes
    );
    assert_eq!(
        out.data["repaired"]["orphan_embeddings"],
        serde_json::Value::Null,
        "a failed repair reports a null value under its name, got: {}",
        out.data
    );
}

#[test]
fn render_doctor_report_output_matches_every_check_repair_recovery_and_verbose_combination() {
    const RETRYABLE: &str = "download interrupted";
    const PERMANENT: &str = "artifact rejected";
    let path = PathBuf::from("/tmp/recall-test.db");

    for healthy in [true, false] {
        for with_repairs in [false, true] {
            for recovery_case in ["skipped", "succeeded", "retryable", "permanent"] {
                for verbose in [false, true] {
                    let mut checks = vec![
                        DoctorCheck::pass("plain"),
                        DoctorCheck::info("noted", "1 note".to_owned()),
                    ];
                    let mut check_json = vec![
                        serde_json::json!({"name": "plain", "ok": true, "detail": null, "remedy": null}),
                        serde_json::json!({"name": "noted", "ok": true, "detail": "1 note", "remedy": null}),
                    ];
                    let mut check_lines = "  plain [ok]\n  noted [ok] 1 note\n".to_owned();
                    let mut notes = Vec::new();
                    if !healthy {
                        checks.push(DoctorCheck::fail("failed", "3 rows".to_owned(), "run fix"));
                        checks.push(DoctorCheck {
                            name: "bare",
                            ok: false,
                            detail: None,
                            remedy: Some("bare remedy"),
                        });
                        check_json.push(serde_json::json!({"name": "failed", "ok": false, "detail": "3 rows", "remedy": "run fix"}));
                        check_json.push(serde_json::json!({"name": "bare", "ok": false, "detail": null, "remedy": "bare remedy"}));
                        check_lines.push_str("  failed [FAIL] 3 rows -> run fix\n  bare [FAIL]\n");
                        notes.push("failed: run fix".to_owned());
                        notes.push("bare: bare remedy".to_owned());
                    }

                    let repairs = with_repairs.then(|| {
                        vec![
                            RepairOutcome {
                                name: "sound",
                                result: Ok(2),
                            },
                            RepairOutcome {
                                name: "broken",
                                result: Err(anyhow::anyhow!("disk I/O error")),
                            },
                        ]
                    });
                    let mut repair_lines = "";
                    let mut repaired = serde_json::Value::Null;
                    if with_repairs {
                        repair_lines =
                            "Repaired sound: 2 row(s)\nRepair failed broken: disk I/O error\n";
                        repaired = serde_json::json!({"sound": 2, "broken": null});
                        notes.push("fix: broken: repair failed: disk I/O error".to_owned());
                    }

                    let (recovery, recovery_line, recovered, retryable) = match recovery_case {
                        "skipped" => (
                            ModelRecovery::Skipped,
                            String::new(),
                            serde_json::Value::Null,
                            serde_json::Value::Null,
                        ),
                        "succeeded" => (
                            ModelRecovery::Succeeded,
                            "Recovered model\n".to_owned(),
                            serde_json::json!(true),
                            serde_json::Value::Null,
                        ),
                        "retryable" => {
                            notes.push(format!("fix: model: recovery failed: {RETRYABLE}"));
                            (
                                ModelRecovery::Failed(RecallError::TempFailure(
                                    RETRYABLE.to_owned(),
                                )),
                                format!("Model recovery failed: {RETRYABLE}\n"),
                                serde_json::json!(false),
                                serde_json::json!(true),
                            )
                        }
                        "permanent" => {
                            notes.push(format!("fix: model: recovery failed: {PERMANENT}"));
                            (
                                ModelRecovery::Failed(RecallError::DataError(PERMANENT.to_owned())),
                                format!("Model recovery failed: {PERMANENT}\n"),
                                serde_json::json!(false),
                                serde_json::json!(false),
                            )
                        }
                        _ => unreachable!(),
                    };
                    let recovery_failed = matches!(recovery, ModelRecovery::Failed(_));

                    let out = render_doctor_report(verbose, &path, &checks, &repairs, &recovery);

                    let health = if healthy { "healthy" } else { "degraded" };
                    let db_line = if verbose {
                        "DB: /tmp/recall-test.db\n"
                    } else {
                        ""
                    };
                    assert_eq!(
                        out.markdown,
                        format!(
                            "Health: {health}\n{repair_lines}{recovery_line}{check_lines}{db_line}"
                        )
                    );
                    assert_eq!(out.notes, notes);
                    // The repairs always include the failed `broken` one.
                    assert_eq!(out.degraded, !healthy || with_repairs || recovery_failed);
                    assert_eq!(
                        out.data,
                        serde_json::json!({
                            "healthy": healthy,
                            "checks": check_json,
                            "repaired": repaired,
                            "model_recovered": recovered,
                            "model_recovery_retryable": retryable,
                        })
                    );
                }
            }
        }
    }
}

#[test]
fn doctor_flags_orphan_chunk_referencing_missing_session() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (1, 'ghost', 'x', 0)",
        [],
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, false, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [FAIL] 1 chunk(s) reference a missing session -> recall rebuild\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    {
                        "detail": "1 chunk(s) reference a missing session",
                        "name": "orphan_chunks",
                        "ok": false,
                        "remedy": "recall rebuild",
                    },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["orphan_chunks: recall rebuild"],
        })
    );
}

#[test]
fn doctor_fix_leaves_unrepairable_orphan_chunk_in_place() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    conn.execute(
        "INSERT INTO qa_chunks (id, session_id, content, timestamp) VALUES (1, 'ghost', 'x', 0)",
        [],
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [FAIL] 1 chunk(s) reference a missing session -> recall rebuild\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    {
                        "detail": "1 chunk(s) reference a missing session",
                        "name": "orphan_chunks",
                        "ok": false,
                        "remedy": "recall rebuild",
                    },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": true,
            "notes": ["orphan_chunks: recall rebuild"],
        })
    );
    assert_eq!(
        stored_rows(&db_path, "SELECT id, session_id FROM qa_chunks", |r| Ok((
            r.get::<_, i64>(0)?,
            r.get::<_, String>(1)?
        ))),
        vec![(1, "ghost".to_owned())]
    );
}

#[test]
fn doctor_fix_breakdown_lists_every_repair_even_when_clean() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn doctor_flags_model_that_fails_to_load() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        probe_result: || anyhow::bail!("isolated model probe failed"),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, false, &Some(db_path.clone()), model_host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] model not loadable: model probe failed -> recall model download\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "model not loadable: model probe failed",
                        "name": "model",
                        "ok": false,
                        "remedy": "recall model download",
                    },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["model: recall model download"],
        })
    );
}

#[test]
fn doctor_flags_model_dimension_mismatch() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        model: || Ok(Arc::new(WidthEmbedder(EMBEDDING_DIMS - 1))),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, false, &Some(db_path.clone()), model_host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] embedding width 767 != expected 768 -> recall model download\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "embedding width 767 != expected 768",
                        "name": "model",
                        "ok": false,
                        "remedy": "recall model download",
                    },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["model: recall model download"],
        })
    );
}

#[test]
fn doctor_reports_not_installed_model_as_info_keeping_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        artifacts: Cell::new(CachedArtifacts::NotInstalled),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, false, &Some(db_path.clone()), model_host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok] model not installed; search runs FTS-only\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "model not installed; search runs FTS-only",
                        "name": "model",
                        "ok": true,
                        "remedy": null,
                    },
                ],
                "healthy": true,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn doctor_fix_recovers_not_installed_model_to_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        artifacts: Cell::new(CachedArtifacts::NotInstalled),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, true, &Some(db_path.clone()), model_host, || {
                host.install();
                Ok(())
            })
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "Recovered model\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": true,
                "model_recovery_retryable": null,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn doctor_fix_recovers_unloadable_model_flipping_healthy() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        artifacts: Cell::new(CachedArtifacts::Corrupt),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, true, &Some(db_path.clone()), model_host, || {
                host.install();
                Ok(())
            })
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "Recovered model\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": true,
                "model_recovery_retryable": null,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn doctor_fix_surfaces_model_recovery_failure() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        artifacts: Cell::new(CachedArtifacts::NotInstalled),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, true, &Some(db_path.clone()), model_host, || {
                Err(RecallError::Internal("MLX backend unavailable".to_owned()))
            })
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "Model recovery failed: MLX backend unavailable\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok] model not installed; search runs FTS-only\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "model not installed; search runs FTS-only",
                        "name": "model",
                        "ok": true,
                        "remedy": null,
                    },
                ],
                "healthy": true,
                "model_recovered": false,
                "model_recovery_retryable": false,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": true,
            "notes": ["fix: model: recovery failed: MLX backend unavailable"],
        })
    );
}

#[test]
fn doctor_fix_failure_marks_transient_recovery_retryable() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        probe_result: || anyhow::bail!("isolated model probe failed"),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, true, &Some(db_path.clone()), model_host, || {
                Err(RecallError::TempFailure("download interrupted".to_owned()))
            })
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "Model recovery failed: download interrupted\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] model not loadable: model probe failed -> recall model download\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "model not loadable: model probe failed",
                        "name": "model",
                        "ok": false,
                        "remedy": "recall model download",
                    },
                ],
                "healthy": false,
                "model_recovered": false,
                "model_recovery_retryable": true,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": true,
            "notes": [
                "model: recall model download",
                "fix: model: recovery failed: download interrupted",
            ],
        })
    );
}

#[test]
fn doctor_fix_refetch_ok_but_probe_still_faults_reports_failed() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        probe_result: || anyhow::bail!("isolated model probe failed"),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, true, &Some(db_path.clone()), model_host, || Ok(()))
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "Repaired orphan_embeddings: 0 row(s)\n",
            "Model recovery failed: re-fetched model still fails its probe\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] model not loadable: model probe failed -> recall model download\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "model not loadable: model probe failed",
                        "name": "model",
                        "ok": false,
                        "remedy": "recall model download",
                    },
                ],
                "healthy": false,
                "model_recovered": false,
                "model_recovery_retryable": false,
                "repaired": { "orphan_embeddings": 0 },
            },
            "degraded": true,
            "notes": [
                "model: recall model download",
                "fix: model: recovery failed: re-fetched model still fails its probe",
            ],
        })
    );
}

#[test]
fn doctor_without_fix_leaves_model_recovered_null() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let failing = || Err(DegradedReason::ProbeFailed);
    let out = run_doctor_with(false, false, &Some(db_path), failing, no_recover).unwrap();

    assert_eq!(out.data["model_recovered"], serde_json::Value::Null);
    assert_eq!(out.data["healthy"], false, "an unloadable model is a fault");
}

#[test]
fn doctor_reports_unhealthy_and_does_not_create_db_when_absent() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("missing.db");
    let host = FakeHost::healthy();

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, false, &Some(db_path.clone()), model_host, no_recover)
        })
        .unwrap();

    assert_eq!(out.markdown, "");
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["no index: run `recall index`"],
        })
    );
    assert!(
        !db_path.exists(),
        "doctor must not create the database while diagnosing"
    );
    assert_eq!(host.probes.get(), 0, "a missing index loads no model");
}

#[test]
fn doctor_reports_unhealthy_on_unopenable_database() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    fs::write(&db_path, b"this is not a sqlite database").unwrap();

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, false, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [FAIL] cannot open database: file is not a database -> remove the database file and run `recall index`\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    {
                        "detail": "cannot open database: file is not a database",
                        "name": "integrity",
                        "ok": false,
                        "remedy": "remove the database file and run `recall index`",
                    },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["integrity: remove the database file and run `recall index`"],
        })
    );
}

#[test]
fn doctor_flags_model_that_fails_to_embed() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    let host = FakeHost {
        model: || Ok(Arc::new(embedder::MockEmbedder::failing_after(0))),
        ..FakeHost::healthy()
    };

    let out = host
        .serve(db_dir.path(), &db_dir.path().join("probe"), |model_host| {
            run_doctor_in(false, false, &Some(db_path.clone()), model_host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] model embed failed: embedding inference failed: mock failure -> recall model download\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "model embed failed: embedding inference failed: mock failure",
                        "name": "model",
                        "ok": false,
                        "remedy": "recall model download",
                    },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["model: recall model download"],
        })
    );
}

#[test]
fn run_db_check_converts_query_error_into_fail_verdict() {
    let check = run_db_check("integrity", || Err(anyhow::anyhow!("disk I/O error")));

    assert!(!check.ok);
    assert!(
        check
            .detail
            .as_deref()
            .unwrap()
            .contains("check could not run"),
        "got: {:?}",
        check.detail
    );
    assert_eq!(check.remedy, Some(CORRUPT_DB_REMEDY));
}

#[test]
fn check_orphan_embeddings_errors_when_table_missing() {
    let conn = Connection::open_in_memory().unwrap();
    assert!(check_orphan_embeddings(&conn).is_err());
}

#[test]
fn check_orphan_chunks_errors_when_table_missing() {
    let conn = Connection::open_in_memory().unwrap();
    assert!(check_orphan_chunks(&conn).is_err());
}

#[test]
fn verbose_doctor_of_a_healthy_index_prints_the_database_path() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(true, false, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        format!(
            concat!(
                "Health: healthy\n",
                "  integrity [ok]\n",
                "  orphan_embeddings [ok]\n",
                "  orphan_chunks [ok]\n",
                "  model [ok]\n",
                "DB: {}\n",
            ),
            db_path.display()
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": false,
            "notes": [],
        })
    );
}

#[test]
fn doctor_of_a_never_indexed_database_file_reports_unhealthy_with_a_failing_index_check() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    drop(rusqlite::Connection::open(&db_path).unwrap());

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, false, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    let detail = format!(
        "no index: database has never been indexed; recall cannot confirm this file ({}) was created by recall — verify the path if this is unexpected",
        db_path.display()
    );
    assert_eq!(
        out.markdown,
        format!("Health: degraded\n  index [FAIL] {detail} -> run `recall index`\n  model [ok]\n")
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": detail, "name": "index", "ok": false, "remedy": "run `recall index`" },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["index: run `recall index`"],
        })
    );
}

#[test]
fn doctor_of_an_index_from_an_older_version_reports_an_out_of_date_schema() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    {
        let conn = rusqlite::Connection::open(&db_path).unwrap();
        conn.execute_batch(
            "CREATE TABLE sessions (session_id TEXT PRIMARY KEY, session_type TEXT);",
        )
        .unwrap();
    }

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, false, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  schema [FAIL] database schema is out of date -> run `recall rebuild`\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    {
                        "detail": "database schema is out of date",
                        "name": "schema",
                        "ok": false,
                        "remedy": "run `recall rebuild`",
                    },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["schema: run `recall rebuild`"],
        })
    );
}

#[test]
fn doctor_on_an_existing_bare_sqlite_db_emits_an_index_fail_with_the_unconfirmed_origin_caveat() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = run_doctor_with(
        false,
        false,
        &Some(db_path.clone()),
        mock_loader,
        no_recover,
    )
    .unwrap();

    assert_eq!(out.data["healthy"], false);
    let index_check = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "index")
        .unwrap_or_else(|| panic!("expected an `index` check, got: {}", out.data));
    assert_eq!(index_check["ok"], false);
    let detail = index_check["detail"].as_str().unwrap();
    assert!(detail.contains("cannot confirm"), "got: {detail}");
    assert!(
        detail.contains(&db_path.display().to_string()),
        "caveat detail must name the resolved DB path to verify; got: {detail}"
    );
    assert_eq!(index_check["remedy"], "run `recall index`");
}

#[test]
fn doctor_fix_refuses_to_graft_recall_tables_into_a_foreign_empty_db() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = bare_empty_db(db_dir.path());

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    let detail = format!(
        "no index: database has never been indexed; recall cannot confirm this file ({}) was created by recall — verify the path if this is unexpected",
        db_path.display()
    );
    assert_eq!(
        out.markdown,
        format!("Health: degraded\n  index [FAIL] {detail} -> run `recall index`\n  model [ok]\n")
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": detail, "name": "index", "ok": false, "remedy": "run `recall index`" },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["index: run `recall index`"],
        })
    );
    assert_eq!(
        stored_rows(&db_path, "SELECT name FROM sqlite_master", |r| r
            .get::<_, String>(0)),
        Vec::<String>::new()
    );
}

#[test]
fn run_doctor_fix_reports_open_failure_on_unopenable_db() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    fs::write(&db_path, b"this is not a sqlite database").unwrap();

    let out = run_doctor_with(false, true, &Some(db_path), mock_loader, no_recover).unwrap();

    let integrity = out.data["checks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "integrity")
        .unwrap();
    assert_eq!(integrity["ok"], false);
    assert!(
        integrity["detail"]
            .as_str()
            .unwrap()
            .contains("cannot open database"),
        "got: {integrity}"
    );
}

#[test]
fn doctor_still_starts_the_probe_after_search_has_recorded_the_environment() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    seed_healthy_index(&open_or_create_db(&db_path).unwrap());
    let records_dir = db_dir.path().join("probe");
    let host = FakeHost::healthy();
    let first_search = host
        .serve(db_dir.path(), &records_dir, |model_host| {
            run_search_in(
                search_command("healthy chunk"),
                &Some(db_path.clone()),
                model_host,
            )
        })
        .unwrap();
    let second_search = host
        .serve(db_dir.path(), &records_dir, |model_host| {
            run_search_in(
                search_command("healthy chunk"),
                &Some(db_path.clone()),
                model_host,
            )
        })
        .unwrap();
    assert!(!first_search.degraded);
    assert!(!second_search.degraded);
    assert_eq!(host.probes.get(), 1, "the second search skipped the probe");

    let out = host
        .serve(db_dir.path(), &records_dir, |model_host| {
            run_doctor_in(false, false, &Some(db_path.clone()), model_host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": false,
            "notes": [],
        })
    );
    assert_eq!(host.probes.get(), 2, "doctor runs the probe every time");
}

/// The number of sessions the index at `db_path` holds.
fn stored_session_count(db_path: &Path) -> i64 {
    let (conn, _) = db::open_db_readonly(db_path).unwrap();
    conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))
        .unwrap()
}

/// The rows `sql` reads from the index at `db_path`, each built by `row`.
fn stored_rows<T>(
    db_path: &Path,
    sql: &str,
    row: impl FnMut(&rusqlite::Row<'_>) -> rusqlite::Result<T>,
) -> Vec<T> {
    let (conn, _) = db::open_db_readonly(db_path).unwrap();
    let mut stmt = conn.prepare(sql).unwrap();
    stmt.query_map([], row)
        .unwrap()
        .map(Result::unwrap)
        .collect()
}

#[test]
fn doctor_fix_of_an_index_built_for_another_model_reports_the_mismatch_once_and_keeps_its_model_identity()
 {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    conn.execute(
        "UPDATE embedding_identity SET fingerprint = 'other-model'",
        [],
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] index embedding identity mismatch; use a separate --db-path for this model or select the matching --model-manifest -> select the matching --model-manifest and index, or build this model in a separate --db-path\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "index embedding identity mismatch; use a separate --db-path for this model or select the matching --model-manifest",
                        "name": "model",
                        "ok": false,
                        "remedy": "select the matching --model-manifest and index, or build this model in a separate --db-path",
                    },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["model: select the matching --model-manifest and index, or build this model in a separate --db-path"],
        })
    );
    assert_eq!(
        stored_rows(
            &db_path,
            "SELECT fingerprint FROM embedding_identity",
            |r| r.get::<_, String>(0)
        ),
        vec!["other-model".to_owned()]
    );
}

#[test]
fn doctor_fix_of_an_index_with_an_outdated_schema_reports_it_and_keeps_the_cached_sessions() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    conn.execute_batch(
        "DROP TABLE messages_vocab;
         DROP TABLE messages;
         CREATE VIRTUAL TABLE messages USING fts5(session_id UNINDEXED, role, text, tokenize='unicode61');
         CREATE VIRTUAL TABLE messages_vocab USING fts5vocab(messages, row);",
    )
    .unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  schema [FAIL] database schema is out of date -> run `recall rebuild`\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    {
                        "detail": "database schema is out of date",
                        "name": "schema",
                        "ok": false,
                        "remedy": "run `recall rebuild`",
                    },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": ["schema: run `recall rebuild`"],
        })
    );
    assert_eq!(stored_session_count(&db_path), 1);
}

#[test]
fn doctor_fix_that_cannot_write_the_index_is_degraded_and_says_why() {
    use std::os::unix::fs::PermissionsExt;

    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    drop(conn);
    fs::set_permissions(&db_path, fs::Permissions::from_mode(0o444)).unwrap();

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: healthy\n",
            "Repair failed orphan_embeddings: attempt to write a readonly database\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [ok]\n",
            "  orphan_chunks [ok]\n",
            "  model [ok]\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    { "detail": null, "name": "model", "ok": true, "remedy": null },
                ],
                "healthy": true,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": { "orphan_embeddings": null },
            },
            "degraded": true,
            "notes": ["fix: orphan_embeddings: repair failed: attempt to write a readonly database"],
        })
    );
}

#[test]
fn doctor_fix_leaves_an_orphan_embedding_in_a_legacy_index_kept_for_rollback() {
    let db_dir = tempfile::TempDir::new().unwrap();
    let db_path = db_dir.path().join("recall.db");
    let conn = open_or_create_db(&db_path).unwrap();
    seed_healthy_index(&conn);
    let orphan = embedder::MockEmbedder::deterministic_vector("orphan");
    conn.execute(
        "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 999, 0)",
        rusqlite::params![embedder::f32_as_bytes(&orphan)],
    )
    .unwrap();
    conn.execute("DROP TABLE embedding_identity", []).unwrap();
    drop(conn);

    let out = FakeHost::healthy()
        .serve(db_dir.path(), &db_dir.path().join("probe"), |host| {
            run_doctor_in(false, true, &Some(db_path.clone()), host, no_recover)
        })
        .unwrap();

    assert_eq!(
        out.markdown,
        concat!(
            "Health: degraded\n",
            "  integrity [ok]\n",
            "  orphan_embeddings [FAIL] 1 embedding(s) reference a missing chunk -> recall doctor --fix\n",
            "  orphan_chunks [ok]\n",
            "  model [FAIL] legacy vector index has no model identity; keep it for rollback and build Gemma in a separate --db-path -> select the matching --model-manifest and index, or build this model in a separate --db-path\n",
        )
    );
    assert_eq!(
        printed_envelope(&out),
        serde_json::json!({
            "data": {
                "checks": [
                    { "detail": null, "name": "integrity", "ok": true, "remedy": null },
                    {
                        "detail": "1 embedding(s) reference a missing chunk",
                        "name": "orphan_embeddings",
                        "ok": false,
                        "remedy": "recall doctor --fix",
                    },
                    { "detail": null, "name": "orphan_chunks", "ok": true, "remedy": null },
                    {
                        "detail": "legacy vector index has no model identity; keep it for rollback and build Gemma in a separate --db-path",
                        "name": "model",
                        "ok": false,
                        "remedy": "select the matching --model-manifest and index, or build this model in a separate --db-path",
                    },
                ],
                "healthy": false,
                "model_recovered": null,
                "model_recovery_retryable": null,
                "repaired": null,
            },
            "degraded": true,
            "notes": [
                "orphan_embeddings: recall doctor --fix",
                "model: select the matching --model-manifest and index, or build this model in a separate --db-path",
            ],
        })
    );
}
