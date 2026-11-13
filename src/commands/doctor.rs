use std::fmt::{Display, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use anyhow::Result;
use rusqlite::Connection;

use crate::commands::build_markdown;
use crate::db;
use crate::db::fix_orphan_embeddings;
use crate::embedding::{EMBEDDING_DIMS, Embed};
use crate::envelope::CommandOutput;
use crate::error::{RecallError, download_error};
use crate::model::{self, DegradedReason, download_and_verify_model};
use crate::runtime::info as cli_info;
use crate::{resolve_db_path, try_load_embedder_cached_with};

struct DoctorCheck {
    name: &'static str,
    ok: bool,
    detail: Option<String>,
    remedy: Option<&'static str>,
}

impl DoctorCheck {
    fn pass(name: &'static str) -> Self {
        Self {
            name,
            ok: true,
            detail: None,
            remedy: None,
        }
    }

    fn info(name: &'static str, detail: String) -> Self {
        Self {
            name,
            ok: true,
            detail: Some(detail),
            remedy: None,
        }
    }

    fn fail(name: &'static str, detail: String, remedy: &'static str) -> Self {
        Self {
            name,
            ok: false,
            detail: Some(detail),
            remedy: Some(remedy),
        }
    }

    fn is_fully_usable(&self) -> bool {
        self.ok && self.detail.is_none()
    }
}

const CORRUPT_DB_REMEDY: &str = "remove the database file and run `recall index`";
const MODEL_INDEX_REMEDY: &str =
    "select the matching --model-manifest and index, or build this model in a separate --db-path";

fn run_db_check(name: &'static str, check: impl FnOnce() -> Result<DoctorCheck>) -> DoctorCheck {
    check().unwrap_or_else(|e| {
        DoctorCheck::fail(name, format!("check could not run: {e}"), CORRUPT_DB_REMEDY)
    })
}

fn db_checks(conn: &Connection) -> Vec<DoctorCheck> {
    vec![
        run_db_check("integrity", || check_integrity(conn)),
        run_db_check("orphan_embeddings", || check_orphan_embeddings(conn)),
        run_db_check("orphan_chunks", || check_orphan_chunks(conn)),
    ]
}

fn open_failure_checks(e: impl Display) -> Vec<DoctorCheck> {
    vec![DoctorCheck::fail(
        "integrity",
        format!("cannot open database: {e}"),
        CORRUPT_DB_REMEDY,
    )]
}

fn empty_index_check(path: &Path) -> DoctorCheck {
    DoctorCheck::fail(
        "index",
        format!(
            "no index: database has never been indexed; recall cannot confirm this file ({}) was created by recall — verify the path if this is unexpected",
            path.display()
        ),
        "run `recall index`",
    )
}

fn check_integrity(conn: &Connection) -> Result<DoctorCheck> {
    let result: String = conn.query_row("PRAGMA quick_check", [], |r| r.get(0))?;
    Ok(if result == "ok" {
        DoctorCheck::pass("integrity")
    } else {
        DoctorCheck::fail(
            "integrity",
            format!("database corrupt: {result}"),
            CORRUPT_DB_REMEDY,
        )
    })
}

fn check_orphan_embeddings(conn: &Connection) -> Result<DoctorCheck> {
    let orphans: i64 = conn.query_row(
        "SELECT COUNT(*) FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM qa_chunks)",
        [],
        |r| r.get(0),
    )?;
    Ok(if orphans == 0 {
        DoctorCheck::pass("orphan_embeddings")
    } else {
        DoctorCheck::fail(
            "orphan_embeddings",
            format!("{orphans} embedding(s) reference a missing chunk"),
            "recall doctor --fix",
        )
    })
}

struct RepairOutcome {
    name: &'static str,
    result: Result<usize>,
}

fn check_orphan_chunks(conn: &Connection) -> Result<DoctorCheck> {
    let orphans: i64 = conn.query_row(
        "SELECT COUNT(*) FROM qa_chunks WHERE session_id NOT IN (SELECT session_id FROM sessions)",
        [],
        |r| r.get(0),
    )?;
    Ok(if orphans == 0 {
        DoctorCheck::pass("orphan_chunks")
    } else {
        DoctorCheck::fail(
            "orphan_chunks",
            format!("{orphans} chunk(s) reference a missing session"),
            "recall rebuild",
        )
    })
}

fn check_model<F>(load_embedder: &F) -> DoctorCheck
where
    F: Fn() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    const REMEDY: &str = "recall model download";
    let embedder = match load_embedder() {
        Ok(e) => e,
        Err(reason @ DegradedReason::NotInstalled) => {
            return DoctorCheck::info("model", format!("{reason}; search runs FTS-only"));
        }
        Err(reason) => {
            return DoctorCheck::fail("model", format!("model not loadable: {reason}"), REMEDY);
        }
    };
    match embedder.embed_query("recall doctor probe") {
        Ok(v) if v.len() == EMBEDDING_DIMS => DoctorCheck::pass("model"),
        Ok(v) => DoctorCheck::fail(
            "model",
            format!("embedding width {} != expected {EMBEDDING_DIMS}", v.len()),
            REMEDY,
        ),
        Err(e) => DoctorCheck::fail("model", format!("model embed failed: {e}"), REMEDY),
    }
}

enum ModelRecovery {
    Skipped,
    Succeeded,
    Failed(RecallError),
}

fn check_model_with_optional_recovery<F, R>(
    recover: bool,
    load_embedder: F,
    recover_model: R,
) -> (DoctorCheck, ModelRecovery)
where
    F: Fn() -> Result<Arc<dyn Embed>, DegradedReason>,
    R: FnOnce() -> Result<(), RecallError>,
{
    let initial = check_model(&load_embedder);
    if !recover || initial.is_fully_usable() {
        return (initial, ModelRecovery::Skipped);
    }
    match recover_model() {
        Ok(()) => {
            let rechecked = check_model(&load_embedder);
            if rechecked.is_fully_usable() {
                (rechecked, ModelRecovery::Succeeded)
            } else {
                (
                    rechecked,
                    ModelRecovery::Failed(RecallError::Internal(
                        "re-fetched model still fails its probe".to_owned(),
                    )),
                )
            }
        }
        Err(e) => (initial, ModelRecovery::Failed(e)),
    }
}

pub(crate) fn run_doctor(
    verbose: bool,
    fix: bool,
    db_path: &Option<PathBuf>,
) -> Result<CommandOutput> {
    run_doctor_in(verbose, fix, db_path, &model::HOST, || {
        download_and_verify_model().map_err(|e| download_error(&e))
    })
}

/// Diagnoses the index at `db_path` against `host`, always probing the model (DR 0013). Under
/// `--fix`, `recover_model` stands for the model download; it runs only when the model is missing
/// or fails its check, and the index's identity check did not fail.
fn run_doctor_in<R>(
    verbose: bool,
    fix: bool,
    db_path: &Option<PathBuf>,
    host: &model::ModelHost<'_>,
    recover_model: R,
) -> Result<CommandOutput>
where
    R: FnOnce() -> Result<(), RecallError>,
{
    let load_embedder = || try_load_embedder_cached_with(|| host.load_probing());
    let path = resolve_db_path(db_path)?;

    if !path.exists() {
        cli_info(&format!("database not found at {}", path.display()));
        cli_info("run `recall index` to create the index.");
        return Ok(CommandOutput::with_notes(
            String::new(),
            serde_json::json!({
                "healthy": false,
                "checks": [],
                "repaired": null,
                "model_recovered": null,
                "model_recovery_retryable": null,
            }),
            true,
            vec!["no index: run `recall index`".to_owned()],
        ));
    }

    let mut repairs: Option<Vec<RepairOutcome>> = None;
    let mut wal_note: Option<String> = None;
    let (mut checks, identity_error) = match db::open_db_readonly(&path) {
        Ok((conn, tier)) => {
            wal_note = db::stale_wal_note(&path, tier);
            let identity = model::check_index(&conn, model::selected());
            let checks = match db::schema_state(&conn) {
                Ok(db::SchemaState::Current) => {
                    // An index built for another model, or a legacy one kept for rollback, stays
                    // unwritten.
                    if fix && identity.is_ok() {
                        repairs = Some(vec![RepairOutcome {
                            name: "orphan_embeddings",
                            result: db::open_existing_db(&path)
                                .and_then(|writer| fix_orphan_embeddings(&writer)),
                        }]);
                    }
                    // Each check reads in a new transaction and sees the repair. An index opened
                    // immutable sits in a read-only directory, where the repair fails instead.
                    db_checks(&conn)
                }
                Ok(db::SchemaState::Stale) => vec![DoctorCheck::fail(
                    "schema",
                    "database schema is out of date".to_owned(),
                    "run `recall rebuild`",
                )],
                Ok(db::SchemaState::Empty) => vec![empty_index_check(&path)],
                Err(e) => open_failure_checks(e),
            };
            (checks, identity.err())
        }
        Err(e) => (open_failure_checks(e), None),
    };
    // Installing the selected model cannot make an index built for another model usable.
    let recover = fix && identity_error.is_none();
    let (mut model_check, model_recovery) =
        check_model_with_optional_recovery(recover, load_embedder, recover_model);
    if let Some(error) = identity_error {
        model_check = DoctorCheck::fail("model", error.to_string(), MODEL_INDEX_REMEDY);
    }
    checks.push(model_check);

    let mut output = render_doctor_report(verbose, &path, &checks, &repairs, &model_recovery);
    if let Some(note) = wal_note {
        output.degraded = true;
        output.notes.push(note);
    }
    Ok(output)
}

fn render_doctor_report(
    verbose: bool,
    path: &Path,
    checks: &[DoctorCheck],
    repairs: &Option<Vec<RepairOutcome>>,
    model_recovery: &ModelRecovery,
) -> CommandOutput {
    let healthy = checks.iter().all(|c| c.ok);
    let notes = doctor_notes(checks, repairs, model_recovery);
    let markdown = doctor_markdown(verbose, path, healthy, checks, repairs, model_recovery);
    let data = doctor_data(healthy, checks, repairs, model_recovery);
    let repair_failed = repairs
        .iter()
        .flatten()
        .any(|outcome| outcome.result.is_err());
    let degraded = !healthy || repair_failed || matches!(model_recovery, ModelRecovery::Failed(_));
    CommandOutput::with_notes(markdown, data, degraded, notes)
}

fn doctor_notes(
    checks: &[DoctorCheck],
    repairs: &Option<Vec<RepairOutcome>>,
    model_recovery: &ModelRecovery,
) -> Vec<String> {
    let mut notes: Vec<String> = checks
        .iter()
        .filter_map(|c| c.remedy.map(|r| format!("{}: {}", c.name, r)))
        .collect();
    if let Some(outcomes) = repairs {
        for o in outcomes {
            if let Err(e) = &o.result {
                notes.push(format!("fix: {}: repair failed: {e}", o.name));
            }
        }
    }
    if let ModelRecovery::Failed(e) = model_recovery {
        notes.push(format!("fix: model: recovery failed: {e}"));
    }
    notes
}

fn doctor_markdown(
    verbose: bool,
    path: &Path,
    healthy: bool,
    checks: &[DoctorCheck],
    repairs: &Option<Vec<RepairOutcome>>,
    model_recovery: &ModelRecovery,
) -> String {
    build_markdown(|buf| {
        writeln!(
            buf,
            "Health: {}",
            if healthy { "healthy" } else { "degraded" }
        )?;
        if let Some(outcomes) = repairs {
            for o in outcomes {
                match &o.result {
                    Ok(n) => {
                        writeln!(buf, "Repaired {}: {n} row(s)", o.name)?;
                    }
                    Err(e) => {
                        writeln!(buf, "Repair failed {}: {e}", o.name)?;
                    }
                }
            }
        }
        match model_recovery {
            ModelRecovery::Succeeded => {
                writeln!(buf, "Recovered model")?;
            }
            ModelRecovery::Failed(e) => {
                writeln!(buf, "Model recovery failed: {e}")?;
            }
            ModelRecovery::Skipped => {}
        }
        for c in checks {
            let status = if c.ok { "ok" } else { "FAIL" };
            match (&c.detail, c.remedy) {
                (Some(detail), Some(remedy)) => {
                    writeln!(buf, "  {} [{status}] {detail} -> {remedy}", c.name)?;
                }
                (Some(detail), None) => {
                    writeln!(buf, "  {} [{status}] {detail}", c.name)?;
                }
                (None, _) => {
                    writeln!(buf, "  {} [{status}]", c.name)?;
                }
            }
        }
        if verbose {
            writeln!(buf, "DB: {}", path.display())?;
        }
        Ok(())
    })
}

fn doctor_data(
    healthy: bool,
    checks: &[DoctorCheck],
    repairs: &Option<Vec<RepairOutcome>>,
    model_recovery: &ModelRecovery,
) -> serde_json::Value {
    let checks_json: Vec<serde_json::Value> = checks
        .iter()
        .map(|c| {
            serde_json::json!({
                "name": c.name,
                "ok": c.ok,
                "detail": c.detail,
                "remedy": c.remedy,
            })
        })
        .collect();
    let repaired = match repairs {
        Some(outcomes) => {
            let map: serde_json::Map<String, serde_json::Value> = outcomes
                .iter()
                .map(|o| {
                    let v = match &o.result {
                        Ok(n) => serde_json::json!(n),
                        Err(_) => serde_json::Value::Null,
                    };
                    (o.name.to_owned(), v)
                })
                .collect();
            serde_json::Value::Object(map)
        }
        None => serde_json::Value::Null,
    };
    let (model_recovered, model_recovery_retryable) = match model_recovery {
        ModelRecovery::Skipped => (serde_json::Value::Null, serde_json::Value::Null),
        ModelRecovery::Succeeded => (serde_json::Value::Bool(true), serde_json::Value::Null),
        ModelRecovery::Failed(e) => (
            serde_json::Value::Bool(false),
            serde_json::Value::Bool(e.retryable()),
        ),
    };
    let data = serde_json::json!({
        "healthy": healthy,
        "checks": checks_json,
        "repaired": repaired,
        "model_recovered": model_recovered,
        "model_recovery_retryable": model_recovery_retryable,
    });
    data
}

#[cfg(test)]
mod tests;
