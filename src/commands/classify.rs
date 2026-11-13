use std::collections::BTreeMap;
use std::fs::{File, Permissions, metadata};
use std::io;
use std::path::{Path, PathBuf};
use std::time::SystemTime;

use anyhow::Result;
use rusqlite::{Connection, ErrorCode, TransactionBehavior};

use crate::envelope::CommandOutput;
use crate::error::RecallError;
use crate::parser::Source;
use crate::{ansi, classify, db, open_or_create_db, parser, resolve_db_path};

pub(crate) struct ClassifyOutcome {
    session_id: String,
    session_type: classify::SessionType,
    excerpt: String,
}

pub(crate) fn read_classification_origin(
    file: File,
    path: &str,
    source: &str,
    session_id: &str,
) -> Result<bool, &'static str> {
    let source = Source::from_db(source).ok_or("unknown source")?;
    let parsed = parser::parse_classification(file, Path::new(path), source)
        .map_err(|_| "read failure")?
        .ok_or("invalid format")?;
    if parsed.session_id != session_id || parsed.origin.identity_conflict {
        return Err("session ID mismatch");
    }
    if !parsed.diagnostics.is_empty() || !parsed.origin.has_records {
        return Err("invalid format");
    }
    Ok(parsed.origin.automated)
}

#[derive(Debug, PartialEq, Eq)]
pub(crate) struct ClassificationFileState {
    stamp: Result<(u64, SystemTime, Permissions), &'static str>,
    access: Result<(), &'static str>,
}

fn classification_file_state(path: &str) -> ClassificationFileState {
    open_classification_file(path).0
}

pub(crate) fn open_classification_file(
    path: &str,
) -> (ClassificationFileState, Result<File, &'static str>) {
    let stamp = (|| {
        let meta = metadata(path).map_err(|e| {
            if e.kind() == io::ErrorKind::NotFound {
                "missing file"
            } else {
                "read failure"
            }
        })?;
        if !meta.is_file() {
            return Err("not a regular file");
        }
        Ok((
            meta.len(),
            meta.modified().map_err(|_| "read failure")?,
            meta.permissions(),
        ))
    })();
    let file = match &stamp {
        Ok(_) => File::open(path).map_err(|_| "read failure"),
        Err(reason) => Err(*reason),
    };
    let access = file.as_ref().map(|_| ()).map_err(|reason| *reason);
    (ClassificationFileState { stamp, access }, file)
}

#[derive(Clone, Copy)]
pub(crate) enum ClassifyPhase {
    Snapshot,
    Prepared,
    WriterWaiting,
    WriterAcquired,
    WriterReleased,
    Updated,
    Retry,
}

pub(crate) fn reclassify_sessions(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
) -> Result<Vec<ClassifyOutcome>> {
    reclassify_sessions_with(conn, all, dry_run, read_classification_origin)
}

pub(crate) fn reclassify_sessions_with(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
    read_origin: impl FnMut(File, &str, &str, &str) -> Result<bool, &'static str>,
) -> Result<Vec<ClassifyOutcome>> {
    reclassify_sessions_observed(conn, all, dry_run, read_origin, |_| {})
}

type ClassifyRow = (String, String, String, Option<String>, Option<String>);

fn read_classify_rows(conn: &mut Connection, all: bool, dry_run: bool) -> Result<Vec<ClassifyRow>> {
    let tx = conn.transaction()?;
    let rows = {
        let mut stmt = tx.prepare(
        "WITH first_turn AS (SELECT session_id, text, MIN(rowid) FROM messages WHERE role = 'user' GROUP BY session_id) \
         SELECT s.session_id, s.source, s.file_path, s.session_type, m.text \
         FROM sessions s LEFT JOIN first_turn m ON m.session_id = s.session_id \
         WHERE ?1 OR s.session_type IS NULL ORDER BY s.session_id",
    ).map_err(|error| {
        if dry_run {
            anyhow::Error::from(RecallError::DatabaseRead(format!(
                "cannot read database structure for classify --dry-run: {error}; \
                 run `recall rebuild` to restore the required schema, then retry"
            )))
        } else {
            error.into()
        }
    })?;
        let rows = stmt.query_map([all], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<String>>(1)?.unwrap_or_default(),
                row.get::<_, Option<String>>(2)?.unwrap_or_default(),
                row.get::<_, Option<String>>(3)?,
                row.get::<_, Option<String>>(4)?,
            ))
        })?;

        rows.collect::<rusqlite::Result<Vec<_>>>()?
    };
    tx.commit()?;
    Ok(rows)
}

struct PreparedClassification {
    outcomes: Vec<ClassifyOutcome>,
    files: Vec<(String, ClassificationFileState)>,
    unavailable: BTreeMap<&'static str, usize>,
    changed: bool,
}

fn prepare_classification(
    rows: Vec<ClassifyRow>,
    dry_run: bool,
    read_origin: &mut impl FnMut(File, &str, &str, &str) -> Result<bool, &'static str>,
) -> PreparedClassification {
    let mut outcomes = Vec::new();
    let mut files = Vec::new();
    let mut unavailable = BTreeMap::<&str, usize>::new();
    let mut changed = false;
    for (session_id, source, path, previous, first_turn) in rows {
        let (before, file) = open_classification_file(&path);
        let (origin, after) = match file {
            Ok(file) => {
                let origin = read_origin(file, &path, &source, &session_id);
                let after = classification_file_state(&path);
                if before != after {
                    changed = true;
                }
                (origin, after)
            }
            Err(reason) => (Err(reason), before),
        };
        files.push((path, after));
        let origin = match origin {
            Ok(origin) => origin,
            Err(reason) => {
                *unavailable.entry(reason).or_default() += 1;
                if previous.is_some() {
                    continue;
                }
                false
            }
        };
        let first_turn = first_turn.as_deref().unwrap_or("");
        outcomes.push(ClassifyOutcome {
            session_id,
            session_type: classify::classify_session(first_turn, origin),
            excerpt: if dry_run {
                ansi::strip_control_chars(first_turn)
                    .trim_start()
                    .chars()
                    .take(80)
                    .map(|c| if c == '\n' { ' ' } else { c })
                    .collect()
            } else {
                String::new()
            },
        });
    }
    PreparedClassification {
        outcomes,
        files,
        unavailable,
        changed,
    }
}

enum ApplyAttempt {
    Applied,
    Retry,
}

fn apply_classification(
    conn: &mut Connection,
    version: i64,
    prepared: &PreparedClassification,
    observe: &mut impl FnMut(ClassifyPhase),
) -> Result<ApplyAttempt> {
    observe(ClassifyPhase::WriterWaiting);
    let tx = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
        Ok(tx) => tx,
        Err(error) if error.sqlite_error_code() == Some(ErrorCode::DatabaseBusy) => {
            return Ok(ApplyAttempt::Retry);
        }
        Err(error) => return Err(error.into()),
    };
    observe(ClassifyPhase::WriterAcquired);
    let current: i64 = tx.query_row("PRAGMA main.data_version", [], |r| r.get(0))?;
    if current != version
        || prepared
            .files
            .iter()
            .any(|(path, state)| classification_file_state(path) != *state)
    {
        tx.rollback()?;
        observe(ClassifyPhase::WriterReleased);
        return Ok(ApplyAttempt::Retry);
    }
    {
        let mut update = tx.prepare("UPDATE sessions SET session_type = ?2 WHERE session_id = ?1 AND session_type IS NOT ?2")?;
        for o in &prepared.outcomes {
            update.execute(rusqlite::params![o.session_id, o.session_type.as_str()])?;
            observe(ClassifyPhase::Updated);
        }
    }
    tx.commit()?;
    observe(ClassifyPhase::WriterReleased);
    Ok(ApplyAttempt::Applied)
}

pub(crate) fn reclassify_sessions_observed(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
    mut read_origin: impl FnMut(File, &str, &str, &str) -> Result<bool, &'static str>,
    mut observe: impl FnMut(ClassifyPhase),
) -> Result<Vec<ClassifyOutcome>> {
    for _ in 0..3 {
        let version: i64 = conn.query_row("PRAGMA main.data_version", [], |r| r.get(0))?;
        let rows = read_classify_rows(conn, all, dry_run)?;
        observe(ClassifyPhase::Snapshot);

        let prepared = prepare_classification(rows, dry_run, &mut read_origin);
        observe(ClassifyPhase::Prepared);
        if prepared.changed {
            observe(ClassifyPhase::Retry);
            continue;
        }
        if !dry_run
            && matches!(
                apply_classification(conn, version, &prepared, &mut observe)?,
                ApplyAttempt::Retry
            )
        {
            observe(ClassifyPhase::Retry);
            continue;
        }
        for (reason, count) in prepared.unavailable {
            eprintln!(
                "warning: provenance unavailable ({reason}): {count} session(s); existing labels retained, unclassified rows use first-turn fallback"
            );
        }
        return Ok(prepared.outcomes);
    }
    Err(RecallError::TempFailure(
        "classification inputs kept changing or the database writer remained busy after 3 attempts; no classification changes applied; rerun `recall classify` with the same options".to_owned()
    ).into())
}

pub(crate) fn run_classify(
    all: bool,
    dry_run: bool,
    db_path: &Option<PathBuf>,
) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;
    let outcomes = if dry_run {
        if path.try_exists()? {
            let (mut conn, tier) = db::open_db_readonly(&path)?;
            if let Some(note) = db::stale_wal_note(&path, tier) {
                return Err(RecallError::DatabaseRead(note).into());
            }
            reclassify_sessions(&mut conn, all, true)?
        } else {
            Vec::new()
        }
    } else {
        let mut conn = open_or_create_db(&path)?;
        reclassify_sessions(&mut conn, all, false)?
    };
    let automated = outcomes
        .iter()
        .filter(|o| o.session_type == classify::SessionType::Automated)
        .count();
    let interactive = outcomes.len() - automated;

    let mut markdown = String::new();
    if dry_run {
        for o in &outcomes {
            markdown.push_str(&format!(
                "{} [{}] {}\n",
                ansi::strip_control_chars(&o.session_id),
                o.session_type.as_str(),
                o.excerpt
            ));
        }
        markdown.push_str(&format!(
            "dry-run: {} session(s) would be classified ({automated} automated, {interactive} interactive)",
            outcomes.len()
        ));
    } else {
        markdown.push_str(&format!(
            "classified {} session(s): {automated} automated, {interactive} interactive",
            outcomes.len()
        ));
    }
    let data = serde_json::json!({
        "classified": outcomes.len(),
        "automated": automated,
        "interactive": interactive,
        "dry_run": dry_run,
    });
    Ok(CommandOutput::ok(markdown, data))
}

#[cfg(test)]
mod tests;
