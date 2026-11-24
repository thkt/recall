use std::fmt::Write;
use std::path::PathBuf;

use anyhow::Result;

use crate::commands::{ReadIndex, build_markdown, identity_mismatch_note, resolve_db_path};
use crate::envelope::CommandOutput;
use crate::model::{self, ModelHost};
use crate::runtime::info as cli_info;

pub(crate) fn run_status(verbose: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    run_status_in(verbose, db_path, &model::HOST)
}

pub(crate) fn run_status_in(
    verbose: bool,
    db_path: &Option<PathBuf>,
    host: &ModelHost<'_>,
) -> Result<CommandOutput> {
    run_status_with(verbose, db_path, || host.artifacts_ready())
}

pub(crate) fn run_status_with(
    verbose: bool,
    db_path: &Option<PathBuf>,
    model_ready: impl FnOnce() -> bool,
) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;

    let model_ok = model_ready();

    if !path.exists() {
        cli_info(&format!("database not found at {}", path.display()));
        cli_info("run `recall index` to create the index.");
        return Ok(CommandOutput::ok(
            String::new(),
            serde_json::json!({
                "sessions": 0,
                "qa_chunks": 0,
                "embedded": 0,
                "model_ready": model_ok,
            }),
        ));
    }

    let index = ReadIndex::open(&path)?;
    let conn = &index.conn;

    let (sessions, chunks, embedded) = if index.is_never_indexed()? {
        (0, 0, 0)
    } else {
        let sessions: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
        let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))?;
        let embedded: i64 =
            conn.query_row("SELECT COUNT(DISTINCT chunk_id) FROM vec_chunks", [], |r| {
                r.get(0)
            })?;
        (sessions, chunks, embedded)
    };

    let markdown = build_markdown(|buf| {
        writeln!(buf, "Sessions: {sessions}")?;
        writeln!(buf, "QA chunks: {chunks}")?;
        writeln!(buf, "Embedded: {embedded}/{chunks}")?;
        if model_ok {
            writeln!(buf, "Model: ready")?;
        } else {
            writeln!(
                buf,
                "Model: not installed (run `{}`)",
                model::DOWNLOAD_COMMAND
            )?;
        }
        if verbose {
            writeln!(buf, "DB: {}", path.display())?;
        }
        Ok(())
    });
    let data = serde_json::json!({
        "sessions": sessions,
        "qa_chunks": chunks,
        "embedded": embedded,
        "model_ready": model_ok,
    });

    let mut out = index.snapshot.finish(CommandOutput::ok(markdown, data));
    if let Some(note) = identity_mismatch_note(conn) {
        out.degraded = true;
        out.notes.push(note);
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
