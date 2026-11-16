use std::fmt::Write;
use std::path::PathBuf;

use anyhow::Result;

use crate::commands::{ReadIndex, build_markdown};
use crate::envelope::{CommandOutput, Note};
use crate::model::{self, cached_artifacts};
use crate::runtime::info as cli_info;
use crate::{db, resolve_db_path};

pub(crate) fn run_status(verbose: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    run_status_with(verbose, db_path, || {
        cached_artifacts().map(|opt| opt.is_some()).unwrap_or(false)
    })
}

/// Takes only whether the model is ready, so status never starts the probe or loads the model.
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

    let (sessions, chunks, embedded) = match index.schema()? {
        db::SchemaState::Empty => (0, 0, 0),
        _ => {
            let sessions: i64 =
                conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
            let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))?;
            let embedded: i64 =
                conn.query_row("SELECT COUNT(DISTINCT chunk_id) FROM vec_chunks", [], |r| {
                    r.get(0)
                })?;
            (sessions, chunks, embedded)
        }
    };

    let markdown = build_markdown(|buf| {
        writeln!(buf, "Sessions: {sessions}")?;
        writeln!(buf, "QA chunks: {chunks}")?;
        writeln!(buf, "Embedded: {embedded}/{chunks}")?;
        writeln!(
            buf,
            "Model: {}",
            if model_ok {
                "ready"
            } else {
                "not installed (run `recall model download`)"
            }
        )?;
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

    // The stale-snapshot note comes before the identity note.
    let mut out = index.finish(CommandOutput::ok(markdown, data));
    if let Err(error) = model::check_index(conn, model::selected()) {
        out.degraded = true;
        out.notes.push(Note::new(format!(
            "semantic search unavailable: {error}; results are text-only"
        )));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
