use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
#[cfg(test)]
use rusqlite::Connection;
use tracing::debug;

use crate::db::fix_orphan_embeddings;
use crate::embedding::Embed;
use crate::envelope::CommandOutput;
use crate::model::{self, DegradedReason, ModelHost, degrade_with_warn};
use crate::runtime::{done, home_dir, info as cli_info, warning};
use crate::{ansi, embedder, index_observer, indexer};
use crate::{
    open_or_create_db, resolve_db_path, search_degraded_note,
    try_load_embedder_cached_with_reporter,
};

fn preserved_embedded_note(count: usize) -> Option<String> {
    (count > 0).then(|| {
        format!(
            "preserved embeddings for {count} session(s) the model could not rebuild this run — semantic search for them stays intact; re-run `recall index` once the model can embed again to refresh any that changed"
        )
    })
}

pub(crate) fn embed_options_from_flags(
    token_budget: Option<usize>,
    forward_pause_ms: Option<u64>,
) -> embedder::EmbedOptions {
    embedder::EmbedOptions {
        token_budget,
        forward_pause: forward_pause_ms.map(Duration::from_millis),
    }
}

pub(crate) fn index_and_report(
    db_path: &Option<PathBuf>,
    force: bool,
    embed_options: embedder::EmbedOptions,
) -> Result<IndexOutcome> {
    let home = home_dir().context("Could not determine home directory")?;
    let claude_dir = indexer::resolve_claude_dir(&home);
    let codex_dir = indexer::resolve_codex_dir(&home);
    let opts = indexer::IndexOptions {
        force,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    let outcome = index_and_report_observed(
        db_path,
        &opts,
        || index_embedder(&model::HOST),
        embed_options,
        &index_observer::Observer::stderr(),
    )?;
    if let Some(note) = &outcome.degraded_note {
        cli_info(note);
    }
    if let Some(note) = preserved_embedded_note(outcome.preserved_embedded) {
        cli_info(&note);
    }
    Ok(outcome)
}

/// Picks the always-probe loader for index and rebuild.
pub(crate) fn index_embedder(host: &ModelHost<'_>) -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_cached_with_reporter(|| host.load_probing(), |_: &str| {})
}

#[derive(Default)]
pub(crate) struct IndexOutcome {
    pub(crate) degraded_note: Option<String>,
    pub(crate) embedded: usize,
    pub(crate) failed_count: usize,
    pub(crate) first_error: Option<String>,
    pub(crate) skipped_roots: Vec<indexer::SkippedRoot>,
    pub(crate) preserved_embedded: usize,
    pub(crate) parse_diagnostics: Vec<indexer::FileParseDiagnostics>,
    pub(crate) observations: serde_json::Value,
}

const EMBED_ERROR_NOTE_MAX_CHARS: usize = 120;

pub(crate) fn index_command_output(outcome: &IndexOutcome) -> CommandOutput {
    debug_assert!(
        outcome.degraded_note.is_none() || outcome.failed_count == 0,
        "model-absent and embed-stop degradation are mutually exclusive"
    );
    let mut notes = Vec::new();
    if let Some(note) = &outcome.degraded_note {
        notes.push(note.clone());
    }
    if outcome.failed_count > 0 {
        let raw = outcome.first_error.as_deref().unwrap_or("unknown error");
        let err: String = ansi::strip_control_chars(raw)
            .chars()
            .take(EMBED_ERROR_NOTE_MAX_CHARS)
            .collect();
        notes.push(format!(
            "{n} chunk(s) failed to embed ({err}); rerun `recall index` to retry",
            n = outcome.failed_count,
        ));
    }
    for skipped in &outcome.skipped_roots {
        notes.push(
            skipped
                .reason
                .note(skipped.source.as_str(), skipped.preserved_sessions),
        );
    }
    if let Some(note) = preserved_embedded_note(outcome.preserved_embedded) {
        notes.push(note);
    }
    notes.extend(
        outcome
            .parse_diagnostics
            .iter()
            .flat_map(indexer::FileParseDiagnostics::notes),
    );
    let degraded = !notes.is_empty();
    let skipped_roots: Vec<_> = outcome
        .skipped_roots
        .iter()
        .map(|s| {
            serde_json::json!({
                "source": s.source.as_str(),
                "preserved_sessions": s.preserved_sessions,
                "reason": s.reason.as_str(),
            })
        })
        .collect();
    let parse_diagnostics: Vec<_> = outcome
        .parse_diagnostics
        .iter()
        .map(|diagnostic| {
            serde_json::json!({
                "file_path": diagnostic.display_path(),
                "source": diagnostic.source.as_str(),
                "invalid_json_lines": diagnostic.lines.invalid_json_lines,
                "invalid_utf8_lines": diagnostic.lines.invalid_utf8_lines,
                "incomplete_tail_lines": diagnostic.lines.incomplete_tail_lines,
                "read_error": diagnostic.read_error,
            })
        })
        .collect();
    let data = serde_json::json!({
        "observations": outcome.observations,
        "parse_diagnostics": parse_diagnostics,
        "embedded": outcome.embedded,
        "failed_count": outcome.failed_count,
        "skipped_roots": skipped_roots,
        "preserved_embedded": outcome.preserved_embedded,
    });
    CommandOutput::with_notes(String::new(), data, degraded, notes)
}

#[cfg(test)]
pub(crate) fn index_and_report_with<F>(
    db_path: &Option<PathBuf>,
    opts: &indexer::IndexOptions,
    load_embedder: F,
) -> Result<IndexOutcome>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    index_and_report_observed(
        db_path,
        opts,
        load_embedder,
        embedder::EmbedOptions::default(),
        &index_observer::Observer::stderr(),
    )
}

pub(crate) fn index_and_report_observed<F>(
    db_path: &Option<PathBuf>,
    opts: &indexer::IndexOptions,
    load_embedder: F,
    embed_options: embedder::EmbedOptions,
    observer: &index_observer::Observer,
) -> Result<IndexOutcome>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    let pipeline = observer.stage("total");
    let path = resolve_db_path(db_path)?;
    let opening = observer.stage("database_open");
    let mut conn = open_or_create_db(&path)?;
    opening.finish("complete");

    let loading = observer.stage("model_load_probe");
    let load_result = load_embedder().and_then(|embedder| {
        embedder
            .embed_document("probe")
            .map(|_| embedder)
            .map_err(degrade_with_warn(
                "embedder inference probe: embed_document",
                DegradedReason::ProbeFailed,
            ))
    });
    let embed_capable = load_result.is_ok();
    loading.finish(if embed_capable {
        "ready"
    } else {
        "unavailable; embedding deferred"
    });

    let stats = indexer::index_from_dirs_observed(&mut conn, opts, embed_capable, observer)?;
    observer.count("sessions_stored", stats.total_sessions);
    let main_msg = if stats.indexed > 0 {
        format!(
            "Indexed {} sessions in {:.1}s",
            stats.indexed, stats.elapsed_secs
        )
    } else if !stats.parse_diagnostics.is_empty() {
        format!(
            "{} sessions stored; {} file(s) with unresolved parsing diagnostics",
            stats.total_sessions,
            stats.parse_diagnostics.len()
        )
    } else {
        format!("{} sessions up to date", stats.total_sessions)
    };
    done(&main_msg);
    for skipped in &stats.skipped_roots {
        warning(
            &skipped
                .reason
                .note(skipped.source.as_str(), skipped.preserved_sessions),
        );
    }
    let skipped_roots = stats.skipped_roots;

    let chunk_stats = indexer::index_chunks_observed(&mut conn, None, observer)?;
    let backfill = observer.stage("legacy_backfill");
    let backfilled = indexer::backfill_rowid_ranges(&mut conn)?;
    if backfilled > 0 {
        debug!(
            sessions = backfilled,
            "linked legacy chunks to rowid ranges"
        );
    }
    let on_backfill_progress = |done: usize, total: usize| {
        backfill.progress(done.saturating_sub(1), total, 0);
    };
    let scanned_backfilled =
        indexer::backfill_session_files(&mut conn, Some(&on_backfill_progress))?;
    let parse_diagnostics = indexer::load_parse_diagnostics(&conn)?;
    for note in parse_diagnostics
        .iter()
        .flat_map(indexer::FileParseDiagnostics::notes)
    {
        warning(&note);
    }
    if scanned_backfilled > 0 {
        debug!(
            sessions = scanned_backfilled,
            "recorded scanned files for legacy sessions"
        );
    }
    backfill.finish("committed");
    if chunk_stats.chunks_created > 0 {
        done(&format!("Created {} chunks", chunk_stats.chunks_created));
    } else {
        done("Chunks up to date");
    }

    if opts.force {
        fix_orphan_embeddings(&conn)?;
    }

    let preserved_embedded = stats.preserved_embedded;
    let mut outcome = match load_result {
        Ok(embedder) => {
            let result = embedder::embed_pending_observed(
                &mut conn,
                embedder.as_ref(),
                None,
                &embed_options,
                observer,
            )?;
            result.warn_if_batches_failed();
            IndexOutcome {
                degraded_note: None,
                embedded: result.embedded,
                failed_count: result.failed_count,
                first_error: result.first_error,
                skipped_roots,
                preserved_embedded,
                parse_diagnostics,
                observations: serde_json::Value::Null,
            }
        }
        Err(reason) => {
            let extraction = observer.stage("pending_extraction");
            let pending_total = embedder::pending_count(&conn)?;
            extraction.finish("complete");
            observer.start_embedding(pending_total);
            IndexOutcome {
                degraded_note: search_degraded_note(reason),
                skipped_roots,
                preserved_embedded,
                parse_diagnostics,
                ..IndexOutcome::default()
            }
        }
    };
    pipeline.finish("complete; see counts for deferred or failed work");
    outcome.observations = observer.snapshot();
    Ok(outcome)
}

#[cfg(test)]
pub(crate) fn embed_all_pending(
    conn: &mut Connection,
    embedder: &dyn Embed,
    options: &embedder::EmbedOptions,
) -> Result<embedder::EmbedResult> {
    let result = embedder::embed_pending_observed(
        conn,
        embedder,
        None,
        options,
        &index_observer::Observer::default(),
    )?;
    result.warn_if_batches_failed();
    Ok(result)
}

#[cfg(test)]
mod tests;
