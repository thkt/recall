mod ansi;
mod chunker;
mod classify;
mod compat;
mod date;
mod db;
mod embedder;
mod embedding;
mod envelope;
mod error;
mod hybrid;
mod index_observer;
mod indexer;
mod model;
mod native;
mod output;
mod parser;
mod runtime;
mod search;
mod storage;

use std::borrow::Cow;
use std::env;
use std::ffi::{OsStr, OsString};
use std::fmt::Display;
use std::fs::{File, OpenOptions, Permissions, create_dir_all, metadata};
use std::io::{self, ErrorKind, Write};
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use crate::compat::filter::escape_like;
use crate::embedding::{EMBEDDING_DIMS, Embed};
use crate::model::cached_artifacts;
use crate::model::handle_probe_if_needed;
use crate::model::{
    DegradedReason, EmbedderDegraded, degrade_with_warn, download_and_verify_model,
    record_degraded, try_load_embedder_default_logging,
};
use crate::runtime::{
    codes, done, exit_error, home_dir, info as cli_info, init_subscriber, try_expand_shorthand,
    warning,
};
use anyhow::{Context, Error, Result};
use clap::error::ErrorKind as ClapErrorKind;
use clap::{Args, Parser, Subcommand};
use rusqlite::{Connection, OptionalExtension};
use tracing::{debug, warn};

use crate::envelope::{CommandOutput, render_json_error, render_json_success};
use crate::error::{RecallError, classify_exit_code, download_error, error_envelope};
use crate::output::{WriteOutcome, write_result};
use crate::parser::Source;

const LOG_FILTER_VERBOSE: &str = "recall=info";
const LOG_FILTER_DEFAULT: &str = "recall=warn";

#[derive(Parser)]
#[command(name = "recall", about = "Search past Claude Code and Codex sessions")]
struct Cli {
    #[command(subcommand)]
    command: Option<Command>,

    /// Show diagnostic output
    #[arg(long, short, global = true)]
    verbose: bool,

    /// Database file path (default: ~/.local/share/recall/recall.db, env: RECALL_DB)
    #[arg(long, env = "RECALL_DB", global = true)]
    db_path: Option<PathBuf>,

    /// Pinned Gemma model manifest. Select an older saved manifest to roll back.
    #[arg(long, env = "RECALL_MODEL_MANIFEST", global = true)]
    model_manifest: Option<PathBuf>,

    /// Emit a machine-readable JSON envelope instead of human-readable text
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse, chunk, and embed new session logs (incremental).
    Index {
        #[command(flatten)]
        embed_flags: EmbedFlags,
    },
    /// Re-parse and re-embed every present session; a missing or unreadable
    /// source root keeps its existing index.
    Rebuild {
        #[command(flatten)]
        embed_flags: EmbedFlags,
    },
    /// Manage the embedding model.
    #[command(subcommand)]
    Model(ModelCommand),
    /// Search indexed sessions (default when query is provided)
    Search {
        /// Search query. Optional when `--file` is given: a query-less
        /// `--file <path>` lists the sessions that touched the path, newest
        /// first. A call with neither a query nor `--file` is a usage error —
        /// rejected declaratively here, with `file_list_outcome` keeping the
        /// same guard for direct (non-CLI) callers.
        #[arg(required_unless_present = "file")]
        query: Option<String>,

        /// Filter by project path (prefix match)
        #[arg(long)]
        project: Option<String>,

        /// Only sessions from last N days
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..))]
        days: Option<i64>,

        /// Filter by source
        #[arg(long, value_enum)]
        source: Option<Source>,

        /// Restrict to sessions that touched this file (exact path, or a
        /// basename / tail sub-path suffix). While Codex file extraction is
        /// unsupported, Codex sessions record no files and are never matched.
        #[arg(long)]
        file: Option<String>,

        /// Max results (1..=100)
        #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u16).range(1..=100))]
        limit: u16,

        /// Exclude the invoking Claude Code session from results. Default when
        /// CLAUDE_CODE_SESSION_ID is set (recall run from inside a session).
        #[arg(long, conflicts_with_all = ["include_current", "only_current"])]
        exclude_current: bool,

        /// Include the invoking session even when CLAUDE_CODE_SESSION_ID is set.
        #[arg(long, conflicts_with_all = ["exclude_current", "only_current"])]
        include_current: bool,

        /// Return only the invoking session (requires CLAUDE_CODE_SESSION_ID).
        #[arg(long, conflicts_with_all = ["exclude_current", "include_current"])]
        only_current: bool,

        /// Include sessions classified automated (hook/script/agent-generated).
        /// Default excludes them.
        #[arg(long)]
        include_automated: bool,
    },
    /// Show full conversation of a session
    Show {
        /// Session ID (prefix match supported)
        session_id: String,
        /// Anchor on a qa_chunk id and show only its message range plus neighbors
        #[arg(long)]
        chunk: Option<i64>,
        /// Per-side neighbor count around the anchored chunk (requires --chunk)
        #[arg(long, default_value_t = DEFAULT_WINDOW)]
        window: i64,
    },
    /// Classify existing sessions from source provenance and their first user turn
    Classify {
        /// Re-classify every session (default: only unclassified)
        #[arg(long)]
        all: bool,
        /// Report what would change without writing
        #[arg(long)]
        dry_run: bool,
    },
    /// Show index and embedding status
    Status,
    /// Diagnose a broken index: corruption, orphaned rows, and a live model
    /// load probe. Read-only by default; `--fix` repairs orphan embeddings and
    /// re-runs the checks. Other remedies point at index/rebuild/model download.
    Doctor {
        /// Delete dangling `vec_chunks` rows whose `chunk_id` no longer resolves
        /// to a `qa_chunks` row (the orphan_embeddings fault), then re-run the
        /// checks so the verdict reflects the repaired state.
        #[arg(long)]
        fix: bool,
    },
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download the embedding model and verify it loads.
    Download,
    /// Print the selected pinned manifest for archiving, updates, or rollback.
    Manifest,
}

/// Embed-pass tuning flags shared verbatim by `index` and `rebuild`
/// (`#[command(flatten)]`), so the two subcommands' flag definitions and docs
/// cannot drift apart.
#[derive(Debug, Args)]
struct EmbedFlags {
    /// Forward-pass token budget override; smaller values shorten each GPU
    /// forward so the host stays responsive during large embed runs. Clamped
    /// down to 256_000 (never up). Gemma packing is additionally capped at
    /// 8192 tokens; one attention window remains atomic.
    #[arg(long, env = "RECALL_TOKEN_BUDGET", value_parser = parse_token_budget)]
    token_budget: Option<usize>,

    /// Pause (in milliseconds) after each GPU forward pass, yielding the GPU
    /// to interactive processes; trades throughput for host responsiveness
    /// during large embed runs.
    #[arg(long, env = "RECALL_FORWARD_PAUSE_MS")]
    forward_pause_ms: Option<u64>,
}

fn create_db_file(path: &Path) -> io::Result<()> {
    create_dir_all(path.parent().unwrap_or(Path::new("")))?;
    let mut opts = OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    opts.open(path)?;
    Ok(())
}

fn format_timestamp(ts_ms: Option<i64>) -> String {
    // Display UTC dates; --days filters a rolling 24-hour window.
    let Some(ts) = ts_ms else {
        return "unknown".to_owned();
    };
    let days = ts.div_euclid(date::MS_PER_DAY);
    let (y, m, d) = date::civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn open_or_create_db(path: &Path) -> Result<Connection> {
    match create_db_file(path) {
        Ok(_) => {}
        Err(e) if e.kind() == ErrorKind::AlreadyExists => {}
        Err(e) => return Err(e).context("Failed to create database file"),
    }
    {
        // Reject a mismatched model before any schema or journal-mode write.
        let (existing, _) = db::open_db_readonly(path)?;
        model::check_index(&existing, model::selected())
            .map_err(|e| RecallError::DatabaseRead(e.to_string()))?;
    }
    let mut conn = db::open_db(path).context("Failed to open database")?;
    model::bind_index(&mut conn, model::selected())?;
    Ok(conn)
}

fn resolve_db_path(db_path: &Option<PathBuf>) -> Result<PathBuf> {
    match db_path {
        Some(p) => Ok(p.clone()),
        None => resolve_default_db_path(
            env::var_os("XDG_DATA_HOME").as_deref(),
            home_dir().as_deref(),
        ),
    }
}

fn resolve_default_db_path(xdg_data_home: Option<&OsStr>, home: Option<&Path>) -> Result<PathBuf> {
    let data_home = match xdg_data_home.map(Path::new) {
        Some(p) if p.is_absolute() => p.to_path_buf(),
        _ => home
            .context("Could not determine home directory")?
            .join(".local")
            .join("share"),
    };
    Ok(data_home
        .join("recall")
        .join("indexes")
        .join(model::selected().identity())
        .join("recall.db"))
}

fn try_load_embedder_cached() -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_cached_with(try_load_embedder_default_logging)
}

fn try_load_embedder_cached_with<F>(open_embedder: F) -> Result<Arc<dyn Embed>, DegradedReason>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    try_load_embedder_cached_with_reporter(open_embedder, cli_info)
}

fn try_load_embedder_cached_with_reporter<F, I>(
    open_embedder: F,
    mut report_info: I,
) -> Result<Arc<dyn Embed>, DegradedReason>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
    I: FnMut(&str),
{
    open_embedder().inspect_err(|&reason| {
        if let Some(note) = search_degraded_note(reason) {
            report_info(&note);
        }
        if reason != DegradedReason::NotInstalled {
            record_degraded(reason, "embedder load");
        }
    })
}

fn search_degraded_note(reason: DegradedReason) -> Option<String> {
    EmbedderDegraded(reason)
        .user_note("recall model download")
        .map(|note| format!("note: {note}."))
}

fn preserved_embedded_note(count: usize) -> Option<String> {
    (count > 0).then(|| {
        format!(
            "preserved embeddings for {count} session(s) the model could not rebuild this run — semantic search for them stays intact; re-run `recall index` once the model can embed again to refresh any that changed"
        )
    })
}

fn search_fallback_notes(reason: DegradedReason) -> Vec<String> {
    EmbedderDegraded(reason)
        .user_note("recall model download")
        .map(|note| vec![format!("semantic search unavailable: {note}")])
        .unwrap_or_default()
}

fn search_degraded_state(load: &Result<Arc<dyn Embed>, DegradedReason>) -> (bool, Vec<String>) {
    match load {
        Ok(_) => (false, Vec::new()),
        Err(reason) => {
            let notes = search_fallback_notes(*reason);
            (!notes.is_empty(), notes)
        }
    }
}

fn parse_token_budget(s: &str) -> Result<usize, ParseIntError> {
    let value: usize = s.parse()?;
    Ok(value.min(embedder::TOKEN_BUDGET_CEILING))
}

fn run_index(
    db_path: &Option<PathBuf>,
    token_budget: Option<usize>,
    forward_pause_ms: Option<u64>,
) -> Result<IndexOutcome> {
    index_and_report(
        db_path,
        false,
        embed_options_from_flags(token_budget, forward_pause_ms),
    )
}

fn run_rebuild(
    db_path: &Option<PathBuf>,
    token_budget: Option<usize>,
    forward_pause_ms: Option<u64>,
) -> Result<IndexOutcome> {
    index_and_report(
        db_path,
        true,
        embed_options_from_flags(token_budget, forward_pause_ms),
    )
}

fn embed_options_from_flags(
    token_budget: Option<usize>,
    forward_pause_ms: Option<u64>,
) -> embedder::EmbedOptions {
    embedder::EmbedOptions {
        token_budget,
        forward_pause: forward_pause_ms.map(Duration::from_millis),
    }
}

fn index_and_report(
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
    let outcome = index_and_report_with_options(
        db_path,
        &opts,
        || try_load_embedder_cached_with_reporter(try_load_embedder_default_logging, |_: &str| {}),
        embed_options,
    )?;
    if let Some(note) = &outcome.degraded_note {
        cli_info(note);
    }
    if let Some(note) = preserved_embedded_note(outcome.preserved_embedded) {
        cli_info(&note);
    }
    Ok(outcome)
}

#[derive(Default)]
struct IndexOutcome {
    degraded_note: Option<String>,
    embedded: usize,
    failed_count: usize,
    first_error: Option<String>,
    skipped_roots: Vec<indexer::SkippedRoot>,
    preserved_embedded: usize,
    parse_diagnostics: Vec<indexer::FileParseDiagnostics>,
    observations: serde_json::Value,
}

const EMBED_ERROR_NOTE_MAX_CHARS: usize = 120;

fn index_command_output(outcome: &IndexOutcome) -> CommandOutput {
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
fn index_and_report_with<F>(
    db_path: &Option<PathBuf>,
    opts: &indexer::IndexOptions,
    load_embedder: F,
) -> Result<IndexOutcome>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    index_and_report_with_options(
        db_path,
        opts,
        load_embedder,
        embedder::EmbedOptions::default(),
    )
}

fn index_and_report_with_options<F>(
    db_path: &Option<PathBuf>,
    opts: &indexer::IndexOptions,
    load_embedder: F,
    embed_options: embedder::EmbedOptions,
) -> Result<IndexOutcome>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    index_and_report_observed(
        db_path,
        opts,
        load_embedder,
        embed_options,
        &index_observer::Observer::stderr(),
    )
}

fn index_and_report_observed<F>(
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
    let outcome: Result<IndexOutcome> = match load_result {
        Ok(embedder) => {
            let result = embedder::embed_pending_observed(
                &mut conn,
                embedder.as_ref(),
                None,
                &embed_options,
                observer,
            )?;
            result.warn_if_batches_failed();
            Ok(IndexOutcome {
                degraded_note: None,
                embedded: result.embedded,
                failed_count: result.failed_count,
                first_error: result.first_error,
                skipped_roots,
                preserved_embedded,
                parse_diagnostics,
                observations: serde_json::Value::Null,
            })
        }
        Err(reason) => {
            let extraction = observer.stage("pending_extraction");
            let pending_total = embedder::pending_count(&conn)?;
            extraction.finish("complete");
            observer.start_embedding(pending_total);
            Ok(IndexOutcome {
                degraded_note: search_degraded_note(reason),
                skipped_roots,
                preserved_embedded,
                parse_diagnostics,
                ..IndexOutcome::default()
            })
        }
    };
    pipeline.finish("complete; see counts for deferred or failed work");
    let mut outcome: IndexOutcome = outcome?;
    outcome.observations = observer.snapshot();
    Ok(outcome)
}

#[cfg(test)]
fn embed_all_pending(
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

fn run_model_download() -> Result<()> {
    download_and_verify_model().map_err(|e| download_error(&e).into())
}

fn search_idle_message(session_count: i64) -> Option<&'static str> {
    (session_count == 0).then_some("No sessions indexed. Run `recall index` first.")
}

fn search_idle_output() -> CommandOutput {
    if let Some(msg) = search_idle_message(0) {
        done(msg);
    }
    CommandOutput::ok(String::new(), serde_json::json!({ "results": [] }))
}

fn require_current_schema(state: db::SchemaState) -> Result<(), RecallError> {
    match state {
        db::SchemaState::Current | db::SchemaState::Empty => Ok(()),
        db::SchemaState::Stale => Err(RecallError::DataError(
            "database schema is out of date; run `recall rebuild` to migrate it".to_owned(),
        )),
    }
}

#[derive(Clone, Copy, Debug)]
struct CurrentSessionFlags {
    exclude: bool,
    include: bool,
    only: bool,
}

fn resolve_current_session(
    flags: CurrentSessionFlags,
    env_session_id: Option<String>,
) -> Result<search::CurrentSession, RecallError> {
    use search::CurrentSession;
    let env_session_id = env_session_id.filter(|s| !s.trim().is_empty());
    if flags.only {
        return match env_session_id {
            Some(id) => Ok(CurrentSession::Only(id)),
            None => Err(RecallError::Usage(
                "--only-current requires CLAUDE_CODE_SESSION_ID; run recall from inside a Claude Code session"
                    .to_owned(),
            )),
        };
    }
    if flags.include {
        return Ok(CurrentSession::Ignore);
    }
    match env_session_id {
        Some(id) => Ok(CurrentSession::Exclude(id)),
        None => {
            if flags.exclude {
                warn!("--exclude-current has no effect: CLAUDE_CODE_SESSION_ID is not set");
            }
            Ok(CurrentSession::Ignore)
        }
    }
}

fn run_search(cmd: Command, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    run_search_with(cmd, db_path, try_load_embedder_cached)
}

fn run_search_with<F>(
    cmd: Command,
    db_path: &Option<PathBuf>,
    load_embedder: F,
) -> Result<CommandOutput>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    let Command::Search {
        query,
        project,
        days,
        source,
        file,
        limit,
        exclude_current,
        include_current,
        only_current,
        include_automated,
    } = cmd
    else {
        unreachable!()
    };

    let current = resolve_current_session(
        CurrentSessionFlags {
            exclude: exclude_current,
            include: include_current,
            only: only_current,
        },
        env::var("CLAUDE_CODE_SESSION_ID").ok(),
    )?;

    let path = resolve_db_path(db_path)?;

    if !path.exists() {
        return Ok(search_idle_output());
    }

    let (conn, tier) = db::open_db_readonly(&path)?;
    match db::schema_state(&conn)? {
        db::SchemaState::Empty => return Ok(search_idle_output()),
        state => require_current_schema(state)?,
    }
    let wal_note = db::stale_wal_note(&path, tier);

    let session_count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    if search_idle_message(session_count).is_some() {
        return Ok(search_idle_output());
    }

    let identity_error = if query.is_some() {
        model::check_index(&conn, model::selected()).err()
    } else {
        None
    };
    let (mut degraded, mut notes, embedder) = if let Some(error) = identity_error {
        (
            true,
            vec![format!(
                "semantic search unavailable: {error}; results are text-only"
            )],
            None,
        )
    } else if query.is_some() {
        let embedder = load_embedder();
        let (degraded, notes) = search_degraded_state(&embedder);
        (degraded, notes, embedder.ok())
    } else {
        (false, Vec::new(), None)
    };

    let outcome = search::search_with_embedder(
        &conn,
        query.as_deref(),
        &search::SearchOptions {
            project,
            days,
            source,
            file,
            limit: limit.into(),
            current,
            include_automated,
            now_ms: None,
        },
        embedder.as_deref(),
    )?;
    let results = outcome.results;

    if outcome.vec_degraded {
        degraded = true;
        notes.push(
            "semantic search unavailable: vector query failed at runtime; results are text-only"
                .to_owned(),
        );
    }

    if outcome.results_incomplete {
        degraded = true;
        notes.push(
            "results may be incomplete: some matched sessions were excluded because their index rows were unreadable or carry an unrecognized source; for unreadable rows, run `recall index` to rebuild"
                .to_owned(),
        );
    }

    if !include_automated {
        let automated: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sessions WHERE session_type = 'automated'",
                [],
                |r| r.get(0),
            )
            .unwrap_or(0);
        if automated > 0 {
            notes.push(format!(
                "{automated} automated session(s) excluded by default; use --include-automated to include them"
            ));
        }
    }

    let markdown = render_results(&results);
    let data = serde_json::json!({
        "results": results.iter().map(result_to_json).collect::<Vec<_>>(),
    });

    if let Some(note) = wal_note {
        degraded = true;
        notes.push(note);
    }

    Ok(CommandOutput::with_notes(markdown, data, degraded, notes))
}

fn write_show_header(
    buf: &mut Vec<u8>,
    session: &parser::SessionData,
    verbose: bool,
    extra: &[String],
) {
    let _ = writeln!(buf, "# {}", session.slug);
    let _ = writeln!(buf, "- session_id: {}", session.session_id);
    let _ = writeln!(buf, "- date: {}", format_timestamp(session.timestamp));
    let _ = writeln!(buf, "- project: {}", session.project);
    let _ = writeln!(buf, "- source: {}", session.source);
    for line in extra {
        let _ = writeln!(buf, "{line}");
    }
    if verbose {
        let _ = writeln!(buf, "- file: {}", session.file_path);
    }
    let _ = writeln!(buf);
}

fn show_session(conn: &Connection, session_id: &str, verbose: bool) -> Result<CommandOutput> {
    let session = resolve_session(conn, session_id)?;

    let mut msg_stmt =
        conn.prepare("SELECT role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let rows = msg_stmt.query_map([&session.session_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    let messages: Vec<(String, String)> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    let mut buf = Vec::new();
    write_show_header(&mut buf, &session, verbose, &[]);
    for (role, text) in &messages {
        let _ = writeln!(buf, "## [{role}]");
        let _ = writeln!(buf, "{text}");
        let _ = writeln!(buf);
    }
    let markdown = String::from_utf8_lossy(&buf).into_owned();

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "messages": messages
            .iter()
            .map(|(role, text)| serde_json::json!({ "role": role, "text": text }))
            .collect::<Vec<_>>(),
    });

    Ok(CommandOutput::ok(markdown, data))
}

const DEFAULT_WINDOW: i64 = 5;
const MAX_WINDOW: i64 = 50;
const MAX_MSG_CHARS: usize = 4000;

fn resolve_session(conn: &Connection, session_id: &str) -> Result<parser::SessionData> {
    let mut stmt = conn.prepare(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id LIKE ?1 ESCAPE '\\' ORDER BY session_id",
    )?;
    let pattern = format!("{}%", escape_like(session_id));
    let rows = stmt.query_map([&pattern], |row| {
        let source_str: String = row.get(1)?;
        Ok(parser::SessionData {
            session_id: row.get(0)?,
            source: Source::from_db(&source_str).unwrap_or_else(|| {
                warn!(source = %source_str, "unknown source, defaulting to claude");
                Source::Claude
            }),
            file_path: row.get(2)?,
            project: row.get(3)?,
            slug: row.get(4)?,
            timestamp: row.get(5)?,
        })
    })?;
    let matches: Vec<parser::SessionData> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    if matches.is_empty() {
        return Err(RecallError::Usage(format!("No session found matching '{session_id}'")).into());
    }
    if matches.len() > 1 {
        let mut msg = format!("Multiple sessions match '{session_id}':\n");
        for s in &matches {
            msg.push_str(&format!("  {}  {}\n", s.session_id, s.slug));
        }
        msg.push_str("Narrow the session ID prefix to match exactly one session");
        return Err(RecallError::Usage(msg).into());
    }
    Ok(matches.into_iter().next().expect("exactly one match"))
}

fn cap_message(text: &str) -> String {
    if text.chars().count() <= MAX_MSG_CHARS {
        return text.to_owned();
    }
    let cut: String = text.chars().take(MAX_MSG_CHARS).collect();
    format!("{cut}…")
}

fn show_session_windowed(
    conn: &Connection,
    session_id: &str,
    verbose: bool,
    chunk: Option<i64>,
    window: i64,
) -> Result<CommandOutput> {
    let Some(chunk_id) = chunk else {
        return show_session(conn, session_id, verbose);
    };

    let session = resolve_session(conn, session_id)?;

    let anchor = conn
        .query_row(
            "SELECT session_id, content, src_rowid_lo, src_rowid_hi \
             FROM qa_chunks WHERE id = ?1",
            [chunk_id],
            |row| {
                Ok((
                    row.get::<_, String>(0)?,
                    row.get::<_, String>(1)?,
                    row.get::<_, Option<i64>>(2)?,
                    row.get::<_, Option<i64>>(3)?,
                ))
            },
        )
        .optional()?
        .ok_or_else(|| RecallError::Usage(format!("No chunk found with id {chunk_id}")))?;
    let (chunk_session, content, lo, hi) = anchor;

    if chunk_session != session.session_id {
        return Err(RecallError::Usage(format!(
            "chunk {chunk_id} belongs to session '{chunk_session}', not '{}'",
            session.session_id
        ))
        .into());
    }

    match (lo, hi) {
        (Some(lo), Some(hi)) => render_windowed(conn, &session, chunk_id, lo, hi, window, verbose),
        _ => Ok(render_chunk_fallback(&session, chunk_id, &content, verbose)),
    }
}

fn render_windowed(
    conn: &Connection,
    session: &parser::SessionData,
    chunk_id: i64,
    lo: i64,
    hi: i64,
    window: i64,
    verbose: bool,
) -> Result<CommandOutput> {
    let n = window.clamp(0, MAX_WINDOW) as usize;

    let mut stmt = conn
        .prepare("SELECT rowid, role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let rows = stmt.query_map([&session.session_id], |row| {
        Ok((
            row.get::<_, i64>(0)?,
            row.get::<_, String>(1)?,
            row.get::<_, String>(2)?,
        ))
    })?;
    let msgs: Vec<(i64, String, String)> = rows.collect::<rusqlite::Result<Vec<_>>>()?;

    let first_body = msgs.iter().position(|(rid, ..)| *rid >= lo);
    let last_body = msgs.iter().rposition(|(rid, ..)| *rid <= hi);
    let slice: &[(i64, String, String)] = match (first_body, last_body) {
        (Some(f), Some(l)) if f <= l => {
            let start = f.saturating_sub(n);
            let end = (l + n).min(msgs.len() - 1);
            &msgs[start..=end]
        }
        _ => &[],
    };

    let mut buf = Vec::new();
    write_show_header(
        &mut buf,
        session,
        verbose,
        &[format!("- chunk: {chunk_id}"), format!("- window: {n}")],
    );
    for (rid, role, text) in slice {
        let marker = if *rid >= lo && *rid <= hi {
            " (chunk)"
        } else {
            ""
        };
        let _ = writeln!(buf, "## [{role}]{marker}");
        let _ = writeln!(buf, "{}", cap_message(text));
        let _ = writeln!(buf);
    }
    let markdown = String::from_utf8_lossy(&buf).into_owned();

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "chunk": chunk_id,
        "window": n,
        "messages": slice
            .iter()
            .map(|(rid, role, text)| serde_json::json!({
                "rowid": rid,
                "role": role,
                "text": cap_message(text),
                "in_chunk": *rid >= lo && *rid <= hi,
            }))
            .collect::<Vec<_>>(),
    });

    Ok(CommandOutput::ok(markdown, data))
}

fn render_chunk_fallback(
    session: &parser::SessionData,
    chunk_id: i64,
    content: &str,
    verbose: bool,
) -> CommandOutput {
    let hint = format!(
        "This chunk predates message-range tracking; run `recall show {}` for the full conversation.",
        session.session_id
    );

    let mut buf = Vec::new();
    write_show_header(
        &mut buf,
        session,
        verbose,
        &[format!("- chunk: {chunk_id}")],
    );
    let _ = writeln!(buf, "{}", cap_message(content));
    let _ = writeln!(buf);
    let _ = writeln!(buf, "{hint}");
    let markdown = String::from_utf8_lossy(&buf).into_owned();

    let data = serde_json::json!({
        "session_id": session.session_id,
        "slug": session.slug,
        "project": session.project,
        "source": session.source.to_string(),
        "timestamp": session.timestamp,
        "chunk": chunk_id,
        "content": cap_message(content),
        "windowed": false,
        "next_step": hint,
    });

    CommandOutput::ok(markdown, data)
}

fn run_show(
    session_id: &str,
    verbose: bool,
    chunk: Option<i64>,
    window: i64,
    db_path: &Option<PathBuf>,
) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        return Err(
            RecallError::Usage("Database not found. Run `recall index` first.".to_owned()).into(),
        );
    }

    let (conn, _tier) = db::open_db_readonly(&path)?;
    match db::schema_state(&conn)? {
        db::SchemaState::Empty => Err(RecallError::Usage(
            "Database not found. Run `recall index` first.".to_owned(),
        )
        .into()),
        state => {
            require_current_schema(state)?;
            show_session_windowed(&conn, session_id, verbose, chunk, window)
        }
    }
}

fn run_status(verbose: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;

    let model_ok = cached_artifacts().map(|opt| opt.is_some()).unwrap_or(false);

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

    let (conn, tier) = db::open_db_readonly(&path)?;

    let (sessions, chunks, embedded) = match db::schema_state(&conn)? {
        db::SchemaState::Empty => (0, 0, 0),
        state => {
            require_current_schema(state)?;
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

    let mut buf = Vec::new();
    let _ = writeln!(buf, "Sessions: {sessions}");
    let _ = writeln!(buf, "QA chunks: {chunks}");
    let _ = writeln!(buf, "Embedded: {embedded}/{chunks}");
    let _ = writeln!(
        buf,
        "Model: {}",
        if model_ok {
            "ready"
        } else {
            "not installed (run `recall model download`)"
        }
    );
    if verbose {
        let _ = writeln!(buf, "DB: {}", path.display());
    }
    let markdown = String::from_utf8_lossy(&buf).into_owned();
    let data = serde_json::json!({
        "sessions": sessions,
        "qa_chunks": chunks,
        "embedded": embedded,
        "model_ready": model_ok,
    });

    let mut notes: Vec<_> = db::stale_wal_note(&path, tier).into_iter().collect();
    if let Err(error) = model::check_index(&conn, model::selected()) {
        notes.push(format!(
            "semantic search unavailable: {error}; results are text-only"
        ));
    }
    Ok(CommandOutput::with_notes(
        markdown,
        data,
        !notes.is_empty(),
        notes,
    ))
}

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

fn fix_orphan_embeddings(conn: &Connection) -> Result<usize> {
    let deleted = conn.execute(
        "DELETE FROM vec_chunks WHERE chunk_id NOT IN (SELECT id FROM qa_chunks)",
        [],
    )?;
    Ok(deleted)
}

struct Repair {
    name: &'static str,
    action: fn(&Connection) -> Result<usize>,
}

struct RepairOutcome {
    name: &'static str,
    result: Result<usize>,
}

const REPAIRS: &[Repair] = &[Repair {
    name: "orphan_embeddings",
    action: fix_orphan_embeddings,
}];

fn run_repairs(conn: &Connection) -> Vec<RepairOutcome> {
    REPAIRS
        .iter()
        .map(|r| RepairOutcome {
            name: r.name,
            result: (r.action)(conn),
        })
        .collect()
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
    fix: bool,
    load_embedder: F,
    recover_model: R,
) -> (DoctorCheck, ModelRecovery)
where
    F: Fn() -> Result<Arc<dyn Embed>, DegradedReason>,
    R: FnOnce() -> Result<(), RecallError>,
{
    let initial = check_model(&load_embedder);
    if !fix || initial.is_fully_usable() {
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

fn run_doctor(verbose: bool, fix: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    run_doctor_with(verbose, fix, db_path, try_load_embedder_cached, || {
        download_and_verify_model().map_err(|e| download_error(&e))
    })
}

fn run_doctor_with<F, R>(
    verbose: bool,
    fix: bool,
    db_path: &Option<PathBuf>,
    load_embedder: F,
    recover_model: R,
) -> Result<CommandOutput>
where
    F: Fn() -> Result<Arc<dyn Embed>, DegradedReason>,
    R: FnOnce() -> Result<(), RecallError>,
{
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
    let mut checks = if fix {
        let empty_foreign_db = matches!(
            db::open_db_readonly(&path).map(|(conn, _)| db::schema_state(&conn)),
            Ok(Ok(db::SchemaState::Empty))
        );
        if empty_foreign_db {
            vec![empty_index_check(&path)]
        } else {
            match open_or_create_db(&path) {
                Ok(conn) => {
                    repairs = Some(run_repairs(&conn));
                    vec![
                        run_db_check("integrity", || check_integrity(&conn)),
                        run_db_check("orphan_embeddings", || check_orphan_embeddings(&conn)),
                        run_db_check("orphan_chunks", || check_orphan_chunks(&conn)),
                    ]
                }
                Err(e)
                    if e.downcast_ref::<RecallError>()
                        .is_some_and(|e| matches!(e, RecallError::DatabaseRead(_))) =>
                {
                    vec![DoctorCheck::fail(
                        "integrity",
                        e.to_string(),
                        MODEL_INDEX_REMEDY,
                    )]
                }
                Err(e) => open_failure_checks(e),
            }
        }
    } else {
        match db::open_db_readonly(&path) {
            Ok((conn, tier)) => {
                wal_note = db::stale_wal_note(&path, tier);
                match db::schema_state(&conn) {
                    Ok(db::SchemaState::Current) => vec![
                        run_db_check("integrity", || check_integrity(&conn)),
                        run_db_check("orphan_embeddings", || check_orphan_embeddings(&conn)),
                        run_db_check("orphan_chunks", || check_orphan_chunks(&conn)),
                    ],
                    Ok(db::SchemaState::Stale) => vec![DoctorCheck::fail(
                        "schema",
                        "database schema is out of date".to_owned(),
                        "run `recall rebuild`",
                    )],
                    Ok(db::SchemaState::Empty) => vec![empty_index_check(&path)],
                    Err(e) => open_failure_checks(e),
                }
            }
            Err(e) => open_failure_checks(e),
        }
    };
    let (mut model_check, model_recovery) =
        check_model_with_optional_recovery(fix, load_embedder, recover_model);
    if let Ok((conn, _)) = db::open_db_readonly(&path)
        && let Err(error) = model::check_index(&conn, model::selected())
    {
        model_check = DoctorCheck::fail(
            "model",
            error.to_string(),
            "select the matching --model-manifest and index, or build this model in a separate --db-path",
        );
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

    let mut buf = Vec::new();
    let _ = writeln!(
        buf,
        "Health: {}",
        if healthy { "healthy" } else { "degraded" }
    );
    if let Some(outcomes) = repairs {
        for o in outcomes {
            match &o.result {
                Ok(n) => {
                    let _ = writeln!(buf, "Repaired {}: {n} row(s)", o.name);
                }
                Err(e) => {
                    let _ = writeln!(buf, "Repair failed {}: {e}", o.name);
                }
            }
        }
    }
    match model_recovery {
        ModelRecovery::Succeeded => {
            let _ = writeln!(buf, "Recovered model");
        }
        ModelRecovery::Failed(e) => {
            let _ = writeln!(buf, "Model recovery failed: {e}");
        }
        ModelRecovery::Skipped => {}
    }
    for c in checks {
        let status = if c.ok { "ok" } else { "FAIL" };
        match (&c.detail, c.remedy) {
            (Some(detail), Some(remedy)) => {
                let _ = writeln!(buf, "  {} [{status}] {detail} -> {remedy}", c.name);
            }
            (Some(detail), None) => {
                let _ = writeln!(buf, "  {} [{status}] {detail}", c.name);
            }
            (None, _) => {
                let _ = writeln!(buf, "  {} [{status}]", c.name);
            }
        }
    }
    if verbose {
        let _ = writeln!(buf, "DB: {}", path.display());
    }
    let markdown = String::from_utf8_lossy(&buf).into_owned();

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

    let degraded = !healthy || matches!(model_recovery, ModelRecovery::Failed(_));
    CommandOutput::with_notes(markdown, data, degraded, notes)
}

fn extract_parent_session(file_path: &str) -> Option<&str> {
    let idx = file_path.find("/subagents/")?;
    let prefix = &file_path[..idx];
    let parent = prefix.rsplit('/').next()?;
    if parent.is_empty() {
        None
    } else {
        Some(parent)
    }
}

fn format_result(w: &mut impl Write, i: usize, r: &search::SearchResult) -> io::Result<()> {
    let s = &r.session;
    let date = format_timestamp(s.timestamp);
    let proj_name = if s.project.is_empty() {
        "unknown"
    } else {
        Path::new(&s.project)
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("unknown")
    };
    let slug = ansi::strip_control_chars(&s.slug);
    let project = ansi::strip_control_chars(&s.project);
    writeln!(
        w,
        "[{}] {} | {} | {} [{}]",
        i + 1,
        date,
        slug,
        proj_name,
        s.source
    )?;
    if !project.is_empty() {
        writeln!(w, "    {project}")?;
    }
    let session_id = ansi::strip_control_chars(&s.session_id);
    writeln!(w, "    ID: {session_id}")?;
    let file_path = ansi::strip_control_chars(&s.file_path);
    if let Some(parent) = extract_parent_session(&file_path) {
        writeln!(w, "    Parent: {parent}")?;
    }
    if !r.excerpt.is_empty() {
        let excerpt = ansi::strip_control_chars(&r.excerpt);
        for line in excerpt.trim().lines() {
            writeln!(w, "    > {line}")?;
        }
    }
    writeln!(w, "    inspect: {}", inspect_hint(&session_id, r.chunk_id))?;
    writeln!(w)?;
    Ok(())
}

fn render_results(results: &[search::SearchResult]) -> String {
    if results.is_empty() {
        return "No matching sessions found.".to_owned();
    }

    let mut buf = Vec::new();
    let _ = writeln!(buf, "Found {} sessions:\n", results.len());
    for (i, r) in results.iter().enumerate() {
        let _ = format_result(&mut buf, i, r);
    }
    String::from_utf8_lossy(&buf).into_owned()
}

fn inspect_hint(session_id: &str, chunk_id: Option<i64>) -> String {
    match chunk_id {
        Some(id) => format!("recall show {session_id} --chunk {id} --window 3"),
        None => format!("recall show {session_id}"),
    }
}

fn result_to_json(r: &search::SearchResult) -> serde_json::Value {
    let s = &r.session;
    serde_json::json!({
        "session_id": s.session_id,
        "project": s.project,
        "slug": s.slug,
        "source": s.source.to_string(),
        "timestamp": s.timestamp,
        "excerpt": r.excerpt,
        "chunk": r.chunk_id,
        "inspect": inspect_hint(&s.session_id, r.chunk_id),
    })
}

fn emit_success(out: &CommandOutput, json_mode: bool) -> io::Result<WriteOutcome> {
    emit_success_to(out, json_mode, &mut io::stdout().lock())
}

fn emit_success_to<W: Write>(
    out: &CommandOutput,
    json_mode: bool,
    w: &mut W,
) -> io::Result<WriteOutcome> {
    let body = if json_mode {
        Cow::Owned(render_json_success(out))
    } else {
        Cow::Borrowed(out.markdown.as_str())
    };
    if body.is_empty() {
        return Ok(WriteOutcome::Written);
    }
    write_result(w, &body)
}

fn emit_error(err: &anyhow::Error, json_mode: bool) {
    if json_mode {
        eprintln!("{}", render_json_error(&error_envelope(err)));
    } else {
        exit_error(&format!("{err:#}"));
    }
}

struct ClassifyOutcome {
    session_id: String,
    session_type: classify::SessionType,
    excerpt: String,
}

fn read_classification_origin(
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
struct ClassificationFileState {
    stamp: Result<(u64, SystemTime, Permissions), &'static str>,
    access: Result<(), &'static str>,
}

fn classification_file_state(path: &str) -> ClassificationFileState {
    open_classification_file(path).0
}

fn open_classification_file(path: &str) -> (ClassificationFileState, Result<File, &'static str>) {
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
enum ClassifyPhase {
    Snapshot,
    Prepared,
    WriterWaiting,
    WriterAcquired,
    WriterReleased,
    Updated,
    Retry,
}

fn reclassify_sessions(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
) -> Result<Vec<ClassifyOutcome>> {
    reclassify_sessions_with(conn, all, dry_run, read_classification_origin)
}

fn reclassify_sessions_with(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
    read_origin: impl FnMut(File, &str, &str, &str) -> Result<bool, &'static str>,
) -> Result<Vec<ClassifyOutcome>> {
    reclassify_sessions_observed(conn, all, dry_run, read_origin, |_| {})
}

fn reclassify_sessions_observed(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
    mut read_origin: impl FnMut(File, &str, &str, &str) -> Result<bool, &'static str>,
    mut observe: impl FnMut(ClassifyPhase),
) -> Result<Vec<ClassifyOutcome>> {
    use rusqlite::{ErrorCode, TransactionBehavior};
    use std::collections::BTreeMap;

    for _ in 0..3 {
        let version: i64 = conn.query_row("PRAGMA main.data_version", [], |r| r.get(0))?;
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
        observe(ClassifyPhase::Snapshot);

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
                    first_turn.trim_start().chars().take(80).collect()
                } else {
                    String::new()
                },
            });
        }
        observe(ClassifyPhase::Prepared);
        if changed {
            observe(ClassifyPhase::Retry);
            continue;
        }
        if !dry_run {
            observe(ClassifyPhase::WriterWaiting);
            let tx = match conn.transaction_with_behavior(TransactionBehavior::Immediate) {
                Ok(tx) => tx,
                Err(error) if error.sqlite_error_code() == Some(ErrorCode::DatabaseBusy) => {
                    observe(ClassifyPhase::Retry);
                    continue;
                }
                Err(error) => return Err(error.into()),
            };
            observe(ClassifyPhase::WriterAcquired);
            let current: i64 = tx.query_row("PRAGMA main.data_version", [], |r| r.get(0))?;
            if current != version
                || files
                    .iter()
                    .any(|(path, state)| classification_file_state(path) != *state)
            {
                tx.rollback()?;
                observe(ClassifyPhase::WriterReleased);
                observe(ClassifyPhase::Retry);
                continue;
            }
            {
                let mut update = tx.prepare("UPDATE sessions SET session_type = ?2 WHERE session_id = ?1 AND session_type IS NOT ?2")?;
                for o in &outcomes {
                    update.execute(rusqlite::params![o.session_id, o.session_type.as_str()])?;
                    observe(ClassifyPhase::Updated);
                }
            }
            tx.commit()?;
            observe(ClassifyPhase::WriterReleased);
        }
        for (reason, count) in unavailable {
            eprintln!(
                "warning: provenance unavailable ({reason}): {count} session(s); existing labels retained, unclassified rows use first-turn fallback"
            );
        }
        return Ok(outcomes);
    }
    Err(RecallError::TempFailure(
        "classification inputs kept changing or the database writer remained busy after 3 attempts; no classification changes applied; rerun `recall classify` with the same options".to_owned()
    ).into())
}

fn run_classify(all: bool, dry_run: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
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
                o.session_id,
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

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "index", "rebuild", "model", "search", "show", "status", "classify", "doctor", "help",
];
const GLOBAL_FLAGS: &[&str] = &["--verbose", "-v", "--json"];

fn parse_cli(args: Vec<OsString>) -> Result<Cli, ExitCode> {
    let parsed = match try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS) {
        Some(expanded) => Cli::try_parse_from(expanded),
        None => Cli::try_parse_from(args),
    };
    parsed.map_err(|e| handle_parse_error(&e))
}

fn handle_parse_error(err: &clap::Error) -> ExitCode {
    let _ = err.print();
    match err.kind() {
        ClapErrorKind::DisplayHelp => ExitCode::SUCCESS,
        _ => ExitCode::from(codes::USAGE),
    }
}

fn no_payload() -> CommandOutput {
    CommandOutput::ok(String::new(), serde_json::Value::Null)
}

fn run(cli: Cli) -> Result<CommandOutput> {
    model::select(cli.model_manifest.as_deref())
        .map_err(|e| RecallError::DatabaseRead(format!("invalid model manifest: {e}")))?;
    let default_filter = if cli.verbose {
        LOG_FILTER_VERBOSE
    } else {
        LOG_FILTER_DEFAULT
    };
    init_subscriber(default_filter);

    match cli.command {
        Some(Command::Index { embed_flags }) => run_index(
            &cli.db_path,
            embed_flags.token_budget,
            embed_flags.forward_pause_ms,
        )
        .map(|o| index_command_output(&o)),
        Some(Command::Rebuild { embed_flags }) => run_rebuild(
            &cli.db_path,
            embed_flags.token_budget,
            embed_flags.forward_pause_ms,
        )
        .map(|o| index_command_output(&o)),
        Some(Command::Model(ModelCommand::Download)) => run_model_download().map(|()| no_payload()),
        Some(Command::Model(ModelCommand::Manifest)) => {
            let data = serde_json::to_value(model::selected())?;
            Ok(CommandOutput::ok(
                serde_json::to_string_pretty(&data)?,
                data,
            ))
        }
        Some(cmd @ Command::Search { .. }) => run_search(cmd, &cli.db_path),
        Some(Command::Show {
            session_id,
            chunk,
            window,
        }) => run_show(&session_id, cli.verbose, chunk, window, &cli.db_path),
        Some(Command::Classify { all, dry_run }) => run_classify(all, dry_run, &cli.db_path),
        Some(Command::Status) => run_status(cli.verbose, &cli.db_path),
        Some(Command::Doctor { fix }) => run_doctor(cli.verbose, fix, &cli.db_path),
        None => Err(RecallError::Usage(
            "A search query is required. Usage: recall search \"query\" or recall index".to_owned(),
        )
        .into()),
    }
}

fn exit_code_for_write(write: io::Result<WriteOutcome>, json_mode: bool) -> ExitCode {
    match write {
        Ok(WriteOutcome::Written | WriteOutcome::PipeClosed) => ExitCode::SUCCESS,
        Err(e) => {
            let err = Error::new(e).context("failed to write output");
            emit_error(&err, json_mode);
            classify_exit_code(&err)
        }
    }
}

fn main() -> ExitCode {
    handle_probe_if_needed();

    let cli = match parse_cli(env::args_os().collect()) {
        Ok(cli) => cli,
        Err(code) => return code,
    };
    let json_mode = cli.json;

    match run(cli) {
        Ok(out) => exit_code_for_write(emit_success(&out, json_mode), json_mode),
        Err(e) => {
            emit_error(&e, json_mode);
            classify_exit_code(&e)
        }
    }
}

#[cfg(test)]
#[path = "main/tests.rs"]
mod tests;
