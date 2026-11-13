mod ansi;
mod chunker;
mod classify;
mod commands;
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

use std::fs::{OpenOptions, create_dir_all};
use std::io::{self, ErrorKind, Write};
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use crate::embedding::Embed;
use crate::model::handle_probe_if_needed;
use crate::model::{DegradedReason, EmbedderDegraded, download_and_verify_model, record_degraded};
use crate::runtime::{
    codes, exit_error, home_dir, info as cli_info, init_subscriber, try_expand_shorthand,
};
use anyhow::{Context, Error, Result};
use clap::error::ErrorKind as ClapErrorKind;
use clap::{Args, Parser, Subcommand};
use rusqlite::Connection;

use crate::envelope::{CommandOutput, render_json_error, render_json_success};
use crate::error::{RecallError, classify_exit_code, download_error, error_envelope};
use crate::output::{WriteOutcome, write_result};

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
    Search(commands::search::SearchArgs),
    /// Show full conversation of a session
    Show {
        /// Session ID (prefix match supported)
        session_id: String,
        /// Anchor on a qa_chunk id and show only its message range plus neighbors
        #[arg(long)]
        chunk: Option<i64>,
        /// Per-side neighbor count around the anchored chunk (requires --chunk)
        #[arg(long, default_value_t = commands::show::DEFAULT_WINDOW)]
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
    /// load probe. Read-only by default; `--fix` repairs orphan embeddings,
    /// recovers the model, and re-runs the checks. Other remedies point at
    /// index/rebuild/model download.
    Doctor {
        /// Delete dangling `vec_chunks` rows whose `chunk_id` no longer resolves
        /// to a `qa_chunks` row (the orphan_embeddings fault), and download and
        /// verify the selected model (~558 MB) when the model check fails, then
        /// re-run the checks so the verdict reflects the repaired state. An index
        /// whose model identity check fails gets neither.
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

fn parse_token_budget(s: &str) -> Result<usize, ParseIntError> {
    let value: usize = s.parse()?;
    Ok(value.min(embedder::TOKEN_BUDGET_CEILING))
}

fn require_current_schema(state: db::SchemaState) -> Result<(), RecallError> {
    match state {
        db::SchemaState::Current | db::SchemaState::Empty => Ok(()),
        db::SchemaState::Stale => Err(RecallError::DataError(
            "database schema is out of date; run `recall rebuild` to migrate it".to_owned(),
        )),
    }
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
        Some(Command::Index { embed_flags }) => commands::index::index_and_report(
            &cli.db_path,
            false,
            commands::index::embed_options_from_flags(
                embed_flags.token_budget,
                embed_flags.forward_pause_ms,
            ),
        )
        .map(|o| commands::index::index_command_output(&o)),
        Some(Command::Rebuild { embed_flags }) => commands::index::index_and_report(
            &cli.db_path,
            true,
            commands::index::embed_options_from_flags(
                embed_flags.token_budget,
                embed_flags.forward_pause_ms,
            ),
        )
        .map(|o| commands::index::index_command_output(&o)),
        Some(Command::Model(ModelCommand::Download)) => {
            download_and_verify_model().map_err(|e| download_error(&e))?;
            Ok(no_payload())
        }
        Some(Command::Model(ModelCommand::Manifest)) => {
            let data = serde_json::to_value(model::selected())?;
            Ok(CommandOutput::ok(
                serde_json::to_string_pretty(&data)?,
                data,
            ))
        }
        Some(Command::Search(args)) => commands::search::run_search(args, &cli.db_path),
        Some(Command::Show {
            session_id,
            chunk,
            window,
        }) => commands::show::run_show(&session_id, cli.verbose, chunk, window, &cli.db_path),
        Some(Command::Classify { all, dry_run }) => {
            commands::classify::run_classify(all, dry_run, &cli.db_path)
        }
        Some(Command::Status) => commands::status::run_status(cli.verbose, &cli.db_path),
        Some(Command::Doctor { fix }) => {
            commands::doctor::run_doctor(cli.verbose, fix, &cli.db_path)
        }
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
