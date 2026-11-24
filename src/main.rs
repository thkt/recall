mod ansi;
mod chunker;
mod classify;
mod commands;
mod date;
mod db;
mod embedder;
mod embedding;
mod envelope;
mod error;
mod fts_query;
mod hybrid;
mod index_observer;
mod indexer;
mod model;
mod native;
mod output;
mod parser;
mod runtime;
mod search;
#[cfg(test)]
mod test_support;

use std::borrow::Cow;
use std::env;
use std::ffi::OsString;
use std::io::{self, Write};
use std::path::PathBuf;
use std::process::ExitCode;

use crate::model::{download_and_verify_model, handle_probe_if_needed};
use crate::runtime::{exit_error, init_subscriber};
use anyhow::{Error, Result};
use clap::error::ErrorKind as ClapErrorKind;
use clap::{Args, Parser, Subcommand};

use crate::envelope::{CommandOutput, render_json_error, render_json_success, render_text_notes};
use crate::error::{ErrorCode, RecallError, classify_exit_code, download_error, error_envelope};
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

    /// Database file path (default: <data dir>/recall/indexes/<model-identity>/recall.db, where
    /// <data dir> is an absolute XDG_DATA_HOME or ~/.local/share; `recall status --verbose` prints
    /// the path in use; env: RECALL_DB)
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
    /// Diagnose a broken index: corruption, orphaned rows, the model identity,
    /// and a live model load probe. Read-only by default; `--fix` repairs orphan
    /// embeddings, recovers the model, and re-runs the checks. Other remedies
    /// point at index/rebuild/model download, --model-manifest, or a separate
    /// --db-path.
    Doctor {
        /// Delete dangling `vec_chunks` rows whose `chunk_id` no longer resolves
        /// to a `qa_chunks` row (the orphan_embeddings fault), and download and
        /// verify the selected model (~558 MB for the default) when it is missing
        /// or fails its check, then re-run the checks so the verdict reflects the
        /// repaired state. An index whose model identity check fails gets neither.
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

#[derive(Debug, Args)]
struct EmbedFlags {
    /// Forward-pass token budget override; smaller values shorten each GPU
    /// forward so the host stays responsive during large embed runs. Gemma
    /// packing is capped at 8192 tokens; one attention window remains atomic.
    #[arg(long, env = "RECALL_TOKEN_BUDGET")]
    token_budget: Option<usize>,

    /// Pause (in milliseconds) after each GPU forward pass, yielding the GPU
    /// to interactive processes; trades throughput for host responsiveness
    /// during large embed runs.
    #[arg(long, env = "RECALL_FORWARD_PAUSE_MS")]
    forward_pause_ms: Option<u64>,
}

fn emit_success(out: &CommandOutput, json_mode: bool) -> io::Result<WriteOutcome> {
    emit_success_to(
        out,
        json_mode,
        &mut io::stdout().lock(),
        &mut io::stderr().lock(),
    )
}

fn emit_success_to<W: Write, E: Write>(
    out: &CommandOutput,
    json_mode: bool,
    w: &mut W,
    notes: &mut E,
) -> io::Result<WriteOutcome> {
    let body = if json_mode {
        Cow::Owned(render_json_success(out))
    } else {
        Cow::Borrowed(out.markdown.as_str())
    };
    let written = if body.is_empty() {
        Ok(WriteOutcome::Written)
    } else {
        write_result(w, &body)
    };
    if !json_mode && written.is_ok() {
        let _ = notes.write_all(render_text_notes(out).as_bytes());
    }
    written
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

fn try_expand_shorthand(args: &[OsString]) -> Option<Vec<OsString>> {
    let mut i = 1;
    while i < args.len() {
        let s = args[i].to_str()?;
        if matches!(s, "--db-path" | "--model-manifest") {
            i += 2;
            continue;
        }
        if s.starts_with("--db-path=")
            || s.starts_with("--model-manifest=")
            || GLOBAL_FLAGS.contains(&s)
        {
            i += 1;
            continue;
        }
        if s == "--help" || s == "-h" || KNOWN_SUBCOMMANDS.contains(&s) {
            return None;
        }
        if s.starts_with('-') {
            return None;
        }
        let mut result = args.to_vec();
        result.insert(i, "search".into());
        return Some(result);
    }
    None
}

fn parse_cli(args: Vec<OsString>) -> Result<Cli, ExitCode> {
    let parsed = match try_expand_shorthand(&args) {
        Some(expanded) => Cli::try_parse_from(expanded),
        None => Cli::try_parse_from(args),
    };
    parsed.map_err(|e| handle_parse_error(&e))
}

fn handle_parse_error(err: &clap::Error) -> ExitCode {
    let _ = err.print();
    match err.kind() {
        ClapErrorKind::DisplayHelp => ExitCode::SUCCESS,
        _ => ExitCode::from(ErrorCode::UsageError.code()),
    }
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
            Ok(CommandOutput::ok(String::new(), serde_json::Value::Null))
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
