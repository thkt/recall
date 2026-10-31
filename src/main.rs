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

use std::fs::{File, OpenOptions, Permissions, create_dir_all, metadata};
use std::io::{self, ErrorKind, Write};
use std::num::ParseIntError;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;
use std::time::SystemTime;

use crate::embedding::Embed;
use crate::model::cached_artifacts;
use crate::model::handle_probe_if_needed;
use crate::model::{
    DegradedReason, EmbedderDegraded, download_and_verify_model, record_degraded,
    try_load_embedder_default_logging,
};
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
        Some(Command::Classify { all, dry_run }) => run_classify(all, dry_run, &cli.db_path),
        Some(Command::Status) => run_status(cli.verbose, &cli.db_path),
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
