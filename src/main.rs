mod ansi;
mod chunker;
mod classify;
mod date;
mod db;
mod embedder;
mod envelope;
mod error;
mod hybrid;
mod indexer;
mod output;
mod parser;
mod search;

use std::env;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::Arc;

use amici::cli::exit_code::codes;
use amici::cli::{Spinner, done, exit_error, info as cli_info, try_expand_shorthand, warning};
use amici::logging::init_subscriber;
use amici::model::embedder::{DegradedReason, try_load_embedder_default_logging};
use amici::model::{EmbedderDegraded, download_and_verify_model, record_degraded};
use amici::storage::{collect_rows, filter::escape_like};
use anyhow::{Context, Error, Result};
use clap::error::ErrorKind as ClapErrorKind;
use clap::{Parser, Subcommand};
use rurico::embed::{Embed, ModelId, cached_artifacts};
use rurico::handle_probe_if_needed;
use rusqlite::Connection;
use tracing::warn;

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

    /// Database file path (default: ~/.recall.db, env: RECALL_DB)
    #[arg(long, env = "RECALL_DB", global = true)]
    db_path: Option<PathBuf>,

    /// Emit a machine-readable JSON envelope instead of human-readable text
    #[arg(long, global = true)]
    json: bool,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Parse and chunk new session logs (incremental). No model calls.
    Index,
    /// Re-parse and re-embed every present session; a missing or unreadable
    /// source root keeps its existing index. No model calls.
    Rebuild,
    /// Manage the embedding model.
    #[command(subcommand)]
    Model(ModelCommand),
    /// Search indexed sessions (default when query is provided)
    Search {
        /// Search query
        query: String,

        /// Filter by project path (prefix match)
        #[arg(long)]
        project: Option<String>,

        /// Only sessions from last N days
        #[arg(long, value_parser = clap::value_parser!(i64).range(1..))]
        days: Option<i64>,

        /// Filter by source
        #[arg(long, value_enum)]
        source: Option<Source>,

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
    },
    /// Classify existing sessions interactive/automated from their first user turn
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
}

#[derive(Debug, Subcommand)]
enum ModelCommand {
    /// Download the embedding model and verify it loads.
    Download,
}

fn create_db_file(path: &Path) -> io::Result<()> {
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
    db::open_db(path).context("Failed to open database")
}

fn resolve_db_path(db_path: &Option<PathBuf>) -> Result<PathBuf> {
    match db_path {
        Some(p) => Ok(p.clone()),
        None => Ok(dirs::home_dir()
            .context("Could not determine home directory")?
            .join(".recall.db")),
    }
}

fn open_cached_embedder() -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_default_logging()
}

fn try_load_embedder_cached() -> Result<Arc<dyn Embed>, DegradedReason> {
    try_load_embedder_cached_with(open_cached_embedder)
}

fn try_load_embedder_cached_with<F>(open_embedder: F) -> Result<Arc<dyn Embed>, DegradedReason>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    try_load_embedder_cached_with_reporter(open_embedder, cli_info)
}

/// Load the cached embedder, returning the [`DegradedReason`] on failure so the
/// caller can build a reason-aware `--json` note. The human-facing stderr note
/// (via `report_info`) and the structured warn (`record_degraded`) are emitted
/// here as side effects; the reason itself is surfaced to the caller rather than
/// flattened to `None`, so the agent channel can say *why* search degraded.
fn try_load_embedder_cached_with_reporter<F, I>(
    open_embedder: F,
    mut report_info: I,
) -> Result<Arc<dyn Embed>, DegradedReason>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
    I: FnMut(&str),
{
    match open_embedder() {
        Ok(e) => Ok(e),
        Err(reason @ DegradedReason::NotInstalled) => {
            if let Some(note) = search_degraded_note(reason) {
                report_info(&note);
            }
            Err(reason)
        }
        Err(reason) => {
            if let Some(note) = search_degraded_note(reason) {
                report_info(&note);
            }
            record_degraded(reason, "embedder load");
            Err(reason)
        }
    }
}

/// The human-facing stderr note for a degraded embedder load (`note: ...`).
fn search_degraded_note(reason: DegradedReason) -> Option<String> {
    EmbedderDegraded(reason)
        .user_note("recall model download")
        .map(|note| format!("note: {note}."))
}

/// The `--json` `notes[]` entry for a search that fell back to FTS-only, derived
/// from the same per-reason source as [`search_degraded_note`] so the agent and
/// human channels never disagree. `NotInstalled` keeps the actionable download
/// hint; a probe or backend failure reports unavailability without it, because
/// re-downloading cannot repair a corrupt cache or a missing backend. An empty
/// list means no degradation note applies (a caller-disabled model), so the
/// caller's `degraded` flag co-varies with it.
fn search_fallback_notes(reason: DegradedReason) -> Vec<String> {
    EmbedderDegraded(reason)
        .user_note("recall model download")
        .map(|note| vec![format!("semantic search unavailable: {note}")])
        .unwrap_or_default()
}

/// Derive the `(degraded, notes)` pair for a search from the embedder load
/// result. A loaded embedder is the ideal path (not degraded, no notes); a
/// failure degrades to FTS-only with a reason-aware note from
/// [`search_fallback_notes`]. `degraded` co-varies with the note (an empty list
/// keeps it false), so the bool and the text never disagree.
fn search_degraded_state(load: &Result<Arc<dyn Embed>, DegradedReason>) -> (bool, Vec<String>) {
    match load {
        Ok(_) => (false, Vec::new()),
        Err(reason) => {
            let notes = search_fallback_notes(*reason);
            (!notes.is_empty(), notes)
        }
    }
}

// -- Subcommands --

fn run_index(db_path: &Option<PathBuf>) -> Result<IndexOutcome> {
    index_and_report(db_path, false)
}

fn run_rebuild(db_path: &Option<PathBuf>) -> Result<IndexOutcome> {
    index_and_report(db_path, true)
}

/// Shared index pipeline for `index` (incremental) and `rebuild` (full).
/// Resolves source dirs from the environment, then delegates to
/// [`index_and_report_with`] with the real cached-embedder loader.
fn index_and_report(db_path: &Option<PathBuf>, force: bool) -> Result<IndexOutcome> {
    let home = dirs::home_dir().context("Could not determine home directory")?;
    let claude_dir = indexer::resolve_claude_dir(&home);
    let codex_dir = indexer::resolve_codex_dir(&home);
    let opts = indexer::IndexOptions {
        force,
        claude_dir: &claude_dir,
        codex_dir: &codex_dir,
    };
    // Load silently: IndexOutcome carries the degraded note and the block below is
    // its single emitter, so passing cli_info to the loader too would print twice.
    let outcome = index_and_report_with(db_path, &opts, || {
        try_load_embedder_cached_with_reporter(open_cached_embedder, |_: &str| {})
    })?;
    if let Some(note) = &outcome.degraded_note {
        cli_info(note);
    }
    Ok(outcome)
}

/// Outcome of an index run, surfaced to the caller for the `--json` envelope.
/// `degraded_note` is `Some` when the embedder could not load (model-less): the
/// index still completes FTS-only. `embedded`/`failed_count`/`first_error` report
/// the embed pass: `failed_count > 0` means some batches failed but the index is
/// complete and those chunks stay pending for the next run. The two degradation
/// reasons are mutually exclusive (a model-less run skips embedding, so
/// `failed_count == 0`).
#[derive(Default)]
struct IndexOutcome {
    degraded_note: Option<String>,
    embedded: usize,
    failed_count: usize,
    first_error: Option<String>,
    /// Source roots that were missing/unreadable this run while still holding
    /// preserved sessions in the DB (#165). Carried into the `--json` envelope so
    /// an agent reading the machine path — not just the human stderr warning —
    /// learns a root outage may be masking real deletions.
    skipped_roots: Vec<indexer::SkippedRoot>,
}

/// Build the `--json` envelope for an index run from its [`IndexOutcome`],
/// mirroring [`search_degraded_state`]: both degradation reasons — a model-less
/// load (`degraded_note`) and an embed batch failure (`failed_count`) — become
/// `notes`, `degraded` co-varies (true iff `notes` is non-empty), and
/// `{ embedded, failed_count }` is the data so an agent reads the magnitude, not
/// just a boolean.
/// Cap on the `first_error` portion of the embed-stop note (chars, not bytes, so
/// truncation never splits a UTF-8 boundary).
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
        // first_error is always Some when failed_count > 0 (both are set in
        // embed_chunks' Err arm together); the fallback is defensive only. Like
        // every other envelope-bound string the error is control-char-stripped,
        // and capped so a verbose MLX error cannot bloat the note.
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
    let data = serde_json::json!({
        "embedded": outcome.embedded,
        "failed_count": outcome.failed_count,
        "skipped_roots": skipped_roots,
    });
    CommandOutput::with_notes(String::new(), data, degraded, notes)
}

/// Index pipeline with an injected embedder loader. Runs FTS indexing + chunking,
/// then embeds every pending chunk via the loaded embedder so `recall search`
/// reads a complete index without embedding on the search path. A model-less
/// loader (`Err`) skips embedding but keeps the FTS index complete, returning a
/// degraded note. `load_embedder` is a seam: tests inject a `MockEmbedder` or a
/// degraded `Err`, production passes [`try_load_embedder_cached`]. Source dirs
/// arrive via `opts` so tests control them without `set_var`.
fn index_and_report_with<F>(
    db_path: &Option<PathBuf>,
    opts: &indexer::IndexOptions,
    load_embedder: F,
) -> Result<IndexOutcome>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    let path = resolve_db_path(db_path)?;
    let mut conn = open_or_create_db(&path)?;

    let sp = Spinner::new("Indexing sessions...");
    let stats = indexer::index_from_dirs(&mut conn, opts)?;
    let main_msg = if stats.indexed > 0 {
        format!(
            "Indexed {} sessions in {:.1}s",
            stats.indexed, stats.elapsed_secs
        )
    } else {
        format!("{} sessions up to date", stats.total_sessions)
    };
    sp.finish_with_detail(&main_msg, stats.parse_error_detail().as_deref());
    for skipped in &stats.skipped_roots {
        warning(
            &skipped
                .reason
                .note(skipped.source.as_str(), skipped.preserved_sessions),
        );
    }
    let skipped_roots = stats.skipped_roots;

    let sp = Spinner::new("Creating chunks...");
    let on_progress = |done: usize, total: usize| {
        sp.set_message(&format!("Creating chunks... {done}/{total} sessions"));
    };
    let chunk_stats = indexer::index_chunks(&mut conn, Some(&on_progress))?;
    if chunk_stats.chunks_created > 0 {
        sp.finish(&format!("Created {} chunks", chunk_stats.chunks_created));
    } else {
        sp.finish("Chunks up to date");
    }

    // Model present: embed all pending. Model absent (degraded): skip embed but
    // keep the completed FTS index and carry a note naming `recall model download`
    // so the wrapper can guide the user (FR-004a/004b).
    match load_embedder() {
        Ok(embedder) => {
            let result = embed_all_pending(&mut conn, embedder.as_ref())?;
            Ok(IndexOutcome {
                degraded_note: None,
                embedded: result.embedded,
                failed_count: result.failed_count,
                first_error: result.first_error,
                skipped_roots,
            })
        }
        Err(reason) => Ok(IndexOutcome {
            degraded_note: search_degraded_note(reason),
            skipped_roots,
            ..IndexOutcome::default()
        }),
    }
}

/// Embed every chunk absent from `vec_chunks`. Incremental by the one-pass
/// pending gate (`embedder::pending_chunks`), so a re-index embeds only new
/// chunks; a `rebuild` re-embeds every present session because it re-parses each
/// one and replaces its chunks, leaving them pending. The pending list is
/// collected once and fed straight to
/// `embed_chunks` — a separate COUNT would re-scan vec_chunks (#138).
fn embed_all_pending(conn: &mut Connection, embedder: &dyn Embed) -> Result<embedder::EmbedResult> {
    let pending = embedder::pending_chunks(conn, usize::MAX)?;
    if pending.is_empty() {
        return Ok(embedder::EmbedResult::default());
    }
    let sp = Spinner::new("Embedding chunks...");
    let result = embedder::embed_chunks(
        conn,
        embedder,
        &pending,
        Some(&|done, total| sp.set_message(&format!("Embedding chunks... {done}/{total}"))),
    )?;
    sp.finish(&format!("Embedded {} chunks", result.embedded));
    // A mid-run batch failure is non-fatal: chunks already embedded are committed,
    // the rest stay pending, and the FTS index is complete and queryable. The next
    // `recall index` retries the remaining pending via the pending gate, so we
    // warn rather than fail. The stop is also surfaced in the index `--json`
    // envelope via `failed_count` (index_command_output).
    result.warn_if_batches_failed();
    Ok(result)
}

fn run_model_download() -> Result<()> {
    download_and_verify_model().map_err(|e| download_error(&e).into())
}

/// Guidance when search runs against an empty index. `recall search` no longer
/// auto-indexes, so zero sessions means the user must run `recall index` first.
fn search_idle_message(session_count: i64) -> Option<&'static str> {
    (session_count == 0).then_some("No sessions indexed. Run `recall index` first.")
}

/// The three mutually exclusive `--*-current` flags, grouped so
/// `resolve_current_session` takes one argument instead of three booleans.
#[derive(Clone, Copy, Debug)]
struct CurrentSessionFlags {
    exclude: bool,
    include: bool,
    only: bool,
}

/// Resolve how `recall search` treats the invoking session, from the
/// `--*-current` flags and the `CLAUDE_CODE_SESSION_ID` env value (read by the
/// caller and passed in, so the truth table is testable without touching process
/// env). With no flag, an env id present defaults to excluding the current
/// session: an agent searching mid-session does not want its own in-progress
/// transcript echoed back. A blank (empty or whitespace-only) env value is
/// treated as absent, so `--only-current` still reports the usage error instead
/// of silently matching no session.
fn resolve_current_session(
    flags: CurrentSessionFlags,
    env_session_id: Option<String>,
) -> Result<search::CurrentSession, RecallError> {
    use search::CurrentSession;
    // A set-but-blank env value (e.g. exported empty in a shell) is not a usable
    // session id; treat it as absent so every mode below behaves consistently.
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

/// Search pipeline with an injected embedder loader. Reads the pre-built index
/// (no indexing, no embedding on the search path — embedding happens at `recall
/// index` time). `load_embedder` is a seam so tests can inject a loaded or
/// degraded embedder and exercise the FTS-fallback path. A failed load degrades
/// to FTS-only.
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
    let conn = open_or_create_db(&path)?;

    // Search reads the pre-built index; indexing happens via `recall index`,
    // not on every search, so search latency no longer depends on scan cost.
    let session_count: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    if let Some(msg) = search_idle_message(session_count) {
        done(msg);
        return Ok(CommandOutput::ok(
            String::new(),
            serde_json::json!({ "results": [] }),
        ));
    }

    let embedder = load_embedder();
    // A failed embedder load degrades search to FTS-only with a reason-aware note
    // for the `--json` channel; see search_degraded_state.
    let (degraded, mut notes) = search_degraded_state(&embedder);
    let embedder = embedder.ok();

    let results = search::search_with_embedder(
        &conn,
        &query,
        &search::SearchOptions {
            project,
            days,
            source,
            limit: limit.into(),
            current,
            include_automated,
            now_ms: None,
        },
        embedder.as_deref(),
    )?;

    // Surface the default automated-session exclusion so an agent consumer can
    // discover --include-automated; skipped when the agent already opted in.
    if !include_automated {
        // Best-effort: a failed count skips the advisory note rather than aborting
        // an otherwise-successful search.
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

    // Search is read-only: embedding happens at `recall index` time, never on the
    // search path, so search latency does not depend on embedding cost.
    Ok(CommandOutput::with_notes(markdown, data, degraded, notes))
}

fn show_session(conn: &Connection, session_id: &str, verbose: bool) -> Result<CommandOutput> {
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
    let matches: Vec<parser::SessionData> = collect_rows::<_, _, _, Error>(rows)?;

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

    let session = &matches[0];

    let mut msg_stmt =
        conn.prepare("SELECT role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let rows = msg_stmt.query_map([&session.session_id], |row| Ok((row.get(0)?, row.get(1)?)))?;
    let messages: Vec<(String, String)> = collect_rows::<_, _, _, Error>(rows)?;

    // Writes to an in-memory Vec are infallible; the pipe is only touched later
    // by the single emit_success -> emit_success_to write path.
    let mut buf = Vec::new();
    let _ = writeln!(buf, "# {}", session.slug);
    let _ = writeln!(buf, "- session_id: {}", session.session_id);
    let _ = writeln!(buf, "- date: {}", format_timestamp(session.timestamp));
    let _ = writeln!(buf, "- project: {}", session.project);
    let _ = writeln!(buf, "- source: {}", session.source);
    if verbose {
        let _ = writeln!(buf, "- file: {}", session.file_path);
    }
    let _ = writeln!(buf);
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

fn run_show(session_id: &str, verbose: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        return Err(
            RecallError::Usage("Database not found. Run `recall index` first.".to_owned()).into(),
        );
    }

    let conn = open_or_create_db(&path)?;
    show_session(&conn, session_id, verbose)
}

fn run_status(verbose: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        cli_info(&format!("database not found at {}", path.display()));
        cli_info("run `recall index` to create the index.");
        return Ok(CommandOutput::ok(
            String::new(),
            serde_json::json!({
                "sessions": 0,
                "qa_chunks": 0,
                "embedded": 0,
                "model_ready": false,
            }),
        ));
    }

    let conn = open_or_create_db(&path)?;

    let sessions: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))?;
    let embedded: i64 = conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))?;

    let model_ok = cached_artifacts(ModelId::DEFAULT)
        .map(|opt| opt.is_some())
        .unwrap_or(false);

    // Writes to an in-memory Vec are infallible, so the io::Result is discarded.
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

    Ok(CommandOutput::ok(markdown, data))
}

// -- Output formatting --

/// Path structure: `{...}/{parent-session-uuid}/subagents/{agent-id}.jsonl`
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
    writeln!(w)?;
    Ok(())
}

/// Render search results into the human-readable listing: empty results yield
/// the no-match line, otherwise a header plus one block per result. Writes go
/// to an in-memory buffer (infallible), so the result emits as a single write.
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

/// The machine payload for one search hit (#67 Phase 2), built explicitly so the
/// `--json` schema stays decoupled from [`search::SearchResult`]'s internal
/// shape (which is deliberately not `Serialize`).
fn result_to_json(r: &search::SearchResult) -> serde_json::Value {
    let s = &r.session;
    serde_json::json!({
        "session_id": s.session_id,
        "project": s.project,
        "slug": s.slug,
        "source": s.source.to_string(),
        "timestamp": s.timestamp,
        "excerpt": r.excerpt,
    })
}

fn render_success_body(out: &CommandOutput, json_mode: bool) -> String {
    if json_mode {
        render_json_success(out)
    } else {
        out.markdown.clone()
    }
}

/// Emit a command's result to stdout: the JSON success envelope when
/// `json_mode`, else the human-readable markdown. Delegates to
/// [`emit_success_to`] with a locked stdout so production and tests share one
/// write path (and the single SIGPIPE boundary).
fn emit_success(out: &CommandOutput, json_mode: bool) -> io::Result<WriteOutcome> {
    emit_success_to(out, json_mode, &mut io::stdout().lock())
}

/// Write a command's result to `w`: the JSON success envelope when `json_mode`,
/// else the human-readable markdown. Empty markdown (a side-effect command in
/// text mode) writes nothing. A closed pipe is a clean stop
/// ([`WriteOutcome::PipeClosed`]); any other write error propagates to the CLI
/// boundary for a non-zero I/O exit code. See [`crate::output`].
fn emit_success_to<W: Write>(
    out: &CommandOutput,
    json_mode: bool,
    w: &mut W,
) -> io::Result<WriteOutcome> {
    let body = render_success_body(out, json_mode);
    if body.is_empty() {
        return Ok(WriteOutcome::Written);
    }
    write_result(w, &body)
}

/// Emit an error to stderr: the JSON error envelope when `json_mode`, else the
/// human-readable `error: <msg>` line. The exit code is decided separately by
/// [`classify_exit_code`], so the rendered shape and the exit code never
/// disagree.
fn emit_error(err: &anyhow::Error, json_mode: bool) {
    if json_mode {
        eprintln!("{}", render_json_error(&error_envelope(err)));
    } else {
        exit_error(&format!("{err:#}"));
    }
}

/// One session's reclassification result (#24 Phase 3).
struct ClassifyOutcome {
    session_id: String,
    session_type: classify::SessionType,
    /// First ~80 chars of the first user turn (after trim), for `--dry-run` display.
    excerpt: String,
}

/// Re-classify existing sessions from their first user turn and update
/// `session_type`. With `all`, every session that has a user turn is re-evaluated;
/// otherwise only those still unclassified (NULL). When `dry_run`, nothing is
/// written — the caller displays the returned outcomes. Sessions without a user
/// turn are left untouched (NULL = interactive).
fn reclassify_sessions(
    conn: &mut Connection,
    all: bool,
    dry_run: bool,
) -> Result<Vec<ClassifyOutcome>> {
    // First user turn per session: the lowest-rowid user message (document order,
    // AS-003). Without `all`, restrict to still-unclassified sessions.
    let base = "SELECT m.session_id, m.text FROM messages m \
        WHERE m.rowid IN (SELECT MIN(rowid) FROM messages WHERE role = 'user' GROUP BY session_id)";
    let sql = if all {
        base.to_owned()
    } else {
        format!(
            "{base} AND m.session_id IN (SELECT session_id FROM sessions WHERE session_type IS NULL)"
        )
    };

    let outcomes = {
        let mut stmt = conn.prepare(&sql)?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut outcomes = Vec::new();
        for row in rows {
            let (session_id, first_turn) = row?;
            let session_type = classify::classify_first_turn(&first_turn);
            // The excerpt is for --dry-run display only; skip the allocation on the
            // write path where it is never read.
            let excerpt = if dry_run {
                first_turn.trim_start().chars().take(80).collect()
            } else {
                String::new()
            };
            outcomes.push(ClassifyOutcome {
                session_id,
                session_type,
                excerpt,
            });
        }
        outcomes
    };

    if !dry_run {
        // One transaction for the whole batch: a single commit instead of one per
        // row, and all-or-nothing so an interrupted run leaves session_type
        // unchanged and is cleanly re-runnable.
        let tx = conn.transaction()?;
        {
            let mut update =
                tx.prepare("UPDATE sessions SET session_type = ?2 WHERE session_id = ?1")?;
            for o in &outcomes {
                update.execute(rusqlite::params![o.session_id, o.session_type.as_str()])?;
            }
        }
        tx.commit()?;
    }

    Ok(outcomes)
}

/// `recall classify`: tag existing sessions interactive/automated from their first
/// user turn (#24 Phase 3). `--all` re-classifies every session; otherwise only
/// unclassified ones. `--dry-run` reports what would change without writing.
fn run_classify(all: bool, dry_run: bool, db_path: &Option<PathBuf>) -> Result<CommandOutput> {
    let path = resolve_db_path(db_path)?;
    let mut conn = open_or_create_db(&path)?;
    let outcomes = reclassify_sessions(&mut conn, all, dry_run)?;
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

// -- Entry point --

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "index", "rebuild", "model", "search", "show", "status", "classify", "help",
];
const GLOBAL_FLAGS: &[&str] = &["--verbose", "-v", "--json"];

/// Parse argv into a [`Cli`], expanding shorthand first. Both expansion
/// branches funnel through one `try_parse_from`, so a parse error prints once.
/// Returns the exit code to use when parsing does not yield a command:
/// success for `--help`, `USAGE` (64) for a real parse error.
fn parse_cli(args: Vec<OsString>) -> Result<Cli, ExitCode> {
    let parsed = match try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS) {
        Some(expanded) => Cli::try_parse_from(expanded),
        None => Cli::try_parse_from(args),
    };
    parsed.map_err(|e| handle_parse_error(&e))
}

/// Print a clap parse error (or `--help` text) and map it to an exit code.
/// `--help` is not a failure (clap writes it to stdout); everything else is a
/// usage error. recall defines no `--version`, so it falls through to `USAGE`
/// like any other unknown flag.
fn handle_parse_error(err: &clap::Error) -> ExitCode {
    let _ = err.print();
    match err.kind() {
        ClapErrorKind::DisplayHelp => ExitCode::SUCCESS,
        _ => ExitCode::from(codes::USAGE),
    }
}

/// The [`CommandOutput`] for a side-effect command (index/embed/model): empty
/// `markdown` and `null` `data`, since its progress already went to stderr. In
/// `--json` mode [`emit_success`] still emits a `data: null` success envelope so
/// an agent gets a parseable signal; in text mode it prints nothing.
fn no_payload() -> CommandOutput {
    CommandOutput::ok(String::new(), serde_json::Value::Null)
}

fn run(cli: Cli) -> Result<CommandOutput> {
    let default_filter = if cli.verbose {
        LOG_FILTER_VERBOSE
    } else {
        LOG_FILTER_DEFAULT
    };
    init_subscriber(default_filter);

    match cli.command {
        Some(Command::Index) => run_index(&cli.db_path).map(|o| index_command_output(&o)),
        Some(Command::Rebuild) => run_rebuild(&cli.db_path).map(|o| index_command_output(&o)),
        Some(Command::Model(ModelCommand::Download)) => run_model_download().map(|()| no_payload()),
        Some(cmd @ Command::Search { .. }) => run_search(cmd, &cli.db_path),
        Some(Command::Show { session_id }) => run_show(&session_id, cli.verbose, &cli.db_path),
        Some(Command::Classify { all, dry_run }) => run_classify(all, dry_run, &cli.db_path),
        Some(Command::Status) => run_status(cli.verbose, &cli.db_path),
        None => Err(RecallError::Usage(
            "A search query is required. Usage: recall search \"query\" or recall index".to_owned(),
        )
        .into()),
    }
}

/// Map the stdout-write outcome of a successful command to an exit code: a
/// written result or a consumer that closed the pipe early is success; any other
/// write failure propagates to a non-zero I/O exit code. The error renders to
/// stderr (never to the already-failing stdout). Split from [`main`] so the
/// write-failure arm is unit-testable without a real broken stdout.
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
mod tests {
    use std::assert_matches;
    use std::fs;

    use super::*;

    // T-CS001: resolve_current_session is the single place the env-exclude policy
    // lives, so this (flags x env) truth table is the contract.
    #[test]
    fn resolve_current_session_truth_table() {
        use search::CurrentSession;
        let flags = |exclude, include, only| CurrentSessionFlags {
            exclude,
            include,
            only,
        };
        let with_env = || Some("sess-x".to_owned());
        let exclude_x = || CurrentSession::Exclude("sess-x".to_owned());

        // no flag: env present → default-exclude; env absent → ignore
        assert_eq!(
            resolve_current_session(flags(false, false, false), with_env()).unwrap(),
            exclude_x()
        );
        assert_eq!(
            resolve_current_session(flags(false, false, false), None).unwrap(),
            CurrentSession::Ignore
        );
        // --exclude-current: env present → exclude; env absent → ignore (warns, no-op)
        assert_eq!(
            resolve_current_session(flags(true, false, false), with_env()).unwrap(),
            exclude_x()
        );
        assert_eq!(
            resolve_current_session(flags(true, false, false), None).unwrap(),
            CurrentSession::Ignore
        );
        // --include-current: ignore regardless of env
        assert_eq!(
            resolve_current_session(flags(false, true, false), with_env()).unwrap(),
            CurrentSession::Ignore
        );
        assert_eq!(
            resolve_current_session(flags(false, true, false), None).unwrap(),
            CurrentSession::Ignore
        );
        // --only-current: env present → only; env absent → usage error
        assert_eq!(
            resolve_current_session(flags(false, false, true), with_env()).unwrap(),
            CurrentSession::Only("sess-x".to_owned())
        );
        assert!(
            resolve_current_session(flags(false, false, true), None).is_err(),
            "--only-current without CLAUDE_CODE_SESSION_ID must be a usage error"
        );
        // a blank (empty/whitespace) env value is treated as absent
        assert_eq!(
            resolve_current_session(flags(false, false, false), Some("  ".to_owned())).unwrap(),
            CurrentSession::Ignore,
            "blank CLAUDE_CODE_SESSION_ID must not produce a degenerate Exclude"
        );
        assert!(
            resolve_current_session(flags(false, false, true), Some(String::new())).is_err(),
            "--only-current with a blank env value must still be a usage error"
        );
    }

    #[test]
    fn test_format_timestamp() {
        for (input, expected) in [
            (None, "unknown"),
            (Some(0), "1970-01-01"),
            (Some(1709251200000), "2024-03-01"),
            (Some(-date::MS_PER_DAY), "1969-12-31"),
        ] {
            assert_eq!(format_timestamp(input), expected, "input: {input:?}");
        }
    }

    #[test]
    fn try_load_embedder_cached_records_degraded_non_install_failure() {
        let result = try_load_embedder_cached_with(|| Err(DegradedReason::ProbeFailed));
        assert!(result.is_err());
    }

    #[test]
    fn try_load_embedder_cached_reports_non_install_degraded_note() {
        let mut notes = Vec::new();
        let result = try_load_embedder_cached_with_reporter(
            || Err(DegradedReason::ProbeFailed),
            |msg| notes.push(msg.to_owned()),
        );

        assert!(result.is_err());
        assert_eq!(
            notes,
            ["note: embedding model unavailable; results from text search only."]
        );
    }

    #[test]
    fn search_degraded_note_covers_backend_failures() {
        assert_eq!(
            search_degraded_note(DegradedReason::BackendUnavailable).as_deref(),
            Some("note: embedding model unavailable; results from text search only.")
        );
        assert_eq!(search_degraded_note(DegradedReason::Disabled), None);
    }

    // The `--json` agent channel carries a reason-aware fallback note, not a fixed
    // "not installed" string: NotInstalled keeps the download hint, a probe/backend
    // failure reports plain unavailability (re-downloading cannot fix either), and a
    // disabled model yields no note so `degraded` stays false. These strings are
    // recall's public `--json` surface, so they are pinned exactly.
    #[test]
    fn search_fallback_notes_are_reason_aware() {
        assert_eq!(
            search_fallback_notes(DegradedReason::NotInstalled),
            [
                "semantic search unavailable: embedding model not installed; run `recall model download` to enable semantic search"
            ]
        );
        assert_eq!(
            search_fallback_notes(DegradedReason::ProbeFailed),
            [
                "semantic search unavailable: embedding model unavailable; results from text search only"
            ]
        );
        assert_eq!(
            search_fallback_notes(DegradedReason::BackendUnavailable),
            [
                "semantic search unavailable: embedding model unavailable; results from text search only"
            ]
        );
        assert!(
            search_fallback_notes(DegradedReason::Disabled).is_empty(),
            "a disabled model is intentional, not degraded: no note"
        );
    }

    // search_degraded_state Ok arm: a loaded embedder is the ideal path. Driven by
    // a MockEmbedder because CI has no real model, so run_search itself only
    // exercises the Err arm — this is the sole coverage of the not-degraded branch.
    #[test]
    fn search_degraded_state_loaded_embedder_is_not_degraded() {
        let loaded: Arc<dyn Embed> = Arc::new(embedder::MockEmbedder::new());
        let (degraded, notes) = search_degraded_state(&Ok(loaded));
        assert!(!degraded, "a loaded embedder must not be marked degraded");
        assert!(notes.is_empty(), "the ideal path carries no notes");
    }

    // search_degraded_state Err arm: a load failure degrades with a reason-aware
    // note, and degraded co-varies with that note.
    #[test]
    fn search_degraded_state_reason_failure_degrades_with_note() {
        let (degraded, notes) = search_degraded_state(&Err(DegradedReason::ProbeFailed));
        assert!(degraded);
        assert_eq!(
            notes,
            [
                "semantic search unavailable: embedding model unavailable; results from text search only"
            ]
        );
    }

    // A caller-disabled model yields no note, so degraded stays false: the
    // invariant degraded == !notes.is_empty() holds even on the Err path.
    #[test]
    fn search_degraded_state_disabled_is_not_degraded() {
        let (degraded, notes) = search_degraded_state(&Err(DegradedReason::Disabled));
        assert!(!degraded, "a disabled model is intentional, not degraded");
        assert!(notes.is_empty());
    }

    // The Ok arm: a loaded embedder is returned so search can rank semantically.
    // Injected via the loader seam to stay independent of whether a model is
    // installed in the test environment (production coverage of this arm would
    // otherwise flip with model presence).
    #[test]
    fn try_load_embedder_cached_returns_loaded_embedder() {
        let loaded: Arc<dyn Embed> = Arc::new(embedder::MockEmbedder::new());
        let result = try_load_embedder_cached_with(|| Ok(loaded));
        assert!(result.is_ok(), "a loaded embedder must be returned");
    }

    // The NotInstalled arm: a missing model degrades search to FTS-only (Err,
    // surfaced as None at the call site) after the install note, rather than
    // aborting the command.
    #[test]
    fn try_load_embedder_cached_returns_err_when_model_not_installed() {
        let result = try_load_embedder_cached_with(|| Err(DegradedReason::NotInstalled));
        assert!(
            result.is_err(),
            "a not-installed model must degrade (Err), not abort"
        );
    }

    fn make_search_result(session_id: &str, project: &str, excerpt: &str) -> search::SearchResult {
        search::SearchResult {
            session: parser::SessionData {
                session_id: session_id.to_owned(),
                source: parser::Source::Claude,
                file_path: "/path/to/file.jsonl".to_owned(),
                project: project.to_owned(),
                slug: "test-slug".to_owned(),
                timestamp: Some(1709251200000),
            },
            excerpt: excerpt.to_owned(),
        }
    }

    #[test]
    fn test_format_result_basic() {
        let r = make_search_result("s1", "/home/me/project", "some excerpt");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("[1] 2024-03-01"));
        assert!(output.contains("test-slug"));
        assert!(output.contains("project [claude]"));
        assert!(output.contains("ID: s1"));
        assert!(output.contains("> some excerpt"));
    }

    // T-RR001: empty results render the no-match line (not a header).
    #[test]
    fn test_render_results_empty_reports_no_match() {
        assert_eq!(render_results(&[]), "No matching sessions found.");
    }

    // T-RR002: non-empty results render a count header plus one block per result.
    #[test]
    fn test_render_results_lists_header_and_each_result() {
        let results = [
            make_search_result("abc123", "/proj/a", "hello world"),
            make_search_result("def456", "/proj/b", "another match"),
        ];
        let out = render_results(&results);
        assert!(out.starts_with("Found 2 sessions:\n"), "got: {out}");
        assert!(out.contains("ID: abc123"), "got: {out}");
        assert!(out.contains("ID: def456"), "got: {out}");
    }

    #[test]
    fn test_format_result_empty_project() {
        let r = make_search_result("s1", "", "excerpt");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("unknown [claude]"));
        assert!(!output.contains("    /"));
    }

    // T-004 (#8/FR-004): the excerpt renders in full — no byte-cap truncation and
    // no "..." marker — so the whole chunk reaches the agent in one search.
    #[test]
    fn test_format_result_excerpt_not_truncated() {
        let long = "a".repeat(300);
        let r = make_search_result("s1", "/proj", &long);
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(
            output.contains(&format!("> {long}")),
            "the full excerpt must render without truncation"
        );
        assert!(
            !output.contains("> ..."),
            "no truncation marker should be emitted, got: {output}"
        );
    }

    struct FailingWriter;

    impl Write for FailingWriter {
        fn write(&mut self, _buf: &[u8]) -> io::Result<usize> {
            Err(io::Error::other("stdout unavailable"))
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    #[test]
    fn emit_success_propagates_non_pipe_write_error() {
        let out = CommandOutput::ok("hello".to_owned(), serde_json::Value::Null);
        let mut writer = FailingWriter;

        let err = emit_success_to(&out, false, &mut writer).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::Other);
    }

    // exit_code_for_write Err arm: a non-pipe stdout write failure surfaces as the
    // I/O exit code (74), the boundary that gives `recall ... > /full/disk` a
    // non-zero status instead of a silent exit 0. The Ok arm is exercised by every
    // integration test that emits a result; only the failure arm is pinned here.
    #[test]
    fn exit_code_for_write_maps_write_failure_to_io_error() {
        let code = exit_code_for_write(Err(io::Error::other("stdout unavailable")), false);
        assert_eq!(code, ExitCode::from(codes::IO_ERR));
    }

    #[test]
    fn exit_code_for_write_maps_written_to_success() {
        let code = exit_code_for_write(Ok(WriteOutcome::Written), false);
        assert_eq!(code, ExitCode::SUCCESS);
    }

    // T-005 (#8 audit/C): the excerpt passes through ansi::strip_control_chars like
    // every peer field (slug/project/session_id/file_path), so stored ANSI/control
    // sequences from indexed content cannot reach the terminal raw. #8 expanded this
    // surface (200-byte snippet -> full chunk content), making the gap material.
    #[test]
    fn test_format_result_excerpt_strips_control_chars() {
        let r = make_search_result("s1", "/proj", "clean\x1b[31mANSI\x1b[0mhere\x07bell");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(
            !output.contains('\x1b') && !output.contains('\x07'),
            "control characters must be stripped from the excerpt, got: {output:?}"
        );
        assert!(
            output.contains("> cleanANSIherebell"),
            "the cleaned excerpt text must render, got: {output:?}"
        );
    }

    #[test]
    fn test_format_result_empty_excerpt() {
        let r = make_search_result("s1", "/proj", "");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(!output.contains("> "));
    }

    #[test]
    fn test_search_idle_message_only_when_unindexed() {
        // `recall search` no longer auto-indexes, so an empty DB must surface guidance.
        assert_eq!(
            search_idle_message(0),
            Some("No sessions indexed. Run `recall index` first.")
        );
        assert_eq!(search_idle_message(5), None);
    }

    #[test]
    fn test_rebuild_is_a_distinct_subcommand() {
        let cli = Cli::try_parse_from(["recall", "rebuild"]).unwrap();
        assert_matches!(cli.command, Some(Command::Rebuild));
    }

    #[test]
    fn test_index_no_longer_accepts_force_flag() {
        assert!(Cli::try_parse_from(["recall", "index", "--force"]).is_err());
        let cli = Cli::try_parse_from(["recall", "index"]).unwrap();
        assert_matches!(cli.command, Some(Command::Index));
    }

    #[test]
    fn test_rebuild_recognized_by_shorthand_expansion() {
        // Without `rebuild` in KNOWN_SUBCOMMANDS, `recall rebuild` would be
        // rewritten to `search rebuild` by shorthand expansion.
        let args: Vec<OsString> = ["recall", "rebuild"].iter().map(OsString::from).collect();
        assert!(try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_none());
    }

    // T-005 (FR-005): with `embed` dropped from KNOWN_SUBCOMMANDS, `recall embed`
    // is not a subcommand — shorthand expansion rewrites it to `search embed`.
    // The inverse of test_rebuild_recognized_by_shorthand_expansion.
    #[test]
    fn embed_token_shorthand_expands_to_search() {
        let args: Vec<OsString> = ["recall", "embed"].iter().map(OsString::from).collect();
        assert!(try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_some());
    }

    // T-007 (FR-008): the user-facing READMEs must describe index-time embedding,
    // not the retired post-search/progressive model. OUTCOME.md carries the same
    // revision but is git-ignored, so this committed test targets the committed
    // artifacts. The banned tokens are architecture/flag terms (substring-safe —
    // "recall embed" is excluded because it is a substring of "embeds"); the
    // embed-at-index rewrite removed all three, so re-adding any must fail here.
    #[test]
    fn readme_describes_index_time_embedding_not_post_search_growth() {
        let en = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.md"))
            .expect("README.md must exist at the crate root");
        let ja = fs::read_to_string(concat!(env!("CARGO_MANIFEST_DIR"), "/README.ja.md"))
            .expect("README.ja.md must exist at the crate root");
        for (name, doc) in [("README.md", en.as_str()), ("README.ja.md", ja.as_str())] {
            for banned in [
                "post-search embedding",
                "Progressive embedding",
                "--no-embed",
            ] {
                assert!(
                    !doc.contains(banned),
                    "{name} still references the retired model: {banned:?}"
                );
            }
        }
    }

    #[test]
    fn test_subagent_file_path_shows_parent_session() {
        let r = search::SearchResult {
            session: parser::SessionData {
                session_id: "agent-a58c408".to_owned(),
                source: parser::Source::Claude,
                file_path: "/home/.claude/projects/proj/abc-def-123/subagents/agent-a58c408.jsonl"
                    .to_owned(),
                project: "/proj".to_owned(),
                slug: "agent-slug".to_owned(),
                timestamp: Some(1709251200000),
            },
            excerpt: "some text".to_owned(),
        };
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("abc-def-123"));
    }

    #[test]
    fn test_normal_session_path_shows_no_parent() {
        let r = make_search_result("s1", "/home/me/project", "excerpt");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(!output.contains("Parent:"));
    }

    fn setup_show_db() -> (tempfile::TempDir, Connection) {
        let (dir, conn) = db::setup_test_db();
        conn.execute(
            "INSERT INTO sessions VALUES ('abc-123', 'claude', '/path/f.jsonl', '/proj', 'my-slug', 1709251200000, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions VALUES ('abc-456', 'claude', '/path/g.jsonl', '/proj', 'other-slug', 1709251200000, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('abc-123', 'user', 'hello world')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('abc-123', 'assistant', 'hi there')",
            [],
        )
        .unwrap();
        (dir, conn)
    }

    #[test]
    fn test_show_session_exact_match() {
        let (_dir, conn) = setup_show_db();
        let out = show_session(&conn, "abc-123", false).unwrap();
        assert!(out.markdown.contains("# my-slug"));
        assert!(out.markdown.contains("- session_id: abc-123"));
        assert!(out.markdown.contains("- date: 2024-03-01"));
        assert!(out.markdown.contains("## [user]\nhello world"));
        assert!(out.markdown.contains("## [assistant]\nhi there"));
        // The `--json` payload carries the same session, machine-readable (#67 Phase 2).
        assert_eq!(out.data["session_id"], "abc-123");
        assert_eq!(out.data["messages"][0]["role"], "user");
        assert_eq!(out.data["messages"][0]["text"], "hello world");
    }

    #[test]
    fn test_show_session_prefix_match() {
        let (_dir, conn) = setup_show_db();
        let out = show_session(&conn, "abc-1", false).unwrap();
        assert!(out.markdown.contains("- session_id: abc-123"));
    }

    #[test]
    fn test_show_session_ambiguous_prefix() {
        let (_dir, conn) = setup_show_db();
        let err = show_session(&conn, "abc", false).unwrap_err();
        assert!(err.to_string().contains("Multiple sessions match"));
    }

    #[test]
    fn test_show_session_not_found() {
        let (_dir, conn) = setup_show_db();
        let err = show_session(&conn, "zzz", false).unwrap_err();
        assert!(err.to_string().contains("No session found matching"));
    }

    #[test]
    fn test_show_session_verbose_includes_file() {
        let (_dir, conn) = setup_show_db();
        let out = show_session(&conn, "abc-123", true).unwrap();
        assert!(out.markdown.contains("- file: /path/f.jsonl"));
    }

    #[test]
    fn test_show_session_message_order() {
        let (_dir, conn) = setup_show_db();
        let out = show_session(&conn, "abc-123", false).unwrap();
        let user_pos = out.markdown.find("## [user]").unwrap();
        let assistant_pos = out.markdown.find("## [assistant]").unwrap();
        assert!(
            user_pos < assistant_pos,
            "user message should come before assistant"
        );
    }

    // A row stored with a source outside the known set ('claude'/'codex') must
    // not abort show: Source::from_db returns None and the row falls back to
    // Claude (with a warn) so the session still renders. Covers the
    // unknown-source guard in show_session's query_map closure.
    #[test]
    fn test_show_session_unknown_source_falls_back_to_claude() {
        let (_dir, conn) = db::setup_test_db();
        conn.execute(
            "INSERT INTO sessions VALUES ('xyz-1', 'bogus', '/f', '/p', 'odd-slug', 1709251200000, 0.0, NULL)",
            [],
        )
        .unwrap();
        let out = show_session(&conn, "xyz-1", false).unwrap();
        assert!(
            out.markdown.contains("- source: claude"),
            "an unknown stored source must fall back to claude, got: {}",
            out.markdown
        );
    }

    /// Seed an 'auto' session (first user turn is a slash-command wrapper) and a
    /// 'human' session (normal turn), both unclassified — the classify tests' fixture.
    fn seed_classify_db() -> (tempfile::TempDir, Connection) {
        let (dir, conn) = db::setup_test_db();
        // 'auto': first user turn is a slash-command wrapper; 'human': a normal
        // turn. Both start unclassified (session_type NULL).
        conn.execute(
            "INSERT INTO sessions VALUES ('auto', 'claude', '/f', '/p', 'a', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('auto', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions VALUES ('human', 'claude', '/f', '/p', 'h', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('human', 'user', 'how do I implement auth')",
            [],
        )
        .unwrap();
        (dir, conn)
    }

    fn session_type_of(conn: &Connection, sid: &str) -> Option<String> {
        conn.query_row(
            "SELECT session_type FROM sessions WHERE session_id = ?1",
            [sid],
            |r| r.get(0),
        )
        .unwrap()
    }

    // T-010 (#24/FR-008): `recall classify --all` re-classifies existing sessions
    // from their first user turn and writes session_type.
    #[test]
    fn test_010_reclassify_all_tags_sessions() {
        let (_dir, mut conn) = seed_classify_db();
        let outcomes = reclassify_sessions(&mut conn, true, false).unwrap();
        assert_eq!(session_type_of(&conn, "auto").as_deref(), Some("automated"));
        assert_eq!(
            session_type_of(&conn, "human").as_deref(),
            Some("interactive")
        );
        assert_eq!(
            outcomes.len(),
            2,
            "both sessions with a user turn are classified"
        );
    }

    // T-011 (#24/FR-009a, FR-009b): --dry-run returns outcomes for display but does
    // not write session_type.
    #[test]
    fn test_011_reclassify_dry_run_does_not_write() {
        let (_dir, mut conn) = seed_classify_db();
        let outcomes = reclassify_sessions(&mut conn, true, true).unwrap();
        assert_eq!(
            session_type_of(&conn, "auto"),
            None,
            "dry-run must not write session_type"
        );
        assert_eq!(session_type_of(&conn, "human"), None);
        let auto = outcomes
            .iter()
            .find(|o| o.session_id == "auto")
            .expect("dry-run still reports the outcome for display");
        assert_eq!(auto.session_type, classify::SessionType::Automated);
        assert!(
            !auto.excerpt.is_empty(),
            "dry-run outcome carries the first-turn excerpt for display"
        );
    }

    // T-012 (#24/FR-008): without --all, only unclassified (NULL) sessions are
    // (re)classified; an already-tagged session is left untouched.
    #[test]
    fn test_012_reclassify_incremental_skips_already_tagged() {
        let (_dir, mut conn) = db::setup_test_db();
        conn.execute(
            "INSERT INTO sessions VALUES ('fresh', 'claude', '/f', '/p', 'f', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('fresh', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();
        // 'tagged' carries an automated first turn but was already tagged interactive;
        // without --all it must stay interactive (not re-evaluated).
        conn.execute(
            "INSERT INTO sessions VALUES ('tagged', 'claude', '/f', '/p', 't', 0, 0.0, 'interactive')",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO messages (session_id, role, text) VALUES ('tagged', 'user', '<command-message>clear</command-message>')",
            [],
        )
        .unwrap();

        let outcomes = reclassify_sessions(&mut conn, false, false).unwrap();

        assert_eq!(
            session_type_of(&conn, "fresh").as_deref(),
            Some("automated"),
            "an unclassified session is classified"
        );
        assert_eq!(
            session_type_of(&conn, "tagged").as_deref(),
            Some("interactive"),
            "an already-tagged session is untouched without --all"
        );
        assert_eq!(
            outcomes.len(),
            1,
            "only the unclassified session is processed"
        );
    }

    // -- #73 段階2 Phase 1: index に full embed を統合 (FR-002 / FR-003 / FR-007) --
    //
    // index_and_report_with is the loader-injection seam the SOW/Spec define
    // (AC-2 #3). These drive it in-process with MockEmbedder so the index-time
    // embed path is covered on a CI runner that has no MLX model (the spawn path
    // would degrade to FTS-only and never exercise embed).
    //
    // Seam shape under test: index_and_report_with(db_path, &IndexOptions, loader).
    // The source dirs MUST be injectable, not only db_path: rebuild (force=true)
    // wipes qa_chunks/sessions/messages then re-scans the source dirs, so T-008
    // cannot stage an FTS term without a per-test claude_dir. set_var is not an
    // option here — Cargo.toml denies unsafe_code and edition-2024 makes
    // std::env::set_var an unsafe fn — so the dir must arrive as an argument.
    // IndexOptions already bundles {force, claude_dir, codex_dir} (indexer.rs:48)
    // and mirrors index_from_dirs(conn, &IndexOptions), keeping the seam at 3 args.

    /// Count chunks with no embedding (the pending set: qa_chunks rows absent
    /// from vec_chunks). The Spec's definition of "fully embedded" is this == 0.
    fn pending_embed_count(conn: &Connection) -> i64 {
        conn.query_row(
            "SELECT COUNT(*) FROM qa_chunks c \
             WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
            [],
            |r| r.get(0),
        )
        .unwrap()
    }

    fn vec_chunk_count(conn: &Connection) -> i64 {
        conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap()
    }

    /// A loader that hands the index pipeline a deterministic in-memory embedder.
    /// `FnOnce` matches the seam's bound; the mock returns 768-dim vectors derived
    /// from text bytes, so no model download or network is touched.
    fn mock_loader() -> Result<Arc<dyn Embed>, DegradedReason> {
        Ok(Arc::new(embedder::MockEmbedder::new()) as Arc<dyn Embed>)
    }

    /// Two temp dirs that do not exist on disk. `collect_sources` gates session
    /// scanning on `is_dir()` (indexer.rs:269), so `index_from_dirs` scans no
    /// files — but its orphan cleanup still deletes any hand-seeded session
    /// whose file_path has no on-disk source, chunks included. Use only where
    /// sessions come from a real source dir (or none); embed-step tests call
    /// `embed_all_pending` directly instead.
    fn absent_source_dirs(root: &Path) -> (PathBuf, PathBuf) {
        (root.join("no_claude"), root.join("no_codex"))
    }

    // T-001 (FR-002): embed_all_pending embeds every pending chunk when a model
    // is present. Given 3 directly-seeded un-embedded chunks, the run embeds all
    // 3 (vec_chunks gains 3 rows; the NOT EXISTS pending set empties). Calls
    // embed_all_pending directly: the embed step is the unit under test, and
    // index_from_dirs' orphan cleanup deletes hand-seeded sessions (no on-disk
    // source), which would silently empty the pending set before embedding.
    // Perspective: equivalence (the "model present, pending > 0" class).
    #[test]
    fn embed_all_pending_embeds_every_pending_chunk() {
        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");
        let mut conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        for i in 1..=3 {
            db::seed_chunk(&conn, i, &format!("pending content {i}"));
        }

        let result = embed_all_pending(&mut conn, &embedder::MockEmbedder::new()).unwrap();

        assert_eq!(result.embedded, 3, "every pending chunk embeds");
        assert_eq!(result.failed_count, 0, "a working embedder fails nothing");
        assert_eq!(pending_embed_count(&conn), 0, "the pending set empties");
        assert_eq!(vec_chunk_count(&conn), 3, "each chunk gains a vec row");
    }

    // T-010 (FR-002, FR-004): a partial embed failure flows through
    // embed_all_pending — it tallies the failed batch, runs
    // warn_if_batches_failed (the tracing-warn path), and stays Ok so the index
    // is non-fatal. Given 3 pending chunks where 'x' poisons their single batch,
    // the run reports embedded=0, failed_count=3, carries the first batch error
    // for the --json note, and leaves the batch pending for the next run.
    // Perspective: error (the failing-embedder arm) + state (EmbedResult tally).
    #[test]
    fn embed_all_pending_partial_failure_reports_failed_count() {
        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");
        let mut conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        db::seed_chunk(&conn, 1, "x");
        for i in 2..=3 {
            db::seed_chunk(&conn, i, &format!("pending content {i}"));
        }

        let result =
            embed_all_pending(&mut conn, &embedder::MockEmbedder::failing_on_text("x")).unwrap();

        assert_eq!(
            result.embedded, 0,
            "the poisoned single batch embeds nothing"
        );
        assert_eq!(
            result.failed_count, 3,
            "the whole 3-chunk batch is reported failed"
        );
        let err = result.first_error.as_deref().unwrap_or_default();
        assert!(
            err.contains("poison text"),
            "the first batch error is carried up for the --json note, got: {err:?}"
        );
        assert_eq!(
            pending_embed_count(&conn),
            3,
            "the failed batch stays pending for the next run"
        );
    }

    // T-002 (FR-003): embedding is incremental — a second run re-embeds nothing.
    // Given 1 already-embedded chunk and 1 new pending chunk, the first
    // embed_all_pending embeds exactly the pending one (vec 1→2); the second
    // leaves the count unchanged (the NOT EXISTS gate skips both).
    // Perspective: hazard (re-run idempotence) + boundary (the already-embedded
    // vs pending split).
    #[test]
    fn embed_all_pending_reembeds_nothing_on_second_run() {
        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");
        let mut conn = open_or_create_db(&db_path).unwrap();
        db::seed_session(&conn, "s1");
        // chunk 1 pre-embedded, chunk 2 pending.
        for (id, content) in [(1, "already embedded"), (2, "new pending")] {
            db::seed_chunk(&conn, id, content);
        }
        let emb = embedder::MockEmbedder::deterministic_vector("already embedded");
        conn.execute(
            "INSERT INTO vec_chunks (embedding, chunk_id, sub_idx) VALUES (?1, 1, 0)",
            [embedder::f32_as_bytes(&emb)],
        )
        .unwrap();

        let first = embed_all_pending(&mut conn, &embedder::MockEmbedder::new()).unwrap();
        assert_eq!(
            first.embedded, 1,
            "the first run embeds only the pending chunk"
        );
        assert_eq!(vec_chunk_count(&conn), 2, "vec gains exactly the new chunk");

        let second = embed_all_pending(&mut conn, &embedder::MockEmbedder::new()).unwrap();
        assert_eq!(second.embedded, 0, "the second run re-embeds nothing");
        assert_eq!(
            vec_chunk_count(&conn),
            2,
            "a second run must re-embed nothing (NOT EXISTS vec_chunks gate)"
        );
    }

    /// Seed a one-session Claude JSONL under `claude_dir` so a rebuild that
    /// re-scans the source re-stages the session and its FTS terms. Mirrors the
    /// integration `seed_indexed_session` fixture (cli_integration.rs:186); the
    /// unique token `quokka` is the FTS round-trip probe for T-008.
    fn seed_claude_source(claude_dir: &Path) {
        fs::create_dir_all(claude_dir).unwrap();
        fs::write(
            claude_dir.join("s1.jsonl"),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"quokka rebuild probe"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    }

    // T-008 (FR-007): rebuild re-embeds every chunk AND rebuilds the FTS index.
    // Given an already-embedded index built from a seeded source, a force=true
    // rebuild wipes then re-scans the source: afterward the pending set is empty
    // (all re-embedded) and the seeded term still matches via FTS search. Two
    // assertions guard both halves FR-007 names (re-embed + FTS reconstruction),
    // so neither half can silently regress. Perspective: state (rebuild is a
    // full-reset transition) + hazard (a destructive rebuild that fails to
    // repopulate FTS would pass an embed-only check).
    #[test]
    fn rebuild_reembeds_all_and_rebuilds_fts_index() {
        let src = tempfile::TempDir::new().unwrap();
        let claude_dir = src.path().join("claude");
        let (_, codex_dir) = absent_source_dirs(src.path());
        seed_claude_source(&claude_dir);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");

        // Precondition: an already-embedded index (force=false index + embed).
        let opts = indexer::IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();
        {
            let conn = open_or_create_db(&db_path).unwrap();
            assert_eq!(
                pending_embed_count(&conn),
                0,
                "precondition: initial index must leave nothing pending"
            );
        }

        // rebuild: force=true re-parses every present session and replaces its
        // chunks/embeddings; missing source roots keep their existing index.
        let rebuild_opts = indexer::IndexOptions {
            force: true,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        index_and_report_with(&Some(db_path.clone()), &rebuild_opts, mock_loader).unwrap();

        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            pending_embed_count(&conn),
            0,
            "rebuild must re-embed every chunk (pending=0 after full reset)"
        );

        let opts = search::SearchOptions::default();
        let hits = search::search(&conn, "quokka", &opts).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "rebuild must reconstruct FTS: the seeded term must return its session"
        );
    }

    // #165 follow-up (SF4-JSON, end-to-end): a source root that vanishes between
    // runs preserves its existing session AND threads the skipped root into the
    // returned IndexOutcome, so index_command_output can surface it on the --json
    // path (the stderr warning loop runs as the human-path side effect). Given a DB
    // seeded from a claude source, a second index pointed at a now-missing claude
    // dir returns skipped_roots naming claude with its one preserved session.
    // Perspective: state (a present→missing root transition) + hazard (a root
    // outage dropped from the outcome would hide the masked deletions #165 names).
    #[test]
    fn index_and_report_with_missing_root_threads_skipped_root_into_outcome() {
        let src = tempfile::TempDir::new().unwrap();
        let claude_dir = src.path().join("claude");
        let (_, codex_dir) = absent_source_dirs(src.path());
        seed_claude_source(&claude_dir);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");

        let opts = indexer::IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        index_and_report_with(&Some(db_path.clone()), &opts, mock_loader).unwrap();

        // Second run: the claude root has vanished. Its session must be preserved
        // and reported, not deleted.
        let gone_claude = src.path().join("claude_gone");
        let missing_opts = indexer::IndexOptions {
            force: false,
            claude_dir: &gone_claude,
            codex_dir: &codex_dir,
        };
        let outcome = index_and_report_with(&Some(db_path), &missing_opts, mock_loader).unwrap();

        assert_eq!(
            outcome.skipped_roots.len(),
            1,
            "the missing claude root must be threaded into the outcome"
        );
        assert_eq!(outcome.skipped_roots[0].source, parser::Source::Claude);
        assert_eq!(
            outcome.skipped_roots[0].preserved_sessions, 1,
            "the one preserved claude session must be counted"
        );
    }

    // -- Phase 2 (AC-4): model-less index degrades, never aborts (FR-004a/004b) --
    //
    // index_and_report_with returns Result<IndexOutcome> carrying the degraded note
    // rather than emitting it, so the note is verifiable in-process without a 4th
    // reporter arg — mirroring search_degraded_state, which returns (degraded,
    // notes). run_index/run_rebuild surface the note, then map to Ok(()).

    // T-004a (FR-004a): a model-less index completes full-text indexing and
    // succeeds (exit 0). Given a seeded source and a loader that returns
    // Err(DegradedReason::NotInstalled), index_and_report_with returns Ok AND the
    // seeded FTS term still returns its session — the FTS index is built even
    // though embedding was skipped. Perspective: branch (the loader Err arm, the
    // sole path a model-less CI runner takes) + hazard (a degraded path that
    // aborted, or that skipped FTS along with embed, would strand search).
    #[test]
    fn index_and_report_with_degraded_loader_completes_fts_and_succeeds() {
        let src = tempfile::TempDir::new().unwrap();
        let claude_dir = src.path().join("claude");
        let (_, codex_dir) = absent_source_dirs(src.path());
        seed_claude_source(&claude_dir);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");

        let opts = indexer::IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
            Err(DegradedReason::NotInstalled)
        });
        assert!(
            outcome.is_ok(),
            "a model-less index must complete FTS and exit 0, not abort"
        );

        let conn = open_or_create_db(&db_path).unwrap();
        let search_opts = search::SearchOptions::default();
        let hits = search::search(&conn, "quokka", &search_opts).unwrap();
        assert_eq!(
            hits.len(),
            1,
            "the FTS index must be complete after a degraded index: the seeded term returns its session"
        );
    }

    // T-004b (FR-004b): a model-less index emits a degraded note that names
    // `recall model download`. Given the same Err(NotInstalled) loader, the
    // returned IndexOutcome carries a note string containing that command, so the
    // user is told how to enable embedding. Perspective: error (the model-absent
    // path surfaces an actionable message, not a silent skip).
    #[test]
    fn index_and_report_with_degraded_loader_emits_model_download_note() {
        let src = tempfile::TempDir::new().unwrap();
        let claude_dir = src.path().join("claude");
        let (_, codex_dir) = absent_source_dirs(src.path());
        seed_claude_source(&claude_dir);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");

        let opts = indexer::IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        let outcome = index_and_report_with(&Some(db_path.clone()), &opts, || {
            Err(DegradedReason::NotInstalled)
        })
        .expect("a degraded index must succeed");

        let note = outcome
            .degraded_note
            .expect("a model-less index must carry a degraded note");
        assert!(
            note.contains("recall model download"),
            "the degraded note must name `recall model download` to guide the user, got: {note}"
        );
    }

    // D5 (#130): the index context was only tested with NotInstalled; pin the
    // other loader-failure variants. BackendUnavailable and ProbeFailed share
    // one equivalence class (amici's user_note renders both as "unavailable",
    // re-downloading would not help), so both must carry that note and skip
    // embedding. The `record_degraded` half of D5 is covered by the loader's
    // own test (try_load_embedder_cached_records_degraded_non_install_failure);
    // production wires that loader in via index_and_report.
    #[test]
    fn index_and_report_with_unavailable_loader_carries_unavailability_note() {
        for reason in [
            DegradedReason::BackendUnavailable,
            DegradedReason::ProbeFailed,
        ] {
            let src = tempfile::TempDir::new().unwrap();
            let claude_dir = src.path().join("claude");
            let (_, codex_dir) = absent_source_dirs(src.path());
            seed_claude_source(&claude_dir);

            let db_dir = tempfile::TempDir::new().unwrap();
            let db_path = db_dir.path().join("recall.db");

            let opts = indexer::IndexOptions {
                force: false,
                claude_dir: &claude_dir,
                codex_dir: &codex_dir,
            };
            let outcome = index_and_report_with(&Some(db_path), &opts, || Err(reason))
                .expect("a degraded index must succeed");

            let note = outcome
                .degraded_note
                .unwrap_or_else(|| panic!("{reason:?} must carry a degraded note"));
            assert!(
                note.contains("unavailable"),
                "{reason:?} must report unavailability (not a download hint), got: {note}"
            );
            assert!(
                !note.to_lowercase().contains("download"),
                "{reason:?} is not repaired by re-downloading, so the note must not \
                 suggest it (that hint belongs to NotInstalled), got: {note}"
            );
            assert_eq!(outcome.embedded, 0, "{reason:?} skips embedding");
            assert_eq!(outcome.failed_count, 0, "{reason:?} fails no batches");
        }
    }

    // D6 (#130): Err(Disabled) yields the same IndexOutcome as a no-op success
    // (degraded_note=None, all counters zero) — a caller-level opt-out is not a
    // degradation, matching the search channel's contract (search_degraded_state
    // returns (false, []) for Disabled). Production never constructs Disabled
    // today (amici's loaders never return it and recall has no opt-out switch);
    // this pin makes the collision explicit for whoever adds that switch.
    #[test]
    fn index_and_report_with_disabled_loader_is_indistinguishable_from_success() {
        let src = tempfile::TempDir::new().unwrap();
        let claude_dir = src.path().join("claude");
        let (_, codex_dir) = absent_source_dirs(src.path());
        seed_claude_source(&claude_dir);

        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");

        let opts = indexer::IndexOptions {
            force: false,
            claude_dir: &claude_dir,
            codex_dir: &codex_dir,
        };
        let outcome =
            index_and_report_with(&Some(db_path), &opts, || Err(DegradedReason::Disabled))
                .expect("a disabled-embedder index must succeed");

        assert_eq!(
            outcome.degraded_note, None,
            "opt-out is not a degradation: no note is carried"
        );
        assert_eq!(outcome.embedded, 0, "a disabled embedder embeds nothing");
        assert_eq!(
            outcome.failed_count, 0,
            "a disabled embedder fails no batches"
        );
        let envelope = index_command_output(&outcome);
        assert!(
            !envelope.degraded,
            "the --json envelope reports a disabled embedder as non-degraded"
        );
    }

    // -- Phase 2 (D2, AC-4..7): index_command_output builds the --json envelope --
    //
    // index_command_output(&IndexOutcome) -> CommandOutput is the pure seam the Spec
    // defines (FR-005): the dispatch maps run_index's IndexOutcome through it instead
    // of no_payload(), so degradation reaches the --json envelope. Testing the pure
    // function directly needs no DB, loader, or process env — every branch is driven
    // by an IndexOutcome literal. The two degradations are exclusive (Spec排他):
    // embed-stop has degraded_note=None + failed_count>0; model-absent has
    // degraded_note=Some + failed_count=0.

    /// An embed-stop outcome: the model loaded and embedded some chunks, but a batch
    /// failed, so failed_count>0 and there is a first_error but no degraded_note.
    fn embed_stop_outcome() -> IndexOutcome {
        IndexOutcome {
            degraded_note: None,
            embedded: 5,
            failed_count: 3,
            first_error: Some("batch: mock failure".to_owned()),
            skipped_roots: Vec::new(),
        }
    }

    /// A model-absent outcome: embedding was skipped, so failed_count=0 and the
    /// degraded_note names `recall model download`.
    fn model_absent_outcome() -> IndexOutcome {
        IndexOutcome {
            degraded_note: Some("note: run `recall model download` to enable.".to_owned()),
            embedded: 0,
            failed_count: 0,
            first_error: None,
            skipped_roots: Vec::new(),
        }
    }

    /// A fully successful outcome: the model loaded and every chunk embedded, so no
    /// degraded_note, no failures, no error.
    fn success_outcome() -> IndexOutcome {
        IndexOutcome {
            degraded_note: None,
            embedded: 8,
            failed_count: 0,
            first_error: None,
            skipped_roots: Vec::new(),
        }
    }

    // T-003 (FR-005, FR-006, FR-008): an embed-stop index surfaces degraded:true with
    // a note naming the failure and the retry command, and carries both magnitudes
    // in the payload. Given failed_count>0 + degraded_note=None,
    // index_command_output sets degraded:true, notes[0] names the failed chunk
    // count, the first batch error, and `recall index` (retry guidance), and the
    // payload reports failed_count and embedded (what did land before the stop).
    // Perspective: branch (the embed-stop arm) + error (a partial failure must be
    // visible, not reported as success).
    #[test]
    fn index_command_output_surfaces_embed_stop_with_failed_count_and_retry() {
        let out = index_command_output(&embed_stop_outcome());

        assert!(
            out.degraded,
            "an embed-stop index is degraded: --json must not report it as a clean success"
        );
        assert!(
            out.notes[0].contains("3 chunk"),
            "the note must name how many chunks failed, got: {:?}",
            out.notes
        );
        assert!(
            out.notes[0].contains("recall index"),
            "the note must guide the agent to retry via `recall index`, got: {:?}",
            out.notes
        );
        assert!(
            out.notes[0].contains("mock failure"),
            "the note carries the first batch error, not the unknown-error fallback, got: {:?}",
            out.notes
        );
        assert_eq!(
            out.data["failed_count"], 3,
            "the payload carries the failure magnitude, not just a bool"
        );
        assert_eq!(
            out.data["embedded"], 5,
            "the payload reports how many chunks did embed despite the stop"
        );
    }

    // T-004 (FR-007): a model-absent index surfaces degraded:true with the
    // model-download note in --json (previously stderr-only). Given
    // degraded_note=Some(model download) + failed_count=0, index_command_output sets
    // degraded:true and a note naming `recall model download`.
    // Perspective: branch (the model-absent arm, exclusive with embed-stop).
    #[test]
    fn index_command_output_surfaces_model_absent_download_note() {
        let out = index_command_output(&model_absent_outcome());

        assert!(
            out.degraded,
            "a model-absent index is degraded: embedding was skipped"
        );
        assert!(
            out.notes
                .iter()
                .any(|n| n.contains("recall model download")),
            "the model-absent note must reach --json (not stderr only), got: {:?}",
            out.notes
        );
    }

    // T-005 (FR-006, FR-008, FR-009, FR-010): a fully successful index is not
    // degraded and carries failed_count=0. Given embedded>0 + failed_count=0 +
    // degraded_note=None, index_command_output sets degraded:false, notes:[], and
    // data.failed_count==0.
    // Perspective: branch (the success arm) + boundary (failed_count at its zero edge).
    #[test]
    fn index_command_output_clean_success_is_not_degraded() {
        let out = index_command_output(&success_outcome());

        assert!(!out.degraded, "a fully embedded index is not degraded");
        assert!(
            out.notes.is_empty(),
            "a clean success carries no notes, got: {:?}",
            out.notes
        );
        assert_eq!(
            out.data["failed_count"], 0,
            "the payload reports zero failures on success"
        );
        assert_eq!(
            out.data["embedded"], 8,
            "the payload reports how many chunks were embedded on success"
        );
    }

    // T-006 (FR-009): degraded co-varies with notes across every outcome —
    // degraded:true ⟺ notes non-empty — replicating search_degraded_state's
    // invariant so the bool and the text never disagree. Checked over all three
    // outcomes (embed-stop, model-absent, success).
    // Perspective: combination (the (degraded, notes_empty) decision table: the
    // (true, empty) and (false, non-empty) rows are the contradictions this forbids).
    #[test]
    fn index_command_output_degraded_covaries_with_notes() {
        for outcome in [
            embed_stop_outcome(),
            model_absent_outcome(),
            success_outcome(),
        ] {
            let out = index_command_output(&outcome);
            assert_eq!(
                out.degraded,
                !out.notes.is_empty(),
                "degraded must equal notes-non-empty (degraded={}, notes={:?})",
                out.degraded,
                out.notes
            );
        }
    }

    // T-009 (FR-006): the embed-stop note sanitizes and bounds first_error — control
    // chars are stripped (matching every other envelope-bound string) and the error
    // is capped so a verbose MLX error cannot bloat the note. The retry guidance
    // must survive the capping.
    // Perspective: boundary (an error far over the cap) + hazard (ANSI escapes
    // forwarded raw into an agent-parsed channel).
    #[test]
    fn index_command_output_sanitizes_and_bounds_first_error_in_note() {
        let outcome = IndexOutcome {
            degraded_note: None,
            embedded: 0,
            failed_count: 2,
            first_error: Some(format!("batch: \x1b[31mboom\x1b[0m {}", "e".repeat(300))),
            skipped_roots: Vec::new(),
        };

        let out = index_command_output(&outcome);

        assert!(
            !out.notes[0].contains('\x1b'),
            "control chars are stripped from the note, got: {:?}",
            out.notes
        );
        assert!(
            out.notes[0].len() < 200,
            "a verbose error is capped, got len {}",
            out.notes[0].len()
        );
        assert!(
            out.notes[0].contains("rerun `recall index`"),
            "the retry guidance survives the capping, got: {:?}",
            out.notes
        );
    }

    // #165 follow-up (SF4-JSON): a skipped source root reaches the --json envelope,
    // not just the human stderr warning. Given skipped_roots non-empty,
    // index_command_output sets degraded:true (co-varying with notes), emits a note
    // naming the source and preserved count, and carries the structured magnitude in
    // data.skipped_roots so an agent reads it without parsing prose.
    // Perspective: branch (the skipped-root arm) + hazard (a root outage masking real
    // deletions must be visible on the machine path #165 names as its audience).
    #[test]
    fn index_command_output_surfaces_skipped_root_in_json() {
        let outcome = IndexOutcome {
            degraded_note: None,
            embedded: 4,
            failed_count: 0,
            first_error: None,
            skipped_roots: vec![indexer::SkippedRoot {
                source: parser::Source::Codex,
                preserved_sessions: 2,
                reason: indexer::SkippedReason::IncompleteEnumeration,
            }],
        };

        let out = index_command_output(&outcome);

        assert!(
            out.degraded,
            "a skipped root is a degradation: --json must not report it as a clean success"
        );
        assert!(
            out.notes
                .iter()
                .any(|n| n.contains("codex") && n.contains("2") && n.contains("recall index")),
            "the note must name the source, the preserved count, and the reconcile command, got: {:?}",
            out.notes
        );
        assert!(
            out.notes
                .iter()
                .any(|n| n.contains("permissions and access")),
            "an incomplete-enumeration root must surface the access remedy, not the missing-root wording, got: {:?}",
            out.notes
        );
        assert_eq!(
            out.data["skipped_roots"][0]["source"], "codex",
            "the payload carries the skipped source, not just prose"
        );
        assert_eq!(
            out.data["skipped_roots"][0]["preserved_sessions"], 2,
            "the payload carries the preserved magnitude an agent can act on"
        );
        assert_eq!(
            out.data["skipped_roots"][0]["reason"], "incomplete_enumeration",
            "the payload carries the machine-readable skip cause so an agent picks the right remedy"
        );
    }

    // #185 T-185-4: with nothing preserved, the structured note channel (the
    // `notes` rendering, separate from the warn! log) must carry the missing-
    // sessions wording and drop the reconcile-deletions framing.
    #[test]
    fn index_command_note_for_zero_preserved_incomplete_root_omits_reconcile() {
        let outcome = IndexOutcome {
            degraded_note: None,
            embedded: 0,
            failed_count: 0,
            first_error: None,
            skipped_roots: vec![indexer::SkippedRoot {
                source: parser::Source::Claude,
                preserved_sessions: 0,
                reason: indexer::SkippedReason::IncompleteEnumeration,
            }],
        };

        let out = index_command_output(&outcome);

        assert!(
            out.notes
                .iter()
                .any(|n| n.contains("claude") && n.contains("some sessions may be missing")),
            "a zero-preserved incomplete root must warn that sessions may be missing, got: {:?}",
            out.notes
        );
        assert!(
            !out.notes
                .iter()
                .any(|n| n.contains("reconcile any real deletions")),
            "with nothing preserved the note must not promise to reconcile deletions, got: {:?}",
            out.notes
        );
        assert_eq!(
            out.data["skipped_roots"][0]["reason"], "incomplete_enumeration",
            "the payload still carries the machine-readable cause at zero preserved"
        );
    }

    // -- Phase 3 (AC-1): search is read-only — it embeds nothing (FR-001) --
    //
    // Seam choice: model-less, try_load_embedder_cached returns Err(NotInstalled)
    // so the embed branch is skipped anyway, making a seam-less vec_chunks-count
    // test vacuous (0 before and after regardless). Injecting a loaded embedder via
    // run_search_with is the only non-vacuous, CI-robust way to prove the read-only
    // search path adds no vectors.

    /// A `Command::Search` for `query` with production defaults: every
    /// `--*-current` flag off and automated sessions excluded. Used by T-003 to
    /// drive `run_search_with` on the read-only search path.
    fn search_command(query: &str) -> Command {
        Command::Search {
            query: query.to_owned(),
            project: None,
            days: None,
            source: None,
            limit: 10,
            exclude_current: false,
            include_current: false,
            only_current: false,
            include_automated: false,
        }
    }

    // T-003 (FR-001): search embeds nothing — vec_chunks is unchanged across a
    // search. Given one session with one pending (un-embedded) chunk and an
    // embedder injected via the loader seam (so the embed branch is reachable on a
    // model-less CI runner), run_search_with leaves vec_chunks empty: the search
    // path reads the pre-built index and never embeds. Asserting count == 0 with the
    // embedder PRESENT proves the absence is the deleted embed call, not a missing
    // model. Perspective: hazard (embedding cost leaking onto the search path) +
    // branch (the embedder-present arm, the only arm that could embed).
    #[test]
    fn run_search_with_embedder_present_embeds_nothing() {
        let db_dir = tempfile::TempDir::new().unwrap();
        let db_path = db_dir.path().join("recall.db");
        {
            let conn = open_or_create_db(&db_path).unwrap();
            // A session is required so search_idle_message does not short-circuit
            // before run_search_with reaches the embedder branch under test.
            db::seed_session(&conn, "s1");
            // One pending chunk: the removed post-search embed warmed the most-recent
            // pending chunks regardless of which results matched, so a retained embed
            // call would push vec_chunks to 1. Asserting 0 proves the call is gone.
            db::seed_chunk(&conn, 1, "pending content");
        }

        run_search_with(
            search_command("anything"),
            &Some(db_path.clone()),
            mock_loader,
        )
        .unwrap();

        let conn = open_or_create_db(&db_path).unwrap();
        assert_eq!(
            vec_chunk_count(&conn),
            0,
            "search must not embed: vec_chunks stays empty even with an embedder present"
        );
    }
}
