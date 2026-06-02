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
use amici::cli::{
    Spinner, done, embed_with_spinners, exit_error, info as cli_info, progress_step,
    try_expand_shorthand,
};
use amici::logging::init_subscriber;
use amici::model::embedder::{DegradedReason, try_load_embedder_default_logging};
use amici::model::{degraded_reason_user_note, download_and_verify_model, record_degraded};
use amici::storage::{collect_rows, filter::escape_like};
use anyhow::{Context, Error, Result};
use clap::error::ErrorKind as ClapErrorKind;
use clap::{Parser, Subcommand};
use rurico::embed::{Embed, ModelId, cached_artifacts};
use rurico::handle_probe_if_needed;
use rusqlite::Connection;
use tracing::{info, warn};

use crate::envelope::{CommandOutput, render_json_error, render_json_success};
use crate::error::{
    RecallError, classify_exit_code, download_error, embedder_error, error_envelope,
};
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

#[derive(Subcommand)]
enum Command {
    /// Parse and chunk new session logs (incremental). No model calls.
    Index,
    /// Drop the index and rebuild from all sessions. No model calls.
    Rebuild,
    /// Embed pending chunks for semantic search.
    Embed,
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

        /// Skip post-search embedding (faster for piped/scripted usage)
        #[arg(long)]
        no_embed: bool,

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

#[derive(Subcommand)]
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

fn try_load_embedder_cached() -> Option<Arc<dyn Embed>> {
    try_load_embedder_cached_with(open_cached_embedder)
}

fn try_load_embedder_cached_with<F>(open_embedder: F) -> Option<Arc<dyn Embed>>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
{
    try_load_embedder_cached_with_reporter(open_embedder, cli_info)
}

fn try_load_embedder_cached_with_reporter<F, I>(
    open_embedder: F,
    mut report_info: I,
) -> Option<Arc<dyn Embed>>
where
    F: FnOnce() -> Result<Arc<dyn Embed>, DegradedReason>,
    I: FnMut(&str),
{
    match open_embedder() {
        Ok(e) => Some(e),
        Err(reason @ DegradedReason::NotInstalled) => {
            if let Some(note) = search_degraded_note(reason) {
                report_info(&note);
            }
            None
        }
        Err(reason) => {
            if let Some(note) = search_degraded_note(reason) {
                report_info(&note);
            }
            record_degraded(reason, "embedder load");
            None
        }
    }
}

fn search_degraded_note(reason: DegradedReason) -> Option<String> {
    degraded_reason_user_note(reason, "recall model download").map(|note| format!("note: {note}."))
}

// -- Subcommands --

fn run_index(db_path: &Option<PathBuf>) -> Result<()> {
    index_and_report(db_path, false)
}

fn run_rebuild(db_path: &Option<PathBuf>) -> Result<()> {
    index_and_report(db_path, true)
}

/// Shared index pipeline for `index` (incremental) and `rebuild` (full).
/// `force` is internalized here so both entry points stay argument-free.
fn index_and_report(db_path: &Option<PathBuf>, force: bool) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    let mut conn = open_or_create_db(&path)?;

    let sp = Spinner::new("Indexing sessions...");
    let stats = indexer::index_sessions(&mut conn, force)?;
    let main_msg = if stats.indexed > 0 {
        format!(
            "Indexed {} sessions in {:.1}s",
            stats.indexed, stats.elapsed_secs
        )
    } else {
        format!("{} sessions up to date", stats.total_sessions)
    };
    sp.finish_with_detail(&main_msg, stats.parse_error_detail().as_deref());

    let sp = Spinner::new("Creating chunks...");
    let chunk_stats = indexer::index_chunks(&mut conn)?;
    if chunk_stats.chunks_created > 0 {
        sp.finish(&format!("Created {} chunks", chunk_stats.chunks_created));
    } else {
        sp.finish("Chunks up to date");
    }

    let model_cached = cached_artifacts(ModelId::default())
        .map(|opt| opt.is_some())
        .unwrap_or(false);
    let model_line = if model_cached {
        "Model: ready (run `recall embed` to embed chunks)"
    } else {
        "Model: not installed (run `recall model download`)"
    };
    progress_step(&[model_line]);

    Ok(())
}

/// Message shown when no chunks are pending embedding. Distinguishes
/// "index not run yet" (no chunks at all) from "everything already embedded".
fn embed_idle_message(total_chunks: i64) -> &'static str {
    if total_chunks == 0 {
        "No chunks to embed. Run `recall index` first."
    } else {
        "All chunks already embedded"
    }
}

fn run_embed(db_path: &Option<PathBuf>) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    let mut conn = open_or_create_db(&path)?;

    let pending: i64 = conn.query_row(
        "SELECT COUNT(*) FROM qa_chunks c \
         WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
        [],
        |r| r.get(0),
    )?;

    if pending == 0 {
        let total_chunks: i64 =
            conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))?;
        done(embed_idle_message(total_chunks));
        return Ok(());
    }

    let pending_u32 = u32::try_from(pending).unwrap_or(u32::MAX);
    let pending_usize = usize::try_from(pending).expect("chunk count fits in usize");

    let result = embed_with_spinners(
        pending_u32,
        |_| load_cached_embedder(),
        |r: &embedder::EmbedResult| format!("Embedded {} chunks", r.embedded),
        |emb, update| {
            embedder::embed_recent_chunks(
                &mut conn,
                emb.as_ref(),
                pending_usize,
                Some(&|n, _| update(&format!("Embedding... {n}/{pending} chunks"))),
            )
        },
    )?;

    if let Some(r) = result {
        r.warn_if_stopped();
    }
    Ok(())
}

fn load_cached_embedder() -> Result<Arc<dyn Embed>> {
    open_cached_embedder().map_err(|reason| embedder_error(reason).into())
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
    let Command::Search {
        query,
        project,
        days,
        source,
        limit,
        no_embed,
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
    let mut conn = open_or_create_db(&path)?;

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

    let embedder = try_load_embedder_cached();
    // A missing embedding model degrades search to FTS-only; surface it so an
    // agent knows semantic ranking was unavailable (the `--json` notes channel).
    let (degraded, mut notes) = match &embedder {
        Some(_) => (false, Vec::new()),
        None => (
            true,
            vec![
                "semantic search unavailable: embedding model not installed (run `recall model download`)"
                    .to_owned(),
            ],
        ),
    };

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

    // Embedding runs after the payload is built, but the result is not emitted
    // until this returns, so it adds latency; `--no-embed` skips it to keep
    // piped/scripted usage responsive.
    if !no_embed && let Some(emb) = embedder.as_deref() {
        let _ = embed_around_results(&mut conn, emb, &results);
    }

    Ok(CommandOutput::with_notes(markdown, data, degraded, notes))
}

#[derive(Default)]
struct PostSearchEmbedOutcome {
    embedded: usize,
    stopped: bool,
}

/// Post-search progressive embedding: a side effect that warms the index around
/// the hits (nearby sessions) plus the most recent chunks. Progress goes to the
/// log, not the result. The caller emits its result only after this returns, so
/// it adds latency; `recall search --no-embed` skips it for interactive use.
fn embed_around_results(
    conn: &mut Connection,
    emb: &dyn Embed,
    results: &[search::SearchResult],
) -> PostSearchEmbedOutcome {
    let session_ids: Vec<String> = results
        .iter()
        .map(|r| r.session.session_id.clone())
        .collect();
    let mut outcome = PostSearchEmbedOutcome::default();
    let mut handle = |result: anyhow::Result<embedder::EmbedResult>| match result {
        Ok(r) => {
            let stopped = r.stopped_at_error.is_some();
            r.warn_if_stopped();
            outcome.embedded += r.embedded;
            outcome.stopped |= stopped;
        }
        Err(e) => {
            warn!(error = %e, "post-search embedding skipped");
        }
    };
    handle(embedder::embed_near_sessions(
        conn,
        emb,
        &session_ids,
        10,
        None,
    ));
    handle(embedder::embed_recent_chunks(conn, emb, 10, None));
    if outcome.embedded > 0 {
        if outcome.stopped {
            info!(
                embedded = outcome.embedded,
                "Embedded chunks before post-search embedding stopped"
            );
        } else {
            info!(
                total = outcome.embedded,
                "Embedded chunks (nearby + recent)"
            );
        }
    }
    outcome
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
    let matches: Vec<parser::SessionData> =
        collect_rows::<_, parser::SessionData, Vec<parser::SessionData>, Error>(rows)?;

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
    let messages: Vec<(String, String)> =
        collect_rows::<_, (String, String), Vec<(String, String)>, Error>(rows)?;

    // Writes to an in-memory Vec are infallible; the pipe is only touched later
    // by the single emit_success -> print_to_stdout.
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

    let model_ok = cached_artifacts(ModelId::default())
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

/// Write fully-rendered `rendered` to stdout. A closed pipe is a clean stop
/// (exit 0); non-pipe write errors propagate to the CLI boundary so they get a
/// non-zero I/O exit code. See [`crate::output`].
fn print_to_stdout(rendered: &str) -> io::Result<WriteOutcome> {
    write_result(&mut io::stdout().lock(), rendered)
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
/// `json_mode`, else the human-readable markdown. Empty markdown (a side-effect
/// command in text mode) prints nothing. Routes through [`print_to_stdout`] so
/// the SIGPIPE boundary covers both modes.
fn emit_success(out: &CommandOutput, json_mode: bool) -> io::Result<WriteOutcome> {
    let body = render_success_body(out, json_mode);
    if body.is_empty() {
        return Ok(WriteOutcome::Written);
    }
    print_to_stdout(&body)
}

#[cfg(test)]
fn emit_success_to_writer<W: Write>(
    out: &CommandOutput,
    json_mode: bool,
    writer: &mut W,
) -> io::Result<WriteOutcome> {
    let body = render_success_body(out, json_mode);
    if body.is_empty() {
        return Ok(WriteOutcome::Written);
    }
    write_result(writer, &body)
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
    "index", "rebuild", "embed", "model", "search", "show", "status", "classify", "help",
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
        Some(Command::Index) => run_index(&cli.db_path).map(|()| no_payload()),
        Some(Command::Rebuild) => run_rebuild(&cli.db_path).map(|()| no_payload()),
        Some(Command::Embed) => run_embed(&cli.db_path).map(|()| no_payload()),
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

fn main() -> ExitCode {
    handle_probe_if_needed();

    let cli = match parse_cli(env::args_os().collect()) {
        Ok(cli) => cli,
        Err(code) => return code,
    };
    let json_mode = cli.json;

    match run(cli) {
        Ok(out) => match emit_success(&out, json_mode) {
            Ok(WriteOutcome::Written | WriteOutcome::PipeClosed) => ExitCode::SUCCESS,
            Err(e) => {
                let err = Error::new(e).context("failed to write output");
                emit_error(&err, json_mode);
                classify_exit_code(&err)
            }
        },
        Err(e) => {
            emit_error(&e, json_mode);
            classify_exit_code(&e)
        }
    }
}

#[cfg(test)]
mod tests {
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
        assert!(result.is_none());
    }

    #[test]
    fn try_load_embedder_cached_reports_non_install_degraded_note() {
        let mut notes = Vec::new();
        let result = try_load_embedder_cached_with_reporter(
            || Err(DegradedReason::ProbeFailed),
            |msg| notes.push(msg.to_owned()),
        );

        assert!(result.is_none());
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

    // The Ok arm: a loaded embedder is returned so search can rank semantically.
    // Injected via the loader seam to stay independent of whether a model is
    // installed in the test environment (production coverage of this arm would
    // otherwise flip with model presence).
    #[test]
    fn try_load_embedder_cached_returns_loaded_embedder() {
        let loaded: Arc<dyn Embed> = Arc::new(embedder::MockEmbedder::new());
        let result = try_load_embedder_cached_with(|| Ok(loaded));
        assert!(result.is_some(), "a loaded embedder must be returned");
    }

    // The NotInstalled arm: a missing model degrades search to FTS-only (None)
    // after surfacing the install note, rather than aborting the command.
    #[test]
    fn try_load_embedder_cached_returns_none_when_model_not_installed() {
        let result = try_load_embedder_cached_with(|| Err(DegradedReason::NotInstalled));
        assert!(
            result.is_none(),
            "a not-installed model must degrade to None, not abort"
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

        let err = emit_success_to_writer(&out, false, &mut writer).unwrap_err();

        assert_eq!(err.kind(), ErrorKind::Other);
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
    fn test_embed_idle_message_distinguishes_unindexed_from_embedded() {
        assert_eq!(
            embed_idle_message(0),
            "No chunks to embed. Run `recall index` first."
        );
        assert_eq!(embed_idle_message(5), "All chunks already embedded");
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
        assert!(matches!(cli.command, Some(Command::Rebuild)));
    }

    #[test]
    fn test_index_no_longer_accepts_force_flag() {
        assert!(Cli::try_parse_from(["recall", "index", "--force"]).is_err());
        let cli = Cli::try_parse_from(["recall", "index"]).unwrap();
        assert!(matches!(cli.command, Some(Command::Index)));
    }

    #[test]
    fn test_rebuild_recognized_by_shorthand_expansion() {
        // Without `rebuild` in KNOWN_SUBCOMMANDS, `recall rebuild` would be
        // rewritten to `search rebuild` by shorthand expansion.
        let args: Vec<OsString> = ["recall", "rebuild"].iter().map(OsString::from).collect();
        assert!(try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS).is_none());
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

    /// Seed one session with one pending (un-embedded) chunk, the fixture both
    /// `embed_around_results` tests share.
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

    fn setup_pending_chunk_db() -> (tempfile::TempDir, Connection) {
        let (dir, conn) = db::setup_test_db();
        conn.execute(
            "INSERT INTO sessions VALUES ('s1', 'claude', '/f', '/p', 'slug', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (1, 's1', 'hello content', 0)",
            [],
        )
        .unwrap();
        (dir, conn)
    }

    // T-EAR001: embed_around_results embeds the pending chunk near a hit session
    // (the embedder-present side effect run_search runs after building its
    // payload). Uses MockEmbedder because CI has no real embedding model.
    #[test]
    fn embed_around_results_embeds_pending_chunk() {
        let (_dir, mut conn) = setup_pending_chunk_db();
        let emb = embedder::MockEmbedder::new();
        let results = [make_search_result("s1", "/p", "hello")];
        let outcome = embed_around_results(&mut conn, &emb, &results);
        let embedded: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(embedded, 1, "the pending chunk should be embedded");
        assert_eq!(outcome.embedded, 1);
        assert!(!outcome.stopped);
    }

    // T-EAR002: a failing embedder is swallowed — embed_around_results is
    // best-effort background work, so it logs and returns (no panic, nothing
    // embedded) and a search still succeeds when post-search embedding fails.
    #[test]
    fn embed_around_results_swallows_embed_failure() {
        let (_dir, mut conn) = setup_pending_chunk_db();
        let emb = embedder::MockEmbedder::failing_after(0);
        let results = [make_search_result("s1", "/p", "hello")];
        let outcome = embed_around_results(&mut conn, &emb, &results);
        let embedded: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(embedded, 0, "a failed embed must leave no vec_chunks");
        assert_eq!(outcome.embedded, 0);
        assert!(outcome.stopped);
    }

    #[test]
    fn embed_around_results_reports_partial_stop() {
        let (_dir, mut conn) = setup_pending_chunk_db();
        conn.execute(
            "INSERT INTO sessions VALUES ('s2', 'claude', '/f2', '/p', 'slug2', 0, 0.0, NULL)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO qa_chunks (id, session_id, content, timestamp) \
             VALUES (2, 's2', 'recent content', 0)",
            [],
        )
        .unwrap();

        let emb = embedder::MockEmbedder::failing_after(1);
        let results = [make_search_result("s1", "/p", "hello")];
        let outcome = embed_around_results(&mut conn, &emb, &results);

        assert_eq!(outcome.embedded, 1);
        assert!(outcome.stopped);
        let embedded: i64 = conn
            .query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))
            .unwrap();
        assert_eq!(
            embedded, 1,
            "only the successful near-session chunk remains"
        );
    }
}
