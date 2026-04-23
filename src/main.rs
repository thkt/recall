mod ansi;
mod chunker;
mod date;
mod db;
mod embedder;
mod hybrid;
mod indexer;
mod parser;
mod search;

use std::env;
use std::ffi::OsString;
use std::fs::OpenOptions;
use std::io::{self, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process;
use std::sync::Arc;

use amici::cli::{
    Spinner, done, embed_with_spinners, exit_error, info as cli_info, progress_step,
    try_expand_shorthand,
};
use amici::logging::init_subscriber;
use amici::model::download_and_verify_model;
use amici::model::embedder::{DegradedReason, try_load_embedder_with};
use amici::storage::filter::escape_like;
use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rurico::embed::{Embed, ModelId, cached_artifacts};
use rurico::model_probe::handle_probe_if_needed;
use rusqlite::Connection;
use tracing::{info, warn};

use crate::parser::Source;

/// Keep in sync with bail! sites: `run()` and `find_candidate_sessions()`.
const USER_ERROR_MARKERS: &[&str] = &["search query is required", "Invalid search query"];

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
}

#[derive(Subcommand)]
enum Command {
    /// Parse and chunk session logs. No model calls.
    Index {
        /// Force full rebuild
        #[arg(long)]
        force: bool,
    },
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
    },
    /// Show full conversation of a session
    Show {
        /// Session ID (prefix match supported)
        session_id: String,
    },
    /// Show index and embedding status
    Status,
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download the embedding model and verify it loads.
    Download,
}

const EXCERPT_MAX_BYTES: usize = 200;

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

fn truncate_str(s: &str, max_bytes: usize) -> &str {
    &s[..s.floor_char_boundary(max_bytes)]
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
    try_load_embedder_with(
        || cached_artifacts(ModelId::default()),
        |e| warn!(error = %e, "failed to delete corrupt model files"),
        |e| warn!(error = %e, "embedder probe failed"),
    )
}

fn try_load_embedder_cached() -> Option<Arc<dyn Embed>> {
    match open_cached_embedder() {
        Ok(e) => Some(e),
        Err(DegradedReason::NotInstalled) => {
            cli_info(
                "note: embedding model not installed; using text search only. Run `recall model download` to enable semantic search.",
            );
            None
        }
        Err(reason) => {
            warn!(%reason, "embedder unavailable; using text search only");
            None
        }
    }
}

// -- Subcommands --

fn run_index(force: bool, db_path: &Option<PathBuf>) -> Result<()> {
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
        done("All chunks already embedded");
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
    open_cached_embedder().map_err(|reason| {
        let hint = match reason {
            DegradedReason::NotInstalled => {
                "embedding model not installed; run `recall model download`"
            }
            DegradedReason::BackendUnavailable => "MLX backend unavailable",
            DegradedReason::ProbeFailed => "embedding model probe failed",
            DegradedReason::Disabled => "embedder disabled",
        };
        anyhow::anyhow!("{hint}")
    })
}

fn run_model_download() -> Result<()> {
    download_and_verify_model().map_err(|e| anyhow::anyhow!("{e}"))
}

fn run_search(cmd: Command, db_path: &Option<PathBuf>) -> Result<()> {
    let Command::Search {
        query,
        project,
        days,
        source,
        limit,
        no_embed,
    } = cmd
    else {
        unreachable!()
    };

    let path = resolve_db_path(db_path)?;
    let mut conn = open_or_create_db(&path)?;

    // Auto-index FTS5 + chunks (fast: skips scan if dirs unchanged)
    let sp = Spinner::new("Indexing...");
    let stats = indexer::index_sessions(&mut conn, false)?;
    let chunk_stats = indexer::index_chunks(&mut conn)?;
    let main_msg = if stats.indexed > 0 || chunk_stats.chunks_created > 0 {
        format!(
            "Indexed {} sessions, {} chunks",
            stats.indexed, chunk_stats.chunks_created
        )
    } else {
        format!("{} sessions", stats.total_sessions)
    };
    sp.finish_with_detail(&main_msg, stats.parse_error_detail().as_deref());

    let embedder = try_load_embedder_cached();

    let results = search::search_with_embedder(
        &conn,
        &query,
        &search::SearchOptions {
            project,
            days,
            source,
            limit: limit.into(),
            now_ms: None,
        },
        embedder.as_deref(),
    )?;

    print_results(&results);

    // Post-search: progressive embedding (search results + recent)
    if !no_embed && let Some(emb) = embedder.as_deref() {
        let session_ids: Vec<String> = results
            .iter()
            .map(|r| r.session.session_id.clone())
            .collect();
        let mut total = 0;
        let mut handle = |result: anyhow::Result<embedder::EmbedResult>| match result {
            Ok(r) => {
                r.warn_if_stopped();
                total += r.embedded;
            }
            Err(e) => {
                warn!(error = %e, "post-search embedding skipped");
            }
        };
        handle(embedder::embed_near_sessions(
            &mut conn,
            emb,
            &session_ids,
            10,
            None,
        ));
        handle(embedder::embed_recent_chunks(&mut conn, emb, 10, None));
        if total > 0 {
            info!(total, "Embedded chunks (nearby + recent)");
        }
    }

    Ok(())
}

fn show_session(
    conn: &Connection,
    session_id: &str,
    verbose: bool,
    w: &mut impl Write,
) -> Result<()> {
    let mut stmt = conn.prepare(
        "SELECT session_id, source, file_path, project, slug, timestamp \
         FROM sessions WHERE session_id LIKE ?1 ESCAPE '\\' ORDER BY session_id",
    )?;
    let pattern = format!("{}%", escape_like(session_id));
    let matches: Vec<parser::SessionData> = stmt
        .query_map([&pattern], |row| {
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
        })?
        .collect::<Result<_, _>>()?;

    if matches.is_empty() {
        anyhow::bail!("No session found matching '{session_id}'");
    }
    if matches.len() > 1 {
        let mut msg = format!("Multiple sessions match '{session_id}':\n");
        for s in &matches {
            msg.push_str(&format!("  {}  {}\n", s.session_id, s.slug));
        }
        msg.push_str("Narrow the session ID prefix to match exactly one session");
        anyhow::bail!(msg);
    }

    let session = &matches[0];

    let mut msg_stmt =
        conn.prepare("SELECT role, text FROM messages WHERE session_id = ?1 ORDER BY rowid")?;
    let messages: Vec<(String, String)> = msg_stmt
        .query_map([&session.session_id], |row| Ok((row.get(0)?, row.get(1)?)))?
        .collect::<Result<_, _>>()?;

    writeln!(w, "# {}", session.slug)?;
    writeln!(w, "- session_id: {}", session.session_id)?;
    writeln!(w, "- date: {}", format_timestamp(session.timestamp))?;
    writeln!(w, "- project: {}", session.project)?;
    writeln!(w, "- source: {}", session.source)?;
    if verbose {
        writeln!(w, "- file: {}", session.file_path)?;
    }
    writeln!(w)?;

    for (role, text) in &messages {
        writeln!(w, "## [{role}]")?;
        writeln!(w, "{text}")?;
        writeln!(w)?;
    }
    Ok(())
}

fn run_show(session_id: &str, verbose: bool, db_path: &Option<PathBuf>) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        anyhow::bail!("Database not found. Run `recall index` first.");
    }

    let conn = open_or_create_db(&path)?;
    let mut w = io::stdout().lock();

    if let Err(e) = show_session(&conn, session_id, verbose, &mut w) {
        if let Some(io_err) = e.downcast_ref::<io::Error>()
            && io_err.kind() == ErrorKind::BrokenPipe
        {
            process::exit(0);
        }
        return Err(e);
    }
    Ok(())
}

fn run_status(verbose: bool, db_path: &Option<PathBuf>) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        cli_info(&format!("database not found at {}", path.display()));
        cli_info("run `recall index` to create the index.");
        return Ok(());
    }

    let conn = open_or_create_db(&path)?;

    let sessions: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))?;
    let embedded: i64 = conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))?;

    println!("Sessions: {sessions}");
    println!("QA chunks: {chunks}");
    println!("Embedded: {embedded}/{chunks}");

    let model_ok = cached_artifacts(ModelId::default())
        .map(|opt| opt.is_some())
        .unwrap_or(false);
    println!(
        "Model: {}",
        if model_ok {
            "ready"
        } else {
            "not installed (run `recall model download`)"
        }
    );

    if verbose {
        println!("DB: {}", path.display());
    }

    Ok(())
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
        let clean = r.excerpt.trim();
        let truncated = truncate_str(clean, EXCERPT_MAX_BYTES);
        for line in truncated.lines() {
            writeln!(w, "    > {line}")?;
        }
        if truncated.len() < clean.len() {
            writeln!(w, "    > ...")?;
        }
    }
    writeln!(w)?;
    Ok(())
}

fn print_result(i: usize, r: &search::SearchResult) {
    if let Err(e) = format_result(&mut io::stdout().lock(), i, r) {
        if e.kind() == ErrorKind::BrokenPipe {
            process::exit(0);
        }
        warn!(error = %e, "failed to write result");
    }
}

fn print_results(results: &[search::SearchResult]) {
    if results.is_empty() {
        println!("No matching sessions found.");
        return;
    }

    println!("Found {} sessions:\n", results.len());

    for (i, r) in results.iter().enumerate() {
        print_result(i, r);
    }
}

// -- Entry point --

const KNOWN_SUBCOMMANDS: &[&str] = &[
    "index", "embed", "model", "search", "show", "status", "help",
];
const GLOBAL_FLAGS: &[&str] = &["--verbose", "-v"];

fn run() -> Result<()> {
    let args: Vec<OsString> = env::args_os().collect();
    let cli = match try_expand_shorthand(&args, KNOWN_SUBCOMMANDS, GLOBAL_FLAGS) {
        Some(expanded) => Cli::parse_from(expanded),
        None => Cli::parse_from(args),
    };

    let default_filter = if cli.verbose {
        LOG_FILTER_VERBOSE
    } else {
        LOG_FILTER_DEFAULT
    };
    init_subscriber(default_filter);

    match cli.command {
        Some(Command::Index { force }) => run_index(force, &cli.db_path),
        Some(Command::Embed) => run_embed(&cli.db_path),
        Some(Command::Model(ModelCommand::Download)) => run_model_download(),
        Some(cmd @ Command::Search { .. }) => run_search(cmd, &cli.db_path),
        Some(Command::Show { session_id }) => run_show(&session_id, cli.verbose, &cli.db_path),
        Some(Command::Status) => run_status(cli.verbose, &cli.db_path),
        None => {
            anyhow::bail!(
                "A search query is required. Usage: recall search \"query\" or recall index"
            )
        }
    }
}

/// Exit codes: 1 = user error (bad query, missing args), 2 = system error (IO, DB).
fn main() {
    handle_probe_if_needed();

    if let Err(e) = run() {
        let msg = format!("{e:#}");
        exit_error(&msg);
        let code = if USER_ERROR_MARKERS.iter().any(|m| msg.contains(m)) {
            1
        } else {
            2
        };
        process::exit(code);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
    fn test_truncate_str() {
        for (input, max, expected) in [
            ("hello", 10, "hello"),
            ("hello", 5, "hello"),
            ("hello world", 5, "hello"),
        ] {
            assert_eq!(
                truncate_str(input, max),
                expected,
                "input: {input:?}, max: {max}"
            );
        }
        // Multibyte: 3 bytes per char, 10 bytes fits 3 chars (9 bytes)
        let result = truncate_str("こんにちは", 10);
        assert_eq!(result, "こんに");
        assert!(result.len() <= 10);
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

    #[test]
    fn test_format_result_empty_project() {
        let r = make_search_result("s1", "", "excerpt");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("unknown [claude]"));
        assert!(!output.contains("    /"));
    }

    #[test]
    fn test_format_result_truncated_excerpt() {
        let long = "a".repeat(300);
        let r = make_search_result("s1", "/proj", &long);
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("..."));
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
            "INSERT INTO sessions VALUES ('abc-123', 'claude', '/path/f.jsonl', '/proj', 'my-slug', 1709251200000, 0.0)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO sessions VALUES ('abc-456', 'claude', '/path/g.jsonl', '/proj', 'other-slug', 1709251200000, 0.0)",
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
        let mut buf = Vec::new();
        show_session(&conn, "abc-123", false, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("# my-slug"));
        assert!(output.contains("- session_id: abc-123"));
        assert!(output.contains("- date: 2024-03-01"));
        assert!(output.contains("## [user]\nhello world"));
        assert!(output.contains("## [assistant]\nhi there"));
    }

    #[test]
    fn test_show_session_prefix_match() {
        let (_dir, conn) = setup_show_db();
        let mut buf = Vec::new();
        show_session(&conn, "abc-1", false, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("- session_id: abc-123"));
    }

    #[test]
    fn test_show_session_ambiguous_prefix() {
        let (_dir, conn) = setup_show_db();
        let mut buf = Vec::new();
        let err = show_session(&conn, "abc", false, &mut buf).unwrap_err();
        assert!(err.to_string().contains("Multiple sessions match"));
    }

    #[test]
    fn test_show_session_not_found() {
        let (_dir, conn) = setup_show_db();
        let mut buf = Vec::new();
        let err = show_session(&conn, "zzz", false, &mut buf).unwrap_err();
        assert!(err.to_string().contains("No session found matching"));
    }

    #[test]
    fn test_show_session_verbose_includes_file() {
        let (_dir, conn) = setup_show_db();
        let mut buf = Vec::new();
        show_session(&conn, "abc-123", true, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(output.contains("- file: /path/f.jsonl"));
    }

    #[test]
    fn test_show_session_message_order() {
        let (_dir, conn) = setup_show_db();
        let mut buf = Vec::new();
        show_session(&conn, "abc-123", false, &mut buf).unwrap();
        let output = String::from_utf8(buf).unwrap();
        let user_pos = output.find("## [user]").unwrap();
        let assistant_pos = output.find("## [assistant]").unwrap();
        assert!(
            user_pos < assistant_pos,
            "user message should come before assistant"
        );
    }
}
