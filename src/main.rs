mod ansi;
mod chunker;
mod date;
mod db;
mod embedder;
mod hybrid;
mod indexer;
mod parser;
mod search;

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use rusqlite::Connection;

use crate::parser::Source;

/// Keep in sync with bail! sites: `run()` and `find_candidate_sessions()`.
const USER_ERROR_MARKERS: &[&str] = &["search query is required", "Invalid search query"];

const HF_REPO: &str = "keitokei1994/ruri-v3-310m-onnx";
const MODEL_FILES: &[&str] = &["model.onnx", "tokenizer.json"];

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
    /// Parse, chunk, and embed session logs
    Index {
        /// Force full rebuild
        #[arg(long)]
        force: bool,

        /// Embed all chunks (slow initial run, enables full hybrid search)
        #[arg(long)]
        embed: bool,
    },
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
    /// Show index and embedding status
    Status,
}

const EXCERPT_MAX_BYTES: usize = 200;

fn create_db_file(path: &Path) -> std::io::Result<()> {
    let mut opts = std::fs::OpenOptions::new();
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
        return "unknown".to_string();
    };
    let days = ts.div_euclid(date::MS_PER_DAY);
    let (y, m, d) = date::civil_from_days(days);
    format!("{y:04}-{m:02}-{d:02}")
}

fn truncate_str(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

fn open_or_create_db(path: &Path) -> Result<Connection> {
    match create_db_file(path) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
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

fn report_index_stats(stats: &indexer::IndexStats, force: bool, verbose: bool) {
    if stats.indexed > 0 {
        eprintln!(
            "Indexed {} sessions in {:.1}s",
            stats.indexed, stats.elapsed_secs
        );
    } else if force {
        eprintln!("Index is up to date ({} sessions)", stats.total_sessions);
    }
    if let Some(ref err) = stats.first_error {
        eprintln!("Failed to parse {} files — {err}", stats.parse_errors);
        if stats.parse_errors > 1 && !verbose {
            eprintln!("  (use --verbose for all errors)");
        }
    }
}

fn try_load_embedder() -> Option<embedder::Embedder> {
    let dir = embedder::model_dir();
    embedder::Embedder::new(&dir).ok()
}

/// Download model files if missing. Returns true if all files are present.
fn ensure_model(verbose: bool) -> Result<bool> {
    let dir = embedder::model_dir();
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("Failed to create model directory: {}", dir.display()))?;

    let mut downloaded = false;
    for filename in MODEL_FILES {
        let path = dir.join(filename);
        if path.exists() {
            continue;
        }
        downloaded = true;
        let url = format!("https://huggingface.co/{HF_REPO}/resolve/main/{filename}");
        eprintln!("Downloading {filename}...");
        let status = std::process::Command::new("curl")
            .args(["-fSL", "--progress-bar", "-o"])
            .arg(&path)
            .arg(&url)
            .status()
            .context("Failed to run curl — is it installed?")?;
        if !status.success() {
            let _ = std::fs::remove_file(&path);
            anyhow::bail!("Download failed for {filename}");
        }
    }

    if downloaded {
        eprintln!("Model downloaded to {}", dir.display());
    } else if verbose {
        eprintln!("Model already downloaded");
    }
    Ok(true)
}

// -- Subcommands --

fn run_index(force: bool, embed: bool, verbose: bool, db_path: &Option<PathBuf>) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    let mut conn = open_or_create_db(&path)?;

    let stats = indexer::index_sessions(&mut conn, force, verbose)?;
    report_index_stats(&stats, force, verbose);

    let chunk_stats = indexer::index_chunks(&mut conn, verbose)?;
    if chunk_stats.chunks_created > 0 {
        eprintln!("Created {} chunks", chunk_stats.chunks_created);
    }

    ensure_model(verbose)?;

    if embed {
        let dir = embedder::model_dir();
        let mut embedder = embedder::Embedder::new(&dir)
            .map_err(|e| anyhow::anyhow!("Failed to load model: {e}"))?;
        let total: i64 = conn.query_row(
            "SELECT COUNT(*) FROM qa_chunks c \
             WHERE NOT EXISTS (SELECT 1 FROM vec_chunks v WHERE v.chunk_id = c.id)",
            [],
            |r| r.get(0),
        )?;
        if total > 0 {
            eprintln!("Embedding {total} chunks (this may take a while)...");
            let result = indexer::embed_recent_chunks(&mut conn, &mut embedder, total as usize)?;
            eprintln!("Embedded {} chunks", result.embedded);
            result.warn_if_stopped();
        } else {
            eprintln!("All chunks already embedded");
        }
    }

    Ok(())
}

fn run_search(cmd: Command, verbose: bool, db_path: &Option<PathBuf>) -> Result<()> {
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
    let stats = indexer::index_sessions(&mut conn, false, verbose)?;
    report_index_stats(&stats, false, verbose);
    let chunk_stats = indexer::index_chunks(&mut conn, verbose)?;
    if chunk_stats.chunks_created > 0 && verbose {
        eprintln!("Created {} chunks", chunk_stats.chunks_created);
    }

    let mut embedder = try_load_embedder();

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
        embedder.as_mut().map(|e| e as &mut dyn embedder::Embed),
    )?;

    print_results(&results);

    // Post-search: progressive embedding (search results + recent)
    if !no_embed && let Some(emb) = embedder.as_mut() {
        let session_ids: Vec<String> = results
            .iter()
            .map(|r| r.session.session_id.clone())
            .collect();
        let mut total = 0;
        if let Ok(r) = indexer::embed_near_sessions(&mut conn, emb, &session_ids, 10) {
            r.warn_if_stopped();
            total += r.embedded;
        }
        if let Ok(r) = indexer::embed_recent_chunks(&mut conn, emb, 10) {
            r.warn_if_stopped();
            total += r.embedded;
        }
        if total > 0 && verbose {
            eprintln!("Embedded {total} chunks (nearby + recent)");
        }
    }

    Ok(())
}

fn run_status(verbose: bool, db_path: &Option<PathBuf>) -> Result<()> {
    let path = resolve_db_path(db_path)?;
    if !path.exists() {
        println!("Database not found at {}", path.display());
        println!("Run `recall index` to create the index.");
        return Ok(());
    }

    let conn = open_or_create_db(&path)?;

    let sessions: i64 = conn.query_row("SELECT COUNT(*) FROM sessions", [], |r| r.get(0))?;
    let chunks: i64 = conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))?;
    let embedded: i64 = conn.query_row("SELECT COUNT(*) FROM vec_chunks", [], |r| r.get(0))?;

    println!("Sessions: {sessions}");
    println!("QA chunks: {chunks}");
    println!("Embedded: {embedded}/{chunks}");

    let model_dir = embedder::model_dir();
    let model_ok = embedder::Embedder::new(&model_dir).is_ok();
    println!(
        "Model: {}",
        if model_ok {
            "ready"
        } else {
            "not downloaded (run `recall model`)"
        }
    );

    if verbose {
        println!("DB: {}", path.display());
        println!("Model dir: {}", model_dir.display());
    }

    Ok(())
}

// -- Output formatting --

/// Extract the parent session UUID from a subagent file path.
///
/// Path structure: `{...}/{parent-session-uuid}/subagents/{agent-id}.jsonl`
/// Returns `None` for non-subagent paths.
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

fn format_result(
    w: &mut impl std::io::Write,
    i: usize,
    r: &search::SearchResult,
) -> std::io::Result<()> {
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
    if let Err(e) = format_result(&mut std::io::stdout().lock(), i, r) {
        if e.kind() == std::io::ErrorKind::BrokenPipe {
            std::process::exit(0);
        }
        eprintln!("Warning: failed to write result: {e}");
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

fn run() -> Result<()> {
    // Backward compat: if first arg is not a subcommand, treat as `search <query>`
    let args: Vec<String> = std::env::args().collect();
    let known_commands = ["index", "search", "status", "help"];

    let cli = if args.len() > 1
        && !args[1].starts_with('-')
        && !known_commands.contains(&args[1].as_str())
    {
        // Legacy: `recall "query"` → `recall search "query"`
        let mut patched = vec![args[0].clone(), "search".to_string()];
        patched.extend_from_slice(&args[1..]);
        Cli::parse_from(patched)
    } else {
        Cli::parse()
    };

    match cli.command {
        Some(Command::Index { force, embed }) => run_index(force, embed, cli.verbose, &cli.db_path),
        Some(cmd @ Command::Search { .. }) => run_search(cmd, cli.verbose, &cli.db_path),
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
    if let Err(e) = run() {
        let msg = format!("{e:#}");
        eprintln!("Error: {msg}");
        let code = if USER_ERROR_MARKERS.iter().any(|m| msg.contains(m)) {
            1
        } else {
            2
        };
        std::process::exit(code);
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
                session_id: session_id.to_string(),
                source: parser::Source::Claude,
                file_path: "/path/to/file.jsonl".to_string(),
                project: project.to_string(),
                slug: "test-slug".to_string(),
                timestamp: Some(1709251200000),
            },
            excerpt: excerpt.to_string(),
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

    // Subagent file path shows parent session
    #[test]
    fn test_subagent_file_path_shows_parent_session() {
        let r = search::SearchResult {
            session: parser::SessionData {
                session_id: "agent-a58c408".to_string(),
                source: parser::Source::Claude,
                file_path: "/home/.claude/projects/proj/abc-def-123/subagents/agent-a58c408.jsonl"
                    .to_string(),
                project: "/proj".to_string(),
                slug: "agent-slug".to_string(),
                timestamp: Some(1709251200000),
            },
            excerpt: "some text".to_string(),
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
}
