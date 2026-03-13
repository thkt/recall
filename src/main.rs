mod ansi;
mod date;
mod db;
mod indexer;
mod parser;
mod search;

use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;
use rusqlite::Connection;

use crate::parser::Source;

/// Error message markers for exit code classification.
/// Keep in sync with bail! sites: main.rs `run()` and search.rs `search_sessions()`.
const USER_ERROR_MARKERS: &[&str] = &["search query is required", "Invalid search query"];

#[derive(Parser)]
#[command(name = "recall", about = "Search past Claude Code and Codex sessions")]
struct Cli {
    /// FTS5 search query (quotes for phrases, AND/OR/NOT)
    query: Option<String>,

    /// Filter to sessions from a specific project path (prefix match)
    #[arg(long)]
    project: Option<String>,

    /// Only sessions from last N days
    #[arg(long, value_parser = clap::value_parser!(i64).range(1..))]
    days: Option<i64>,

    /// Filter by source (claude or codex)
    #[arg(long, value_enum)]
    source: Option<Source>,

    /// Max results (1..=100, default: 10)
    #[arg(long, default_value_t = 10, value_parser = clap::value_parser!(u16).range(1..=100))]
    limit: u16,

    /// Force full rebuild of the index
    #[arg(long)]
    reindex: bool,

    /// Show diagnostic output
    #[arg(long, short)]
    verbose: bool,

    /// Database file path (default: ~/.recall.db, env: RECALL_DB)
    #[arg(long, env = "RECALL_DB")]
    db_path: Option<std::path::PathBuf>,
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

fn report_index_stats(stats: &indexer::IndexStats, reindex: bool, verbose: bool) {
    if stats.indexed > 0 {
        eprintln!(
            "Indexed {} sessions in {:.1}s",
            stats.indexed, stats.elapsed_secs
        );
    } else if reindex {
        eprintln!("Index is up to date ({} sessions)", stats.total_sessions);
    }
    if let Some(ref err) = stats.first_error {
        eprintln!("Failed to parse {} files — {err}", stats.parse_errors);
        if stats.parse_errors > 1 && !verbose {
            eprintln!("  (use --verbose for all errors)");
        }
    }
}

fn run_with_cli(cli: Cli) -> Result<()> {
    let db_path = match cli.db_path {
        Some(ref p) => p.clone(),
        None => dirs::home_dir()
            .context("Could not determine home directory")?
            .join(".recall.db"),
    };

    let mut conn = open_or_create_db(&db_path)?;

    let stats = indexer::index_sessions(&mut conn, cli.reindex, cli.verbose)?;
    report_index_stats(&stats, cli.reindex, cli.verbose);

    let Some(query) = cli.query else {
        if !cli.reindex {
            anyhow::bail!("A search query is required (or use --reindex to rebuild the index)");
        }
        return Ok(());
    };

    let results = search::search(
        &conn,
        &query,
        &search::SearchOptions {
            project: cli.project,
            days: cli.days,
            source: cli.source,
            limit: cli.limit.into(),
            now_ms: None,
        },
    )?;

    print_results(&results, &stats);

    Ok(())
}

fn run() -> Result<()> {
    run_with_cli(Cli::parse())
}

/// Extract the parent session UUID from a subagent file path.
///
/// Path structure: `{...}/{parent-session-uuid}/subagents/{agent-id}.jsonl`
/// Returns `None` for non-subagent paths.
fn extract_parent_session(file_path: &str) -> Option<&str> {
    let idx = file_path.find("/subagents/")?;
    let prefix = &file_path[..idx];
    let parent = prefix.rsplit('/').next()?;
    if parent.is_empty() { None } else { Some(parent) }
}

fn format_result(w: &mut impl std::io::Write, i: usize, r: &search::SearchResult) -> std::io::Result<()> {
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
    let file_path = ansi::strip_control_chars(&s.file_path);
    writeln!(w, "[{}] {} | {} | {} [{}]", i + 1, date, slug, proj_name, s.source)?;
    if !project.is_empty() {
        writeln!(w, "    {project}")?;
    }
    let session_id = ansi::strip_control_chars(&s.session_id);
    writeln!(w, "    ID: {session_id}")?;
    if let Some(parent) = extract_parent_session(&file_path) {
        writeln!(w, "    Parent: {parent}")?;
    }
    if !file_path.is_empty() {
        writeln!(w, "    File: {file_path}")?;
    }
    if !r.excerpt.is_empty() {
        let clean = r.excerpt.replace('\n', " ");
        let clean = clean.trim();
        let truncated = truncate_str(clean, EXCERPT_MAX_BYTES);
        if truncated.len() < clean.len() {
            writeln!(w, "    > {truncated}...")?;
        } else {
            writeln!(w, "    > {clean}")?;
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

fn print_results(results: &[search::SearchResult], stats: &indexer::IndexStats) {
    if results.is_empty() {
        println!("No matching sessions found.");
        return;
    }

    println!(
        "Found {} sessions (index: {} sessions, {} messages):\n",
        results.len(),
        stats.total_sessions,
        stats.total_messages
    );

    for (i, r) in results.iter().enumerate() {
        print_result(i, r);
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
            assert_eq!(truncate_str(input, max), expected, "input: {input:?}, max: {max}");
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

    // T-011: Subagent file path shows parent session (FR-004)
    #[test]
    fn test_011_subagent_file_path_shows_parent_session() {
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
        assert!(
            output.contains("abc-def-123"),
            "T-011: output should contain parent session UUID 'abc-def-123', got:\n{output}"
        );
    }

    // T-012: Normal session path shows no parent (FR-004)
    #[test]
    fn test_012_normal_session_path_shows_no_parent() {
        let r = make_search_result("s1", "/home/me/project", "excerpt");
        let mut buf = Vec::new();
        format_result(&mut buf, 0, &r).unwrap();
        let output = String::from_utf8(buf).unwrap();
        assert!(
            !output.contains("Parent:"),
            "T-012: normal session should not contain 'Parent:' info, got:\n{output}"
        );
    }
}
