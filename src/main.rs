mod date;
mod db;
mod indexer;
mod parser;
mod search;

use std::path::Path;

use anyhow::{Context, Result};
use clap::Parser;

use crate::parser::Source;

#[derive(Parser)]
#[command(name = "recall", about = "Search past Claude Code and Codex sessions")]
struct Cli {
    /// FTS5 search query (quotes for phrases, AND/OR/NOT)
    query: String,

    /// Filter to sessions from a specific project path (prefix match)
    #[arg(long)]
    project: Option<String>,

    /// Only sessions from last N days
    #[arg(long)]
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
}

const EXCERPT_MAX_BYTES: usize = 200;

fn strip_control_chars(s: &str) -> std::borrow::Cow<'_, str> {
    if s.bytes().all(|b| b >= 0x20 || b == b'\n') {
        return std::borrow::Cow::Borrowed(s);
    }
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            if chars.peek() == Some(&'[') {
                chars.next();
                while let Some(&next) = chars.peek() {
                    chars.next();
                    if next.is_ascii_alphabetic() {
                        break;
                    }
                }
            }
        } else if !c.is_control() || c == '\n' {
            out.push(c);
        }
    }
    std::borrow::Cow::Owned(out)
}

fn format_timestamp(ts_ms: i64) -> String {
    if ts_ms == 0 {
        return "unknown".to_string();
    }
    let secs = ts_ms / 1000;
    let days = secs / 86400;
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

fn run() -> Result<()> {
    let cli = Cli::parse();

    let home = dirs::home_dir().context("Could not determine home directory")?;
    let db_path = home.join(".recall.db");

    // create_new(true) is atomic — no TOCTOU race, no exists() check needed.
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        match std::fs::OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&db_path)
        {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {}
            Err(e) => return Err(e).context("Failed to create database file"),
        }
    }

    let conn = db::open_db(&db_path).context("Failed to open database")?;

    let stats = indexer::index_sessions(&conn, cli.reindex, cli.verbose)?;
    if stats.indexed > 0 {
        eprintln!(
            "Indexed {} sessions in {:.1}s",
            stats.indexed, stats.elapsed_secs
        );
    }
    if stats.skipped > 0 {
        eprintln!("Skipped {} files (parse errors)", stats.skipped);
    }

    let results = search::search(
        &conn,
        &cli.query,
        &search::SearchOptions {
            project: cli.project,
            days: cli.days,
            source: cli.source,
            limit: cli.limit as usize,
            now_ms: None,
        },
    )?;

    print_results(&results, &stats);

    Ok(())
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
        let slug = strip_control_chars(&s.slug);
        let project = strip_control_chars(&s.project);
        let file_path = strip_control_chars(&s.file_path);
        println!(
            "[{}] {} | {} | {} [{}]",
            i + 1,
            date,
            slug,
            proj_name,
            s.source
        );
        if !project.is_empty() {
            println!("    {project}");
        }
        println!("    ID: {}", s.session_id);
        if !file_path.is_empty() {
            println!("    File: {file_path}");
        }
        if !r.excerpt.is_empty() {
            let clean = r.excerpt.replace('\n', " ");
            let clean = clean.trim();
            let truncated = truncate_str(clean, EXCERPT_MAX_BYTES);
            if truncated.len() < clean.len() {
                println!("    > {truncated}...");
            } else {
                println!("    > {clean}");
            }
        }
        println!();
    }
}

fn main() {
    if let Err(e) = run() {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_timestamp_zero() {
        assert_eq!(format_timestamp(0), "unknown");
    }

    #[test]
    fn test_format_timestamp_epoch() {
        assert_eq!(format_timestamp(1000), "1970-01-01");
    }

    #[test]
    fn test_format_timestamp_known_date() {
        // 2024-03-01 00:00:00 UTC = 1709251200000 ms
        assert_eq!(format_timestamp(1709251200000), "2024-03-01");
    }

    #[test]
    fn test_truncate_str_short() {
        assert_eq!(truncate_str("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_str_exact() {
        assert_eq!(truncate_str("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_str_ascii() {
        assert_eq!(truncate_str("hello world", 5), "hello");
    }

    #[test]
    fn test_truncate_str_multibyte() {
        let s = "こんにちは"; // 15 bytes (3 per char)
        let result = truncate_str(s, 10);
        assert_eq!(result, "こんに"); // 9 bytes fits in 10
        assert!(result.len() <= 10);
    }

    #[test]
    fn test_strip_control_chars_ansi() {
        assert_eq!(
            strip_control_chars("hello \x1b[31mred\x1b[0m world"),
            "hello red world"
        );
    }

    #[test]
    fn test_strip_control_chars_clean() {
        assert_eq!(strip_control_chars("normal text"), "normal text");
    }

    #[test]
    fn test_strip_control_chars_preserves_newlines() {
        assert_eq!(strip_control_chars("line1\nline2"), "line1\nline2");
    }
}
