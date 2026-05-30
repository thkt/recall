//! CLI exit-code contract (ADR-0066 Group 2, #82). Spawns the real binary and
//! pins the sysexits codes agents branch on, exercising the parse / dispatch /
//! error-classification glue that unit tests cannot reach in-process.

use std::fs;
use std::path::Path;
use std::process::Command;

use tempfile::TempDir;

/// A `recall` invocation isolated from the developer's real sessions and DB.
/// `RECALL_DB` points at a path inside `dir`; the source dirs point nowhere so
/// indexing never touches `~/.claude` or `~/.codex`.
fn recall(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_recall"));
    cmd.env("RECALL_DB", dir.join("recall.db"))
        .env("RECALL_CLAUDE_DIR", dir.join("no-claude"))
        .env("RECALL_CODEX_DIR", dir.join("no-codex"));
    cmd
}

fn exit_code(cmd: &mut Command) -> i32 {
    cmd.output()
        .expect("spawn recall binary")
        .status
        .code()
        .expect("process returned an exit code")
}

// T-CLI001: an unknown flag is a usage error (clap parse → USAGE 64).
#[test]
fn unknown_flag_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("--no-such-flag")), 64);
}

// T-CLI002: no subcommand prints the "search query required" usage error (64).
#[test]
fn no_subcommand_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(&mut recall(dir.path())), 64);
}

// T-CLI003: --help is not a failure; clap writes help to stdout and exits 0.
#[test]
fn help_exits_success() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("--help")), 0);
}

// T-CLI004: `show` against a database that was never created is a usage error
// (run `recall index` first), not a crash.
#[test]
fn show_without_database_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).args(["show", "deadbeef"])), 64);
}

// T-CLI005: searching an empty (but created) index is a success that prints the
// "no sessions indexed" guidance, exit 0 — not an error.
#[test]
fn search_empty_index_exits_success() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["search", "anything", "--no-embed"])),
        0
    );
}

// T-CLI007: a bare query with no subcommand is shorthand-expanded to `search`,
// exercising the shorthand-expansion parse branch. Empty index → success (0).
#[test]
fn bare_query_shorthand_expands_to_search() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["someword", "--no-embed"])),
        0
    );
}

// T-CLI006: `show` of an unknown id against an existing index resolves to no
// match → usage error (64). Creating the index first (via search) makes the DB
// exist, so this exercises the "No session found" path, not "Database not found".
#[test]
fn show_unknown_id_in_existing_index_exits_usage() {
    let dir = TempDir::new().unwrap();
    // `index` creates the (empty) database without needing the embedding model.
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);
    assert_eq!(exit_code(recall(dir.path()).args(["show", "deadbeef"])), 64);
}

// T-CLI008: `status` against a created index prints the stats to stdout and
// exits 0. Regression guard for #67 Phase 1: the status output now routes
// through `output::write_result` (a closed pipe stops cleanly) instead of bare
// `println!`, so this pins that the reroute preserves the visible output and the
// success exit code.
#[test]
fn status_prints_stats_and_exits_success() {
    let dir = TempDir::new().unwrap();
    // `index` creates the (empty) database so `status` reports counts.
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);
    let out = recall(dir.path())
        .arg("status")
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "status should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Sessions:") && stdout.contains("QA chunks:"),
        "status stdout should list stats, got: {stdout}"
    );
}

// T-CLI009: `status --verbose` adds the DB path line and still exits 0. Covers
// the verbose branch of the status output that the non-verbose T-CLI008 skips.
#[test]
fn status_verbose_includes_db_path() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);
    let out = recall(dir.path())
        .args(["status", "--verbose"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "status --verbose should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("DB:"),
        "verbose status stdout should list the DB path, got: {stdout}"
    );
}

// T-CLI010: searching an index that contains a matching session prints the
// "Found N sessions" listing and exits 0. The core OUTCOME path (search returns
// results) end to end: seed a Claude session JSONL, index it, then search a term
// it contains. Exercises the result-rendering path that the empty-index tests
// (which short-circuit on the "no sessions indexed" guidance) never reach.
#[test]
fn search_with_matching_index_lists_results() {
    let dir = TempDir::new().unwrap();
    let claude_dir = dir.path().join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("s1.jsonl"),
        r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world recall"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();
    // Only `index` reads the source dir; search then reads the built DB.
    assert_eq!(
        exit_code(
            recall(dir.path())
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0
    );
    let out = recall(dir.path())
        .args(["search", "hello", "--no-embed"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Found 1 sessions:"),
        "search stdout should list the matching session, got: {stdout}"
    );
}
