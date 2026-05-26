//! CLI exit-code contract (ADR-0066 Group 2, #82). Spawns the real binary and
//! pins the sysexits codes agents branch on, exercising the parse / dispatch /
//! error-classification glue that unit tests cannot reach in-process.

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
