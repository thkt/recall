//! CLI exit-code contract (ADR-0066 Group 2, #82). Spawns the real binary and
//! pins the sysexits codes agents branch on, exercising the parse / dispatch /
//! error-classification glue that unit tests cannot reach in-process.

use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

use rusqlite::Connection;
use tempfile::TempDir;

/// A `recall` invocation isolated from the developer's real sessions and DB.
/// `RECALL_DB` points at a path inside `dir`; the source dirs point nowhere so
/// indexing never touches `~/.claude` or `~/.codex`. `CLAUDE_CODE_SESSION_ID` is
/// cleared so a suite run from inside a live Claude Code session does not inherit
/// the invoking session id — `search` now excludes that session by default, so
/// inheritance would leak the runner's environment into results. Tests that
/// exercise the exclusion set the variable explicitly.
fn recall(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_recall"));
    cmd.env("RECALL_DB", dir.join("recall.db"))
        .env("RECALL_CLAUDE_DIR", dir.join("no-claude"))
        .env("RECALL_CODEX_DIR", dir.join("no-codex"))
        .env_remove("CLAUDE_CODE_SESSION_ID");
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

// -- #67 Phase 2: global `--json` envelope (T-CLI011..T-CLI017) --
//
// These spawn the real binary with the global `--json` flag and pin the wire
// envelope an agent consumes. Several assert on the response body, not the exit
// code alone: a usage error (T-CLI012) exits 64, but a plain-text rejection
// would too, so the `{"error":{"code":"USAGE_ERROR",...}}` body is what proves
// the envelope path ran. The contract keys (`results`, `session_id`, `excerpt`,
// `messages`, `sessions`, `qa_chunks`, `embedded`, `degraded`, `notes`) are the
// public surface; assert them verbatim.

/// Seed a one-session Claude index in `dir` so `--json` search has a hit to
/// emit. Mirrors the T-CLI010 fixture (JSONL seed → `index` builds the DB);
/// only `index` reads the source dir, so search then reads the built DB.
fn seed_indexed_session(dir: &Path) {
    let claude_dir = dir.join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("s1.jsonl"),
        r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world recall"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();
    assert_eq!(
        exit_code(
            recall(dir)
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0,
        "index should build the DB and exit 0"
    );
}

// T-CLI011: `search <q> --json` against a matching index prints the success
// envelope to stdout and exits 0. The machine path of the core OUTCOME (search
// returns results): parses the envelope and asserts data.results carries the hit
// (session_id + excerpt), and that degraded co-varies with notes — asserting the
// invariant, not a fixed value, keeps it portable across machines that have or
// lack the cached embedding model. Perspective: normal (the JSON happy path).
#[test]
fn search_json_emits_success_envelope_with_results() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let out = recall(dir.path())
        .args(["search", "hello", "--no-embed", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    let results = v["data"]["results"]
        .as_array()
        .unwrap_or_else(|| panic!("data.results should be an array, got: {stdout}"));
    assert_eq!(
        results.len(),
        1,
        "the one seeded session should be the single hit, got: {stdout}"
    );
    assert!(
        results[0]["session_id"].is_string() && results[0]["excerpt"].is_string(),
        "each result row should carry session_id and excerpt, got: {stdout}"
    );
    // degraded and notes co-vary: search sets degraded=true with an FTS-fallback
    // note exactly when the embedding model is absent. Asserting the invariant
    // (not a fixed value) keeps the test portable across machines that have or
    // lack the cached model.
    let degraded = v["degraded"]
        .as_bool()
        .unwrap_or_else(|| panic!("degraded should be a bool, got: {stdout}"));
    let notes = v["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes should be an array, got: {stdout}"));
    assert_eq!(
        degraded,
        !notes.is_empty(),
        "degraded must coincide with a non-empty notes list, got: {stdout}"
    );
}

// T-CLI012: `--json` with no subcommand prints the usage error as a JSON
// envelope on stderr and exits 64. Assert on the body, not the exit code alone:
// today clap rejects the unknown `--json` flag with exit 64 too, so an exit-only
// check would pass against plain-text stderr (a false green). The body must be
// `{"error":{"code":"USAGE_ERROR",...}}` with a next_step. Perspective: error.
#[test]
fn no_subcommand_json_emits_error_envelope_on_stderr() {
    let dir = TempDir::new().unwrap();
    let out = recall(dir.path())
        .arg("--json")
        .output()
        .expect("spawn recall binary");
    assert_eq!(
        out.status.code(),
        Some(64),
        "missing query is a usage error (64)"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(r#""error""#) && stderr.contains(r#""code":"USAGE_ERROR""#),
        "stderr should be a JSON error envelope with USAGE_ERROR, got: {stderr}"
    );
    assert!(
        stderr.contains(r#""next_step""#),
        "the error envelope should carry next_step guidance, got: {stderr}"
    );
}

// T-CLI013: `status --json` against a created index prints the stats as a
// success envelope on stdout and exits 0. The data payload carries the
// machine-readable counters (`sessions`, `qa_chunks`, `embedded`) — the
// snake_case keys, not the human "Sessions:" / "QA chunks:" labels.
// Perspective: normal.
#[test]
fn status_json_emits_stats_envelope() {
    let dir = TempDir::new().unwrap();
    // `index` creates the (empty) DB so `status` has counts to report.
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);
    let out = recall(dir.path())
        .args(["status", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "status --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(r#""data""#),
        "stdout should be a success envelope, got: {stdout}"
    );
    assert!(
        stdout.contains(r#""sessions""#)
            && stdout.contains(r#""qa_chunks""#)
            && stdout.contains(r#""embedded""#),
        "data should carry sessions/qa_chunks/embedded counters, got: {stdout}"
    );
}

// T-CLI014: `--json` does not change the exit code of an empty-index search.
// Searching an empty (but created) index is a success (exit 0) with or without
// `--json` — the flag selects the output format, never the exit class.
// Perspective: boundary (empty index, the zero-result edge) + regression (exit
// code invariant across the flag). Pairs with T-CLI005 (the `--json`-less twin).
#[test]
fn search_empty_index_json_exit_code_matches_human() {
    let human = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(human.path()).arg("index")), 0);
    let human_code = exit_code(recall(human.path()).args(["search", "anything", "--no-embed"]));

    let json = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(json.path()).arg("index")), 0);
    let json_code =
        exit_code(recall(json.path()).args(["search", "anything", "--no-embed", "--json"]));

    assert_eq!(human_code, 0, "empty-index search (human) should exit 0");
    assert_eq!(
        json_code, human_code,
        "--json must not change the exit code of an empty-index search"
    );
}

// T-CLI015: without `--json`, search still prints the human-readable listing and
// exits 0 — the Phase-2 envelope work must not regress the default output that
// T-CLI010 pins. Regression guard: the same matching-index fixture must yield
// the "Found N sessions:" text (not JSON) when `--json` is absent.
#[test]
fn search_without_json_keeps_human_output() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let out = recall(dir.path())
        .args(["search", "hello", "--no-embed"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Found 1 sessions:"),
        "default output should stay human-readable, got: {stdout}"
    );
    assert!(
        !stdout.contains(r#""data""#),
        "default output must not emit the JSON envelope, got: {stdout}"
    );
}

// T-CLI016: `show <id> --json` against a matching index prints the success
// envelope to stdout and exits 0. The data payload carries the session identity
// (`session_id`) and the conversation as `messages` (each a role/text object) —
// the machine path of `show`. Perspective: normal (the JSON happy path).
#[test]
fn show_json_emits_session_envelope() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let out = recall(dir.path())
        .args(["show", "s1", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "show --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains(r#""data""#) && stdout.contains(r#""session_id":"s1""#),
        "stdout should be a success envelope carrying the session id, got: {stdout}"
    );
    assert!(
        stdout.contains(r#""messages""#)
            && stdout.contains(r#""role":"user""#)
            && stdout.contains(r#""text":"hello world recall""#),
        "data should carry the conversation messages, got: {stdout}"
    );
}

// T-CLI017: without `--json`, `show` prints the human-readable markdown and
// exits 0 — the envelope work must not regress the default `show` output. The
// markdown carries the slug header and the role-tagged message body, and must
// not emit the JSON envelope. Perspective: regression. Pairs with T-CLI016.
#[test]
fn show_without_json_keeps_human_output() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let out = recall(dir.path())
        .args(["show", "s1"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "show should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("# s1") && stdout.contains("## [user]"),
        "default output should stay human-readable markdown, got: {stdout}"
    );
    assert!(
        !stdout.contains(r#""data""#),
        "default output must not emit the JSON envelope, got: {stdout}"
    );
}

// T-CLI018: a side-effect command (`index`) with `--json` emits the no_payload
// envelope — data:null, degraded:false, empty notes — and exits 0. Pins the
// machine contract for commands that produce no result payload (the only source
// of `data:null` in the envelope). Progress spinners go to stderr, so stdout is
// exactly the one envelope; parsed as JSON, not substring-matched.
#[test]
fn index_json_emits_null_data_envelope() {
    let dir = TempDir::new().unwrap();
    let out = recall(dir.path())
        .args(["index", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "index --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    assert!(
        v["data"].is_null(),
        "a side-effect command must emit data:null, got: {stdout}"
    );
    assert_eq!(
        v["degraded"],
        serde_json::json!(false),
        "index is not a degraded path, got: {stdout}"
    );
    assert_eq!(
        v["notes"],
        serde_json::json!([]),
        "index carries no notes, got: {stdout}"
    );
}

// T-CLI019: `status --json` against a never-created index reports zero counters
// with model_ready:false as a success envelope, exit 0 — the no-database branch
// of run_status. Pairs with T-CLI013, which runs `index` first (the live path).
#[test]
fn status_json_without_index_reports_zero_counts() {
    let dir = TempDir::new().unwrap();
    let out = recall(dir.path())
        .args(["status", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(
        out.status.code(),
        Some(0),
        "status --json should exit 0 without an index"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    assert_eq!(
        v["data"]["sessions"],
        serde_json::json!(0),
        "no index means zero sessions, got: {stdout}"
    );
    assert_eq!(
        v["data"]["model_ready"],
        serde_json::json!(false),
        "the no-database branch reports model_ready:false, got: {stdout}"
    );
}

/// Seed a two-session Claude index (s1, s2) that both match "hello", then index
/// it. The current-session tests differ only in the flag and env they pass next,
/// so the seed lives here once.
fn seed_two_matching_sessions(dir: &Path) {
    let claude_dir = dir.join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    for sid in ["s1", "s2"] {
        fs::write(
            claude_dir.join(format!("{sid}.jsonl")),
            r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world recall"},"timestamp":"2026-03-01T00:00:00Z"}"#,
        )
        .unwrap();
    }
    assert_eq!(
        exit_code(
            recall(dir)
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0,
        "index should build the DB and exit 0"
    );
}

// T-CLI020: CLAUDE_CODE_SESSION_ID excludes the invoking session by default (#77).
// Two sessions (s1, s2) both match the query. With the env var set to s1 and no
// flag, the default drops s1 and keeps s2 — proving the binary reads the env var
// and threads it into SearchOptions.current, the glue the in-process resolve unit
// test cannot reach. The baseline (env cleared by the helper) lists both, so the
// assertion cannot pass vacuously by returning nothing.
#[test]
fn search_excludes_current_session_by_env_default() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    // Baseline: no invoking session (helper clears the env) → both are candidates.
    let baseline = recall(dir.path())
        .args(["search", "hello", "--no-embed"])
        .output()
        .expect("spawn recall binary");
    let baseline_out = String::from_utf8_lossy(&baseline.stdout);
    assert!(
        baseline_out.contains("ID: s1") && baseline_out.contains("ID: s2"),
        "baseline (no env) should list both sessions, got: {baseline_out}"
    );

    // CLAUDE_CODE_SESSION_ID=s1 → s1 is the invoking session, excluded by default.
    let out = recall(dir.path())
        .args(["search", "hello", "--no-embed"])
        .env("CLAUDE_CODE_SESSION_ID", "s1")
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ID: s2"),
        "the non-current session must remain, got: {stdout}"
    );
    assert!(
        !stdout.contains("ID: s1"),
        "the invoking session (s1) must be excluded by default, got: {stdout}"
    );
}

// T-CLI022: --include-current overrides the default exclusion (#77). With
// CLAUDE_CODE_SESSION_ID=s1 the flag keeps s1, so both s1 and s2 appear — the
// end-to-end path for the Ignore mode that the default/exclude tests never reach.
#[test]
fn search_include_current_keeps_invoking_session() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    let out = recall(dir.path())
        .args(["search", "hello", "--no-embed", "--include-current"])
        .env("CLAUDE_CODE_SESSION_ID", "s1")
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ID: s1") && stdout.contains("ID: s2"),
        "--include-current must keep the invoking session, got: {stdout}"
    );
}

// T-CLI023: --only-current restricts results to the invoking session (#77). With
// CLAUDE_CODE_SESSION_ID=s1, only s1 appears even though s2 also matches — the
// positive end-to-end path that T-CLI021 (the env-absent usage error) omits.
#[test]
fn search_only_current_restricts_to_invoking_session() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    let out = recall(dir.path())
        .args(["search", "hello", "--no-embed", "--only-current"])
        .env("CLAUDE_CODE_SESSION_ID", "s1")
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("ID: s1"),
        "--only-current must keep the invoking session, got: {stdout}"
    );
    assert!(
        !stdout.contains("ID: s2"),
        "--only-current must drop other sessions, got: {stdout}"
    );
}

// T-CLI021: --only-current without CLAUDE_CODE_SESSION_ID is a usage error (64).
// resolve_current_session has nothing to filter to, so it returns
// RecallError::Usage; this pins that it classifies to the sysexits USAGE code end
// to end (not a crash or a silently empty result). resolve runs before the DB is
// opened, so no index is needed.
#[test]
fn only_current_without_env_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["search", "hello", "--no-embed", "--only-current"])),
        64
    );
}

// T-CLI024 (#24 Phase 3): `recall classify` reports outcomes end to end, and
// --dry-run previews without persisting. The session's first user turn is a
// slash-command wrapper, so it classifies automated.
#[test]
fn classify_reports_and_dry_run_previews() {
    let dir = TempDir::new().unwrap();
    let claude_dir = dir.path().join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("s1.jsonl"),
        r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"<command-message>clear</command-message>"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();
    assert_eq!(
        exit_code(
            recall(dir.path())
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0,
        "index should build the DB and exit 0"
    );

    let dry = recall(dir.path())
        .args(["classify", "--all", "--dry-run"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(
        dry.status.code(),
        Some(0),
        "classify --dry-run should exit 0"
    );
    let dry_out = String::from_utf8_lossy(&dry.stdout);
    assert!(dry_out.contains("dry-run"), "dry-run output: {dry_out}");

    let run = recall(dir.path())
        .args(["classify", "--all"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(run.status.code(), Some(0), "classify --all should exit 0");
    let out = String::from_utf8_lossy(&run.stdout);
    assert!(out.contains("classified"), "classify output: {out}");
}

// T-CLI025 (#24 audit/N): a default search surfaces a note that automated sessions
// are excluded, so an AI-agent consumer can discover --include-automated.
#[test]
fn search_notes_automated_exclusion() {
    let dir = TempDir::new().unwrap();
    let claude_dir = dir.path().join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("auto.jsonl"),
        r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"<command-message>clear</command-message>"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();
    fs::write(
        claude_dir.join("human.jsonl"),
        r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"authentication notes"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();
    assert_eq!(
        exit_code(
            recall(dir.path())
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0
    );

    let out = recall(dir.path())
        .args(["search", "authentication", "--no-embed", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("--include-automated"),
        "default search must note the automated exclusion: {stdout}"
    );
}

// T-CLI026 (#24 audit/N): when the index has no automated sessions, the search
// output carries no exclusion note (the hint does not appear spuriously).
#[test]
fn search_omits_automated_note_when_none_excluded() {
    let dir = TempDir::new().unwrap();
    let claude_dir = dir.path().join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join("human.jsonl"),
        r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"authentication notes"},"timestamp":"2026-03-01T00:00:00Z"}"#,
    )
    .unwrap();
    assert_eq!(
        exit_code(
            recall(dir.path())
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0
    );

    let out = recall(dir.path())
        .args(["search", "authentication", "--no-embed", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("--include-automated"),
        "no automated sessions means no exclusion note: {stdout}"
    );
}

// -- #73 stage 1: `recall hook-handler` (SessionEnd hook → FTS index) (T-001..T-009) --
//
// These spawn the real binary as the SessionEnd hook would: a JSON payload on
// stdin, no subcommand args beyond `hook-handler`. The discriminator that proves
// the feature exists (rather than the current shorthand expansion, which rewrites
// `hook-handler` to `search "hook-handler"`) is FR-004 / NFR-003: hook-handler
// writes NOTHING to stdout (index counts go to tracing). A `search` of the same
// argv prints idle guidance or results, so asserting stdout is empty fails today
// (Red) and passes only once the silent index path is wired (Green). Exit-0-only
// would pass vacuously against the current binary.

/// Write a Claude session JSONL into `dir/claude` but do NOT index it. Mirrors the
/// first half of `seed_indexed_session`; the missing `index` call is deliberate —
/// in these tests *hook-handler* is the indexer (FR-003), so pre-indexing would
/// hide whether it ran. Returns the source dir to point `RECALL_CLAUDE_DIR` at.
///
/// `term` is embedded in the message so a later `search` can hit this one session
/// by a session-specific word (not the shared "hello world recall" of the other
/// fixtures, which would not prove hook-handler indexed *this* file).
fn seed_session_unindexed(dir: &Path, slug: &str, term: &str) -> PathBuf {
    let claude_dir = dir.join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(
        claude_dir.join(format!("{slug}.jsonl")),
        format!(
            r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":"{term}"}},"timestamp":"2026-03-01T00:00:00Z"}}"#
        ),
    )
    .unwrap();
    claude_dir
}

/// Spawn `recall hook-handler` with `payload` on stdin and capture all three
/// streams. The stdin handle is taken and dropped in an inner scope so the child
/// sees EOF — hook-handler reads via `serde_json::from_reader`, which blocks until
/// the pipe closes. `cmd` must already carry the `recall(dir)` env (RECALL_DB +
/// source dirs) so the index lands in the test DB, not `~/.claude`.
fn spawn_hook_handler(mut cmd: Command, payload: &[u8]) -> Output {
    let mut child = cmd
        .arg("hook-handler")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("spawn recall hook-handler");
    {
        let mut stdin = child.stdin.take().expect("hook-handler stdin is piped");
        stdin.write_all(payload).expect("write payload to stdin");
    } // drop stdin → EOF so from_reader returns
    child.wait_with_output().expect("wait for hook-handler")
}

// T-001 (FR-001/FR-002/FR-003): a full 6-field SessionEnd payload on stdin makes
// `recall hook-handler` index the seeded session and exit 0. The discriminator is
// the DB side effect, not stdout: the current argv shorthand-expands to `search
// "hook-handler"`, which routes its idle guidance to stderr and leaves stdout empty
// in both worlds — so only an indexed `sessions` row separates the real hook from
// the shorthand. The row-exists assertion is a clean Red today (shorthand-search
// opens+migrates the DB but indexes nothing → 0 rows). The empty-stdout line is a
// secondary FR-004 guard, not the Red trigger. Perspective: normal (AC1 happy path).
#[test]
fn hook_handler_full_payload_indexes_session_and_exits_zero() {
    let dir = TempDir::new().unwrap();
    let claude_dir = seed_session_unindexed(dir.path(), "s1", "zephyrine quasar token");
    let payload = br#"{"session_id":"s1","transcript_path":"/tmp/t.jsonl","cwd":"/proj","hook_event_name":"SessionEnd","reason":"clear","exit_code":0}"#;
    let out = spawn_hook_handler(
        {
            let mut c = recall(dir.path());
            c.env("RECALL_CLAUDE_DIR", &claude_dir);
            c
        },
        payload,
    );
    assert_eq!(out.status.code(), Some(0), "hook-handler should exit 0");
    let (count, _) = sessions_count_and_max_mtime(dir.path());
    assert!(
        count >= 1,
        "a full payload must FTS-index the seeded session, got {count} rows"
    );
    // TC-001: index_chunks is a distinct second stage with its own `?`; assert it
    // produced chunks (what stage-2 embed consumes), else a silent chunk no-op stays green.
    let chunks: i64 = {
        let conn = Connection::open(dir.path().join("recall.db")).unwrap();
        conn.query_row("SELECT COUNT(*) FROM qa_chunks", [], |r| r.get(0))
            .unwrap()
    };
    assert!(
        chunks >= 1,
        "hook-handler must also run index_chunks (FR-003), got {chunks} chunks"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.is_empty(),
        "hook-handler must be silent on stdout (FR-004); got: {stdout}"
    );
}

// T-002 (FR-002 input-error path): empty stdin must not crash. The parse fails, so
// hook-handler logs a warning and exits 0 without attempting an index (CON-006).
// The discriminator is the warning on stderr (the spec's "tracing 警告"): the DB
// cannot tell the two worlds apart (parse-fail indexes nothing either way) and
// stdout is empty in both, so stderr is the only observable. amici's subscriber
// writes to stderr at the default `recall=warn` filter, so the warn! shows without
// --verbose. The negative guard (no "No sessions indexed") is grounded in the
// observed shorthand-search output and stays Red regardless of the exact wording.
// Perspective: boundary (empty input).
#[test]
fn hook_handler_empty_stdin_warns_and_exits_zero() {
    let dir = TempDir::new().unwrap();
    let out = spawn_hook_handler(recall(dir.path()), b"");
    assert_eq!(
        out.status.code(),
        Some(0),
        "empty stdin is a no-op exit 0, not a crash"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("payload"),
        "empty stdin must log a parse-skip warning mentioning the payload (FR-002), got: {stderr}"
    );
    assert!(
        !stderr.contains("No sessions indexed"),
        "hook-handler must not fall through to the shorthand search path, got: {stderr}"
    );
}

// T-003 (FR-002 input-error path): malformed JSON (`{broken`) must not crash. Same
// contract as the empty case — a stderr warning + exit 0, no index attempt. The
// warning on stderr is the discriminator; the negative guard pins that the input
// never reaches the shorthand search. Perspective: error (invalid input).
#[test]
fn hook_handler_malformed_json_warns_and_exits_zero() {
    let dir = TempDir::new().unwrap();
    let out = spawn_hook_handler(recall(dir.path()), b"{broken");
    assert_eq!(
        out.status.code(),
        Some(0),
        "malformed JSON is a no-op exit 0, not a crash"
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("payload"),
        "malformed JSON must log a parse-skip warning mentioning the payload (FR-002), got: {stderr}"
    );
    assert!(
        !stderr.contains("No sessions indexed"),
        "hook-handler must not fall through to the shorthand search path, got: {stderr}"
    );
}

// T-006 (FR-003, AC3): after hook-handler runs, a follow-up `search <term>
// --no-embed` finds the session it indexed — proving hook-handler (not a prior
// `index`) built the FTS index. The fixture is deliberately NOT pre-indexed and
// uses a session-specific term, so a hit can only come from hook-handler's index
// pass. Asserting the hit ("Found 1 sessions:") guards against the vacuous pass
// where the shorthand `search` indexes nothing and returns zero results.
// Perspective: normal (end-to-end search-after-hook).
#[test]
fn hook_handler_index_makes_session_searchable() {
    let dir = TempDir::new().unwrap();
    let claude_dir = seed_session_unindexed(dir.path(), "s1", "zephyrine quasar token");
    let hook = spawn_hook_handler(
        {
            let mut c = recall(dir.path());
            c.env("RECALL_CLAUDE_DIR", &claude_dir);
            c
        },
        br#"{"session_id":"s1","transcript_path":"/tmp/t.jsonl","cwd":"/proj","hook_event_name":"SessionEnd","reason":"clear","exit_code":0}"#,
    );
    assert_eq!(hook.status.code(), Some(0), "hook-handler should exit 0");

    let out = recall(dir.path())
        .args(["search", "zephyrine", "--no-embed"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("Found 1 sessions:"),
        "hook-handler must FTS-index the session so search hits it, got: {stdout}"
    );
}

/// `MAX(mtime)` and row count of the `sessions` table in the test DB, opened
/// read-only via the same path `recall(dir)` sets in `RECALL_DB`. Used by T-008 to
/// verify the second hook-handler run is a near-no-op at the DB level, independent
/// of stdout. `mtime` is the source file's modified time (indexer stores
/// fs::metadata().modified()), so an unchanged JSONL keeps it constant.
fn sessions_count_and_max_mtime(dir: &Path) -> (i64, f64) {
    let conn = Connection::open(dir.join("recall.db")).expect("open test DB");
    conn.query_row(
        "SELECT COUNT(*), COALESCE(MAX(mtime), 0.0) FROM sessions",
        [],
        |row| Ok((row.get(0)?, row.get(1)?)),
    )
    .expect("query sessions count and max mtime")
}

// T-008 (FR-003 / NFR-004): running hook-handler twice on the same unchanged tree
// leaves `sessions` unchanged — COUNT(*) and MAX(mtime) after the second run equal
// the first. This validates the idempotent near-no-op at the DB level, so it does
// not depend on stdout suppression. The baseline `count1 >= 1` assertion guards the
// vacuous `0 == 0` pass: against a binary that indexes nothing, both runs would
// store zero sessions and the equality would hold for the wrong reason.
// Perspective: hazard (double-fire) + state (re-run is a no-op).
#[test]
fn hook_handler_double_run_is_idempotent_on_sessions_table() {
    let dir = TempDir::new().unwrap();
    let claude_dir = seed_session_unindexed(dir.path(), "s1", "zephyrine quasar token");
    let payload = br#"{"session_id":"s1","transcript_path":"/tmp/t.jsonl","cwd":"/proj","hook_event_name":"SessionEnd","reason":"clear","exit_code":0}"#;
    let spawn = |claude: &Path| {
        let mut c = recall(dir.path());
        c.env("RECALL_CLAUDE_DIR", claude);
        c
    };

    let first = spawn_hook_handler(spawn(&claude_dir), payload);
    assert_eq!(first.status.code(), Some(0), "first run should exit 0");
    let (count1, mtime1) = sessions_count_and_max_mtime(dir.path());
    assert!(
        count1 >= 1,
        "the first hook-handler run must index the seeded session (got {count1} rows)"
    );

    let second = spawn_hook_handler(spawn(&claude_dir), payload);
    assert_eq!(second.status.code(), Some(0), "second run should exit 0");
    let (count2, mtime2) = sessions_count_and_max_mtime(dir.path());

    assert_eq!(
        count2, count1,
        "a second run on the unchanged tree must not add sessions rows"
    );
    assert_eq!(
        mtime2, mtime1,
        "an unchanged JSONL keeps MAX(mtime) constant across re-runs (near-no-op)"
    );
}

// T-009 (FR-005, AC4): `recall hook-handler --help` documents the three hook-setup
// items so the docs are the mitigation for the un-detectable silent failure
// (CON-001): (a) the settings.json registration example, (b) the `which recall`
// PATH caveat for `sh -c`, (c) the Codex manual-index guidance. Today the argv
// shorthand-expands to `search --help`, whose text carries none of these, so the
// assertions fail until hook-handler owns its help. Perspective: normal (docs
// content contract).
#[test]
fn hook_handler_help_documents_registration_path_and_codex() {
    let dir = TempDir::new().unwrap();
    let out = recall(dir.path())
        .args(["hook-handler", "--help"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "--help exits 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        stdout.contains("settings.json"),
        "(a) help must show the SessionEnd settings.json example, got: {stdout}"
    );
    assert!(
        stdout.contains("which recall"),
        "(b) help must carry the `which recall` PATH caveat, got: {stdout}"
    );
    assert!(
        stdout.contains("Codex"),
        "(c) help must point Codex users at manual indexing, got: {stdout}"
    );
}

// T-010 (FR-002 DB-error path): a valid payload but an unopenable DB (RECALL_DB
// points at a directory, which SQLite cannot open as a file) makes hook-handler
// propagate the failure as a nonzero exit — the boundary's DB-error side, opposite
// the input-error path (T-002/T-003) that exits 0. Pins that index/DB failures are
// not swallowed by the hook. Perspective: error.
#[test]
fn hook_handler_db_error_exits_nonzero() {
    let dir = TempDir::new().unwrap();
    let out = spawn_hook_handler(
        {
            let mut c = recall(dir.path());
            // Override RECALL_DB to the temp dir itself: opening a directory as a
            // SQLite file fails, exercising the DB-error propagation path.
            c.env("RECALL_DB", dir.path());
            c
        },
        br#"{"session_id":"s1","hook_event_name":"SessionEnd"}"#,
    );
    assert!(
        !out.status.success(),
        "DB open failure must exit nonzero (FR-002 boundary), got: {:?}",
        out.status
    );
}
