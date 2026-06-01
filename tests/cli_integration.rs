//! CLI exit-code contract (ADR-0066 Group 2, #82). Spawns the real binary and
//! pins the sysexits codes agents branch on, exercising the parse / dispatch /
//! error-classification glue that unit tests cannot reach in-process.

use std::fs;
use std::path::Path;
use std::process::Command;

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
