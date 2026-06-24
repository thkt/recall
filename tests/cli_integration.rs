//! CLI exit-code contract (recall ADR-0006, extending scout ADR-0066 Group 2; #82). Spawns the real binary and
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
        exit_code(recall(dir.path()).args(["search", "anything"])),
        0
    );
}

// T-CLI007: a bare query with no subcommand is shorthand-expanded to `search`,
// exercising the shorthand-expansion parse branch. Empty index → success (0).
#[test]
fn bare_query_shorthand_expands_to_search() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).args(["someword"])), 0);
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
        .args(["search", "hello"])
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
        .args(["search", "hello", "--json"])
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
    let human_code = exit_code(recall(human.path()).args(["search", "anything"]));

    let json = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(json.path()).arg("index")), 0);
    let json_code = exit_code(recall(json.path()).args(["search", "anything", "--json"]));

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
        .args(["search", "hello"])
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

// T-CLI018 (#130 D2/FR-008): `index --json` emits a structured embed summary —
// `data` carries `{ embedded, failed_count }` (no longer null) so an agent reads
// how much embedding happened, and exits 0. An empty index embeds nothing and
// fails nothing regardless of whether the model is present, so the counts are a
// deterministic 0/0; the degraded flag and notes depend on model presence and are
// covered deterministically by the `index_command_output` unit tests (T-003..006).
// Progress spinners go to stderr, so stdout is exactly the one envelope.
#[test]
fn index_json_emits_structured_embed_summary() {
    let dir = TempDir::new().unwrap();
    let out = recall(dir.path())
        .args(["index", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "index --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    assert_eq!(
        v["data"]["embedded"],
        serde_json::json!(0),
        "an empty index embeds nothing, got: {stdout}"
    );
    assert_eq!(
        v["data"]["failed_count"],
        serde_json::json!(0),
        "an empty index has no embed failures, got: {stdout}"
    );
}

// T-CLI027: `rebuild --json` mirrors the index envelope contract through the
// rebuild dispatch arm (run_rebuild → index_command_output). On an empty index
// the wipe-and-rebuild embeds nothing and fails nothing, so the counts are a
// deterministic 0/0 regardless of model presence, and the exit is 0.
#[test]
fn rebuild_json_emits_structured_embed_summary() {
    let dir = TempDir::new().unwrap();
    let out = recall(dir.path())
        .args(["rebuild", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "rebuild --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    assert_eq!(
        v["data"]["embedded"],
        serde_json::json!(0),
        "an empty rebuild embeds nothing, got: {stdout}"
    );
    assert_eq!(
        v["data"]["failed_count"],
        serde_json::json!(0),
        "an empty rebuild has no embed failures, got: {stdout}"
    );
}

// T-CLI028 (ADR-0001, #211): `doctor --json` against a built index freezes the
// diagnostic envelope an agent consumes. The `data` payload pins the top keys
// `healthy` / `checks`, each check object's `name` / `ok` / `detail` / `remedy`,
// and the four check `name` tokens (`integrity` / `orphan_embeddings` /
// `orphan_chunks` / `model`) an agent pattern-matches. A valid seeded DB always
// takes the four-check arm (open succeeds), so the name set is deterministic; the
// model check's verdict still varies by host, so this asserts the envelope
// invariant `degraded == !healthy` rather than a fixed `healthy` value, keeping
// the test portable. Exit is 0 because a successful command maps to SUCCESS
// regardless of the degraded flag (exit_code_for_write). Perspective: contract
// freeze (key set) + invariant (degraded co-varies with healthy).
#[test]
fn doctor_json_freezes_diagnostic_envelope_key_set() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let out = recall(dir.path())
        .args(["doctor", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "doctor --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    let healthy = v["data"]["healthy"]
        .as_bool()
        .unwrap_or_else(|| panic!("data.healthy should be a bool, got: {stdout}"));
    assert!(
        v["data"]
            .get("repaired")
            .is_some_and(serde_json::Value::is_null),
        "a read-only doctor (no --fix) freezes repaired=null, got: {stdout}"
    );
    let degraded = v["degraded"]
        .as_bool()
        .unwrap_or_else(|| panic!("degraded should be a bool, got: {stdout}"));
    assert_eq!(
        degraded, !healthy,
        "envelope degraded must be the negation of data.healthy, got: {stdout}"
    );
    let checks = v["data"]["checks"]
        .as_array()
        .unwrap_or_else(|| panic!("data.checks should be an array, got: {stdout}"));
    for c in checks {
        let obj = c
            .as_object()
            .unwrap_or_else(|| panic!("each check should be an object, got: {stdout}"));
        for key in ["name", "ok", "detail", "remedy"] {
            assert!(
                obj.contains_key(key),
                "each check must carry the `{key}` key, got: {stdout}"
            );
        }
    }
    let mut names: Vec<&str> = checks
        .iter()
        .map(|c| {
            c["name"]
                .as_str()
                .unwrap_or_else(|| panic!("check name should be a string, got: {stdout}"))
        })
        .collect();
    names.sort_unstable();
    assert_eq!(
        names,
        ["integrity", "model", "orphan_chunks", "orphan_embeddings"],
        "a built index freezes exactly these four check tokens, got: {stdout}"
    );
}

// T-CLI019: `status --json` against a never-created index reports zero counters
// as a success envelope, exit 0 — the no-database branch of run_status. With
// HF_HOME pointed at an empty cache the model is not installed, so model_ready is
// false here; T-CLI020 covers the same branch when the model IS cached. Pairs
// with T-CLI013, which runs `index` first (the live path).
#[test]
fn status_json_without_index_reports_zero_counts() {
    let dir = TempDir::new().unwrap();
    let hf_home = TempDir::new().unwrap();
    let out = recall(dir.path())
        .env("HF_HOME", hf_home.path())
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
        "an empty HF cache means the model is not installed, got: {stdout}"
    );
}

/// Populate an HF Hub cache under `hf_home` with valid DEFAULT-model artifacts so
/// `cached_artifacts(ModelId::DEFAULT)` verifies as ready. Builds the
/// `hub/models--{slug}/refs/{revision}` → `snapshots/{hash}/` layout
/// `hf_hub::Cache::from_env` expects (root is `$HF_HOME/hub`). The fixtures mirror
/// rurico's `test_support` (VALID_CONFIG_JSON / MINIMAL_TOKENIZER_JSON /
/// FAKE_BACKBONE_KEY), which is `pub(crate)` and so unreachable from here.
fn setup_cached_default_model(hf_home: &Path) {
    // ruri-v3-310m is ModelId::DEFAULT; the revision keys the refs file name.
    let repo_slug = "cl-nagoya--ruri-v3-310m";
    let revision = "18b60fb8c2b9df296fb4212bb7d23ef94e579cd3";
    let commit = "abc123";
    let repo_dir = hf_home.join("hub").join(format!("models--{repo_slug}"));
    let refs_dir = repo_dir.join("refs");
    fs::create_dir_all(&refs_dir).unwrap();
    fs::write(refs_dir.join(revision), commit).unwrap();
    let snapshot = repo_dir.join("snapshots").join(commit);
    fs::create_dir_all(&snapshot).unwrap();

    // config.json must parse as a valid ModernBERT Config (artifacts.rs verify_config).
    let config = r#"{
        "vocab_size": 1000,
        "hidden_size": 768,
        "num_hidden_layers": 2,
        "num_attention_heads": 12,
        "intermediate_size": 3072,
        "max_position_embeddings": 512,
        "layer_norm_eps": 1e-5,
        "pad_token_id": 0,
        "global_attn_every_n_layers": 3,
        "global_rope_theta": 160000.0,
        "local_attention": 128,
        "local_rope_theta": 10000.0
    }"#;
    fs::write(snapshot.join("config.json"), config).unwrap();

    // Minimal BPE tokenizer accepted by tokenizers 0.22+ (verify_tokenizer).
    let tokenizer = r#"{
        "version": "1.0",
        "model": {"type": "BPE", "vocab": {}, "merges": []},
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": null,
        "post_processor": null,
        "decoder": null,
        "truncation": null,
        "padding": null
    }"#;
    fs::write(snapshot.join("tokenizer.json"), tokenizer).unwrap();

    // safetensors carrying one `layers.` backbone key (verify_embed_kind) and no
    // reranker key, with one f32 of weight data.
    fs::write(
        snapshot.join("model.safetensors"),
        fake_safetensors(&["layers.0.attn.Wo.weight"]),
    )
    .unwrap();
}

/// Structurally valid safetensors with the given tensor keys, each one f32 of
/// data. Mirrors rurico `test_support::write_fake_safetensors`.
fn fake_safetensors(keys: &[&str]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    header.insert("__metadata__".to_owned(), serde_json::json!({}));
    let mut offset = 0usize;
    for &key in keys {
        let end = offset + 4;
        header.insert(
            key.to_owned(),
            serde_json::json!({"dtype": "F32", "shape": [1], "data_offsets": [offset, end]}),
        );
        offset = end;
    }
    let header_json = serde_json::to_vec(&header).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(header_json.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_json);
    for _ in keys {
        out.extend_from_slice(&0f32.to_le_bytes());
    }
    out
}

// T-CLI020: `status --json` reports model_ready:true when the model artifacts are
// cached but no index DB exists (#166). The no-database branch must reflect the
// real HF cache state, not a hard-coded false — an agent can `recall model
// download` before its first `recall index`. Counts stay zero (no DB). Pairs with
// T-CLI019 (empty cache → false) and exercises the regression directly.
// Perspective: hazard (model installed before index) + branch (no-DB true path).
#[test]
fn status_json_without_index_reports_model_ready_when_cached() {
    let dir = TempDir::new().unwrap();
    let hf_home = TempDir::new().unwrap();
    setup_cached_default_model(hf_home.path());
    let out = recall(dir.path())
        .env("HF_HOME", hf_home.path())
        .args(["status", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "status --json should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let v: serde_json::Value = serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"));
    assert_eq!(
        v["data"]["model_ready"],
        serde_json::json!(true),
        "no-database branch must report the cached model as ready, got: {stdout}"
    );
    assert_eq!(
        v["data"]["sessions"],
        serde_json::json!(0),
        "status must not create the DB; counts stay zero, got: {stdout}"
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
        .args(["search", "hello"])
        .output()
        .expect("spawn recall binary");
    let baseline_out = String::from_utf8_lossy(&baseline.stdout);
    assert!(
        baseline_out.contains("ID: s1") && baseline_out.contains("ID: s2"),
        "baseline (no env) should list both sessions, got: {baseline_out}"
    );

    // CLAUDE_CODE_SESSION_ID=s1 → s1 is the invoking session, excluded by default.
    let out = recall(dir.path())
        .args(["search", "hello"])
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
        .args(["search", "hello", "--include-current"])
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
        .args(["search", "hello", "--only-current"])
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
        exit_code(recall(dir.path()).args(["search", "hello", "--only-current"])),
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
        .args(["search", "authentication", "--json"])
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
        .args(["search", "authentication", "--json"])
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        !stdout.contains("--include-automated"),
        "no automated sessions means no exclusion note: {stdout}"
    );
}

// -- #73 段階2 Phase 4 (AC-3): --no-embed retired (FR-006) --
//
// FR-005 (the retired `embed` subcommand) is verified by a main.rs unit test
// (embed_token_shorthand_expands_to_search): with "embed" dropped from
// KNOWN_SUBCOMMANDS, `recall embed` shorthand-expands to a search instead of
// erroring, so the observable is subcommand membership, not an exit code.

// T-006 (FR-006): `recall search <q> --no-embed` is a usage error (64). With the
// --no-embed arg definition removed, the flag is unknown: clap fails to parse, and
// handle_parse_error (main.rs:1051) maps every non-help clap error to the sysexits
// USAGE code (64) — the same mapping T-CLI001 pins for `--no-such-flag`. A bare
// query is supplied so the failure is the unknown flag, not a missing positional.
// This spawns the real binary because the parse/dispatch/exit glue is reachable
// only end to end. Perspective: error (the retired flag must fail, not be silently
// ignored).
#[test]
fn search_no_embed_flag_is_retired_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["search", "anything", "--no-embed"])),
        64
    );
}
