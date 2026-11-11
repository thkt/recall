use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

use tempfile::TempDir;

#[cfg(unix)]
#[path = "support/root_skip.rs"]
mod root_skip;

fn recall(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_recall"));
    cmd.env("RECALL_DB", dir.join("recall.db"))
        .env("RECALL_MODEL_DIR", dir.join("no-model"))
        .env_remove("RECALL_MODEL_MANIFEST")
        .env("RECALL_CLAUDE_DIR", dir.join("no-claude"))
        .env("RECALL_CODEX_DIR", dir.join("no-codex"))
        .env_remove("RECALL_CODEX_ARCHIVED_DIR")
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

#[test]
fn unknown_flag_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("--no-such-flag")), 64);
}

#[test]
fn no_subcommand_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(&mut recall(dir.path())), 64);
}

#[test]
fn help_exits_success() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("--help")), 0);
}

#[test]
fn show_without_database_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).args(["show", "deadbeef"])), 64);
}

#[test]
fn search_empty_index_exits_success() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["search", "anything"])),
        0
    );
    assert!(
        !dir.path().join("recall.db").exists(),
        "missing-DB search must not create the index file"
    );
}

#[test]
fn bare_query_shorthand_expands_to_search() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).args(["someword"])), 0);
}

#[test]
fn show_unknown_id_in_existing_index_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);
    assert_eq!(exit_code(recall(dir.path()).args(["show", "deadbeef"])), 64);
}

#[test]
fn status_prints_stats_and_exits_success() {
    let dir = TempDir::new().unwrap();
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

#[test]
fn upgrading_an_old_embedding_table_tells_the_user_to_run_recall_index() {
    let dir = TempDir::new().unwrap();
    rusqlite::Connection::open(dir.path().join("recall.db"))
        .unwrap()
        .execute_batch("CREATE TABLE vec_chunks (chunk_id INTEGER, embedding BLOB);")
        .unwrap();

    let out = recall(dir.path())
        .arg("index")
        .output()
        .expect("spawn recall binary");

    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(out.status.code(), Some(0), "{stderr}");
    assert!(stderr.contains("embeddings schema changed"), "{stderr}");
    assert!(stderr.contains("run `recall index`"), "{stderr}");
}

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

#[test]
fn status_json_emits_stats_envelope() {
    let dir = TempDir::new().unwrap();
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
    let stderr = String::from_utf8_lossy(&out.stderr);
    for stage in [
        "model_load_probe",
        "enumeration",
        "fts_transaction",
        "chunking",
        "pending_extraction",
    ] {
        assert!(
            stderr.contains(&format!("index: {stage}: started")),
            "{stderr}"
        );
    }
    assert!(v["data"]["observations"]["seconds"]["total"].is_number());
    assert_eq!(v["data"]["observations"]["counts"]["files_discovered"], 0);
}

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

fn setup_cached_default_model(root: &Path) -> PathBuf {
    let output = recall(root).args(["model", "manifest"]).output().unwrap();
    assert!(output.status.success());
    let mut spec: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    for (name, key, bytes, checksum) in [
        (
            "model.gguf",
            "model_sha256",
            b"fixture GGUF".as_slice(),
            "b28d94311aea5e81429ea07fc7a4af5c118665010865bf8bd9114d17a23f0820",
        ),
        (
            "tokenizer.json",
            "tokenizer_sha256",
            b"{}".as_slice(),
            "44136fa355b3678a1146ad16f7e8649e94fb4fc21fe77e8310c060f61caaff8a",
        ),
    ] {
        fs::write(root.join(name), bytes).unwrap();
        spec[key] = checksum.into();
    }
    let path = root.join("manifest.json");
    fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    path
}

#[test]
fn status_json_without_index_reports_model_ready_when_cached() {
    let dir = TempDir::new().unwrap();
    let hf_home = TempDir::new().unwrap();
    let manifest = setup_cached_default_model(hf_home.path());
    let out = recall(dir.path())
        .env("HF_HOME", hf_home.path())
        .env("RECALL_MODEL_DIR", hf_home.path())
        .arg("--model-manifest")
        .arg(manifest)
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

#[test]
fn search_excludes_current_session_by_env_default() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    let baseline = recall(dir.path())
        .args(["search", "hello"])
        .output()
        .expect("spawn recall binary");
    let baseline_out = String::from_utf8_lossy(&baseline.stdout);
    assert!(
        baseline_out.contains("ID: s1") && baseline_out.contains("ID: s2"),
        "baseline (no env) should list both sessions, got: {baseline_out}"
    );

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

#[test]
fn only_current_without_env_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["search", "hello", "--only-current"])),
        64
    );
}

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

#[test]
fn classify_preview_of_missing_database_keeps_the_path_absent() {
    let dir = TempDir::new().unwrap();
    for path in [
        dir.path().join("recall.db"),
        dir.path().join("absent/nested/recall.db"),
    ] {
        for all in [false, true] {
            let mut cmd = recall(dir.path());
            cmd.arg("--db-path")
                .arg(&path)
                .args(["classify", "--dry-run", "--json"]);
            if all {
                cmd.arg("--all");
            }
            let out = cmd.output().unwrap();
            assert_eq!(out.status.code(), Some(0), "{:?}", out.stderr);
            let envelope: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
            assert_eq!(
                envelope,
                serde_json::json!({
                    "data": {"classified":0,"automated":0,"interactive":0,"dry_run":true},
                    "degraded": false, "notes": []
                })
            );
            assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 0);
        }
    }
}

fn index_claude_sessions(dir: &Path, sessions: &[(&str, &[&str])]) {
    let claude_dir = dir.join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    for (name, lines) in sessions {
        fs::write(claude_dir.join(format!("{name}.jsonl")), lines.join("\n")).unwrap();
    }
    assert_eq!(
        exit_code(
            recall(dir)
                .env("RECALL_CLAUDE_DIR", &claude_dir)
                .arg("index")
        ),
        0
    );
}

fn search_stdout(dir: &Path, args: &[&str]) -> String {
    let out = recall(dir)
        .arg("search")
        .args(args)
        .arg("--json")
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "search should exit 0");
    String::from_utf8_lossy(&out.stdout).into_owned()
}

const AUTOMATED_OPENER: &str = r#"{"type":"user","cwd":"/other","message":{"role":"user","content":"<command-message>clear</command-message>"},"timestamp":"2026-03-01T00:00:00Z"}"#;
const AUTOMATED_AUTH_TURN: &str = r#"{"type":"user","cwd":"/other","message":{"role":"user","content":"authentication flow review"},"timestamp":"2026-03-01T00:01:00Z"}"#;
const HUMAN_AUTH: &str = r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"authentication notes"},"timestamp":"2026-03-01T00:00:00Z"}"#;
const HUMAN_UNRELATED: &str = r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"deployment checklist"},"timestamp":"2026-03-01T00:00:00Z"}"#;

#[test]
fn search_notes_automated_sessions_that_match_the_query() {
    let dir = TempDir::new().unwrap();
    index_claude_sessions(
        dir.path(),
        &[
            ("auto", &[AUTOMATED_OPENER, AUTOMATED_AUTH_TURN]),
            ("human", &[HUMAN_AUTH]),
            ("human2", &[HUMAN_AUTH]),
        ],
    );

    let stdout = search_stdout(dir.path(), &["authentication"]);

    assert!(
        stdout.contains("1 automated session(s)") && stdout.contains("--include-automated"),
        "a matching automated session must be counted in the note: {stdout}"
    );
}

#[test]
fn search_omits_automated_note_when_no_automated_session_matches_the_query() {
    let dir = TempDir::new().unwrap();
    index_claude_sessions(
        dir.path(),
        &[("auto", &[AUTOMATED_OPENER]), ("human", &[HUMAN_AUTH])],
    );

    let stdout = search_stdout(dir.path(), &["authentication"]);

    assert!(
        !stdout.contains("--include-automated"),
        "an automated session that does not match the query was not excluded from it: {stdout}"
    );
}

#[test]
fn search_omits_automated_note_when_filters_exclude_the_automated_match() {
    let dir = TempDir::new().unwrap();
    index_claude_sessions(
        dir.path(),
        &[
            ("auto", &[AUTOMATED_OPENER, AUTOMATED_AUTH_TURN]),
            ("human", &[HUMAN_AUTH]),
        ],
    );

    let stdout = search_stdout(dir.path(), &["authentication", "--project", "/proj"]);

    assert!(
        !stdout.contains("--include-automated"),
        "--project /proj already rules out the /other automated session: {stdout}"
    );
}

#[test]
fn search_notes_automated_match_even_when_no_interactive_session_matches() {
    let dir = TempDir::new().unwrap();
    index_claude_sessions(
        dir.path(),
        &[
            ("auto", &[AUTOMATED_OPENER, AUTOMATED_AUTH_TURN]),
            ("human", &[HUMAN_UNRELATED]),
        ],
    );

    let stdout = search_stdout(dir.path(), &["authentication"]);

    assert!(
        stdout.contains(r#""results":[]"#) && stdout.contains("1 automated session(s)"),
        "with no interactive hit, the note is the only pointer to the automated match: {stdout}"
    );
}

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

#[test]
fn search_no_embed_flag_is_retired_exits_usage() {
    let dir = TempDir::new().unwrap();
    assert_eq!(
        exit_code(recall(dir.path()).args(["search", "anything", "--no-embed"])),
        64
    );
}

#[cfg(unix)]
#[test]
fn read_commands_work_in_readonly_directory() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    let db = dir.path().join("recall.db");
    for suffix in ["-wal", "-shm"] {
        let _ = fs::remove_file(db.with_file_name(format!("recall.db{suffix}")));
    }

    let set_mode = |mode: u32| {
        fs::set_permissions(dir.path(), fs::Permissions::from_mode(mode)).unwrap();
    };
    set_mode(0o555);

    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let search_out = recall(dir.path())
        .args(["search", "hello"])
        .output()
        .expect("spawn recall binary");
    let status_out = recall(dir.path())
        .arg("status")
        .output()
        .expect("spawn recall binary");
    let classify_out = recall(dir.path())
        .args(["classify", "--all", "--dry-run"])
        .output()
        .expect("spawn recall binary");

    set_mode(0o755);

    for (name, out) in [
        ("search", &search_out),
        ("status", &status_out),
        ("classify", &classify_out),
    ] {
        let stderr = String::from_utf8_lossy(&out.stderr);
        assert!(
            !stderr.contains("Error code 14"),
            "{name} must not emit SQLite code 14 in a read-only dir; stderr: {stderr}"
        );
        assert_eq!(
            out.status.code(),
            Some(0),
            "{name} should exit 0 in a read-only dir; stderr: {stderr}"
        );
    }

    for suffix in ["-wal", "-shm"] {
        let side = db.with_file_name(format!("recall.db{suffix}"));
        assert!(
            !side.exists(),
            "read-only open must not create recall.db{suffix}"
        );
    }
}

/// Indexes one session, then leaves a nonempty `-wal` beside the DB the way an
/// interrupted writer does, so a read from a read-only directory sees a stale snapshot.
fn seed_stale_wal(dir: &Path) {
    seed_indexed_session(dir);
    let db = dir.join("recall.db");
    let _ = fs::remove_file(db.with_file_name("recall.db-shm"));
    fs::write(db.with_file_name("recall.db-wal"), b"uncheckpointed-frames").unwrap();
}

/// Makes a directory read-only until dropped, so a failed assertion still leaves
/// `TempDir` able to delete it.
#[cfg(unix)]
struct ReadOnlyDir<'a>(&'a Path);

#[cfg(unix)]
impl<'a> ReadOnlyDir<'a> {
    fn new(dir: &'a Path) -> Self {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).unwrap();
        Self(dir)
    }
}

#[cfg(unix)]
impl Drop for ReadOnlyDir<'_> {
    fn drop(&mut self) {
        use std::os::unix::fs::PermissionsExt;
        let _ = fs::set_permissions(self.0, fs::Permissions::from_mode(0o755));
    }
}

#[cfg(unix)]
#[test]
fn show_on_an_immutable_open_with_stale_wal_surfaces_degraded_note() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());
    let _read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let out = recall(dir.path())
        .args(["show", "s1", "--json"])
        .output()
        .expect("spawn recall binary");

    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(envelope["degraded"], true, "{envelope}");
    assert_eq!(
        envelope["notes"].as_array().map(Vec::len),
        Some(1),
        "{envelope}"
    );
    root_skip::assert_copy_based_remedy(envelope["notes"][0].as_str().unwrap(), "show");
}

#[cfg(unix)]
#[test]
fn show_of_a_session_missing_from_a_stale_snapshot_reports_the_write_ahead_log() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());
    let _read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let out = recall(dir.path())
        .args(["show", "s2", "--json"])
        .output()
        .expect("spawn recall binary");

    assert_eq!(out.status.code(), Some(65));
    assert!(out.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&out.stderr).unwrap();
    assert_eq!(error["error"]["code"], "DATA_ERROR");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(
        message.contains("No session found matching 's2'"),
        "{message}"
    );
    root_skip::assert_copy_based_remedy(message, "show");
}

#[test]
fn show_on_an_index_from_an_older_version_asks_for_a_rebuild() {
    let dir = TempDir::new().unwrap();
    rusqlite::Connection::open(dir.path().join("recall.db"))
        .unwrap()
        .execute_batch("CREATE TABLE sessions (session_id TEXT PRIMARY KEY);")
        .unwrap();

    let out = recall(dir.path())
        .args(["show", "s1", "--json"])
        .output()
        .expect("spawn recall binary");

    assert_eq!(out.status.code(), Some(65));
    let error: serde_json::Value = serde_json::from_slice(&out.stderr).unwrap();
    assert_eq!(error["error"]["code"], "DATA_ERROR");
    let message = error["error"]["message"].as_str().unwrap();
    assert!(message.contains("run `recall rebuild`"), "{message}");
}

#[test]
fn show_from_a_writable_directory_reads_the_write_ahead_log_without_a_note() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());

    let out = recall(dir.path())
        .args(["show", "s1", "--json"])
        .output()
        .expect("spawn recall binary");

    assert_eq!(
        out.status.code(),
        Some(0),
        "{}",
        String::from_utf8_lossy(&out.stderr)
    );
    let envelope: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(envelope["degraded"], false, "{envelope}");
    assert_eq!(envelope["notes"], serde_json::json!([]), "{envelope}");
}

#[cfg(unix)]
#[test]
fn immutable_open_with_stale_wal_surfaces_degraded_note() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());
    let read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    let outputs = [
        (
            "search",
            recall(dir.path())
                .args(["search", "hello", "--json"])
                .output(),
        ),
        (
            "status",
            recall(dir.path()).args(["status", "--json"]).output(),
        ),
        (
            "doctor",
            recall(dir.path()).args(["doctor", "--json"]).output(),
        ),
    ];
    let outputs: Vec<(&str, Output)> = outputs
        .into_iter()
        .map(|(name, out)| (name, out.expect("spawn recall binary")))
        .collect();
    let classify_out = recall(dir.path())
        .args(["classify", "--all", "--dry-run", "--json"])
        .output()
        .expect("spawn recall binary");

    drop(read_only);

    assert_eq!(
        classify_out.status.code(),
        Some(65),
        "{}",
        String::from_utf8_lossy(&classify_out.stderr)
    );
    assert!(classify_out.stdout.is_empty());
    let error: serde_json::Value = serde_json::from_slice(&classify_out.stderr).unwrap();
    assert_eq!(error["error"]["code"], "DATA_ERROR");
    assert_eq!(error["error"]["retryable"], false);
    assert!(error["error"].get("next_step").is_none(), "{error}");
    let message = error["error"]["message"].as_str().unwrap();
    root_skip::assert_copy_based_remedy(message, "classify");

    for (name, out) in &outputs {
        let stdout = String::from_utf8_lossy(&out.stdout);
        let v: serde_json::Value = serde_json::from_str(stdout.trim()).unwrap_or_else(|e| {
            panic!("{name} stdout should be one JSON envelope, got {stdout:?}: {e}")
        });
        assert_eq!(
            v["degraded"].as_bool(),
            Some(true),
            "{name} must mark degraded on an immutable open with a stale -wal, got: {stdout}"
        );
        let notes = v["notes"]
            .as_array()
            .unwrap_or_else(|| panic!("{name} notes should be an array, got: {stdout}"));
        assert!(
            notes
                .iter()
                .any(|n| n.as_str().is_some_and(|s| s.contains("write-ahead log"))),
            "{name} notes must carry the stale write-ahead-log warning, got: {stdout}"
        );

        let remedy = notes
            .iter()
            .find_map(|n| n.as_str().filter(|s| s.contains("write-ahead log")))
            .unwrap_or_else(|| {
                panic!("{name} should carry the stale write-ahead-log note, got: {stdout}")
            });
        root_skip::assert_copy_based_remedy(remedy, name);
    }
}

#[cfg(unix)]
#[test]
fn root_probe_文字列は_skip_if_root_の定義以外に重複しない() {
    root_skip::assert_probe_literal_not_inlined(
        include_str!("cli_integration.rs"),
        "tests/cli_integration.rs",
    );
}

#[cfg(unix)]
#[test]
fn integration_canary_は非_root_で_green_root_では_panic_で赤化する() {
    root_skip::assert_root_canary();
}

#[test]
fn index_reports_parse_loss_on_json_and_human_surfaces_until_source_repair() {
    let dir = TempDir::new().unwrap();
    let claude = dir.path().join("claude");
    let codex = dir.path().join("codex");
    fs::create_dir(&claude).unwrap();
    fs::create_dir(&codex).unwrap();
    let path = claude.join("partial.jsonl");
    let good = b"{\"type\":\"user\",\"message\":\"secret-conversation-body\"}\n";
    let mut contents = good.to_vec();
    contents
        .extend_from_slice(b"{broken-secret-body\n\xff\n{\"type\":\"progress\"}\n{\"message\":");
    fs::write(&path, contents).unwrap();
    fs::write(codex.join("ignored.jsonl"), "{\"type\":\"event_msg\"}\n").unwrap();
    let run = |args: &[&str]| {
        recall(dir.path())
            .env("HF_HOME", dir.path().join("empty-cache"))
            .env("RECALL_CLAUDE_DIR", &claude)
            .env("RECALL_CODEX_DIR", &codex)
            .args(args)
            .output()
            .unwrap()
    };
    for command in ["index", "index", "rebuild"] {
        let out = run(&[command, "--json"]);
        assert_eq!(out.status.code(), Some(0));
        let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
        let mut keys: Vec<_> = v.as_object().unwrap().keys().map(String::as_str).collect();
        keys.sort_unstable();
        assert_eq!(keys, ["data", "degraded", "notes"]);
        assert_eq!(v["degraded"], true);
        let diagnostics = v["data"]["parse_diagnostics"].as_array().unwrap();
        assert_eq!(diagnostics.len(), 1, "excluded Codex events are not loss");
        assert_eq!(
            diagnostics[0],
            serde_json::json!({
                "file_path": path.to_str().unwrap(), "source": "claude",
                "invalid_json_lines": 1, "invalid_utf8_lines": 1,
                "incomplete_tail_lines": 1, "read_error": false,
            })
        );
        let notes = v["notes"].as_array().unwrap();
        assert!(
            notes
                .iter()
                .any(|n| n.as_str().unwrap().contains("recall model download"))
        );
        assert!(
            notes
                .iter()
                .any(|n| n.as_str().unwrap().contains("repair the source"))
        );
        assert!(
            notes
                .iter()
                .any(|n| n.as_str().unwrap().contains("next append"))
        );
        assert!(!String::from_utf8_lossy(&out.stdout).contains("secret-"));
        assert!(!String::from_utf8_lossy(&out.stderr).contains("secret-"));
    }
    let out = run(&["index"]);
    assert_eq!(out.status.code(), Some(0));
    let human = format!(
        "{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    assert!(human.contains("1 invalid JSON") && human.contains("1 invalid UTF-8"));
    assert!(human.contains("1 incomplete final line") && human.contains("next append"));
    assert!(!human.contains("secret-"));
    fs::write(
        &path,
        [
            good.as_slice(),
            b"{\"type\":\"assistant\",\"message\":\"repaired-answer\"}\n",
        ]
        .concat(),
    )
    .unwrap();
    let out = run(&["index", "--json"]);
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["data"]["parse_diagnostics"], serde_json::json!([]));
    assert_eq!(
        v["degraded"], true,
        "model absence still degrades the repaired index"
    );
    let out = run(&["show", "partial", "--json"]);
    assert_eq!(out.status.code(), Some(0));
    let body = String::from_utf8_lossy(&out.stdout);
    assert!(body.contains("secret-conversation-body") && body.contains("repaired-answer"));
}

#[cfg(unix)]
#[test]
fn index_reports_real_file_read_failure_then_recovers() {
    use std::os::unix::fs::PermissionsExt;

    let dir = TempDir::new().unwrap();
    let claude = dir.path().join("claude");
    fs::create_dir(&claude).unwrap();
    let path = claude.join("unreadable.jsonl");
    fs::write(
        &path,
        "{\"type\":\"user\",\"message\":\"readable-again\"}\n",
    )
    .unwrap();
    let run = || {
        recall(dir.path())
            .env("HF_HOME", dir.path().join("empty-cache"))
            .env("RECALL_CLAUDE_DIR", &claude)
            .args(["index", "--json"])
            .output()
            .unwrap()
    };
    fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
    let out = run();
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    assert_eq!(out.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["data"]["parse_diagnostics"][0]["read_error"], true);
    assert_eq!(v["degraded"], true);
    assert!(
        v["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str().unwrap().contains("check file access"))
    );
    let out = run();
    assert_eq!(out.status.code(), Some(0));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(v["data"]["parse_diagnostics"], serde_json::json!([]));
}

const SESSION_LINE: &str = r#"{"type":"user","cwd":"/proj","message":{"role":"user","content":"hello world recall"},"timestamp":"2026-03-01T00:00:00Z"}"#;

fn write_session(dir: &Path, body: &str) -> PathBuf {
    let claude_dir = dir.join("claude");
    fs::create_dir_all(&claude_dir).unwrap();
    fs::write(claude_dir.join("s1.jsonl"), body).unwrap();
    claude_dir
}

fn index_stderr(dir: &Path, claude_dir: &Path) -> String {
    let out = recall(dir)
        .env("RECALL_CLAUDE_DIR", claude_dir)
        .arg("index")
        .output()
        .expect("spawn recall binary");
    assert_eq!(out.status.code(), Some(0), "index should exit 0");
    String::from_utf8_lossy(&out.stderr).into_owned()
}

fn assert_in_order(stderr: &str, first: &str, second: &str) {
    let position = |text: &str| {
        stderr
            .find(text)
            .unwrap_or_else(|| panic!("stderr should contain {text:?}, got: {stderr}"))
    };
    assert!(position(first) < position(second), "got: {stderr}");
}

#[test]
fn index_stderr_summarizes_new_and_unchanged_runs_in_order() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), SESSION_LINE);

    let first = index_stderr(dir.path(), &claude_dir);
    assert_in_order(&first, "Indexed 1 sessions in ", "Created 1 chunks");

    let second = index_stderr(dir.path(), &claude_dir);
    assert_in_order(&second, "1 sessions up to date", "Chunks up to date");
}

#[test]
fn index_stderr_summarizes_files_with_unresolved_parse_diagnostics() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), &format!("{SESSION_LINE}\nnot json\n"));
    index_stderr(dir.path(), &claude_dir);

    let unchanged = index_stderr(dir.path(), &claude_dir);

    assert!(
        unchanged.contains("1 sessions stored; 1 file(s) with unresolved parsing diagnostics"),
        "got: {unchanged}"
    );
}

#[test]
fn index_stderr_warns_when_a_root_with_indexed_sessions_is_unavailable() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), SESSION_LINE);
    index_stderr(dir.path(), &claude_dir);

    let stderr = index_stderr(dir.path(), &dir.path().join("absent"));

    assert!(
        stderr.contains("claude root unavailable — preserved 1 existing session(s)"),
        "got: {stderr}"
    );
}

/// #398: an empty `CLAUDE_CONFIG_DIR` must neither turn a `projects/` dir in the cwd into
/// the Claude root nor fall back to the default root; either run would read an empty root
/// in full and delete the session indexed from the configured one.
#[test]
fn empty_claude_config_dir_keeps_indexed_sessions_when_cwd_has_a_projects_dir() {
    let dir = TempDir::new().unwrap();
    let home = dir.path().join("home");
    let config_dir = dir.path().join("config");
    let work_dir = dir.path().join("work");
    fs::create_dir_all(home.join(".claude").join("projects")).unwrap();
    let session_dir = config_dir.join("projects").join("p");
    fs::create_dir_all(&session_dir).unwrap();
    fs::write(session_dir.join("s1.jsonl"), SESSION_LINE).unwrap();
    fs::create_dir_all(work_dir.join("projects")).unwrap();
    let isolated = || {
        let mut cmd = recall(dir.path());
        cmd.env_remove("RECALL_CLAUDE_DIR").env("HOME", &home);
        cmd
    };
    let search_hello = || {
        let out = isolated()
            .env_remove("CLAUDE_CONFIG_DIR")
            .args(["search", "hello"])
            .output()
            .expect("spawn recall binary");
        assert_eq!(out.status.code(), Some(0), "search should exit 0");
        String::from_utf8_lossy(&out.stdout).into_owned()
    };

    let first = isolated()
        .env("CLAUDE_CONFIG_DIR", &config_dir)
        .arg("index")
        .output()
        .expect("spawn recall binary");
    assert_eq!(first.status.code(), Some(0), "first index should exit 0");
    let before = search_hello();
    assert!(
        before.contains("Found 1 sessions:"),
        "premise: the first index should store the session, got: {before}"
    );

    let second = isolated()
        .env("CLAUDE_CONFIG_DIR", "")
        .current_dir(&work_dir)
        .arg("index")
        .output()
        .expect("spawn recall binary");
    assert_eq!(second.status.code(), Some(0), "second index should exit 0");
    let after = search_hello();

    assert!(
        after.contains("Found 1 sessions:"),
        "an empty CLAUDE_CONFIG_DIR run should keep the indexed session, got: {after}"
    );
    let stderr = String::from_utf8_lossy(&second.stderr);
    assert!(
        stderr.contains("claude root could not be fully read — preserved 1 existing session(s)"),
        "the empty CLAUDE_CONFIG_DIR run should report the Claude root as not read, got: {stderr}"
    );
    assert!(
        stderr.contains("check that the root is an absolute path"),
        "the note should point at the root's path, got: {stderr}"
    );
}

#[test]
fn classify_warns_once_when_a_session_source_is_missing() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), SESSION_LINE);
    index_stderr(dir.path(), &claude_dir);
    fs::remove_file(claude_dir.join("s1.jsonl")).unwrap();

    let out = recall(dir.path())
        .args(["classify", "--all"])
        .output()
        .expect("spawn recall binary");

    assert_eq!(out.status.code(), Some(0), "classify should exit 0");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert_eq!(
        stderr
            .matches("warning: provenance unavailable (missing file): 1 session(s)")
            .count(),
        1,
        "got: {stderr}"
    );
}

#[test]
fn recall_search_with_a_not_query_returns_only_the_sessions_without_the_excluded_word() {
    let dir = TempDir::new().unwrap();
    let turn = |text: &str| {
        format!(
            r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":"{text}"}},"timestamp":"2026-03-01T00:00:00Z"}}"#
        )
    };
    let wanted = turn("authentication notes");
    let excluded = turn("authentication and deployment notes");
    let unrelated = turn("checklist for release");
    index_claude_sessions(
        dir.path(),
        &[
            ("wanted", &[wanted.as_str()]),
            ("excluded", &[excluded.as_str()]),
            ("unrelated", &[unrelated.as_str()]),
        ],
    );

    let out = recall(dir.path())
        .args(["search", "authentication NOT deployment"])
        .output()
        .expect("spawn recall binary");

    let stdout = String::from_utf8_lossy(&out.stdout);
    assert_eq!(out.status.code(), Some(0), "{stdout}");
    assert!(
        stdout.contains("ID: wanted"),
        "the session without the excluded word must be returned: {stdout}"
    );
    assert!(
        !stdout.contains("ID: excluded") && !stdout.contains("ID: unrelated"),
        "sessions with the excluded word or without the first word must be absent: {stdout}"
    );
}

fn user_line(text: &str, timestamp: &str) -> String {
    format!(
        r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":"{text}"}},"timestamp":"{timestamp}"}}"#
    )
}

fn search_envelope(dir: &Path, args: &[&str]) -> serde_json::Value {
    let stdout = search_stdout(dir, args);
    serde_json::from_str(stdout.trim())
        .unwrap_or_else(|e| panic!("stdout should be one JSON envelope, got {stdout:?}: {e}"))
}

fn first_session_id(envelope: &serde_json::Value) -> String {
    envelope["data"]["results"][0]["session_id"]
        .as_str()
        .unwrap_or_else(|| panic!("no first result in: {envelope}"))
        .to_owned()
}

fn mark_source_unknown(dir: &Path, session_id: &str) {
    let changed = rusqlite::Connection::open(dir.join("recall.db"))
        .unwrap()
        .execute(
            "UPDATE sessions SET source = 'gemini' WHERE session_id = ?1",
            [session_id],
        )
        .unwrap();
    assert_eq!(changed, 1, "session {session_id} should exist in the index");
}

fn incomplete_note(envelope: &serde_json::Value) -> Option<String> {
    envelope["notes"]
        .as_array()
        .unwrap_or_else(|| panic!("notes should be an array, got: {envelope}"))
        .iter()
        .filter_map(|n| n.as_str())
        .find(|n| n.contains("incomplete"))
        .map(str::to_owned)
}

#[test]
fn search_with_limit_one_prints_the_first_result_of_the_default_limit() {
    let dir = TempDir::new().unwrap();
    // Equal text gives equal raw ranks, so only the recency blend separates the
    // sessions. "z-newest" sorts last by id, so a cut before the blend drops it.
    let sessions: Vec<(String, Vec<String>)> = [
        ("a1", "2026-01-01T00:00:00Z"),
        ("a2", "2026-01-02T00:00:00Z"),
        ("a3", "2026-01-03T00:00:00Z"),
        ("a4", "2026-01-04T00:00:00Z"),
        ("a5", "2026-01-05T00:00:00Z"),
        ("z-newest", "2026-10-01T00:00:00Z"),
    ]
    .iter()
    .map(|(name, ts)| {
        (
            (*name).to_owned(),
            vec![user_line("authentication notes", ts)],
        )
    })
    .collect();
    let refs: Vec<(&str, Vec<&str>)> = sessions
        .iter()
        .map(|(n, l)| (n.as_str(), l.iter().map(String::as_str).collect()))
        .collect();
    let borrowed: Vec<(&str, &[&str])> = refs.iter().map(|(n, l)| (*n, l.as_slice())).collect();
    index_claude_sessions(dir.path(), &borrowed);

    let default_limit = search_envelope(dir.path(), &["authentication"]);
    let limit_one = search_envelope(dir.path(), &["authentication", "--limit", "1"]);

    assert_eq!(
        first_session_id(&default_limit),
        "z-newest",
        "the recency blend should put the newest session first: {default_limit}"
    );
    assert_eq!(
        limit_one["data"]["results"].as_array().map(Vec::len),
        Some(1),
        "{limit_one}"
    );
    assert_eq!(
        first_session_id(&limit_one),
        first_session_id(&default_limit),
        "--limit 1 must print the head of the default limit: {limit_one}"
    );
}

#[test]
fn search_that_skips_only_an_unknown_source_session_outside_the_results_is_not_degraded() {
    let dir = TempDir::new().unwrap();
    let valid = user_line("authentication", "2026-03-01T00:00:00Z");
    let weird = user_line(
        "authentication retry logic for the session refresh path in the gateway service",
        "2026-03-01T00:00:00Z",
    );
    index_claude_sessions(dir.path(), &[("valid", &[&valid]), ("weird", &[&weird])]);
    mark_source_unknown(dir.path(), "weird");

    let envelope = search_envelope(dir.path(), &["authentication", "--limit", "1"]);

    assert_eq!(first_session_id(&envelope), "valid", "{envelope}");
    assert_eq!(
        incomplete_note(&envelope),
        None,
        "a skipped session that cannot reach the results must not raise the note: {envelope}"
    );
    assert_eq!(
        envelope["degraded"].as_bool(),
        envelope["notes"].as_array().map(|n| !n.is_empty()),
        "degraded must come only from the remaining notes: {envelope}"
    );
}

#[test]
fn search_that_skips_an_unknown_source_session_that_could_rank_says_so_in_the_note() {
    let dir = TempDir::new().unwrap();
    let valid = user_line("authentication notes", "2026-03-01T00:00:00Z");
    let weird = user_line("authentication notes", "2026-03-01T00:00:00Z");
    index_claude_sessions(dir.path(), &[("valid", &[&valid]), ("weird", &[&weird])]);
    mark_source_unknown(dir.path(), "weird");

    let envelope = search_envelope(dir.path(), &["authentication", "--limit", "1"]);

    let note = incomplete_note(&envelope).unwrap_or_else(|| {
        panic!("a skipped session that could rank must raise the note: {envelope}")
    });
    assert_eq!(envelope["degraded"], true, "{envelope}");
    assert!(
        note.contains("could have ranked"),
        "the note must say a skipped session could have ranked in the results: {note}"
    );
}
