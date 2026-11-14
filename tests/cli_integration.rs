use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::TempDir;

#[cfg(unix)]
#[path = "support/root_skip.rs"]
mod root_skip;

#[path = "support/stale_snapshot_note.rs"]
mod stale_snapshot_note;
use stale_snapshot_note::READ_ONLY_DIR_NOTE;

fn recall(dir: &Path) -> Command {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_recall"));
    cmd.env("RECALL_DB", dir.join("recall.db"))
        .env("RECALL_MODEL_DIR", dir.join("no-model"))
        .env_remove("RECALL_MODEL_MANIFEST")
        .env("RECALL_CLAUDE_DIR", dir.join("no-claude"))
        .env("RECALL_CODEX_DIR", dir.join("no-codex"))
        .env_remove("RECALL_CODEX_ARCHIVED_DIR")
        .env_remove("CLAUDE_CONFIG_DIR")
        .env_remove("RECALL_TOKEN_BUDGET")
        .env_remove("RECALL_FORWARD_PAUSE_MS")
        // Keep the model cache and the probe record off the developer's own directories.
        .env("XDG_DATA_HOME", dir.join("data"))
        .env("XDG_CACHE_HOME", dir.join("cache"))
        .env("HF_HOME", dir.join("hf"))
        .env_remove("__RECALL_GEMMA_PROBE")
        .env_remove("__RECALL_GEMMA_SPEC")
        .env_remove("CLAUDE_CODE_SESSION_ID")
        // clap colours its errors and help when this is set, even into a pipe.
        .env_remove("CLICOLOR_FORCE")
        .env_remove("RUST_LOG");
    cmd
}

fn exit_code(cmd: &mut Command) -> i32 {
    cmd.output()
        .expect("spawn recall binary")
        .status
        .code()
        .expect("process returned an exit code")
}

/// What a run shows the user: its exit code, stdout and stderr.
fn outcome(cmd: &mut Command) -> (Option<i32>, String, String) {
    let out = cmd.output().expect("spawn recall binary");
    (
        out.status.code(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

fn listing(dir: &Path) -> Vec<String> {
    let mut names: Vec<String> = fs::read_dir(dir)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().into_string().unwrap())
        .collect();
    names.sort();
    names
}

/// `recall --help`, whose `--db-path` line names the `RECALL_DB` that `recall()` sets.
fn help_text(db: &Path) -> String {
    format!(
        concat!(
            "Search past Claude Code and Codex sessions\n",
            "\n",
            "Usage: recall [OPTIONS] [COMMAND]\n",
            "\n",
            "Commands:\n",
            "  index     Parse, chunk, and embed new session logs (incremental)\n",
            "  rebuild   Re-parse and re-embed every present session; a missing or unreadable source root keeps its existing index\n",
            "  model     Manage the embedding model\n",
            "  search    Search indexed sessions (default when query is provided)\n",
            "  show      Show full conversation of a session\n",
            "  classify  Classify existing sessions from source provenance and their first user turn\n",
            "  status    Show index and embedding status\n",
            "  doctor    Diagnose a broken index: corruption, orphaned rows, and a live model load probe. Read-only by default; `--fix` repairs orphan embeddings, recovers the model, and re-runs the checks. Other remedies point at index/rebuild/model download\n",
            "  help      Print this message or the help of the given subcommand(s)\n",
            "\n",
            "Options:\n",
            "  -v, --verbose\n",
            "          Show diagnostic output\n",
            "      --db-path <DB_PATH>\n",
            "          Database file path (default: <data dir>/recall/indexes/<manifest-sha256>/recall.db, where <data dir> is an absolute XDG_DATA_HOME or ~/.local/share; env: RECALL_DB) [env: RECALL_DB={}]\n",
            "      --model-manifest <MODEL_MANIFEST>\n",
            "          Pinned Gemma model manifest. Select an older saved manifest to roll back [env: RECALL_MODEL_MANIFEST=]\n",
            "      --json\n",
            "          Emit a machine-readable JSON envelope instead of human-readable text\n",
            "  -h, --help\n",
            "          Print help\n",
        ),
        db.display()
    )
}

/// What a `--json` run shows the user, with stdout parsed as the one-line envelope it prints.
fn json_outcome(cmd: &mut Command) -> (Option<i32>, serde_json::Value, String) {
    let (code, stdout, stderr) = outcome(cmd);
    let envelope = stdout
        .strip_suffix('\n')
        .filter(|line| !line.contains('\n'))
        .and_then(|line| serde_json::from_str(line).ok())
        .unwrap_or_else(|| {
            panic!("stdout should be one JSON line; exit {code:?}, stdout {stdout:?}, stderr {stderr:?}")
        });
    (code, envelope, stderr)
}

const NO_SESSIONS_INDEXED: &str = "No sessions indexed. Run `recall index` first.\n";

/// What `status` and `doctor` print on stderr when `db` does not exist.
fn database_not_found(db: &Path) -> String {
    format!(
        "database not found at {}\nrun `recall index` to create the index.\n",
        db.display()
    )
}

const NO_QUERY: &str = "A search query is required. Usage: recall search \"query\" or recall index";
const ONLY_CURRENT_WITHOUT_A_SESSION_ID: &str = "error: --only-current requires CLAUDE_CODE_SESSION_ID; run recall from inside a Claude Code session\n";

#[test]
fn unknown_flag_exits_usage() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).arg("--no-such-flag")),
        (
            Some(64),
            String::new(),
            concat!(
                "error: unexpected argument '--no-such-flag' found\n",
                "\n",
                "Usage: recall [OPTIONS] [COMMAND]\n",
                "\n",
                "For more information, try '--help'.\n",
            )
            .to_owned()
        )
    );
}

#[test]
fn no_subcommand_exits_usage() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(&mut recall(dir.path())),
        (Some(64), String::new(), format!("error: {NO_QUERY}\n"))
    );
}

#[test]
fn help_exits_success() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).arg("--help")),
        (
            Some(0),
            help_text(&dir.path().join("recall.db")),
            String::new()
        )
    );
}

#[test]
fn show_without_database_exits_usage() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).args(["show", "deadbeef"])),
        (
            Some(64),
            String::new(),
            "error: Database not found. Run `recall index` first.\n".to_owned()
        )
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
}

#[test]
fn search_without_an_index_says_to_run_recall_index_and_creates_no_file() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "anything"])),
        (Some(0), String::new(), NO_SESSIONS_INDEXED.to_owned())
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
}

#[test]
fn a_bare_word_runs_a_search_for_it() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).arg("someword")),
        (Some(0), String::new(), NO_SESSIONS_INDEXED.to_owned())
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
}

#[test]
fn status_verbose_includes_db_path() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);

    assert_eq!(
        outcome(recall(dir.path()).args(["status", "--verbose"])),
        (
            Some(0),
            format!(
                "Sessions: 0\nQA chunks: 0\nEmbedded: 0/0\nModel: not installed (run `recall model download`)\nDB: {}\n",
                dir.path().join("recall.db").display()
            ),
            String::new()
        )
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

/// `recall search hello` when `seed_indexed_session` indexed only `s1`.
const S1_MATCH: &str = concat!(
    "Found 1 sessions:\n",
    "\n",
    "[1] 2026-03-01 | s1 | proj [claude]\n",
    "    /proj\n",
    "    ID: s1\n",
    "    > hello world recall\n",
    "    inspect: recall show s1 --chunk 1 --window 3\n",
    "\n",
);

/// The `--json` result row of `S1_MATCH`.
fn s1_match() -> serde_json::Value {
    serde_json::json!({
        "chunk": 1,
        "excerpt": "hello world recall",
        "inspect": "recall show s1 --chunk 1 --window 3",
        "project": "/proj",
        "session_id": "s1",
        "slug": "s1",
        "source": "claude",
        "timestamp": 1_772_323_200_000_i64,
    })
}

/// The `--json` data of `recall show s1` when `seed_indexed_session` indexed `s1`.
fn show_s1() -> serde_json::Value {
    serde_json::json!({
        "messages": [{"role": "user", "text": "hello world recall"}],
        "project": "/proj",
        "session_id": "s1",
        "slug": "s1",
        "source": "claude",
        "timestamp": 1_772_323_200_000_i64,
    })
}

/// The `--json` data of `recall status` on an index of one unembedded session.
fn status_of_one_session() -> serde_json::Value {
    serde_json::json!({"embedded": 0, "model_ready": false, "qa_chunks": 1, "sessions": 1})
}

/// The `--json` data of `recall doctor` on a sound index without the model installed.
fn doctor_of_a_sound_index_without_the_model() -> serde_json::Value {
    serde_json::json!({
        "checks": [
            {"detail": null, "name": "integrity", "ok": true, "remedy": null},
            {"detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null},
            {"detail": null, "name": "orphan_chunks", "ok": true, "remedy": null},
            {
                "detail": "model not installed; search runs FTS-only",
                "name": "model",
                "ok": true,
                "remedy": null,
            },
        ],
        "healthy": true,
        "model_recovered": null,
        "model_recovery_retryable": null,
        "repaired": null,
    })
}

const MODEL_NOT_INSTALLED_STDERR: &str =
    "note: embedding model not installed; run `recall model download` to enable semantic search.\n";
const MODEL_NOT_INSTALLED_NOTE: &str = "semantic search unavailable: embedding model not installed; run `recall model download` to enable semantic search";
const STATUS_OF_ONE_SESSION: &str = "Sessions: 1\nQA chunks: 1\nEmbedded: 0/1\nModel: not installed (run `recall model download`)\n";

#[test]
fn search_json_emits_success_envelope_with_results() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    assert_eq!(
        json_outcome(recall(dir.path()).args(["search", "hello", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [s1_match()]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn no_subcommand_json_emits_error_envelope_on_stderr() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).arg("--json")),
        (
            Some(64),
            String::new(),
            concat!(
                r#"{"error":{"code":"USAGE_ERROR","message":"A search query is required. Usage: recall search \"query\" or recall index","next_step":"See `recall --help`; if the index is missing, run `recall index` first.","retryable":false}}"#,
                "\n"
            )
            .to_owned()
        )
    );
}

#[test]
fn status_json_emits_stats_envelope() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    assert_eq!(
        json_outcome(recall(dir.path()).args(["status", "--json"])),
        (
            Some(0),
            serde_json::json!({"data": status_of_one_session(), "degraded": false, "notes": []}),
            String::new()
        )
    );
}

#[test]
fn search_of_an_index_without_sessions_says_to_run_recall_index_in_text_and_json() {
    let dir = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(dir.path()).arg("index")), 0);

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "anything"])),
        (Some(0), String::new(), NO_SESSIONS_INDEXED.to_owned())
    );
    assert_eq!(
        json_outcome(recall(dir.path()).args(["search", "anything", "--json"])),
        (
            Some(0),
            serde_json::json!({"data": {"results": []}, "degraded": false, "notes": []}),
            NO_SESSIONS_INDEXED.to_owned()
        )
    );
}

#[test]
fn search_without_json_keeps_human_output() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "hello"])),
        (
            Some(0),
            S1_MATCH.to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn show_json_emits_session_envelope() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    assert_eq!(
        json_outcome(recall(dir.path()).args(["show", "s1", "--json"])),
        (
            Some(0),
            serde_json::json!({"data": show_s1(), "degraded": false, "notes": []}),
            String::new()
        )
    );
}

#[test]
fn show_without_json_keeps_human_output() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    assert_eq!(
        outcome(recall(dir.path()).args(["show", "s1"])),
        (
            Some(0),
            concat!(
                "# s1\n",
                "- session_id: s1\n",
                "- date: 2026-03-01\n",
                "- project: /proj\n",
                "- source: claude\n",
                "\n",
                "## [user]\n",
                "hello world recall\n",
                "\n",
            )
            .to_owned(),
            String::new()
        )
    );
}

#[test]
fn index_json_without_sessions_reports_nothing_to_embed_and_the_missing_model() {
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
fn rebuild_json_reparses_an_unchanged_session_and_reports_it_unembedded() {
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
fn doctor_json_of_an_index_without_the_model_reports_it_healthy_with_the_model_as_info() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    assert_eq!(
        json_outcome(recall(dir.path()).args(["doctor", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": doctor_of_a_sound_index_without_the_model(),
                "degraded": false,
                "notes": [],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn status_json_without_index_reports_zero_counts() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        json_outcome(recall(dir.path()).args(["status", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"embedded": 0, "model_ready": false, "qa_chunks": 0, "sessions": 0},
                "degraded": false,
                "notes": [],
            }),
            database_not_found(&dir.path().join("recall.db"))
        )
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
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
    let model_dir = TempDir::new().unwrap();
    let manifest = setup_cached_default_model(model_dir.path());

    assert_eq!(
        json_outcome(
            recall(dir.path())
                .env("RECALL_MODEL_DIR", model_dir.path())
                .arg("--model-manifest")
                .arg(manifest)
                .args(["status", "--json"])
        ),
        (
            Some(0),
            serde_json::json!({
                "data": {"embedded": 0, "model_ready": true, "qa_chunks": 0, "sessions": 0},
                "degraded": false,
                "notes": [],
            }),
            database_not_found(&dir.path().join("recall.db"))
        )
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
}

#[test]
fn doctor_without_an_index_says_to_run_recall_index() {
    let dir = TempDir::new().unwrap();
    let hint = database_not_found(&dir.path().join("recall.db"));

    assert_eq!(
        outcome(recall(dir.path()).arg("doctor")),
        (Some(0), String::new(), hint.clone())
    );
    assert_eq!(
        json_outcome(recall(dir.path()).args(["doctor", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {
                    "checks": [],
                    "healthy": false,
                    "model_recovered": null,
                    "model_recovery_retryable": null,
                    "repaired": null,
                },
                "degraded": true,
                "notes": ["no index: run `recall index`"],
            }),
            hint
        )
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
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
        outcome(recall(dir.path()).args(["search", "hello", "--only-current"])),
        (
            Some(64),
            String::new(),
            ONLY_CURRENT_WITHOUT_A_SESSION_ID.to_owned()
        )
    );
}

/// What `recall search hello` prints when it lists both sessions of `seed_two_matching_sessions`.
const BOTH_SEEDED_SESSIONS: &str = concat!(
    "Found 2 sessions:\n",
    "\n",
    "[1] 2026-03-01 | s1 | proj [claude]\n",
    "    /proj\n",
    "    ID: s1\n",
    "    > hello world recall\n",
    "    inspect: recall show s1 --chunk 1 --window 3\n",
    "\n",
    "[2] 2026-03-01 | s2 | proj [claude]\n",
    "    /proj\n",
    "    ID: s2\n",
    "    > hello world recall\n",
    "    inspect: recall show s2 --chunk 2 --window 3\n",
    "\n",
);

/// The tracing line carries a timestamp and color codes, so tests look for its message.
const EXCLUDE_CURRENT_HAS_NO_EFFECT: &str =
    "--exclude-current has no effect: CLAUDE_CODE_SESSION_ID is not set\n";

#[test]
fn search_exclude_current_with_a_whitespace_only_session_id_lists_every_session_and_warns() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    let out = recall(dir.path())
        .args(["search", "hello", "--exclude-current"])
        .env("CLAUDE_CODE_SESSION_ID", "  ")
        .output()
        .expect("spawn recall binary");

    assert_eq!(out.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&out.stdout), BOTH_SEEDED_SESSIONS);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(EXCLUDE_CURRENT_HAS_NO_EFFECT),
        "a whitespace-only id counts as unset, so --exclude-current warns: {stderr}"
    );
}

#[test]
fn search_only_current_with_a_whitespace_only_session_id_exits_usage() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(
            recall(dir.path())
                .args(["search", "hello", "--only-current"])
                .env("CLAUDE_CODE_SESSION_ID", "  ")
        ),
        (
            Some(64),
            String::new(),
            ONLY_CURRENT_WITHOUT_A_SESSION_ID.to_owned()
        )
    );
}

#[test]
fn search_exclude_current_without_a_session_id_lists_every_session_and_warns() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    let out = recall(dir.path())
        .args(["search", "hello", "--exclude-current"])
        .output()
        .expect("spawn recall binary");

    assert_eq!(out.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&out.stdout), BOTH_SEEDED_SESSIONS);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains(EXCLUDE_CURRENT_HAS_NO_EFFECT),
        "--exclude-current without the env must warn that it has no effect: {stderr}"
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
fn search_no_embed_flag_is_retired_exits_usage() {
    let dir = TempDir::new().unwrap();

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "anything", "--no-embed"])),
        (
            Some(64),
            String::new(),
            concat!(
                "error: unexpected argument '--no-embed' found\n",
                "\n",
                "  tip: to pass '--no-embed' as a value, use '-- --no-embed'\n",
                "\n",
                "Usage: recall search <QUERY>\n",
                "\n",
                "For more information, try '--help'.\n",
            )
            .to_owned()
        )
    );
}

#[cfg(unix)]
#[test]
fn read_commands_work_in_readonly_directory() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let db = dir.path().join("recall.db");
    let _ = fs::remove_file(db.with_file_name("recall.db-wal"));
    let _ = fs::remove_file(db.with_file_name("recall.db-shm"));
    let read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "hello"])),
        (
            Some(0),
            S1_MATCH.to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
    assert_eq!(
        outcome(recall(dir.path()).arg("status")),
        (Some(0), STATUS_OF_ONE_SESSION.to_owned(), String::new())
    );
    assert_eq!(
        outcome(recall(dir.path()).args(["classify", "--all", "--dry-run"])),
        (
            Some(0),
            "s1 [interactive] hello world recall\ndry-run: 1 session(s) would be classified (0 automated, 1 interactive)\n".to_owned(),
            String::new()
        )
    );
    drop(read_only);
    assert_eq!(listing(dir.path()), ["claude", "recall.db"]);
}

#[cfg(unix)]
#[test]
fn status_reads_an_index_given_by_a_relative_path_from_a_read_only_working_directory() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let db = dir.path().join("recall.db");
    let _ = fs::remove_file(db.with_file_name("recall.db-wal"));
    let _ = fs::remove_file(db.with_file_name("recall.db-shm"));
    let _read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    assert_eq!(
        outcome(
            recall(dir.path())
                .current_dir(dir.path())
                .env("RECALL_DB", "recall.db")
                .arg("status")
        ),
        (Some(0), STATUS_OF_ONE_SESSION.to_owned(), String::new())
    );
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
fn show_of_a_session_missing_from_a_stale_snapshot_reports_the_write_ahead_log() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());
    let _read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    assert_eq!(
        outcome(recall(dir.path()).args(["show", "s2", "--json"])),
        (
            Some(65),
            String::new(),
            format!(
                r#"{{"error":{{"code":"DATA_ERROR","message":"No session found matching 's2'; {READ_ONLY_DIR_NOTE}","retryable":false}}}}"#
            ) + "\n"
        )
    );
}

/// An index from before the trigram tokenizer and before embeddings: one session whose log
/// sits under the Claude root, which is missing in `recall()`'s environment.
fn seed_pre_trigram_index(dir: &Path) {
    rusqlite::Connection::open(dir.join("recall.db"))
        .unwrap()
        .execute_batch(
            "CREATE TABLE sessions (
                 session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT,
                 project TEXT, slug TEXT, timestamp INTEGER, mtime REAL
             );
             CREATE VIRTUAL TABLE messages USING fts5(
                 session_id UNINDEXED, role, text, tokenize='porter unicode61'
             );
             INSERT INTO sessions VALUES ('s1', 'claude', '/gone/s1.jsonl', '/p', 'slug', 0, 0.0);
             INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello world recall');",
        )
        .unwrap();
}

#[test]
fn index_upgrading_a_pre_trigram_index_keeps_the_sessions_of_a_missing_root_searchable() {
    let dir = TempDir::new().unwrap();
    seed_pre_trigram_index(dir.path());

    // Its stderr carries stage timings and the read warning #445 reports, so only the code counts.
    let (index_code, _, index_stderr) = outcome(recall(dir.path()).arg("index"));
    assert_eq!(index_code, Some(0), "{index_stderr}");

    assert_eq!(
        outcome(recall(dir.path()).arg("status")),
        (Some(0), STATUS_OF_ONE_SESSION.to_owned(), String::new())
    );
    assert_eq!(
        outcome(recall(dir.path()).args(["search", "hello"])),
        (
            Some(0),
            "Found 1 sessions:\n\n[1] 1970-01-01 | slug | p [claude]\n    /p\n    ID: s1\n    > hello world recall\n    inspect: recall show s1 --chunk 1 --window 3\n\n".to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn show_on_an_index_from_an_older_version_asks_for_a_rebuild() {
    let dir = TempDir::new().unwrap();
    rusqlite::Connection::open(dir.path().join("recall.db"))
        .unwrap()
        .execute_batch("CREATE TABLE sessions (session_id TEXT PRIMARY KEY);")
        .unwrap();

    assert_eq!(
        outcome(recall(dir.path()).args(["show", "s1", "--json"])),
        (
            Some(65),
            String::new(),
            concat!(
                r#"{"error":{"code":"DATA_ERROR","message":"database schema is out of date; run `recall rebuild` to migrate it","retryable":false}}"#,
                "\n"
            )
            .to_owned()
        )
    );
}

#[test]
fn show_from_a_writable_directory_adds_no_stale_snapshot_note() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());

    assert_eq!(
        json_outcome(recall(dir.path()).args(["show", "s1", "--json"])),
        (
            Some(0),
            serde_json::json!({"data": show_s1(), "degraded": false, "notes": []}),
            String::new()
        )
    );
}

#[cfg(unix)]
#[test]
fn read_commands_on_a_stale_snapshot_in_a_read_only_directory_ask_to_index_a_copy() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());
    let _read_only = ReadOnlyDir::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    assert_eq!(
        json_outcome(recall(dir.path()).args(["show", "s1", "--json"])),
        (
            Some(0),
            serde_json::json!({"data": show_s1(), "degraded": true, "notes": [READ_ONLY_DIR_NOTE]}),
            String::new()
        )
    );
    assert_eq!(
        json_outcome(recall(dir.path()).args(["search", "hello", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [s1_match()]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE, READ_ONLY_DIR_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
    assert_eq!(
        json_outcome(recall(dir.path()).args(["status", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": status_of_one_session(),
                "degraded": true,
                "notes": [READ_ONLY_DIR_NOTE],
            }),
            String::new()
        )
    );
    assert_eq!(
        json_outcome(recall(dir.path()).args(["doctor", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": doctor_of_a_sound_index_without_the_model(),
                "degraded": true,
                "notes": [READ_ONLY_DIR_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
    assert_eq!(
        outcome(recall(dir.path()).args(["classify", "--all", "--dry-run", "--json"])),
        (
            Some(65),
            String::new(),
            format!(
                r#"{{"error":{{"code":"DATA_ERROR","message":"{READ_ONLY_DIR_NOTE}","retryable":false}}}}"#
            ) + "\n"
        )
    );
}

#[cfg(unix)]
#[test]
fn cli_tests_leave_the_root_probe_file_name_to_the_shared_root_check() {
    root_skip::assert_probe_literal_not_inlined(
        include_str!("cli_integration.rs"),
        "tests/cli_integration.rs",
    );
}

#[cfg(unix)]
#[test]
fn the_cli_integration_suite_fails_when_run_as_root() {
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
    // It is dated in the future, which always gets the largest boost, so the
    // order holds however late the test runs.
    let a1 = user_line("authentication notes", "2026-01-01T00:00:00Z");
    let a2 = user_line("authentication notes", "2026-01-02T00:00:00Z");
    let a3 = user_line("authentication notes", "2026-01-03T00:00:00Z");
    let a4 = user_line("authentication notes", "2026-01-04T00:00:00Z");
    let a5 = user_line("authentication notes", "2026-01-05T00:00:00Z");
    let newest = user_line("authentication notes", "2999-01-01T00:00:00Z");
    index_claude_sessions(
        dir.path(),
        &[
            ("a1", &[&a1]),
            ("a2", &[&a2]),
            ("a3", &[&a3]),
            ("a4", &[&a4]),
            ("a5", &[&a5]),
            ("z-newest", &[&newest]),
        ],
    );

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
fn search_that_skips_only_an_unknown_source_session_outside_the_results_adds_no_incomplete_note() {
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
