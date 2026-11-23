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
        // why-not: the developer's own model cache and probe record, which a test run must not touch
        .env("XDG_DATA_HOME", dir.join("data"))
        .env("XDG_CACHE_HOME", dir.join("cache"))
        .env("HF_HOME", dir.join("hf"))
        .env_remove("__RECALL_GEMMA_PROBE")
        .env_remove("__RECALL_GEMMA_SPEC")
        .env_remove("CLAUDE_CODE_SESSION_ID")
        // why-not: CLICOLOR_FORCE, because clap then colours its errors and help even into a pipe
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
            "  doctor    Diagnose a broken index: corruption, orphaned rows, the model identity, and a live model load probe. Read-only by default; `--fix` repairs orphan embeddings, recovers the model, and re-runs the checks. Other remedies point at index/rebuild/model download, --model-manifest, or a separate --db-path\n",
            "  help      Print this message or the help of the given subcommand(s)\n",
            "\n",
            "Options:\n",
            "  -v, --verbose\n",
            "          Show diagnostic output\n",
            "      --db-path <DB_PATH>\n",
            "          Database file path (default: <data dir>/recall/indexes/<model-identity>/recall.db, where <data dir> is an absolute XDG_DATA_HOME or ~/.local/share; `recall status --verbose` prints the path in use; env: RECALL_DB) [env: RECALL_DB={}]\n",
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
fn indexing_over_an_old_embedding_table_with_rows_refuses_and_keeps_them() {
    let dir = TempDir::new().unwrap();
    let db = dir.path().join("recall.db");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute_batch(
            "CREATE TABLE vec_chunks (chunk_id INTEGER, embedding BLOB);
             INSERT INTO vec_chunks VALUES (7, x'00');",
        )
        .unwrap();

    let (code, stdout, stderr) = outcome(recall(dir.path()).arg("index"));

    assert_eq!((code, stdout.as_str()), (Some(65), ""));
    assert!(
        stderr.ends_with(
            "\nerror: legacy vector index has no model identity; keep it for rollback and build Gemma in a separate --db-path\n"
        ),
        "{stderr}"
    );
    assert_eq!(
        rusqlite::Connection::open(&db)
            .unwrap()
            .query_row("SELECT chunk_id FROM vec_chunks", [], |r| r
                .get::<_, i64>(0))
            .unwrap(),
        7
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

const SHOW_S1: &str = concat!(
    "# s1\n",
    "- session_id: s1\n",
    "- date: 2026-03-01\n",
    "- project: /proj\n",
    "- source: claude\n",
    "\n",
    "## [user]\n",
    "hello world recall\n",
    "\n",
);

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

fn status_of_one_session() -> serde_json::Value {
    serde_json::json!({"embedded": 0, "model_ready": false, "qa_chunks": 1, "sessions": 1})
}

fn doctor_of_a_sound_index_without_the_model() -> serde_json::Value {
    serde_json::json!({
        "checks": [
            {"detail": null, "name": "integrity", "ok": true, "remedy": null},
            {"detail": null, "name": "orphan_embeddings", "ok": true, "remedy": null},
            {"detail": null, "name": "orphan_chunks", "ok": true, "remedy": null},
            {"detail": null, "name": "model_identity", "ok": true, "remedy": null},
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

macro_rules! model_not_installed {
    () => {
        "note: embedding model not installed; run `recall model download` to enable semantic search."
    };
}
const MODEL_NOT_INSTALLED_STDERR: &str = concat!(model_not_installed!(), "\n");
const MODEL_NOT_INSTALLED_INDEX_NOTE: &str = model_not_installed!();

fn index_json_outcome_without_stage_timings(
    cmd: &mut Command,
) -> (Option<i32>, serde_json::Value, String) {
    let (code, mut envelope, stderr) = json_outcome(cmd);
    let seconds = envelope["data"]["observations"]
        .as_object_mut()
        .and_then(|observations| observations.remove("seconds"))
        .unwrap_or_else(|| panic!("an index envelope carries its stage seconds: {envelope}"));
    assert!(seconds["total"].is_number(), "{seconds}");
    (code, envelope, stderr)
}

fn index_envelope_without_the_model(
    counts: &[(&str, u64)],
    parse_diagnostics: &serde_json::Value,
    notes: &[String],
) -> serde_json::Value {
    let notes = [&[MODEL_NOT_INSTALLED_INDEX_NOTE.to_owned()], notes].concat();
    serde_json::json!({
        "data": {
            "embedded": 0,
            "failed_count": 0,
            "observations": {"counts": index_counts(counts)},
            "parse_diagnostics": parse_diagnostics,
            "preserved_embedded": 0,
            "skipped_roots": [],
        },
        "degraded": true,
        "notes": notes,
    })
}

fn index_counts(nonzero: &[(&str, u64)]) -> serde_json::Value {
    let mut counts = serde_json::json!({
        "chunks_created_committed": 0, "chunks_failed": 0, "chunks_matched_committed": 0,
        "chunks_pending_snapshot": 0, "chunks_reconciled_created_committed": 0,
        "chunks_remaining_snapshot": 0, "chunks_removed_committed": 0, "chunks_save_failed": 0,
        "chunks_saved": 0, "chunks_stale": 0, "chunks_unattempted": 0,
        "embeddings_reused_committed": 0, "files_deferred": 0, "files_discovered": 0,
        "files_empty": 0, "files_failed": 0, "files_remaining": 0, "files_unchanged": 0,
        "files_updated": 0, "files_updated_committed": 0, "inference_batches": 0,
        "inference_chunks": 0, "sessions_chunk_pending": 0, "sessions_chunk_remaining": 0,
        "sessions_chunked_committed": 0, "sessions_chunked_empty": 0, "sessions_stored": 0,
    });
    for (name, value) in nonzero {
        assert!(counts.get(name).is_some(), "no index count is named {name}");
        counts[name] = (*value).into();
    }
    counts
}

const ONE_SESSION_PARSED_AND_CHUNKED: &[(&str, u64)] = &[
    ("chunks_created_committed", 1),
    ("chunks_pending_snapshot", 1),
    ("chunks_remaining_snapshot", 1),
    ("chunks_unattempted", 1),
    ("files_discovered", 1),
    ("files_updated", 1),
    ("files_updated_committed", 1),
    ("sessions_chunk_pending", 1),
    ("sessions_chunked_committed", 1),
    ("sessions_stored", 1),
];
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
        (Some(0), SHOW_S1.to_owned(), String::new())
    );
}

#[test]
fn index_json_without_sessions_reports_nothing_to_embed_the_missing_model_and_its_stage_lines() {
    let dir = TempDir::new().unwrap();

    let (code, envelope, stderr) =
        index_json_outcome_without_stage_timings(recall(dir.path()).args(["index", "--json"]));

    assert_eq!(
        (code, envelope),
        (
            Some(0),
            index_envelope_without_the_model(&[], &serde_json::json!([]), &[])
        )
    );
    assert!(
        stderr.starts_with("index: total: started (not yet complete)\n"),
        "only a run of the binary reaches the stage lines the observer writes to stderr: {stderr}"
    );
}

#[test]
fn rebuild_json_reparses_an_unchanged_session_and_reports_it_unembedded() {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());

    let (code, envelope, _) = index_json_outcome_without_stage_timings(
        recall(dir.path())
            .env("RECALL_CLAUDE_DIR", dir.path().join("claude"))
            .args(["rebuild", "--json"]),
    );

    assert_eq!(
        (code, envelope),
        (
            Some(0),
            index_envelope_without_the_model(
                ONE_SESSION_PARSED_AND_CHUNKED,
                &serde_json::json!([]),
                &[]
            )
        )
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

    assert_eq!(
        outcome(
            recall(dir.path())
                .args(["search", "hello"])
                .env("CLAUDE_CODE_SESSION_ID", "s1")
        ),
        (
            Some(0),
            S2_MATCH.to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn search_include_current_keeps_invoking_session() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    assert_eq!(
        outcome(
            recall(dir.path())
                .args(["search", "hello", "--include-current"])
                .env("CLAUDE_CODE_SESSION_ID", "s1")
        ),
        (
            Some(0),
            BOTH_SEEDED_SESSIONS.to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn search_only_current_restricts_to_invoking_session() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    assert_eq!(
        outcome(
            recall(dir.path())
                .args(["search", "hello", "--only-current"])
                .env("CLAUDE_CODE_SESSION_ID", "s1")
        ),
        (
            Some(0),
            S1_MATCH.to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
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

const S2_MATCH: &str = concat!(
    "Found 1 sessions:\n",
    "\n",
    "[1] 2026-03-01 | s2 | proj [claude]\n",
    "    /proj\n",
    "    ID: s2\n",
    "    > hello world recall\n",
    "    inspect: recall show s2 --chunk 2 --window 3\n",
    "\n",
);

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

const EXCLUDE_CURRENT_HAS_NO_EFFECT_STDERR: &str = concat!(
    "warning: --exclude-current has no effect: CLAUDE_CODE_SESSION_ID is not set\n",
    model_not_installed!(),
    "\n",
);

#[test]
fn search_exclude_current_with_a_whitespace_only_session_id_lists_every_session_and_warns() {
    let dir = TempDir::new().unwrap();
    seed_two_matching_sessions(dir.path());

    assert_eq!(
        outcome(
            recall(dir.path())
                .args(["search", "hello", "--exclude-current"])
                .env("CLAUDE_CODE_SESSION_ID", "  ")
        ),
        (
            Some(0),
            BOTH_SEEDED_SESSIONS.to_owned(),
            EXCLUDE_CURRENT_HAS_NO_EFFECT_STDERR.to_owned()
        )
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

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "hello", "--exclude-current"])),
        (
            Some(0),
            BOTH_SEEDED_SESSIONS.to_owned(),
            EXCLUDE_CURRENT_HAS_NO_EFFECT_STDERR.to_owned()
        )
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

    assert_eq!(
        outcome(recall(dir.path()).args(["classify", "--all", "--dry-run"])),
        (
            Some(0),
            concat!(
                "s1 [automated] <command-message>clear</command-message>\n",
                "dry-run: 1 session(s) would be classified (1 automated, 0 interactive)\n",
            )
            .to_owned(),
            String::new()
        )
    );
    assert_eq!(
        outcome(recall(dir.path()).args(["classify", "--all"])),
        (
            Some(0),
            "classified 1 session(s): 1 automated, 0 interactive\n".to_owned(),
            String::new()
        )
    );
}

#[test]
fn classify_preview_of_missing_database_keeps_the_path_absent() {
    let dir = TempDir::new().unwrap();
    let nothing_classified = serde_json::json!({
        "data": {"automated": 0, "classified": 0, "dry_run": true, "interactive": 0},
        "degraded": false,
        "notes": [],
    });

    assert_eq!(
        json_outcome(
            recall(dir.path())
                .arg("--db-path")
                .arg(dir.path().join("recall.db"))
                .args(["classify", "--dry-run", "--json"])
        ),
        (Some(0), nothing_classified.clone(), String::new())
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
    assert_eq!(
        json_outcome(
            recall(dir.path())
                .arg("--db-path")
                .arg(dir.path().join("absent/nested/recall.db"))
                .args(["classify", "--dry-run", "--json", "--all"])
        ),
        (Some(0), nothing_classified, String::new())
    );
    assert_eq!(listing(dir.path()), Vec::<String>::new());
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

fn authentication_notes_match(session_id: &str, chunk: u64) -> serde_json::Value {
    serde_json::json!({
        "chunk": chunk,
        "excerpt": "authentication notes",
        "inspect": format!("recall show {session_id} --chunk {chunk} --window 3"),
        "project": "/proj",
        "session_id": session_id,
        "slug": session_id,
        "source": "claude",
        "timestamp": 1_772_323_200_000_i64,
    })
}

const AUTOMATED_EXCLUDED_NOTE: &str = "1 automated session(s) matching this search were excluded by default; use --include-automated to include them";
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

    assert_eq!(
        json_outcome(recall(dir.path()).args(["search", "authentication", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [authentication_notes_match("human", 3), authentication_notes_match("human2", 4)]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE, AUTOMATED_EXCLUDED_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

#[test]
fn search_omits_automated_note_when_no_automated_session_matches_the_query() {
    let dir = TempDir::new().unwrap();
    index_claude_sessions(
        dir.path(),
        &[("auto", &[AUTOMATED_OPENER]), ("human", &[HUMAN_AUTH])],
    );

    assert_eq!(
        json_outcome(recall(dir.path()).args(["search", "authentication", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [authentication_notes_match("human", 2)]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
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

    assert_eq!(
        json_outcome(recall(dir.path()).args([
            "search",
            "authentication",
            "--project",
            "/proj",
            "--json"
        ])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [authentication_notes_match("human", 3)]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
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

    assert_eq!(
        json_outcome(recall(dir.path()).args(["search", "authentication", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": []},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE, AUTOMATED_EXCLUDED_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
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
    let read_only = ReadOnlyUntilDropped::new(dir.path());
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
    let _read_only = ReadOnlyUntilDropped::new(dir.path());
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

#[cfg(unix)]
#[test]
fn status_of_an_unreadable_database_in_a_read_only_directory_names_the_path_it_was_given() {
    use std::os::unix::fs::PermissionsExt;

    let root = TempDir::new().unwrap();
    let dir_whose_uri_escapes_a_space_and_a_hash = root.path().join("a b#c");
    fs::create_dir(&dir_whose_uri_escapes_a_space_and_a_hash).unwrap();
    let db = dir_whose_uri_escapes_a_space_and_a_hash.join("recall.db");
    fs::write(&db, b"").unwrap();
    fs::set_permissions(&db, fs::Permissions::from_mode(0o000)).unwrap();
    let _read_only = ReadOnlyUntilDropped::new(&dir_whose_uri_escapes_a_space_and_a_hash);
    if root_skip::skip_if_root(&dir_whose_uri_escapes_a_space_and_a_hash) {
        return;
    }

    assert_eq!(
        outcome(recall(root.path()).env("RECALL_DB", &db).arg("status")),
        (
            Some(74),
            String::new(),
            format!(
                "error: unable to open database file: {}: Error code 14: unable to open database file\n",
                db.display()
            )
        )
    );
}

fn seed_stale_wal(dir: &Path) {
    seed_indexed_session(dir);
    let db = dir.join("recall.db");
    let _ = fs::remove_file(db.with_file_name("recall.db-shm"));
    fs::write(db.with_file_name("recall.db-wal"), b"uncheckpointed-frames").unwrap();
}

#[cfg(unix)]
struct ReadOnlyUntilDropped<'a>(&'a Path);

#[cfg(unix)]
impl<'a> ReadOnlyUntilDropped<'a> {
    fn new(dir: &'a Path) -> Self {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir, fs::Permissions::from_mode(0o555)).unwrap();
        Self(dir)
    }
}

#[cfg(unix)]
impl Drop for ReadOnlyUntilDropped<'_> {
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
    let _read_only = ReadOnlyUntilDropped::new(dir.path());
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

const UNSUPPORTED_INDEX: &str = "this index's schema predates recall 0.14.0 and cannot be opened; build a new index with `recall index` at another `--db-path`, or upgrade this one by running `recall classify` from recall 0.16.1 on it";

fn seed_v0_13_index(dir: &Path) {
    rusqlite::Connection::open(dir.join("recall.db"))
        .unwrap()
        .execute_batch(
            "CREATE TABLE sessions (
                 session_id TEXT PRIMARY KEY, source TEXT, file_path TEXT, project TEXT,
                 slug TEXT, timestamp INTEGER, mtime REAL, session_type TEXT,
                 files_scanned INTEGER
             );
             CREATE VIRTUAL TABLE messages USING fts5(
                 session_id UNINDEXED, role, text, tokenize='trigram'
             );
             INSERT INTO sessions (session_id, source, file_path, project, slug, timestamp)
                 VALUES ('s1', 'claude', '/gone/s1.jsonl', '/p', 'slug', 0);
             INSERT INTO messages (session_id, role, text) VALUES ('s1', 'user', 'hello world');",
        )
        .unwrap();
}

#[test]
fn every_command_refuses_an_index_older_than_v0_14_with_one_message_and_leaves_the_file_as_it_was()
{
    let dir = TempDir::new().unwrap();
    seed_v0_13_index(dir.path());
    let stored = || {
        let mut names: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name())
            .collect();
        names.sort();
        (names, fs::read(dir.path().join("recall.db")).unwrap())
    };
    let before = stored();
    let refusal = format!("error: {UNSUPPORTED_INDEX}\n");

    for args in [
        vec!["search", "hello"],
        vec!["status"],
        vec!["show", "s1"],
        vec!["classify", "--dry-run"],
        vec!["classify"],
    ] {
        assert_eq!(
            outcome(recall(dir.path()).args(&args)),
            (Some(65), String::new(), refusal.clone()),
            "recall {args:?}"
        );
    }
    for command in ["index", "rebuild"] {
        // why-not: a whole-stderr match, because stage timings fill the rest of stderr
        let (code, stdout, stderr) = outcome(recall(dir.path()).arg(command));
        assert_eq!((code, stdout.as_str()), (Some(65), ""), "recall {command}");
        assert!(stderr.ends_with(&refusal), "recall {command}: {stderr}");
    }
    let (_, doctor, _) = outcome(recall(dir.path()).arg("doctor"));
    assert!(
        doctor.contains(
            "  schema [FAIL] this index's schema predates recall 0.14.0 and cannot be opened -> build a new index with `recall index` at another `--db-path`, or upgrade this one by running `recall classify` from recall 0.16.1 on it\n"
        ),
        "{doctor}"
    );
    assert!(stored() == before, "a refused command changed the index");
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
    let _read_only = ReadOnlyUntilDropped::new(dir.path());
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
fn text_read_commands_on_a_stale_snapshot_in_a_read_only_directory_print_the_note_on_stderr() {
    let dir = TempDir::new().unwrap();
    seed_stale_wal(dir.path());
    let _read_only = ReadOnlyUntilDropped::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }
    let note = format!("note: {READ_ONLY_DIR_NOTE}\n");

    assert_eq!(
        outcome(recall(dir.path()).args(["show", "s1"])),
        (Some(0), SHOW_S1.to_owned(), note.clone())
    );
    assert_eq!(
        outcome(recall(dir.path()).args(["search", "hello"])),
        (
            Some(0),
            S1_MATCH.to_owned(),
            format!("{MODEL_NOT_INSTALLED_STDERR}{note}")
        )
    );
    assert_eq!(
        outcome(recall(dir.path()).arg("status")),
        (Some(0), STATUS_OF_ONE_SESSION.to_owned(), note.clone())
    );
    assert_eq!(
        outcome(recall(dir.path()).arg("doctor")),
        (
            Some(0),
            concat!(
                "Health: healthy\n",
                "  integrity [ok]\n",
                "  orphan_embeddings [ok]\n",
                "  orphan_chunks [ok]\n",
                "  model_identity [ok]\n",
                "  model [ok] model not installed; search runs FTS-only\n",
            )
            .to_owned(),
            format!("{MODEL_NOT_INSTALLED_STDERR}{note}")
        )
    );
}

#[cfg(unix)]
#[test]
fn search_of_a_stale_snapshot_that_shows_no_sessions_keeps_the_note() {
    let no_sessions = TempDir::new().unwrap();
    assert_eq!(exit_code(recall(no_sessions.path()).arg("index")), 0);
    let no_schema = TempDir::new().unwrap();
    rusqlite::Connection::open(no_schema.path().join("recall.db"))
        .unwrap()
        .execute_batch("PRAGMA journal_mode = WAL;")
        .unwrap();
    let _ = fs::remove_file(no_sessions.path().join("recall.db-shm"));
    fs::write(
        no_sessions.path().join("recall.db-wal"),
        b"uncheckpointed-frames",
    )
    .unwrap();
    fs::write(
        no_schema.path().join("recall.db-wal"),
        b"uncheckpointed-frames",
    )
    .unwrap();
    let _no_sessions_read_only = ReadOnlyUntilDropped::new(no_sessions.path());
    let _no_schema_read_only = ReadOnlyUntilDropped::new(no_schema.path());
    if root_skip::skip_if_root(no_sessions.path()) {
        return;
    }
    let text = (
        Some(0),
        String::new(),
        format!("{NO_SESSIONS_INDEXED}note: {READ_ONLY_DIR_NOTE}\n"),
    );

    assert_eq!(
        outcome(recall(no_sessions.path()).args(["search", "anything"])),
        text,
        "every session may sit in the log this snapshot cannot read"
    );
    assert_eq!(
        json_outcome(recall(no_sessions.path()).args(["search", "anything", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": []},
                "degraded": true,
                "notes": [READ_ONLY_DIR_NOTE],
            }),
            NO_SESSIONS_INDEXED.to_owned()
        )
    );
    assert_eq!(
        outcome(recall(no_schema.path()).args(["search", "anything"])),
        text,
        "every session may sit in the log this snapshot cannot read"
    );
}

#[cfg(unix)]
#[test]
fn search_and_status_of_a_stale_snapshot_with_an_older_schema_add_the_note_to_the_error() {
    let dir = TempDir::new().unwrap();
    rusqlite::Connection::open(dir.path().join("recall.db"))
        .unwrap()
        .execute_batch(
            "PRAGMA journal_mode = WAL; CREATE TABLE sessions (session_id TEXT PRIMARY KEY);",
        )
        .unwrap();
    fs::write(dir.path().join("recall.db-wal"), b"uncheckpointed-frames").unwrap();
    let _read_only = ReadOnlyUntilDropped::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }
    let error = (
        Some(65),
        String::new(),
        format!("error: {UNSUPPORTED_INDEX}; {READ_ONLY_DIR_NOTE}\n"),
    );

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "anything"])),
        error,
        "the newer schema may sit in the log this snapshot cannot read"
    );
    assert_eq!(
        outcome(recall(dir.path()).arg("status")),
        error,
        "the newer schema may sit in the log this snapshot cannot read"
    );
}

#[cfg(unix)]
#[test]
fn status_of_a_stale_snapshot_built_for_another_model_lists_the_snapshot_note_before_the_identity_note()
 {
    let dir = TempDir::new().unwrap();
    seed_indexed_session(dir.path());
    let db = dir.path().join("recall.db");
    rusqlite::Connection::open(&db)
        .unwrap()
        .execute(
            "UPDATE embedding_identity SET fingerprint = 'another model'",
            [],
        )
        .unwrap();
    let _ = fs::remove_file(db.with_file_name("recall.db-shm"));
    fs::write(db.with_file_name("recall.db-wal"), b"uncheckpointed-frames").unwrap();
    let _read_only = ReadOnlyUntilDropped::new(dir.path());
    if root_skip::skip_if_root(dir.path()) {
        return;
    }

    assert_eq!(
        json_outcome(recall(dir.path()).args(["status", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": status_of_one_session(),
                "degraded": true,
                "notes": [
                    READ_ONLY_DIR_NOTE,
                    "semantic search unavailable: index embedding identity mismatch; use a separate --db-path for this model or select the matching --model-manifest; results are text-only",
                ],
            }),
            String::new()
        )
    );
}

#[test]
fn text_search_and_status_print_the_notes_they_add_on_stderr() {
    let automated = TempDir::new().unwrap();
    index_claude_sessions(
        automated.path(),
        &[
            ("auto", &[AUTOMATED_OPENER, AUTOMATED_AUTH_TURN]),
            ("human", &[HUMAN_AUTH]),
        ],
    );
    let other_model = TempDir::new().unwrap();
    seed_indexed_session(other_model.path());
    rusqlite::Connection::open(other_model.path().join("recall.db"))
        .unwrap()
        .execute(
            "UPDATE embedding_identity SET fingerprint = 'another model'",
            [],
        )
        .unwrap();
    let unknown_source = TempDir::new().unwrap();
    index_claude_sessions(
        unknown_source.path(),
        &[
            (
                "valid",
                &[&user_line("authentication notes", "2026-03-01T00:00:00Z")],
            ),
            (
                "weird",
                &[&user_line("authentication notes", "2026-03-01T00:00:00Z")],
            ),
        ],
    );
    mark_source_unknown(unknown_source.path(), "weird");

    let identity_note = "note: semantic search unavailable: index embedding identity mismatch; use a separate --db-path for this model or select the matching --model-manifest; results are text-only\n";

    assert_eq!(
        outcome(recall(automated.path()).args(["search", "authentication"])),
        (
            Some(0),
            concat!(
                "Found 1 sessions:\n",
                "\n",
                "[1] 2026-03-01 | human | proj [claude]\n",
                "    /proj\n",
                "    ID: human\n",
                "    > authentication notes\n",
                "    inspect: recall show human --chunk 3 --window 3\n",
                "\n",
            )
            .to_owned(),
            format!("{MODEL_NOT_INSTALLED_STDERR}note: {AUTOMATED_EXCLUDED_NOTE}\n")
        )
    );
    assert_eq!(
        outcome(recall(other_model.path()).args(["search", "hello"])),
        (Some(0), S1_MATCH.to_owned(), identity_note.to_owned())
    );
    assert_eq!(
        outcome(recall(other_model.path()).arg("status")),
        (
            Some(0),
            STATUS_OF_ONE_SESSION.to_owned(),
            identity_note.to_owned()
        )
    );
    // why-not: a whole-stderr match, because a timestamped tracing line logs the skipped session first
    let (_, _, stderr) =
        outcome(recall(unknown_source.path()).args(["search", "authentication", "--limit", "1"]));
    assert!(
        stderr.ends_with(&format!("\nnote: {INCOMPLETE_RESULTS_NOTE}\n")),
        "{stderr}"
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
        let mut cmd = recall(dir.path());
        cmd.env("RECALL_CLAUDE_DIR", &claude)
            .env("RECALL_CODEX_DIR", &codex)
            .args(args);
        cmd
    };
    let skipped_lines = format!(
        "{}: skipped 1 invalid JSON line(s) and 1 invalid UTF-8 line(s); repair the source file, then rerun `recall index` (use `recall rebuild` if size and mtime are unchanged)",
        path.display()
    );
    let incomplete_tail = format!(
        "{}: 1 incomplete final line(s), possibly still being written; wait for the next append, then rerun `recall index`; if the writer has stopped, repair the source file",
        path.display()
    );
    let diagnostics = serde_json::json!([{
        "file_path": path.to_str().unwrap(),
        "incomplete_tail_lines": 1,
        "invalid_json_lines": 1,
        "invalid_utf8_lines": 1,
        "read_error": false,
        "source": "claude",
    }]);
    let notes = [skipped_lines.clone(), incomplete_tail.clone()];
    let parsed_with_the_excluded_codex_log_read_as_empty = [
        ONE_SESSION_PARSED_AND_CHUNKED,
        &[("files_discovered", 2), ("files_empty", 1)],
    ]
    .concat();
    let unchanged = [
        ("chunks_pending_snapshot", 1),
        ("chunks_remaining_snapshot", 1),
        ("chunks_unattempted", 1),
        ("files_discovered", 2),
        ("files_empty", 1),
        ("files_unchanged", 1),
        ("sessions_stored", 1),
    ];

    let (code, first, first_stderr) =
        index_json_outcome_without_stage_timings(&mut run(&["index", "--json"]));
    assert_eq!(
        (code, first),
        (
            Some(0),
            index_envelope_without_the_model(
                &parsed_with_the_excluded_codex_log_read_as_empty,
                &diagnostics,
                &notes
            )
        ),
        "excluded Codex events are not parse loss"
    );
    let (code, second, second_stderr) =
        index_json_outcome_without_stage_timings(&mut run(&["index", "--json"]));
    assert_eq!(
        (code, second),
        (
            Some(0),
            index_envelope_without_the_model(&unchanged, &diagnostics, &notes)
        )
    );
    let (code, rebuilt, rebuilt_stderr) =
        index_json_outcome_without_stage_timings(&mut run(&["rebuild", "--json"]));
    assert_eq!(
        (code, rebuilt),
        (
            Some(0),
            index_envelope_without_the_model(
                &parsed_with_the_excluded_codex_log_read_as_empty,
                &diagnostics,
                &notes
            )
        )
    );
    let (code, stdout, stderr) = outcome(&mut run(&["index"]));
    assert_eq!((code, stdout), (Some(0), String::new()));
    // why-not: a whole-stderr match, because stage timings fill the rest of stderr
    assert!(
        stderr.contains("\n1 sessions stored; 1 file(s) with unresolved parsing diagnostics\n"),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("\nwarning: {skipped_lines}\n")),
        "{stderr}"
    );
    assert!(
        stderr.contains(&format!("\nwarning: {incomplete_tail}\n")),
        "{stderr}"
    );
    let repeats_skipped_text =
        "the skipped lines hold conversation text, which no output may repeat";
    assert!(
        !first_stderr.contains("secret-"),
        "{repeats_skipped_text}: {first_stderr}"
    );
    assert!(
        !second_stderr.contains("secret-"),
        "{repeats_skipped_text}: {second_stderr}"
    );
    assert!(
        !rebuilt_stderr.contains("secret-"),
        "{repeats_skipped_text}: {rebuilt_stderr}"
    );
    assert!(
        !stderr.contains("secret-"),
        "{repeats_skipped_text}: {stderr}"
    );

    fs::write(
        &path,
        [
            good.as_slice(),
            b"{\"type\":\"assistant\",\"message\":\"repaired-answer\"}\n",
        ]
        .concat(),
    )
    .unwrap();
    let (code, repaired, _) =
        index_json_outcome_without_stage_timings(&mut run(&["index", "--json"]));

    assert_eq!(
        (code, repaired),
        (
            Some(0),
            index_envelope_without_the_model(
                &[
                    ("chunks_pending_snapshot", 1),
                    ("chunks_reconciled_created_committed", 1),
                    ("chunks_remaining_snapshot", 1),
                    ("chunks_removed_committed", 1),
                    ("chunks_unattempted", 1),
                    ("files_discovered", 2),
                    ("files_empty", 1),
                    ("files_updated", 1),
                    ("files_updated_committed", 1),
                    ("sessions_stored", 1),
                ],
                &serde_json::json!([]),
                &[]
            )
        )
    );
    assert_eq!(
        json_outcome(&mut run(&["show", "partial", "--json"])),
        (
            Some(0),
            serde_json::json!({
                "data": {
                    "messages": [
                        {"role": "user", "text": "secret-conversation-body"},
                        {"role": "assistant", "text": "repaired-answer"},
                    ],
                    "project": "",
                    "session_id": "partial",
                    "slug": "partial",
                    "source": "claude",
                    "timestamp": null,
                },
                "degraded": false,
                "notes": [],
            }),
            String::new()
        )
    );
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
        let mut cmd = recall(dir.path());
        cmd.env("RECALL_CLAUDE_DIR", &claude)
            .args(["index", "--json"]);
        cmd
    };

    fs::set_permissions(&path, fs::Permissions::from_mode(0o000)).unwrap();
    let (code, unreadable, _) = index_json_outcome_without_stage_timings(&mut run());
    fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
    let (recovered_code, recovered, _) = index_json_outcome_without_stage_timings(&mut run());

    assert_eq!(
        (code, unreadable),
        (
            Some(0),
            index_envelope_without_the_model(
                &[("files_discovered", 1), ("files_failed", 1)],
                &serde_json::json!([{
                    "file_path": path.to_str().unwrap(),
                    "incomplete_tail_lines": 0,
                    "invalid_json_lines": 0,
                    "invalid_utf8_lines": 0,
                    "read_error": true,
                    "source": "claude",
                }]),
                &[format!(
                    "{}: 1 file could not be read; check file access and UTF-8 path encoding, then rerun `recall index`; existing indexed data was retained",
                    path.display()
                )]
            )
        )
    );
    assert_eq!(
        (recovered_code, recovered),
        (
            Some(0),
            index_envelope_without_the_model(
                ONE_SESSION_PARSED_AND_CHUNKED,
                &serde_json::json!([]),
                &[]
            )
        )
    );
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

    // why-not: a whole-stderr match, because stage timings fill it and the first summary line ends in the run's time
    let first = index_stderr(dir.path(), &claude_dir);
    assert_in_order(&first, "\nIndexed 1 sessions in ", "\nCreated 1 chunks\n");

    let second = index_stderr(dir.path(), &claude_dir);
    assert_in_order(
        &second,
        "\n1 sessions up to date\n",
        "\nChunks up to date\n",
    );
}

#[test]
fn index_stderr_summarizes_the_scan_before_chunking_starts() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), SESSION_LINE);

    let stderr = index_stderr(dir.path(), &claude_dir);

    assert_in_order(
        &stderr,
        "\nIndexed 1 sessions in ",
        "\nindex: chunking: started",
    );
}

#[test]
fn index_stderr_warns_about_parse_diagnostics_after_the_backfill_and_before_the_chunk_summary() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), &format!("{SESSION_LINE}\nnot json\n"));

    let stderr = index_stderr(dir.path(), &claude_dir);

    let parse_warning = "skipped 1 invalid JSON line(s)";
    assert_in_order(
        &stderr,
        "\nindex: legacy_backfill: committed",
        parse_warning,
    );
    assert_in_order(&stderr, parse_warning, "\nCreated 1 chunks\n");
}

#[test]
fn index_stderr_summarizes_files_with_unresolved_parse_diagnostics() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), &format!("{SESSION_LINE}\nnot json\n"));
    index_stderr(dir.path(), &claude_dir);

    let unchanged = index_stderr(dir.path(), &claude_dir);

    assert!(
        unchanged.contains("\n1 sessions stored; 1 file(s) with unresolved parsing diagnostics\n"),
        "{unchanged}"
    );
}

#[test]
fn index_stderr_warns_when_a_root_with_indexed_sessions_is_unavailable() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), SESSION_LINE);
    index_stderr(dir.path(), &claude_dir);

    let stderr = index_stderr(dir.path(), &dir.path().join("absent"));

    assert!(
        stderr.contains("\nwarning: claude root unavailable — preserved 1 existing session(s), none deleted this run; re-run `recall index` once the root is back so a full scan can reconcile any genuinely deleted sessions\n"),
        "{stderr}"
    );
}

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
        let mut cmd = isolated();
        cmd.env_remove("CLAUDE_CONFIG_DIR")
            .args(["search", "hello"]);
        cmd
    };
    let found_the_session = (
        Some(0),
        S1_MATCH.to_owned(),
        MODEL_NOT_INSTALLED_STDERR.to_owned(),
    );

    let (code, _, stderr) = outcome(
        isolated()
            .env("CLAUDE_CONFIG_DIR", &config_dir)
            .arg("index"),
    );
    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(outcome(&mut search_hello()), found_the_session);

    let (code, _, stderr) = outcome(
        isolated()
            .env("CLAUDE_CONFIG_DIR", "")
            .current_dir(&work_dir)
            .arg("index"),
    );

    assert_eq!(code, Some(0), "{stderr}");
    assert_eq!(
        outcome(&mut search_hello()),
        found_the_session,
        "neither the cwd's projects/ dir nor the default root becomes the Claude root; reading either empty root in full would delete the session"
    );
    // why-not: a whole-stderr match, because stage timings fill the rest of stderr
    assert!(
        stderr.contains("\nwarning: claude root could not be fully read — preserved 1 existing session(s), none deleted this run; check that the root is an absolute path and check the directory's permissions and access, then re-run `recall index` so the next full scan can reconcile any genuinely deleted sessions\n"),
        "{stderr}"
    );
}

#[test]
fn classify_warns_once_when_a_session_source_is_missing() {
    let dir = TempDir::new().unwrap();
    let claude_dir = write_session(dir.path(), SESSION_LINE);
    index_stderr(dir.path(), &claude_dir);
    fs::remove_file(claude_dir.join("s1.jsonl")).unwrap();

    assert_eq!(
        outcome(recall(dir.path()).args(["classify", "--all"])),
        (
            Some(0),
            "classified 0 session(s): 0 automated, 0 interactive\n".to_owned(),
            "warning: provenance unavailable (missing file): 1 session(s); existing labels retained, unclassified rows use first-turn fallback\n".to_owned()
        )
    );
}

#[test]
fn recall_search_with_a_not_query_returns_only_the_sessions_without_the_excluded_word() {
    let dir = TempDir::new().unwrap();
    let wanted = user_line("authentication notes", "2026-03-01T00:00:00Z");
    let excluded = user_line(
        "authentication and deployment notes",
        "2026-03-01T00:00:00Z",
    );
    let unrelated = user_line("checklist for release", "2026-03-01T00:00:00Z");
    index_claude_sessions(
        dir.path(),
        &[
            ("wanted", &[wanted.as_str()]),
            ("excluded", &[excluded.as_str()]),
            ("unrelated", &[unrelated.as_str()]),
        ],
    );

    assert_eq!(
        outcome(recall(dir.path()).args(["search", "authentication NOT deployment"])),
        (
            Some(0),
            concat!(
                "Found 1 sessions:\n",
                "\n",
                "[1] 2026-03-01 | wanted | proj [claude]\n",
                "    /proj\n",
                "    ID: wanted\n",
                "    > authentication notes\n",
                "    inspect: recall show wanted --chunk 3 --window 3\n",
                "\n",
            )
            .to_owned(),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
    );
}

const INCOMPLETE_RESULTS_NOTE: &str = "results may be incomplete: some matched sessions that could have ranked in these results were excluded because their index rows were unreadable or carry an unrecognized source; for unreadable rows, run `recall index` to rebuild";

fn result_ids(envelope: &serde_json::Value) -> Vec<&str> {
    envelope["data"]["results"]
        .as_array()
        .unwrap()
        .iter()
        .map(|result| result["session_id"].as_str().unwrap())
        .collect()
}

fn user_line(text: &str, timestamp: &str) -> String {
    format!(
        r#"{{"type":"user","cwd":"/proj","message":{{"role":"user","content":"{text}"}},"timestamp":"{timestamp}"}}"#
    )
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

#[test]
fn search_with_limit_one_prints_the_first_result_of_the_default_limit() {
    let dir = TempDir::new().unwrap();
    let text_with_equal_raw_ranks = "authentication notes";
    let a1 = user_line(text_with_equal_raw_ranks, "2026-01-01T00:00:00Z");
    let a2 = user_line(text_with_equal_raw_ranks, "2026-01-02T00:00:00Z");
    let a3 = user_line(text_with_equal_raw_ranks, "2026-01-03T00:00:00Z");
    let a4 = user_line(text_with_equal_raw_ranks, "2026-01-04T00:00:00Z");
    let a5 = user_line(text_with_equal_raw_ranks, "2026-01-05T00:00:00Z");
    let future_date_with_the_largest_recency_boost = "2999-01-01T00:00:00Z";
    let newest = user_line(
        text_with_equal_raw_ranks,
        future_date_with_the_largest_recency_boost,
    );
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

    let (code, default_limit, stderr) =
        json_outcome(recall(dir.path()).args(["search", "authentication", "--json"]));

    // why-not: comparing the whole order, because a1 to a5 differ only in age and reorder as they grow older
    assert_eq!(
        (
            code,
            result_ids(&default_limit)[0],
            result_ids(&default_limit).len(),
            &default_limit["notes"],
            stderr
        ),
        (
            Some(0),
            "z-newest",
            6,
            &serde_json::json!([MODEL_NOT_INSTALLED_NOTE]),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        ),
        "only the recency blend ranks z-newest first; it sorts last by id, so a cut before the blend drops it"
    );
    assert_eq!(
        json_outcome(recall(dir.path()).args([
            "search",
            "authentication",
            "--limit",
            "1",
            "--json"
        ])),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [{
                    "chunk": 6,
                    "excerpt": "authentication notes",
                    "inspect": "recall show z-newest --chunk 6 --window 3",
                    "project": "/proj",
                    "session_id": "z-newest",
                    "slug": "z-newest",
                    "source": "claude",
                    "timestamp": 32_472_144_000_000_i64,
                }]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE],
            }),
            MODEL_NOT_INSTALLED_STDERR.to_owned()
        )
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

    // why-not: comparing stderr, because it holds a timestamped tracing warning for the skipped session
    let (code, envelope, _) = json_outcome(recall(dir.path()).args([
        "search",
        "authentication",
        "--limit",
        "1",
        "--json",
    ]));

    assert_eq!(
        (code, envelope),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [{
                    "chunk": 1,
                    "excerpt": "authentication",
                    "inspect": "recall show valid --chunk 1 --window 3",
                    "project": "/proj",
                    "session_id": "valid",
                    "slug": "valid",
                    "source": "claude",
                    "timestamp": 1_772_323_200_000_i64,
                }]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE],
            })
        )
    );
}

#[test]
fn search_that_skips_an_unknown_source_session_that_could_rank_says_so_in_the_note() {
    let dir = TempDir::new().unwrap();
    let valid = user_line("authentication notes", "2026-03-01T00:00:00Z");
    let weird = user_line("authentication notes", "2026-03-01T00:00:00Z");
    index_claude_sessions(dir.path(), &[("valid", &[&valid]), ("weird", &[&weird])]);
    mark_source_unknown(dir.path(), "weird");

    // why-not: comparing stderr, because it holds a timestamped tracing warning for the skipped session
    let (code, envelope, _) = json_outcome(recall(dir.path()).args([
        "search",
        "authentication",
        "--limit",
        "1",
        "--json",
    ]));

    assert_eq!(
        (code, envelope),
        (
            Some(0),
            serde_json::json!({
                "data": {"results": [authentication_notes_match("valid", 1)]},
                "degraded": true,
                "notes": [MODEL_NOT_INSTALLED_NOTE, INCOMPLETE_RESULTS_NOTE],
            })
        )
    );
}
