use rusqlite::Connection;
use serde_json::{Value, json};
use std::{
    env, fs,
    path::Path,
    process::{Command, Output},
};
use tempfile::TempDir;

fn cli(root: &Path) -> Command {
    let mut command = Command::new(env!("CARGO_BIN_EXE_recall"));
    command
        .env("RECALL_DB", root.join("index.db"))
        .env("RECALL_MODEL_DIR", root.join("model"))
        .env("RECALL_CLAUDE_DIR", root.join("claude"))
        .env("RECALL_CODEX_DIR", root.join("codex"))
        .env("XDG_DATA_HOME", root.join("data"))
        .env("XDG_CACHE_HOME", root.join("cache"))
        .env("HF_HOME", root.join("hf"))
        .env_remove("RECALL_MODEL_MANIFEST")
        .env_remove("CLAUDE_CODE_SESSION_ID")
        .env_remove("__RECALL_GEMMA_PROBE")
        .env_remove("__RECALL_GEMMA_SPEC");
    command
}
fn run(command: &mut Command) -> Output {
    command.output().unwrap()
}
fn value(output: &Output) -> Value {
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).unwrap()
}
fn manifest(root: &Path) -> Value {
    serde_json::from_slice(&run(cli(root).args(["model", "manifest"])).stdout).unwrap()
}
fn fixture(root: &Path) {
    fs::create_dir_all(root.join("claude")).unwrap();
    fs::write(root.join("claude/public.jsonl"), concat!(
        "{\"type\":\"user\",\"cwd\":\"/fixture\",\"message\":{\"role\":\"user\",\"content\":\"quokka rollback instructions\"},\"timestamp\":\"2026-03-01T00:00:00Z\"}\n",
        "{\"type\":\"assistant\",\"message\":{\"role\":\"assistant\",\"content\":\"Keep the previous database for rollback.\"}}\n"
    )).unwrap();
}

#[test]
fn manifest_validation_and_cache_failures_are_explicit() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    let original = manifest(root);
    let path = root.join("manifest.json");
    for (field, invalid) in [
        ("dimensions", json!(1024)),
        ("revision", json!("main")),
        ("pooling", json!("max")),
        ("backend_revision", json!("0".repeat(40))),
        ("model_url", json!("http://example.invalid/model")),
        ("unknown_option", json!(true)),
    ] {
        let mut spec = original.clone();
        spec[field] = invalid;
        fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
        let out = run(cli(root)
            .arg("--model-manifest")
            .arg(&path)
            .args(["model", "manifest"]));
        assert!(
            !out.status.success(),
            "invalid manifest field accepted: {field}"
        );
        assert!(!root.join("index.db").exists());
    }
    fixture(root);
    assert!(run(cli(root).arg("index")).status.success());
    fs::create_dir(root.join("model")).unwrap();
    fs::write(root.join("model/model.gguf"), b"not weights").unwrap();
    fs::write(root.join("model/tokenizer.json"), b"not a tokenizer").unwrap();
    let out = value(&run(cli(root).args(["search", "quokka", "--json"])));
    assert_eq!(out["degraded"], true);
    assert!(out["data"]["results"].as_array().unwrap().len() == 1);
    let mut spec = original;
    spec["model_url"] = json!("https://127.0.0.1:9/model.gguf");
    fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    let out = run(cli(root)
        .arg("--model-manifest")
        .arg(&path)
        .args(["model", "download"]));
    assert_eq!(out.status.code(), Some(75));
    assert_eq!(
        fs::read(root.join("model/model.gguf")).unwrap(),
        b"not weights"
    );
}

#[test]
fn foreign_index_remains_readable_and_cannot_be_rebound() {
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    fixture(root);
    assert!(run(cli(root).arg("index")).status.success());
    let path = root.join("index.db");
    Connection::open(&path)
        .unwrap()
        .execute("UPDATE embedding_identity SET fingerprint='foreign'", [])
        .unwrap();
    let before = fs::read(&path).unwrap();
    let out = value(&run(cli(root).args(["search", "quokka", "--json"])));
    assert_eq!(out["degraded"], true);
    assert_eq!(out["data"]["results"].as_array().unwrap().len(), 1);
    let status = value(&run(cli(root).args(["status", "--json"])));
    assert_eq!(status["degraded"], true);
    assert!(
        status["notes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|n| n.as_str().unwrap().contains("identity mismatch"))
    );
    assert_eq!(run(cli(root).arg("rebuild")).status.code(), Some(65));
    assert_eq!(fs::read(&path).unwrap(), before);
}

#[test]
#[ignore = "requires explicitly staged pinned public artifacts; runs under CI coverage"]
fn public_model_cli_contract() {
    let model =
        env::var_os("RECALL_GEMMA_TEST_ARTIFACT_DIR").expect("explicit public model directory");
    let dir = TempDir::new().unwrap();
    let root = dir.path();
    fixture(root);
    let invoke = || {
        let mut c = cli(root);
        c.env("RECALL_MODEL_DIR", &model);
        c
    };
    let download = run(invoke().args(["model", "download"]));
    let stderr = String::from_utf8_lossy(&download.stderr);
    let probe = match download.status.code() {
        Some(0) => "success",
        Some(70) if stderr.contains("backend unavailable") => "backend unavailable",
        Some(75) if stderr.contains("isolated model probe timed out") => "timeout",
        status => panic!("unexpected public-model download/probe result {status:?}: {stderr}"),
    };
    let check_state = |output: &Value| {
        if output["degraded"] == true {
            assert!(
                output["notes"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .any(|n| n.as_str().unwrap().contains("embedding model unavailable")),
                "verified artifacts must degrade with an explicit model-unavailable note: {output}"
            );
        } else {
            assert_eq!(output["degraded"], false);
        }
    };
    let indexed = value(&run(invoke().args(["index", "--json"])));
    check_state(&indexed);
    assert_eq!(
        indexed["data"]["embedded"].as_u64().unwrap() > 0,
        indexed["degraded"] == false
    );
    let first = value(&run(invoke().args(["search", "quokka", "--json"])));
    check_state(&first);
    assert_eq!(first["data"]["results"].as_array().unwrap().len(), 1);
    let spec = manifest(root);
    let path = root.join("archived.json");
    fs::write(&path, serde_json::to_vec(&spec).unwrap()).unwrap();
    let again = value(&run(invoke()
        .arg("--model-manifest")
        .arg(&path)
        .args(["search", "quokka", "--json"])));
    check_state(&again);
    assert_eq!(first["data"], again["data"]);
    let out = run(invoke()
        .env("__RECALL_GEMMA_PROBE", "1")
        .env("__RECALL_GEMMA_SPEC", "{}"));
    assert_eq!(out.status.code(), Some(1));
    println!(
        "public model CLI: download probe={probe}; index degraded={}; search degraded={}; isolated index and archived manifest verified",
        indexed["degraded"], first["degraded"]
    );
}
