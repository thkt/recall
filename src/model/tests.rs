#![cfg(test)]
use super::*;
#[test]
fn checksum_format_and_persisted_identity_survive_digest_updates() {
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("artifact");
    for (bytes, expected) in [
        (
            Vec::new(),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
        ),
        (
            b"abc".to_vec(),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
        ),
        (
            vec![b'a'; 1024 * 1024 + 1],
            "4a3f0c0c213adea174f9a3d4c13177315b588bdb2e9c1012d3d0bf0453ca0f6a",
        ),
    ] {
        fs::write(&path, bytes).unwrap();
        assert_eq!(hash_file(&path).unwrap(), expected);
    }
    assert_eq!(
        ModelSpec::default().identity(),
        "2af2d2da81d827b88831ac2c7a9a52438d9d917daa4b99f5f45b4005867d265e"
    );
}

#[test]
fn process_model_selection_is_idempotent_and_cannot_change_mid_run() {
    let original = selected().clone();
    select(None).unwrap();
    select(None).unwrap();
    let mut next = original.clone();
    next.document_prefix.push_str("candidate ");
    let dir = tempfile::TempDir::new().unwrap();
    let path = dir.path().join("candidate.json");
    fs::write(&path, serde_json::to_vec(&next).unwrap()).unwrap();
    assert!(
        select(Some(&path))
            .unwrap_err()
            .to_string()
            .contains("already selected")
    );
    assert_eq!(selected().identity(), original.identity());
}
#[cfg(unix)]
#[test]
fn probe_distinguishes_success_unavailable_failure_and_reaps_timeouts() {
    for (script, expected) in [
        ("exit 0", Some(true)),
        ("exit 2", Some(false)),
        ("exit 1", None),
    ] {
        let actual = run_probe(
            Command::new("/bin/sh").args(["-c", script]),
            Duration::from_secs(1),
        );
        match expected {
            Some(value) => assert_eq!(actual.unwrap(), value),
            None => assert!(actual.unwrap_err().to_string().contains("probe failed")),
        }
    }
    let dir = tempfile::TempDir::new().unwrap();
    let pid_file = dir.path().join("pid");
    let mut cmd = Command::new("/bin/sh");
    cmd.args(["-c", "echo $$ > \"$1\"; exec sleep 60", "probe"])
        .arg(&pid_file);
    let result = run_probe(&mut cmd, Duration::from_millis(200));
    assert!(result.unwrap_err().to_string().contains("timed out"));
    let pid = fs::read_to_string(pid_file).unwrap();
    assert!(
        !Command::new("/bin/kill")
            .args(["-0", pid.trim()])
            .stderr(Stdio::null())
            .status()
            .unwrap()
            .success(),
        "timed-out child must have been killed and reaped before returning"
    );
    assert!(
        run_probe(
            &mut Command::new(dir.path().join("absent")),
            Duration::from_secs(1)
        )
        .is_err()
    );
}
#[test]
#[ignore = "requires explicitly staged public model artifacts"]
fn public_model_contract_cache_and_loader() {
    assert!(env::var_os("RECALL_MODEL_DIR").is_some());
    let root = cached_artifacts().unwrap().unwrap();
    assert!(root.join("model.gguf").is_file());
    let model = load_verified().unwrap();
    assert_eq!(
        model
            .embed_query("public cache verification")
            .unwrap()
            .len(),
        768
    );
    assert!(gpu_available().is_ok());
}
#[test]
fn verified_artifact_replaces_only_after_complete_checksum_match() {
    let dir = tempfile::TempDir::new().unwrap();
    let dest = dir.path().join("model.gguf");
    fs::write(&dest, b"old").unwrap();
    let bytes = b"new verified weights";
    let digest = digest_hex(&Sha256::digest(bytes));
    install_artifact(dir.path(), "model.gguf", &digest, &mut bytes.as_slice()).unwrap();
    assert_eq!(fs::read(dest).unwrap(), bytes);
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}
#[test]
fn bad_checksum_and_partial_stream_preserve_previous_artifact() {
    struct Failed;
    impl Read for Failed {
        fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
            Err(io::Error::other("interrupted download"))
        }
    }
    let dir = tempfile::TempDir::new().unwrap();
    let dest = dir.path().join("model.gguf");
    fs::write(&dest, b"old").unwrap();
    assert!(
        install_artifact(
            dir.path(),
            "model.gguf",
            &"0".repeat(64),
            &mut b"wrong".as_slice()
        )
        .is_err()
    );
    assert!(install_artifact(dir.path(), "model.gguf", &"0".repeat(64), &mut Failed).is_err());
    assert_eq!(fs::read(dest).unwrap(), b"old");
    assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
}
#[test]
fn occupied_temporary_path_is_never_deleted_or_overwritten() {
    let dir = tempfile::TempDir::new().unwrap();
    let temp = dir
        .path()
        .join(format!(".model.gguf.{}.download", process::id()));
    fs::write(&temp, b"someone else").unwrap();
    assert!(
        install_artifact(
            dir.path(),
            "model.gguf",
            &"0".repeat(64),
            &mut b"new".as_slice()
        )
        .is_err()
    );
    assert_eq!(fs::read(temp).unwrap(), b"someone else");
    assert!(!dir.path().join("model.gguf").exists());
}
fn empty_index() -> Connection {
    let c = Connection::open_in_memory().unwrap();
    c.execute_batch("CREATE TABLE vec_chunks(id INTEGER)")
        .unwrap();
    c
}
#[test]
fn model_or_tokenizer_or_prefix_changes_select_a_new_index_identity() {
    let old = ModelSpec::default();
    let id = old.identity();
    for field in ["weights", "tokenizer", "prefix", "revision"] {
        let mut next = old.clone();
        match field {
            "weights" => next.model_sha256 = "0".repeat(64),
            "tokenizer" => next.tokenizer_sha256 = "1".repeat(64),
            "prefix" => next.query_prefix.push_str("new "),
            _ => next.revision = "2".repeat(40),
        }
        next.validate().unwrap();
        assert_ne!(id, next.identity());
    }
}
#[test]
fn legacy_vectors_are_rejected_without_stamping_or_deleting_them() {
    let mut c = empty_index();
    c.execute("INSERT INTO vec_chunks VALUES(7)", []).unwrap();
    assert!(bind_index(&mut c, &ModelSpec::default()).is_err());
    assert_eq!(
        c.query_row("SELECT id FROM vec_chunks", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        7
    );
    assert_eq!(
        c.query_row(
            "SELECT count(*) FROM sqlite_master WHERE name='embedding_identity'",
            [],
            |r| r.get::<_, i64>(0)
        )
        .unwrap(),
        0
    );
}
#[test]
fn different_model_cannot_rebind_an_existing_index_and_old_model_can_return() {
    let old = ModelSpec::default();
    let mut next = old.clone();
    next.document_prefix.push_str("next ");
    let mut c = empty_index();
    bind_index(&mut c, &old).unwrap();
    c.execute("INSERT INTO vec_chunks VALUES(19)", []).unwrap();
    assert!(bind_index(&mut c, &next).is_err());
    check_index(&c, &old).unwrap();
    bind_index(&mut c, &old).unwrap();
    assert_eq!(
        c.query_row("SELECT id FROM vec_chunks", [], |r| r.get::<_, i64>(0))
            .unwrap(),
        19
    );
    let mut new = empty_index();
    bind_index(&mut new, &next).unwrap();
    check_index(&new, &next).unwrap();
    assert!(check_index(&new, &old).is_err());
}
#[test]
fn invalid_profile_shapes_fail_closed() {
    assert!(
        ModelSpec {
            model_sha256: "not a checksum".into(),
            ..ModelSpec::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        ModelSpec {
            dimensions: 1024,
            ..ModelSpec::default()
        }
        .validate()
        .is_err()
    );
    assert!(
        ModelSpec {
            revision: "main".into(),
            ..ModelSpec::default()
        }
        .validate()
        .is_err()
    );
}
