use super::*;
use std::{
    io::{BufRead, BufReader, Write as _},
    net::TcpListener,
    process::{Command, Stdio},
    thread,
    time::Instant,
};

fn spec() -> ModelSpec {
    ModelSpec {
        model_sha256: digest_hex(&Sha256::digest(b"shared weights")),
        tokenizer_sha256: digest_hex(&Sha256::digest(b"shared tokenizer")),
        // Reuse tests must work without any accessible download endpoint.
        model_url: "http://127.0.0.1:1/model".into(),
        tokenizer_url: "http://127.0.0.1:1/tokenizer".into(),
        ..ModelSpec::default()
    }
}

fn child(root: &Path, mode: &str) -> Command {
    let mut command = Command::new(env::current_exe().unwrap());
    command.args(["--exact", "model::cache::tests::cache_child", "--nocapture"]);
    for key in [
        "HF_HOME",
        "XDG_CACHE_HOME",
        "SAE_MODEL_DIR",
        "RECALL_MODEL_DIR",
    ] {
        command.env_remove(key);
    }
    command
        .env("HOME", root)
        .env("__MODEL_CACHE_TEST_ROOT", root)
        .env("__MODEL_CACHE_TEST_MODE", mode);
    command
}

#[cfg(unix)]
fn same_file(left: &Path, right: &Path) {
    use std::os::unix::fs::MetadataExt;
    let left = fs::metadata(left).unwrap();
    let right = fs::metadata(right).unwrap();
    assert_eq!((left.dev(), left.ino()), (right.dev(), right.ino()));
}

#[test]
fn cache_child() {
    let Ok(mode) = env::var("__MODEL_CACHE_TEST_MODE") else {
        return;
    };
    let home = PathBuf::from(env::var_os("__MODEL_CACHE_TEST_ROOT").unwrap());
    let spec = spec();
    let shared = cache_dir("recall", &spec);
    match mode.as_str() {
        "paths" => {
            let expected = PathBuf::from(env::var_os("__MODEL_CACHE_TEST_BASE").unwrap());
            assert_eq!(shared.parent().unwrap(), expected.join("embedding-models"));
            assert_eq!(shared, cache_dir("sae", &spec));
            assert_eq!(shared.file_name().unwrap().len(), 64);
        }
        "override" => {
            assert_eq!(shared, home.join("explicit-recall"));
            assert_eq!(cache_dir("sae", &spec), home.join("explicit-sae"));
            assert!(
                cached_artifacts("recall", &spec, &shared)
                    .unwrap()
                    .is_none()
            );
            assert!(
                cached_artifacts("sae", &spec, &cache_dir("sae", &spec))
                    .unwrap()
                    .is_none()
            );
            assert!(!shared.exists());
        }
        "reuse" => {
            for app in ["recall", "sae"] {
                assert_eq!(cache_dir(app, &spec), shared);
                assert_eq!(
                    cached_artifacts(app, &spec, &shared).unwrap(),
                    Some(shared.clone())
                );
                // Download also reuses verified files without a live server.
                download_artifacts(app, &spec, &shared).unwrap();
            }
            let source = home.join(".cache/sae/models/old-sae");
            assert_eq!(
                fs::read(source.join("model.gguf")).unwrap(),
                b"shared weights"
            );
            #[cfg(unix)]
            for name in ["model.gguf", "tokenizer.json"] {
                same_file(&source.join(name), &shared.join(name));
            }
            // Corruption is detected on reads, and a download can repair it from legacy bytes.
            fs::rename(shared.join("model.gguf"), shared.join("saved-model")).unwrap();
            fs::write(shared.join("model.gguf"), b"corrupt").unwrap();
            assert!(
                cached_artifacts("recall", &spec, &shared)
                    .unwrap_err()
                    .to_string()
                    .contains("checksum")
            );
            download_artifacts("recall", &spec, &shared).unwrap();
            assert_eq!(
                fs::read(shared.join("model.gguf")).unwrap(),
                b"shared weights"
            );
        }
        "download" => {
            let address = env::var("__MODEL_CACHE_TEST_SERVER").unwrap();
            let spec = ModelSpec {
                model_url: format!("{address}/model"),
                tokenizer_url: format!("{address}/tokenizer"),
                ..spec
            };
            let app = env::var("__MODEL_CACHE_TEST_APP").unwrap();
            download_artifacts(&app, &spec, &cache_dir(&app, &spec)).unwrap();
        }
        "exit-with-lock" => {
            let _lock = lock_cache(&shared).unwrap();
            process::exit(0);
        }
        _ => panic!("unknown cache test mode"),
    }
}

#[test]
fn shared_cache_ignores_app_encoder_identity_but_separates_different_artifacts() {
    let original = spec();
    let mut other = original.clone();
    other.implementation = "other-app".into();
    other.query_prefix.push_str("different prefix");
    other.backend_revision = "0".repeat(40);
    other.model_url.push_str("?mirror");
    assert_ne!(original.identity(), other.identity());
    assert_eq!(artifact_identity(&original), artifact_identity(&other));
    other.model_sha256 = "1".repeat(64);
    assert_ne!(artifact_identity(&original), artifact_identity(&other));
    other = original.clone();
    other.tokenizer_sha256 = "2".repeat(64);
    assert_ne!(artifact_identity(&original), artifact_identity(&other));
}

#[test]
fn shared_cache_respects_hf_xdg_home_and_explicit_overrides() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    for (hf, xdg, expected) in [
        (
            Some(root.join("hf")),
            Some(root.join("xdg")),
            root.join("hf"),
        ),
        (None, Some(root.join("xdg")), root.join("xdg")),
        (None, Some(PathBuf::from("relative")), root.join(".cache")),
        (None, None, root.join(".cache")),
    ] {
        let mut command = child(root, "paths");
        if let Some(hf) = hf {
            command.env("HF_HOME", hf);
        }
        if let Some(xdg) = xdg {
            command.env("XDG_CACHE_HOME", xdg);
        }
        assert!(
            command
                .env("__MODEL_CACHE_TEST_BASE", expected)
                .status()
                .unwrap()
                .success()
        );
    }
    let legacy = root.join(".cache/sae/models/old-sae");
    fs::create_dir_all(&legacy).unwrap();
    fs::write(legacy.join("model.gguf"), b"shared weights").unwrap();
    fs::write(legacy.join("tokenizer.json"), b"shared tokenizer").unwrap();
    assert!(
        child(root, "override")
            .env("SAE_MODEL_DIR", root.join("explicit-sae"))
            .env("RECALL_MODEL_DIR", root.join("explicit-recall"))
            .env("HF_HOME", root.join("hf"))
            .status()
            .unwrap()
            .success()
    );
}

#[test]
fn existing_peer_cache_is_verified_and_reused_offline_without_copying() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path();
    let valid = root.join(".cache/sae/models/old-sae");
    let corrupt = root.join(".cache/recall/models/old-recall");
    fs::create_dir_all(&valid).unwrap();
    fs::create_dir_all(&corrupt).unwrap();
    fs::write(valid.join("model.gguf"), b"shared weights").unwrap();
    fs::write(valid.join("tokenizer.json"), b"shared tokenizer").unwrap();
    fs::write(corrupt.join("model.gguf"), b"invalid legacy model").unwrap();
    fs::write(corrupt.join("tokenizer.json"), b"invalid legacy tokenizer").unwrap();
    // A newly configured cache root must still discover the original home cache.
    assert!(
        child(root, "reuse")
            .env("HF_HOME", root.join("new-hf"))
            .status()
            .unwrap()
            .success()
    );
    assert_eq!(
        fs::read(corrupt.join("model.gguf")).unwrap(),
        b"invalid legacy model"
    );
}

#[test]
fn independently_cached_artifacts_can_be_combined_and_invalid_legacy_is_skipped() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("shared");
    let weights = dir.path().join("old-recall");
    let tokenizer = dir.path().join("old-sae");
    let corrupt = dir.path().join("invalid");
    for path in [&root, &weights, &tokenizer, &corrupt] {
        fs::create_dir_all(path).unwrap();
    }
    fs::write(weights.join("model.gguf"), b"shared weights").unwrap();
    fs::write(tokenizer.join("tokenizer.json"), b"shared tokenizer").unwrap();
    fs::write(corrupt.join("model.gguf"), b"invalid").unwrap();
    let legacy = [corrupt, weights, tokenizer];
    download_artifacts_at(&root, &spec(), &legacy).unwrap();
    assert_eq!(validate_at(&root, &spec()).unwrap(), Some(root.clone()));
    #[cfg(unix)]
    {
        same_file(&root.join("model.gguf"), &legacy[1].join("model.gguf"));
        same_file(
            &root.join("tokenizer.json"),
            &legacy[2].join("tokenizer.json"),
        );
    }
}

#[test]
fn concurrent_sae_and_recall_download_each_artifact_only_once() {
    let dir = tempfile::tempdir().unwrap();
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    listener.set_nonblocking(true).unwrap();
    let address = format!("http://{}", listener.local_addr().unwrap());
    let server = thread::spawn(move || {
        for (path, body) in [
            ("/model", &b"shared weights"[..]),
            ("/tokenizer", &b"shared tokenizer"[..]),
        ] {
            let deadline = Instant::now() + Duration::from_secs(15);
            let mut stream = loop {
                match listener.accept() {
                    Ok((stream, _)) => break stream,
                    Err(error) if error.kind() == ErrorKind::WouldBlock => {
                        assert!(Instant::now() < deadline, "missing request {path}");
                        thread::sleep(Duration::from_millis(5));
                    }
                    Err(error) => panic!("{error}"),
                }
            };
            stream.set_nonblocking(false).unwrap();
            stream
                .set_read_timeout(Some(Duration::from_secs(5)))
                .unwrap();
            let mut request = BufReader::new(&stream);
            let mut line = String::new();
            request.read_line(&mut line).unwrap();
            assert_eq!(line.split_whitespace().nth(1), Some(path));
            loop {
                line.clear();
                request.read_line(&mut line).unwrap();
                if line == "\r\n" || line.is_empty() {
                    break;
                }
            }
            drop(request);
            write!(
                stream,
                "HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len()
            )
            .unwrap();
            stream.write_all(body).unwrap();
        }
    });
    let mut children = Vec::new();
    for app in ["sae", "recall"] {
        children.push(
            child(dir.path(), "download")
                .env("__MODEL_CACHE_TEST_SERVER", &address)
                .env("__MODEL_CACHE_TEST_APP", app)
                .stdout(Stdio::null())
                .spawn()
                .unwrap(),
        );
    }
    for mut child in children {
        assert!(child.wait().unwrap().success());
    }
    server.join().unwrap();
}

#[test]
fn cache_lock_survives_process_exit_without_stale_locking() {
    let dir = tempfile::tempdir().unwrap();
    assert!(
        child(dir.path(), "exit-with-lock")
            .status()
            .unwrap()
            .success()
    );
    let base = dir.path().join(".cache/embedding-models");
    let root = fs::read_dir(base).unwrap().next().unwrap().unwrap().path();
    let lock = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join(".install.lock"))
        .unwrap();
    lock.try_lock()
        .expect("exiting a process must release its lock");
    let second = fs::OpenOptions::new()
        .read(true)
        .write(true)
        .open(root.join(".install.lock"))
        .unwrap();
    assert!(
        second.try_lock().is_err(),
        "all writers must lock the same persistent file"
    );
}

#[test]
#[ignore = "requires explicitly staged public model artifacts in a fixture HOME"]
fn public_existing_model_is_reused_without_network_or_inference() {
    let spec = ModelSpec::default();
    let root = cache_dir("recall", &spec);
    let legacy = legacy_dirs();
    assert!(!legacy.is_empty(), "stage an old sae or recall cache first");
    assert_eq!(
        cached_artifacts("recall", &spec, &root).unwrap(),
        Some(root.clone())
    );
    assert_eq!(
        cached_artifacts("sae", &spec, &cache_dir("sae", &spec)).unwrap(),
        Some(root.clone())
    );
    #[cfg(unix)]
    for (name, _, _) in artifacts(&spec) {
        same_file(&legacy[0].join(name), &root.join(name));
    }
}

#[test]
fn model_and_tokenizer_updates_reuse_the_unchanged_artifact() {
    let dir = tempfile::tempdir().unwrap();
    let old = dir.path().join("old-shared");
    fs::create_dir_all(&old).unwrap();
    fs::write(old.join("model.gguf"), b"shared weights").unwrap();
    fs::write(old.join("tokenizer.json"), b"shared tokenizer").unwrap();
    let next_tokenizer = dir.path().join("next-tokenizer");
    fs::create_dir_all(&next_tokenizer).unwrap();
    fs::write(next_tokenizer.join("tokenizer.json"), b"updated tokenizer").unwrap();
    let next = ModelSpec {
        tokenizer_sha256: digest_hex(&Sha256::digest(b"updated tokenizer")),
        ..spec()
    };
    let cache = dir.path().join("new-shared");
    download_artifacts_at(&cache, &next, &[old.clone(), next_tokenizer]).unwrap();
    #[cfg(unix)]
    same_file(&old.join("model.gguf"), &cache.join("model.gguf"));
    assert_eq!(validate_at(&cache, &next).unwrap(), Some(cache));
}
