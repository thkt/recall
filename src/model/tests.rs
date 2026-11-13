#![cfg(test)]
use super::*;
use crate::embedder::MockEmbedder;
use std::cell::Cell;
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
#[should_panic(expected = "a unit test reached the host's model cache")]
fn a_unit_test_that_reads_the_model_cache_without_a_staged_model_fails() {
    let _ = cached_artifacts();
}
#[test]
#[ignore = "requires explicitly staged public model artifacts"]
fn public_model_contract_cache_and_loader() {
    assert!(env::var_os("RECALL_MODEL_DIR").is_some());
    let artifacts = cached_artifacts().unwrap().unwrap();
    assert!(artifacts.root.join("model.gguf").is_file());
    let model = artifacts.load().unwrap();
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

#[test]
fn probe_command_hands_the_child_the_parent_verified_artifact_root() {
    let root = Path::new("/verified/artifacts");

    let command = probe_command(Path::new("/bin/recall"), root, "{}");

    let handed = command
        .get_envs()
        .find(|(key, _)| *key == "__RECALL_GEMMA_ARTIFACTS")
        .and_then(|(_, value)| value);
    assert_eq!(handed, Some(root.as_os_str()));
}

#[test]
fn probe_child_uses_the_handed_root_without_rehashing_it() {
    let dir = tempfile::TempDir::new().unwrap();
    fs::write(dir.path().join("model.gguf"), b"not the pinned bytes").unwrap();
    fs::write(dir.path().join("tokenizer.json"), b"{}").unwrap();

    let artifacts = probe_artifacts(Some(dir.path().as_os_str().to_owned())).unwrap();

    assert_eq!(
        artifacts.root,
        dir.path(),
        "the parent already verified these bytes, so the child must not re-check them"
    );
}

#[test]
fn probe_child_without_a_handed_root_is_an_error() {
    let error = probe_artifacts(None)
        .err()
        .expect("a missing root must fail");

    assert!(
        error.to_string().contains("__RECALL_GEMMA_ARTIFACTS"),
        "{error}"
    );
}

/// Replaces only the probe child, the GPU check and the model load; the probe record is real.
pub(crate) struct FakeHost {
    pub(crate) probe_result: fn() -> Result<bool>,
    pub(crate) gpu: fn() -> Result<bool>,
    pub(crate) model: fn() -> Result<Arc<dyn Embed>>,
    pub(crate) probes: Cell<usize>,
    pub(crate) loads: Cell<usize>,
}

impl FakeHost {
    pub(crate) fn healthy() -> Self {
        Self {
            probe_result: || Ok(true),
            gpu: || Ok(true),
            model: || Ok(Arc::new(MockEmbedder::new())),
            probes: Cell::new(0),
            loads: Cell::new(0),
        }
    }

    pub(crate) fn run<R>(&self, f: impl FnOnce(&LoadSteps<'_>) -> R) -> R {
        let probe = |_: &Path| {
            self.probes.set(self.probes.get() + 1);
            (self.probe_result)()
        };
        let load = |_: &Path| {
            self.loads.set(self.loads.get() + 1);
            (self.model)()
        };
        f(&LoadSteps {
            probe: &probe,
            gpu_available: &self.gpu,
            load: &load,
        })
    }

    /// Hands `f` a model host whose artifacts resolve to `root` and whose record store,
    /// keyed `env`, lives in `records_dir`; commands then pick their loader as in production.
    pub(crate) fn serve<R>(
        &self,
        root: &Path,
        records_dir: &Path,
        f: impl FnOnce(&ModelHost<'_>) -> R,
    ) -> R {
        self.run(|steps| {
            let resolve = || {
                let records = ProbeRecords::new(records_dir.to_path_buf(), "env".to_owned());
                Ok((root.to_path_buf(), Some(records)))
            };
            f(&ModelHost {
                resolve: &resolve,
                steps,
            })
        })
    }
}

fn store(dir: &tempfile::TempDir) -> ProbeRecords {
    ProbeRecords::new(dir.path().join("probe"), "env".to_owned())
}

fn failing_self_check() -> Result<Arc<dyn Embed>> {
    Ok(Arc::new(MockEmbedder::failing_after(0)))
}

#[test]
fn search_loader_with_a_matching_record_loads_without_starting_the_probe() {
    let dir = tempfile::TempDir::new().unwrap();
    let records = store(&dir);
    records.record_success();
    let host = FakeHost::healthy();

    let loaded = host.run(|steps| load_for_search(dir.path(), Some(&records), steps));

    assert!(loaded.is_ok());
    assert_eq!(host.probes.get(), 0);
    assert_eq!(host.loads.get(), 1);
    assert!(
        records.may_skip(),
        "a finished skipped load keeps the record"
    );
}

#[test]
fn search_loader_skip_path_without_a_gpu_returns_backend_unavailable_and_clears_the_record() {
    let dir = tempfile::TempDir::new().unwrap();
    let records = store(&dir);
    records.record_success();
    let host = FakeHost {
        gpu: || Ok(false),
        ..FakeHost::healthy()
    };

    let loaded = host.run(|steps| load_for_search(dir.path(), Some(&records), steps));

    assert_eq!(loaded.err(), Some(DegradedReason::BackendUnavailable));
    assert_eq!(host.probes.get(), 0);
    assert_eq!(host.loads.get(), 0);
    assert!(!records.may_skip());
}

#[test]
fn search_loader_skip_path_load_or_self_check_failure_clears_the_record() {
    let failures: [FakeHost; 3] = [
        FakeHost {
            gpu: || anyhow::bail!("backend init failed"),
            ..FakeHost::healthy()
        },
        FakeHost {
            model: || anyhow::bail!("model load failed"),
            ..FakeHost::healthy()
        },
        FakeHost {
            model: failing_self_check,
            ..FakeHost::healthy()
        },
    ];
    for host in failures {
        let dir = tempfile::TempDir::new().unwrap();
        let records = store(&dir);
        records.record_success();

        let loaded = host.run(|steps| load_for_search(dir.path(), Some(&records), steps));

        assert_eq!(loaded.err(), Some(DegradedReason::ProbeFailed));
        assert_eq!(host.probes.get(), 0);
        assert!(!records.may_skip());
        let _ = host.run(|steps| load_for_search(dir.path(), Some(&records), steps));
        assert_eq!(host.probes.get(), 1, "the next search runs the probe");
    }
}

#[test]
fn search_loader_runs_the_probe_when_the_record_is_missing_or_the_environment_is_marked_no_skip() {
    let dir = tempfile::TempDir::new().unwrap();
    let records = store(&dir);
    let host = FakeHost::healthy();

    assert!(
        host.run(|steps| load_for_search(dir.path(), Some(&records), steps))
            .is_ok()
    );
    assert_eq!(host.probes.get(), 1, "no record: the probe runs");

    drop(records.begin_skipped_load().unwrap());
    for expected in [2, 3] {
        assert!(
            host.run(|steps| load_for_search(dir.path(), Some(&records), steps))
                .is_ok()
        );
        assert_eq!(
            host.probes.get(),
            expected,
            "a crashed skipped load keeps the environment probing even after a new success"
        );
    }

    assert!(
        !dir.path().join("probe").join("env.ok").exists(),
        "an environment marked no-skip gets no record that nothing would read"
    );

    let probes_without_store = host.run(|steps| load_for_search(dir.path(), None, steps));
    assert!(probes_without_store.is_ok());
    assert_eq!(host.probes.get(), 4, "no usable store: the probe runs");
}

#[test]
fn search_loader_runs_the_probe_when_the_running_marker_cannot_be_created() {
    let dir = tempfile::TempDir::new().unwrap();
    let records = store(&dir);
    records.record_success();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir.path().join("probe"), fs::Permissions::from_mode(0o500)).unwrap();
    }
    let host = FakeHost::healthy();

    let loaded = host.run(|steps| load_for_search(dir.path(), Some(&records), steps));

    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(dir.path().join("probe"), fs::Permissions::from_mode(0o700)).unwrap();
    }
    assert!(loaded.is_ok(), "search must not fail or degrade");
    assert_eq!(host.probes.get(), 1);
}

#[test]
fn default_loader_runs_the_probe_even_when_a_matching_record_exists() {
    let dir = tempfile::TempDir::new().unwrap();
    let records = store(&dir);
    records.record_success();
    let host = FakeHost::healthy();

    let loaded = host.run(|steps| load_probing(dir.path(), Some(&records), steps));

    assert!(loaded.is_ok());
    assert_eq!(host.probes.get(), 1);
    assert!(records.may_skip());
}

#[test]
fn default_loader_records_success_only_after_the_parent_self_check_passes() {
    let not_recorded: [(FakeHost, Option<DegradedReason>); 4] = [
        (
            FakeHost {
                probe_result: || Ok(false),
                ..FakeHost::healthy()
            },
            Some(DegradedReason::BackendUnavailable),
        ),
        (
            FakeHost {
                probe_result: || anyhow::bail!("isolated model probe timed out"),
                ..FakeHost::healthy()
            },
            Some(DegradedReason::ProbeFailed),
        ),
        (
            FakeHost {
                model: || anyhow::bail!("model load failed"),
                ..FakeHost::healthy()
            },
            Some(DegradedReason::ProbeFailed),
        ),
        // A failed parent self-check keeps today's result: the model is returned unrecorded.
        (
            FakeHost {
                model: failing_self_check,
                ..FakeHost::healthy()
            },
            None,
        ),
    ];
    for (host, reason) in not_recorded {
        let dir = tempfile::TempDir::new().unwrap();
        let records = store(&dir);
        records.record_success();

        let loaded = host.run(|steps| load_probing(dir.path(), Some(&records), steps));

        assert_eq!(loaded.err(), reason);
        assert!(
            !records.may_skip(),
            "a failure clears and withholds the record"
        );
    }

    let dir = tempfile::TempDir::new().unwrap();
    let records = store(&dir);
    let loaded = FakeHost::healthy().run(|steps| load_probing(dir.path(), Some(&records), steps));
    assert!(loaded.is_ok());
    assert!(records.may_skip());
}
