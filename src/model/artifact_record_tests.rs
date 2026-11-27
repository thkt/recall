use super::*;
use crate::commands::doctor::run_doctor_in;
use crate::commands::index::index_embedder;
use crate::commands::open_or_create_db;
use crate::commands::search::run_search_in;
use crate::commands::status::run_status_in;
use crate::embedder::MockEmbedder;
use crate::embedding::Embed;
use crate::error::RecallError;
use crate::model::tests::{FakeHost, SharedCell};
use crate::model::{ModelHost, cache, hash_file};
use crate::test_support::{search_command, seed_healthy_index};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

struct SyntheticModel {
    dir: tempfile::TempDir,
    root: PathBuf,
    spec: ModelSpec,
    hashes: SharedCell<usize>,
}

impl SyntheticModel {
    fn new() -> Self {
        let dir = tempfile::TempDir::new().unwrap();
        let root = dir.path().join("model");
        fs::create_dir(&root).unwrap();
        fs::write(root.join("model.gguf"), b"synthetic weights").unwrap();
        fs::write(root.join("tokenizer.json"), b"{\"synthetic\":true}").unwrap();
        let spec = ModelSpec {
            model_sha256: hash_file(&root.join("model.gguf")).unwrap(),
            tokenizer_sha256: hash_file(&root.join("tokenizer.json")).unwrap(),
            ..ModelSpec::default()
        };
        Self {
            dir,
            root,
            spec,
            hashes: SharedCell::new(0),
        }
    }

    fn records_dir(&self) -> PathBuf {
        self.dir.path().join("verified-artifacts")
    }

    fn probe_dir(&self) -> PathBuf {
        self.dir.path().join("probe")
    }

    fn resolve_with(
        &self,
        spec: &ModelSpec,
        check: ArtifactCheck,
    ) -> Result<ResolvedArtifacts, DegradedReason> {
        let records = ArtifactRecords::new(self.records_dir());
        resolve_artifacts(check, &self.root, spec, Some(&records), || {
            self.hashes.set(self.hashes.get() + 1);
            cache::cached_artifacts("recall", spec, &self.root)
        })
    }

    fn resolve(&self, check: ArtifactCheck) -> Result<ResolvedArtifacts, DegradedReason> {
        self.resolve_with(&self.spec, check)
    }

    fn serve<R>(&self, host: &FakeHost, f: impl FnOnce(&ModelHost<'_>) -> R) -> R {
        host.serve_resolving(&|check| self.resolve(check), &self.probe_dir(), f)
    }

    fn search(&self, host: &FakeHost) -> Result<Arc<dyn Embed>, DegradedReason> {
        self.serve(host, search_loader)
    }

    fn record_count(&self) -> usize {
        fs::read_dir(self.records_dir()).map_or(0, |entries| {
            entries
                .flatten()
                .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "json"))
                .count()
        })
    }

    fn record_path(&self) -> PathBuf {
        ArtifactRecords::new(self.records_dir())
            .record(&self.root, &self.spec)
            .path
    }
}

fn search_loader(model_host: &ModelHost<'_>) -> Result<Arc<dyn Embed>, DegradedReason> {
    model_host.load_for_search()
}

fn no_recover() -> Result<(), RecallError> {
    panic!("model recovery must not run")
}

#[test]
fn search_after_a_passed_check_returns_the_same_directory_without_hashing() {
    let model = SyntheticModel::new();

    let first = model.resolve(ArtifactCheck::TrustRecord).unwrap();
    let second = model.resolve(ArtifactCheck::TrustRecord).unwrap();

    assert_eq!(
        (
            model.hashes.get(),
            first.root,
            second.root,
            second.trusted_record.is_some()
        ),
        (1, model.root.clone(), model.root.clone(), true)
    );
}

#[test]
fn search_hashes_a_same_size_rewrite_with_its_mtime_restored_and_clears_the_record() {
    let model = SyntheticModel::new();
    model.resolve(ArtifactCheck::HashAndRecord).unwrap();
    let path = model.root.join("model.gguf");
    let original = fs::metadata(&path).unwrap();
    thread::sleep(Duration::from_millis(20));

    fs::write(&path, b"tampered weights!").unwrap();
    fs::File::options()
        .write(true)
        .open(&path)
        .unwrap()
        .set_modified(original.modified().unwrap())
        .unwrap();

    let rewritten = fs::metadata(&path).unwrap();
    assert_eq!(
        (rewritten.len(), rewritten.modified().unwrap()),
        (original.len(), original.modified().unwrap()),
        "the rewrite keeps the size and mtime"
    );
    let resolved = model.resolve(ArtifactCheck::TrustRecord);
    assert_eq!(
        (resolved.err(), model.hashes.get(), model.record_count()),
        (Some(DegradedReason::ProbeFailed), 2, 0)
    );
}

#[test]
fn index_doctor_and_status_hash_every_time_and_status_writes_no_record() {
    let model = SyntheticModel::new();
    let db_path = model.dir.path().join("recall.db");
    seed_healthy_index(&open_or_create_db(&db_path).unwrap());
    let db = Some(db_path);
    let host = FakeHost::healthy();
    let mut steps = Vec::new();
    let mut observe =
        |name: &'static str| steps.push((name, model.hashes.get(), model.record_count()));

    let status = model
        .serve(&host, |h| run_status_in(false, &db, h))
        .unwrap();
    assert_eq!(status.data["model_ready"], true);
    observe("status without a record");
    let search = model
        .serve(&host, |h| {
            run_search_in(search_command("healthy chunk"), &db, h)
        })
        .unwrap();
    assert!(!search.degraded);
    observe("first search");
    model
        .serve(&host, |h| {
            run_search_in(search_command("healthy chunk"), &db, h)
        })
        .unwrap();
    observe("second search");
    assert!(model.serve(&host, index_embedder).is_ok());
    observe("index");
    let doctor = model
        .serve(&host, |h| run_doctor_in(false, false, &db, h, no_recover))
        .unwrap();
    assert_eq!(doctor.data["healthy"], true);
    observe("doctor");
    model
        .serve(&host, |h| run_status_in(false, &db, h))
        .unwrap();
    observe("status with a record");

    assert_eq!(
        steps,
        [
            ("status without a record", 1, 0),
            ("first search", 2, 1),
            ("second search", 2, 1),
            ("index", 3, 1),
            ("doctor", 4, 1),
            ("status with a record", 5, 1),
        ]
    );
}

#[test]
fn search_hashes_again_when_a_manifest_checksum_changes() {
    let changes: [fn(&mut ModelSpec); 2] = [
        |s| s.model_sha256 = "0".repeat(64),
        |s| s.tokenizer_sha256 = "0".repeat(64),
    ];
    for change in changes {
        let model = SyntheticModel::new();
        model.resolve(ArtifactCheck::HashAndRecord).unwrap();
        let mut next = model.spec.clone();
        change(&mut next);

        let resolved = model.resolve_with(&next, ArtifactCheck::TrustRecord);

        assert_eq!(
            (resolved.err(), model.hashes.get()),
            (Some(DegradedReason::ProbeFailed), 2),
            "{next:?}"
        );
    }
}

#[test]
fn search_hashes_and_loads_when_the_record_cannot_be_written_or_read() {
    let unwritable = SyntheticModel::new();
    fs::create_dir(unwritable.records_dir()).unwrap();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(unwritable.records_dir(), fs::Permissions::from_mode(0o500)).unwrap();
    }
    let unreadable = SyntheticModel::new();
    fs::create_dir_all(unreadable.record_path()).unwrap();

    for model in [&unwritable, &unreadable] {
        let host = FakeHost::healthy();
        let loads: Vec<bool> = (0..2).map(|_| model.search(&host).is_ok()).collect();
        assert_eq!((loads, model.hashes.get()), (vec![true, true], 2));
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        fs::set_permissions(unwritable.records_dir(), fs::Permissions::from_mode(0o700)).unwrap();
    }
}

#[test]
fn a_failed_parent_self_check_on_the_probe_path_after_a_skipped_hash_clears_the_record() {
    let model = SyntheticModel::new();
    assert!(model.search(&FakeHost::healthy()).is_ok());
    fs::remove_file(model.probe_dir().join("env.ok")).unwrap();
    let failing = FakeHost {
        model: || Ok(Arc::new(MockEmbedder::failing_after(0))),
        ..FakeHost::healthy()
    };

    let returned = model.search(&failing).is_ok();
    let after_failure = (
        failing.probes.get(),
        model.hashes.get(),
        model.record_count(),
    );
    assert!(model.search(&FakeHost::healthy()).is_ok());

    assert_eq!(
        (returned, after_failure, model.hashes.get()),
        (true, (1, 1, 0), 2)
    );
}

#[test]
fn a_failed_load_after_a_skipped_hash_clears_the_record_and_the_next_search_hashes() {
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
            model: || Ok(Arc::new(MockEmbedder::failing_after(0))),
            ..FakeHost::healthy()
        },
    ];
    for failing in failures {
        let model = SyntheticModel::new();
        let healthy = FakeHost::healthy();
        assert!(model.search(&healthy).is_ok());

        let failed = model.search(&failing);
        let after_failure = (model.hashes.get(), model.record_count());
        assert!(model.search(&healthy).is_ok());

        assert_eq!(
            (failed.err(), after_failure, model.hashes.get()),
            (Some(DegradedReason::ProbeFailed), (1, 0), 2)
        );
    }
}
