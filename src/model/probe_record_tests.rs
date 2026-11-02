use super::*;
use std::time::Duration;

fn snapshot() -> Snapshot {
    Snapshot {
        binary_len: 1_000,
        binary_modified_ns: 1_700_000_000_000_000_000,
        model_identity: "model".to_owned(),
        os_build: "25A100".to_owned(),
        hardware_model: "Mac15,3".to_owned(),
        ggml_vars: vec![("GGML_METAL_NDEBUG".into(), "1".into())],
    }
}

fn records(dir: &Path) -> ProbeRecords {
    ProbeRecords::new(dir.join("probe"), "env".to_owned())
}

#[test]
fn environment_key_changes_when_any_single_snapshot_element_changes() {
    let base = snapshot();
    let changes: [fn(&mut Snapshot); 6] = [
        |s| s.binary_len += 1,
        |s| s.binary_modified_ns += 1,
        |s| s.model_identity.push('2'),
        |s| s.os_build = "25A101".to_owned(),
        |s| s.hardware_model = "Mac16,1".to_owned(),
        |s| {
            s.ggml_vars
                .push(("GGML_METAL_NO_RESIDENCY".into(), "1".into()))
        },
    ];
    for change in changes {
        let mut next = base.clone();
        change(&mut next);
        assert_ne!(base.key(), next.key(), "{next:?}");
    }
    assert_eq!(base.key(), snapshot().key());
}

#[test]
fn environment_key_ignores_non_ggml_variables_and_the_order_of_ggml_variables() {
    let var = |name: &str, value: &str| (OsString::from(name), OsString::from(value));
    let forward = ggml_vars([
        var("PATH", "/bin"),
        var("GGML_A", "1"),
        var("GGML_B", "2"),
        var("XGGML_C", "3"),
    ]);
    let reversed = ggml_vars([var("GGML_B", "2"), var("HOME", "/h"), var("GGML_A", "1")]);
    assert_eq!(forward, reversed);
    assert_eq!(forward, [var("GGML_A", "1"), var("GGML_B", "2")]);
    let key = |ggml_vars| {
        Snapshot {
            ggml_vars,
            ..snapshot()
        }
        .key()
    };
    assert_eq!(key(forward), key(reversed));
}

#[cfg(target_vendor = "apple")]
#[test]
fn host_snapshot_reads_a_non_empty_os_build_and_hardware_model() {
    let host = Snapshot::host("model").unwrap();

    assert!(!host.os_build.trim().is_empty());
    assert!(!host.hardware_model.trim().is_empty());
    assert!(host.binary_len > 0);
}

#[test]
fn probe_record_directory_follows_xdg_cache_home_and_stays_outside_embedding_models() {
    let home = OsStr::new("/home/u");
    let xdg = OsStr::new("/xdg/cache");

    let dir = record_dir(None, Some(xdg), Some(home)).unwrap();

    assert_eq!(dir, Path::new("/xdg/cache/recall/probe"));
    assert!(!dir.starts_with("/xdg/cache/embedding-models"));
    assert_eq!(
        record_dir(Some(OsStr::new("/hf")), Some(xdg), Some(home)).unwrap(),
        Path::new("/hf/recall/probe"),
        "HF_HOME comes first, as for the model cache"
    );
    assert_eq!(
        record_dir(None, Some(OsStr::new("relative")), Some(home)).unwrap(),
        Path::new("/home/u/.cache/recall/probe"),
        "a relative XDG_CACHE_HOME is ignored"
    );
    assert_eq!(record_dir(None, None, None), None);
}

#[test]
fn running_marker_whose_lock_is_free_is_reported_as_a_crash_and_disables_skipping_for_that_environment()
 {
    let dir = tempfile::TempDir::new().unwrap();
    let env = records(dir.path());
    let other = ProbeRecords::new(dir.path().join("probe"), "other".to_owned());
    env.record_success();
    other.record_success();
    assert!(env.may_skip());

    // Dropping the marker without finishing it is what a crash leaves behind.
    drop(env.begin_skipped_load().unwrap());

    assert!(!env.may_skip(), "a crashed skipped load must stop skipping");
    env.record_success();
    assert!(
        !env.may_skip(),
        "a later successful probe must not lift the no-skip state"
    );
    assert!(other.may_skip(), "another environment keeps skipping");
}

#[test]
fn running_marker_held_for_more_than_thirty_seconds_is_reported_as_a_hang() {
    let dir = tempfile::TempDir::new().unwrap();
    let env = records(dir.path());
    env.record_success();
    let marker = env.begin_skipped_load().unwrap();

    assert!(
        env.may_skip(),
        "a concurrent search may skip while another skipped load is running"
    );
    File::options()
        .write(true)
        .open(&marker.path)
        .unwrap()
        .set_modified(SystemTime::now() - Duration::from_secs(31))
        .unwrap();
    assert!(!env.may_skip(), "a skipped load held past 30s is a hang");
    marker.finish();
    assert!(!env.may_skip());
}

#[test]
fn unreadable_or_unwritable_record_is_treated_as_absent_without_an_error() {
    let dir = tempfile::TempDir::new().unwrap();
    let unreadable = records(dir.path());
    fs::create_dir_all(unreadable.path("ok")).unwrap();
    assert!(
        !unreadable.may_skip(),
        "a record that cannot be read is absent"
    );
    unreadable.record_success();
    assert_eq!(
        fs::read_dir(&unreadable.dir).unwrap().count(),
        1,
        "a failed record write leaves no temporary file"
    );

    let blocked = dir.path().join("file");
    fs::write(&blocked, b"not a directory").unwrap();
    let unwritable = ProbeRecords::new(blocked.join("probe"), "env".to_owned());
    unwritable.record_success();
    unwritable.clear();
    assert!(!unwritable.may_skip());
    assert!(unwritable.begin_skipped_load().is_err());

    let mismatched = records(dir.path().join("other").as_path());
    fs::create_dir_all(&mismatched.dir).unwrap();
    fs::write(mismatched.path("ok"), b"different-key").unwrap();
    assert!(!mismatched.may_skip(), "a record for another key is absent");
}

#[test]
fn marker_unlinked_by_its_finished_owner_after_being_opened_is_not_a_crash() {
    let dir = tempfile::TempDir::new().unwrap();
    let env = records(dir.path());
    let marker = env.begin_skipped_load().unwrap();
    let opened = File::open(&marker.path).unwrap();
    let path = marker.path.clone();

    marker.finish();

    assert!(matches!(
        marker_state(&path, &opened).unwrap(),
        MarkerState::Finished
    ));
}

#[cfg(unix)]
#[test]
fn marker_that_cannot_be_opened_blocks_the_skip_without_disabling_skipping() {
    use std::os::unix::fs::PermissionsExt;
    let dir = tempfile::TempDir::new().unwrap();
    let env = records(dir.path());
    env.record_success();
    let marker = env.path("1.0.running");
    fs::write(&marker, b"").unwrap();
    fs::set_permissions(&marker, fs::Permissions::from_mode(0o000)).unwrap();

    assert!(!env.may_skip());
    assert!(
        !env.no_skip(),
        "an uninspectable marker is not a confirmed crash"
    );

    fs::set_permissions(&marker, fs::Permissions::from_mode(0o600)).unwrap();
    assert!(
        !env.may_skip(),
        "once readable, the unlocked marker is a crash"
    );
    assert!(env.no_skip());
}
