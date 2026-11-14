use std::path::Path;

#[cfg(unix)]
pub fn skip_if_root(dir: &Path) -> bool {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let is_root = fs::File::create(dir.join(".root_probe")).is_ok();
    if is_root {
        let _ = fs::remove_file(dir.join(".root_probe"));
        fs::set_permissions(dir, fs::Permissions::from_mode(0o755)).unwrap();
    }
    is_root
}

#[cfg(unix)]
pub fn assert_root_canary() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if skip_if_root(dir.path()) {
        panic!(
            "running as root: permission-bits tests cannot reproduce read-only dirs and are \
             skipped; run the suite as non-root"
        );
    }

    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
}

pub fn assert_probe_literal_not_inlined(source: &str, file_label: &str) {
    let probe_literal = concat!(".", "root_probe");
    assert_eq!(
        source.matches(probe_literal).count(),
        0,
        "{probe_literal:?} must appear only in root_skip::skip_if_root's definition, \
         not duplicated inline in {file_label}"
    );
}
