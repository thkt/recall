//! Shared root-bypass probe, factored out of `src/db/tests.rs`'s inline
//! `root_bypasses_permission_bits` (and a second inline copy in the
//! `stale_wal_note` read-only-dir test) so both sites detect a
//! permission-bypassing effective user (e.g. root) through one contract
//! instead of two independently-drifting copies.
//!
//! This file lives under `tests/support/`, a subdirectory, so Cargo does not
//! compile it as its own integration-test binary (only `.rs` files directly
//! under `tests/` get that treatment). It is instead pulled into the unit
//! test module tree via `#[path]` from `src/db/tests.rs`.
//!
//! U-001 does not add marker output (e.g. `println!`) for the root-detected
//! case: nextest captures stdout/stderr by default, so a marker would not
//! surface anywhere a human or CI would see it. Making a root test run
//! visible is a separate concern left to a canary test, not this helper.
//!
//! Read-only-dir test convention: a test that clears a dir's write bits routes
//! through `skip_if_root` and restores the write bits (chmod 0o755) before any
//! assert, so a panicking assert cannot leave the read-only TempDir undroppable.

use std::path::Path;

/// True when the effective user bypasses `dir`'s permission bits (e.g.
/// running as root), detected empirically by attempting to create a file
/// under `dir`.
///
/// # Precondition
/// The caller must already have cleared `dir`'s write bits (e.g.
/// `chmod 0o555`) before calling this. Calling it on a directory that is
/// still writable by the current user gives a **false positive**: the probe
/// creation succeeds for an unrelated reason (the dir simply is writable),
/// not because permission bits are being bypassed, and this function cannot
/// tell the two apart.
///
/// # Return value
/// - Probe creation succeeds (bypass detected, or the precondition above was
///   violated): the probe file is removed, `dir`'s permissions are restored
///   to `0o755`, and this returns `true` so the caller can skip the
///   assertions that assume enforced permission bits.
/// - Probe creation fails: nothing under `dir` is touched, and this returns
///   `false`.
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

/// Root canary shared by the unit-test (`src/db/tests.rs`) and integration-test
/// (`tests/cli_integration.rs`) suites, which compile separately and so cannot
/// share a `#[test]` fn directly. Turns a root-run suite loud (panics) instead
/// of letting the permission-bit tests silently skip, and on a non-root run
/// restores the read-only TempDir's write bits before Drop.
#[cfg(unix)]
pub fn assert_root_canary() {
    use std::fs;
    use std::os::unix::fs::PermissionsExt;

    let dir = tempfile::TempDir::new().unwrap();
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o555)).unwrap();

    if skip_if_root(dir.path()) {
        // skip_if_root already restored dir's write bits to 0o755 before
        // reporting root, so TempDir's Drop can still clean up after the panic
        // unwinds.
        panic!(
            "running as root: permission-bits tests cannot reproduce read-only dirs and are \
             skipped; run the suite as non-root"
        );
    }

    // Non-root: skip_if_root left dir's write bits untouched (still 0o555).
    // Restore them so TempDir's Drop can clean up.
    fs::set_permissions(dir.path(), fs::Permissions::from_mode(0o755)).unwrap();
}

/// Asserts the raw `.root_probe` probe-file literal is not inlined anywhere in
/// `source` (each caller passes `include_str!` of its own file), so the literal
/// lives only in `skip_if_root`'s definition above and cannot silently
/// re-appear as a hand-rolled inline copy in a test file. `file_label` names the
/// scanned file for the failure message. Shared by both suites for the same
/// reason as `assert_root_canary`.
pub fn assert_probe_literal_not_inlined(source: &str, file_label: &str) {
    // Built via concat! so this helper's own source text does not self-match if
    // a caller ever include_str!s the file this function lives in.
    let probe_literal = concat!(".", "root_probe");
    assert_eq!(
        source.matches(probe_literal).count(),
        0,
        "{probe_literal:?} must appear only in root_skip::skip_if_root's definition, \
         not duplicated inline in {file_label}"
    );
}

/// Asserts `note` carries the read-only-dir "copy the DB elsewhere" remedy
/// contract in one shared place, so the three tests that exercise it (the
/// read-only-dir immutable-tier test and the macOS read-only-mount test in
/// `src/db/tests.rs`, and the CLI `--json` degraded-note test in
/// `tests/cli_integration.rs`) hold the same wording contract instead of
/// drifting inline trio asserts. `context` labels the calling site in failure
/// messages.
///
/// The assertions, pinned against `src/db.rs`'s `stale_wal_note`
/// read-only branch (production wording): the write-ahead-log wording survives;
/// the copy-based `recall index --db-path <copy>` remedy is present; the
/// `-wal` and `-shm` sidecars are each named; and no bare `recall index` run
/// form (which a read-only dir cannot execute) leaks through.
#[track_caller]
pub fn assert_copy_based_remedy(note: &str, context: &str) {
    assert!(
        note.contains("write-ahead log"),
        "[{context}] note must carry the write-ahead-log wording, got: {note}"
    );
    assert!(
        note.contains("recall index --db-path <copy>"),
        "[{context}] note must guide a copy-based `recall index --db-path` remedy, got: {note}"
    );
    assert!(
        note.contains("-wal"),
        "[{context}] note must name the -wal sidecar file to copy, got: {note}"
    );
    assert!(
        note.contains("-shm"),
        "[{context}] note must name the -shm sidecar file to copy, got: {note}"
    );
    assert!(
        !note.contains("run `recall index`"),
        "[{context}] a read-only location cannot run a bare `recall index`; got: {note}"
    );
}
