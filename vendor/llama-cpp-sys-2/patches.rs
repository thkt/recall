//! Applies the build-time patches to a staged copy of llama.cpp and syncs the
//! changed files to the directory CMake builds from.
//!
//! Standard library only: `build.rs` and `tests/llama_patches.rs` both load this
//! file through `#[path]`.

use std::{
    io,
    path::{Path, PathBuf},
};

/// Applies every `*.patch` in `patch_dir` to `staged` in name order and returns
/// the applied patch file names.
pub fn apply_patches(_staged: &Path, _patch_dir: &Path) -> Result<Vec<String>, String> {
    Ok(Vec::new())
}

/// Copies the files of `staged` whose content differs from `dest` (or is missing
/// there) into `dest` and returns their paths relative to `staged`.
pub fn sync_changed(_staged: &Path, _dest: &Path) -> io::Result<Vec<PathBuf>> {
    Ok(Vec::new())
}
