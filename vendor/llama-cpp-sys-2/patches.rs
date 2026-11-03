//! Applies the build-time patches to a staged copy of llama.cpp and syncs the
//! changed files to the directory CMake builds from.
//!
//! Standard library only: `build.rs` and `tests/llama_patches.rs` both load this
//! file through `#[path]`.

use std::{
    fs, io,
    path::{Component, Path, PathBuf},
    process::Command,
};

/// Git variables that would point `git apply` at a repository other than the one
/// it is meant to ignore.
const GIT_LOCATION_VARS: [&str; 4] = [
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_COMMON_DIR",
];

/// Patches may change files below this directory only. bindgen and cc keep
/// reading `include/`, `ggml/include` and `common/` from the original tree.
const PATCHABLE_DIR: &str = "src";

/// Applies every `*.patch` in `patch_dir` to `staged` in name order and returns
/// the applied patch file names.
pub fn apply_patches(staged: &Path, patch_dir: &Path) -> Result<Vec<String>, String> {
    let staged = staged
        .canonicalize()
        .map_err(|e| format!("cannot resolve staging directory {}: {e}", staged.display()))?;
    let mut applied = Vec::new();
    for patch in patch_files(patch_dir)? {
        let name = patch
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        let text = fs::read(&patch)
            .map_err(|e| format!("{name}: cannot read {}: {e}", patch.display()))?;
        check_paths_below_src(&name, &String::from_utf8_lossy(&text))?;
        git_apply(&name, &staged, &patch, false)?;
        // `git apply` exits 0 for a patch it silently skipped, so confirm that
        // the patch is now present by reversing it as a dry run.
        git_apply(&name, &staged, &patch, true)
            .map_err(|e| format!("{name}: patch was not applied to the staged copy: {e}"))?;
        applied.push(name);
    }
    Ok(applied)
}

/// Copies the files of `staged` whose content differs from `dest` (or is missing
/// there) into `dest` and returns their paths relative to `staged`.
pub fn sync_changed(staged: &Path, dest: &Path) -> io::Result<Vec<PathBuf>> {
    let mut rewritten = Vec::new();
    sync_dir(staged, dest, Path::new(""), &mut rewritten)?;
    Ok(rewritten)
}

fn patch_files(patch_dir: &Path) -> Result<Vec<PathBuf>, String> {
    let entries = fs::read_dir(patch_dir)
        .map_err(|e| format!("cannot read patch directory {}: {e}", patch_dir.display()))?;
    let mut files = Vec::new();
    for entry in entries {
        let path = entry
            .map_err(|e| format!("cannot list patch directory {}: {e}", patch_dir.display()))?
            .path();
        if path.extension().is_some_and(|ext| ext == "patch") && path.is_file() {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

fn check_paths_below_src(name: &str, patch: &str) -> Result<(), String> {
    for line in patch.lines() {
        for path in header_paths(line) {
            if !is_below_src(path) {
                return Err(format!(
                    "{name}: changes `{path}`, but patches may only change files below {PATCHABLE_DIR}/"
                ));
            }
        }
    }
    Ok(())
}

fn strip_side_prefix(path: &str) -> &str {
    let path = path.split('\t').next().unwrap_or(path);
    path.strip_prefix("a/")
        .or_else(|| path.strip_prefix("b/"))
        .unwrap_or(path)
}

/// The file paths a patch header line names, without the `a/` or `b/` prefix.
fn header_paths(line: &str) -> Vec<&str> {
    if let Some(rest) = line.strip_prefix("diff --git ") {
        return match rest.split_once(" b/") {
            Some((old, new)) => vec![strip_side_prefix(old), new],
            None => vec![rest],
        };
    }
    for prefix in ["--- ", "+++ "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return if rest.starts_with("/dev/null") {
                vec![]
            } else {
                vec![strip_side_prefix(rest)]
            };
        }
    }
    for prefix in ["rename from ", "rename to ", "copy from ", "copy to "] {
        if let Some(rest) = line.strip_prefix(prefix) {
            return vec![rest];
        }
    }
    vec![]
}

fn is_below_src(path: &str) -> bool {
    let path = Path::new(path);
    path.starts_with(PATCHABLE_DIR) && path.components().all(|c| matches!(c, Component::Normal(_)))
}

/// Runs `git apply` (or, with `reverse_check`, a dry run of its reversal) inside
/// `staged`. `GIT_CEILING_DIRECTORIES` stops git from finding a repository above
/// `staged`: inside one, `git apply` prints `Skipped patch` and exits 0.
fn git_apply(name: &str, staged: &Path, patch: &Path, reverse_check: bool) -> Result<(), String> {
    let mut command = Command::new("git");
    command.arg("apply").arg("--whitespace=nowarn");
    if reverse_check {
        command.args(["--reverse", "--check"]);
    }
    command.arg(patch).current_dir(staged);
    for var in GIT_LOCATION_VARS {
        command.env_remove(var);
    }
    if let Some(parent) = staged.parent() {
        command.env("GIT_CEILING_DIRECTORIES", parent);
    }
    let output = command
        .output()
        .map_err(|e| format!("{name}: cannot run git apply: {e}"))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        return Err(format!("{name}: git apply failed: {}", stderr.trim()));
    }
    if stderr.contains("Skipped patch") {
        return Err(format!(
            "{name}: git apply skipped the patch: {}",
            stderr.trim()
        ));
    }
    Ok(())
}

fn sync_dir(
    staged: &Path,
    dest: &Path,
    relative: &Path,
    rewritten: &mut Vec<PathBuf>,
) -> io::Result<()> {
    for entry in fs::read_dir(staged.join(relative))? {
        let entry = entry?;
        let relative = relative.join(entry.file_name());
        let from = staged.join(&relative);
        if entry.file_type()?.is_dir() {
            sync_dir(staged, dest, &relative, rewritten)?;
            continue;
        }
        // A symlink whose target is missing cannot be copied.
        if fs::metadata(&from).is_err() {
            continue;
        }
        let to = dest.join(&relative);
        if same_content(&from, &to)? {
            continue;
        }
        if let Some(parent) = to.parent() {
            fs::create_dir_all(parent)?;
        }
        fs::copy(&from, &to)?;
        rewritten.push(relative);
    }
    Ok(())
}

fn same_content(a: &Path, b: &Path) -> io::Result<bool> {
    let Ok(b_meta) = fs::metadata(b) else {
        return Ok(false);
    };
    if fs::metadata(a)?.len() != b_meta.len() {
        return Ok(false);
    }
    Ok(fs::read(a)? == fs::read(b)?)
}
