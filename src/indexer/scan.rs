use std::collections::{HashMap, HashSet};
use std::env;
use std::ffi::OsString;
use std::fs::{self, DirEntry};
use std::io;
use std::mem;
use std::path::{Path, PathBuf};

use tracing::{debug, warn};

use super::{IndexOptions, SkippedReason};
use crate::parser::{Source, is_compressed_log, is_session_log};

pub(crate) fn resolve_claude_dir_with<F>(home: &Path, get: F) -> PathBuf
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(dir) = get("RECALL_CLAUDE_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(dir) = get("CLAUDE_CONFIG_DIR") {
        return PathBuf::from(dir).join("projects");
    }
    home.join(".claude").join("projects")
}

pub(crate) fn resolve_claude_dir(home: &Path) -> PathBuf {
    resolve_claude_dir_with(home, |key| env::var_os(key))
}

pub(crate) fn resolve_codex_dir_with<F>(home: &Path, get: F) -> PathBuf
where
    F: Fn(&str) -> Option<OsString>,
{
    get("RECALL_CODEX_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| home.join(".codex").join("sessions"))
}

pub(crate) fn resolve_codex_dir(home: &Path) -> PathBuf {
    resolve_codex_dir_with(home, |key| env::var_os(key))
}

pub(super) struct ScanOutcome {
    pub(super) sources: Vec<(PathBuf, Source)>,
    /// Compressed logs left unparsed because their plain copy sits beside them.
    /// They are still on disk, so a row pointing at one must not be deleted.
    pub(super) set_aside: Vec<PathBuf>,
    pub(super) scanned: HashSet<Source>,
    pub(super) skipped: HashMap<Source, SkippedReason>,
}

pub(super) fn collect_sources(opts: &IndexOptions) -> ScanOutcome {
    let mut out = ScanOutcome {
        sources: Vec::new(),
        set_aside: Vec::new(),
        scanned: HashSet::new(),
        skipped: HashMap::new(),
    };
    classify_root(opts.claude_dir, Source::Claude, &mut out);
    classify_root(opts.codex_dir, Source::Codex, &mut out);
    out.set_aside = set_aside_compressed_copies_of_plain_logs(&mut out.sources);
    out
}

/// While Codex compresses a log, the plain file and its `.zst` copy coexist.
/// Indexing both would make each run replace the session row with the other path.
fn set_aside_compressed_copies_of_plain_logs(sources: &mut Vec<(PathBuf, Source)>) -> Vec<PathBuf> {
    let plain: HashSet<PathBuf> = sources
        .iter()
        .filter(|(path, _)| !is_compressed_log(path))
        .map(|(path, _)| path.clone())
        .collect();
    let (kept, set_aside): (Vec<_>, Vec<_>) =
        mem::take(sources).into_iter().partition(|(path, _)| {
            !is_compressed_log(path) || !plain.contains(&path.with_extension(""))
        });
    *sources = kept;
    set_aside.into_iter().map(|(path, _)| path).collect()
}

fn classify_root(dir: &Path, source: Source, out: &mut ScanOutcome) {
    if !dir.is_dir() {
        out.skipped.insert(source, SkippedReason::MissingRoot);
    } else if collect_jsonl_files(dir, &mut out.sources, source) {
        out.scanned.insert(source);
    } else {
        out.skipped
            .insert(source, SkippedReason::IncompleteEnumeration);
    }
}

pub(super) const MAX_DIR_DEPTH: usize = 10;

pub(super) fn collect_jsonl_files(
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
) -> bool {
    collect_jsonl_files_inner(dir, out, source, 0)
}

fn collect_jsonl_files_inner(
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    depth: usize,
) -> bool {
    if depth >= MAX_DIR_DEPTH {
        warn!(
            max_depth = MAX_DIR_DEPTH,
            path = %dir.display(),
            "depth limit reached"
        );
        return false;
    }
    let entries = match fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            warn!(path = %dir.display(), error = %e, "cannot read dir");
            return false;
        }
    };
    collect_from_entries(entries, dir, out, source, depth)
}

pub(super) fn collect_from_entries(
    entries: impl Iterator<Item = io::Result<DirEntry>>,
    dir: &Path,
    out: &mut Vec<(PathBuf, Source)>,
    source: Source,
    depth: usize,
) -> bool {
    let mut fully_read = true;
    for entry in entries {
        let (entry, ft) = match entry.and_then(|e| e.file_type().map(|ft| (e, ft))) {
            Ok(pair) => pair,
            Err(e) => {
                warn!(path = %dir.display(), error = %e, "cannot read entry");
                fully_read = false;
                continue;
            }
        };
        if ft.is_symlink() {
            debug!(path = %entry.path().display(), "skipping symlink");
            continue;
        }
        let path = entry.path();
        if ft.is_dir() {
            fully_read &= collect_jsonl_files_inner(&path, out, source, depth + 1);
        } else if is_session_log(&path, source) {
            out.push((path, source));
        }
    }
    fully_read
}
