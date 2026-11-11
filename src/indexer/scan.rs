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

/// Values are taken as given; `classify_roots` refuses a root that is not an absolute path.
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

/// `RECALL_CODEX_DIR` alone turns the default archive off, so pointing recall at
/// another sessions tree never mixes in the real `~/.codex/archived_sessions`.
/// An empty `RECALL_CODEX_ARCHIVED_DIR` turns the archive off on its own; any other
/// value that is not an absolute path is refused by `classify_roots`.
pub(crate) fn resolve_codex_archived_dir_with<F>(home: &Path, get: F) -> Option<PathBuf>
where
    F: Fn(&str) -> Option<OsString>,
{
    if let Some(dir) = get("RECALL_CODEX_ARCHIVED_DIR") {
        return (!dir.is_empty()).then(|| PathBuf::from(dir));
    }
    get("RECALL_CODEX_DIR")
        .is_none()
        .then(|| home.join(".codex").join("archived_sessions"))
}

pub(crate) fn resolve_codex_archived_dir(home: &Path) -> Option<PathBuf> {
    resolve_codex_archived_dir_with(home, |key| env::var_os(key))
}

pub(super) struct ScanOutcome {
    pub(super) sources: Vec<(PathBuf, Source)>,
    /// Compressed logs left unparsed because their plain copy sits beside them.
    /// They are still on disk, so a row pointing at one must not be deleted.
    pub(super) set_aside: Vec<PathBuf>,
    pub(super) removal: RemovalGate,
    pub(super) skipped: HashMap<Source, SkippedReason>,
}

/// Decides which rows a run may delete because their file was not found (DR 0004).
#[derive(Default)]
pub(super) struct RemovalGate {
    fully_read: HashSet<Source>,
    missing_roots: Vec<(Source, PathBuf)>,
}

impl RemovalGate {
    #[cfg(test)]
    pub(super) fn fully_read(sources: impl IntoIterator<Item = Source>) -> Self {
        Self {
            fully_read: sources.into_iter().collect(),
            missing_roots: Vec::new(),
        }
    }

    pub(super) fn is_fully_read(&self, source: Source) -> bool {
        self.fully_read.contains(&source)
    }

    /// A row may go only when every present root of its source was read in full
    /// and the row does not lie under a root that is missing this run.
    pub(super) fn allows(&self, source: Option<Source>, path: &Path) -> bool {
        source.is_some_and(|source| {
            self.is_fully_read(source)
                && !self
                    .missing_roots
                    .iter()
                    .any(|(owner, root)| *owner == source && path.starts_with(root))
        })
    }
}

pub(super) fn collect_sources(opts: &IndexOptions) -> ScanOutcome {
    let mut out = ScanOutcome {
        sources: Vec::new(),
        set_aside: Vec::new(),
        removal: RemovalGate::default(),
        skipped: HashMap::new(),
    };
    classify_roots(Source::Claude, &[opts.claude_dir], &mut out);
    let codex_roots: Vec<&Path> = [opts.codex_dir]
        .into_iter()
        .chain(opts.codex_archived_dir)
        .collect();
    classify_roots(Source::Codex, &codex_roots, &mut out);
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

/// One unreadable root blocks every deletion for its source: a session that just moved
/// into that root would otherwise be deleted from the root it left. A root counts as
/// missing when it does not exist or is not a directory; `is_dir()` would also call it
/// missing when `stat` fails on a permission error. A root that is not an absolute path,
/// such as one from an empty environment variable, is not read and counts as unreadable:
/// it would resolve against the working directory and change from run to run.
fn classify_roots(source: Source, roots: &[&Path], out: &mut ScanOutcome) {
    let mut any_present = false;
    let mut all_read = true;
    for root in roots {
        if !root.is_absolute() {
            warn!(
                path = %root.display(),
                "session root is not an absolute path; not reading it and keeping every {source} session",
                source = source.as_str(),
            );
            all_read = false;
            continue;
        }
        match fs::metadata(root) {
            Ok(meta) if meta.is_dir() => {
                any_present = true;
                all_read &= collect_jsonl_files(root, &mut out.sources, source);
            }
            Err(e)
                if !matches!(
                    e.kind(),
                    io::ErrorKind::NotFound | io::ErrorKind::NotADirectory
                ) =>
            {
                warn!(path = %root.display(), error = %e, "cannot inspect root");
                all_read = false;
            }
            _ => out.removal.missing_roots.push((source, root.to_path_buf())),
        }
    }
    if !all_read {
        out.skipped
            .insert(source, SkippedReason::IncompleteEnumeration);
    } else if any_present {
        out.removal.fully_read.insert(source);
        if out.removal.missing_roots.iter().any(|(s, _)| *s == source) {
            out.skipped.insert(source, SkippedReason::MissingRoot);
        }
    } else {
        out.skipped.insert(source, SkippedReason::MissingRoot);
    }
}

const MAX_DIR_DEPTH: usize = 10;

fn collect_jsonl_files(dir: &Path, out: &mut Vec<(PathBuf, Source)>, source: Source) -> bool {
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
