//! Keep this cache protocol identical in sae and recall. Index identities remain app-specific.
use super::{ModelSpec, digest_hex, hash_file, install_artifact};
use anyhow::{Result, ensure};
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::ErrorKind,
    path::{Path, PathBuf},
    process,
    time::Duration,
};

fn override_dir(app: &str) -> Option<PathBuf> {
    env::var_os(format!("{}_MODEL_DIR", app.to_ascii_uppercase())).map(PathBuf::from)
}

fn home_cache() -> PathBuf {
    env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_default()
        .join(".cache")
}

fn cache_bases() -> Vec<PathBuf> {
    let mut bases = Vec::new();
    if let Some(hf) = env::var_os("HF_HOME") {
        bases.push(PathBuf::from(hf));
    }
    if let Some(xdg) = env::var_os("XDG_CACHE_HOME").map(PathBuf::from)
        && xdg.is_absolute()
    {
        bases.push(xdg);
    }
    bases.push(home_cache());
    bases
}

pub(super) fn artifact_identity(spec: &ModelSpec) -> String {
    // Only artifact bytes determine the shared key, never prefixes or app/backend identity.
    digest_hex(&Sha256::digest(format!(
        "{}\n{}\n",
        spec.model_sha256, spec.tokenizer_sha256
    )))
}

pub(super) fn cache_dir(app: &str, spec: &ModelSpec) -> PathBuf {
    override_dir(app).unwrap_or_else(|| {
        cache_bases()[0]
            .join("embedding-models")
            .join(artifact_identity(spec))
    })
}

fn legacy_dirs() -> Vec<PathBuf> {
    let mut dirs = Vec::new();
    for base in cache_bases() {
        // Also reuse unchanged artifacts after a tokenizer/model update or cache-base change.
        if let Ok(entries) = fs::read_dir(base.join("embedding-models")) {
            dirs.extend(entries.flatten().map(|entry| entry.path()));
        }
        for app in ["sae", "recall"] {
            if let Ok(entries) = fs::read_dir(base.join(app).join("models")) {
                dirs.extend(entries.flatten().map(|entry| entry.path()));
            }
        }
    }
    dirs.sort();
    dirs.dedup();
    dirs
}

fn artifacts(spec: &ModelSpec) -> [(&str, &str, &str); 2] {
    [
        ("model.gguf", &spec.model_url, &spec.model_sha256),
        (
            "tokenizer.json",
            &spec.tokenizer_url,
            &spec.tokenizer_sha256,
        ),
    ]
}

fn validate_at(root: &Path, spec: &ModelSpec) -> Result<Option<PathBuf>> {
    if artifacts(spec)
        .iter()
        .any(|(name, _, _)| !root.join(name).is_file())
    {
        return Ok(None);
    }
    for (name, _, expected) in artifacts(spec) {
        ensure!(
            hash_file(&root.join(name))? == expected,
            "{name} checksum mismatch"
        );
    }
    Ok(Some(root.to_path_buf()))
}

fn lock_cache(root: &Path) -> Result<fs::File> {
    fs::create_dir_all(root)?;
    let mut options = fs::OpenOptions::new();
    options.read(true).write(true).create(true).truncate(false);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    let lock = options.open(root.join(".install.lock"))?;
    // The OS releases this on exit/crash. Never unlink it: waiters must share the same inode.
    lock.lock()?;
    Ok(lock)
}

fn reuse_artifact(root: &Path, name: &str, expected: &str, legacy: &[PathBuf]) -> Result<bool> {
    for dir in legacy {
        let source = dir.join(name);
        if !source.is_file() || hash_file(&source).ok().as_deref() != Some(expected) {
            continue;
        }
        let temporary = root.join(format!(".{name}.{}.reuse", process::id()));
        match fs::hard_link(&source, &temporary) {
            Ok(()) => {
                let result = (|| -> Result<()> {
                    ensure!(
                        hash_file(&temporary)? == expected,
                        "legacy checksum mismatch"
                    );
                    fs::rename(&temporary, root.join(name))?;
                    Ok(())
                })();
                if result.is_err() {
                    let _ = fs::remove_file(&temporary);
                }
                result?;
            }
            Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                return Err(error.into());
            }
            // A cache may live on a different filesystem, where hard links are unavailable.
            Err(_) => install_artifact(root, name, expected, &mut fs::File::open(source)?)?,
        }
        return Ok(true);
    }
    Ok(false)
}

pub(super) fn cached_artifacts(
    app: &str,
    spec: &ModelSpec,
    root: &Path,
) -> Result<Option<PathBuf>> {
    if override_dir(app).is_some()
        || artifacts(spec)
            .iter()
            .all(|(name, _, _)| root.join(name).is_file())
    {
        return validate_at(root, spec);
    }
    let legacy = legacy_dirs();
    if !legacy.is_empty() {
        let _lock = lock_cache(root)?;
        for (name, _, expected) in artifacts(spec) {
            if !root.join(name).is_file() {
                reuse_artifact(root, name, expected, &legacy)?;
            }
        }
    }
    validate_at(root, spec)
}

pub(super) fn download_artifacts(app: &str, spec: &ModelSpec, root: &Path) -> Result<()> {
    let legacy = if override_dir(app).is_some() {
        Vec::new()
    } else {
        legacy_dirs()
    };
    download_artifacts_at(root, spec, &legacy)
}

pub(super) fn download_artifacts_at(
    root: &Path,
    spec: &ModelSpec,
    legacy: &[PathBuf],
) -> Result<()> {
    let _lock = lock_cache(root)?;
    let mut client: Option<Client> = None;
    for (name, url, expected) in artifacts(spec) {
        let destination = root.join(name);
        if destination.is_file() && hash_file(&destination)? == expected {
            continue;
        }
        if reuse_artifact(root, name, expected, legacy)? {
            continue;
        }
        let client = match &client {
            Some(client) => client,
            None => client.insert(
                Client::builder()
                    .timeout(Duration::from_secs(600))
                    .build()?,
            ),
        };
        install_artifact(
            root,
            name,
            expected,
            &mut client.get(url).send()?.error_for_status()?,
        )?;
    }
    Ok(())
}

#[cfg(test)]
#[path = "cache_tests.rs"]
mod tests;
