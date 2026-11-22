use super::probe_record::{host_state_dir, write_atomically};
use super::{DegradedReason, ModelSpec, digest_hex};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs,
    io::{self, ErrorKind},
    path::{Path, PathBuf},
};

const ARTIFACT_NAMES: [&str; 2] = ["model.gguf", "tokenizer.json"];

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ArtifactCheck {
    TrustRecord,
    HashAndRecord,
    HashOnly,
}

pub(crate) struct ResolvedArtifacts {
    pub(crate) root: PathBuf,
    pub(crate) trusted_record: Option<ArtifactRecord>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct FileIdentity {
    dev: u64,
    ino: u64,
    size: u64,
    mtime_sec: i64,
    mtime_nsec: i64,
    ctime_sec: i64,
    ctime_nsec: i64,
}

#[derive(Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RecordContents {
    artifacts: [FileIdentity; 2],
}

fn file_identity(path: &Path) -> io::Result<FileIdentity> {
    use std::os::unix::fs::MetadataExt;
    let metadata = fs::metadata(path)?;
    Ok(FileIdentity {
        dev: metadata.dev(),
        ino: metadata.ino(),
        size: metadata.size(),
        mtime_sec: metadata.mtime(),
        mtime_nsec: metadata.mtime_nsec(),
        ctime_sec: metadata.ctime(),
        ctime_nsec: metadata.ctime_nsec(),
    })
}

fn current_identities(root: &Path) -> Option<[FileIdentity; 2]> {
    let read = |name: &str| file_identity(&root.join(name));
    match read(ARTIFACT_NAMES[0]).and_then(|model| Ok([model, read(ARTIFACT_NAMES[1])?])) {
        Ok(identities) => Some(identities),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => {
            tracing::warn!(%error, "could not read the model artifact file identities; the artifacts are hashed");
            None
        }
    }
}

pub(crate) struct ArtifactRecords {
    dir: PathBuf,
}

impl ArtifactRecords {
    pub(crate) fn new(dir: PathBuf) -> Self {
        Self { dir }
    }

    pub(super) fn host() -> Option<Self> {
        host_state_dir("verified-artifacts").map(Self::new)
    }

    fn record(&self, root: &Path, spec: &ModelSpec) -> ArtifactRecord {
        let mut hash = Sha256::new();
        let mut field = |bytes: &[u8]| {
            hash.update(bytes.len().to_le_bytes());
            hash.update(bytes);
        };
        field(b"recall-artifact-record-v1");
        field(root.as_os_str().as_encoded_bytes());
        field(spec.model_sha256.as_bytes());
        field(spec.tokenizer_sha256.as_bytes());
        ArtifactRecord {
            path: self
                .dir
                .join(format!("{}.json", digest_hex(&hash.finalize()))),
        }
    }
}

pub(crate) struct ArtifactRecord {
    path: PathBuf,
}

impl ArtifactRecord {
    fn vouches_for(&self, identities: &[FileIdentity; 2]) -> bool {
        let text = match fs::read_to_string(&self.path) {
            Ok(text) => text,
            Err(error) if error.kind() == ErrorKind::NotFound => return false,
            Err(error) => {
                tracing::warn!(%error, "could not read the model artifact record; the artifacts are hashed");
                return false;
            }
        };
        serde_json::from_str::<RecordContents>(&text).is_ok_and(|contents| {
            contents
                == RecordContents {
                    artifacts: *identities,
                }
        })
    }

    fn write(&self, identities: [FileIdentity; 2]) {
        let contents = RecordContents {
            artifacts: identities,
        };
        let written = serde_json::to_string(&contents)
            .map_err(io::Error::other)
            .and_then(|json| write_atomically(&self.path, &json));
        if let Err(error) = written {
            tracing::warn!(%error, "could not record the verified model artifacts");
        }
    }

    pub(crate) fn clear(&self) {
        match fs::remove_file(&self.path) {
            Err(error) if error.kind() != ErrorKind::NotFound => {
                tracing::warn!(%error, "could not clear the model artifact record");
            }
            _ => {}
        }
    }
}

pub(crate) fn resolve_artifacts(
    check: ArtifactCheck,
    root: &Path,
    spec: &ModelSpec,
    records: Option<&ArtifactRecords>,
    hash: impl FnOnce() -> anyhow::Result<Option<PathBuf>>,
) -> Result<ResolvedArtifacts, DegradedReason> {
    let record = records.map(|records| records.record(root, spec));
    let before = record
        .as_ref()
        .filter(|_| check != ArtifactCheck::HashOnly)
        .and_then(|_| current_identities(root));
    if check == ArtifactCheck::TrustRecord
        && let (Some(trusted), Some(identities)) = (&record, &before)
        && trusted.vouches_for(identities)
    {
        tracing::debug!("artifact sha256 skipped: the files are unchanged since they passed");
        return Ok(ResolvedArtifacts {
            root: root.to_path_buf(),
            trusted_record: record,
        });
    }
    match hash() {
        Ok(None) => Err(DegradedReason::NotInstalled),
        Err(error) => {
            if check != ArtifactCheck::HashOnly {
                tracing::warn!(%error, "model artifact validation failed");
            }
            if let Some(record) = &record {
                record.clear();
            }
            Err(DegradedReason::ProbeFailed)
        }
        Ok(Some(verified)) => {
            if let (Some(record), Some(before)) = (&record, before)
                && current_identities(root) == Some(before)
            {
                record.write(before);
            }
            Ok(ResolvedArtifacts {
                root: verified,
                trusted_record: None,
            })
        }
    }
}

#[cfg(test)]
#[path = "artifact_record_tests.rs"]
mod tests;
