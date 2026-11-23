use super::digest_hex;
use sha2::{Digest, Sha256};
use std::{
    env,
    ffi::{CStr, OsStr, OsString},
    fs::{self, File, OpenOptions, TryLockError},
    io::{self, ErrorKind, Write},
    path::{Path, PathBuf},
    process, ptr,
    sync::atomic::{AtomicU64, Ordering},
    time::{Duration, SystemTime, UNIX_EPOCH},
};

const HANG_AFTER_THE_PROBE_CHILD_TIMEOUT: Duration = Duration::from_secs(30);

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) struct Snapshot {
    pub(super) binary_len: u64,
    pub(super) binary_modified_ns: u128,
    pub(super) model_identity: String,
    pub(super) os_build: String,
    pub(super) hardware_model: String,
    pub(super) ggml_vars: Vec<(OsString, OsString)>,
}

impl Snapshot {
    fn host(model_identity: &str) -> io::Result<Self> {
        let binary = fs::metadata(env::current_exe()?)?;
        let modified = binary
            .modified()?
            .duration_since(UNIX_EPOCH)
            .map_err(io::Error::other)?;
        Ok(Self {
            binary_len: binary.len(),
            binary_modified_ns: modified.as_nanos(),
            model_identity: model_identity.to_owned(),
            os_build: sysctl_string(c"kern.osversion")?,
            hardware_model: sysctl_string(c"hw.model")?,
            ggml_vars: ggml_vars(env::vars_os()),
        })
    }

    pub(super) fn key(&self) -> String {
        let mut hash = Sha256::new();
        let mut length_prefixed_field = |bytes: &[u8]| {
            hash.update(bytes.len().to_le_bytes());
            hash.update(bytes);
        };
        length_prefixed_field(b"recall-probe-record-v1");
        length_prefixed_field(&self.binary_len.to_le_bytes());
        length_prefixed_field(&self.binary_modified_ns.to_le_bytes());
        length_prefixed_field(self.model_identity.as_bytes());
        length_prefixed_field(self.os_build.as_bytes());
        length_prefixed_field(self.hardware_model.as_bytes());
        length_prefixed_field(&self.ggml_vars.len().to_le_bytes());
        for (name, value) in &self.ggml_vars {
            length_prefixed_field(name.as_encoded_bytes());
            length_prefixed_field(value.as_encoded_bytes());
        }
        digest_hex(&hash.finalize())
    }
}

pub(super) fn ggml_vars(
    vars: impl IntoIterator<Item = (OsString, OsString)>,
) -> Vec<(OsString, OsString)> {
    let mut ggml: Vec<_> = vars
        .into_iter()
        .filter(|(name, _)| name.as_encoded_bytes().starts_with(b"GGML_"))
        .collect();
    ggml.sort();
    ggml
}

#[allow(unsafe_code)]
fn sysctl_string(name: &CStr) -> io::Result<String> {
    let mut len: libc::size_t = 0;
    // SAFETY: `name` is NUL-terminated; a null output buffer asks only for the length.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            ptr::null_mut(),
            &raw mut len,
            ptr::null_mut(),
            0,
        )
    };
    (status == 0)
        .then_some(())
        .ok_or_else(io::Error::last_os_error)?;
    let mut buf = vec![0_u8; len];
    // SAFETY: `buf` has `len` writable bytes, and the kernel writes at most `len` and updates it.
    let status = unsafe {
        libc::sysctlbyname(
            name.as_ptr(),
            buf.as_mut_ptr().cast(),
            &raw mut len,
            ptr::null_mut(),
            0,
        )
    };
    (status == 0)
        .then_some(())
        .ok_or_else(io::Error::last_os_error)?;
    buf.truncate(len);
    Ok(String::from_utf8_lossy(&buf)
        .trim_end_matches('\0')
        .to_owned())
}

pub(super) fn record_dir(
    hf_home: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
) -> Option<PathBuf> {
    state_dir(hf_home, xdg_cache_home, home, "probe")
}

pub(super) fn host_state_dir(leaf: &str) -> Option<PathBuf> {
    state_dir(
        env::var_os("HF_HOME").as_deref(),
        env::var_os("XDG_CACHE_HOME").as_deref(),
        env::var_os("HOME").as_deref(),
        leaf,
    )
}

fn state_dir(
    hf_home: Option<&OsStr>,
    xdg_cache_home: Option<&OsStr>,
    home: Option<&OsStr>,
    leaf: &str,
) -> Option<PathBuf> {
    let base = hf_home
        .map(PathBuf::from)
        .or_else(|| {
            xdg_cache_home
                .map(PathBuf::from)
                .filter(|path| path.is_absolute())
        })
        .or_else(|| home.map(|home| Path::new(home).join(".cache")))?;
    Some(base.join("recall").join(leaf))
}

fn next_nonce() -> u64 {
    static NONCE: AtomicU64 = AtomicU64::new(0);
    NONCE.fetch_add(1, Ordering::Relaxed)
}

fn owner_only_new_file() -> OpenOptions {
    let mut options = OpenOptions::new();
    options.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options
}

pub(super) fn write_atomically(target: &Path, contents: &str) -> io::Result<()> {
    target.parent().map_or(Ok(()), fs::create_dir_all)?;
    let temp = target.with_file_name(format!(".{}.{}.tmp", process::id(), next_nonce()));
    let mut file = owner_only_new_file().open(&temp)?;
    let result = file
        .write_all(contents.as_bytes())
        .and_then(|()| fs::rename(&temp, target));
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}

enum MarkerState {
    Finished,
    Running,
    Abandoned,
}

pub(crate) struct ProbeRecords {
    dir: PathBuf,
    key: String,
}

impl ProbeRecords {
    pub(crate) fn new(dir: PathBuf, key: String) -> Self {
        Self { dir, key }
    }

    pub(super) fn host(model_identity: &str) -> Option<Self> {
        let dir = record_dir(
            env::var_os("HF_HOME").as_deref(),
            env::var_os("XDG_CACHE_HOME").as_deref(),
            env::var_os("HOME").as_deref(),
        )?;
        let snapshot = Snapshot::host(model_identity)
            .inspect_err(|error| tracing::warn!(%error, "probe record environment unavailable; the isolated probe runs"))
            .ok()?;
        Some(Self::new(dir, snapshot.key()))
    }

    fn path(&self, suffix: &str) -> PathBuf {
        self.dir.join(format!("{}.{suffix}", self.key))
    }

    pub(super) fn may_skip(&self) -> bool {
        if self.no_skip() || !self.recorded() {
            return false;
        }
        match self.found_abandoned_marker() {
            Ok(false) => true,
            Err(error) => {
                tracing::warn!(%error, "could not inspect the skipped model load markers; the isolated probe runs");
                false
            }
            Ok(true) => {
                tracing::warn!(
                    "a model load that skipped the isolated probe did not finish; this environment runs the probe from now on"
                );
                self.forbid_skipping();
                false
            }
        }
    }

    pub(super) fn no_skip(&self) -> bool {
        match fs::symlink_metadata(self.path("no-skip")) {
            Ok(_) => true,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => {
                tracing::warn!(%error, "could not check the model probe no-skip marker; the isolated probe runs");
                true
            }
        }
    }

    fn recorded(&self) -> bool {
        match fs::read_to_string(self.path("ok")) {
            Ok(content) => content.trim() == self.key,
            Err(error) if error.kind() == ErrorKind::NotFound => false,
            Err(error) => {
                tracing::warn!(%error, "could not read the model probe record; the isolated probe runs");
                false
            }
        }
    }

    fn found_abandoned_marker(&self) -> io::Result<bool> {
        let prefix = format!("{}.", self.key);
        for entry in fs::read_dir(&self.dir)? {
            let path = entry?.path();
            let name = path.file_name().unwrap_or_default().to_string_lossy();
            if !(name.starts_with(&prefix) && name.ends_with(".running")) {
                continue;
            }
            let file = match File::open(&path) {
                Ok(file) => file,
                Err(error) if error.kind() == ErrorKind::NotFound => continue,
                Err(error) => return Err(error),
            };
            if matches!(marker_state(&path, &file)?, MarkerState::Abandoned) {
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn forbid_skipping(&self) {
        if let Err(error) = write_atomically(&self.path("no-skip"), "") {
            tracing::warn!(%error, "could not record that this environment must run the probe");
        }
        self.clear();
    }

    pub(super) fn record_success(&self) {
        if let Err(error) = write_atomically(&self.path("ok"), &self.key) {
            tracing::warn!(%error, "could not record the successful model probe");
        }
    }

    pub(super) fn clear(&self) {
        match fs::remove_file(self.path("ok")) {
            Err(error) if error.kind() != ErrorKind::NotFound => {
                tracing::warn!(%error, "could not clear the model probe record");
            }
            _ => {}
        }
    }

    pub(super) fn begin_skipped_load(&self) -> io::Result<RunningMarker> {
        fs::create_dir_all(&self.dir)?;
        let id = format!("{}.{}", process::id(), next_nonce());
        let staging = self.dir.join(format!(".{}.{id}.starting", self.key));
        let file = owner_only_new_file().open(&staging)?;
        let path = self.path(&format!("{id}.running"));
        match file.lock().and_then(|()| fs::rename(&staging, &path)) {
            Ok(()) => Ok(RunningMarker { path, _lock: file }),
            Err(error) => {
                let _ = fs::remove_file(&staging);
                Err(error)
            }
        }
    }
}

pub(super) struct RunningMarker {
    path: PathBuf,
    _lock: File,
}

impl RunningMarker {
    pub(super) fn finish(self) {
        // why-not: never release the lock before this unlink, or a reader takes the unlocked marker for a crash
        if let Err(error) = fs::remove_file(&self.path) {
            tracing::warn!(%error, "could not remove the skipped model load marker");
        }
    }
}

fn marker_state(path: &Path, file: &File) -> io::Result<MarkerState> {
    match file.try_lock() {
        Ok(()) if still_linked(path, file)? => Ok(MarkerState::Abandoned),
        Ok(()) => Ok(MarkerState::Finished),
        Err(TryLockError::WouldBlock) => {
            let age = SystemTime::now()
                .duration_since(file.metadata()?.modified()?)
                .unwrap_or_default();
            Ok(if age > HANG_AFTER_THE_PROBE_CHILD_TIMEOUT {
                MarkerState::Abandoned
            } else {
                MarkerState::Running
            })
        }
        Err(TryLockError::Error(error)) => Err(error),
    }
}

fn still_linked(path: &Path, file: &File) -> io::Result<bool> {
    use std::os::unix::fs::MetadataExt;
    let opened = file.metadata()?;
    match fs::metadata(path) {
        Ok(current) => Ok((current.dev(), current.ino()) == (opened.dev(), opened.ino())),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(false),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
#[path = "probe_record_tests.rs"]
mod tests;
