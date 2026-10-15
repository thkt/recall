//! Pinned artifacts and an index identity; model updates never reuse old vectors.
use crate::{
    embedding::Embed,
    native::{Gemma, gpu_available},
};
use anyhow::{Context, Result, ensure};
use reqwest::blocking::Client;
use rusqlite::{Connection, OptionalExtension};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    env, fmt, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    process::{self, Command, Stdio},
    sync::{Arc, OnceLock},
    thread,
    time::{Duration, Instant},
};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ModelSpec {
    pub model_id: String,
    pub revision: String,
    pub model_url: String,
    pub model_sha256: String,
    pub tokenizer_revision: String,
    pub tokenizer_url: String,
    pub tokenizer_sha256: String,
    pub dimensions: usize,
    pub max_tokens: usize,
    pub overlap_tokens: usize,
    pub query_prefix: String,
    pub document_prefix: String,
    pub pooling: String,
    pub implementation: String,
    pub backend_revision: String,
    pub bindings_revision: String,
}
impl Default for ModelSpec {
    fn default() -> Self {
        Self {
            model_id:"embeddinggemma-2-bf16".into(),revision:"bfcd298762cc34d0357ece5ebdd31791a3a374d8".into(),
            model_url:"https://huggingface.co/ggml-org/embeddinggemma-2-GGUF/resolve/bfcd298762cc34d0357ece5ebdd31791a3a374d8/embeddinggemma-2-BF16.gguf".into(),
            model_sha256:"68bae29d62fb8c7d23e98d21fd4662753ddd636e6b62b8a70dfe059a9844f216".into(),
            tokenizer_revision:"914f7f89142e33e77833254d9c9b90c3cef7303b".into(),
            tokenizer_url:"https://huggingface.co/google/embeddinggemma-2/resolve/914f7f89142e33e77833254d9c9b90c3cef7303b/tokenizer.json".into(),
            tokenizer_sha256:"4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4".into(),
            dimensions:768,max_tokens:8192,overlap_tokens:2048,
            query_prefix:"task: search result | query: ".into(),document_prefix:"title: none | text: ".into(),
            pooling:"mean-l2-noncausal".into(),implementation:"recall-gemma-llama-v1".into(),
            backend_revision:"1b70350c92563449576bcbbf552994b1d1ed847d".into(),
            bindings_revision:"849e04446cb7b6a1fe3cee650e635a081ec072be".into(),
        }
    }
}
impl ModelSpec {
    pub fn validate(&self) -> Result<()> {
        for hash in [&self.model_sha256, &self.tokenizer_sha256] {
            ensure!(
                hash.len() == 64
                    && hash
                        .bytes()
                        .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
                "artifact SHA256 must be 64 lowercase hex characters"
            );
        }
        for revision in [&self.revision, &self.tokenizer_revision] {
            ensure!(
                revision.len() == 40 && revision.bytes().all(|b| b.is_ascii_hexdigit()),
                "model revisions must be pinned full commit IDs"
            );
        }
        ensure!(!self.model_id.is_empty(), "model_id must not be empty");
        ensure!(
            self.dimensions == 768 && self.max_tokens == 8192 && self.overlap_tokens == 2048,
            "unsupported model dimensions or token splitting contract"
        );
        ensure!(
            self.pooling == "mean-l2-noncausal" && self.implementation == "recall-gemma-llama-v1",
            "unsupported encoder implementation or pooling"
        );
        ensure!(
            self.backend_revision == "1b70350c92563449576bcbbf552994b1d1ed847d"
                && self.bindings_revision == "849e04446cb7b6a1fe3cee650e635a081ec072be",
            "manifest requires a different encoder backend; use its matching binary"
        );
        for url in [&self.model_url, &self.tokenizer_url] {
            ensure!(
                url.starts_with("https://"),
                "artifact downloads require HTTPS"
            );
        }
        Ok(())
    }
    pub fn identity(&self) -> String {
        format!(
            "{:x}",
            Sha256::digest(serde_json::to_vec(self).expect("model spec is serializable"))
        )
    }
}
static SELECTED: OnceLock<ModelSpec> = OnceLock::new();
pub fn selected() -> &'static ModelSpec {
    SELECTED.get_or_init(ModelSpec::default)
}
pub fn select(path: Option<&Path>) -> Result<()> {
    let spec = match path {
        Some(path) => serde_json::from_reader(fs::File::open(path)?)?,
        None => ModelSpec::default(),
    };
    spec.validate()?;
    match SELECTED.set(spec.clone()) {
        Ok(()) => Ok(()),
        Err(_) if selected().identity() == spec.identity() => Ok(()),
        Err(_) => anyhow::bail!("a different model is already selected in this process"),
    }
}
pub fn hash_file(path: &Path) -> Result<String> {
    let mut h = Sha256::new();
    let mut f = fs::File::open(path)?;
    let mut buf = vec![0; 1024 * 1024];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(format!("{:x}", h.finalize()))
}
pub fn cache_dir() -> PathBuf {
    if let Some(p) = env::var_os("RECALL_MODEL_DIR") {
        return p.into();
    }
    let base = env::var_os("HF_HOME")
        .map(PathBuf::from)
        .map(|p| p.join("recall"))
        .or_else(|| {
            env::var_os("XDG_CACHE_HOME")
                .map(PathBuf::from)
                .filter(|p| p.is_absolute())
                .map(|p| p.join("recall"))
        })
        .unwrap_or_else(|| dirs::home_dir().unwrap_or_default().join(".cache/recall"));
    base.join("models").join(selected().identity())
}
pub struct ModelId;
impl ModelId {
    pub const DEFAULT: Self = Self;
}
pub fn cached_artifacts(_id: ModelId) -> Result<Option<PathBuf>> {
    let root = cache_dir();
    if !root.join("model.gguf").is_file() || !root.join("tokenizer.json").is_file() {
        return Ok(None);
    }
    ensure!(
        hash_file(&root.join("model.gguf"))? == selected().model_sha256,
        "model checksum mismatch"
    );
    ensure!(
        hash_file(&root.join("tokenizer.json"))? == selected().tokenizer_sha256,
        "tokenizer checksum mismatch"
    );
    Ok(Some(root))
}
fn load_verified() -> Result<Arc<dyn Embed>> {
    let root = cached_artifacts(ModelId::DEFAULT)?.context("model not installed")?;
    Ok(Arc::new(Gemma::load(
        &root.join("model.gguf"),
        &root.join("tokenizer.json"),
        selected().clone(),
    )?))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DegradedReason {
    NotInstalled,
    ProbeFailed,
    BackendUnavailable,
    #[cfg(test)]
    Disabled,
}
impl fmt::Display for DegradedReason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::NotInstalled => "model not installed",
                Self::ProbeFailed => "model probe failed",
                Self::BackendUnavailable => "backend unavailable",
                #[cfg(test)]
                Self::Disabled => "model disabled",
            }
        )
    }
}
pub struct EmbedderDegraded(pub DegradedReason);
impl EmbedderDegraded {
    pub fn user_note(&self, download: &str) -> Option<String> {
        match self.0 {
            #[cfg(test)]
            DegradedReason::Disabled => None,
            DegradedReason::NotInstalled => Some(format!(
                "embedding model not installed; run `{download}` to enable semantic search"
            )),
            DegradedReason::ProbeFailed | DegradedReason::BackendUnavailable => {
                Some("embedding model unavailable; results from text search only".into())
            }
        }
    }
}
pub fn record_degraded(reason: DegradedReason, stage: &str) {
    tracing::warn!(%reason,stage,"semantic search unavailable");
}
pub fn degrade_with_warn<E: fmt::Display>(
    stage: &'static str,
    reason: DegradedReason,
) -> impl FnOnce(E) -> DegradedReason {
    move |e| {
        tracing::warn!(error=%e,stage,"model operation failed");
        reason
    }
}

pub fn handle_probe_if_needed() {
    if env::var_os("__RECALL_GEMMA_PROBE").is_none() {
        return;
    }
    let result = (|| -> Result<()> {
        let spec: ModelSpec = serde_json::from_str(&env::var("__RECALL_GEMMA_SPEC")?)?;
        spec.validate()?;
        SELECTED
            .set(spec)
            .map_err(|_| anyhow::anyhow!("model already selected"))?;
        if !gpu_available()? {
            process::exit(2);
        }
        let model = load_verified()?;
        let v = model.embed_query("model probe")?;
        ensure!(
            v.len() == 768 && v.iter().all(|x| x.is_finite()),
            "invalid probe embedding"
        );
        Ok(())
    })();
    process::exit(if result.is_ok() { 0 } else { 1 });
}
fn probe() -> Result<bool> {
    let mut child = Command::new(env::current_exe()?)
        .env("__RECALL_GEMMA_PROBE", "1")
        .env("__RECALL_GEMMA_SPEC", serde_json::to_string(selected())?)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()?;
    let deadline = Instant::now() + Duration::from_secs(30);
    loop {
        if let Some(status) = child.try_wait()? {
            if status.code() == Some(2) {
                return Ok(false);
            }
            ensure!(status.success(), "isolated model probe failed");
            return Ok(true);
        }
        if Instant::now() >= deadline {
            let _ = child.kill();
            let _ = child.wait();
            anyhow::bail!("isolated model probe timed out")
        }
        thread::sleep(Duration::from_millis(10));
    }
}
pub fn try_load_embedder_default_logging() -> Result<Arc<dyn Embed>, DegradedReason> {
    match cached_artifacts(ModelId::DEFAULT) {
        Ok(None) => return Err(DegradedReason::NotInstalled),
        Err(error) => {
            tracing::warn!(%error,"model artifact validation failed");
            return Err(DegradedReason::ProbeFailed);
        }
        Ok(Some(_)) => {}
    }
    match probe() {
        Ok(false) => Err(DegradedReason::BackendUnavailable),
        Ok(true) => {
            load_verified().map_err(degrade_with_warn("model load", DegradedReason::ProbeFailed))
        }
        Err(error) => Err(degrade_with_warn(
            "model probe",
            DegradedReason::ProbeFailed,
        )(error)),
    }
}
#[derive(Debug, thiserror::Error)]
pub enum ModelDownloadError {
    #[error("model download failed: {0}")]
    DownloadFailed(String),
    #[error("model probe failed: {0:?}")]
    ProbeFailed(Option<String>),
    #[error("backend unavailable")]
    BackendUnavailable,
}

/// Replace only a checksum-verified artifact. Failed/partial downloads preserve
/// the prior file, and cleanup never removes a temporary file we did not create.
fn install_artifact(root: &Path, name: &str, expected: &str, reader: &mut impl Read) -> Result<()> {
    let temp = root.join(format!(".{name}.{}.download", process::id()));
    let mut opts = fs::OpenOptions::new();
    opts.write(true).create_new(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        opts.mode(0o600);
    }
    let mut file = opts.open(&temp)?;
    let result = (|| -> Result<()> {
        io::copy(reader, &mut file)?;
        file.flush()?;
        file.sync_all()?;
        ensure!(hash_file(&temp)? == expected, "download checksum mismatch");
        fs::rename(&temp, root.join(name))?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temp);
    }
    result
}
pub fn download_and_verify_model() -> Result<(), ModelDownloadError> {
    let install = (|| -> Result<()> {
        let root = cache_dir();
        fs::create_dir_all(&root)?;
        let client = Client::builder()
            .timeout(Duration::from_secs(600))
            .build()?;
        for (name, url, expected) in [
            (
                "model.gguf",
                &selected().model_url,
                &selected().model_sha256,
            ),
            (
                "tokenizer.json",
                &selected().tokenizer_url,
                &selected().tokenizer_sha256,
            ),
        ] {
            let dest = root.join(name);
            if dest.exists() && hash_file(&dest)? == *expected {
                continue;
            }
            install_artifact(
                &root,
                name,
                expected,
                &mut client.get(url).send()?.error_for_status()?,
            )?;
        }
        Ok(())
    })();
    install.map_err(|e| ModelDownloadError::DownloadFailed(e.to_string()))?;
    match probe() {
        Ok(true) => Ok(()),
        Ok(false) => Err(ModelDownloadError::BackendUnavailable),
        Err(error) => Err(ModelDownloadError::ProbeFailed(Some(error.to_string()))),
    }
}

/// Missing identity plus existing vectors is legacy Ruri, never implicitly Gemma.
pub fn check_index(conn: &Connection, spec: &ModelSpec) -> Result<()> {
    let table:bool=conn.query_row("SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='embedding_identity' AND type='table')",[],|r|r.get(0))?;
    if table {
        let saved: Option<String> = conn
            .query_row(
                "SELECT fingerprint FROM embedding_identity WHERE singleton=1",
                [],
                |r| r.get(0),
            )
            .optional()?;
        ensure!(
            saved.as_deref() == Some(spec.identity().as_str()),
            "index embedding identity mismatch; use a separate --db-path for this model or select the matching --model-manifest"
        );
        return Ok(());
    }
    let vectors: bool = conn.query_row(
        "SELECT EXISTS(SELECT 1 FROM sqlite_master WHERE name='vec_chunks')",
        [],
        |r| r.get(0),
    )?;
    if vectors {
        let count: bool =
            conn.query_row("SELECT EXISTS(SELECT 1 FROM vec_chunks)", [], |r| r.get(0))?;
        ensure!(
            !count,
            "legacy vector index has no model identity; keep it for rollback and build Gemma in a separate --db-path"
        );
    }
    Ok(())
}
pub fn bind_index(conn: &mut Connection, spec: &ModelSpec) -> Result<()> {
    let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
    check_index(&tx, spec)?;
    tx.execute_batch("CREATE TABLE IF NOT EXISTS embedding_identity (singleton INTEGER PRIMARY KEY CHECK(singleton=1), fingerprint TEXT NOT NULL, manifest TEXT NOT NULL)")?;
    tx.execute(
        "INSERT OR IGNORE INTO embedding_identity VALUES(1,?1,?2)",
        rusqlite::params![spec.identity(), serde_json::to_string(spec)?],
    )?;
    tx.commit()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn verified_artifact_replaces_only_after_complete_checksum_match() {
        let dir = tempfile::TempDir::new().unwrap();
        let dest = dir.path().join("model.gguf");
        fs::write(&dest, b"old").unwrap();
        let bytes = b"new verified weights";
        let digest = format!("{:x}", Sha256::digest(bytes));
        install_artifact(dir.path(), "model.gguf", &digest, &mut bytes.as_slice()).unwrap();
        assert_eq!(fs::read(dest).unwrap(), bytes);
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn bad_checksum_and_partial_stream_preserve_previous_artifact() {
        struct Failed;
        impl Read for Failed {
            fn read(&mut self, _: &mut [u8]) -> io::Result<usize> {
                Err(io::Error::other("interrupted download"))
            }
        }
        let dir = tempfile::TempDir::new().unwrap();
        let dest = dir.path().join("model.gguf");
        fs::write(&dest, b"old").unwrap();
        assert!(
            install_artifact(
                dir.path(),
                "model.gguf",
                &"0".repeat(64),
                &mut b"wrong".as_slice()
            )
            .is_err()
        );
        assert!(install_artifact(dir.path(), "model.gguf", &"0".repeat(64), &mut Failed).is_err());
        assert_eq!(fs::read(dest).unwrap(), b"old");
        assert_eq!(fs::read_dir(dir.path()).unwrap().count(), 1);
    }
    #[test]
    fn occupied_temporary_path_is_never_deleted_or_overwritten() {
        let dir = tempfile::TempDir::new().unwrap();
        let temp = dir
            .path()
            .join(format!(".model.gguf.{}.download", process::id()));
        fs::write(&temp, b"someone else").unwrap();
        assert!(
            install_artifact(
                dir.path(),
                "model.gguf",
                &"0".repeat(64),
                &mut b"new".as_slice()
            )
            .is_err()
        );
        assert_eq!(fs::read(temp).unwrap(), b"someone else");
        assert!(!dir.path().join("model.gguf").exists());
    }
    fn empty_index() -> Connection {
        let c = Connection::open_in_memory().unwrap();
        c.execute_batch("CREATE TABLE vec_chunks(id INTEGER)")
            .unwrap();
        c
    }
    #[test]
    fn model_or_tokenizer_or_prefix_changes_select_a_new_index_identity() {
        let old = ModelSpec::default();
        let id = old.identity();
        for field in ["weights", "tokenizer", "prefix", "revision"] {
            let mut next = old.clone();
            match field {
                "weights" => next.model_sha256 = "0".repeat(64),
                "tokenizer" => next.tokenizer_sha256 = "1".repeat(64),
                "prefix" => next.query_prefix.push_str("new "),
                _ => next.revision = "2".repeat(40),
            }
            next.validate().unwrap();
            assert_ne!(id, next.identity());
        }
    }
    #[test]
    fn legacy_vectors_are_rejected_without_stamping_or_deleting_them() {
        let mut c = empty_index();
        c.execute("INSERT INTO vec_chunks VALUES(7)", []).unwrap();
        assert!(bind_index(&mut c, &ModelSpec::default()).is_err());
        assert_eq!(
            c.query_row("SELECT id FROM vec_chunks", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            7
        );
        assert_eq!(
            c.query_row(
                "SELECT count(*) FROM sqlite_master WHERE name='embedding_identity'",
                [],
                |r| r.get::<_, i64>(0)
            )
            .unwrap(),
            0
        );
    }
    #[test]
    fn different_model_cannot_rebind_an_existing_index_and_old_model_can_return() {
        let old = ModelSpec::default();
        let mut next = old.clone();
        next.document_prefix.push_str("next ");
        let mut c = empty_index();
        bind_index(&mut c, &old).unwrap();
        c.execute("INSERT INTO vec_chunks VALUES(19)", []).unwrap();
        assert!(bind_index(&mut c, &next).is_err());
        check_index(&c, &old).unwrap();
        bind_index(&mut c, &old).unwrap();
        assert_eq!(
            c.query_row("SELECT id FROM vec_chunks", [], |r| r.get::<_, i64>(0))
                .unwrap(),
            19
        );
        let mut new = empty_index();
        bind_index(&mut new, &next).unwrap();
        check_index(&new, &next).unwrap();
        assert!(check_index(&new, &old).is_err());
    }
    #[test]
    fn invalid_profile_shapes_fail_closed() {
        assert!(
            ModelSpec {
                model_sha256: "not a checksum".into(),
                ..ModelSpec::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ModelSpec {
                dimensions: 1024,
                ..ModelSpec::default()
            }
            .validate()
            .is_err()
        );
        assert!(
            ModelSpec {
                revision: "main".into(),
                ..ModelSpec::default()
            }
            .validate()
            .is_err()
        );
    }
}
