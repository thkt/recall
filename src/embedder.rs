use std::path::{Path, PathBuf};

use ort::ep;
use ort::session::Session;
use ort::value::Tensor;

pub(crate) const EMBEDDING_DIMS: usize = 768;
pub(crate) const QUERY_PREFIX: &str = "検索クエリ: ";
pub(crate) const DOCUMENT_PREFIX: &str = "検索文書: ";

const MODEL_SUBDIR: &str = "recall/models/ruri-v3-310m";
const MODEL_FILENAME: &str = "model.onnx";
const TOKENIZER_FILENAME: &str = "tokenizer.json";

#[derive(Debug)]
pub(crate) enum EmbedError {
    ModelNotFound { path: PathBuf },
    DimensionMismatch { expected: usize, actual: usize },
    Inference(String),
    Tokenizer(String),
}

impl std::fmt::Display for EmbedError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EmbedError::ModelNotFound { path } => write!(
                f,
                "Model not found at {}. Run `recall model` to download.",
                path.display()
            ),
            EmbedError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
            EmbedError::Inference(msg) => write!(f, "ONNX inference error: {msg}"),
            EmbedError::Tokenizer(msg) => write!(f, "Tokenizer error: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

pub(crate) fn model_dir() -> PathBuf {
    if let Some(dir) = std::env::var_os("XDG_DATA_HOME") {
        return PathBuf::from(dir).join(MODEL_SUBDIR);
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".local/share")
        .join(MODEL_SUBDIR)
}

#[cfg(test)]
pub(crate) fn validate_dims(embedding: &[f32]) -> Result<(), EmbedError> {
    if embedding.len() != EMBEDDING_DIMS {
        return Err(EmbedError::DimensionMismatch {
            expected: EMBEDDING_DIMS,
            actual: embedding.len(),
        });
    }
    Ok(())
}

/// Mean pooling over token embeddings with attention mask.
///
/// `data` is a flat `[seq_len * hidden_size]` slice (row-major).
/// Tokens where `attention_mask[t] == 0` are excluded from the average.
pub(crate) fn mean_pooling(
    data: &[f32],
    seq_len: usize,
    hidden_size: usize,
    attention_mask: &[u32],
) -> Vec<f32> {
    let mut result = vec![0.0f32; hidden_size];
    let mut mask_sum = 0.0f32;

    for t in 0..seq_len {
        let m = attention_mask[t];
        if m > 0 {
            let mf = m as f32;
            let offset = t * hidden_size;
            for d in 0..hidden_size {
                result[d] += data[offset + d] * mf;
            }
            mask_sum += mf;
        }
    }

    if mask_sum > 0.0 {
        for v in &mut result {
            *v /= mask_sum;
        }
    }

    result
}

pub(crate) fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

pub(crate) trait Embed: std::fmt::Debug {
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, EmbedError>;
}

pub(crate) struct Embedder {
    session: Session,
    tokenizer: tokenizers::Tokenizer,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    pub(crate) fn new(dir: &Path) -> Result<Self, EmbedError> {
        let model_path = dir.join(MODEL_FILENAME);
        let tokenizer_path = dir.join(TOKENIZER_FILENAME);

        if !model_path.exists() {
            return Err(EmbedError::ModelNotFound { path: model_path });
        }
        if !tokenizer_path.exists() {
            return Err(EmbedError::ModelNotFound {
                path: tokenizer_path,
            });
        }

        let cache_dir = dir.join("coreml_cache");
        let coreml = ep::CoreML::default()
            .with_compute_units(ep::coreml::ComputeUnits::All)
            .with_model_format(ep::coreml::ModelFormat::MLProgram)
            .with_model_cache_dir(cache_dir.display().to_string())
            .build();

        let session = Session::builder()
            .map_err(|e| EmbedError::Inference(e.to_string()))?
            .with_execution_providers([coreml])
            .map_err(|e| EmbedError::Inference(e.to_string()))?
            .with_intra_threads(1)
            .map_err(|e| EmbedError::Inference(e.to_string()))?
            .commit_from_file(&model_path)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        Ok(Self { session, tokenizer })
    }

    fn embed_with_prefix(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        let prefixed = format!("{prefix}{text}");
        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let input_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();
        let attention_mask_i64: Vec<i64> =
            encoding.get_attention_mask().iter().map(|&m| m as i64).collect();
        let seq_len = input_ids.len();
        let shape = [1i64, seq_len as i64];

        let input_ids_tensor =
            Tensor::from_array((shape, input_ids)).map_err(|e| EmbedError::Inference(e.to_string()))?;
        let attention_mask_tensor =
            Tensor::from_array((shape, attention_mask_i64)).map_err(|e| EmbedError::Inference(e.to_string()))?;

        let outputs = self
            .session
            .run(ort::inputs![input_ids_tensor, attention_mask_tensor])
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let (out_shape, data) = outputs[0]
            .try_extract_tensor::<f32>()
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        // Shape: [1, seq_len, hidden_size]
        let hidden_size = out_shape[2] as usize;
        if hidden_size != EMBEDDING_DIMS {
            return Err(EmbedError::DimensionMismatch {
                expected: EMBEDDING_DIMS,
                actual: hidden_size,
            });
        }

        let mask: Vec<u32> = encoding.get_attention_mask().to_vec();
        let mut pooled = mean_pooling(data, seq_len, hidden_size, &mask);
        l2_normalize(&mut pooled);

        Ok(pooled)
    }
}

impl Embed for Embedder {
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_with_prefix(text, QUERY_PREFIX)
    }

    fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_with_prefix(text, DOCUMENT_PREFIX)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // T-003: dimension mismatch (FR-002)
    #[test]
    fn test_003_validate_dims_rejects_wrong_size() {
        let wrong = vec![0.0f32; 256];
        let err = validate_dims(&wrong).unwrap_err();
        assert!(
            matches!(err, EmbedError::DimensionMismatch { actual: 256, .. }),
            "expected DimensionMismatch, got: {err}"
        );
    }

    #[test]
    fn test_003_validate_dims_accepts_correct_size() {
        let correct = vec![0.0f32; EMBEDDING_DIMS];
        assert!(validate_dims(&correct).is_ok());
    }

    // T-014: model not found (FR-001)
    #[test]
    fn test_014_embedder_new_model_not_found() {
        let err = Embedder::new(Path::new("/nonexistent/path")).unwrap_err();
        assert!(
            matches!(err, EmbedError::ModelNotFound { .. }),
            "expected ModelNotFound, got: {err}"
        );
        assert!(
            err.to_string().contains("recall model"),
            "error message should mention `recall model`: {err}"
        );
    }

    // T-016: mean pooling excludes mask=0 tokens (FR-002)
    #[test]
    fn test_016_mean_pooling_excludes_masked_tokens() {
        // 3 tokens, 4 dims. Flat row-major: [t0d0, t0d1, ..., t2d3]
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,       // token 0, mask=1
            5.0, 6.0, 7.0, 8.0,       // token 1, mask=1
            100.0, 100.0, 100.0, 100.0, // token 2, mask=0 (excluded)
        ];
        let mask = vec![1u32, 1, 0];
        let result = mean_pooling(&data, 3, 4, &mask);
        // Mean of tokens 0,1: (1+5)/2=3, (2+6)/2=4, (3+7)/2=5, (4+8)/2=6
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    #[test]
    fn test_016_mean_pooling_all_masked() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![0u32, 0];
        let result = mean_pooling(&data, 2, 2, &mask);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    // T-017: L2 normalize produces unit norm (FR-002)
    #[test]
    fn test_017_l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-6,
            "norm should be 1.0, got {norm}"
        );
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_017_l2_normalize_zero_vector() {
        let mut v = vec![0.0f32, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    // model_dir resolution
    #[test]
    fn test_model_dir_with_xdg() {
        let key = "XDG_DATA_HOME";
        let saved = std::env::var(key).ok();
        unsafe { std::env::set_var(key, "/tmp/test-xdg") };
        let dir = model_dir();
        match saved {
            Some(v) => unsafe { std::env::set_var(key, v) },
            None => unsafe { std::env::remove_var(key) },
        }
        assert_eq!(
            dir,
            PathBuf::from("/tmp/test-xdg/recall/models/ruri-v3-310m")
        );
    }
}
