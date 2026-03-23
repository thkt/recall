use std::path::PathBuf;

use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;

pub(crate) const EMBEDDING_DIMS: usize = 768;
pub(crate) const QUERY_PREFIX: &str = "検索クエリ: ";
pub(crate) const DOCUMENT_PREFIX: &str = "検索文書: ";

/// Paths to model files (SafeTensors + config + tokenizer).
#[derive(Debug, Clone)]
pub(crate) struct ModelPaths {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    /// Construct paths assuming all files are in a single directory.
    #[cfg(test)]
    pub(crate) fn from_dir(dir: &std::path::Path) -> Self {
        Self {
            model: dir.join("model.safetensors"),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
        }
    }
}

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
                "Model not found at {}. Run `recall index` to download.",
                path.display()
            ),
            EmbedError::DimensionMismatch { expected, actual } => {
                write!(f, "Dimension mismatch: expected {expected}, got {actual}")
            }
            EmbedError::Inference(msg) => write!(f, "Inference error: {msg}"),
            EmbedError::Tokenizer(msg) => write!(f, "Tokenizer error: {msg}"),
        }
    }
}

impl std::error::Error for EmbedError {}

/// Select the best available device: Metal GPU if available, otherwise CPU.
pub(crate) fn select_device() -> Device {
    #[cfg(feature = "metal")]
    {
        if let Ok(device) = Device::new_metal(0) {
            return device;
        }
        eprintln!("Warning: Metal unavailable, using CPU");
    }
    Device::Cpu
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

    for (t, &m) in attention_mask.iter().enumerate().take(seq_len) {
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
    model: modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

impl Embedder {
    pub(crate) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        if !paths.model.exists() {
            return Err(EmbedError::ModelNotFound {
                path: paths.model.clone(),
            });
        }
        if !paths.config.exists() {
            return Err(EmbedError::ModelNotFound {
                path: paths.config.clone(),
            });
        }
        if !paths.tokenizer.exists() {
            return Err(EmbedError::ModelNotFound {
                path: paths.tokenizer.clone(),
            });
        }

        let device = select_device();

        let config_text = std::fs::read_to_string(&paths.config)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let config: modernbert::Config = serde_json::from_str(&config_text)
            .map_err(|e| EmbedError::Inference(format!("config.json parse error: {e}")))?;

        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                std::slice::from_ref(&paths.model),
                DType::F32,
                &device,
            )
            .map_err(|e| EmbedError::Inference(e.to_string()))?
        };

        // ruri-v3-310m keys: "embeddings.tok_embeddings.weight"
        // candle expects: "model.embeddings.tok_embeddings.weight"
        // Strip the "model." prefix candle prepends.
        let vb = vb.rename_f(|name| {
            name.strip_prefix("model.")
                .unwrap_or(name)
                .to_string()
        });

        let model = modernbert::ModernBert::load(vb, &config)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn embed_with_prefix(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        let prefixed = format!("{prefix}{text}");
        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        // Retain attention mask as Vec<u32> for mean pooling (DA: must not lose this)
        let attention_mask_u32: Vec<u32> = encoding.get_attention_mask().to_vec();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();

        // Create candle tensors (u32, shape [1, seq_len])
        let input_ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let attention_mask_tensor = Tensor::new(attention_mask_u32.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        // Forward pass: output shape [1, seq_len, hidden_size]
        let output = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        // Move to CPU and extract flat f32 data for mean_pooling
        let output_cpu = output
            .to_device(&Device::Cpu)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        // [1, seq_len, hidden_size] -> [seq_len * hidden_size]
        let flat: Vec<f32> = output_cpu
            .squeeze(0)
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1())
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let hidden_size = flat.len() / seq_len;
        if hidden_size != EMBEDDING_DIMS {
            return Err(EmbedError::DimensionMismatch {
                expected: EMBEDDING_DIMS,
                actual: hidden_size,
            });
        }

        let mut pooled = mean_pooling(&flat, seq_len, hidden_size, &attention_mask_u32);
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
    use std::path::Path;

    use super::*;

    // T-001: dimension mismatch (FR-001)
    #[test]
    fn test_001_validate_dims_rejects_wrong_size() {
        let wrong = vec![0.0f32; 256];
        let err = validate_dims(&wrong).unwrap_err();
        assert!(
            matches!(err, EmbedError::DimensionMismatch { actual: 256, .. }),
            "expected DimensionMismatch, got: {err}"
        );
    }

    // T-002: correct dims accepted (FR-001)
    #[test]
    fn test_002_validate_dims_accepts_correct_size() {
        let correct = vec![0.0f32; EMBEDDING_DIMS];
        assert!(validate_dims(&correct).is_ok());
    }

    // T-003: mean pooling excludes mask=0 tokens (FR-001)
    #[test]
    fn test_003_mean_pooling_excludes_masked_tokens() {
        #[rustfmt::skip]
        let data: Vec<f32> = vec![
            1.0, 2.0, 3.0, 4.0,       // token 0, mask=1
            5.0, 6.0, 7.0, 8.0,       // token 1, mask=1
            100.0, 100.0, 100.0, 100.0, // token 2, mask=0 (excluded)
        ];
        let mask = vec![1u32, 1, 0];
        let result = mean_pooling(&data, 3, 4, &mask);
        assert_eq!(result, vec![3.0, 4.0, 5.0, 6.0]);
    }

    // T-004: all masked returns zero vector (FR-001)
    #[test]
    fn test_004_mean_pooling_all_masked() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![0u32, 0];
        let result = mean_pooling(&data, 2, 2, &mask);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    // T-005: L2 normalize produces unit norm (FR-001)
    #[test]
    fn test_005_l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm should be 1.0, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    // T-006: zero vector stays zero (FR-001)
    #[test]
    fn test_006_l2_normalize_zero_vector() {
        let mut v = vec![0.0f32, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    // T-007: select_device returns CPU when Metal unavailable (FR-002)
    #[test]
    fn test_007_select_device_returns_cpu_without_metal() {
        let device = select_device();
        #[cfg(not(feature = "metal"))]
        assert!(
            matches!(device, Device::Cpu),
            "expected Device::Cpu without metal feature"
        );
        #[cfg(feature = "metal")]
        {
            // On Apple Silicon: Metal. On CI/other: CPU fallback.
            assert!(
                matches!(device, Device::Cpu | Device::Metal(_)),
                "expected Device::Cpu or Device::Metal"
            );
        }
    }

    // T-008: model not found (FR-003)
    #[test]
    fn test_008_embedder_new_model_not_found() {
        let paths = ModelPaths::from_dir(Path::new("/nonexistent/path"));
        let err = Embedder::new(&paths).unwrap_err();
        assert!(
            matches!(err, EmbedError::ModelNotFound { .. }),
            "expected ModelNotFound, got: {err}"
        );
        assert!(
            err.to_string().contains("recall index"),
            "error message should mention `recall index`: {err}"
        );
    }

    // T-009: ModelPaths::from_dir constructs correct paths
    #[test]
    fn test_009_model_paths_from_dir() {
        let paths = ModelPaths::from_dir(Path::new("/tmp/test-models"));
        assert_eq!(
            paths.model,
            PathBuf::from("/tmp/test-models/model.safetensors")
        );
        assert_eq!(paths.config, PathBuf::from("/tmp/test-models/config.json"));
        assert_eq!(
            paths.tokenizer,
            PathBuf::from("/tmp/test-models/tokenizer.json")
        );
    }
}
