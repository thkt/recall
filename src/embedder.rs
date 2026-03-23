#[cfg(all(feature = "candle", feature = "mlx"))]
compile_error!("features `candle` and `mlx` are mutually exclusive — enable only one");

use std::path::PathBuf;

#[cfg(feature = "candle")]
use candle_core::{DType, Device, Tensor};
#[cfg(feature = "candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "candle")]
use candle_transformers::models::modernbert;

pub(crate) const EMBEDDING_DIMS: usize = 768;
pub(crate) const QUERY_PREFIX: &str = "検索クエリ: ";
pub(crate) const DOCUMENT_PREFIX: &str = "検索文書: ";

#[derive(Debug, Clone)]
pub(crate) struct ModelPaths {
    pub model: PathBuf,
    pub config: PathBuf,
    pub tokenizer: PathBuf,
}

impl ModelPaths {
    #[cfg(test)]
    pub(crate) fn from_dir(dir: &std::path::Path) -> Self {
        Self {
            model: dir.join("model.safetensors"),
            config: dir.join("config.json"),
            tokenizer: dir.join("tokenizer.json"),
        }
    }

    pub(crate) fn validate(&self) -> Result<(), EmbedError> {
        for path in [&self.model, &self.config, &self.tokenizer] {
            if !path.exists() {
                return Err(EmbedError::ModelNotFound { path: path.clone() });
            }
        }
        Ok(())
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
    fn embed_documents_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        texts.iter().map(|t| self.embed_document(t)).collect()
    }
}

// --- candle backend ---

#[cfg(feature = "candle")]
pub(crate) fn select_device() -> Device {
    Device::Cpu
}

#[cfg(feature = "candle")]
pub(crate) struct Embedder {
    model: modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
    device: Device,
}

#[cfg(feature = "candle")]
impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

#[cfg(feature = "candle")]
impl Embedder {
    pub(crate) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        paths.validate()?;
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
        let vb = vb.rename_f(|name| name.strip_prefix("model.").unwrap_or(name).to_string());

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

        let attention_mask_u32: Vec<u32> = encoding.get_attention_mask().to_vec();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();

        let input_ids_tensor = Tensor::new(input_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let attention_mask_tensor = Tensor::new(attention_mask_u32.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let output = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let output_cpu = output
            .to_device(&Device::Cpu)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

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

#[cfg(feature = "candle")]
impl Embed for Embedder {
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_with_prefix(text, QUERY_PREFIX)
    }

    fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_with_prefix(text, DOCUMENT_PREFIX)
    }
}

// --- mlx backend ---

#[cfg(feature = "mlx")]
pub(crate) struct Embedder {
    model: crate::modernbert::ModernBert,
    tokenizer: tokenizers::Tokenizer,
}

#[cfg(feature = "mlx")]
impl std::fmt::Debug for Embedder {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Embedder").finish_non_exhaustive()
    }
}

#[cfg(feature = "mlx")]
impl Embedder {
    pub(crate) fn new(paths: &ModelPaths) -> Result<Self, EmbedError> {
        paths.validate()?;
        let config_text = std::fs::read_to_string(&paths.config)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let config: crate::modernbert::Config = serde_json::from_str(&config_text)
            .map_err(|e| EmbedError::Inference(format!("config.json parse error: {e}")))?;

        let model = crate::modernbert::ModernBert::load(&paths.model, &config)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        let tokenizer = tokenizers::Tokenizer::from_file(&paths.tokenizer)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        Ok(Self { model, tokenizer })
    }

    fn embed_with_prefix(&mut self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError> {
        let prefixed = format!("{prefix}{text}");
        let encoding = self
            .tokenizer
            .encode(prefixed, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let attention_mask_u32: Vec<u32> = encoding.get_attention_mask().to_vec();
        let input_ids: Vec<u32> = encoding.get_ids().to_vec();
        let seq_len = input_ids.len();

        let output = self
            .model
            .forward(&input_ids, &attention_mask_u32, 1, seq_len as i32)
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        output
            .eval()
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let flat: &[f32] = output.as_slice();

        let hidden_size = flat.len() / seq_len;
        if hidden_size != EMBEDDING_DIMS {
            return Err(EmbedError::DimensionMismatch {
                expected: EMBEDDING_DIMS,
                actual: hidden_size,
            });
        }

        let mut pooled = mean_pooling(flat, seq_len, hidden_size, &attention_mask_u32);
        l2_normalize(&mut pooled);

        Ok(pooled)
    }
}

#[cfg(feature = "mlx")]
impl Embed for Embedder {
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_with_prefix(text, QUERY_PREFIX)
    }

    fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_with_prefix(text, DOCUMENT_PREFIX)
    }

    fn embed_documents_batch(&mut self, texts: &[&str]) -> Result<Vec<Vec<f32>>, EmbedError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }

        let prefixed: Vec<String> = texts
            .iter()
            .map(|t| format!("{DOCUMENT_PREFIX}{t}"))
            .collect();
        let encodings = self
            .tokenizer
            .encode_batch(prefixed, true)
            .map_err(|e| EmbedError::Tokenizer(e.to_string()))?;

        let batch_size = encodings.len();
        let max_seq_len = encodings.iter().map(|e| e.get_ids().len()).max().unwrap();

        let mut input_ids = vec![0u32; batch_size * max_seq_len];
        let mut attention_mask = vec![0u32; batch_size * max_seq_len];
        for (i, enc) in encodings.iter().enumerate() {
            let ids = enc.get_ids();
            let mask = enc.get_attention_mask();
            let offset = i * max_seq_len;
            input_ids[offset..offset + ids.len()].copy_from_slice(ids);
            attention_mask[offset..offset + mask.len()].copy_from_slice(mask);
        }

        let output = self
            .model
            .forward(
                &input_ids,
                &attention_mask,
                batch_size as i32,
                max_seq_len as i32,
            )
            .map_err(|e| EmbedError::Inference(e.to_string()))?;

        output
            .eval()
            .map_err(|e| EmbedError::Inference(e.to_string()))?;
        let flat: &[f32] = output.as_slice();

        let hidden_size = EMBEDDING_DIMS;
        let stride = max_seq_len * hidden_size;
        let mut results = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let seq_data = &flat[i * stride..(i + 1) * stride];
            let mask_slice = &attention_mask[i * max_seq_len..(i + 1) * max_seq_len];
            let mut pooled = mean_pooling(seq_data, max_seq_len, hidden_size, mask_slice);
            l2_normalize(&mut pooled);
            results.push(pooled);
        }

        Ok(results)
    }
}

/// Returns deterministic 768-dim vectors derived from text bytes.
#[cfg(test)]
#[derive(Debug)]
pub(crate) struct MockEmbedder {
    call_count: usize,
    fail_after: Option<usize>,
}

#[cfg(test)]
impl MockEmbedder {
    pub(crate) fn new() -> Self {
        Self {
            call_count: 0,
            fail_after: None,
        }
    }

    pub(crate) fn failing_after(n: usize) -> Self {
        Self {
            call_count: 0,
            fail_after: Some(n),
        }
    }

    pub(crate) fn deterministic_vector(text: &str) -> Vec<f32> {
        let mut v = vec![0.0f32; EMBEDDING_DIMS];
        for (i, b) in text.bytes().enumerate() {
            v[i % EMBEDDING_DIMS] += b as f32;
        }
        l2_normalize(&mut v);
        v
    }
}

#[cfg(test)]
impl Embed for MockEmbedder {
    fn embed_query(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        self.embed_document(text)
    }

    fn embed_document(&mut self, text: &str) -> Result<Vec<f32>, EmbedError> {
        if let Some(limit) = self.fail_after {
            if self.call_count >= limit {
                return Err(EmbedError::Inference("mock failure".to_string()));
            }
        }
        self.call_count += 1;
        Ok(Self::deterministic_vector(text))
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

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

    #[test]
    fn test_004_mean_pooling_all_masked() {
        let data = vec![1.0f32, 2.0, 3.0, 4.0];
        let mask = vec![0u32, 0];
        let result = mean_pooling(&data, 2, 2, &mask);
        assert_eq!(result, vec![0.0, 0.0]);
    }

    #[test]
    fn test_005_l2_normalize_produces_unit_norm() {
        let mut v = vec![3.0f32, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6, "norm should be 1.0, got {norm}");
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_006_l2_normalize_zero_vector() {
        let mut v = vec![0.0f32, 0.0];
        l2_normalize(&mut v);
        assert_eq!(v, vec![0.0, 0.0]);
    }

    #[cfg(feature = "candle")]
    #[test]
    fn test_007_select_device_returns_cpu() {
        let device = select_device();
        assert!(matches!(device, Device::Cpu), "expected Device::Cpu");
    }

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

    // Benchmark: measure per-chunk embedding latency
    #[test]
    #[ignore]
    #[cfg_attr(feature = "mlx", serial_test::serial)]
    fn bench_embed_latency() {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.model("cl-nagoya/ruri-v3-310m".to_string());
        let paths = ModelPaths {
            model: repo.get("model.safetensors").unwrap(),
            config: repo.get("config.json").unwrap(),
            tokenizer: repo.get("tokenizer.json").unwrap(),
        };

        let load_start = std::time::Instant::now();
        let mut embedder = Embedder::new(&paths).expect("load model");
        eprintln!("Model load: {:.1}s", load_start.elapsed().as_secs_f64());

        let texts = [
            "Rustでの認証実装方法について教えてください",
            "How to implement authentication in Rust using argon2",
            "クイックソートアルゴリズムの説明",
            "The quick brown fox jumps over the lazy dog",
            "mlx unified memory embedding backend migration plan",
            "candle Metal per-op GPU transfer overhead causes slowness",
            "ModernBERT transformer layers alternating local global attention",
            "SQLite FTS5 full text search with Japanese tokenization support",
            "GitHub Actions CI workflow for macOS and Linux compilation",
            "Progressive embedding strategy for search result improvement",
        ];

        // Warmup
        let _ = embedder.embed_document("warmup text");

        let mut times = Vec::new();
        for (i, text) in texts.iter().enumerate() {
            let start = std::time::Instant::now();
            let result = embedder.embed_document(text).expect("embed");
            let ms = start.elapsed().as_secs_f64() * 1000.0;
            times.push(ms);
            eprintln!("  [{:2}] {:6.1}ms ({}d)", i + 1, ms, result.len());
        }

        let avg = times.iter().sum::<f64>() / times.len() as f64;
        let min = times.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = times.iter().cloned().fold(0.0f64, f64::max);
        eprintln!("\nSequential ({} chunks):", texts.len());
        eprintln!("  Avg: {avg:.0}ms  Min: {min:.0}ms  Max: {max:.0}ms");

        for batch_size in [8, 10, 16, 32] {
            let mut batch_texts: Vec<&str> = Vec::new();
            for i in 0..batch_size {
                batch_texts.push(texts[i % texts.len()]);
            }
            let batch_start = std::time::Instant::now();
            let batch_results = embedder.embed_documents_batch(&batch_texts).expect("batch");
            let batch_ms = batch_start.elapsed().as_secs_f64() * 1000.0;
            let batch_per_chunk = batch_ms / batch_results.len() as f64;
            eprintln!(
                "\nBatch ({} chunks): {batch_ms:.1}ms total, {batch_per_chunk:.1}ms/chunk, {:.1}x speedup",
                batch_results.len(),
                avg / batch_per_chunk
            );
        }

        // Verify batch correctness (use batch=10)
        let verify_texts: Vec<&str> = texts.iter().copied().collect();
        let verify_results = embedder
            .embed_documents_batch(&verify_texts)
            .expect("verify");
        for (i, result) in verify_results.iter().enumerate() {
            assert_eq!(result.len(), EMBEDDING_DIMS, "batch[{i}] should be 768d");
            let norm: f32 = result.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "batch[{i}] should be L2 normalized, got norm={norm}"
            );
        }
    }

    #[test]
    #[ignore] // requires model download
    #[cfg_attr(feature = "mlx", serial_test::serial)]
    fn test_embed_documents_batch_matches_sequential() {
        let api = hf_hub::api::sync::Api::new().unwrap();
        let repo = api.model("cl-nagoya/ruri-v3-310m".to_string());
        let paths = ModelPaths {
            model: repo.get("model.safetensors").unwrap(),
            config: repo.get("config.json").unwrap(),
            tokenizer: repo.get("tokenizer.json").unwrap(),
        };
        let mut embedder = Embedder::new(&paths).expect("load model");

        let texts = [
            "Rustでの認証実装方法",
            "How to implement auth",
            "クイックソートの説明",
        ];

        // Sequential
        let sequential: Vec<Vec<f32>> = texts
            .iter()
            .map(|t| embedder.embed_document(t).expect("embed"))
            .collect();

        // Batch
        let text_refs: Vec<&str> = texts.iter().copied().collect();
        let batch = embedder.embed_documents_batch(&text_refs).expect("batch");

        assert_eq!(batch.len(), sequential.len());

        // Cosine similarity > 0.999 for each pair
        for (i, (b, s)) in batch.iter().zip(sequential.iter()).enumerate() {
            assert_eq!(b.len(), EMBEDDING_DIMS, "batch[{i}] dims");
            let cosine: f32 = b.iter().zip(s.iter()).map(|(a, b)| a * b).sum();
            assert!(
                cosine > 0.999,
                "batch[{i}] vs sequential cosine={cosine}, expected >0.999"
            );
        }

        // Empty input
        let empty: Vec<&str> = Vec::new();
        assert!(
            embedder
                .embed_documents_batch(&empty)
                .expect("empty")
                .is_empty()
        );
    }
}
