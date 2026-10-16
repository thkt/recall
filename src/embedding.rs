use std::time::Duration;

pub const EMBEDDING_DIMS: usize = 768;

#[derive(Debug, thiserror::Error)]
pub enum EmbedError {
    #[error("empty embedding sequence")]
    EmptySequence,
    #[error("embedding inference failed: {message}")]
    Inference {
        message: String,
        #[source]
        source: Option<anyhow::Error>,
    },
}

impl From<anyhow::Error> for EmbedError {
    fn from(error: anyhow::Error) -> Self {
        Self::Inference {
            message: error.to_string(),
            source: Some(error),
        }
    }
}

pub struct ChunkedEmbedding(Vec<Vec<f32>>);
impl ChunkedEmbedding {
    pub fn try_new(chunks: Vec<Vec<f32>>) -> Result<Self, EmbedError> {
        if chunks.is_empty() || chunks.iter().any(Vec::is_empty) {
            return Err(EmbedError::EmptySequence);
        }
        Ok(Self(chunks))
    }
    pub fn chunks(&self) -> &[Vec<f32>] {
        &self.0
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct EmbedOptions {
    pub token_budget: Option<usize>,
    pub forward_pause: Option<Duration>,
}

pub trait Embed: Send + Sync {
    fn embed_query(&self, text: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_document(&self, text: &str) -> Result<ChunkedEmbedding, EmbedError>;
    fn embed_documents_batch(&self, texts: &[&str]) -> Result<Vec<ChunkedEmbedding>, EmbedError>;
    fn embed_text(&self, text: &str, prefix: &str) -> Result<Vec<f32>, EmbedError>;
    fn embed_documents_batch_with_options(
        &self,
        texts: &[&str],
        _options: &EmbedOptions,
    ) -> Result<Vec<ChunkedEmbedding>, EmbedError> {
        self.embed_documents_batch(texts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_empty_chunk_never_becomes_a_persistable_embedding() {
        for chunks in [vec![], vec![vec![]], vec![vec![1.], vec![]]] {
            assert!(matches!(
                ChunkedEmbedding::try_new(chunks),
                Err(EmbedError::EmptySequence)
            ));
        }
    }
}
