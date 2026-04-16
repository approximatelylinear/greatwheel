//! Candle-based ColBERT encoder for Reason-ModernColBERT.
//!
//! Loads two safetensors files from a HuggingFace model directory:
//!
//! - `model.safetensors`         — the ModernBERT backbone (22 layers, 768 hidden)
//! - `1_Dense/model.safetensors` — the ColBERT projection head (Linear 768→128, no bias)
//!
//! At encode time:
//!   1. Prepend `[Q] ` to the query text.
//!   2. Tokenize with the model's tokenizer.
//!   3. Run ModernBERT forward → `[1, T, 768]` hidden states.
//!   4. Project with the dense head → `[1, T, 128]`.
//!   5. L2-normalize along the last dimension.
//!   6. Strip padding tokens (where attention mask is 0).
//!
//! The output is `Array2<f32>` of shape `(num_real_tokens, 128)`, ready to
//! pass to `BlobReranker::rerank` or `FirstStageRetriever::candidates`.
//!
//! Loading: use `CandleColbertEncoder::from_hf("lightonai/Reason-ModernColBERT")`
//! to download via `hf-hub`, or `from_dir("/path/to/local/model")` to load
//! from a directory you've already pulled.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use async_trait::async_trait;
use candle_core::{safetensors as candle_safetensors, DType, Device, Tensor};
use candle_nn::{linear_no_bias, Linear, Module, VarBuilder};
use candle_transformers::models::modernbert::{Config as ModernBertConfig, ModernBert};
use ndarray::Array2;
use tokenizers::Tokenizer;

use crate::colbert::ColbertEncoder;
use crate::error::MemoryError;

/// Default ColBERT projection dimension (Reason-ModernColBERT and most v2-era ColBERTs).
const PROJ_DIM: usize = 128;

/// Reason-ModernColBERT prepends `[Q] ` to query text. From the model's
/// `config_sentence_transformers.json`.
const QUERY_PREFIX: &str = "[Q] ";

/// Max query length from the same config (`query_length`).
const QUERY_MAX_TOKENS: usize = 128;

pub struct CandleColbertEncoder {
    model: ModernBert,
    projection: Linear,
    tokenizer: Tokenizer,
    device: Device,
    /// Set to `model.config.pad_token_id` so we can build attention masks.
    pad_token_id: u32,
}

impl CandleColbertEncoder {
    /// Load from a local directory containing `config.json`, `tokenizer.json`,
    /// `model.safetensors`, and `1_Dense/model.safetensors`.
    pub fn from_dir(dir: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let dir = dir.as_ref();
        let device = Device::cuda_if_available(0)
            .map_err(|e| MemoryError::Embedding(format!("device init: {e}")))?;

        // Config
        let config_path = dir.join("config.json");
        let config_str = std::fs::read_to_string(&config_path)
            .map_err(|e| MemoryError::Embedding(format!("read config.json: {e}")))?;
        let config: ModernBertConfig = serde_json::from_str(&config_str)
            .map_err(|e| MemoryError::Embedding(format!("parse config.json: {e}")))?;
        let pad_token_id = config.pad_token_id;

        // Tokenizer. The Reason-ModernColBERT tokenizer.json has padding to
        // 127 and truncation to 127 baked in (the ColBERT "query expansion"
        // trick — pad with [MASK] tokens so the model can use them as soft
        // expansion). But this model has `do_query_expansion=false`, so we
        // disable both and use the real attention mask instead. The Python
        // encoder does the same override.
        let tokenizer_path = dir.join("tokenizer.json");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| MemoryError::Embedding(format!("load tokenizer.json: {e}")))?;
        tokenizer.with_padding(None);
        let _ = tokenizer.with_truncation(None);

        // Backbone weights. The Reason-ModernColBERT checkpoint stores keys
        // *without* the `model.` prefix (because it was saved as a bare
        // ModernBertModel, not as a ForMaskedLM head). candle's `ModernBert::load`
        // expects the prefixed names, so we load the safetensors into a HashMap
        // and prepend `model.` to every key before constructing the VarBuilder.
        let backbone_path = dir.join("model.safetensors");
        let raw_tensors = candle_safetensors::load(&backbone_path, &device)
            .map_err(|e| MemoryError::Embedding(format!("load model.safetensors: {e}")))?;
        let mut prefixed: HashMap<String, Tensor> = HashMap::with_capacity(raw_tensors.len());
        for (k, v) in raw_tensors {
            prefixed.insert(format!("model.{k}"), v);
        }
        let vb_backbone = VarBuilder::from_tensors(prefixed, DType::F32, &device);
        let model = ModernBert::load(vb_backbone, &config)
            .map_err(|e| MemoryError::Embedding(format!("ModernBert::load: {e}")))?;

        // ColBERT projection head — separate file in 1_Dense/
        let dense_path = dir.join("1_Dense").join("model.safetensors");
        let vb_dense = unsafe {
            VarBuilder::from_mmaped_safetensors(&[dense_path], DType::F32, &device).map_err(
                |e| MemoryError::Embedding(format!("load 1_Dense/model.safetensors: {e}")),
            )?
        };
        // The dense head is stored as `linear.weight` in sentence-transformers checkpoints.
        // candle's `linear_no_bias` expects the weight under the `vb` root, so we pp into "linear".
        let projection = linear_no_bias(config.hidden_size, PROJ_DIM, vb_dense.pp("linear"))
            .map_err(|e| MemoryError::Embedding(format!("load dense projection: {e}")))?;

        Ok(Self {
            model,
            projection,
            tokenizer,
            device,
            pad_token_id,
        })
    }

    /// Convenience constructor that downloads from HuggingFace via `hf-hub`.
    /// Uses the default cache (`~/.cache/huggingface`).
    pub async fn from_hf(repo_id: &str) -> Result<Self, MemoryError> {
        use hf_hub::api::tokio::Api;

        let api = Api::new().map_err(|e| MemoryError::Embedding(format!("hf-hub init: {e}")))?;
        let repo = api.model(repo_id.to_string());

        // Download all the files we need into the cache and find the directory.
        let config = repo
            .get("config.json")
            .await
            .map_err(|e| MemoryError::Embedding(format!("download config.json: {e}")))?;
        repo.get("tokenizer.json")
            .await
            .map_err(|e| MemoryError::Embedding(format!("download tokenizer.json: {e}")))?;
        repo.get("model.safetensors")
            .await
            .map_err(|e| MemoryError::Embedding(format!("download model.safetensors: {e}")))?;
        repo.get("1_Dense/model.safetensors").await.map_err(|e| {
            MemoryError::Embedding(format!("download 1_Dense/model.safetensors: {e}"))
        })?;

        // The downloaded files share a parent directory in the cache.
        let dir: PathBuf = config
            .parent()
            .ok_or_else(|| MemoryError::Embedding("hf cache layout".into()))?
            .to_path_buf();
        Self::from_dir(dir)
    }

    /// Internal helper that does the work for both query (with [Q] prefix +
    /// length cap) and any future doc encoding (with [D] prefix). Today only
    /// query is wired up.
    fn encode_text(&self, text: &str, max_tokens: usize) -> Result<Array2<f32>, MemoryError> {
        // Tokenize
        let encoding = self
            .tokenizer
            .encode(text, true) // add special tokens (CLS/SEP)
            .map_err(|e| MemoryError::Embedding(format!("tokenize: {e}")))?;

        let mut ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.len() > max_tokens {
            ids.truncate(max_tokens);
        }
        // Pad to max_tokens with pad_token_id (matches sentence-transformers behavior).
        let real_len = ids.len();
        while ids.len() < max_tokens {
            ids.push(self.pad_token_id);
        }

        // Build attention mask (1 for real tokens, 0 for padding)
        let mask: Vec<u32> = (0..max_tokens)
            .map(|i| if i < real_len { 1u32 } else { 0u32 })
            .collect();

        let input_ids = Tensor::from_vec(ids, (1, max_tokens), &self.device)
            .map_err(|e| MemoryError::Embedding(format!("input_ids tensor: {e}")))?;
        let attention_mask = Tensor::from_vec(mask, (1, max_tokens), &self.device)
            .map_err(|e| MemoryError::Embedding(format!("attention_mask tensor: {e}")))?;

        // Forward → (1, T, 768)
        let hidden = self
            .model
            .forward(&input_ids, &attention_mask)
            .map_err(|e| MemoryError::Embedding(format!("modernbert forward: {e}")))?;

        // Project → (1, T, 128)
        let projected = self
            .projection
            .forward(&hidden)
            .map_err(|e| MemoryError::Embedding(format!("projection: {e}")))?;

        // L2 normalize along the last dim. candle has no built-in normalize, so do it by hand:
        // x / sqrt(sum(x^2, dim=-1, keepdim=True) + eps)
        let squared = projected
            .sqr()
            .map_err(|e| MemoryError::Embedding(format!("sqr: {e}")))?;
        let sum = squared
            .sum_keepdim(2)
            .map_err(|e| MemoryError::Embedding(format!("sum_keepdim: {e}")))?;
        let norm = (sum + 1e-12_f64)
            .and_then(|t| t.sqrt())
            .map_err(|e| MemoryError::Embedding(format!("norm sqrt: {e}")))?;
        let normalized = projected
            .broadcast_div(&norm)
            .map_err(|e| MemoryError::Embedding(format!("normalize: {e}")))?;

        // Strip padding: take only the first `real_len` tokens.
        // shape: (1, T, 128) → (real_len, 128)
        let trimmed = normalized
            .narrow(1, 0, real_len)
            .map_err(|e| MemoryError::Embedding(format!("narrow padding: {e}")))?
            .squeeze(0)
            .map_err(|e| MemoryError::Embedding(format!("squeeze batch: {e}")))?;

        // Pull to CPU as Vec<f32> and reshape into ndarray
        let host = trimmed
            .to_dtype(DType::F32)
            .and_then(|t| t.to_device(&Device::Cpu))
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| MemoryError::Embedding(format!("host transfer: {e}")))?;
        let arr = Array2::from_shape_vec((real_len, PROJ_DIM), host)
            .map_err(|e| MemoryError::Embedding(format!("reshape ndarray: {e}")))?;
        Ok(arr)
    }
}

#[async_trait]
impl ColbertEncoder for CandleColbertEncoder {
    async fn encode_query(&self, text: &str) -> Result<Array2<f32>, MemoryError> {
        let prefixed = format!("{QUERY_PREFIX}{text}");
        // Forward pass is sync from candle's perspective; nothing to await.
        self.encode_text(&prefixed, QUERY_MAX_TOKENS)
    }
}

/// Convenience: returns the encoder wrapped in an `Arc<dyn ColbertEncoder>`
/// so it can be plugged into a `ColbertStore` directly.
pub fn arc_encoder(enc: CandleColbertEncoder) -> Arc<dyn ColbertEncoder> {
    Arc::new(enc)
}
