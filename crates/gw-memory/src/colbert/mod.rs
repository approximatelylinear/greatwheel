//! ColBERT-style late-interaction retrieval pipeline.
//!
//! Three-stage architecture, each stage independently swappable:
//!
//! 1. **Encoder** (`ColbertEncoder`) — turns query text into a token tensor.
//!    Today: candle ModernBERT loading Reason-ModernColBERT.
//! 2. **First-stage retriever** (`FirstStageRetriever`) — generates candidate
//!    docids from a query. Today: usearch over flattened token vectors.
//!    Could also be tantivy BM25, dense single-vector ANN, etc.
//! 3. **Reranker** (`BlobReranker`, in `colbert_blobs.rs`) — concrete struct,
//!    not a trait, since there's no plausible alternative implementation
//!    we'd want to swap.
//!
//! See `docs/design-colbert-production.md` for the architecture rationale
//! and the BrowseComp empirical numbers that justify it.

pub mod candle_encoder;
pub mod usearch_retriever;

use std::sync::Arc;

use async_trait::async_trait;
use ndarray::{Array2, ArrayView2};

use crate::colbert_blobs::{BlobReranker, ScoredDoc};
use crate::error::MemoryError;

/// Encodes text into ColBERT token vectors. Output shape is
/// `(num_tokens, 128)` and L2-normalized along the last dimension.
#[async_trait]
pub trait ColbertEncoder: Send + Sync {
    /// Encode a query string. The implementation is responsible for adding
    /// any model-specific prefix (e.g. `[Q] ` for ColBERT) before tokenizing.
    async fn encode_query(&self, text: &str) -> Result<Array2<f32>, MemoryError>;
}

/// First-stage candidate generator. Given a query (and its pre-encoded token
/// tensor for dense retrievers), returns the top-K candidate document IDs.
#[async_trait]
pub trait FirstStageRetriever: Send + Sync {
    /// Returns at most `k` candidate docids ranked by the retriever's own
    /// scoring. The implementation may use either the raw `text` (sparse
    /// retrievers like BM25) or the pre-encoded `query_tokens` (dense /
    /// late-interaction retrievers).
    async fn candidates(
        &self,
        text: &str,
        query_tokens: ArrayView2<'_, f32>,
        k: usize,
    ) -> Result<Vec<String>, MemoryError>;
}

/// Composition of an encoder, a first-stage retriever, and a blob reranker.
///
/// The full pipeline is: `encode → first stage → blob rerank → top-K`.
/// All three components are pluggable; this struct just wires them together
/// and exposes a single `search` entry point.
pub struct ColbertStore {
    encoder: Arc<dyn ColbertEncoder>,
    first_stage: Arc<dyn FirstStageRetriever>,
    reranker: BlobReranker,
    /// Number of candidates to pull from the first stage before reranking.
    /// Empirically validated default: 200 (see design doc).
    pub first_stage_k: usize,
}

impl ColbertStore {
    pub fn new(
        encoder: Arc<dyn ColbertEncoder>,
        first_stage: Arc<dyn FirstStageRetriever>,
        reranker: BlobReranker,
    ) -> Self {
        Self {
            encoder,
            first_stage,
            reranker,
            first_stage_k: 200,
        }
    }

    /// Run the full pipeline. Returns the top `k` reranked documents.
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<ScoredDoc>, MemoryError> {
        let q = self.encoder.encode_query(query).await?;
        let candidates = self
            .first_stage
            .candidates(query, q.view(), self.first_stage_k)
            .await?;
        let mut scored = self.reranker.rerank(q.view(), &candidates).await?;
        scored.truncate(k);
        Ok(scored)
    }

    /// Access the underlying reranker, e.g. to call `rerank()` directly with
    /// candidates from a non-trait first stage in tests or experiments.
    pub fn reranker(&self) -> &BlobReranker {
        &self.reranker
    }

    /// Access the underlying encoder. Useful when you want to time stages
    /// individually or reuse the encoded query for something else.
    pub fn encoder(&self) -> &dyn ColbertEncoder {
        &*self.encoder
    }

    /// Access the underlying first-stage retriever.
    pub fn first_stage(&self) -> &dyn FirstStageRetriever {
        &*self.first_stage
    }
}
