pub mod colbert;
pub mod colbert_blobs;
pub mod corpus;
pub mod error;
pub mod fusion;
pub mod graph;
pub mod hybrid;
pub mod lance;
pub mod postgres;
pub mod tantivy_store;
pub mod temporal;

use chrono::{DateTime, Utc};
use gw_core::{AgentId, CallContext, MemoryKind, SessionId, UserId};
use serde::{Deserialize, Serialize};

pub use error::MemoryError;
pub use hybrid::HybridStore;

/// A memory record returned from recall queries.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub key: String,
    pub value: serde_json::Value,
    pub score: f32,
    /// Hindsight memory type classification.
    #[serde(default)]
    pub kind: MemoryKind,
    /// Confidence score (only meaningful for `Opinion` kind).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f32>,
    /// When the fact occurred (distinct from when it was stored).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurred_at: Option<DateTime<Utc>>,
    /// End of occurrence interval (None = point event).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub occurred_end: Option<DateTime<Utc>>,
    /// Canonical entity names extracted from this memory.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub entities: Vec<String>,
}

/// Metadata for storing a memory (input to store operations).
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct MemoryMeta {
    pub kind: MemoryKind,
    pub confidence: Option<f32>,
    pub occurred_at: Option<DateTime<Utc>>,
    pub occurred_end: Option<DateTime<Utc>>,
    pub entities: Vec<String>,
}

/// Options for memory recall queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallOpts {
    pub top_k: usize,
    pub mode: SearchMode,
    pub scope: MemoryScope,
    /// Optional: filter by memory kind.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub kind_filter: Option<MemoryKind>,
    /// Optional: exclude opinion memories with confidence below this threshold.
    /// Default: None (no filtering). Typical value: 0.3.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence_threshold: Option<f32>,
}

/// How to search memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    Vector,
    FullText,
    Hybrid {
        alpha: f32,
    },
    /// Four-channel retrieval: vector + BM25 + graph + temporal, fused via RRF.
    Full {
        /// Max graph traversal depth (default 2).
        #[serde(default = "default_graph_hops")]
        graph_hops: usize,
        /// Activation decay per hop (default 0.5).
        #[serde(default = "default_graph_decay")]
        graph_decay: f32,
        /// Recency decay sigma in days when no temporal expression detected (default 7).
        #[serde(default = "default_recency_sigma")]
        recency_sigma_days: f64,
    },
}

fn default_graph_hops() -> usize {
    2
}
fn default_graph_decay() -> f32 {
    0.5
}
fn default_recency_sigma() -> f64 {
    7.0
}

/// Scope of memory search.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MemoryScope {
    Org,
    User(UserId),
    Agent(AgentId),
    Session(SessionId),
}

/// Trait for the hybrid memory store (LanceDB + Postgres).
#[allow(async_fn_in_trait)]
pub trait MemoryStore {
    /// Store a memory with optional Hindsight metadata.
    async fn store(
        &self,
        ctx: &CallContext,
        key: &str,
        value: serde_json::Value,
        meta: Option<MemoryMeta>,
    ) -> Result<(), MemoryError>;

    async fn recall(
        &self,
        ctx: &CallContext,
        query: &str,
        opts: RecallOpts,
    ) -> Result<Vec<MemoryRecord>, MemoryError>;

    async fn forget(&self, ctx: &CallContext, key: &str) -> Result<(), MemoryError>;
}

impl MemoryStore for HybridStore {
    async fn store(
        &self,
        ctx: &CallContext,
        key: &str,
        value: serde_json::Value,
        meta: Option<MemoryMeta>,
    ) -> Result<(), MemoryError> {
        self.store(ctx, key, value, meta).await
    }

    async fn recall(
        &self,
        ctx: &CallContext,
        query: &str,
        opts: RecallOpts,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        self.recall(ctx, query, opts).await
    }

    async fn forget(&self, ctx: &CallContext, key: &str) -> Result<(), MemoryError> {
        self.forget(ctx, key).await
    }
}
