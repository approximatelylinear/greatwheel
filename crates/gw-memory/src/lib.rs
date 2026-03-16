pub mod corpus;
pub mod error;
pub mod fusion;
pub mod hybrid;
pub mod lance;
pub mod postgres;
pub mod tantivy_store;

use gw_core::{AgentId, CallContext, SessionId, UserId};
use serde::{Deserialize, Serialize};

pub use error::MemoryError;
pub use hybrid::HybridStore;

/// A memory record returned from recall queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryRecord {
    pub key: String,
    pub value: serde_json::Value,
    pub score: f32,
}

/// Options for memory recall queries.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecallOpts {
    pub top_k: usize,
    pub mode: SearchMode,
    pub scope: MemoryScope,
}

/// How to search memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SearchMode {
    Vector,
    FullText,
    Hybrid { alpha: f32 },
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
    async fn store(
        &self,
        ctx: &CallContext,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), MemoryError>;

    async fn recall(
        &self,
        ctx: &CallContext,
        query: &str,
        opts: RecallOpts,
    ) -> Result<Vec<MemoryRecord>, MemoryError>;

    async fn forget(
        &self,
        ctx: &CallContext,
        key: &str,
    ) -> Result<(), MemoryError>;
}

impl MemoryStore for HybridStore {
    async fn store(
        &self,
        ctx: &CallContext,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), MemoryError> {
        self.store(ctx, key, value).await
    }

    async fn recall(
        &self,
        ctx: &CallContext,
        query: &str,
        opts: RecallOpts,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        self.recall(ctx, query, opts).await
    }

    async fn forget(
        &self,
        ctx: &CallContext,
        key: &str,
    ) -> Result<(), MemoryError> {
        self.forget(ctx, key).await
    }
}
