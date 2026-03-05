use gw_core::{AgentId, CallContext, SessionId, UserId};
use serde::{Deserialize, Serialize};

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
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    async fn recall(
        &self,
        ctx: &CallContext,
        query: &str,
        opts: RecallOpts,
    ) -> Result<Vec<serde_json::Value>, Box<dyn std::error::Error + Send + Sync>>;

    async fn forget(
        &self,
        ctx: &CallContext,
        key: &str,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Hybrid store implementation backed by LanceDB + Postgres.
pub struct HybridStore {
    // TODO: LanceDB connection, sqlx pool
}

impl HybridStore {
    pub fn new() -> Self {
        Self {}
    }
}
