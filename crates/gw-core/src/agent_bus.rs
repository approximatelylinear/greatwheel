//! Inter-agent message bus trait.
//!
//! Migrated from the standalone `gw-bus` crate — the trait lives here
//! in `gw-core` alongside the other core abstractions.

use crate::{AgentId, CallContext, Task};

/// Trait for inter-agent message bus.
#[allow(async_fn_in_trait)]
pub trait AgentBus: Send + Sync {
    /// Synchronous agent call — caller pauses until child completes.
    async fn call(
        &self,
        ctx: &CallContext,
        agent: AgentId,
        task: Task,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>>;

    /// Async fire-and-forget notification — queued as a new task.
    async fn notify(
        &self,
        ctx: &CallContext,
        agent: AgentId,
        message: serde_json::Value,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
