use gw_core::EntryId;
use gw_runtime::AgentError;

/// Errors from the conversation loop.
#[derive(Debug, thiserror::Error)]
pub enum LoopError {
    #[error("LLM error: {0}")]
    Llm(String),
    #[error("agent error: {0}")]
    Agent(#[from] AgentError),
    #[error("channel closed")]
    ChannelClosed,
    #[error("session ended")]
    SessionEnded,
    #[error("entry not found: {0:?}")]
    EntryNotFound(EntryId),
    #[error("no snapshot available for branch restore")]
    NoSnapshot,
    #[error("nothing to compact")]
    NothingToCompact,
}
