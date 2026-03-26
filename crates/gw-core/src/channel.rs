//! Channel adapter traits.
//!
//! Migrated from the standalone `gw-channels` crate — the traits live here
//! in `gw-core` alongside the other core abstractions.

use crate::{LoopEvent, SessionId, Task, TaskId};

/// A sender for loop events, runtime-agnostic.
///
/// Implementations can wrap `tokio::sync::mpsc::UnboundedSender`,
/// `std::sync::mpsc::Sender`, or any other channel type.
pub type EventSender = Box<dyn Fn(LoopEvent) -> Result<(), LoopEvent> + Send + Sync>;

/// Trait for channel adapters (HTTP, WebSocket, CLI, Slack, etc.).
///
/// Each adapter converts its protocol into LoopEvents and listens
/// for outbound events to send back through its protocol.
#[allow(async_fn_in_trait)]
pub trait ChannelAdapter: Send + Sync {
    /// Unique channel identifier (e.g., "http", "ws", "slack-C04N8BKRM").
    fn channel_id(&self) -> &str;

    /// Start listening for inbound messages. Send LoopEvents via event_tx.
    async fn start(
        &self,
        session_id: SessionId,
        event_tx: EventSender,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Handle an outbound event (Response, InputRequest, TurnComplete, etc.).
    async fn handle_outbound(
        &self,
        event: &LoopEvent,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}

/// Legacy trait for task-based channel adapters.
#[allow(async_fn_in_trait)]
pub trait TaskChannelAdapter: Send + Sync {
    /// Receive inbound tasks from this channel.
    async fn recv(&self) -> Result<Task, Box<dyn std::error::Error + Send + Sync>>;

    /// Send a message back through this channel.
    async fn send(
        &self,
        task_id: TaskId,
        message: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
