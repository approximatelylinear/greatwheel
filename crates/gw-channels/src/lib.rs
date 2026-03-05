use gw_core::Task;

/// Trait for channel adapters (HTTP, WebSocket, CLI, Slack, etc.).
#[allow(async_fn_in_trait)]
pub trait ChannelAdapter {
    /// Receive inbound tasks from this channel.
    async fn recv(&self) -> Result<Task, Box<dyn std::error::Error + Send + Sync>>;

    /// Send a message back through this channel.
    async fn send(
        &self,
        task_id: gw_core::TaskId,
        message: String,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;
}
