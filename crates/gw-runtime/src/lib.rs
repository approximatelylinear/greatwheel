use gw_core::{AgentId, CallContext, SessionId};

/// Manages ouros sessions — creation, dispatch, snapshot, eviction.
pub struct SessionManager {
    // TODO: session pool, ouros integration
}

impl SessionManager {
    pub fn new() -> Self {
        Self {}
    }
}

/// Trait for the host function bridge — ouros calls pause here.
#[allow(async_fn_in_trait)]
pub trait HostFunctionBridge {
    /// Dispatch a host function call from within an ouros session.
    async fn dispatch(
        &self,
        ctx: &CallContext,
        function: &str,
        args: serde_json::Value,
    ) -> Result<serde_json::Value, Box<dyn std::error::Error + Send + Sync>>;
}
