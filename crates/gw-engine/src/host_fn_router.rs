//! Composable host function router.
//!
//! Collects host function handlers from plugins and dispatches calls
//! by function name. Falls back to an optional inner bridge for
//! functions not registered via plugins.

use gw_core::{HostFnHandler, HostFnRegistration, PluginError};
use serde_json::Value;
use std::collections::HashMap;

/// Routes host function calls to registered plugin handlers.
///
/// Built once at engine init from every plugin's `register_host_fn*`
/// calls. An `Arc<HostFnRouter>` is threaded into each
/// `ConversationBridge` so agent host function calls can be resolved
/// through the plugin system before the bridge's hardcoded matches
/// or fallback `inner` bridge.
///
/// ### Dispatch and the sync/async bridge
///
/// `HostBridge::call()` is synchronous (ouros callbacks are sync), but
/// handlers can be either sync or async. [`dispatch`] bridges the gap:
/// sync handlers run inline, async handlers are resolved via
/// `tokio::task::block_in_place` + `Handle::current().block_on(...)`.
///
/// This is safe **only inside a multi-threaded tokio runtime**
/// (`#[tokio::main]` or `#[tokio::test(flavor = "multi_thread")]`).
/// Current-thread runtimes will panic at the `block_in_place` call.
/// `gw-server` uses multi-threaded tokio, so production is fine; tests
/// that exercise async handlers must specify the multi-thread flavor.
pub struct HostFnRouter {
    handlers: HashMap<String, HostFnRegistration>,
}

impl HostFnRouter {
    pub fn new(handlers: HashMap<String, HostFnRegistration>) -> Self {
        Self { handlers }
    }

    /// Look up a registration by function name.
    pub fn get(&self, function: &str) -> Option<&HostFnRegistration> {
        self.handlers.get(function)
    }

    /// List all registered function names.
    pub fn function_names(&self) -> Vec<&str> {
        self.handlers.keys().map(|s| s.as_str()).collect()
    }

    /// Check if a function is registered.
    pub fn has(&self, function: &str) -> bool {
        self.handlers.contains_key(function)
    }

    /// Dispatch a host function call by name.
    ///
    /// Returns:
    ///   - `None` if the function is not registered (caller should
    ///     fall through to its own dispatch)
    ///   - `Some(Ok(value))` if the handler ran and returned a value
    ///   - `Some(Err(e))` if the handler ran and returned an error
    ///
    /// Sync handlers run inline. Async handlers are resolved via
    /// `block_in_place` + `Handle::current().block_on(...)`. See the
    /// struct docs for the runtime requirement.
    pub fn dispatch(
        &self,
        function: &str,
        args: Vec<Value>,
        kwargs: HashMap<String, Value>,
    ) -> Option<Result<Value, PluginError>> {
        let registration = self.handlers.get(function)?;
        // TODO(capability): once enforcement lands, check
        // `registration.capability` against the active agent's grants
        // before invoking the handler.
        let result = match &registration.handler {
            HostFnHandler::Sync(f) => f(args, kwargs),
            HostFnHandler::Async(f) => {
                let fut = f(args, kwargs);
                tokio::task::block_in_place(|| tokio::runtime::Handle::current().block_on(fut))
            }
        };
        Some(result)
    }
}
