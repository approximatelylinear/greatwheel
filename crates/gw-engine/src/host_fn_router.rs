//! Composable host function router.
//!
//! Collects host function handlers from plugins and dispatches calls
//! by function name. Falls back to an optional inner bridge for
//! functions not registered via plugins.

use gw_core::HostFnHandler;
use std::collections::HashMap;

/// Routes host function calls to registered plugin handlers.
///
/// Used by the engine to build the `HostBridge` that each REPL session receives.
/// Plugin-registered functions take precedence; unmatched calls fall through
/// to the conversation bridge or return an error.
pub struct HostFnRouter {
    handlers: HashMap<String, HostFnHandler>,
}

impl HostFnRouter {
    pub fn new(handlers: HashMap<String, HostFnHandler>) -> Self {
        Self { handlers }
    }

    /// Look up a handler by function name.
    pub fn get(&self, function: &str) -> Option<&HostFnHandler> {
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
}
