use gw_runtime::{AgentError, HostBridge};
use ouros::Object;
use serde_json::Value;
use std::collections::HashMap;

/// Returns Object::None for any function call.
pub struct MockBridge;

impl HostBridge for MockBridge {
    fn call(
        &mut self,
        _function: &str,
        _args: Vec<Value>,
        _kwargs: HashMap<String, Value>,
    ) -> Result<Object, AgentError> {
        Ok(Object::None)
    }
}
