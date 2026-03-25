use gw_runtime::{AgentError, HostBridge};
use ouros::Object;
use serde_json::Value;
use std::collections::HashMap;

/// Records host calls and returns Object::None for any function.
pub struct MockBridge {
    pub calls: Vec<MockCall>,
}

#[derive(Debug, Clone)]
pub struct MockCall {
    pub function: String,
    pub args: Vec<Value>,
    pub kwargs: HashMap<String, Value>,
}

impl MockBridge {
    pub fn new() -> Self {
        Self { calls: Vec::new() }
    }
}

impl HostBridge for MockBridge {
    fn call(
        &mut self,
        function: &str,
        args: Vec<Value>,
        kwargs: HashMap<String, Value>,
    ) -> Result<Object, AgentError> {
        self.calls.push(MockCall {
            function: function.to_string(),
            args,
            kwargs,
        });
        Ok(Object::None)
    }
}
