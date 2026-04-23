use ouros::{
    CollectStringPrint, NoLimitTracker, Object, ReplProgress, ReplSession, ResourceLimits,
    RunProgress, Runner,
};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{info, warn};

/// Errors during agent execution.
#[derive(Debug, thiserror::Error)]
pub enum AgentError {
    #[error("ouros parse error: {0}")]
    Parse(String),
    #[error("ouros runtime error: {0}")]
    Runtime(String),
    #[error("unknown host function: {0}")]
    UnknownFunction(String),
    #[error("host function error in {function}: {message}")]
    HostFunction { function: String, message: String },
    #[error("execution limit exceeded")]
    LimitExceeded,
}

/// Result of running an agent to completion.
#[derive(Debug)]
pub struct AgentResult {
    /// The final return value from the agent code (converted to JSON).
    pub value: Value,
    /// All print output captured during execution.
    pub stdout: String,
}

/// Trait for resolving host function calls from within ouros.
pub trait HostBridge: Send {
    fn call(
        &mut self,
        function: &str,
        args: Vec<Value>,
        kwargs: HashMap<String, Value>,
    ) -> Result<Object, AgentError>;
}

/// Convert a serde_json::Value into an ouros Object.
pub fn json_to_object(val: Value) -> Object {
    Object::from_json_value(val)
}

/// Convert an ouros Object into a serde_json::Value.
pub fn object_to_json(obj: &Object) -> Value {
    obj.to_json_value()
}

/// Extract positional args from ouros Objects to JSON Values.
fn args_to_json(args: &[Object]) -> Vec<Value> {
    args.iter().map(|o| o.to_json_value()).collect()
}

/// Extract keyword args from ouros Object pairs to a HashMap.
fn kwargs_to_map(kwargs: &[(Object, Object)]) -> HashMap<String, Value> {
    kwargs
        .iter()
        .filter_map(|(k, v)| {
            if let Object::String(key) = k {
                Some((key.clone(), v.to_json_value()))
            } else {
                None
            }
        })
        .collect()
}

fn runtime_error(msg: String) -> ouros::Exception {
    ouros::Exception::new(ouros::ExcType::RuntimeError, Some(msg))
}

fn permission_error(msg: String) -> ouros::Exception {
    ouros::Exception::new(ouros::ExcType::PermissionError, Some(msg))
}

// ─── Single-script agent execution ─────────────────────────────────────────

/// Run a complete Python script inside ouros, dispatching external function calls
/// through the provided `HostBridge`.
#[tracing::instrument(name = "invoke_agent", skip(code, inputs, bridge))]
pub fn run_agent(
    code: &str,
    inputs: Vec<(String, Object)>,
    external_functions: Vec<String>,
    bridge: &mut dyn HostBridge,
) -> Result<AgentResult, AgentError> {
    let input_names: Vec<String> = inputs.iter().map(|(n, _)| n.clone()).collect();
    let input_values: Vec<Object> = inputs.into_iter().map(|(_, v)| v).collect();

    let runner = Runner::new(code.to_owned(), "agent.py", input_names, external_functions)
        .map_err(|e| AgentError::Parse(format!("{e}")))?;

    let mut print_buf = CollectStringPrint::new();

    let mut progress = runner
        .start(input_values, NoLimitTracker, &mut print_buf)
        .map_err(|e| AgentError::Runtime(format!("{e}")))?;

    loop {
        match progress {
            RunProgress::Complete(obj) => {
                let value = object_to_json(&obj);
                info!("agent completed");
                return Ok(AgentResult {
                    value,
                    stdout: print_buf.output().to_string(),
                });
            }
            RunProgress::FunctionCall {
                function_name,
                args,
                kwargs,
                call_id: _,
                state,
            } => {
                let _span =
                    tracing::info_span!("host_function", function = %function_name).entered();
                let json_args = args_to_json(&args);
                let json_kwargs = kwargs_to_map(&kwargs);

                let result = bridge.call(&function_name, json_args, json_kwargs);
                match result {
                    Ok(return_value) => {
                        progress = state
                            .run(return_value, &mut print_buf)
                            .map_err(|e| AgentError::Runtime(format!("{e}")))?;
                    }
                    Err(e) => {
                        warn!(function = %function_name, error = %e, "host function failed");
                        progress = state
                            .run(runtime_error(format!("{e}")), &mut print_buf)
                            .map_err(|e| AgentError::Runtime(format!("{e}")))?;
                    }
                }
            }
            RunProgress::OsCall { state, .. } => {
                warn!("agent attempted OS call — denied");
                progress = state
                    .run(
                        permission_error("OS calls are not permitted in agent sandboxes".into()),
                        &mut print_buf,
                    )
                    .map_err(|e| AgentError::Runtime(format!("{e}")))?;
            }
            RunProgress::ResolveFutures(_) => {
                return Err(AgentError::Runtime(
                    "async futures not supported in sync agent execution".into(),
                ));
            }
        }
    }
}

// ─── rLM-style REPL agent ──────────────────────────────────────────────────

/// Result of executing a single code block in the REPL.
#[derive(Debug)]
pub struct ReplExecResult {
    /// The return value of the code block (last expression).
    pub value: Value,
    /// Captured print output from this execution.
    pub stdout: String,
    /// Whether a FINAL() call was made.
    pub is_final: bool,
    /// The FINAL() argument if is_final is true.
    pub final_value: Option<Value>,
}

/// Persistent REPL agent session backed by ouros::ReplSession.
///
/// Supports the rLM pattern: the Rust host drives the loop, sending
/// LLM-generated code blocks to this session for execution. External
/// functions (search, llm_query, etc.) pause execution and get resolved
/// by the host bridge.
pub struct ReplAgent {
    session: ReplSession,
    bridge: Box<dyn HostBridge>,
    final_value: Option<Object>,
}

impl ReplAgent {
    /// Create a new REPL agent with the given external functions.
    pub fn new(external_functions: Vec<String>, bridge: Box<dyn HostBridge>) -> Self {
        let session = ReplSession::new(external_functions, "agent.py");
        Self {
            session,
            bridge,
            final_value: None,
        }
    }

    /// Inject a variable into the REPL session.
    pub fn set_variable(&mut self, name: &str, value: Object) -> Result<(), AgentError> {
        self.session
            .set_variable(name, value)
            .map_err(|e| AgentError::Runtime(format!("set_variable failed: {e}")))
    }

    /// Read a variable from the REPL session.
    pub fn get_variable(&self, name: &str) -> Option<Object> {
        self.session.get_variable(name)
    }

    /// Whether a FINAL() call has been made.
    pub fn is_finished(&self) -> bool {
        self.final_value.is_some()
    }

    /// Get the final value if FINAL() was called.
    pub fn final_value(&self) -> Option<Value> {
        self.final_value.as_ref().map(object_to_json)
    }

    /// Summarize REPL state as "name: type = value" lines for scalars,
    /// "name: type (N items)" for collections, "name: type" for complex objects.
    pub fn state_summary(&self) -> String {
        self.session
            .list_variables()
            .into_iter()
            .map(|(name, ty)| {
                if let Some(obj) = self.session.get_variable(&name) {
                    let val = object_to_json(&obj);
                    match &val {
                        Value::Number(_) | Value::Bool(_) | Value::Null => {
                            format!("{name}: {ty} = {val}")
                        }
                        Value::String(s) if s.len() <= 80 => {
                            format!("{name}: {ty} = {val}")
                        }
                        Value::String(s) => {
                            format!("{name}: {ty} ({} chars)", s.len())
                        }
                        Value::Array(a) => {
                            if a.len() <= 5 {
                                format!("{name}: {ty} = {val}")
                            } else {
                                format!("{name}: {ty} ({} items)", a.len())
                            }
                        }
                        Value::Object(m) => {
                            let keys: Vec<&str> = m.keys().map(|k| k.as_str()).collect();
                            if keys.len() <= 5 {
                                format!("{name}: {ty} keys={keys:?}")
                            } else {
                                format!("{name}: {ty} ({} keys)", keys.len())
                            }
                        }
                    }
                } else {
                    format!("{name}: {ty}")
                }
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// Get all variables as a JSON map.
    pub fn get_all_variables(&self) -> HashMap<String, Value> {
        self.session
            .list_variables()
            .into_iter()
            .filter_map(|(name, _)| {
                self.session
                    .get_variable(&name)
                    .map(|obj| (name, object_to_json(&obj)))
            })
            .collect()
    }

    /// Serialize the REPL session to bytes for snapshot/restore.
    pub fn save_snapshot(&self) -> Result<Vec<u8>, AgentError> {
        self.session
            .save()
            .map_err(|e| AgentError::Runtime(format!("save_snapshot failed: {e}")))
    }

    /// Restore a REPL agent from a snapshot.
    pub fn restore_snapshot(bytes: &[u8], bridge: Box<dyn HostBridge>) -> Result<Self, AgentError> {
        let session = ReplSession::load(bytes, ResourceLimits::new())
            .map_err(|e| AgentError::Runtime(format!("restore_snapshot failed: {e}")))?;
        Ok(Self {
            session,
            bridge,
            final_value: None,
        })
    }

    /// Get names of defined functions in the REPL.
    pub fn get_definitions(&self) -> Vec<String> {
        self.session
            .list_variables()
            .into_iter()
            .filter(|(_, ty)| ty == "function")
            .map(|(name, _)| name)
            .collect()
    }

    /// Execute a code block in the REPL session.
    ///
    /// External function calls are resolved synchronously through the bridge.
    /// If the code calls FINAL(value), sets `is_final = true`.
    #[tracing::instrument(
        name = "repl.execute",
        skip(self, code),
        fields(gw.code_length = code.len(), gw.is_final = tracing::field::Empty)
    )]
    pub fn execute(&mut self, code: &str) -> Result<ReplExecResult, AgentError> {
        let mut print_buf = CollectStringPrint::new();

        // Reset per-call FINAL tracking. `self.final_value` is cumulative
        // agent state for snapshot purposes, but `ReplExecResult::is_final`
        // is per-block: "did THIS code block call FINAL()?" — which is
        // what `ConversationLoop::handle_turn` relies on. Without this
        // reset, turn 2+ see `is_final=true` forever because turn 1 left
        // a final_value set.
        self.final_value = None;

        let mut progress = self
            .session
            .execute_interactive(code, &mut print_buf)
            .map_err(|e| AgentError::Runtime(format!("{e}")))?;

        loop {
            match progress {
                ReplProgress::Complete(obj) => {
                    let stdout = print_buf.output().to_string();
                    let is_final = self.final_value.is_some();
                    let final_val = self.final_value.as_ref().map(object_to_json);
                    return Ok(ReplExecResult {
                        value: object_to_json(&obj),
                        stdout,
                        is_final,
                        final_value: final_val,
                    });
                }
                ReplProgress::FunctionCall {
                    function_name,
                    args,
                    kwargs,
                    call_id: _,
                } => {
                    // Intercept FINAL() calls
                    if function_name == "FINAL" {
                        let final_obj = args.into_iter().next().unwrap_or(Object::None);
                        self.final_value = Some(final_obj.clone());
                        tracing::Span::current().record("gw.is_final", true);
                        // Resume with the value itself so the code continues
                        progress = self
                            .session
                            .resume(final_obj, &mut print_buf)
                            .map_err(|e| AgentError::Runtime(format!("{e}")))?;
                        continue;
                    }

                    let _span =
                        tracing::info_span!("host_function", function = %function_name).entered();
                    let json_args = args_to_json(&args);
                    let json_kwargs = kwargs_to_map(&kwargs);

                    let result = self.bridge.call(&function_name, json_args, json_kwargs);
                    match result {
                        Ok(return_value) => {
                            progress = self
                                .session
                                .resume(return_value, &mut print_buf)
                                .map_err(|e| AgentError::Runtime(format!("{e}")))?;
                        }
                        Err(e) => {
                            warn!(function = %function_name, error = %e, "host function failed");
                            progress = self
                                .session
                                .resume(runtime_error(format!("{e}")), &mut print_buf)
                                .map_err(|e| AgentError::Runtime(format!("{e}")))?;
                        }
                    }
                }
                ReplProgress::ProxyCall { call_id: _, .. } => {
                    progress = self
                        .session
                        .resume(
                            runtime_error("proxy calls not supported".into()),
                            &mut print_buf,
                        )
                        .map_err(|e| AgentError::Runtime(format!("{e}")))?;
                }
                ReplProgress::ResolveFutures { .. } => {
                    return Err(AgentError::Runtime(
                        "async futures not supported in REPL agent".into(),
                    ));
                }
            }
        }
    }
}

/// Extract ```repl``` or ```python``` code blocks from LLM output.
pub fn extract_code_blocks(text: &str) -> Vec<String> {
    let mut blocks = Vec::new();
    let mut in_block = false;
    let mut current_block = String::new();

    for line in text.lines() {
        let trimmed = line.trim();
        if !in_block {
            if trimmed.starts_with("```repl")
                || trimmed.starts_with("```python")
                || trimmed == "```" && blocks.is_empty() && !current_block.is_empty()
            {
                in_block = true;
                current_block.clear();
            }
        } else if trimmed == "```" {
            if !current_block.trim().is_empty() {
                blocks.push(current_block.trim().to_string());
            }
            current_block.clear();
            in_block = false;
        } else {
            current_block.push_str(line);
            current_block.push('\n');
        }
    }

    blocks
}

/// Check if LLM output contains a FINAL() or FINAL_VAR() directive outside code blocks.
/// Skips content inside ```...``` fenced blocks to avoid capturing FINAL(variable_name)
/// from code that should be executed instead.
pub fn extract_final_answer(text: &str) -> Option<String> {
    let mut in_code_block = false;
    for line in text.lines() {
        let trimmed = line.trim();
        // Track fenced code blocks
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            continue;
        }
        if in_code_block {
            continue;
        }
        // FINAL("literal answer") outside of code blocks — only match quoted strings
        if let Some(rest) = trimmed.strip_prefix("FINAL(") {
            if let Some(answer) = rest.strip_suffix(')') {
                let clean = answer.trim().trim_matches('"').trim_matches('\'');
                // Only accept if the argument was quoted (literal string, not a variable name)
                if answer.trim().starts_with('"') || answer.trim().starts_with('\'') {
                    return Some(clean.to_string());
                }
                // Unquoted — likely a variable reference, skip (let code execution handle it)
            }
        }
        // FINAL_VAR(variable_name) — handled by the caller reading from session
        if trimmed.starts_with("FINAL_VAR(") {
            return Some(trimmed.to_string());
        }
    }
    None
}

/// Strip `<think>...</think>` blocks from LLM output.
/// Used for thinking models (qwen3.5, etc.) that wrap reasoning in think tags.
pub fn strip_think_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            rest = &rest[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> — skip everything after it.
            return result.trim().to_string();
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}
