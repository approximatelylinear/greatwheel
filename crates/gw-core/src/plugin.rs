//! Plugin trait and supporting types for the Greatwheel plugin framework.
//!
//! Plugins register capabilities during `init()` via the `PluginContext` API.
//! The engine collects registrations and wires them into the runtime.

use serde_json::Value;
use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;

use crate::SessionId;

// ─── Plugin trait ─────────────────────────────────────────────────────────────

/// A Greatwheel plugin.
///
/// Plugins are the primary extension point. During `init`, a plugin registers
/// capabilities (LLM backends, memory stores, channels, host functions,
/// lifecycle event handlers, HTTP routes) via the `PluginContext`.
pub trait Plugin: Send + Sync + 'static {
    /// Unique name (e.g., "slack-channel", "pinecone-memory").
    fn name(&self) -> &str;

    /// Semantic version of this plugin.
    fn version(&self) -> &str {
        "0.1.0"
    }

    /// Declare capabilities and dependencies.
    fn manifest(&self) -> PluginManifest {
        PluginManifest::default()
    }

    /// Called once at startup. Register capabilities via `ctx`.
    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError>;

    /// Called on graceful shutdown. Flush state, close connections.
    fn shutdown(&self) -> Result<(), PluginError> {
        Ok(())
    }
}

// ─── Manifest ─────────────────────────────────────────────────────────────────

/// Declares what a plugin provides and requires.
#[derive(Debug, Clone, Default)]
pub struct PluginManifest {
    /// Capabilities this plugin provides (e.g., "llm:openai", "channel:slack").
    pub provides: Vec<String>,
    /// Capabilities this plugin requires (e.g., "memory", "llm").
    pub requires: Vec<String>,
    /// Load order hint — lower runs first (default 100).
    pub priority: u32,
}

// ─── Lifecycle events ─────────────────────────────────────────────────────────

/// Lifecycle events that plugins can subscribe to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LifecycleEvent {
    // Server lifecycle
    BeforeStartup,
    AfterStartup,
    BeforeShutdown,

    // Session lifecycle
    SessionCreated,
    SessionResumed,
    SessionEvicted,
    SessionEnded,

    // Turn lifecycle (the rLM loop)
    BeforeTurn,
    AfterContextBuild,
    BeforeLlmCall,
    AfterLlmCall,
    BeforeCodeExec,
    AfterCodeExec,
    BeforeHostCall,
    AfterHostCall,
    AfterTurn,
    TurnError,

    // Memory lifecycle
    BeforeMemoryStore,
    AfterMemoryRecall,

    // Agent lifecycle
    AgentSpawned,
    AgentCompleted,
}

/// Result of handling a lifecycle event.
pub enum EventResult {
    /// Proceed normally.
    Continue,
    /// Payload was modified, proceed with changes.
    Modified,
    /// Skip remaining processing, use this value as the result.
    ShortCircuit(Value),
}

/// Typed payloads for built-in lifecycle events.
pub enum EventData {
    /// Server lifecycle events (BeforeStartup, AfterStartup, BeforeShutdown).
    Server,

    /// Session lifecycle events.
    Session {
        session_id: SessionId,
    },

    /// BeforeTurn — user message received.
    Turn {
        session_id: SessionId,
        message: String,
    },

    /// AfterContextBuild / BeforeLlmCall — LLM messages assembled.
    Messages {
        session_id: SessionId,
        messages: Vec<LlmMessageData>,
    },

    /// AfterLlmCall — raw LLM response.
    LlmResponse {
        session_id: SessionId,
        content: String,
        input_tokens: Option<u32>,
        output_tokens: Option<u32>,
    },

    /// BeforeCodeExec — extracted code about to execute.
    Code {
        session_id: SessionId,
        code: String,
    },

    /// AfterCodeExec — REPL execution result.
    ExecResult {
        session_id: SessionId,
        value: Value,
        stdout: String,
        is_final: bool,
    },

    /// BeforeHostCall / AfterHostCall — host function invocation.
    HostCall {
        session_id: SessionId,
        function: String,
        args: Vec<Value>,
    },

    /// AfterTurn — turn complete.
    TurnComplete {
        session_id: SessionId,
        response: String,
        iterations: usize,
        input_tokens: u32,
        output_tokens: u32,
    },

    /// TurnError — turn failed.
    Error {
        session_id: SessionId,
        error: String,
    },

    /// Memory events.
    Memory {
        key: String,
        value: Option<Value>,
    },

    /// Plugin-defined custom events.
    Custom(Box<dyn Any + Send>),
}

impl fmt::Debug for EventData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Server => write!(f, "Server"),
            Self::Session { session_id } => f.debug_struct("Session").field("session_id", session_id).finish(),
            Self::Turn { session_id, .. } => f.debug_struct("Turn").field("session_id", session_id).finish(),
            Self::Messages { session_id, messages } => f.debug_struct("Messages").field("session_id", session_id).field("count", &messages.len()).finish(),
            Self::LlmResponse { session_id, input_tokens, output_tokens, .. } => f.debug_struct("LlmResponse").field("session_id", session_id).field("input_tokens", input_tokens).field("output_tokens", output_tokens).finish(),
            Self::Code { session_id, .. } => f.debug_struct("Code").field("session_id", session_id).finish(),
            Self::ExecResult { session_id, is_final, .. } => f.debug_struct("ExecResult").field("session_id", session_id).field("is_final", is_final).finish(),
            Self::HostCall { session_id, function, .. } => f.debug_struct("HostCall").field("session_id", session_id).field("function", function).finish(),
            Self::TurnComplete { session_id, iterations, input_tokens, output_tokens, .. } => f.debug_struct("TurnComplete").field("session_id", session_id).field("iterations", iterations).field("input_tokens", input_tokens).field("output_tokens", output_tokens).finish(),
            Self::Error { session_id, error } => f.debug_struct("Error").field("session_id", session_id).field("error", error).finish(),
            Self::Memory { key, .. } => f.debug_struct("Memory").field("key", key).finish(),
            Self::Custom(_) => write!(f, "Custom(..)"),
        }
    }
}

/// Simplified message representation for event payloads.
#[derive(Debug, Clone)]
pub struct LlmMessageData {
    pub role: String,
    pub content: String,
}

/// Payload passed to event handlers.
pub struct EventPayload {
    pub event: LifecycleEvent,
    pub data: EventData,
}

/// A sync event handler function.
pub type EventHandler = Arc<dyn Fn(&mut EventPayload) -> EventResult + Send + Sync>;

// ─── Plugin context ───────────────────────────────────────────────────────────

/// Typed map for sharing state between plugins.
#[derive(Default)]
pub struct SharedState {
    map: HashMap<TypeId, Box<dyn Any + Send + Sync>>,
}

impl SharedState {
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T> {
        self.map
            .get(&TypeId::of::<T>())
            .and_then(|v| v.downcast_ref())
    }

    pub fn insert<T: Send + Sync + 'static>(&mut self, value: T) {
        self.map.insert(TypeId::of::<T>(), Box::new(value));
    }
}

/// Registration context passed to `Plugin::init()`.
///
/// Provides read/write access to shared state (so Plugin B can read what
/// Plugin A provided) and methods to register capabilities.
pub struct PluginContext<'a> {
    /// This plugin's config section from TOML.
    pub config: &'a Value,

    /// Shared state — readable and writable. Plugins initialized earlier
    /// have already inserted their values; this plugin can read those
    /// and insert its own for downstream plugins.
    pub shared: &'a mut SharedState,

    /// Collected registrations.
    pub(crate) registrations: &'a mut PluginRegistrations,
}

impl<'a> PluginContext<'a> {
    pub fn new(
        config: &'a Value,
        shared: &'a mut SharedState,
        registrations: &'a mut PluginRegistrations,
    ) -> Self {
        Self {
            config,
            shared,
            registrations,
        }
    }

    /// Subscribe to a lifecycle event.
    pub fn on(&mut self, event: LifecycleEvent, handler: EventHandler) {
        self.registrations
            .event_handlers
            .entry(event)
            .or_default()
            .push(handler);
    }

    /// Register a host function callable from the Python REPL.
    pub fn register_host_fn(&mut self, name: &str, handler: HostFnHandler) {
        self.registrations
            .host_functions
            .insert(name.to_string(), handler);
    }

    /// Expose a typed value for downstream plugins to access.
    pub fn provide<T: Send + Sync + 'static>(&mut self, value: T) {
        self.shared.insert(value);
    }
}

/// Collected registrations from a single plugin's init().
#[derive(Default)]
pub struct PluginRegistrations {
    pub event_handlers: HashMap<LifecycleEvent, Vec<EventHandler>>,
    pub host_functions: HashMap<String, HostFnHandler>,
}

// ─── Host function types ──────────────────────────────────────────────────────

/// A registered host function handler.
///
/// Takes positional args, keyword args, and returns a JSON value or error.
/// Sync for now — the handler runs on the REPL execution thread.
pub type HostFnHandler = Arc<
    dyn Fn(Vec<Value>, HashMap<String, Value>) -> Result<Value, PluginError> + Send + Sync,
>;

// ─── Errors ───────────────────────────────────────────────────────────────────

/// Errors from plugin operations.
#[derive(Debug, thiserror::Error)]
pub enum PluginError {
    #[error("plugin init failed: {0}")]
    Init(String),

    #[error("missing required capability: {0}")]
    MissingCapability(String),

    #[error("plugin config error: {0}")]
    Config(String),

    #[error("host function error: {0}")]
    HostFunction(String),

    #[error("plugin panicked: {0}")]
    Panic(String),

    #[error("{0}")]
    Other(String),
}

impl fmt::Display for PluginManifest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "provides={:?}, requires={:?}, priority={}",
            self.provides, self.requires, self.priority
        )
    }
}
