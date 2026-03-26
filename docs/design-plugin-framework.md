# Design: Plugin & Lifecycle Framework

**Status:** In progress (Phases 1-2 implemented)
**Date:** 2026-03-26

---

## 0. Implementation Status

Phases 1 and 2 from the implementation plan are complete. The foundational
types and engine crate are implemented and wired into the server.

| Component | Status | Location |
|-----------|--------|----------|
| `Plugin` trait (sync) | Done | `gw-core/src/plugin.rs` |
| `PluginManifest` | Done | `gw-core/src/plugin.rs` |
| `PluginContext` | Partial | `gw-core/src/plugin.rs` — `on()`, `register_host_fn()`, `provide()` implemented; `register_llm_backend()`, `register_memory_store()`, `register_channel()`, `register_bus()`, `register_routes()` deferred to Phase 3 |
| `LifecycleEvent` (23 variants) | Done | `gw-core/src/plugin.rs` |
| `EventData` (typed enum) | Done | `gw-core/src/plugin.rs` |
| `SharedState` | Done | `gw-core/src/plugin.rs` |
| `AgentBus` trait | Done (migrated) | `gw-core/src/agent_bus.rs` |
| `ChannelAdapter` trait | Done (migrated) | `gw-core/src/channel.rs` |
| `PluginRegistry` | Done | `gw-engine/src/registry.rs` |
| `EventDispatcher` | Done | `gw-engine/src/dispatcher.rs` |
| `HostFnRouter` | Done | `gw-engine/src/host_fn_router.rs` |
| `GreatWheelEngine` | Done | `gw-engine/src/engine.rs` |
| Server integration | Done | `gw-server/src/main.rs` — init, lifecycle events, shutdown |
| Built-in plugins (Ollama, HybridStore, OTel) | Not started | Phase 3 |
| Event dispatch in `ConversationLoop` | Not started | Phase 4 |

---

## 0b. Motivation

Greatwheel is currently a single integrated system — every component (LLM,
memory, channels, tracing) is wired together in `gw-server/main.rs`. This
works well for our own use but makes it hard for others to build on top of.

To become a framework, we need to answer one question: *how does someone
use Greatwheel without forking it?*

The answer is a plugin system that lets external code:
- Replace or compose core subsystems (LLM backends, memory stores, channels)
- Extend the host function bridge with new capabilities
- Hook into lifecycle events at every stage of the rLM loop
- Expose custom HTTP routes alongside the built-in API
- Bring their own configuration, initialized from TOML

---

## 1. Design Principles

**Registration over declaration.** Plugins push capabilities into a registry
during init. The core never needs to know what plugins exist at compile time.
This avoids a god-trait with 20 optional methods.

**Events over middleware chains.** Lifecycle hooks are discrete named events
with typed payloads. Handlers can inspect, modify, or short-circuit. No need
to reason about middleware ordering — events have natural ordering from the
rLM loop structure itself.

**Composition over replacement.** A plugin that registers a memory store
doesn't delete the existing one — it composes. The framework supports
stacking (e.g., a caching layer in front of vector search). Replacement is
opt-in via explicit config.

**Compiled-in first, dynamic later.** Plugins are Rust crates added as
dependencies. No dynamic loading, no ABI concerns. A `dyn Plugin` trait
object is the extension point. Dynamic `.so` loading can come later once
APIs stabilize.

---

## 2. The Plugin Trait

A plugin is a struct implementing one trait in `gw-core`:

```rust
// gw-core/src/plugin.rs

pub trait Plugin: Send + Sync + 'static {
    /// Unique name (e.g., "slack-channel", "pinecone-memory")
    fn name(&self) -> &str;

    /// Semantic version of this plugin
    fn version(&self) -> &str { "0.1.0" }

    /// Declare capabilities and dependencies
    fn manifest(&self) -> PluginManifest { PluginManifest::default() }

    /// Called once at startup. Register capabilities via ctx.
    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError>;

    /// Called on graceful shutdown. Flush state, close connections.
    fn shutdown(&self) -> Result<(), PluginError> { Ok(()) }
}
```

> **Note:** `init` and `shutdown` are sync. This was a deliberate decision —
> most plugins only inspect config and register handlers during init, which
> doesn't require async. An `async` variant (`init_async`) will be added
> when needed (e.g., for plugins that must connect to external services
> during startup).

This is deliberately minimal. A plugin doesn't need to implement every
subsystem — it registers only what it provides.

### PluginManifest

Declares what the plugin offers and what it requires to load:

```rust
pub struct PluginManifest {
    /// Capabilities this plugin provides (e.g., "llm:openai", "channel:slack")
    pub provides: Vec<String>,
    /// Capabilities this plugin requires (e.g., "memory", "llm")
    pub requires: Vec<String>,
    /// Load order hint — lower runs first (default 100)
    pub priority: u32,
}
```

The engine uses manifests for:
- **Dependency checking** — fail fast if a required capability isn't provided
- **Load ordering** — plugins that provide "memory" init before plugins that require it
- **Conflict detection** — warn if two plugins both provide "llm" without explicit config

---

## 3. PluginContext — the Registration API

During `init`, the plugin receives a mutable context to register capabilities:

```rust
pub struct PluginContext<'a> {
    /// This plugin's config section from TOML (e.g., [plugins.slack])
    pub config: &'a Value,

    /// Shared state — readable and writable. Plugins initialized earlier
    /// have already inserted their values.
    pub shared: &'a mut SharedState,

    /// Collected registrations (internal).
    pub(crate) registrations: &'a mut PluginRegistrations,
}

impl<'a> PluginContext<'a> {
    // --- Implemented ---

    /// Register a host function callable from the Python REPL.
    pub fn register_host_fn(&mut self, name: &str, handler: HostFnHandler);    // ✓

    /// Subscribe to a lifecycle event.
    pub fn on(&mut self, event: LifecycleEvent, handler: EventHandler);         // ✓

    /// Expose a value for downstream plugins to access.
    pub fn provide<T: Send + Sync + 'static>(&mut self, value: T);              // ✓

    // --- Planned (Phase 3) ---

    /// Register an LLM backend by name (e.g., "openai", "anthropic")
    pub fn register_llm_backend(&mut self, name: &str, factory: LlmClientFactory);

    /// Register a memory store. Composes with existing by default.
    pub fn register_memory_store(&mut self, store: Box<dyn MemoryStore>);

    /// Register a channel adapter (Slack, Discord, CLI, etc.)
    pub fn register_channel(&mut self, adapter: Box<dyn ChannelAdapter>);

    /// Register an agent bus implementation.
    pub fn register_bus(&mut self, bus: Box<dyn AgentBus>);

    /// Mount an Axum router at /plugins/{prefix}/*
    pub fn register_routes(&mut self, prefix: &str, router: axum::Router);
}
```

### SharedState

Plugins sometimes need to share things — a database pool, an HTTP client,
an auth provider. `SharedState` is a typed map (backed by `TypeMap` or
`anymap`):

```rust
impl SharedState {
    pub fn get<T: Send + Sync + 'static>(&self) -> Option<&T>;
}
```

A plugin that sets up a Postgres pool can `ctx.provide(pool)`. A later
plugin can `ctx.shared.get::<PgPool>()`. This avoids hard-wiring specific
services while keeping type safety.

---

## 4. Lifecycle Events

Events are emitted at natural points in the rLM loop and server lifecycle.
Each event carries a typed payload that handlers can read and (where
appropriate) mutate.

### Event Definitions

```rust
pub enum LifecycleEvent {
    // --- Server lifecycle ---
    BeforeStartup,          // config loaded, plugins initialized, about to listen
    AfterStartup,           // server accepting connections
    BeforeShutdown,         // graceful shutdown initiated

    // --- Session lifecycle ---
    SessionCreated,         // new conversation session opened
    SessionResumed,         // session restored from persistence
    SessionEvicted,         // session evicted due to idle timeout
    SessionEnded,           // session explicitly closed

    // --- Turn lifecycle (the rLM loop) ---
    BeforeTurn,             // user message received, before any processing
    AfterContextBuild,      // context assembled, before LLM call
    BeforeLlmCall,          // messages ready, about to call LLM
    AfterLlmCall,           // LLM responded, raw output available
    BeforeCodeExec,         // code extracted, about to execute in REPL
    AfterCodeExec,          // REPL execution complete
    BeforeHostCall,         // REPL paused on host function call
    AfterHostCall,          // host function resolved, about to resume
    AfterTurn,              // turn complete, final result available
    TurnError,              // turn failed with error

    // --- Memory lifecycle ---
    BeforeMemoryStore,      // about to persist a memory
    AfterMemoryRecall,      // memories retrieved, before delivery

    // --- Agent lifecycle ---
    AgentSpawned,           // sub-agent created via bus
    AgentCompleted,         // sub-agent returned result
}
```

### Event Handlers

```rust
pub type EventHandler = Arc<
    dyn Fn(&mut EventPayload) -> EventResult + Send + Sync
>;

pub enum EventResult {
    /// Proceed normally
    Continue,
    /// I modified the payload, proceed with changes
    Modified,
    /// Skip remaining processing, use this value as the result
    ShortCircuit(serde_json::Value),
}
```

### EventPayload

A payload is a struct with typed fields relevant to the event. Rather than
one mega-enum, we use a thin wrapper:

```rust
pub struct EventPayload {
    pub event: LifecycleEvent,
    pub session_id: Option<SessionId>,
    pub data: Box<dyn Any + Send>,
}
```

Handlers downcast `data` to the expected type for their event. Typed
helper constructors prevent mistakes:

```rust
// What the engine constructs internally:
let payload = EventPayload::before_llm_call(session_id, &mut messages);

// What a handler does:
fn my_guardrail(payload: &mut EventPayload) -> EventResult {
    let messages = payload.data.downcast_mut::<Vec<Message>>().unwrap();
    // inspect or modify messages
    EventResult::Continue
}
```

### Dispatch Model

Events dispatch to all subscribers in priority order (from the plugin's
manifest). For turn-lifecycle events, this means:

```
BeforeTurn: [logging(10), auth-check(50), rate-limit(80)]
BeforeLlmCall: [guardrails(20), rag-inject(50)]
AfterLlmCall: [guardrails(20), billing(90)]
```

If any handler returns `ShortCircuit`, remaining handlers for that event
are skipped and the engine uses the provided value. This lets guardrails
block a call without needing the rest of the pipeline.

---

## 5. Host Function Router

Currently `HostBridge` is a single `call(function, args, kwargs)` method.
This is already the right shape — we just make it composable.

### HostFnRouter

```rust
pub type HostFnHandler = Arc<
    dyn Fn(Vec<Value>, HashMap<String, Value>, &CallContext)
        -> BoxFuture<'_, Result<Object, AgentError>>
    + Send + Sync
>;

pub struct HostFnRouter {
    handlers: HashMap<String, HostFnHandler>,
    fallback: Option<Box<dyn HostBridge>>,
}

impl HostBridge for HostFnRouter {
    async fn call(&self, function: &str, args: Vec<Value>, kwargs: HashMap<String, Value>)
        -> Result<Object, AgentError>
    {
        if let Some(handler) = self.handlers.get(function) {
            handler(args, kwargs, &self.call_context).await
        } else if let Some(fallback) = &self.fallback {
            fallback.call(function, args, kwargs).await
        } else {
            Err(AgentError::UnknownFunction(function.to_string()))
        }
    }
}
```

Plugins register handlers during init:

```rust
// In a weather plugin's init():
ctx.register_host_fn("weather.forecast", Arc::new(|args, _kwargs, _ctx| {
    Box::pin(async move {
        let city = args[0].as_str().ok_or(AgentError::BadArg)?;
        let forecast = weather_api::get(city).await?;
        Ok(forecast.into())
    })
}));
```

Now any Python agent can call `weather.forecast("Amsterdam")`. The REPL
pauses, Rust fulfills the call, REPL resumes with the result. The rLM
discovers available host functions via the system prompt, which the engine
builds from the registry.

### Namespacing Convention

Host functions use dot-separated namespaces:
- `llm.*` — LLM operations (built-in)
- `memory.*` — memory store (built-in)
- `channel.*` — channel I/O (built-in)
- `weather.*` — weather plugin
- `jira.*` — Jira plugin

The engine auto-generates the SDK section of the system prompt from
registered functions, so the rLM always knows what's available.

---

## 6. The Engine

### Replacing the Monolithic main()

Current state: `gw-server/main.rs` does all wiring in a single function.
Proposed: extract a `GreatWheelEngine` that is the framework's public API.

```rust
// New crate: gw-engine (or module in gw-server)

pub struct GreatWheelEngine {
    config: AppConfig,
    plugins: Vec<Box<dyn Plugin>>,
}

impl GreatWheelEngine {
    pub fn from_config(path: &str) -> Result<Self, EngineError>;

    pub fn add_plugin(&mut self, plugin: impl Plugin) -> &mut Self;

    /// Initialize all plugins, wire components, start serving
    pub async fn start(self) -> Result<RunningEngine, EngineError>;
}

pub struct RunningEngine {
    shutdown_tx: tokio::sync::oneshot::Sender<()>,
    handle: tokio::task::JoinHandle<Result<(), EngineError>>,
}

impl RunningEngine {
    pub async fn wait_for_shutdown(self) -> Result<(), EngineError>;
}
```

### Init Sequence

```
1. Load config (TOML)
2. Sort plugins by manifest.priority
3. Check dependency graph (all requires are satisfiable)
4. For each plugin:
   a. Create PluginContext with plugin's config section
   b. Call plugin.init(&mut ctx)
   c. Merge registrations into global PluginRegistry
5. Build HostFnRouter from all registered host functions
6. Build merged Axum router (core routes + plugin routes)
7. Emit BeforeStartup event
8. Start listening
9. Emit AfterStartup event
```

### Shutdown Sequence

```
1. Emit BeforeShutdown event
2. Stop accepting connections
3. Wait for in-flight requests (with timeout)
4. For each plugin (reverse init order):
   a. Call plugin.shutdown()
5. Flush tracing
6. Close database connections
```

### User-Facing Code

Building a custom system on Greatwheel becomes:

```rust
use greatwheel::GreatWheelEngine;
use greatwheel_slack::SlackPlugin;
use greatwheel_openai::OpenAiPlugin;
use my_company::InternalToolsPlugin;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let engine = GreatWheelEngine::from_config("config/greatwheel.toml")?
        .add_plugin(SlackPlugin::new())
        .add_plugin(OpenAiPlugin::new())
        .add_plugin(InternalToolsPlugin::new());

    engine.start().await?.wait_for_shutdown().await
}
```

---

## 7. Plugin Configuration

Each plugin gets a TOML section under `[plugins.<name>]`:

```toml
[plugins.slack]
enabled = true
bot_token = "${SLACK_BOT_TOKEN}"
channels = ["#agents", "#alerts"]

[plugins.openai]
enabled = true
api_key = "${OPENAI_API_KEY}"
default_model = "gpt-4o"

[plugins.guardrails]
enabled = true
block_patterns = ["SSN", "credit card"]
```

Environment variable expansion (`${VAR}`) is handled by the engine before
passing config to the plugin. Plugins with `enabled = false` are skipped
entirely.

---

## 8. Concrete Plugin Examples

### Example A: Slack Channel Plugin

```rust
pub struct SlackPlugin { /* ... */ }

#[async_trait]
impl Plugin for SlackPlugin {
    fn name(&self) -> &str { "slack" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["channel:slack".into()],
            requires: vec![],
            priority: 50,
        }
    }

    async fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let token = ctx.config["bot_token"].as_str().unwrap();
        let adapter = SlackAdapter::connect(token).await?;
        ctx.register_channel(Box::new(adapter));
        Ok(())
    }
}
```

### Example B: Guardrails Plugin

```rust
pub struct GuardrailsPlugin { /* ... */ }

#[async_trait]
impl Plugin for GuardrailsPlugin {
    fn name(&self) -> &str { "guardrails" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["guardrails".into()],
            requires: vec!["llm".into()],
            priority: 10, // runs early
        }
    }

    async fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let patterns: Vec<String> = /* from config */;

        ctx.on(LifecycleEvent::BeforeLlmCall, Arc::new(move |payload| {
            let messages = payload.data.downcast_ref::<Vec<Message>>().unwrap();
            for msg in messages {
                for pattern in &patterns {
                    if msg.content.contains(pattern) {
                        return EventResult::ShortCircuit(
                            json!({"error": "blocked by content policy"})
                        );
                    }
                }
            }
            EventResult::Continue
        }));

        ctx.on(LifecycleEvent::AfterLlmCall, Arc::new(move |payload| {
            // screen LLM output too
            EventResult::Continue
        }));

        Ok(())
    }
}
```

### Example C: Custom Host Functions Plugin

```rust
pub struct JiraPlugin { /* ... */ }

#[async_trait]
impl Plugin for JiraPlugin {
    fn name(&self) -> &str { "jira" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["host_fn:jira".into()],
            requires: vec![],
            priority: 50,
        }
    }

    async fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let base_url = ctx.config["url"].as_str().unwrap().to_string();
        let api_token = ctx.config["token"].as_str().unwrap().to_string();

        ctx.register_host_fn("jira.create_issue", Arc::new(move |args, kwargs, _ctx| {
            let url = base_url.clone();
            let token = api_token.clone();
            Box::pin(async move {
                let summary = args[0].as_str().ok_or(AgentError::BadArg)?;
                let issue = jira_api::create(&url, &token, summary).await?;
                Ok(json!({"key": issue.key, "url": issue.url}).into())
            })
        }));

        ctx.register_host_fn("jira.search", Arc::new(move |args, kwargs, _ctx| {
            // ...
        }));

        Ok(())
    }
}
```

The rLM can then write:

```python
issue = jira.create_issue("Agent detected anomaly in metrics")
send_message(f"Created {issue['key']}: {issue['url']}")
```

---

## 9. Crate Restructuring

### Proposed Changes

| Change | Rationale |
|--------|-----------|
| Create `gw-engine` crate | Engine is a library; `gw-server` becomes a thin binary |
| Move `AgentBus` trait from `gw-bus` into `gw-core` | Trait-only crate adds friction, no implementations to justify a crate |
| Move `ChannelAdapter` trait from `gw-channels` into `gw-core` | Same — trait lives with the other core types |
| Delete `gw-bus`, `gw-channels` as standalone crates | Implementations move into plugins or `gw-engine` defaults |
| Add `plugin.rs` to `gw-core` | Plugin trait, PluginManifest, LifecycleEvent, EventPayload |
| Add `host_fn_router.rs` to `gw-engine` | Composable host function dispatch |
| Add `registry.rs` to `gw-engine` | PluginRegistry that collects all registrations |

### Resulting Crate Layout

```
gw-core        — types, traits (Plugin, HostBridge, MemoryStore, AgentBus,
                 ChannelAdapter, LifecycleEvent, etc.)
gw-runtime     — ouros integration (unchanged)
gw-llm         — Ollama/sglang client (unchanged, becomes a built-in plugin)
gw-memory      — HybridStore (unchanged, becomes a built-in plugin)
gw-loop        — conversation loop (add event dispatch hooks)
gw-trace       — OTel tracing (unchanged, becomes a built-in plugin)
gw-scheduler   — rate limiting (unchanged)
gw-engine      — NEW: PluginRegistry, HostFnRouter, GreatWheelEngine,
                 built-in plugin wrappers for llm/memory/trace
gw-server      — thin binary: loads config, adds plugins, calls engine.start()
gw-bench       — benchmark harness (unchanged)
```

### Dependency Graph After Restructuring

```
                    gw-core (traits + types)
                   /    |    \        \
           gw-llm  gw-memory gw-runtime  gw-trace
               \      |       /          /
                gw-loop (conversation)  /
                    \       |         /
                     gw-engine ------
                        |
                     gw-server (binary)
```

---

## 10. Converting Existing Components to Built-In Plugins

The existing subsystems don't disappear — they become the default plugins
that ship with Greatwheel. This validates the plugin API against real code.

```rust
// gw-engine/src/builtins/llm.rs

pub struct OllamaPlugin;

#[async_trait]
impl Plugin for OllamaPlugin {
    fn name(&self) -> &str { "ollama" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["llm:ollama".into()],
            ..Default::default()
        }
    }

    async fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let proxy_url = ctx.config["proxy_url"].as_str().unwrap();
        let direct_url = ctx.config["direct_url"].as_str().unwrap();
        let client = OllamaClient::new(proxy_url, direct_url);
        ctx.register_llm_backend("ollama", /* factory wrapping client */);
        ctx.provide(Arc::new(client)); // share with downstream plugins
        Ok(())
    }
}
```

Similarly for `HybridMemoryPlugin`, `OtelTracePlugin`, `HttpChannelPlugin`.

---

## 11. Integrating Events into the rLM Loop

The conversation loop (`gw-loop/src/conversation.rs`) currently has
`IterationCallback` as a one-off hook. Replace it with event dispatch:

```rust
// In ConversationLoop::handle_turn()

// Before turn
self.dispatch(LifecycleEvent::BeforeTurn, &mut TurnPayload { message, session_id });

// After building context
let mut ctx = build_turn_context(/* ... */);
self.dispatch(LifecycleEvent::AfterContextBuild, &mut ContextPayload { messages: &mut ctx });

// Before LLM call
self.dispatch(LifecycleEvent::BeforeLlmCall, &mut LlmCallPayload { messages: &mut ctx });

let response = self.llm.chat(ctx).await?;

// After LLM call
self.dispatch(LifecycleEvent::AfterLlmCall, &mut LlmResponsePayload { response: &mut response });

// Before code exec
let code = extract_code_blocks(&response);
self.dispatch(LifecycleEvent::BeforeCodeExec, &mut CodePayload { code: &code });

let result = self.repl.execute(&code).await?;

// After code exec
self.dispatch(LifecycleEvent::AfterCodeExec, &mut ExecResultPayload { result: &result });
```

`IterationCallback` becomes syntactic sugar over subscribing to the
relevant turn events.

---

## 12. Decisions (all resolved)

| Question | Decision | Rationale |
|----------|----------|-----------|
| Should `gw-engine` be a separate crate or a module in `gw-server`? | **Separate crate** | The whole point is others building on this — they need a library to depend on, not just a binary |
| Async vs sync event handlers? | **Sync first, plan for async** | Most handlers inspect/mutate data and don't need I/O. Design the `EventHandler` type so we can add an `on_async()` variant later without breaking existing sync handlers |
| Should `EventPayload::data` use `Any` downcasting or a typed enum? | **Typed enum for built-in events, `Any` escape hatch for plugin-defined events** | Built-in events get compile-time safety; plugins that define custom events use `Any` downcasting since the core can't know their types ahead of time |
| Dynamic plugin loading (`.so` files)? | **Deferred** | Compiled-in plugins cover the primary use case. ABI stability is a real burden. Revisit when the plugin API has stabilized through real usage |
| Should plugins depend on other plugins or on capabilities? | **Capability-based** | "I need `llm`" not "I need `OllamaPlugin`". Keeps plugins interchangeable — any plugin providing the `llm` capability satisfies the dependency |
| How to handle plugin panics? | **Catch with `catch_unwind`, log, disable, continue** | A misbehaving plugin should not take down the whole runtime. Catch the panic, log it with full context, disable the plugin for the remainder of the process, and continue serving with remaining plugins |

---

## 13. Implementation Plan

### Phase 1 — Foundations
- Add `Plugin`, `PluginManifest`, `PluginContext`, `LifecycleEvent` to `gw-core`
- Move `AgentBus` and `ChannelAdapter` traits into `gw-core`
- Implement `HostFnRouter`
- Delete empty `gw-bus` and `gw-channels` crates

### Phase 2 — Engine
- Create `gw-engine` crate with `PluginRegistry` and `GreatWheelEngine`
- Extract wiring logic from `gw-server/main.rs` into `gw-engine`
- `gw-server` becomes a thin binary calling `engine.start()`

### Phase 3 — Built-in Plugins
- Wrap existing Ollama, HybridStore, OTel, HTTP channel as built-in plugins
- Validate that the plugin API can express everything the current main() does

### Phase 4 — Event Dispatch
- Add event dispatch hooks to `ConversationLoop`
- Replace `IterationCallback` with lifecycle event subscriptions
- Add events to `SessionManager` (session created/ended/evicted)

### Phase 5 — Documentation & Example Plugins
- Write a "Building Your First Plugin" guide
- Ship 1-2 example plugins (e.g., a logging plugin, a webhook notifier)
- Document the full lifecycle event reference

### Phase 6 — Stabilization
- Gather feedback from real plugin development
- Finalize trait signatures (breaking changes before 1.0)
- Consider dynamic loading if demand exists
