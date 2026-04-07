# Design: gw-kb Agent Integration via Async Plugin Host Functions

**Status:** Draft
**Date:** 2026-04-07
**Motivation:** Expose `gw-kb` read operations (search, explore, topic
lookup) to rLM agents running inside ouros sessions, so agents can
actually use the knowledge base we built. Fixes a latent wiring gap
in the plugin host function system along the way.

---

## 0. Summary

`gw-kb` is a working knowledge base with hybrid search, a typed topic
graph, per-topic LLM summaries, and spreading-activation exploration.
None of it is reachable from an agent today: agents call host functions
through `HostBridge::call()`, and no bridge in the codebase dispatches
to the `HostFnRouter` where plugin host functions are registered. The
existing `memory.extract_entities` host function from the hindsight
plugin is **dead code** — registered at startup but unreachable at
runtime.

This document proposes a three-phase change that:

1. **Unblocks the plugin host function system** by adding async handler
   support and wiring `HostFnRouter` into `ConversationBridge`'s
   dispatch. Fixes the hindsight dead-code issue as a side effect.
2. **Adds a KB plugin** (`gw-engine/src/builtins/kb.rs`) that registers
   four read-only host functions: `kb.search`, `kb.explore`, `kb.topic`,
   `kb.topics`.
3. **Validates end-to-end** with a simple agent script and manual
   verification.

After this lands, an agent can say `hits = kb.search("centroid
pruning", k=5)` from inside Python and get structured results back,
with source attribution, topic membership, and scores.

---

## 1. Problem and context

### 1.1 What we have

The plugin system in `gw-core::plugin` defines:

```rust
pub type HostFnHandler = Arc<
    dyn Fn(Vec<Value>, HashMap<String, Value>)
        -> Result<Value, PluginError> + Send + Sync,
>;
// "Sync for now — the handler runs on the REPL execution thread."
```

Plugins register host functions in their `init()` method via
`ctx.register_host_fn(name, handler)`. `GreatWheelEngine::init()`
builds a `HostFnRouter` from all registered handlers and stores it on
the engine struct.

Three hindsight plugins (`hindsight_retain`, `hindsight_recall`,
`hindsight_opinions`) are added in `gw-server/main.rs`, and one of them
(`hindsight_retain`) registers `memory.extract_entities` as a host
function.

### 1.2 The wiring gap

Despite the above, **no bridge dispatches through the router**:

- `gw-server/main.rs` builds the engine, calls `engine.before_startup()`,
  and then constructs `SessionManager` **without passing the engine or
  router to it**. The engine goes out of reach of the session layer.
- `SessionManager::with_pg` internally creates `ConversationBridge`
  instances for each session. `ConversationBridge::call()` has a
  hardcoded match on `send_message` / `ask_user` / `compact_session`
  and falls through to an optional `inner: Box<dyn HostBridge>`. The
  inner bridge has no knowledge of plugin-registered handlers.
- `gw-bench`'s `BrowseCompBridge` is a parallel world that hand-wires
  its own `search` / `vector_search` / `get_document` / `llm_query`
  functions with no router reference.

**Confirmed by `grep`:** `HostFnRouter` is constructed only inside
`gw-engine`. No other crate references it.

Result: `memory.extract_entities` is registered during startup, lives
in the router, and is never callable from an agent. Any KB plugin we
add now would hit the same dead end.

### 1.3 Why agent integration matters

Three immediate use cases unlock once agents can call the KB:

1. **BrowseComp benchmark integration.** The benchmark runs an agent
   that retrieves passages to answer questions. Today that goes through
   `BrowseCompBridge`'s hand-wired `search`; with KB integration the
   benchmark can use `kb.search` or `kb.explore` to test whether the KB
   improves retrieval quality.
2. **Research agents.** An autoresearch-style agent (see
   `docs/design-autoresearch.md`) can use the KB as its primary
   corpus: `kb.topic("carolingian-empire")` to pull an existing
   synthesis, `kb.explore("query")` to walk the graph.
3. **Session-local learning.** A conversation agent can reference
   `kb.search` results mid-session rather than relying solely on
   agent memory.

---

## 2. Goals and non-goals

### 2.1 Goals

- G1. Plugin-registered host functions are actually callable from an
  agent through `ConversationBridge`.
- G2. Handlers can be **async**, so they can hit Postgres / LanceDB /
  tantivy / the embedder without contorted bridging.
- G3. Four read-only KB operations exposed to agents:
  `kb.search`, `kb.explore`, `kb.topic`, `kb.topics`.
- G4. Each KB host function declares a `kb.read` capability string.
  Enforcement is future work but the declaration lands now.
- G5. The existing hindsight `memory.extract_entities` function becomes
  reachable as a side effect, without any changes to the hindsight code
  beyond whatever minor surface changes the `HostFnHandler` refactor
  forces.

### 2.2 Non-goals

- NG1. Full async end-to-end through `HostBridge::call()`. The trait
  stays sync because ouros calls it from a sync context. Async handlers
  are bridged via `block_in_place` + `block_on` at the dispatch site.
- NG2. Capability *enforcement*. We add the capability string to the
  registration; we do not enforce it in this slice. Agents in dev mode
  are default-granted `kb.read`.
- NG3. Migrating `BrowseCompBridge` to use the router. It stays
  hand-wired until the benchmark integration work explicitly needs it.
- NG4. Write operations from agents (ingest, organize, merge, link,
  classify, synthesize, clean, feed management). These are operator
  actions and remain CLI-only.
- NG5. `NullBridge` router integration. Test helpers stay unchanged.

---

## 3. Design

### 3.1 Async-capable `HostFnHandler`

`gw-core::plugin::HostFnHandler` becomes an enum:

```rust
use futures::future::BoxFuture;

pub enum HostFnHandler {
    Sync(Arc<dyn Fn(Vec<Value>, HashMap<String, Value>)
        -> Result<Value, PluginError> + Send + Sync>),
    Async(Arc<dyn Fn(Vec<Value>, HashMap<String, Value>)
        -> BoxFuture<'static, Result<Value, PluginError>>
        + Send + Sync>),
}
```

`PluginContext` gains two constructors:

```rust
fn register_host_fn_sync<F>(&mut self, name: &str, capability: Option<&str>, handler: F)
where F: Fn(Vec<Value>, HashMap<String, Value>) -> Result<Value, PluginError> + Send + Sync + 'static;

fn register_host_fn_async<F, Fut>(&mut self, name: &str, capability: Option<&str>, handler: F)
where F: Fn(Vec<Value>, HashMap<String, Value>) -> Fut + Send + Sync + 'static,
      Fut: Future<Output = Result<Value, PluginError>> + Send + 'static;
```

The existing `register_host_fn` keeps its signature as a thin alias for
`register_host_fn_sync` (with `None` capability) so the hindsight
plugins don't need to change their calls. This keeps the refactor
additive.

`capability: Option<&str>` lets a plugin declare what the function
needs, e.g. `Some("kb.read")`. Stored alongside the handler. Not
enforced in this slice (see NG2).

### 3.2 Router dispatch

`HostFnRouter` gains a dispatch method:

```rust
impl HostFnRouter {
    pub fn dispatch(
        &self,
        function: &str,
        args: Vec<Value>,
        kwargs: HashMap<String, Value>,
    ) -> Option<Result<Value, PluginError>> {
        let handler = self.handlers.get(function)?;
        Some(match handler {
            HostFnHandler::Sync(f) => f(args, kwargs),
            HostFnHandler::Async(f) => {
                // Block the current runtime worker until the future
                // resolves. Safe inside multi-threaded tokio (which
                // gw-server uses); panics in current_thread runtimes.
                tokio::task::block_in_place(|| {
                    tokio::runtime::Handle::current().block_on(f(args, kwargs))
                })
            }
        })
    }
}
```

`dispatch()` returns `Option<Result>`:
- `None` — function not registered; caller should fall through
- `Some(Ok(value))` — success
- `Some(Err(e))` — function ran but failed

The `block_in_place` + `Handle::current().block_on` pattern is
canonical tokio. It requires a multi-threaded runtime, which
`gw-server` already uses via `#[tokio::main]`.

**Call path:**
```
ouros → HostBridge::call() (sync)
    → ConversationBridge::call() match
        → if router.dispatch() returns Some, use it
        → else fall through to hardcoded matches / inner bridge
```

### 3.3 Wiring the router into `ConversationBridge`

`ConversationBridge` currently has:

```rust
pub struct ConversationBridge {
    event_tx: Sender<LoopEvent>,
    ask_handle: AskHandle,
    inner: Option<Box<dyn HostBridge>>,
}
```

It gains an optional router:

```rust
pub struct ConversationBridge {
    event_tx: Sender<LoopEvent>,
    ask_handle: AskHandle,
    inner: Option<Box<dyn HostBridge>>,
    plugin_router: Option<Arc<HostFnRouter>>,
}
```

`ConversationBridge::new` gains a `plugin_router: Option<Arc<HostFnRouter>>`
parameter. The existing call sites (session_manager, tests) pass `None`
unless they're specifically providing one.

`ConversationBridge::call()` adds router dispatch after the hardcoded
match block and before the `inner` fallback:

```rust
impl HostBridge for ConversationBridge {
    fn call(&mut self, function: &str, args: ..., kwargs: ...) -> Result<Object, AgentError> {
        match function {
            "send_message" | "channel.send" => { ... }
            "ask_user" | "channel.ask" => { ... }
            "compact_session" | "session.compact" => { ... }
            _ => {
                // NEW: try the plugin router before the inner bridge
                if let Some(router) = &self.plugin_router {
                    if let Some(result) = router.dispatch(function, args.clone(), kwargs.clone()) {
                        return result
                            .map(json_to_object)
                            .map_err(|e| AgentError::HostFunction {
                                function: function.to_string(),
                                message: e.to_string(),
                            });
                    }
                }
                // existing fallback
                if let Some(inner) = &mut self.inner {
                    inner.call(function, args, kwargs)
                } else {
                    Err(AgentError::UnknownFunction(function.to_string()))
                }
            }
        }
    }
}
```

Argument cloning into router dispatch is acceptable at our call rates
(low hundreds of host calls per agent turn).

### 3.4 Threading the router from `gw-server`

`gw-server/main.rs` currently:

```rust
let engine = GreatWheelEngine::new()
    .add_plugin(HindsightRetainPlugin)
    ...
    .init(&config.plugins)?;
engine.before_startup();

let session_mgr = Arc::new(match session_pool {
    Some(pool) => SessionManager::with_pg(...),
    None => SessionManager::new(...),
});
```

After the change:

```rust
let engine = GreatWheelEngine::new()
    .add_plugin(HindsightRetainPlugin)
    .add_plugin(KbPlugin::new(kb_stores.clone()))  // NEW
    ...
    .init(&config.plugins)?;
engine.before_startup();
let plugin_router = Arc::new(engine.host_fn_router().clone());

let session_mgr = Arc::new(match session_pool {
    Some(pool) => SessionManager::with_pg(
        llm_factory,
        LoopConfig::default(),
        Duration::from_secs(30 * 60),
        pool,
        Some(plugin_router),   // NEW
    ),
    ...
});
```

`SessionManager::new` and `::with_pg` gain a
`plugin_router: Option<Arc<HostFnRouter>>` parameter that they store
and pass to each `ConversationBridge` they create internally.

`HostFnRouter` needs to be cheap-to-clone for this Arc wrapping.
Currently it holds `HashMap<String, HostFnHandler>`. `HostFnHandler`
wraps `Arc<dyn Fn ...>` so cloning the map is O(n) in handler count but
allocation-free per handler. We'll wrap the whole router in `Arc` and
share, not clone.

### 3.5 KB plugin

New file: `crates/gw-engine/src/builtins/kb.rs`

```rust
pub struct KbPlugin {
    stores: Arc<KbStores>,
}

impl KbPlugin {
    pub fn new(stores: KbStores) -> Self {
        Self { stores: Arc::new(stores) }
    }
}

impl Plugin for KbPlugin {
    fn name(&self) -> &str { "kb" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "host_fn:kb.search".into(),
                "host_fn:kb.explore".into(),
                "host_fn:kb.topic".into(),
                "host_fn:kb.topics".into(),
            ],
            requires: vec![],
            priority: 50,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let stores = Arc::clone(&self.stores);

        // kb.search(query: str, k: int = 5) -> list[dict]
        let s = Arc::clone(&stores);
        ctx.register_host_fn_async("kb.search", Some("kb.read"), move |args, kwargs| {
            let s = Arc::clone(&s);
            Box::pin(async move {
                let query = get_str(&args, &kwargs, 0, "query")?;
                let k = get_usize(&args, &kwargs, 1, "k").unwrap_or(5);
                let hits = gw_kb::search::hybrid_search(&s, &query, k)
                    .await
                    .map_err(|e| PluginError::Runtime(e.to_string()))?;
                Ok(serde_json::to_value(hits).unwrap())
            })
        });

        // kb.explore(query: str, k: int = 10) -> list[dict]
        // kb.topic(slug: str) -> dict | None
        // kb.topics(limit: int = 50) -> list[dict]
        // (similar pattern)

        Ok(())
    }
}
```

Helpers `get_str`, `get_usize` pull positional args first, kwargs
second, with clear error messages if missing.

`SearchHit` and other return types already `derive(Serialize)` in
`gw-kb`, so `serde_json::to_value` gives us plain dicts on the agent
side.

### 3.6 KbStores construction in gw-server

Add a `[kb]` section to `config/greatwheel.toml`:

```toml
[kb]
lance_path = "data/kb-lancedb"
tantivy_path = "data/kb-tantivy"
embedding_model = "nomic-ai/nomic-embed-text-v1.5"
embedding_dim = 768
enabled = true
```

Wire into `gw-server/main.rs`:

```rust
let kb_stores = if config.kb.enabled {
    Some(build_kb_stores(&config, session_pool.clone()).await?)
} else {
    None
};

// ...
let mut engine = GreatWheelEngine::new()
    .add_plugin(HindsightRetainPlugin)
    ...;
if let Some(stores) = kb_stores {
    engine = engine.add_plugin(KbPlugin::new(stores));
}
```

`build_kb_stores` is a helper that constructs `KbStores` the same way
`gw-kb/src/bin/gw_kb.rs::build_stores` does, sharing the Postgres pool
with the rest of gw-server.

`kb.enabled = false` keeps the server bootable without KB data
directories, useful for CI and minimal deployments.

### 3.7 Capability system (stub)

`PluginContext::register_host_fn_async` takes a
`capability: Option<&str>`. The registry records this but the router
does not enforce it yet. When enforcement lands (future work), the
dispatch site will check the active agent's capability set before
invoking the handler.

For the KB plugin, every function declares `Some("kb.read")`.

In dev mode the default agent capability set should include `kb.read`
so agents can call these functions out of the box. Adding this to the
agent defaults is a one-line change in `gw-core::agent` — flagged here
as a loose end to tie.

---

## 4. Return shapes

Agents receive plain dicts. Field names match gw-kb internal types.

### `kb.search(query, k=5)`
Returns `list[dict]`:
```python
[{
    "chunk_id": "uuid-string",
    "source_id": "uuid-string",
    "source_title": "PLAID: ...",
    "source_url": "https://arxiv.org/pdf/2205.09707",
    "heading_path": ["# PLAID ...", "## 4.2 Centroid Interaction"],
    "content": "## **4.2 Centroid Interaction** ...",
    "score": 0.0325
}, ...]
```

### `kb.explore(query, k=10)`
Returns `list[dict]`:
```python
[{
    "topic_id": "uuid-string",
    "label": "Carolingian Empire",
    "slug": "carolingian-empire",
    "chunk_count": 47,
    "score": 0.3901
}, ...]
```

### `kb.topic(slug)`
Returns `dict | None`:
```python
{
    "topic_id": "uuid-string",
    "label": "Cosmology",
    "slug": "cosmology",
    "chunk_count": 64,
    "first_seen": "2026-04-07T16:35:03Z",
    "last_seen": "2026-04-07T16:43:03Z",
    "summary": "Big Bang nucleosynthesis stands as a pivotal epoch ...",
    "summary_at": "2026-04-07T17:01:00Z",
    "neighbors": [
        {"kind": "subtopic_of", "direction": "incoming", "label": "Fast Radio Bursts", "slug": "fast-radio-bursts", "confidence": 0.958},
        {"kind": "related", "label": "ΛCDM model", "slug": "cdm-model", "confidence": 0.979},
        ...
    ]
}
```
Returns `None` if the slug does not match any topic.

### `kb.topics(limit=50)`
Returns `list[dict]`:
```python
[{
    "topic_id": "uuid-string",
    "label": "Machine Learning",
    "slug": "machine-learning",
    "chunk_count": 408,
    "source_count": 38,
    "last_seen": "2026-04-07T19:02:00Z"
}, ...]
```

---

## 5. Phases and sequencing

### Phase 1 — Foundation (unblock the plugin system)

1. Enum `HostFnHandler` in `gw-core::plugin`. Keep old
   `register_host_fn` as a thin alias for the sync constructor. Add
   async constructor + capability param.
2. `HostFnRouter::dispatch()` with block-on bridge.
3. `ConversationBridge::plugin_router` field + dispatch in `call()`.
4. Thread the router through `SessionManager::new` / `::with_pg`.
5. In `gw-server/main.rs`, extract router from engine and pass to
   session manager.
6. Verify: `memory.extract_entities` is now callable from an agent
   (dormant hindsight function comes alive as a side effect).

### Phase 2 — KB plugin

7. `crates/gw-engine/src/builtins/kb.rs` with `KbPlugin` and the four
   async host functions.
8. `[kb]` section in `config/greatwheel.toml` + `KbStores` construction
   in `gw-server/main.rs`.
9. Add `kb` builtin to `gw-engine::builtins::mod.rs`.
10. Register `KbPlugin` in `gw-server/main.rs` engine builder
    (conditional on `kb.enabled`).

### Phase 3 — Validation

11. Smoke-test agent script (`examples/kb_agent_smoke.py` or inline in
    a test) that calls `kb.search`, `kb.topic`, `kb.explore`,
    `kb.topics` and prints results.
12. Run the agent against the current 53-source / 176-topic corpus.
13. Verify error paths: missing slug returns `None`, missing required
    arg returns `PluginError::Runtime`.

---

## 6. Risks and mitigations

**R1. `block_in_place` panics in single-threaded runtimes.**
The `gw-server` binary uses `#[tokio::main]` which defaults to
multi-threaded. But tests may use `#[tokio::test]` which defaults to
current-thread. Mitigation: document the expectation in the `dispatch`
doc comment; any tests exercising plugin handlers must use
`#[tokio::test(flavor = "multi_thread")]`.

**R2. Long-running async handlers block a runtime worker.**
A `kb.search` call takes hundreds of ms (embedder + tantivy + lance +
postgres). A `kb.synthesize` call would take several seconds. The
block-on pattern ties up a runtime worker for the duration. Mitigation:
KB operations exposed to agents are read-only retrieval; write
operations stay CLI-only. Document the runtime-worker cost clearly.

**R3. `HostFnHandler` enum refactor breaks the hindsight plugins.**
The existing `ctx.register_host_fn(name, handler)` call site needs to
keep working. Mitigation: keep `register_host_fn` as a thin alias that
constructs `HostFnHandler::Sync` with `capability: None`. Zero-diff for
the hindsight plugin body.

**R4. `KbStores` initialization fails at startup.**
`build_stores` in gw-kb today loads the sentence-transformers model
lazily on first call, but the LanceDB and tantivy stores are opened
eagerly. If the lance path is missing, server boot fails. Mitigation:
`kb.enabled = false` in config skips KB plugin entirely. Document the
setup requirement.

**R5. Agent observability.**
Host function calls from agents should be traceable in OTel spans.
Mitigation: wrap `router.dispatch()` in a `tracing::info_span!` with
`gen_ai.host_fn.name` and `duration_ms` fields. Matches existing
tracing conventions.

**R6. Shared `KbStores` across async tasks.**
Multiple concurrent agent sessions may call `kb.search` simultaneously.
`KbStores` holds `PgPool` (thread-safe), `Arc<KbLanceStore>`,
`Arc<KbTantivyStore>`, `Arc<Embedder>`, and `Arc<OllamaClient>` — all
`Send + Sync`. The Embedder uses PyO3 which acquires the GIL
internally, so concurrent embed calls serialize on the GIL. Acceptable
for current load; documented here as a scaling boundary.

---

## 7. Testing

- **Unit:** `HostFnHandler` enum construction and dispatch, both
  variants, via synthetic handlers.
- **Integration:** A test in `crates/gw-loop/tests` that builds a
  `ConversationBridge` with a router containing a stub plugin handler
  and verifies `call()` dispatches to it.
- **End-to-end:** Run an agent script against a gw-server with KB
  enabled. Verify all four functions return sensible data for the
  current corpus.

---

## 8. Open questions

- **Q1.** Should the KB plugin take a pre-built `KbStores` or build its
  own from config? Taking a pre-built one is more flexible (server
  controls lifetime and can inject test doubles); building its own is
  simpler. Leaning toward pre-built.
- **Q2.** How should capability grants be surfaced in the CLI `gw-bench`
  runner? For this slice, benchmarks hand-wire their own bridges and
  don't go through the router, so capability enforcement is moot there.
  Revisit when the benchmark starts using the KB plugin path.
- **Q3.** Do we want a way for agents to introspect available host
  functions at runtime (e.g. `list_host_fns()`)? Useful for rLM agents
  that adapt to environment. Deferred — nice-to-have.

---

## 9. Files touched

- `crates/gw-core/src/plugin.rs` — enum refactor, async constructor
- `crates/gw-engine/src/host_fn_router.rs` — `dispatch()` with block-on
- `crates/gw-engine/src/builtins/kb.rs` — NEW, the plugin
- `crates/gw-engine/src/builtins/mod.rs` — register `kb` submodule
- `crates/gw-loop/src/bridge.rs` — `plugin_router` field + dispatch
- `crates/gw-loop/src/session.rs` — thread router through `SessionManager`
- `crates/gw-server/src/main.rs` — build `KbStores`, add plugin, pass router
- `config/greatwheel.toml` — `[kb]` section
- `crates/gw-loop/tests/conversation_test.rs` — integration test
- `examples/kb_agent_smoke.py` — NEW, smoke test script

---

## 10. Acceptance criteria

- [ ] `HostFnHandler` is an enum with `Sync` and `Async` variants.
- [ ] `ConversationBridge::call()` routes unknown function names
      through the plugin router before the inner fallback.
- [ ] `gw-server` constructs the router from the engine and passes it
      to `SessionManager`.
- [ ] `memory.extract_entities` (hindsight) is callable from an agent
      without any additional wiring — confirms the foundation works.
- [ ] `KbPlugin` registers four read-only host functions under the
      `kb.read` capability.
- [ ] An agent script calls `kb.search("centroid pruning", k=5)` and
      receives five ranked hits from the corpus with source
      attribution.
- [ ] An agent script calls `kb.topic("cosmology")` and receives the
      stored synthesis plus neighbor graph.
- [ ] Server boot with `kb.enabled = false` still works.
- [ ] All existing tests pass. New unit and integration tests added.
