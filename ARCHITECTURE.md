# Greatwheel — Architecture Vision (v3)

## One-liner

An agentic runtime where rLM-coordinated agents execute as Python code
inside sandboxed ouros sessions, with a Rust core managing memory, routing,
and inter-agent communication — multi-tenant, observable, and fine-tunable.

---

## Core Insight

An rLM (Recursive Language Model) is not a special model — it's an
**inference strategy**. The model receives a task but never sees the full
context directly. Instead, context is stored in a Python REPL environment
as variables. The model writes Python code to search, partition, transform,
and recursively sub-query that context — using the REPL as its thinking
space. It can spawn recursive LM calls as function invocations in code.

Ouros gives us the perfect execution substrate: a sandboxed Python runtime
that **pauses on external function calls** and produces serializable
snapshots. The model writes code, ouros executes it, and every host
function call (LLM, memory, agents) is a pause point where the Rust
runtime takes over.

This means:
- Agent logic is readable Python, not opaque framework glue.
- The rLM's context window stays clean — it never sees the full input,
  only what it explicitly queries via code.
- Every pause point is serializable — agents survive restarts.
- The host controls all I/O — perfect auditability and rate limiting.
- An agent with **no predefined code** is still useful — it has the REPL
  and our SDK, so the rLM can build its own tools on the fly.

---

## Decisions (all resolved)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Persistence | **PostgreSQL** | Relational backbone for tenancy, tasks, agent defs, audit logs |
| Vector store | **LanceDB** (embedded, Rust-native) | Hybrid search (vector + FTS + SQL), Lance columnar format, no external service |
| LLM provider | **Ollama (Qwen)** first | Local-first, no API keys for dev; provider trait allows adding others later |
| Auth model | Per-agent tool allowlists + per-user defaults | Agents get explicit capability grants; users inherit org defaults with overrides |
| Observability | **OpenTelemetry** with GenAI semantic conventions | Industry standard; GenAI semconv for `gen_ai.agent.*`, `gen_ai.usage.*` spans |
| Trace storage | Traces → Postgres (+ optional OTLP) | Enables behavior correction and fine-tuning dataset extraction |
| Fine-tuning | **rl-play** integration | Transparent Ollama proxy captures sessions → SFT/DPO/PPO training pipeline |
| Agent hot-reload | **Yes** — session state survives code updates | File watcher; new code loads into existing session on next turn |
| Agent versioning | **Version history + rollback** | Every source change recorded; rollback to any prior version |
| Multi-tenancy | **Multi-org, multi-user from the start** | Org → User → Session hierarchy in core types and DB schema |
| Session lifecycle | **Per-org configurable idle timeout** | Snapshot-and-evict after configurable idle period; tune as we learn |
| Rate limiting | **Hard + soft limits** for users and orgs | Soft = warning, hard = reject; not billing by tokens yet |
| Secrets management | **Session key only** | Agents never see secrets; sensitive calls go through our SDK using session key |
| Deployment | **Docker Compose** from the start | greatwheel + Postgres + Ollama + rl-play as services |

---

## System Layers

```
┌─────────────────────────────────────────────────────┐
│                   Channel Layer                      │
│   HTTP API · WebSocket · CLI · Webhooks (Slack,etc) │
└──────────────────────┬──────────────────────────────┘
                       │ Task (scoped to org + user)
┌──────────────────────▼──────────────────────────────┐
│                  Router / Scheduler                   │
│   Triage agent classifies → dispatches to workers,   │
│   manages concurrency, priority, retries per org     │
└──────────────────────┬──────────────────────────────┘
                       │ AgentInvocation
┌──────────────────────▼──────────────────────────────┐
│                  Agent Runtime                        │
│                                                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐    │
│  │  Session A  │  │  Session B  │  │  Session C  │    │
│  │  (ouros)    │  │  (ouros)    │  │  (ouros)    │    │
│  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘    │
│        │               │               │            │
│        ▼               ▼               ▼            │
│  ┌──────────────────────────────────────────────┐   │
│  │           Host Function Bridge                │   │
│  │                                               │   │
│  │  llm.complete()    memory.recall()            │   │
│  │  agent.call()      channel.send()             │   │
│  │  tool.*()          state.checkpoint()         │   │
│  └──────────────────────────────────────────────┘   │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                  Core Services                        │
│                                                      │
│  ┌──────────┐ ┌──────────────┐ ┌──────────┐        │
│  │ LLM Gate │ │ Memory Store │ │ Msg Bus  │        │
│  │          │ │ (Lance+PG)   │ │          │        │
│  └────┬─────┘ └──────────────┘ └──────────┘        │
│       │       ┌──────────┐ ┌────────────────┐      │
│       │       │ Registry │ │ Trace Recorder │      │
│       │       │ (agents) │ │ (OTel GenAI)   │      │
│       │       └──────────┘ └────────────────┘      │
└───────┼─────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────┐
│               rl-play (transparent proxy)            │
│   Intercepts LLM calls, captures sessions,          │
│   groups by session-id for fine-tuning pipelines     │
│   SFT · DPO · PPO → ollama create                   │
└───────┬─────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────┐
│                   Ollama (Qwen)                      │
└─────────────────────────────────────────────────────┘
        │
┌───────▼─────────────────────────────────────────────┐
│                   PostgreSQL                         │
│  orgs · users · sessions · tasks · agent_defs       │
│  agent_versions · snapshots · traces · rate_limits   │
└─────────────────────────────────────────────────────┘
```

---

## The rLM Execution Model

Based on the [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/)
paradigm. The key insight: the rLM's context window never gets clogged
because it never sees the full context directly. Instead, context is
stored as Python variables in the ouros session, and the rLM writes code
to inspect, partition, and query it.

### How a Turn Executes

```
1. User sends message → Task created
2. Task dispatched to agent's ouros session
3. Context injected as Python variable:
     _task = {"payload": "...", "id": "..."}
     # Prior session state (variables, functions) already live
4. rLM receives:
     - System prompt (agent definition + available SDK functions)
     - Truncated session state summary (variable names + types)
     - The user message
5. rLM outputs Python code block(s)
6. Ouros executes the code:
     - Pure computation runs immediately
     - Host function calls (llm, memory, agent, channel) → PAUSE
     - Rust runtime fulfills the call → RESUME
     - Output/return value captured
7. rLM sees the execution output, may emit more code
8. Loop until rLM produces a FINAL() response or limit is hit
9. Session state persists — all variables survive to next turn
```

### The "Bare Agent" — No Code Required

An agent with no predefined entrypoint source is still functional.
The rLM has access to the SDK globals (`llm`, `memory`, `agent`,
`channel`, `tool`) and can build its own tools on the fly:

```python
# rLM writes this in turn 1 of a fresh session
def search_and_summarize(query, top_k=5):
    """Helper I'll reuse across turns."""
    results = memory.recall(query, top_k=top_k, mode="hybrid")
    summary = llm.complete(
        system="Summarize these search results concisely.",
        messages=[{"role": "user", "content": str(results)}],
    )
    return summary

# rLM uses it immediately
answer = search_and_summarize(_task["payload"])
channel.send(answer)

# Turn 2: search_and_summarize is still defined and available
```

### Recursive Sub-queries

The rLM can decompose complex tasks by spawning sub-LLM calls, exactly
as described in the rLM paper — the REPL is the decomposition engine:

```python
# rLM receives a large document as context
# Instead of cramming it into one LLM call, it decomposes:

chunks = [_context[i:i+2000] for i in range(0, len(_context), 2000)]
summaries = []
for chunk in chunks:
    s = llm.complete(
        system="Extract key facts from this text chunk.",
        messages=[{"role": "user", "content": chunk}],
    )
    summaries.append(s)

# Now synthesize with a clean context window
final = llm.complete(
    system="Synthesize a comprehensive answer from these summaries.",
    messages=[
        {"role": "user", "content": _task["payload"]},
        {"role": "context", "content": "\n".join(summaries)},
    ],
)
FINAL(final)
```

---

## Multi-Tenancy Model

```
Org (tenant boundary)
├── Settings (default model, rate limits, session idle timeout)
├── Users
│   ├── default tool permissions (inherited by agents they invoke)
│   ├── rate limits (soft + hard, token budgets)
│   └── Sessions (conversations with agents)
├── Agents (scoped to org)
│   ├── tool allowlist (what host functions this agent can call)
│   ├── model config (which Ollama model, params)
│   ├── resource limits (memory, time, max LLM calls)
│   └── version history (source snapshots, rollback)
└── Memory (org-scoped, optionally user-scoped)
```

```rust
struct OrgId(Uuid);
struct UserId(Uuid);

struct Org {
    id: OrgId,
    name: String,
    settings: OrgSettings,
}

struct OrgSettings {
    default_model: String,             // e.g. "qwen2.5:7b"
    session_idle_timeout: Duration,    // snapshot-and-evict after this
    rate_limits: RateLimitConfig,
}

struct RateLimitConfig {
    /// Soft limit: emit warning, continue
    soft_token_limit: Option<u64>,
    /// Hard limit: reject requests
    hard_token_limit: Option<u64>,
    /// Per-user overrides
    per_user_overrides: HashMap<UserId, RateLimitConfig>,
}

struct User {
    id: UserId,
    org_id: OrgId,
    name: String,
    default_permissions: ToolPermissions,
    rate_limits: Option<RateLimitConfig>,  // override org defaults
}

struct Session {
    id: SessionId,
    org_id: OrgId,
    user_id: UserId,
    agent_id: AgentId,
    ouros_session: OurosSessionHandle,
    created_at: DateTime<Utc>,
    last_active_at: DateTime<Utc>,
    // After org.settings.session_idle_timeout of inactivity:
    // snapshot ouros state → store in DB → evict from memory
}
```

---

## Key Components

### 1. Channel Layer
Accepts inbound tasks and normalizes them into a `Task` envelope:

```rust
struct Task {
    id: TaskId,
    org_id: OrgId,
    user_id: UserId,
    channel: Channel,
    payload: String,
    context: Option<Value>,
    reply_to: Option<ReplyHandle>,
    parent_task: Option<TaskId>,
}
```

Channels are pluggable via `ChannelAdapter` trait.
Auth middleware resolves org + user identity before task creation.

### 2. Router / Scheduler
- Triage agent (rLM-powered) classifies and delegates tasks.
- Per-org concurrency limits and work queue.
- Retries with backoff, dead-letter on failure.
- Respects agent allowlists × user permissions intersection.
- Checks rate limits before dispatching (soft → warn, hard → reject).

### 3. Agent Runtime

```rust
struct AgentDef {
    id: AgentId,
    org_id: OrgId,
    name: String,
    description: String,
    system_prompt: String,
    source: AgentSource,       // can be None — "bare agent"
    tools: ToolPermissions,
    model: ModelConfig,
    limits: ResourceLimits,
    current_version: u32,
}

enum AgentSource {
    /// No predefined code — rLM builds tools on the fly via REPL
    Bare,
    /// Watched for changes → hot-reload + version bump
    File(PathBuf),
    /// Stored in DB (via API)
    Inline(String),
    /// Pulled from git
    Git { repo: String, path: String, rev: String },
}
```

**Hot-reload:** File watcher detects changes → new source loaded into
existing ouros session on next invocation → session state (variables)
preserved → version bumped and old source archived.

### 4. Agent Versioning

Every source change creates a version record:

```rust
struct AgentVersion {
    agent_id: AgentId,
    version: u32,
    source_snapshot: String,    // the Python source at this version
    created_at: DateTime<Utc>,
    created_by: VersionTrigger, // FileWatch, ApiUpdate, GitSync
}

enum VersionTrigger {
    FileWatch,    // hot-reload detected file change
    ApiUpdate,    // user updated via API
    GitSync,      // git pull brought new revision
    Rollback,     // explicit rollback to prior version
}
```

Rollback: `PUT /agents/{id}/rollback?version=3` → loads version 3's
source, creates a new version record (version N+1) pointing to version
3's source, hot-reloads into active sessions.

### 5. Host Function Bridge

Enforced by per-agent + per-user permission intersection:

```rust
struct ToolPermissions {
    allowed: HashSet<ToolName>,
    denied: HashSet<ToolName>,
}

fn effective_permissions(agent: &AgentDef, user: &User) -> ToolPermissions {
    agent.tools.intersect(&user.default_permissions)
}
```

Host function traits (all receive `CallContext` for auth/tracing/tenancy):

```rust
struct CallContext {
    org_id: OrgId,
    user_id: UserId,
    session_id: SessionId,
    agent_id: AgentId,
    task_id: TaskId,
    session_key: SessionKey,     // the ONLY secret the agent can use
    span: tracing::Span,
    permissions: ToolPermissions,
}

#[async_trait]
trait LlmGate {
    async fn complete(&self, ctx: &CallContext, req: CompletionRequest) -> CompletionResponse;
    async fn embed(&self, ctx: &CallContext, text: &str) -> Vec<f32>;
}

#[async_trait]
trait MemoryStore {
    async fn store(&self, ctx: &CallContext, key: &str, value: Value) -> Result<(), MemoryError>;
    async fn recall(&self, ctx: &CallContext, query: &str, opts: RecallOpts) -> Result<Vec<MemoryRecord>, MemoryError>;
    async fn forget(&self, ctx: &CallContext, key: &str) -> Result<(), MemoryError>;
}

#[async_trait]
trait AgentBus {
    async fn call(&self, ctx: &CallContext, agent: AgentId, task: Task) -> AgentResult;
    async fn notify(&self, ctx: &CallContext, agent: AgentId, message: Value);
}

#[async_trait]
trait ChannelOut {
    async fn send(&self, ctx: &CallContext, message: String);
    async fn ask(&self, ctx: &CallContext, question: String) -> String;
}
```

### 6. Secrets Management — Session Key Model

Agents **never** have direct access to secrets (API keys, credentials).
Instead:

```
┌─────────────────────────────────────────────┐
│ Agent (ouros sandbox)                        │
│                                              │
│  # The ONLY credential available:            │
│  SESSION_KEY = env("GW_SESSION_KEY")         │
│                                              │
│  # To perform a sensitive action:            │
│  result = tool.authorized_action(            │
│      action="web_search",                    │
│      params={"query": "rust async"},         │
│  )                                           │
│  # This calls through to our SDK, which      │
│  # resolves the session key → agent → org    │
│  # → retrieves the actual API key server-side│
└──────────────┬──────────────────────────────┘
               │ host function call (pauses sandbox)
               ▼
┌──────────────────────────────────────────────┐
│ Host Function Bridge (Rust)                   │
│                                               │
│  1. Validate session_key → resolve agent/org  │
│  2. Check agent's tool permissions            │
│  3. Retrieve actual secret from secure store  │
│  4. Execute the action with real credentials  │
│  5. Return sanitized result to sandbox        │
└──────────────────────────────────────────────┘
```

```rust
struct SessionKey(String);  // opaque token, one per agent session

/// Server-side secret resolution — agent never sees this
struct SecretStore {
    // org_id + secret_name → encrypted value
    // e.g. ("org_abc", "serpapi_key") → "sk-..."
}
```

The session key is:
- Generated when a session starts
- Scoped to one agent + one session
- The only env var injected into the ouros sandbox
- Used by our Python SDK to authenticate host function calls
- Revocable (kill a session → key becomes invalid)

### 7. Core Services

**LLM Gate → rl-play → Ollama**

LLM calls flow through rl-play as a transparent proxy to Ollama.
This gives us automatic session-grouped conversation capture for
fine-tuning — no extra instrumentation needed.

```
gw-llm crate                    rl-play proxy                Ollama
     │                               │                         │
     │  POST /v1/chat/completions    │                         │
     │  x-session-id: {session_id}   │                         │
     │ ─────────────────────────────▶│                         │
     │                               │  forward to Ollama      │
     │                               │────────────────────────▶│
     │                               │◀────────────────────────│
     │                               │  log to rl_play.db      │
     │◀──────────────────────────────│  (session + messages)   │
     │                               │                         │
```

The `gw-llm` crate:
- Routes to rl-play's proxy URL (not directly to Ollama)
- Sets `x-session-id` header = greatwheel's session ID
- This means rl-play automatically groups all LLM calls for a
  greatwheel session, enabling:
  - `rl-play export --format sft` → supervised fine-tuning data
  - `rl-play export --format dpo` → preference pairs from refinements
  - `rl-play train --method ppo` → RL fine-tuning with reward model
  - `rl-play publish` → push fine-tuned model back to Ollama

```rust
struct LlmGateConfig {
    /// Points to rl-play proxy, NOT directly to Ollama
    proxy_url: String,            // default: http://rl-play:8000
    default_model: String,        // e.g. "qwen2.5:7b"
    embedding_model: String,      // e.g. "nomic-embed-text"
    /// Embedding calls bypass rl-play (no fine-tuning value)
    ollama_direct_url: String,    // default: http://ollama:11434
}
```

**Memory Store (LanceDB + tantivy + Postgres)** — Hybrid search:
- **LanceDB**: vector similarity search via Ollama embeddings.
  Per-org table isolation (`memory_{org_id}`). Embedded, no external service.
- **tantivy**: true BM25 full-text search via an embedded tantivy index.
  Indexes key, text, org_id, and scope IDs (user, agent, session).
  Replaces PostgreSQL's `tsvector`/`ts_rank` for higher-quality keyword
  relevance scoring. Index lives on disk at a configurable path
  (`data/tantivy`). Supports `rebuild_from_rows()` to sync from Postgres
  on startup.
- **Postgres**: source of truth for memory values (JSONB), scope
  metadata, and upsert semantics via `ON CONFLICT`. No longer used for
  FTS — only persistence and value lookups.
- **Fusion**: Reciprocal Rank Fusion (RRF) merges vector + BM25 results.
  Ported from the Python `search_server.py` reference implementation.
- **Short-term**: ouros session state (Python variables).

All writes go to all three stores: Postgres + LanceDB concurrently
(`tokio::try_join!`), then tantivy (sync, fast).
Embedding via `OllamaClient::embed()` with batch processing (32),
truncation (8192 chars), individual retry on batch failure, and
zero-vector fallback.

```rust
struct RecallOpts {
    top_k: usize,
    mode: SearchMode,
    scope: MemoryScope,
}

enum SearchMode {
    Vector,
    FullText,
    Hybrid { alpha: f32 },
}

enum MemoryScope {
    Org,                    // search all org memory
    User(UserId),           // scoped + org-wide (NULL user_id)
    Agent(AgentId),         // scoped + org-wide (NULL agent_id)
    Session(SessionId),     // scoped + org-wide (NULL session_id)
}
```

**Message Bus** — Inter-agent and agent-human communication:
- `agent.call()` — synchronous, caller pauses, child span created.
- `agent.notify()` — async fire-and-forget, queued as new task.
- `channel.send()` / `channel.ask()` — talk to humans via channel.

**Snapshot Store** — Ouros snapshots persisted to Postgres.
Enables suspend/resume, crash recovery, idle eviction.

**Agent Registry** — Agent defs in Postgres + file watcher for
hot-reload + version history.

---

## Observability — OTel GenAI Tracing

Every operation emits OpenTelemetry spans using the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/).
Instrumentation uses the `tracing` crate with `#[instrument]` as the
single API surface, layered with `tracing-opentelemetry` for OTLP
export and a custom `PostgresTraceLayer` for database persistence.

### Architecture

The `gw-trace` crate provides `init_tracing(config, pg_pool)` which
builds a three-layer `tracing_subscriber::Registry`:

1. **`fmt::Layer` + `EnvFilter`** — always active, console output.
2. **`tracing_opentelemetry::Layer`** — OTLP gRPC export to Jaeger/etc
   (when `exporter = "otlp"` or `"both"`).
3. **`PostgresTraceLayer`** — fire-and-forget INSERT into the `traces`
   table for spans matching our naming conventions (when
   `postgres_export = true` and a `PgPool` is provided).

Configuration via `[tracing]` in `greatwheel.toml` or env vars in
`gw-bench`:

```toml
[tracing]
exporter = "console"               # "console", "otlp", or "both"
# otlp_endpoint = "http://localhost:4317"
postgres_export = false
service_name = "greatwheel"
```

### Instrumented Crates

**`gw-llm`** — `#[instrument]` on all LLM methods:
- `chat_with_options()` → span `gen_ai.chat` with model, token counts
  recorded from the Ollama response via `Span::current().record()`.
- `embed()` → span `gen_ai.embeddings` with model, batch size.
- `chat_stream()` → span `gen_ai.chat` with `gw.streaming = true`.

**`gw-runtime`** — `#[instrument]` on execution entry points:
- `run_agent()` → span `invoke_agent`.
- `ReplAgent::execute()` → span `repl.execute` with `gw.code_length`,
  `gw.is_final` (recorded when `FINAL()` is called).
- Host function dispatch → `info_span!("host_function", function = %name)`
  wrapping each bridge call in both sync and REPL execution paths.

**`gw-memory`** — `#[instrument]` on `HybridStore` methods:
- `store()` → span `memory.store` with `gw.memory.key`.
- `recall()` → span `memory.recall` with `gw.memory.search_mode`,
  `gw.memory.top_k`, `gw.memory.results_count`.
- `forget()` → span `memory.forget` with `gw.memory.key`.

**`gw-bench`** — Full rLM loop instrumentation:
- `run_single_query()` → span `rlm.question` with `gw.model`.
- `pre_search()` → span `rlm.pre_search`.
- `run_rlm_loop()` → span `rlm.loop` with `gw.max_iterations`.
- Per-iteration → `info_span!("rlm.iteration", n = iteration + 1)`.
- `BrowseCompBridge` methods → `host_function` spans for `search`,
  `get_document`, `llm_query`, `batch_llm_query`.

**`gw-server`** — Calls `gw_trace::init_tracing()` at startup,
`gw_trace::shutdown_tracing()` on exit. Postgres pool is connected
before tracing init so the trace layer can use it.

### PostgresTraceLayer

A `tracing_subscriber::Layer` that captures completed spans matching
these name prefixes: `gen_ai.*`, `memory.*`, `host_function`,
`invoke_agent`, `repl.*`, `rlm.*`. All other spans are ignored.

On span close:
- Extracts recorded fields (model, tokens, duration, custom attributes).
- Spawns a `tokio::spawn` fire-and-forget INSERT into the `traces` table.
- Uses `eprintln` (not `tracing`) for INSERT errors to avoid recursion.

### Span Hierarchy — rLM Benchmark

```
rlm.question{query_id, gw.model}
├── rlm.pre_search
│   ├── gen_ai.chat{model, input_tokens, output_tokens}  (decompose query)
│   ├── host_function{function="search"} ×5
│   ├── gen_ai.chat{...}                                 (refine)
│   └── host_function{function="search"} ×3
├── rlm.loop{gw.max_iterations}
│   ├── rlm.iteration{n=1}
│   │   ├── gen_ai.chat{model, input_tokens, output_tokens}
│   │   ├── repl.execute{gw.code_length}
│   │   │   ├── host_function{function="search"}
│   │   │   ├── host_function{function="get_document"}
│   │   │   └── host_function{function="llm_query"}
│   │   │       └── gen_ai.chat{...}
│   │   └── repl.execute{...}
│   ├── rlm.iteration{n=2}
│   │   └── ...
│   └── rlm.iteration{n=3}
│       └── repl.execute{gw.is_final=true}
```

### Span Hierarchy — Server Agent Invocation

```
invoke_agent "triage"                          (gen_ai.agent.name = "triage")
├── gen_ai.chat{model, input_tokens, output_tokens}
├── invoke_agent "research"                    (child agent call)
│   ├── gen_ai.chat{...}
│   ├── memory.recall{search_mode, top_k}
│   │   └── gen_ai.embeddings{batch_size}      (query embedding)
│   ├── gen_ai.chat{...}
│   └── memory.store{key}
│       └── gen_ai.embeddings{batch_size}      (value embedding)
└── channel.send                               (reply to user)
```

### Key Attributes

| Attribute | Example | Set by |
|-----------|---------|--------|
| `gen_ai.operation.name` | `chat`, `embeddings` | `gw-llm` |
| `gen_ai.request.model` | `"qwen2.5:7b"` | `gw-llm` |
| `gen_ai.provider.name` | `"ollama"` | `gw-llm` |
| `gen_ai.usage.input_tokens` | `1200` | `gw-llm` (from response) |
| `gen_ai.usage.output_tokens` | `85` | `gw-llm` (from response) |
| `gw.streaming` | `true` | `gw-llm` (chat_stream) |
| `gw.batch_size` | `32` | `gw-llm` (embed) |
| `gw.code_length` | `245` | `gw-runtime` |
| `gw.is_final` | `true` | `gw-runtime` |
| `gw.memory.key` | `"user_prefs"` | `gw-memory` |
| `gw.memory.search_mode` | `Hybrid { alpha: 0.5 }` | `gw-memory` |
| `gw.memory.top_k` | `5` | `gw-memory` |
| `gw.model` | `"qwen2.5:7b"` | `gw-bench` |
| `gw.max_iterations` | `12` | `gw-bench` |

### Verification

```bash
# Console spans — hierarchical output with token counts and timing
RUST_LOG=info cargo run --bin gw-bench -- [args]

# OTLP — send to Jaeger
docker run -p 16686:16686 -p 4317:4317 jaegertracing/all-in-one
GW_TRACE_EXPORTER=both GW_TRACE_OTLP_ENDPOINT=http://localhost:4317 \
  cargo run --bin gw-bench -- [args]
# View traces at http://localhost:16686

# Postgres export
# Set postgres_export = true in greatwheel.toml, then:
SELECT operation_name, model, input_tokens, output_tokens, duration_ms
FROM traces ORDER BY created_at DESC LIMIT 20;
```

### Trace Recording → Fine-Tuning Pipeline

Two complementary recording paths:

**Path 1: OTel traces → Postgres**
Structured spans with GenAI attributes. Good for querying agent behavior,
identifying failures, reviewing decision trees. Exportable as JSONL for
analysis.

**Path 2: rl-play session capture**
Raw LLM conversations captured at the proxy layer, grouped by session ID.
This is the primary fine-tuning data source:

```
greatwheel traces (OTel)     rl-play sessions
┌──────────────────────┐     ┌──────────────────────┐
│ Structured spans     │     │ Raw conversations    │
│ Agent decision tree  │     │ Multi-turn sessions  │
│ Token usage metrics  │     │ Input/output pairs   │
│ Latency data         │     │                      │
└──────────┬───────────┘     └──────────┬───────────┘
           │                            │
           ▼                            ▼
┌──────────────────────┐     ┌──────────────────────┐
│ Behavior analysis    │     │ rl-play export       │
│ Failure detection    │     │   --format sft|dpo   │
│ Prompt improvement   │     │                      │
└──────────────────────┘     │ rl-play train        │
                             │   --method sft|dpo|ppo│
                             │                      │
                             │ rl-play publish      │
                             │   → new Ollama model │
                             └──────────────────────┘
```

The feedback loop: observe agent behavior (OTel traces) → identify
what to improve → export training data (rl-play) → fine-tune →
publish improved model → agents automatically use it.

---

## Rate Limiting

```rust
struct RateLimitState {
    org_id: OrgId,
    user_id: Option<UserId>,
    tokens_used: AtomicU64,
    period_start: DateTime<Utc>,
}

enum RateLimitResult {
    /// Under soft limit — proceed normally
    Ok,
    /// Over soft limit — proceed but emit warning
    SoftLimitExceeded { used: u64, soft_limit: u64 },
    /// Over hard limit — reject the request
    HardLimitExceeded { used: u64, hard_limit: u64 },
}
```

Rate limits are checked at two levels:
1. **Org level**: total token budget across all users and agents.
2. **User level**: per-user budget (defaults from org, with overrides).

Token counts come from Ollama's response (approximate is fine — we're
not billing by tokens yet). Limits reset on a configurable period
(daily/weekly/monthly per org settings).

---

## Postgres Schema

```sql
-- Tenant isolation
CREATE TABLE orgs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name TEXT NOT NULL UNIQUE,
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,
    email TEXT,
    default_permissions JSONB NOT NULL DEFAULT '{}',
    rate_limits JSONB,  -- per-user overrides, nullable = use org defaults
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, email)
);

-- Agent definitions with versioning
CREATE TABLE agent_defs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    source_type TEXT NOT NULL,        -- 'bare', 'file', 'inline', 'git'
    source_ref TEXT,                  -- path, inline code, or git ref (null for bare)
    tool_permissions JSONB NOT NULL,
    model_config JSONB NOT NULL,
    resource_limits JSONB NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name)
);

CREATE TABLE agent_versions (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agent_defs(id),
    version INTEGER NOT NULL,
    source_snapshot TEXT NOT NULL,     -- full Python source at this version
    trigger TEXT NOT NULL,            -- 'file_watch', 'api_update', 'git_sync', 'rollback'
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (agent_id, version)
);

-- Sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    user_id UUID NOT NULL REFERENCES users(id),
    agent_id UUID NOT NULL REFERENCES agent_defs(id),
    session_key TEXT NOT NULL UNIQUE,  -- opaque token for agent auth
    snapshot BYTEA,
    status TEXT NOT NULL DEFAULT 'active',  -- active, suspended, evicted
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_active_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Task log
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    user_id UUID NOT NULL REFERENCES users(id),
    session_id UUID REFERENCES sessions(id),
    parent_task_id UUID REFERENCES tasks(id),
    channel TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',
    result JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

-- OTel trace storage
CREATE TABLE traces (
    id BIGSERIAL PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL UNIQUE,
    parent_span_id TEXT,
    operation_name TEXT NOT NULL,
    org_id UUID NOT NULL REFERENCES orgs(id),
    agent_id UUID REFERENCES agent_defs(id),
    session_id UUID REFERENCES sessions(id),
    model TEXT,
    provider TEXT,
    input_messages JSONB,
    output_messages JSONB,
    input_tokens INTEGER,
    output_tokens INTEGER,
    duration_ms BIGINT NOT NULL,
    status TEXT NOT NULL,
    attributes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX idx_traces_org_agent ON traces(org_id, agent_id, created_at);
CREATE INDEX idx_traces_trace_id ON traces(trace_id);

-- Org-level secrets (agent never sees these directly)
CREATE TABLE org_secrets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,              -- e.g. "serpapi_key"
    encrypted_value BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name)
);

-- Hybrid memory store (LanceDB vectors + Postgres FTS)
CREATE TABLE memories (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID NOT NULL REFERENCES orgs(id),
    user_id      UUID,
    agent_id     UUID,
    session_id   UUID,
    key          TEXT NOT NULL,
    value        JSONB NOT NULL,
    text_content TEXT NOT NULL,
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text_content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, key)
);
CREATE INDEX idx_memories_tsv ON memories USING GIN(tsv);
CREATE INDEX idx_memories_org ON memories(org_id);
CREATE INDEX idx_memories_org_user ON memories(org_id, user_id);
CREATE INDEX idx_memories_org_agent ON memories(org_id, agent_id);
CREATE INDEX idx_memories_org_session ON memories(org_id, session_id);

-- Rate limit tracking
CREATE TABLE rate_limit_counters (
    org_id UUID NOT NULL REFERENCES orgs(id),
    user_id UUID REFERENCES users(id),
    period_start TIMESTAMPTZ NOT NULL,
    tokens_used BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, COALESCE(user_id, '00000000-0000-0000-0000-000000000000'), period_start)
);
```

---

## Crate Structure

```
greatwheel/
├── Cargo.toml                 # workspace root
├── crates/
│   ├── gw-core/               # OrgId, UserId, Task, AgentDef, permissions,
│   │                          # CallContext, SessionKey, rate limit types
│   ├── gw-runtime/            # ouros integration, session manager,
│   │                          # host function bridge, hot-reload,
│   │                          # snapshots, session lifecycle (idle eviction)
│   ├── gw-llm/                # rl-play proxy client, Ollama direct client,
│   │                          # provider trait, token tracking, embed()
│   ├── gw-memory/             # LanceDB + tantivy + Postgres hybrid store,
│   │                          # embedding pipeline, BM25 via tantivy,
│   │                          # RRF fusion, search modes, scope filtering
│   ├── gw-bus/                # message bus, agent.call/notify routing
│   ├── gw-channels/           # channel adapters (HTTP, WS, CLI, Slack)
│   ├── gw-scheduler/          # task queue, per-org concurrency, retries,
│   │                          # rate limit enforcement
│   ├── gw-trace/              # OTel GenAI instrumentation, init_tracing(),
│   │                          # PostgresTraceLayer, OTLP exporter, config
│   └── gw-server/             # binary — wires everything, HTTP/WS API,
│                              # auth middleware, config loading,
│                              # agent version management API
├── agents/                    # agent definitions (Python source + config)
│   ├── triage.py
│   ├── research.py
│   └── agents.toml
├── migrations/                # Postgres schema (sqlx migrations)
├── config/
│   └── greatwheel.toml
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── README.md
```

---

## Docker Compose Topology

```yaml
services:
  greatwheel:
    build: .
    ports:
      - "8080:8080"       # HTTP API
      - "8081:8081"       # WebSocket
    environment:
      DATABASE_URL: postgres://gw:gw@postgres:5432/greatwheel
      RLPLAY_URL: http://rl-play:8000    # LLM calls go here
      OLLAMA_URL: http://ollama:11434    # embeddings go direct
      LANCE_DATA_DIR: /data/lance
    volumes:
      - lance-data:/data/lance
      - ./agents:/agents
    depends_on:
      - postgres
      - rl-play
      - ollama

  rl-play:
    build:
      context: ../rl-play        # sibling project
    ports:
      - "8000:8000"
    environment:
      OLLAMA_BASE_URL: http://ollama:11434
    volumes:
      - rl-play-data:/data
    depends_on:
      - ollama

  postgres:
    image: postgres:17
    environment:
      POSTGRES_DB: greatwheel
      POSTGRES_USER: gw
      POSTGRES_PASSWORD: gw
    volumes:
      - pg-data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama-models:/root/.ollama
    # deploy:
    #   resources:
    #     reservations:
    #       devices:
    #         - capabilities: [gpu]

volumes:
  pg-data:
  lance-data:
  ollama-models:
  rl-play-data:
```

---

## Implementation Status

### Complete
1. ~~`cargo init` workspace + crate stubs + `docker-compose.yml`.~~
2. ~~Postgres migrations for the core schema.~~ — 5 migration files (orgs/users, agent_defs/versions, sessions/tasks, traces, secrets/rate_limits).
3. ~~Integrate ouros — spike the session → pause → resume loop.~~ — `ReplAgent` in `gw-runtime` with persistent REPL sessions, `HostBridge` trait for host function dispatch, `FINAL()` interception, variable injection/retrieval.
4. ~~Ollama client (`gw-llm`) via rl-play proxy — `complete()` + `embed()`.~~ — Non-streaming and streaming chat, optional `think` parameter, per-call model override.
5. ~~Wire up a minimal rLM agent.~~ — Implemented in `gw-bench` for BrowseComp-Plus: rLM writes code → ouros executes → host functions (`search`, `llm_query`, `get_document`) pause → Rust fulfills → resume → response. Full benchmark harness with multi-run voting and trajectory recording.
6. ~~LanceDB + tantivy + Postgres hybrid memory search (`gw-memory`).~~ — `HybridStore` implementation: LanceDB for vector search (per-org tables, Ollama embeddings via `OllamaClient::embed()`), tantivy for true BM25 full-text search (replaces Postgres `tsvector`/`ts_rank`), Postgres for value persistence and scope metadata, RRF fusion. `TantivyStore` with on-disk index, org/scope filtering via boolean queries, `rebuild_from_rows()` for startup sync. `MemoryStore` trait, `MemoryError` (thiserror), `MemoryRecord` return type. Concurrent triple-store writes/deletes. Migration `006_memory.sql`.
10. ~~Channel layer (HTTP API first).~~ — `gw-server` serves HTTP via Axum: `/api/chat` (streaming SSE), `/api/models`, `/api/config`, `/health`. Embedded chat UI with model selector, system prompt editor, token display.

7. ~~OTel instrumentation with GenAI semconv + Postgres trace export (`gw-trace`).~~ — Three-layer tracing subscriber (console + OTLP + Postgres). `#[instrument]` spans on `gw-llm` (chat/embed/stream with token recording), `gw-runtime` (agent/REPL execution + host function dispatch), `gw-memory` (store/recall/forget), `gw-bench` (full rLM loop hierarchy). `PostgresTraceLayer` captures matching spans to `traces` table. Config via `[tracing]` TOML section or env vars.

### Trait/type definitions only
- `gw-bus` — `AgentBus` trait (`call`/`notify`). No concrete implementation.
- `gw-channels` — `ChannelAdapter` trait. No HTTP/WS/Slack adapter implementations.
- `gw-scheduler` — `Scheduler`, `RateLimiter` structs, `RateLimitResult` enum. No queue or enforcement logic.

### Next Steps
8. Session lifecycle (idle timeout → snapshot → evict).
9. Agent versioning + hot-reload.
11. Rate limiting — soft + hard (`gw-scheduler`).
12. Session key auth model.
13. Inter-agent message bus — concrete `AgentBus` implementation (`gw-bus`).
14. Agent SDK — Python-side `llm`, `memory`, `agent`, `channel` globals wired to host bridge.
15. Multi-tenancy auth middleware.
16. Episodic memory + thread weaving (see [`docs/design-episodes.md`](docs/design-episodes.md)).
