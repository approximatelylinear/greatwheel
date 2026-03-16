# Greatwheel

An RLM-coordinated agentic runtime where Python agents execute inside sandboxed [ouros](https://crates.io/crates/ouros) sessions, with a Rust core managing memory, routing, and inter-agent communication — multi-tenant, observable, and fine-tunable.

## Core Idea

A **Recursive Language Model** (rLM) is not a special model — it's an inference strategy. The model receives a task but never sees the full context directly. Instead, context lives as Python variables in a sandboxed REPL. The model writes code to search, partition, transform, and recursively sub-query that context.

Ouros gives us the execution substrate: a sandboxed Python runtime that **pauses on external function calls** and produces serializable snapshots. Every host function call (LLM, memory, inter-agent) is a pause point where the Rust runtime takes over.

```mermaid
sequenceDiagram
    participant User
    participant Channel as Channel Layer
    participant Sched as Scheduler
    participant Ouros as Ouros Session
    participant Bridge as Host Bridge
    participant LLM as rl-play → Ollama
    participant Mem as Memory Store

    User->>Channel: Send message
    Channel->>Sched: Create Task
    Sched->>Ouros: Dispatch to agent session

    loop rLM execution loop
        Ouros->>Ouros: Execute Python code
        Ouros-->>Bridge: llm.complete() — PAUSE
        Bridge->>LLM: Forward completion request
        LLM-->>Bridge: Response + tokens
        Bridge-->>Ouros: RESUME with result
    end

    Ouros-->>Bridge: memory.store() — PAUSE
    Bridge->>Mem: Persist to Lance + Postgres
    Mem-->>Bridge: OK
    Bridge-->>Ouros: RESUME

    Ouros->>Channel: channel.send(response)
    Channel->>User: Reply
```

## Architecture

```mermaid
graph TB
    subgraph Channel["Channel Layer"]
        HTTP["HTTP API :8080"]
        WS["WebSocket :8081"]
        CLI["CLI"]
        Webhooks["Webhooks"]
    end

    subgraph Scheduler["Router / Scheduler"]
        Triage["Triage Agent"]
        Queue["Task Queue"]
    end

    subgraph Runtime["Agent Runtime"]
        SA["Session A<br/><i>ouros sandbox</i>"]
        SB["Session B<br/><i>ouros sandbox</i>"]
        HB["Host Function Bridge"]
    end

    subgraph Services["Core Services"]
        LLMGate["LLM Gate"]
        Memory["Memory Store<br/><i>Lance + PG</i>"]
        Bus["Message Bus"]
        Registry["Agent Registry"]
        Tracer["Trace Recorder<br/><i>OTel GenAI</i>"]
    end

    RLPlay["rl-play<br/><i>fine-tuning proxy</i>"]
    Ollama["Ollama<br/><i>Qwen 2.5</i>"]
    PG[("PostgreSQL")]

    HTTP & WS & CLI & Webhooks --> Triage
    Triage --> Queue
    Queue --> SA & SB
    SA & SB -->|pause| HB
    HB --> LLMGate & Memory & Bus
    LLMGate -->|chat| RLPlay
    LLMGate -->|embed| Ollama
    RLPlay --> Ollama
    Memory & Registry & Tracer --> PG

    style Channel fill:#4A9EFF15,stroke:#4A9EFF
    style Scheduler fill:#FF8C4215,stroke:#FF8C42
    style Runtime fill:#4ADE8015,stroke:#4ADE80
    style Services fill:#A78BFA15,stroke:#A78BFA
    style RLPlay fill:#FBBF2415,stroke:#FBBF24
    style Ollama fill:#F8717115,stroke:#F87171
    style PG fill:#22D3EE15,stroke:#22D3EE
```

## Crate Structure

```mermaid
graph BT
    core["gw-core<br/><i>shared types</i>"]
    runtime["gw-runtime<br/><i>ouros integration</i>"]
    llm["gw-llm<br/><i>LLM provider</i>"]
    memory["gw-memory<br/><i>hybrid memory</i>"]
    bus["gw-bus<br/><i>message bus</i>"]
    channels["gw-channels<br/><i>channel adapters</i>"]
    scheduler["gw-scheduler<br/><i>task queue</i>"]
    trace["gw-trace<br/><i>OTel tracing</i>"]
    server["gw-server<br/><i>binary</i>"]

    runtime --> core
    llm --> core
    memory --> core
    bus --> core
    channels --> core
    scheduler --> core
    trace --> core
    server --> runtime & llm & memory & bus & channels & scheduler & trace

    style core fill:#A78BFA22,stroke:#A78BFA
    style server fill:#ffffff15,stroke:#888
```

| Crate | Purpose | Key Types |
|-------|---------|-----------|
| `gw-core` | Shared vocabulary | `OrgId`, `UserId`, `Task`, `AgentDef`, `CallContext`, `ToolPermissions` |
| `gw-runtime` | Ouros session management | `SessionManager`, `HostFunctionBridge` |
| `gw-llm` | LLM provider (via rl-play) | `LlmGate`, `OllamaClient`, `CompletionRequest` |
| `gw-memory` | Hybrid search (Lance + PG) | `MemoryStore`, `HybridStore`, `SearchMode` |
| `gw-bus` | Inter-agent communication | `AgentBus` |
| `gw-channels` | Inbound/outbound adapters | `ChannelAdapter` |
| `gw-scheduler` | Task queue + rate limiting | `Scheduler`, `RateLimiter` |
| `gw-trace` | OTel GenAI instrumentation | `TraceRecorder`, `TraceRecord` |
| `gw-server` | Wires everything, serves API | `main()`, config, HTTP routes |

## Data Model

```mermaid
erDiagram
    orgs ||--o{ users : has
    orgs ||--o{ agent_defs : owns
    orgs ||--o{ org_secrets : stores
    orgs ||--o{ rate_limit_counters : tracks
    agent_defs ||--o{ agent_versions : versions
    users ||--o{ sessions : opens
    agent_defs ||--o{ sessions : runs_in
    sessions ||--o{ tasks : contains
    tasks ||--o{ tasks : parent
    orgs ||--o{ traces : recorded_for

    orgs {
        uuid id PK
        text name UK
        jsonb settings
    }
    users {
        uuid id PK
        uuid org_id FK
        text name
        text email
        jsonb default_permissions
    }
    agent_defs {
        uuid id PK
        uuid org_id FK
        text name
        text system_prompt
        text source_type
        jsonb tool_permissions
        int current_version
    }
    sessions {
        uuid id PK
        uuid org_id FK
        uuid user_id FK
        uuid agent_id FK
        text session_key UK
        bytea snapshot
        text status
    }
    tasks {
        uuid id PK
        uuid session_id FK
        text channel
        text payload
        text status
        jsonb result
    }
    traces {
        bigint id PK
        text trace_id
        text span_id UK
        text operation_name
        int input_tokens
        int output_tokens
    }
```

## Docker Topology

```mermaid
graph LR
    subgraph compose["docker-compose"]
        GW["greatwheel<br/>:8080 :8081"]
        RL["rl-play<br/>:8000"]
        PG["postgres:17<br/>:5432"]
        OL["ollama<br/>:11434"]
    end

    GW -->|depends_on| RL & PG & OL
    RL -->|depends_on| OL
    GW -.->|DATABASE_URL| PG
    GW -.->|RLPLAY_URL| RL
    RL -.->|OLLAMA_BASE_URL| OL

    lance["lance-data"] -.-> GW
    pgdata["pg-data"] -.-> PG
    models["ollama-models"] -.-> OL

    style GW fill:#4ADE8022,stroke:#4ADE80
    style RL fill:#FBBF2422,stroke:#FBBF24
    style PG fill:#22D3EE22,stroke:#22D3EE
    style OL fill:#F8717122,stroke:#F87171
```

## Fine-Tuning Pipeline

All LLM calls flow through **rl-play**, a transparent proxy that captures conversations grouped by session ID:

```mermaid
flowchart LR
    A["Agent calls<br/>llm.complete()"] --> B["gw-llm<br/>+ x-session-id header"]
    B --> C["rl-play proxy<br/><i>captures & forwards</i>"]
    C --> D["Ollama"]
    D --> C --> B --> A

    C --> E["rl-play.db<br/><i>session-grouped<br/>conversations</i>"]
    E --> F["rl-play export<br/>--format sft|dpo"]
    F --> G["rl-play train<br/>--method sft|dpo|ppo"]
    G --> H["rl-play publish<br/>→ new Ollama model"]
    H -.->|agents use<br/>improved model| D

    style C fill:#FBBF2422,stroke:#FBBF24
    style E fill:#A78BFA22,stroke:#A78BFA
```

## Quickstart

**Prerequisites:** Rust (stable), Ollama running locally with a model pulled (`ollama pull qwen2.5:7b`)

```bash
# Run the server (Postgres optional for dev — chat works without it)
cargo run --bin greatwheel -- --config config/greatwheel.toml

# Open the chat UI
open http://localhost:8090
```

**With Docker Compose** (full stack):

```bash
docker compose -f docker/docker-compose.yml up
```

## Development

```bash
# Check all crates compile
cargo check --workspace

# Run tests
cargo test --workspace

# Run with debug logging
RUST_LOG=debug cargo run --bin greatwheel -- --config config/greatwheel.toml
```

### Build Requirements

LanceDB requires `protoc`:

```bash
# Debian/Ubuntu
apt-get install protobuf-compiler

# Or install from release
curl -sL https://github.com/protocolbuffers/protobuf/releases/download/v29.3/protoc-29.3-linux-x86_64.zip -o protoc.zip
unzip protoc.zip -d ~/.local
```

## Project Status

The architecture is defined in [`ARCHITECTURE.md`](ARCHITECTURE.md). Current state:

### Implemented
- [x] Workspace with 9 crates + `gw-bench` and correct dependencies
- [x] Core types — `OrgId`, `Task`, `AgentDef`, `CallContext`, `ToolPermissions`, etc. (`gw-core`)
- [x] Ouros integration — `ReplAgent` with persistent REPL sessions, host function bridge via `HostBridge` trait, `FINAL()` interception, variable injection/retrieval (`gw-runtime`)
- [x] Ollama client — non-streaming, streaming (SSE), optional `think` parameter for reasoning models, model override per call (`gw-llm`)
- [x] HTTP server — Axum with `/api/chat` (streaming SSE), `/api/models`, `/api/config`, `/health` (`gw-server`)
- [x] Chat UI — embedded `chat.html` with dark theme, model selector, system prompt editor, real-time token display
- [x] Hybrid search — BM25s (sparse) + LanceDB/Ollama (dense vector) with RRF fusion, HTTP search server, index builders with resume support (`bench/browsecomp/`)
- [x] BrowseComp-Plus benchmark harness — rLM REPL agent with `search()`, `get_document()`, `llm_query()`, `batch_llm_query()` host functions, multi-run voting, trajectory recording (`gw-bench`)
- [x] Postgres migrations — 5 files: orgs/users, agent_defs/versions, sessions/tasks, traces, secrets/rate_limits
- [x] Docker Compose — 4 services (greatwheel, rl-play, postgres, ollama) + Dockerfiles
- [x] Config — `greatwheel.toml` with server, database, LLM, agents, session sections
- [x] Architecture explorer ([`greatwheel-explorer.html`](greatwheel-explorer.html))

### Trait/type definitions only (no implementation)
- [ ] Hybrid memory as a Rust crate — types defined (`RecallOpts`, `SearchMode`, `MemoryScope`), `MemoryStore` trait defined, no Rust LanceDB/Postgres integration yet (`gw-memory`). Working Python implementation exists in `bench/browsecomp/` (BM25s + LanceDB + RRF fusion)
- [ ] Inter-agent message bus — `AgentBus` trait defined (`call`/`notify`), no concrete implementation (`gw-bus`)
- [ ] Channel adapters — `ChannelAdapter` trait defined, no HTTP/WS/Slack implementations (`gw-channels`)
- [ ] Task scheduler — `Scheduler`/`RateLimiter` structs stubbed, `RateLimitResult` enum defined, no queue or enforcement (`gw-scheduler`)
- [ ] OTel tracing — `TraceRecord` struct defined, no Postgres persistence or OTLP export (`gw-trace`)

### Not started
- [ ] Agent SDK integration — `triage.py` references SDK that doesn't exist yet
- [ ] Agent hot-reload + versioning
- [ ] Session lifecycle (idle timeout → snapshot → evict)
- [ ] Multi-tenancy auth middleware
- [ ] Session key auth model

## License

TBD
