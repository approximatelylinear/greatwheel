# Greatwheel — Project Conventions

## What is this?
An rLM-coordinated agentic runtime where Python agents execute inside ouros sandboxes,
with a Rust core managing memory, routing, and inter-agent communication.

## Architecture
See `ARCHITECTURE.md` for the full design. Key points:
- 9-crate Rust workspace under `crates/`
- Postgres for persistence, LanceDB for vector search
- Ollama (via rl-play proxy) for LLM inference
- ouros for sandboxed Python agent execution

## Crate layout
- `gw-core` — shared types (IDs, Task, AgentDef, CallContext, permissions)
- `gw-runtime` — ouros integration, session management
- `gw-llm` — LLM provider trait + Ollama client (via rl-play)
- `gw-memory` — hybrid memory (LanceDB + Postgres)
- `gw-bus` — inter-agent message bus
- `gw-channels` — channel adapters (HTTP, WS, etc.)
- `gw-scheduler` — task queue + rate limiting
- `gw-trace` — OTel GenAI tracing
- `gw-server` — binary that wires everything together

## Conventions
- Use workspace dependencies (define versions in root `Cargo.toml`)
- Types shared across crates go in `gw-core`
- All async code uses `tokio`
- Tracing via the `tracing` crate (not `log`)
- Database access via `sqlx` with compile-time query checking where possible
- Prefer `thiserror` for error types when adding error handling

## Running
```bash
# Dev (needs Postgres + Ollama running)
cargo run --bin greatwheel -- --config config/greatwheel.toml

# Docker
docker compose -f docker/docker-compose.yml up
```

## Testing
```bash
cargo test --workspace
```
