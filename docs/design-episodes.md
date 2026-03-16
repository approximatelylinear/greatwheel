# Design: Slate Integration — Episodes, Thread Weaving, and Adaptive Routing

**Status:** Draft
**Date:** 2026-03-14
**Motivation:** [Slate: moving beyond ReAct and RLM](https://randomlabs.ai/blog/slate) — Random Labs

---

## 1. Context

Random Labs' Slate post presents a taxonomy of agent architectures — ReAct,
markdown planning, task trees, RLM, Devin/Manus/Altera, Claude Code/Codex —
and proposes **Thread Weaving** as a unifying pattern that combines the
expressivity of ReAct with the context isolation of subagents and the
decomposition flexibility of RLM.

Greatwheel is built on the **rLM paradigm**: agents write Python in ouros
sandboxes, context lives as Python variables (not in the LLM context window),
and host function calls (`llm.complete()`, `memory.recall()`, `agent.call()`)
are synchronization points where the Rust runtime takes over.

This document describes how to integrate Slate's key insights into
greatwheel without abandoning the rLM foundation.

### 1.1 Where We Sit in Slate's Taxonomy

| Property | Slate | Greatwheel (current) | Greatwheel (proposed) |
|---|---|---|---|
| Planning | implicit (orchestrator) | implicit (rLM writes code) | unchanged |
| Decomposition | implicit (thread dispatch) | REPL + `agent.call()` | + episode composability |
| Synchronization | per-episode | per-host-call (ouros pause) | + episode boundaries |
| Context isolation | per-thread | per-session (ouros) | unchanged |
| Context compaction | episode compress | none (variables persist) | + episode summaries |
| Parallel execution | native | `agent.notify()` only | + parallel `agent.call()` |
| Expressivity | high | high (Python REPL) | unchanged |
| Model routing | per-thread | per-agent | + per-call override |

### 1.2 What We Already Do Well

Slate's post identifies several problems. Our architecture already addresses
some of them in ways Slate does not:

- **Ouros pause points** give us finer-grained synchronization than Slate's
  per-episode model. Every host function call is a sync boundary where the
  runtime can inspect state, enforce limits, and record traces.

- **Context lives in Python variables**, not the LLM context window. The
  rLM never sees the full context — it writes code to query it. This
  sidesteps the "dumb zone" / context rot problem (Hong, Troynikov & Huber,
  2025) that motivates much of Slate's design.

- **rl-play fine-tuning loop.** We can extract SFT/DPO/PPO training data
  from agent traces and push improved models back to Ollama. Slate has no
  equivalent — they rely entirely on frontier API models.

- **Serializable snapshots.** Ouros sessions can be checkpointed, evicted,
  and restored. Agents survive restarts. Slate threads are ephemeral.

### 1.3 What We're Missing

Three gaps that Slate exposes:

1. **No episodic memory.** `agent.call()` returns a `serde_json::Value`.
   The full trace of what the child did is captured in OTel spans but never
   compressed into a reusable summary. The calling agent gets a result but
   no structured history. Future agents get nothing.

2. **No recursion guard.** `agent.call()` allows unbounded recursive
   spawning. The rLM paper itself was limited to depth=1. Slate notes that
   unbounded recursion leads to overdecomposition.

3. **Single model per agent.** `OllamaClient::chat()` already accepts an
   optional model override, but the rLM has no Python-side API to use it,
   and there's no allowlist to constrain which models an agent may use.

---

## 2. Design

### 2.1 Episodes

An episode is a compressed, structured summary of work done by an agent
during a task. It captures what was accomplished, what was learned, and
what failed — not the step-by-step trace.

#### 2.1.1 Type Definition

```rust
// gw-core/src/episode.rs

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EpisodeId(pub Uuid);

/// Compressed summary of work done during a task or sub-task.
/// Produced on sub-agent completion or explicit checkpoint.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Episode {
    pub id: EpisodeId,
    pub task_id: TaskId,
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub org_id: OrgId,

    /// The original task payload — what was requested.
    pub objective: String,

    /// LLM-generated summary of what was accomplished.
    pub summary: String,

    /// Structured outcomes.
    pub artifacts: Vec<Artifact>,

    /// What didn't work — so downstream agents avoid repeating failures.
    pub failures: Vec<String>,

    /// Token cost of the work this episode represents.
    pub token_cost: TokenCost,

    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Artifact {
    pub kind: ArtifactKind,
    pub description: String,
    /// Pointer to stored data (memory key, file path, variable name).
    pub reference: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ArtifactKind {
    /// Information discovered.
    Finding,
    /// Choice made, with rationale in `description`.
    Decision,
    /// File or code modified.
    CodeChange,
    /// Data persisted to gw-memory.
    MemoryStored,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TokenCost {
    pub input_tokens: u64,
    pub output_tokens: u64,
}
```

#### 2.1.2 Episode Generation

Episodes are generated at two points:

**a) Sub-agent completion.** When `agent.call()` returns, the runtime
generates an episode by calling the LLM with a compressed span tree.

The span tree is extracted from `gw-trace` and pruned to operation names,
durations, error status, and key attributes — not full message content.
This keeps the summarization call small.

```
System: Summarize the work done by an agent. Given the objective and
action trace, produce a JSON episode: { summary, artifacts, failures }.
Include key findings, decisions, and what failed. Be concise.

User: Objective: {task.payload}
Actions: {compressed_spans}
Result: {agent_result}
```

The summarization model should be the cheapest available (see §2.3 for
model routing). A bad summary wastes tokens in downstream context; a
missing summary loses information entirely. Err on the side of generating
one.

**Skip condition:** If the sub-agent completed in a single turn with
fewer than 500 total tokens, skip summarization and use the raw result
as the episode summary. The overhead isn't worth it for trivial tasks.

**b) Explicit checkpoint.** The rLM calls `episode.checkpoint()` to
snapshot progress without ending the session:

```python
episode.checkpoint(
    summary="Analyzed 3/5 modules. Circular dep found in auth ↔ user.",
    artifacts=[{"kind": "finding", "description": "auth <-> user circular"}],
)
```

This is a host function call, so it pauses ouros and the Rust runtime
persists the episode.

#### 2.1.3 Episode Storage

Dual storage:

- **Postgres** — structured, queryable by task/agent/session/org/time.
- **LanceDB** — summary field embedded for vector search via `gw-memory`,
  so agents can semantically recall relevant past episodes.

```sql
-- New migration: episodes table

CREATE TABLE episodes (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    task_id     UUID NOT NULL REFERENCES tasks(id),
    agent_id    UUID NOT NULL REFERENCES agent_defs(id),
    session_id  UUID NOT NULL REFERENCES sessions(id),
    org_id      UUID NOT NULL REFERENCES orgs(id),
    objective   TEXT NOT NULL,
    summary     TEXT NOT NULL,
    artifacts   JSONB NOT NULL DEFAULT '[]',
    failures    JSONB NOT NULL DEFAULT '[]',
    input_tokens  BIGINT NOT NULL DEFAULT 0,
    output_tokens BIGINT NOT NULL DEFAULT 0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_episodes_task  ON episodes(task_id);
CREATE INDEX idx_episodes_agent ON episodes(agent_id, created_at);
CREATE INDEX idx_episodes_org   ON episodes(org_id, created_at);
```

LanceDB table `episodes_{org_id}` stores `(episode_id, summary_vector)`
alongside the existing memory tables, searchable via `memory.recall()`
with a new `MemoryScope::Episodes`.

### 2.2 Episode Composability

This is the core Slate insight: episodes are not just return values —
they're composable context that flows between agents.

#### 2.2.1 Changes to AgentBus

```rust
// gw-bus/src/lib.rs

use gw_core::{AgentId, CallContext, Episode, Task};

pub trait AgentBus {
    /// Synchronous call with optional episode context from prior work.
    async fn call(
        &self,
        ctx: &CallContext,
        agent: AgentId,
        task: Task,
        context_episodes: Vec<Episode>,
    ) -> Result<AgentResult, AgentError>;

    /// Fire-and-forget notification (unchanged).
    async fn notify(
        &self,
        ctx: &CallContext,
        agent: AgentId,
        message: serde_json::Value,
    ) -> Result<(), AgentError>;
}

/// Result of agent.call() — now includes the episode.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentResult {
    /// The agent's direct response.
    pub response: serde_json::Value,
    /// Compressed summary of the work done.
    pub episode: Option<Episode>,
}
```

#### 2.2.2 Context Injection

When `context_episodes` is non-empty, the runtime injects them into the
child agent's ouros session as a Python variable before execution begins:

```python
# Injected by gw-runtime before agent code runs
_prior_work = [
    {
        "agent": "research",
        "objective": "Find Stripe rate limit documentation",
        "summary": "Stripe rate limits: 100 req/s test, 10k req/s live...",
        "artifacts": [
            {"kind": "finding", "description": "Rate limits vary by endpoint"},
        ],
        "failures": ["Stripe changelog page returned 403"],
    },
]
```

The rLM can read, filter, and transform `_prior_work` like any other
Python variable — it never enters the LLM context window directly unless
the rLM explicitly references it in a prompt.

#### 2.2.3 Python SDK

```python
# Dispatch a research agent, get episode back
research = agent.call(
    "research",
    task="Find Stripe rate limit docs",
)
# research.response — the direct answer
# research.episode  — structured summary of what happened

# Pass that episode as context to the implementer
impl = agent.call(
    "implementer",
    task="Implement Stripe rate limiter",
    context=[research.episode],
)

# Recall relevant episodes from past work
past = memory.recall(
    "rate limiting",
    scope="episodes",
    top_k=3,
)
```

#### 2.2.4 Parallel Agent Calls

Slate emphasizes parallel thread dispatch. Currently `agent.call()` is
synchronous — the caller blocks. We add a parallel variant:

```python
# Dispatch multiple agents in parallel, collect episodes
results = agent.call_parallel([
    ("research", "Find Stripe rate limits"),
    ("research", "Find Braintree rate limits"),
    ("research", "Find Adyen rate limits"),
])
# results: list of AgentResult, each with .response and .episode
```

Implementation: `gw-bus` spawns tasks concurrently via `tokio::join!` and
collects results. The calling agent's ouros session is paused for the
duration (single host function call from its perspective).

### 2.3 Per-Call Model Routing

Slate dispatches different models to different threads. Our `OllamaClient`
already accepts `model: Option<&str>` — the plumbing exists. What's
missing is the Python-side API and an allowlist.

#### 2.3.1 Allowed Models

```rust
// gw-core — extend ModelConfig

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    /// Default model for this agent.
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
    /// Models this agent may use via override. Empty = default only.
    pub allowed_models: Vec<String>,
}
```

`LlmGate` (in `gw-llm`) validates: if the rLM requests a model not in
`allowed_models` and not equal to `model`, reject the call.

#### 2.3.2 Python SDK

```python
# Cheap model for extraction
names = llm.complete(
    model="qwen3:4b",
    system="Extract person names.",
    messages=[{"role": "user", "content": text}],
)

# Expensive model for reasoning
plan = llm.complete(
    model="qwen3:32b",
    system="Design migration strategy.",
    messages=[{"role": "user", "content": schema}],
)

# Default model (from agent config) — existing behavior
answer = llm.complete(
    system="Answer the question.",
    messages=[{"role": "user", "content": question}],
)
```

#### 2.3.3 Episode Summarization Model

Episode generation (§2.1.2) should use the cheapest allowed model. The
runtime selects: if `allowed_models` contains a model tagged as "fast"
in org settings, use it; otherwise fall back to the agent's default.

This avoids burning expensive reasoning tokens on summaries.

### 2.4 Recursion Guard

`agent.call()` currently has no depth limit. The rLM paper used depth=1.
Slate warns about overdecomposition with unbounded recursion.

#### 2.4.1 Depth Tracking

```rust
// gw-core — extend ResourceLimits

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_llm_calls: Option<u32>,
    pub max_execution_seconds: Option<u64>,
    /// Max depth of agent.call() recursion. Default: 3.
    pub max_call_depth: Option<u32>,
}
```

#### 2.4.2 Enforcement

`CallContext` gains a `call_depth: u32` field, incremented each time
`agent.call()` creates a child task. The bus checks before spawning:

```rust
// gw-bus implementation

async fn call(&self, ctx: &CallContext, agent: AgentId, task: Task, ...) -> Result<AgentResult, AgentError> {
    let limit = self.get_resource_limits(ctx.agent_id).await?
        .max_call_depth
        .unwrap_or(3);

    if ctx.call_depth >= limit {
        return Err(AgentError::MaxCallDepthExceeded {
            depth: ctx.call_depth,
            limit,
        });
    }

    let child_ctx = ctx.with_depth(ctx.call_depth + 1);
    // ... spawn child with child_ctx
}
```

This is simpler and faster than walking the `parent_task_id` chain in
Postgres on every call. The depth is carried in-memory through the
context.

Default of 3 means: orchestrator → worker → sub-worker → leaf.
Configurable per agent for cases that genuinely need deeper recursion.

---

## 3. Expressivity Preservation

Slate's central design principle: agent harnesses must preserve
**expressivity** — the range of behaviors the system can reach. Rigid
task trees, mandatory plan-then-execute patterns, and gated step
sequences all reduce expressivity.

This is an argument *for* our rLM approach. The Python REPL is maximally
expressive — the model can do anything Python can do. None of the changes
in this design constrain that expressivity:

- Episodes are **additive** — they provide context, not constraints.
- The recursion guard limits depth, not breadth. An agent can dispatch
  many parallel calls at any depth.
- Model routing expands capability — agents can pick the right tool for
  each sub-problem.

The one risk is **over-structuring episodes**. If we require agents to
produce episodes in a rigid schema, we lose the flexibility of natural
language summaries. The `Artifact` type should remain optional — the
`summary` field is the primary carrier and it's free-form text.

---

## 4. Knowledge Overhang

Slate introduces the concept of **knowledge overhang**: knowledge the
model has access to theoretically but can't access tactically without
scaffolding like chain-of-thought or file-based planning.

Our architecture already addresses this better than most: the rLM writes
Python code to explore context, which forces it to externalize reasoning.
The REPL is a natural chain-of-thought medium — the model literally
writes code to query what it knows.

Episodes extend this: by making the results of past reasoning available
as structured context (`_prior_work`), we reduce the knowledge overhang
for downstream agents. They don't need to re-derive what a prior agent
already figured out.

---

## 5. Comparison with Slate's Architecture

| Slate concept | Greatwheel equivalent | Gap |
|---|---|---|
| Thread | ouros session + `agent.call()` | None — sessions are more capable (persistent, serializable) |
| Episode | **new** (this design) | Closing it |
| Thread Weaving | orchestrator rLM dispatching `agent.call()` with episodes | Closing it |
| Orchestrator | triage agent / any rLM that calls other agents | None |
| Bounded work units | ouros pause points (per host call) | We're more granular |
| Episode as input | `context_episodes` parameter on `agent.call()` | Closing it |
| Cross-model composition | `model` parameter on `llm.complete()` | Closing it (§2.3) |
| Parallel dispatch | `agent.call_parallel()` | Closing it (§2.2.4) |
| Context compaction | episode summarization at call boundaries | Closing it |

What we have that Slate doesn't:
- Serializable session snapshots (crash recovery, idle eviction)
- Fine-tuning pipeline (rl-play → SFT/DPO/PPO → improved models)
- Variables-as-context (never hit the dumb zone)
- Per-host-call synchronization (more granular than per-episode)

---

## 6. Implementation Plan

| Phase | Change | Crate(s) | Dependencies |
|---|---|---|---|
| 1 | `Episode`, `EpisodeId`, `Artifact`, `TokenCost` types | `gw-core` | None |
| 1 | `max_call_depth` on `ResourceLimits`, `call_depth` on `CallContext` | `gw-core` | None |
| 2 | Recursion guard enforcement | `gw-bus` | Phase 1 |
| 2 | `AgentResult` type with `episode` field | `gw-bus` | Phase 1 |
| 2 | `allowed_models` on `ModelConfig` | `gw-core` | None |
| 3 | Episode generation (span compression + LLM summarization) | `gw-runtime` | Phase 1, `gw-llm`, `gw-trace` |
| 3 | Model allowlist validation in `OllamaClient` | `gw-llm` | Phase 2 |
| 4 | Episodes Postgres migration | `migrations/` | Phase 1 |
| 4 | Episode embedding + `MemoryScope::Episodes` | `gw-memory` | Phase 1 |
| 5 | `context_episodes` on `AgentBus::call()`, context injection | `gw-bus`, `gw-runtime` | Phase 2-4 |
| 5 | `agent.call_parallel()` | `gw-bus`, `gw-runtime` | Phase 2 |
| 6 | Python SDK: `episode.checkpoint()`, `agent.call(context=)`, `llm.complete(model=)` | Agent SDK | Phase 3-5 |

Phases 1-2 are small, safe, and can land independently.
Phase 3 is the core — episode generation requires the LLM and trace systems.
Phases 4-5 make episodes persistent and composable.
Phase 6 is the agent-facing API.

---

## 7. Open Questions

**Episode summarization cost.** Every sub-agent completion triggers an LLM
call. Mitigation: skip for trivial tasks (§2.1.2), use cheapest model
(§2.3.3). Is this sufficient, or do we need a token-budget threshold?

**Episode retention.** Episodes accumulate per-org. Options: per-org TTL,
LRU eviction, or rely on vector search to surface only relevant ones.
Start with no eviction and measure growth.

**Auto-checkpoint.** Should the runtime auto-checkpoint after N host calls
or N tokens consumed? Or leave it entirely to the rLM? Start manual-only,
add auto-checkpoint if agents consistently lose context in long sessions.

**Episode size.** Should episodes have a max summary length? A sub-agent
that did extensive work could produce a summary large enough to crowd the
parent's context. Propose: cap at 2000 tokens, let the LLM prioritize.

**Parallel call semantics.** If one parallel call fails, do we cancel the
others? Return partial results? Propose: collect all results, mark failed
ones with errors, let the caller decide.

---

## 8. References

- [Slate: moving beyond ReAct and RLM](https://randomlabs.ai/blog/slate) — Random Labs
- [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) — Alex Zhang
- [Context Rot](https://research.trychroma.com/context-rot) — Hong, Troynikov & Huber (Chroma, 2025)
- [ADaPT: As-Needed Decomposition and Planning with LLMs](https://arxiv.org/abs/2311.05772) — Prasad et al.
- [Acquisition of Chess Knowledge in AlphaZero](https://www.pnas.org/doi/10.1073/pnas.2206625119) — McGrath et al. (PNAS 2022)
- [AlphaGo (Nature)](https://www.nature.com/articles/nature16961) — Silver et al.
- [Context Engineering in Manus](https://manus.im/blog/context-engineering) — Lance Martin
