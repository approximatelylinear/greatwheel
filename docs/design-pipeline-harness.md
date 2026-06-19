# Design: Pipeline Harness — What Greatwheel Would Need to Host a Clearwing-Shaped Application

**Status:** Proposal
**Date:** 2026-04-17
**Depends on:** [Apps](design-apps.md), [Plugin Framework](design-plugin-framework.md)

---

## 0. Problem

Clearwing (Lazarus-AI) is an autonomous offensive-security tool — a
source-code hunter and network-pentest agent — built on LangGraph. Its
source-hunt mode is interesting precisely because it is **not** a single
ReAct loop: it is a staged pipeline (preprocess → rank → per-file hunter
pool → adversarial verifier → exploiter → variant loop → auto-patcher →
report), with a different LLM client per stage, tiered budget allocation,
argument-level tool guardrails, per-hunter Docker sandboxes, and a
cross-run "mechanism memory" that injects abstract lessons back into
hunter prompts.

The question this doc is scoped to: **if someone wanted to build a
clearwing-shaped application on top of greatwheel, what is missing from
the harness today?** Not "what features should we copy" — that's app
work — but: which gaps are execution-substrate-level, the kind an
application cannot paper over from the SDK side?

This doc is deliberately narrow. Application-level patterns (evidence
ladders, verifier agents, variant loops, knowledge graphs) are discussed
only to separate them from real harness gaps.

---

## 1. Clearwing's Harness Shape (for reference)

Three substrate facts that shape everything else:

1. **Two orchestration modes in one codebase.** A LangGraph ReAct loop
   with ~63 bind-tools (network-pentest), and a staged DAG
   (`SourceHuntRunner`) where each stage is an independently configurable
   unit.
2. **Per-stage LLM client.** `SourceHuntRunner(ranker_llm, hunter_llm,
   verifier_llm, exploiter_llm)` — cheap models rank, strong ones hunt.
3. **Per-hunter disposable Docker containers** with ASan/UBSan/MSan
   toolchains. The sandbox is the substrate where compilation, fuzzing,
   and exploit PoCs actually execute — not where the LLM "thinks."

Supporting substrate features Clearwing depends on:

- Argument-level guardrails (`input_guardrail_tool_names`) that validate
  `target=` before a scanner shells out.
- Budget tiers (Tier A/B/C, 70/25/5 split, rollover).
- Capability probing (`capabilities.has("knowledge")`) so optional
  subsystems degrade gracefully.
- Cross-run persistent stores outside the session: mechanism memory
  (JSONL), knowledge graph (networkx → JSON).

Everything else in Clearwing (Finding dataclass, evidence levels,
variant loop, specialist dispatch, patch oracle) is *application logic*
written against these substrate features.

---

## 2. Greatwheel's Harness Shape Today

Crate inventory, through the lens of "what an agent author can count on":

| Primitive | Where | Shape |
|-----------|-------|-------|
| Execution substrate | `gw-runtime` + ouros | Python REPL, pauses on host calls, snapshots serializable |
| LLM | `gw-llm` | `chat()` with per-call model override, `embed()`, streaming, via rl-play |
| Memory | `gw-memory` | Vector (LanceDB) + BM25 (tantivy) + JSONB (Postgres), RRF fusion, scope filters |
| Inter-agent | `gw-bus` | `AgentBus` trait (`call`/`notify`) — **trait-only, no concrete impl** |
| Routing | `gw-scheduler` | Types defined; **no queue or enforcement** |
| Permissions | `gw-core::ToolPermissions` | Name-based allow/deny sets |
| Rate limits | `gw-core::RateLimitConfig` | Flat soft + hard token caps, org + per-user |
| Secrets | session-key model | Agent never sees real credentials |
| Tracing | `gw-trace` | OTel GenAI semconv → console + OTLP + Postgres |
| Fine-tuning capture | rl-play proxy | Session-grouped LLM calls → SFT/DPO/PPO |
| Agents | `AgentDef` | Single model, single source (bare/file/inline/git), hot-reloadable |

The canonical execution pattern is one agent, one rLM loop, one model,
one ouros session. An agent can spawn sub-agents via `agent.call`, but
that remains one-rLM-loop-calls-another.

---

## 3. Where the Shapes Diverge

Not every difference is a harness gap. The test: can an application
author build it from the Python SDK alone, or does the runtime have to
change?

### 3.1 Things an application can already build today

These are genuinely app-level and do **not** need harness changes:

- **Staged pipelines as rLM code.** The rLM model is Turing-complete
  Python. A `SourceHuntRunner` equivalent is just a function that
  sequences calls to `agent.call(ranker)`, `agent.call(hunter)`, etc.
- **Adversarial verifier pattern.** `agent.call` to a verifier sub-agent
  with a deliberately independent system prompt is two lines of Python.
- **Evidence-level ladder / Finding dataclass.** Pure data. Store in
  `memory.store` with a JSON payload; add a `confidence` field in the
  value blob. No schema change needed in the runtime.
- **Variant fixpoint loop, patch-oracle, specialist dispatch.** All
  ordinary control flow over the existing SDK surface.
- **Knowledge graph.** An app-scoped plugin with its own persistence
  and host functions; the apps design already contemplates this.

If the harness didn't change at all, an ambitious author could build 70%
of Clearwing's source-hunt pipeline today. The remaining 30% is where
the real harness gaps are.

### 3.2 Real harness gaps — the remaining 30%

Ranked by "will this block an app author who has already committed to
greatwheel":

1. **Non-Python execution substrate.** Clearwing compiles C with ASan,
   runs libFuzzer, shells out to nmap/Kali. Ouros sandboxes Python
   code. There is no greatwheel substrate for "disposable container,
   mount the workspace read-only, run this binary, capture stdout +
   exit code + filesystem artifacts." Today an agent wanting to do this
   has to use a host function that shells out on the *host*, which
   collapses the sandbox boundary and loses the per-call isolation
   Clearwing depends on. **This is the largest gap.**

2. **Argument-level tool guardrails.** `ToolPermissions` is
   name-granular. An agent can either call `scan_ports` or it can't —
   there is no way to express "may call `scan_ports` only with targets
   inside CIDR X." Clearwing's input guardrails catch hallucinated
   `target=` values before anything ships. In greatwheel today, this
   logic would have to live inside every tool's own body, which is both
   error-prone and invisible to the permission audit.

3. **Per-call model override in the Python SDK.** `gw-llm` exposes
   per-call model override in Rust, but the Python SDK path into
   `llm.complete()` fixes on the agent's `ModelConfig`. To let one
   agent's rLM cheaply-rank-then-expensively-hunt inside a single
   session, `llm.complete(model=..., ...)` needs to be plumbed through
   the host bridge — and the permission model needs to decide what
   "allowed models" means at the agent/org level.

4. **Richer budget / rate-limit model.** `RateLimitConfig` is flat soft
   + hard tokens per org/user. A pipeline author needs at least
   **categorical budgets** ("Tier A has 70% of this session's budget,
   Tier C has 5%, unspent rolls down"). This isn't a blocker — an app
   can track its own budget — but once an app tracks its own budget,
   the org-level cap becomes cosmetic, which defeats the purpose of
   having it in the harness.

5. **Session-prelude hook.** Mechanism memory requires a "run this
   memory recall and inject the result into the system prompt before
   turn 1 begins" step. Today an agent handles this itself on every
   turn, which wastes context and tokens. A declarative prelude on
   `AgentDef` (`prelude: Option<PreludeSpec>`) would let mechanism
   memory, persona reminders, and retrieved context be uniformly
   expressed.

6. **Runtime capability registry.** Clearwing's `capabilities.has(name)`
   is runtime, not compile-time. Greatwheel features are compiled into
   crates; there's no notion of "this optional subsystem failed init,
   flag it as absent, and let agents / tools query whether it's
   available." This matters when a deployment wants to run without
   Postgres-backed traces, or without a specific plugin.

7. **Pipeline-as-first-class.** An rLM agent implementing a pipeline
   *can* express stages, but every stage looks like "more code in the
   same loop." There is no harness-level `Pipeline { stages: [...] }`
   with: per-stage model config, per-stage retries, per-stage budget,
   per-stage trace span root, resume-from-stage-N on crash. This is
   the most debatable entry — it *can* be built in rLM code, but the
   observability and control-plane wins from making it first-class are
   real. Listed last because opinions differ on whether this is a
   runtime concern or an SDK library concern.

What's **not** on this list, despite being notable about Clearwing:
adversarial verifiers, evidence ladders, variant loops, specialist
dispatch, patch oracles, Finding dataclasses, knowledge graphs. Those
are all application code in Clearwing too — they just happen to share
a codebase with the framework bits.

---

## 4. Proposed Primitives

One proposal per gap. Sized for minimum viable, not ideal. Each is
independently adoptable.

### 4.1 Second sandbox substrate — `gw-sandbox`

Not a replacement for ouros; a sibling. Purpose: run a binary in a
disposable container, capture output, tear down.

```rust
// crates/gw-sandbox/src/lib.rs
#[async_trait]
pub trait SandboxBackend {
    async fn run(
        &self,
        ctx: &CallContext,
        spec: SandboxSpec,
    ) -> Result<SandboxResult, SandboxError>;
}

pub struct SandboxSpec {
    /// OCI image ref, e.g. "greatwheel/asan-toolchain:latest"
    pub image: String,
    /// Command + argv
    pub cmd: Vec<String>,
    /// Files to stage read-only into /workspace
    pub mounts_ro: Vec<Mount>,
    /// A scratch dir for outputs (tmpfs, returned as artifacts)
    pub scratch_mb: u32,
    pub timeout: Duration,
    pub memory_mb: u32,
    pub network: NetworkPolicy,   // None | ScopedEgress(allowlist)
}

pub struct SandboxResult {
    pub exit_code: i32,
    pub stdout: Bytes,
    pub stderr: Bytes,
    pub artifacts: Vec<Artifact>, // files written to /scratch
    pub duration: Duration,
}
```

Exposed to agents as a host function:

```python
# In the rLM
result = sandbox.run(
    image="greatwheel/asan-toolchain:latest",
    cmd=["./configure", "&&", "make", "fuzz_target"],
    workspace=memory.get_file("clone:myrepo"),
    timeout_s=600,
)
if result.exit_code != 0:
    channel.send(f"build failed: {result.stderr.decode()[:500]}")
```

Open choices: Docker SDK vs. podman vs. Firecracker; how artifacts are
returned (streamed back through the host bridge, or written to a known
memory key); how network policy is enforced. The MVP is Docker + all
artifacts returned in-memory up to a size cap.

**Why this is unavoidable:** ouros's pause-on-host-call model gives us
pure-Python isolation, but there is no supported way to invoke
non-Python tooling from inside ouros without routing through a host
function that executes on the *host*. The moment that host function
shells out to `nmap`, the agent's sandbox boundary becomes the host
itself. A second substrate is the only honest answer.

### 4.2 Argument-level guardrails on `ToolPermissions`

Extend the existing permission struct rather than replacing it:

```rust
struct ToolPermissions {
    allowed: HashSet<ToolName>,
    denied: HashSet<ToolName>,
    /// New: per-tool argument validators.
    /// Keyed by tool name; each entry is a list of predicates.
    arg_rules: HashMap<ToolName, Vec<ArgRule>>,
}

enum ArgRule {
    /// Field must equal one of these values
    EqualsOneOf { field: String, values: Vec<Value> },
    /// Field must match this regex
    Matches { field: String, pattern: String },
    /// Field must be an IP inside this CIDR list
    InCidr { field: String, cidrs: Vec<String> },
    /// Custom predicate from a plugin
    Plugin { plugin_id: String, rule_id: String },
}
```

Enforcement at the `HostBridge` dispatch site, before the tool body
runs. Rejected calls surface to the agent as a typed error it can
handle, and emit a `permission_denied` span attribute.

The `Plugin` variant exists because expressing "is this target covered
by the pentest engagement's scope-of-work?" in a static rule language
is hopeless — an engagement-scope plugin owns that logic.

### 4.3 Per-call `model=` in the Python SDK

Two changes:

1. Plumb the existing `gw-llm` per-call model override through the
   Python SDK: `llm.complete(messages=..., model="qwen2.5:32b")`.
2. Add a per-agent `allowed_models: Vec<ModelPattern>` field. Default
   is `[agent.model_config.default]`. Setting `allowed_models` to
   `["qwen2.5:*"]` is how an app author opts into multi-model routing
   for a single agent without giving the agent arbitrary model access.

Rejected overrides fail the host call with a clear error; accepted
overrides show up in the trace as `gen_ai.request.model` on the
`gen_ai.chat` span, which already exists.

### 4.4 Categorical budgets

Extend `RateLimitConfig` with a `categories` map:

```rust
struct RateLimitConfig {
    soft_token_limit: Option<u64>,
    hard_token_limit: Option<u64>,
    per_user_overrides: HashMap<UserId, RateLimitConfig>,
    /// New: named budget categories, e.g.
    ///   "tier_a" -> BudgetCategory { share: 0.70, rollover: true }
    categories: HashMap<String, BudgetCategory>,
}

struct BudgetCategory {
    /// Fraction of the containing (org or user) budget
    share: f32,
    /// Unspent budget rolls down to the next-lower category
    rollover: bool,
}
```

Category is tagged on the host call:

```python
llm.complete(messages=..., budget_category="tier_a")
```

The rate limiter debits the category first, then falls back to
unreserved budget, then rejects. A pipeline author tags each stage's
calls with the right category and gets "70/25/5 with rollover" for
free.

### 4.5 Session-prelude hook

Add a `prelude: Option<PreludeSpec>` field on `AgentDef`:

```rust
enum PreludeSpec {
    /// Run a memory recall and inject the top-K results as a
    /// system-prompt suffix at session start.
    MemoryRecall {
        query: PreludeQuery,       // static string or Jinja-ish template
        scope: MemoryScope,
        top_k: usize,
        mode: SearchMode,
    },
    /// Run a Python function against the session; its return value is
    /// prepended to the first user turn as a system message.
    Python { source: String },
}
```

This is where mechanism memory plugs in: `PreludeSpec::MemoryRecall`
with a category filter, run once per session at turn zero.

### 4.6 Runtime capability registry

One crate, one struct, one query point:

```rust
// gw-core::capabilities
pub struct CapabilityRegistry { /* ... */ }

impl CapabilityRegistry {
    pub fn declare(&self, name: &str, present: bool);
    pub fn has(&self, name: &str) -> bool;
    pub fn snapshot(&self) -> HashMap<String, bool>;
}
```

Crate initializers declare presence on startup. An agent host function
exposes `capabilities.has("knowledge_graph")`. The registry snapshots
into every trace as a resource attribute, so traces are
self-describing.

### 4.7 Pipelines as first-class (optional, phase 2)

A pipeline is an `AgentSource` variant. Stages are typed; the runtime
handles retries, per-stage spans, resumption. Sketch:

```rust
enum AgentSource {
    Bare,
    File(PathBuf),
    Inline(String),
    Git { /* ... */ },
    /// New
    Pipeline(PipelineSpec),
}

struct PipelineSpec {
    stages: Vec<Stage>,
}

struct Stage {
    name: String,
    agent: AgentRef,              // which agent runs this stage
    model: Option<ModelConfig>,   // overrides agent default
    retries: u32,
    budget_category: Option<String>,
    on_failure: FailurePolicy,    // Skip | Fail | ContinueWithSentinel
    checkpoint: bool,             // persist output to sessions.snapshot
}
```

Deferred because rLM can already express this. Worth doing only once
we have two or three concrete pipeline apps asking for the same
affordances (per-stage retry, resume-from-stage) in parallel.

---

## 5. What We Deliberately Don't Build

- **A Finding dataclass in `gw-core`.** Clearwing's `Finding` is
  domain-specific. Apps define their own payloads and store them in
  memory; the harness stays agnostic.
- **An adversarial-verifier primitive.** Two agents calling each other
  is exactly what `agent.call` is for. Naming the pattern in the
  harness has no payoff.
- **A knowledge-graph crate.** App-scoped plugin. An ops app that
  wants one builds it; the harness doesn't prescribe a graph backend.
- **A LangGraph port.** The rLM model is our answer to "how do agents
  express control flow." Adding a second orchestration DSL splits the
  mental model without a clear win.

---

## 6. Phasing

Ordered by blast radius and dependency:

1. **Phase 1 (small, unlocks a lot):** 4.3 per-call `model=`, 4.6
   capability registry, 4.5 session-prelude hook. None of these
   requires a new crate; all ship behind additive API changes.
2. **Phase 2 (medium):** 4.2 argument-level guardrails, 4.4
   categorical budgets. Both extend existing structs. Guardrails touch
   every tool dispatch path, so they warrant a dedicated test pass.
3. **Phase 3 (large):** 4.1 `gw-sandbox`. New crate, OCI dependency,
   security review surface. This is the one that actually unblocks
   source-hunt-style apps. Do it when the demand is concrete.
4. **Phase 4 (optional):** 4.7 first-class pipelines. Revisit after
   two apps have hand-rolled one.

---

## 7. Open Questions

- **Sandbox backend choice.** Docker is the obvious pick (broad image
  ecosystem, Clearwing uses it, matches dev machines). Podman is more
  ergonomic for rootless. Firecracker is more isolated but image
  ecosystem is thin. Decide when we start 4.1, not before.
- **Per-call model override and fine-tuning capture.** Today rl-play
  groups by session ID. If one session uses three models, the capture
  is still one session — which may or may not be what the fine-tuning
  pipeline wants. Likely fine; flagging.
- **Where do guardrail rules live — `AgentDef` or a new table?**
  Putting them on `AgentDef` keeps the blast radius small but bloats
  the struct. A separate `guardrail_policies` table keyed by agent
  (and optionally overridable per user) scales better. Lean toward the
  separate table.
- **Budget categories vs. stage-tagged budgets.** If we ship 4.7
  (pipelines) we might not need 4.4 (categorical budgets) as a
  general primitive — a stage already has a natural tag. But 4.4 is
  useful for non-pipeline apps too, so probably still worth doing.
- **Capability registry and graceful degradation.** If a plugin
  declares `has("knowledge_graph") = false`, should `memory.recall`
  with `scope=knowledge` fail loudly or fall back to a warning? Bias
  toward fail-loud; silent degradation is how observability bugs
  compound.

---

## 8. Summary

The short answer to "what prevents us from hosting a clearwing-shaped
application today" is: **one big thing, a few small ones, and a lot of
things that look like gaps but are actually app-level work.**

The big thing is the second execution substrate — ouros is not, and
should not become, a general-purpose binary runner. The small things
are argument-level guardrails, per-call model routing, categorical
budgets, a session-prelude hook, and a runtime capability registry.
Everything else interesting about Clearwing — the verifier pattern, the
evidence ladder, the variant loop, the knowledge graph, the specialist
dispatch — is application logic and stays application logic.

A pragmatic path: ship Phase 1 in a sprint, ship Phase 2 when a
concrete app asks for it, commit to Phase 3 the first time a serious
non-Python tooling workload is on the table. That order keeps the
harness honest — every primitive we add is one an app has actually
asked for.
