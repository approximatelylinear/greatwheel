# Design: BrowseComp Experiment Dashboard (on `gw-ui`)

**Status:** Past mode end-to-end. Backend (steps 1, 2, 5, 6), frontend renderers + catalog (step 3), and `BenchAgent` v1 + runnable `bench_dashboard` example (step 4) landed. Live mode (step 7+) and Planning mode (step 9) still pending.
**Date:** 2026-05-05

---

## 0. Implementation Status

Backend host-fn surface is complete; frontend renderers and the
`BenchAgent` are not yet started. Smoke against real `runs/` (56
experiments) returns sensible numbers — top runs match prior
EXPERIMENTS.md history.

| Component | Status | Location |
|-----------|--------|----------|
| `crates/gw-bench-ui/` crate (workspace member) | Done | `crates/gw-bench-ui/` |
| `BenchPaths` / `BenchStore` / `BenchPlugin` | Done | `src/{paths,store,plugin}.rs` |
| Run JSON + API return types | Done | `src/types.rs` |
| Exact + fuzzy scoring (port of `quick_eval.py`) | Done | `src/score.rs` |
| Host fns: `list_experiments`, `load_run`, `load_query`, `load_config`, `compare`, `gold` | Done (step 1) | `src/plugin.rs` |
| Host fns: `read_annotations`, `write_annotation` | Done (step 2) | `src/plugin.rs` |
| Host fns: `difficulty_matrix`, `query_history` | Done (step 5) | `src/plugin.rs` |
| Host fn: `cost_summary` | Done (step 6) | `src/plugin.rs` |
| Annotations sidecar (`runs/<slug>/notes.json`) | Done | `src/store.rs::{read,write}_annotations` |
| Test fixtures + 32 tests | Done | `tests/store_test.rs`, `tests/data/` |
| Smoke example | Done | `examples/smoke.rs` |
| Custom React renderers (`Code`, `DifficultyMatrix`, `CostTrend`, `Markdown`) | Done (step 3) | `frontend/src/widgets/{CodeBlock,DifficultyMatrixWidget,CostTrendWidget}.tsx` + inline `Markdown` |
| Widget catalog registration | Done (step 3) | `frontend/src/jr/{catalog.ts,registry.tsx}` |
| `BenchAgent` prompt + first-turn pinned widgets | Done (step 4) | `crates/gw-bench-ui/examples/bench_dashboard.rs` (`SYSTEM_PROMPT`) |
| `bench_dashboard` runnable example | Done (step 4) | `crates/gw-bench-ui/examples/bench_dashboard.rs` |
| Translator cases for `Code` / `Markdown` / `DifficultyMatrix` / `CostTrend` | Done (step 4) | `frontend/src/jr/translate.ts` |
| Live mode (filesystem watcher) | Not started | `src/store.rs` (TBD) |
| `ops_status` host fn | Not started | `src/plugin.rs` (TBD) |
| Spawn (`bench:spawn` capability) | Deferred to v3 | — |

**To pick up in a fresh session, read this section, then jump to §12 for
the next step (Live mode Layer 1 — filesystem watcher, `list_running`,
`tail_run`, `RunningExperiments` + `LiveRunProgress`, step 7).**

Quick verify the stack still works:

```bash
cargo test -p gw-bench-ui                       # 32 tests
cargo run -p gw-bench-ui --example smoke        # walks real runs/
cargo build -p gw-bench-ui --example bench_dashboard
# Past-mode dashboard end-to-end (needs Ollama or OPENAI_API_KEY):
#   cargo run -p gw-bench-ui --example bench_dashboard
#   then open http://localhost:5173/?session=<printed uuid>
```

---

## 1. Motivation

`bench/browsecomp/EXPERIMENTS.md` is currently the system of record for
what's been tried, what worked, and what didn't. The actual artifacts —
30 per-query JSONs per run × ~50 runs in `runs/` — are read with `jq`,
ad-hoc Python, and grep. The data is rich (timing breakdowns,
tool-call counts, token usage, retrieved docids, full reasoning trace)
and almost none of it surfaces unless someone asks a specific question.

We have `gw-ui` working end-to-end (chat + widgets + interaction +
canvas pane). Pointing it at the bench produces a dashboard for free,
keeps the analysis loop conversational, and exercises gw-ui on a real
internal use case before any external demo.

---

## 2. Design Principles

**Agent-driven, not REST-driven.** A `BenchAgent` running in the
existing rLM loop is the dashboard's only backend. It reads run files
through host functions, emits widgets, and answers analysis questions
in the same turn. No separate API layer, no FastAPI sidecar.

**`runs/` is the source of truth.** Experiments are directories of
JSONs on disk, exactly as they are today. The bench writes per-query
JSONs incrementally (verified: 29 timestamps over 12 minutes in
`apr19-gepa-v3-native/`), so live progress is tractable from the
filesystem alone — no in-process hooks needed for v1.

**Three concerns, one agent.** The dashboard supports live monitoring
of in-flight runs, analysis of finished runs, and planning of the
next set. They share host functions and widgets, but have different
streaming needs and pinning logic. One agent handles all three;
the user's question selects the concern.

**Annotations are durable, narrow writes.** v1 is read-mostly, but
notes/tags on runs are valuable enough that the dashboard writes
sidecar JSONs (`runs/<slug>/notes.json`). This is the only `bench:write`
capability before v3. Spawning experiments stays out until v3.

**Persistent state in the canvas, transient analysis in chat.** The
currently-focused entity (running experiment, finished run, hypothesis
ledger) lives in pinned canvas widgets. Per-query drill-downs and
comparisons are inline transient widgets that scroll with the
conversation.

---

## 3. The three concerns

### 3a. Live — what's running right now

**Pinned canvas widgets:**
- **`RunningExperiments`** — table of `runs/<slug>/` whose mtime is
  within the last N seconds, or that carry a `_running` marker file.
  Columns: slug, queries done / 30, p50 so far, projected completion,
  uptime.
- **`OpsStatus`** — health of the long-lived sidecar processes
  (`qwen3_embed_server` :8003, `colbert_server` :8002,
  `search_server_qwen3` :8000+offset, ollama / rl-play). Up/down/last
  ping. Surfaced everywhere because forgetting to start one of these
  is a recurring failure mode (per `design-bench-worktrees.md`).

**Inline widgets (per running experiment):**
- **`LiveRunProgress`** — progress bar + per-query status grid as
  rows fill in. Streamed from filesystem events.
- **`LiveQueryTrace`** *(v1.5)* — for a query currently in flight,
  show the agent's tool calls and LLM responses as they happen. Needs
  a per-query trace sidecar (see §8).

### 3b. Past — analyzing finished runs

**Pinned canvas widgets:**
- **`ExperimentList`** — every directory in `runs/`. Columns: slug,
  query count, exact / fuzzy, p50, mtime, annotation tags.
- **`SelectedRunSummary`** — header card for the focused run
  (metadata, scores, timing, notes).

**Inline widgets:**
- **`PerQueryTable`** — 30 rows: status / score / iterations / tokens
  / termination_reason. Click a row → drill-down.
- **`QueryDetail`** — for one `(run, query_id)`: question, gold,
  model answer, full trace, retrieved docids with rank.
- **`ComparisonTable`** — `(run_a, run_b)` per-query win/loss/tie.
- **`ConfigDiff`** — TOML diff between two runs' configs.
- **`DifficultyMatrix`** — `30 × N` grid (rows = queries, columns =
  recent runs) colored by correct/wrong/timeout. Drives "what to try
  next" — persistently-hard queries are where the next hypothesis
  lives.
- **`QueryHistory`** — inverse of `DifficultyMatrix`. Pick query 747,
  see every run's answer and retrieved docids. For understanding
  whether a specific intervention moved a specific hard case.
- **`AnnotationsPanel`** — notes and tags on the focused run.
  Editable (the only writable widget in v1).
- **`CostTrend`** — token usage and dollar spend per run, rolling
  window. Free aggregate over `usage` fields; matters now that
  OpenAI `gpt-5.4` is in the mix alongside Ollama.

### 3c. Planning — deciding what to try next

**Pinned canvas widget:**
- **`HypothesisLedger`** — structured rendering of the "Things that
  help" / "Things that don't help" tables in `EXPERIMENTS.md`. Each
  row is a hypothesis with citations to specific runs and a
  confidence indicator (number of confirming / disconfirming runs).
  Tagged annotations on runs feed this directly.

**Inline widgets:**
- **`ExperimentNotes`** — relevant section of `EXPERIMENTS.md`
  rendered as markdown when the agent cites it.
- **`ProposalCard`** — agent-generated suggestion for the next
  experiment: hypothesis, config delta from baseline, predicted
  outcome, runs to compare against. Buttons: *file as TODO* (writes
  to `EXPERIMENTS.md` in v2) / *spawn now* (v3).

The agent in this concern reads `EXPERIMENTS.md`, the difficulty
matrix, and annotations; emits proposals; never spawns. Spawn is v3.

---

## 4. User stories

| # | Story | Concern | Status |
|---|---|---|---|
| 1 | See currently-running experiments with live progress and results | Live | v1 (coarse, filesystem-tailing) → v1.5 (per-query trace) |
| 2 | Discuss the next set of experiments using `EXPERIMENTS.md` | Planning | v1 |
| 3 | Kick off an experiment by chatting | (spawn) | **v3** |
| 4 | Cross-run query difficulty matrix | Past | v1 |
| 5 | Per-query history across all runs | Past | v1 |
| 6 | Annotate runs with notes / hypothesis tags | Past + Planning | v1 |
| 7 | Operational status panel (services / ports) | Live | v1 |
| 8 | Cost & token-spend trend | Past | v1 |

Stories 4–8 came out of the "what other tasks would be useful" pass.
None of them require new infrastructure beyond §5; they're widget +
host-fn additions.

---

## 5. Host functions

A new plugin in `gw-bench` exposes filesystem-bound host fns. Three
capability tiers, only the first two in v1.

### `bench:read` (v1)

| Name | Returns | Used by |
|------|---------|---------|
| `bench.list_experiments()` | `[{slug, n_queries, exact, fuzzy, p50_ms, mtime, tags}]` | `ExperimentList` |
| `bench.load_run(slug)` | `{metadata, queries: [{query_id, status, score, iterations, ...}], notes}` | `SelectedRunSummary`, `PerQueryTable` |
| `bench.load_query(slug, query_id)` | `{question, gold, answer, trace, retrieved_docids, timing, usage}` | `QueryDetail` |
| `bench.load_config(slug)` | `{toml_path, contents}` | `ConfigDiff` |
| `bench.compare(slug_a, slug_b)` | `{per_query: [...], deltas: {exact, fuzzy, p50_ms, tokens}}` | `ComparisonTable` |
| `bench.gold(query_id)` | `{question, answer, gold_docids}` | `QueryHistory`, `QueryDetail` |
| `bench.difficulty_matrix(slugs)` | `{queries: [qid], runs: [slug], cells: [[status]]}` | `DifficultyMatrix` |
| `bench.query_history(query_id, slugs?)` | `[{slug, status, answer, retrieved_docids}]` | `QueryHistory` |
| `bench.list_running()` | `[{slug, queries_done, eta_sec, started_at}]` | `RunningExperiments` |
| `bench.tail_run(slug, since_ts)` | `{new_queries: [...], n_done: int}` | `LiveRunProgress` (polled or SSE) |
| `bench.ops_status()` | `[{name, port, healthy, last_ping}]` | `OpsStatus` |
| `bench.cost_summary(window?)` | `[{slug, input_tokens, output_tokens, est_usd}]` | `CostTrend` |
| `bench.read_annotations(slug)` | `{notes: [{ts, text}], tags: [string]}` | `AnnotationsPanel` |
| `bench.read_experiments_md(section?)` | `{markdown}` | `HypothesisLedger`, `ExperimentNotes` |

### `bench:write` (v1, narrow)

| Name | Returns | Used by |
|------|---------|---------|
| `bench.write_annotation(slug, text, tags?)` | `{ok}` | `AnnotationsPanel` |

### `bench:spawn` (v3)

| Name | Returns |
|------|---------|
| `bench.spawn_experiment(name, config_toml)` | `{worktree_path, search_port, status_stream_id}` |
| `bench.cancel_experiment(name)` | `{ok}` |

---

## 6. Live mode mechanics

The structurally novel part. Three layers, increasing fidelity.

**Layer 1 (v1) — directory tail.** A filesystem watcher on `runs/`
(via the `notify` crate, already a workspace dep) emits one event per
new `run_*.json` file. The plugin maintains a `slug → {n_done,
last_ts, cells}` map; `bench.list_running` and `bench.tail_run` read
it. Detection: a run is "live" if its directory's newest file is
< 5 min old, or a `_running` marker file exists. Per-query granularity
is enough for `LiveRunProgress` and the difficulty-matrix cell to
flip live.

**Layer 2 (v1.5) — per-query trace tail.** Bench harness writes
`runs/<slug>/_trace_<query_id>.jsonl` append-only as the agent runs
(one JSON line per tool call / LLM response). Dashboard tails the
file via the same watcher. `LiveQueryTrace` widget shows the lines
as they appear. Bench-side change: ~20 lines in
`retrieval_benchmark_v2.py` (or its rLM equivalent) to emit the trace
file. Decoupled from gw-server entirely.

**Layer 3 (v2) — `LoopEvent` over AG-UI.** Refactor the bench harness
to run agents through `gw-server`'s session model so traces flow
natively as `LoopEvent`s and the dashboard subscribes to the same
SSE stream. Removes the JSONL detour, unifies live trace with chat
infrastructure. Requires bench harness to be a gw-server caller, not
a standalone Python script — bigger lift, deferred.

V1 ships with Layer 1. Layer 2 lands when watching a run's reasoning
in real time becomes the bottleneck for debugging.

---

## 7. Annotations

Sidecar file at `runs/<slug>/notes.json`. Schema:

```json
{
  "notes": [{"ts": "2026-05-04T12:34:00Z", "text": "ignore — server crashed at q15"}],
  "tags": ["tests-hypothesis:passage-rrf", "reproduces:1331262"]
}
```

`bench.load_run` merges this into its return value so the rest of the
widgets see annotations without a second call. `AnnotationsPanel` is
the editing surface; `bench.write_annotation` appends a note and
unions tags. Tags are free-form strings with conventions enforced
only by the agent's prompt — no schema today, EXPERIMENTS.md
narrative is the schema.

`HypothesisLedger` reads annotations across all runs to count
confirming / disconfirming evidence per hypothesis.

---

## 8. Agent shape

One `BenchAgent`, mode-aware via the question. System prompt
sketch:

> You are a research assistant for BrowseComp retrieval experiments.
> The user asks questions about runs in `runs/*` (live or finished),
> hypotheses recorded in `EXPERIMENTS.md`, and what to try next. Load
> data via `bench.*` host fns; present results by emitting widgets.
> Pin one navigational widget to the canvas: `RunningExperiments` if
> any runs are live, otherwise `ExperimentList` for past analysis, or
> `HypothesisLedger` when the conversation is about planning. Keep
> chat replies short — the data is in the widgets.

Canvas pinning is agent-managed via `ui.pin_to_canvas` /
`ui.supersede_widget`, exactly as in `design-gw-ui.md` §7. No
explicit "mode" enum in the agent — it picks a pinned widget based
on what the user just asked.

---

## 9. Read paths

Three sample flows, one per concern.

```
ChatPane "what's running?"
  → BenchAgent turn
    → host: bench.list_running()
    → host: bench.ops_status()
    → host: ui.pin_to_canvas(RunningExperiments)
    → host: ui.emit_widget(OpsStatus, …)
  → CanvasPane updates
```

```
ChatPane "compare apr19-passages-4096 vs apr19-gepa-v3-native"
  → BenchAgent turn
    → host: bench.compare("apr19-passages-4096", "apr19-gepa-v3-native")
    → host: bench.load_config(both)
    → host: ui.emit_widget(ComparisonTable, …)
    → host: ui.emit_widget(ConfigDiff, …)
  → ChatPane appends both widgets
```

```
ChatPane "what should we try next?"
  → BenchAgent turn
    → host: bench.read_experiments_md(section="Things that don't help")
    → host: bench.difficulty_matrix(recent 5 slugs)
    → host: bench.read_annotations(each)
    → host: ui.pin_to_canvas(HypothesisLedger)
    → host: ui.emit_widget(ProposalCard, …)
  → CanvasPane shows ledger; ChatPane shows proposal
```

No host fn returns more than ~100KB on the heaviest paths.
`bench.load_query` (full traces) is the exception and is only
fetched on click.

---

## 10. Widget catalog (v1)

All A2UI, registered in `frontend/src/jr/registry.ts`. Twelve widgets
total.

| Widget | Concern | Renderer notes |
|---|---|---|
| `ExperimentList` | Past | Table + per-row button |
| `SelectedRunSummary` | Past | Card |
| `PerQueryTable` | Past | Table + per-row button |
| `QueryDetail` | Past | Card + Code (trace) + Table (docids) |
| `ComparisonTable` | Past | Table with color-coded winner column |
| `ConfigDiff` | Past | Code (unified TOML diff) |
| `DifficultyMatrix` | Past | Custom React grid (json-render `Grid` doesn't exist; needs renderer) |
| `QueryHistory` | Past | Table |
| `AnnotationsPanel` | Past | Card with form (textarea + tag input) |
| `CostTrend` | Past | Custom React sparkline component |
| `RunningExperiments` | Live | Table |
| `LiveRunProgress` | Live | Card + progress bar (SSE-driven) |
| `OpsStatus` | Live | Table with status dot |
| `HypothesisLedger` | Planning | Table with citation links |
| `ExperimentNotes` | Planning | Markdown |
| `ProposalCard` | Planning | Card + buttons |

That's actually sixteen, not twelve. Several are
`Card`+`Table`+`Button` compositions that the existing
`@json-render/react` registry handles directly; the genuinely new
renderers are `Code` (TOML / JSON syntax highlight),
`DifficultyMatrix` (a colored grid), `CostTrend` (a sparkline), and a
markdown renderer wired to `react-markdown` (already a frontend dep).
Four custom components.

---

## 11. Server wiring

- New module `gw-bench-ui` (or inside `gw-bench`) with the plugin
  definition and host-fn impls. Registered alongside `UiPlugin` in
  the `gw-server` startup.
- New runnable example
  `crates/gw-bench/examples/bench_dashboard.rs`, modeled on
  `gw-ui/examples/echo_server.rs`. Spawns a `BenchAgent` session,
  prints the session UUID, serves the frontend on a port that
  doesn't collide with the bench's own search servers (8780 is
  free).
- Filesystem watcher on `runs/` is owned by the plugin, started in
  `init`, shut down on plugin teardown. Single watcher per server
  process, fan-out to per-session `bench.tail_run` callers via a
  broadcast channel.
- Health probes for `OpsStatus` are short HTTP gets to known
  ports with a 200ms timeout; cached for 5s to avoid hammering.

---

## 12. Implementation order

Each step ends at a runnable system, same convention as
`design-gw-ui.md`. Backend was implemented out of step order
(steps 1, 2, 5, 6 first) so the entire host-fn surface for Past mode +
the cross-run analytics is in place before any frontend work begins;
this lets the frontend session focus purely on rendering and agent
prompting.

1. ✅ **`bench:read` host fns + tests.** `list_experiments`, `load_run`,
   `load_query`, `load_config`, `compare`, `gold`. Tested against
   fixtures under `crates/gw-bench-ui/tests/data/`. **Note:** the
   crate is `gw-bench-ui` (not `gw-bench`); fixtures live there.
2. ✅ **Annotations.** `read_annotations`, `write_annotation`, sidecar
   JSON at `runs/<slug>/notes.json`, atomic tmp+rename writes.
   `RunDetail.notes` and `RunDetail.tags` populate from sidecar.
3. ✅ **Custom renderers.** `Code`, `DifficultyMatrix`, `CostTrend`,
   and `Markdown` registered in `frontend/src/jr/{catalog,registry}.tsx`;
   custom React components live in `frontend/src/widgets/`. Typecheck
   and `vite build` clean. No backend wiring yet — these light up in
   step 4 when the agent emits them.
4. ✅ **Past-mode widgets + agent.** `BenchAgent` v1 system prompt
   landed in `crates/gw-bench-ui/examples/bench_dashboard.rs`,
   covering ExperimentList → SelectedRunSummary + PerQueryTable →
   QueryDetail → ComparisonTable + ConfigDiff (unified diff via
   `difflib` in the agent) → DifficultyMatrix → QueryHistory →
   CostTrend → AnnotationsPanel (write_annotation re-emits the
   panel). Translator cases for the four new widget types added to
   `frontend/src/jr/translate.ts` (per-cell ActionBindings on
   DifficultyMatrix mirror the EntityCloud per-point pattern).
   `cargo build -p gw-bench-ui --example bench_dashboard` is green;
   live smoke against an LLM is the user's next step.
5. ✅ **Difficulty matrix + query history.** Two host fns landed
   (`difficulty_matrix`, `query_history`); widgets pending in step 3.
   Cell schema decided in §13.
6. ✅ **Cost trend.** Host fn `cost_summary` with optional
   `window_days`; widget pending in step 3. Per-model pricing is a
   placeholder table (§13 open).
7. **Live mode (Layer 1).** Filesystem watcher, `list_running`,
   `tail_run`, `RunningExperiments` and `LiveRunProgress`. Agent
   prompt updated to pin the live widget when runs are detected.
8. **Ops status.** Health probes, `OpsStatus` widget.
9. **Planning mode.** `read_experiments_md`, `HypothesisLedger`,
   `ProposalCard`. Agent prompt extended.
10. **Dashboard binary.** `bench_dashboard.rs` example tying it
    together as one runnable thing.
11. *(v1.5)* Live trace (Layer 2): bench writes per-query
    `_trace_*.jsonl`; `LiveQueryTrace` widget tails it.
12. *(v3)* Spawn: `bench:spawn` capability, `bench.spawn_experiment`,
    `bench.cancel_experiment`, integration with the worktree
    spawner from `design-bench-worktrees.md`. New `RunningExperiment`
    fields for live spawn status.

The MVP target is "Past mode usable end-to-end" — that requires
finishing steps 3 and 4 (frontend + agent). Backend work for steps 5
and 6 is already done so those widgets can land in the same frontend
session for free.

---

## 13. Open questions

- ✅ **Difficulty-matrix cell semantics.** *Resolved during
  implementation.* Cells classify on correctness only —
  `Exact / Fuzzy / Wrong / Error / Missing` — where `Error` fires
  only on `status == "error"`, **not** on
  `termination_reason == "timeout"`. Real runs frequently terminate
  on timeout but still emit a correct "Exact Answer:" line; treating
  those as Error hid 13 correct cells out of 87 in the smoke run.
  Termination details remain available via
  `QuerySummary.termination_reason`, which the UI can layer as a
  small icon overlay if needed. Reopen if "did the gold doc appear
  in retrieved_docids" becomes a needed dimension.
- **`load_config` slug-to-TOML resolution is unreliable.** Most real
  run slugs (`apr19-baseline-native`, `colbert-passage-rerank-v2`,
  …) don't map cleanly to `bench/browsecomp/configs/*.toml`. v1
  tries `<slug>.toml` then strips a leading `<date>-` prefix —
  misses on most. Two options: (a) embed the resolved config inside
  each `run_*.json` at write time in `gw-bench`, then `load_config`
  reads from the run JSON; (b) write a `_config.toml` sidecar
  alongside the run JSONs. Lean toward (a): one less file, no
  ambiguity. Needs a 5-line change to `gw-bench/src/main.rs`.
- **Cost-summary pricing is placeholder.** Per-model rates for OpenAI
  models are hardcoded guesses; local Ollama returns `est_usd: None`.
  Move to a config-driven table (e.g.,
  `bench/browsecomp/pricing.toml`) once we have stable rates.
- **Where does `EXPERIMENTS.md` live in the dashboard?** v1 plan: it
  stays as the markdown narrative; `HypothesisLedger` is a
  *projection* of it, not a replacement. Open: should the dashboard
  ever *write* to `EXPERIMENTS.md`, or is that a manual curation
  step? Lean toward manual until v2 shows the projection is reliable
  enough to round-trip.
- **GEPA runs.** `bench/browsecomp/gepa_runs/` has a different schema
  (`candidates.json`, `gepa_state.bin`, `run_log.json`). Probably a
  second set of host fns (`bench.list_gepa_runs`, etc.) and a
  `GepaCandidateTree` widget. Defer; not blocking v1.
- **Multi-session vs single-session.** Echo_server today is fixed to
  one session. Dashboard sessions are cheap; do we let the user open
  several (one per analysis thread)? V1 single-session; v2
  multi-session is a frontend tab change.
- **Long traces.** `run_*.json` results can be ~MB.
  `bench.load_query` either caps trace length and offers a "load
  full trace" follow-up, or returns it as a
  `WidgetPayload::Reference` to a static file route. Lean toward
  reference — keeps the AG-UI SSE channel clean.
- **Auth.** Same answer as `gw-ui` today: signed widget tokens off
  `SessionKey`. The dashboard reads files and writes annotations; no
  new threat surface relative to the rest of `gw-server`.

## 14. Lessons from backend implementation

Recording for the next session.

- **`gw-bench/src/main.rs`'s schema comment is stale.** It claims
  `termination_reason ∈ {"final_called", "max_turns", "timeout",
  "refusal_rejected", "llm_error", "max_turns_fallback"}` — actual
  values include `final_called_code` (not in the comment) and the
  dominant value across recent runs is `timeout` paired with
  `status="max_turns_fallback"`. The deserializer is permissive
  (string fields), so this didn't break anything; flagging only so
  any future code that pattern-matches on these values knows to look
  at real data first. The data drove the §13 cell-semantics
  decision.
- **Python `quick_eval.py` port had one bug.** `re.sub(r"[^\w\s]", "", text)`
  *removes* punctuation; my first port replaced with space, producing
  "richard c  larson" (double space) and missing the
  cleaned-substring path. Caught by a unit test on the first try.
  The current Rust impl filters non-alphanumeric/non-whitespace
  chars out, matching Python.
- **Naming convention for host fns is flat at the registration
  site.** Manifests use `host_fn:bench.list_experiments` for
  capability scoping; `register_host_fn_async` takes the bare name
  `"list_experiments"`. Mirrors the `gw-ui` pattern. Python agents
  in ouros call by bare identifier.
- **`pick_string_list` helper** in `plugin.rs` accepts an optional
  list-of-strings kwarg, returning empty `Vec` for missing/null.
  Reused for `tags`, `slugs`, etc. Worth lifting to a shared
  plugin-utils crate when a third plugin needs it.
