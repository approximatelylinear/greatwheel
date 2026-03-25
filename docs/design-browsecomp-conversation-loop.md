# Design: BrowseComp + Conversation Loop Integration

**Status:** Proposed
**Date:** 2026-03-23

---

## 1. Problem Statement

`gw-bench` has a hand-rolled rLM loop in `run_rlm_loop()` (~400 lines)
that duplicates core logic already implemented in `gw-loop`'s
`ConversationLoop`. The hand-rolled version manages its own message
history, iteration counting, FINAL detection, and trajectory recording.
Meanwhile, `ConversationLoop` provides all of this plus a `SessionTree`,
steering injection, branching, compaction, snapshots, and Postgres
persistence — none of which the benchmark currently benefits from.

**Goal:** Replace `run_rlm_loop()` with `ConversationLoop::handle_turn()`
while preserving existing accuracy (11/30 baseline) and enabling new
capabilities.

---

## 2. Key Architectural Differences

| Aspect | gw-bench (current) | gw-loop ConversationLoop |
|--------|-------------------|--------------------------|
| Message history | Flat `Vec<Message>`, full history every call | `SessionTree` entries, `build_turn_context()` with recency window |
| REPL output feedback | Explicit user message: `"REPL output:\n```\n{output}```"` | **Skipped** — `build_turn_context()` omits CodeExecution entries, relies on `state_summary()` |
| Iteration prompts | Dynamically built per iteration with coaching nudges | Single user message per turn; steering via `inject_steering()` |
| Turn model | One "turn" = entire query (all 12 iterations) | One "turn" = one user message through the rLM inner loop |
| Trajectory | Explicit `Vec<TrajectoryMessage>` | Implicit via `SessionTree` entries |
| Refusal rejection | Inline in the loop, injects follow-up user message | No equivalent |
| Fallback extraction | Separate `fallback_extract()` after max iterations | No equivalent |
| Think-tag stripping | `strip_think_tags()` on LLM response | Not present |

---

## 3. Critical Gap: REPL Output Feedback

The most consequential difference. In gw-bench, after code execution,
the REPL output (search results, document text, llm_query responses) is
appended as a user message. The LLM sees the actual results in its
message history.

In gw-loop, `build_turn_context()` explicitly **skips** `CodeExecution`
entries. The LLM only sees variable names and values via `state_summary()`.

For BrowseComp this is fatal — the LLM must see search results and
document content to reason about the next step.

**Resolution:** Add `include_code_output: bool` to `LoopConfig`. When
true, `build_turn_context()` renders `CodeExecution` entries as synthetic
user messages with the format `"REPL output:\n```\n{stdout}\n{result}```"`.

---

## 4. Implementation Plan

### Phase 1: Extend gw-loop for Benchmark-Style Execution

Five changes to `gw-loop`, all backward-compatible (new config flags
default to off):

**1.1 — CodeExecution feedback in `build_turn_context()`**

Add `include_code_output: bool` to `LoopConfig` (default: `false`).
When true, `CodeExecution` entries in the path become user messages in
the prompt. This is the single most important change.

**1.2 — Think-tag stripping**

Move `strip_think_tags()` to `gw-runtime` as a utility. Call it in
`handle_turn()` after receiving the LLM response, before code extraction.
Only active when the LLM model uses thinking tags (qwen3.5, etc.).

**1.3 — IterationCallback trait**

```rust
pub trait IterationCallback: Send {
    /// Called before each LLM call. Returns optional steering message.
    fn before_iteration(
        &mut self,
        iteration: usize,
        max_iterations: usize,
        repl: &ReplAgent,
    ) -> Option<String>;

    /// Called when max iterations reached without FINAL.
    /// Returns optional fallback answer.
    fn on_max_iterations(&mut self, query: &str) -> Option<String>;
}
```

Added to `LoopConfig`. When present, `handle_turn()` calls
`before_iteration()` before each LLM call and appends the result as a
user message in the tree (not as a steering message — it should appear
as a normal turn in the LLM history).

**1.4 — Answer validator**

Add `answer_validator: Option<Box<dyn Fn(&str) -> bool + Send>>` to
`LoopConfig`. When FINAL is called and the validator rejects the answer,
the loop continues with a user message: "That answer was rejected.
Please try again." This replaces gw-bench's refusal rejection logic.

**1.5 — Output truncation**

Add `repl_output_max_chars: usize` to `LoopConfig` (default: 0 = no
limit). When `include_code_output` is true and output exceeds this,
truncate with `"...\n[truncated]"`.

### Phase 2: Adapt BrowseComp for ConversationLoop

**2.1 — Wrap BrowseCompBridge**

`ConversationBridge` already has `inner: Option<Box<dyn HostBridge>>`.
BrowseComp functions (`search`, `vector_search`, `get_document`,
`llm_query`, `batch_llm_query`) don't collide with conversation
functions (`ask_user`, `send_message`, `compact_session`). The wrapping
is zero-conflict:

```rust
let conv_bridge = ConversationBridge::new(event_tx, ask_handle, Some(Box::new(browsecomp_bridge)));
let repl = ReplAgent::new(external_fns, Box::new(conv_bridge));
```

**2.2 — Implement BenchIterationCallback**

Preserves the exact iteration prompt logic from gw-bench:

```rust
struct BenchIterationCallback {
    query: String,
    model: String,
    llm: OllamaClient,
}

impl IterationCallback for BenchIterationCallback {
    fn before_iteration(&mut self, iter: usize, max: usize, repl: &ReplAgent) -> Option<String> {
        let vars = build_variables_info(repl);
        if iter == 0 {
            Some(first_iteration_prompt(&self.query, &vars))
        } else if iter >= max - 2 {
            Some(final_iteration_prompt(&self.query, &vars))
        } else {
            Some(standard_iteration_prompt(&self.query, iter, max, &vars))
        }
    }

    fn on_max_iterations(&mut self, query: &str) -> Option<String> {
        // fallback_extract() using self.llm
    }
}
```

### Phase 3: Rewrite run_single_query

**3.1 — New flow:**

1. Pre-search pipeline runs unchanged — produces `context_hits`.
2. Create `BrowseCompBridge` + wrap in `ConversationBridge`.
3. Create `ReplAgent` with external functions: `search`, `vector_search`,
   `get_document`, `llm_query`, `batch_llm_query`, `FINAL`.
4. Seed REPL variables: `context`, `question`, `answer_type`.
5. Create `ConversationLoop` with:
   - `include_code_output: true`
   - `recency_window: 20` (keep all iterations within a single query)
   - `iteration_callback: BenchIterationCallback`
   - `answer_validator: |a| !is_refusal_answer(a)`
   - `auto_compact_after_turns: None`
   - `snapshot_policy: disabled`
6. Call `loop.handle_turn("Begin investigating.")`.
7. Extract answer from `TurnResult.response`.
8. Build `RunRecord` from `SessionTree` entries.

**3.2 — RunRecord from SessionTree**

Walk `tree.path_to_leaf()` and translate:
- `UserMessage` → `TrajectoryMessage { role: "user" }`
- `AssistantMessage` → `TrajectoryMessage { role: "assistant" }` +
  `ResultEntry` for each code block
- `CodeExecution` → `ResultEntry { entry_type: "tool_call" }`

This preserves GEPA-compatible JSON output.

### Phase 4: New Capabilities (Post-Integration)

**4.1 — Postgres trajectory persistence**

`--postgres-url` flag on gw-bench. Each query's tree persists to
Postgres via `flush_to_pg()`. Enables SQL-based failure analysis across
runs.

**4.2 — Branching retry**

After a query completes with fallback extraction (low confidence):
1. `loop.save_snapshot()` after pre-search
2. `loop.switch_branch(pre_search_entry, false)` to reset
3. Re-run with different steering ("Try a completely different angle")
4. Compare answers from both branches, pick best

Controlled by `--retry-branches N` flag.

**4.3 — Failure-driven steering**

When code execution fails, the `IterationCallback` inspects recent tree
entries and adjusts the next prompt. Example: if `get_document` returned
empty, steer to "That document was empty. Try a different docid."

---

## 5. Dependency Chain

```
Phase 1 (gw-loop extensions)
  1.1 CodeExecution feedback    ← prerequisite for Phase 3
  1.2 Think-tag stripping       ← prerequisite for Phase 3
  1.3 IterationCallback trait   ← prerequisite for Phase 2.2
  1.4 Answer validator          ← prerequisite for Phase 3
  1.5 Output truncation         ← prerequisite for Phase 3

Phase 2 (bench adaptation)
  2.1 Wrap BrowseCompBridge     ← depends on 1.3
  2.2 BenchIterationCallback    ← depends on 1.3

Phase 3 (integration)
  3.1 Rewrite run_single_query  ← depends on 1.*, 2.*
  3.2 RunRecord from tree       ← depends on 3.1

Phase 4 (new capabilities)     ← depends on 3.*
  4.1 Postgres persistence
  4.2 Branching retry
  4.3 Failure-driven steering
```

---

## 6. Risk Assessment

**High: accuracy regression from context format change.**
`build_turn_context()` formats messages differently than the flat
`Vec<Message>` approach. Mitigation: set `recency_window` high (20+)
for benchmark use. Run A/B comparison on 5 queries before full run.

**High: REPL output feedback fidelity.**
The current code carefully formats output with truncation and combines
stdout + return values. The new `CodeExecution` entry stores `stdout`
and `result` separately. The rendering in `build_turn_context()` must
replicate the exact same format. Mitigation: extract formatting logic
into a shared utility.

**Medium: iteration prompt timing.**
In gw-bench, iteration prompts are the user message for each LLM call.
In ConversationLoop, subsequent rLM iterations don't have user messages.
The `IterationCallback` must append these as `UserMessage` entries, not
steering messages (which get `[Steering]` prefix). Mitigation: when
callback returns a prompt, append as `EntryType::UserMessage`.

**Low: non-Send handling.**
Both systems solve this. gw-bench runs sync with `block_on()`. Making
`run_single_query` async eliminates the issue.

**Low: BrowseCompBridge delegation.**
ConversationBridge fallthrough routing is already tested. BrowseComp
function names don't collide with conversation function names.

---

## 7. What NOT to Change

- Pre-search pipeline (runs before ConversationLoop)
- BrowseCompBridge host function implementations
- BenchConfig TOML format and CLI flags
- RunRecord JSON output schema
- GEPA evaluator and scoring logic

---

## 8. Success Criteria

1. `cargo test --workspace` passes
2. RunRecord JSON schema unchanged
3. Accuracy >= 11/30 (no regression)
4. Optional: branching retry improves to >= 13/30
5. Optional: Postgres persistence enables SQL trajectory analysis

---

## 9. Estimated Effort

| Phase | Scope | Effort |
|-------|-------|--------|
| Phase 1 | 5 backward-compatible gw-loop extensions | ~200 LOC |
| Phase 2 | Bridge wrapping + callback | ~150 LOC |
| Phase 3 | run_single_query rewrite + RunRecord translation | ~300 LOC |
| Phase 4 | New capabilities | ~200 LOC each |
| **Total (Phases 1-3)** | Core integration | **~650 LOC** |
