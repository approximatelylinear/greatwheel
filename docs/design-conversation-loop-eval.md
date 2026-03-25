# Design: Conversation Loop Evaluation Framework

**Status:** Implemented
**Date:** 2026-03-20 (design), 2026-03-23 (implementation complete)
**Companion to:** [design-conversation-loop.md](design-conversation-loop.md)

---

## 0. Implementation Status

The evaluation framework is fully operational in three modes:

### Current scores

**Mode 1 — Deterministic (no LLM):** 40/40 assertions, 9 scenarios, 8 capabilities

```
branching            4/4
compaction           7/7
context_management   4/4
follow_up            2/2
session_resumption   7/7
state_persistence    3/3
steering             2/2
tree_integrity      11/11
```

**Mode 2 — Live (qwen3.5:9b):** 47/47 assertions, 17 scenarios, 3 capabilities

```
error_recovery         2/2
instruction_retention  7/7
state_persistence     38/38
```

**Mode 3 — Adapter (benchmark conversion):** MINT 10/10, tau-bench 6/6

### What's built

| Component | Status |
|-----------|--------|
| Scenario TOML parser | **Done** — `scenario.rs` with 5 turn types, 9 assertion kinds |
| Deterministic runner | **Done** — `runner.rs` with mock LLM + real ouros REPL |
| Live runner | **Done** — `live_runner.rs` with `OllamaLlmClient` + real ouros REPL |
| Per-capability reporter | **Done** — `reporter.rs` with table output |
| MINT adapter | **Done** — `adapters/mint.rs`, JSONL → Scenario |
| tau-bench adapter | **Done** — `adapters/tau_bench.rs`, JSON → Scenario |
| conv-loop skill | **Done** — autonomous development loop |
| Float-tolerant assertions | **Done** — `values_approx_equal()` with 1e-9 tolerance |
| Snapshot/restore turns | **Done** — `snapshot`, `restore_snapshot` turn types |
| Branch switch turns | **Done** — `switch_branch` with snapshot+replay restore |

### Scenario catalog

**Deterministic (S01-S10):**
| ID | Name | Capability | Assertions |
|----|------|-----------|------------|
| S01 | Variable persistence | state_persistence | 1 |
| S02 | Function definition persistence | state_persistence | 2 |
| S03 | Context recency window | context_management | 4 |
| S04 | Compaction preserves REPL state | compaction | 7 |
| S05 | Branch switch and restore | branching | 4 |
| S06 | Steering message injection | steering | 2 |
| S07 | Follow-up sequencing | follow_up | 2 |
| S09 | Session suspend and resume | session_resumption | 7 |
| S10 | Tree integrity under stress | tree_integrity | 11 |

**Live (L01-L17):**
| ID | Name | Capability | Assertions |
|----|------|-----------|------------|
| L01 | Variable persistence | state_persistence | 2 |
| L02 | Function definition persistence | state_persistence | 2 |
| L03 | Multi-turn computation | state_persistence | 3 |
| L04 | Instruction retention | instruction_retention | 4 |
| L05 | Data pipeline (5 turns) | state_persistence | 3 |
| L06 | Iterative refinement | state_persistence | 2 |
| L07 | Long-range variable reference | instruction_retention | 3 |
| L08 | Complex data structures | state_persistence | 3 |
| L09 | Error recovery | error_recovery | 2 |
| L10 | Conditional accumulation | state_persistence | 2 |
| L11 | String processing | state_persistence | 3 |
| L12 | Dictionary aggregation | state_persistence | 2 |
| L13 | Fibonacci algorithm | state_persistence | 4 |
| L14 | Matrix operations | state_persistence | 3 |
| L15 | Stateful counter pattern | state_persistence | 3 |
| L16 | Nested data traversal | state_persistence | 2 |
| L17 | Set operations | state_persistence | 4 |

### Running

```bash
# Build
cargo build --bin conv-loop-bench

# Mode 1: Deterministic
cargo run --bin conv-loop-bench -- --mode deterministic

# Mode 2: Live
cargo run --bin conv-loop-bench -- --mode live \
    --scenarios bench/conv-loop/scenarios/live --model "qwen3.5:9b"

# Mode 3: Adapter (MINT)
cargo run --bin conv-loop-bench -- --mode adapter \
    --adapter mint --data bench/conv-loop/data/mint_sample.jsonl \
    --model "qwen3.5:9b"

# Mode 3: Adapter (tau-bench)
cargo run --bin conv-loop-bench -- --mode adapter \
    --adapter tau-bench --data bench/conv-loop/data/tau_bench_sample.json \
    --model "qwen3.5:9b"
```

### Remaining work

- **LLM-judged assertions** — `InstructionFollowed { instruction, turns_ago }` using an LLM to evaluate free-text compliance
- **S08: Channel routing** — needs multi-channel support in the eval harness
- **Full benchmark datasets** — download tau-bench/MINT from GitHub/HuggingFace, run at scale
- **Results TSV logging** — currently manual; could auto-append after each run

---

## 1. Problem Statement

The browsecomp-loop skill iterates on a scalar metric: accuracy out of
30 queries. Change a prompt → run → score → keep or revert. That works
because the benchmark has a fixed evaluation function and the variable
is the prompt.

The conversation loop is different. We're building **system infrastructure**
— session tree, context builder, event dispatch, compaction, branching,
channel multiplexing. The things that can go wrong are structural:

- Context builder drops a constraint from turn 4 when building turn 8's prompt.
- Compaction loses a variable the rLM defined in turn 2.
- Branch restore fails to recover a function definition.
- Steering message arrives but isn't injected before the next LLM call.
- Follow-up message fires too early, interrupting an in-progress turn.
- Channel routing sends a response to the wrong adapter.

A single accuracy number can't diagnose these. We need **per-capability
assertions** that tell us exactly which structural property broke.

---

## 2. Evaluation Architecture

### 2.1 Test Scenario Structure

Each scenario is a scripted multi-turn conversation that exercises one
or more capabilities of the conversation loop. Scenarios are deterministic
scripts, not LLM-generated — the evaluation framework drives both sides
of the conversation.

```rust
// bench/conv-loop/src/scenario.rs

/// A scripted multi-turn scenario.
pub struct Scenario {
    pub id: String,
    pub name: String,
    pub description: String,
    /// Which capabilities this scenario tests.
    pub capabilities: Vec<Capability>,
    /// The scripted turns.
    pub turns: Vec<ScenarioTurn>,
    /// Assertions checked after each turn and at scenario end.
    pub assertions: Vec<Assertion>,
}

/// A single turn in a scripted scenario.
pub enum ScenarioTurn {
    /// Simulate a user message.
    UserMessage {
        content: String,
        channel: String,
    },
    /// Simulate a steering message (arrives mid-turn).
    SteeringMessage {
        content: String,
        /// Inject after this many host function calls in the current turn.
        inject_after_n_calls: u32,
    },
    /// Simulate a follow-up (queued for after current turn ends).
    FollowUp {
        content: String,
    },
    /// Trigger compaction.
    Compact,
    /// Switch to a different branch.
    SwitchBranch {
        target_turn: usize,
        summarize: bool,
    },
    /// Wait for the current turn to complete before proceeding.
    WaitForTurnComplete,
    /// Assert something about the current state.
    Assert(Assertion),
}
```

### 2.2 Capabilities

Each capability is a discrete structural property of the conversation loop.
Scenarios can test one capability in isolation or combine several.

```rust
pub enum Capability {
    /// Context builder includes recent messages correctly.
    ContextRecency,
    /// REPL variables survive across turns.
    VariablePersistence,
    /// Function definitions survive across turns.
    DefinitionPersistence,
    /// Instructions given in turn N are followed in turn N+M.
    InstructionRetention,
    /// Compaction preserves REPL state and produces usable summary.
    CompactionFidelity,
    /// Branch switch restores correct REPL snapshot.
    BranchRestore,
    /// Branch summary captures key information from abandoned path.
    BranchSummary,
    /// Steering messages arrive before the next LLM call.
    SteeringInjection,
    /// Follow-up messages fire only after turn completion.
    FollowUpTiming,
    /// Responses route to the correct channel.
    ChannelRouting,
    /// channel.ask() blocks until reply and returns it.
    InteractiveInput,
    /// Session can be suspended (evicted) and resumed correctly.
    SessionResumption,
    /// Auto-snapshot fires according to policy.
    AutoSnapshot,
    /// Session tree records all entries with correct parent linkage.
    TreeIntegrity,
}
```

### 2.3 Assertions

Assertions are the evaluation primitives. Each checks one specific
property of the loop state after a turn.

```rust
pub enum Assertion {
    // === REPL state ===

    /// Variable exists in REPL with expected value.
    VariableEquals {
        name: String,
        expected: Value,
    },
    /// Variable exists (any value).
    VariableExists {
        name: String,
    },
    /// Function is defined and callable.
    FunctionDefined {
        name: String,
    },

    // === Context builder ===

    /// The LLM prompt for this turn contains the given substring.
    PromptContains {
        substring: String,
    },
    /// The LLM prompt does NOT contain the given substring
    /// (e.g., a compacted turn should not appear verbatim).
    PromptExcludes {
        substring: String,
    },
    /// The LLM prompt includes exactly N user/assistant message pairs.
    PromptMessageCount {
        expected: usize,
    },

    // === Session tree ===

    /// The session tree has exactly N entries.
    TreeEntryCount {
        expected: usize,
    },
    /// The active leaf is an entry of the given type.
    ActiveLeafType {
        expected_type: String,
    },
    /// Entry at position N in the root→leaf path has the given type.
    PathEntryType {
        position: usize,
        expected_type: String,
    },

    // === Events ===

    /// A Response event was emitted to the given channel.
    ResponseEmitted {
        channel: String,
    },
    /// A Response event contains the given substring.
    ResponseContains {
        substring: String,
    },
    /// No Response event was emitted (for negative tests).
    NoResponseEmitted,

    // === Timing ===

    /// Steering message was injected before the Nth LLM call in this turn.
    SteeringBeforeLlmCall {
        call_index: usize,
    },
    /// Follow-up was NOT processed during the current turn.
    FollowUpNotProcessedDuringTurn,

    // === Instruction following (LLM-judged) ===

    /// The agent's response follows an instruction given N turns ago.
    /// This is the one assertion that requires an LLM judge.
    InstructionFollowed {
        instruction: String,
        turns_ago: usize,
    },
}
```

### 2.4 Two Evaluation Modes

**Mode 1: Deterministic (no LLM).** The rLM is replaced with a mock
that returns scripted code blocks. This tests the loop infrastructure
in isolation — does the context builder work? does compaction preserve
state? does branching restore snapshots? These tests are fast, reliable,
and run in CI.

**Mode 2: End-to-end (with LLM).** The real rLM runs against scenarios
drawn from multi-turn benchmarks (tau-bench, MINT, MultiChallenge). This
tests whether the loop infrastructure actually helps the LLM follow
instructions across turns. These tests are slower and non-deterministic,
run as a benchmark suite.

```
┌─────────────────────────────────────────────┐
│          Scenario Runner                     │
│                                              │
│  Load scenario                               │
│  For each turn:                              │
│    → Send event to ConversationLoop          │
│    → Wait for turn completion                │
│    → Run assertions against loop state       │
│  End: run final assertions                   │
│  Report: per-assertion pass/fail + summary   │
└──────────────────┬──────────────────────────┘
                   │
          ┌────────┴─────────┐
          ▼                  ▼
┌──────────────────┐  ┌──────────────────┐
│  Mode 1: Mock    │  │  Mode 2: Live    │
│                  │  │                  │
│  MockReplAgent   │  │  ReplAgent       │
│  MockLlmClient   │  │  OllamaClient   │
│  Scripted code   │  │  Real inference  │
│  blocks          │  │                  │
│                  │  │  + LLM-judged    │
│  Fast, CI-safe   │  │  assertions      │
└──────────────────┘  └──────────────────┘
```

---

## 3. Scenario Catalog

### 3.1 Core Infrastructure Scenarios (Mode 1 — deterministic)

These use mock LLM/REPL responses to test loop mechanics in isolation.

#### S01: Basic Multi-Turn Variable Persistence

Tests: `VariablePersistence`, `ContextRecency`

```
Turn 1: User says "Set x to 42"
        Mock LLM returns: ```python\nx = 42\nFINAL("done")\n```
        Assert: VariableEquals { name: "x", expected: 42 }

Turn 2: User says "What is x?"
        Mock LLM returns: ```python\nFINAL(str(x))\n```
        Assert: VariableEquals { name: "x", expected: 42 }
        Assert: ResponseContains { substring: "42" }
```

#### S02: Function Definition Persistence

Tests: `DefinitionPersistence`

```
Turn 1: User says "Define a helper"
        Mock LLM returns: ```python\ndef double(n): return n * 2\nFINAL("defined")\n```
        Assert: FunctionDefined { name: "double" }

Turn 2: User says "Use the helper"
        Mock LLM returns: ```python\nresult = double(21)\nFINAL(str(result))\n```
        Assert: ResponseContains { substring: "42" }
```

#### S03: Context Window Recency

Tests: `ContextRecency`

```
Turns 1-8: User sends numbered messages, mock LLM sets variables
Turn 9: User says "New question"
        Assert: PromptContains { substring: "Turn 7" }  // within window
        Assert: PromptContains { substring: "Turn 8" }
        Assert: PromptExcludes { substring: "Turn 1" }  // outside window
        Assert: PromptExcludes { substring: "Turn 2" }
```

#### S04: Compaction Preserves REPL State

Tests: `CompactionFidelity`, `VariablePersistence`

```
Turns 1-6: Build up variables (a, b, c, helper_fn)
Turn 7: Trigger Compact
        Assert: VariableEquals { name: "a", ... }
        Assert: VariableEquals { name: "b", ... }
        Assert: VariableEquals { name: "c", ... }
        Assert: FunctionDefined { name: "helper_fn" }
        Assert: PromptExcludes { substring: "Turn 1 content" }
        Assert: PromptContains { substring: "Summary" }  // compaction summary present
```

#### S05: Branch Switch and Restore

Tests: `BranchRestore`, `BranchSummary`

```
Turns 1-3: Build state (x=1, y=2, z=3)
Turn 4: User says "Try approach A" → mock sets x=100
Turn 5: SwitchBranch to Turn 3 (with summarize=true)
        Assert: VariableEquals { name: "x", expected: 1 }  // restored to pre-branch
        Assert: VariableEquals { name: "y", expected: 2 }
        Assert: VariableEquals { name: "z", expected: 3 }
        Assert: PathEntryType { position: -1, expected_type: "BranchSummary" }
Turn 6: User says "Try approach B" → mock sets x=200
        Assert: VariableEquals { name: "x", expected: 200 }
```

#### S06: Steering Message Injection

Tests: `SteeringInjection`

```
Turn 1: User says "Analyze the codebase" (triggers multi-step rLM loop)
        After 2 host calls: inject SteeringMessage "Focus on auth module"
        Assert: SteeringBeforeLlmCall { call_index: 1 }  // injected before next LLM call
        Assert: PromptContains { substring: "Focus on auth module" }
```

#### S07: Follow-Up Timing

Tests: `FollowUpTiming`

```
Turn 1: User says "Search for X" (triggers rLM loop)
        Queue FollowUp "Also search for Y"
        Assert: FollowUpNotProcessedDuringTurn  // not processed mid-turn
        WaitForTurnComplete
Turn 2: (automatically triggered by follow-up)
        Assert: PromptContains { substring: "Also search for Y" }
```

#### S08: Channel Routing

Tests: `ChannelRouting`

```
Turn 1: UserMessage { content: "Hello", channel: "slack" }
        Mock LLM calls channel.send("Hi there")
        Assert: ResponseEmitted { channel: "slack" }

Turn 2: UserMessage { content: "Hello again", channel: "http" }
        Mock LLM calls channel.send("Hi again")
        Assert: ResponseEmitted { channel: "http" }
```

#### S09: Session Suspend and Resume

Tests: `SessionResumption`, `VariablePersistence`

```
Turns 1-3: Build state (data={"key": "value"}, process_fn defined)
Action: Suspend session (simulate idle eviction)
Action: Resume session
Turn 4: User says "Use the data"
        Assert: VariableEquals { name: "data", expected: {"key": "value"} }
        Assert: FunctionDefined { name: "process_fn" }
```

#### S10: Tree Integrity Under Stress

Tests: `TreeIntegrity`

```
Turns 1-5: Normal conversation
Turn 6: Branch to turn 3
Turns 7-9: New branch
Turn 10: Compact
Turn 11: Branch to turn 8
Assert: All parent_id links are valid
Assert: Root→leaf path is correct
Assert: No orphaned entries
Assert: Compaction node is on the active path
```

### 3.2 Instruction Following Scenarios (Mode 2 — with LLM)

These use the real rLM and adapt tasks from existing benchmarks. The
assertions use both deterministic checks (REPL state) and LLM-judged
checks (instruction following quality).

#### S11: Retained Constraint (adapted from MultiChallenge)

```
Turn 1: "From now on, always respond in bullet points."
Turn 2: "What are the benefits of Rust?"
        Assert: InstructionFollowed {
            instruction: "respond in bullet points",
            turns_ago: 1,
        }
Turn 3-5: Various questions
Turn 6: "What's the difference between async and threads?"
        Assert: InstructionFollowed {
            instruction: "respond in bullet points",
            turns_ago: 5,
        }
```

#### S12: Tool Use Across Turns (adapted from MINT)

```
Turn 1: "Search for documents about rate limiting"
        Assert: at least one search() host call occurred
        Assert: VariableExists { name containing "results" or "docs" }
Turn 2: "Now find the specific rate limits for Stripe"
        Assert: search() call uses terms from turn 1 results
        Assert: VariableExists for Stripe-specific data
Turn 3: "Compare those with what you found earlier"
        Assert: response references both prior result sets
```

#### S13: Multi-Step Task with Compaction (adapted from tau-bench)

```
Turns 1-4: Step-by-step airline booking task
            (search flights, select, enter passenger info, confirm)
Turn 5: Compact
Turn 6: "What flight did I book?"
        Assert: response contains correct flight details
        Assert: PromptExcludes (raw booking steps not in prompt)
        Assert: VariableExists (booking details in REPL)
```

#### S14: Branching Exploration

```
Turns 1-2: "Research the best database for time series data"
Turn 3: "Let's explore InfluxDB" → mock searches
Turn 4: "Actually, let's try TimescaleDB instead"
        SwitchBranch to turn 2
Turn 5: "Let's explore TimescaleDB" → mock searches
Turn 6: "Compare what we found about InfluxDB with TimescaleDB"
        Assert: BranchSummary for InfluxDB path is in context
        Assert: response references both databases
```

### 3.3 Benchmark Adapter Scenarios (Mode 2 — automated)

These load tasks from external benchmarks and run them through the
conversation loop, checking both task completion and structural properties.

#### tau-bench Adapter

```rust
/// Load a tau-bench task and run it as a multi-turn scenario.
fn adapt_tau_bench(task: TauBenchTask) -> Scenario {
    // 1. Convert tau-bench user simulation into ScenarioTurns.
    // 2. Map tau-bench tool definitions to host functions.
    // 3. Add structural assertions:
    //    - VariablePersistence after each turn
    //    - TreeIntegrity at end
    //    - ContextRecency at each turn
    // 4. Add tau-bench's own task-completion check as final assertion.
}
```

#### MINT Adapter

```rust
/// Load a MINT task and run it as a multi-turn scenario.
fn adapt_mint(task: MintTask) -> Scenario {
    // MINT is the closest match — it already uses code execution.
    // 1. Map MINT tool calls to host functions.
    // 2. Map MINT user feedback to follow-up messages.
    // 3. Add structural assertions on REPL state.
    // 4. Add MINT's own correctness check as final assertion.
}
```

---

## 4. Metrics and Reporting

### 4.1 Per-Assertion Results

Every assertion produces a structured result:

```rust
pub struct AssertionResult {
    pub assertion: Assertion,
    pub passed: bool,
    pub actual: Option<String>,    // what was observed
    pub message: Option<String>,   // diagnostic detail
    pub turn: usize,               // which turn this was checked at
}
```

### 4.2 Per-Capability Scores

Aggregate assertion results by capability:

```
Capability                 Pass  Fail  Score
─────────────────────────  ────  ────  ─────
ContextRecency               8     0  1.000
VariablePersistence         12     1  0.923
DefinitionPersistence        4     0  1.000
InstructionRetention         3     2  0.600  ← problem area
CompactionFidelity           6     0  1.000
BranchRestore                3     1  0.750  ← problem area
BranchSummary                2     0  1.000
SteeringInjection            4     0  1.000
FollowUpTiming               2     0  1.000
ChannelRouting               4     0  1.000
SessionResumption            2     0  1.000
TreeIntegrity                5     0  1.000
─────────────────────────  ────  ────  ─────
TOTAL                       55     4  0.932
```

### 4.3 Results File

Similar to browsecomp's `results.tsv`, but structured around capabilities:

```
bench/conv-loop/results.tsv

commit  total_score  context  variables  instructions  compaction  branching  steering  status  description
```

### 4.4 Failure Diagnostics

When an assertion fails, the framework emits a diagnostic block:

```
FAIL S05:BranchRestore turn=5
  assertion: VariableEquals { name: "x", expected: 1 }
  actual:    x = 100
  diagnosis: Branch switch did not restore REPL snapshot.
             The snapshot at turn 3 was not found — check
             auto-snapshot policy (currently every_n_turns=3,
             but only 2 turns elapsed since last snapshot).
  affected:  BranchRestore capability
  suggested: Lower snapshot frequency or add snapshot before branch.
```

This is the structural equivalent of browsecomp's per-query failure
analysis. Instead of "query 17 failed because the document wasn't
retrieved," we get "branch restore failed because the snapshot policy
didn't fire at the right time."

---

## 5. The Development Loop

### 5.1 How It Differs from browsecomp-loop

| Aspect | browsecomp-loop | conv-loop |
|--------|----------------|-----------|
| Variable | Prompts, search params | System architecture (types, logic, policies) |
| Metric | Scalar accuracy (X/30) | Per-capability score vector |
| Feedback | "Accuracy went up/down" | "BranchRestore broke because snapshot policy gap" |
| Eval speed | ~5 min (30 LLM queries) | ~10s mode 1 (deterministic), ~5 min mode 2 (LLM) |
| Iteration target | `main.rs` prompts | `gw-loop`, `gw-runtime`, `gw-core` |
| Revert condition | Accuracy dropped | Any capability regressed |

### 5.2 The Loop

```
LOOP:

1. Check state:
   - Read bench/conv-loop/results.tsv
   - Identify lowest-scoring capabilities
   - Read failing scenario diagnostics

2. Form hypothesis:
   - "BranchRestore fails because restore_snapshot doesn't
     replay definitions" → fix restore logic
   - "InstructionRetention drops at turn 6 because context
     window is 5" → increase recency window or add instruction
     persistence to REPL variables
   - "CompactionFidelity fails for non-serializable variables"
     → improve snapshot serialization

3. Edit the relevant crate:
   - gw-loop/src/ for loop logic, context builder, compaction
   - gw-runtime/src/ for REPL agent extensions
   - gw-core/src/ for type changes

4. Build: cargo build --release --bin conv-loop-bench

5. Run Mode 1 (deterministic):
   cargo run --release --bin conv-loop-bench -- --mode deterministic
   (Fast — runs in seconds. Catches structural regressions.)

6. If Mode 1 passes, run Mode 2 (with LLM):
   cargo run --release --bin conv-loop-bench -- --mode live \
       --scenarios bench/conv-loop/scenarios/
   (Slower — catches instruction-following regressions.)

7. Log results to bench/conv-loop/results.tsv

8. If any capability regressed: revert and rethink
   If all capabilities maintained or improved: keep commit

9. GOTO 1
```

### 5.3 When to Add New Scenarios

- **Implementing a new feature** (e.g., compaction) → write S04 first,
  then implement until it passes. TDD for infrastructure.
- **Found a bug in production** → write a scenario that reproduces it,
  then fix.
- **Adding a benchmark adapter** (e.g., tau-bench) → adapter converts
  external tasks into scenarios automatically.

---

## 6. File Layout

```
bench/conv-loop/
├── src/
│   ├── main.rs           # CLI: --mode deterministic|live|adapter
│   ├── scenario.rs       # Scenario, ScenarioTurn (5 types), Assertion (9 kinds)
│   ├── runner.rs         # deterministic execution (mock LLM + real ouros)
│   ├── live_runner.rs    # live execution (OllamaLlmClient + real ouros)
│   ├── mock.rs           # MockBridge for deterministic mode
│   ├── assertions.rs     # assertion evaluation + float-tolerant comparison
│   ├── reporter.rs       # per-capability scoring table
│   └── adapters/
│       ├── mod.rs        # BenchmarkAdapter trait
│       ├── mint.rs       # MINT JSONL → Scenario converter
│       └── tau_bench.rs  # tau-bench JSON → Scenario converter
├── scenarios/
│   ├── s01_variable_persistence.toml
│   ├── s02_definition_persistence.toml
│   ├── s03_context_recency.toml
│   ├── s04_compaction.toml
│   ├── s05_branch_restore.toml
│   ├── s06_steering.toml
│   ├── s07_follow_up.toml
│   ├── s09_session_resume.toml
│   ├── s10_tree_integrity.toml
│   └── live/
│       ├── l01_variable_persistence.toml
│       ├── l02_function_persistence.toml
│       ├── l03_multi_turn_computation.toml
│       ├── l04_instruction_retention.toml
│       ├── l05_data_pipeline.toml
│       ├── l06_iterative_refinement.toml
│       ├── l07_long_range_reference.toml
│       ├── l08_complex_structures.toml
│       ├── l09_error_recovery.toml
│       ├── l10_conditional_accumulation.toml
│       ├── l11_string_processing.toml
│       ├── l12_dict_aggregation.toml
│       ├── l13_fibonacci.toml
│       ├── l14_matrix_ops.toml
│       ├── l15_stateful_counter.toml
│       ├── l16_recursive_data.toml
│       └── l17_set_operations.toml
├── data/
│   ├── mint_sample.jsonl       # 10 sample MINT tasks
│   └── tau_bench_sample.json   # 3 sample tau-bench tasks
└── results.tsv
```

Scenarios are defined in TOML for readability and non-programmer
editability. The runner parses them into `Scenario` structs.

### 6.1 Scenario TOML Format (as implemented)

```toml
name = "S05: Branch switch and restore"
capability = "branching"
system_prompt = "You are a helpful assistant with a Python REPL."
recency_window = 100  # optional, default 100

# Turn types:

[[turns]]
type = "user_message"
content = "Set x to 1"
mock_response = """         # only for deterministic mode
` ` `python
x = 1
` ` `
"""

[[turns]]
type = "snapshot"              # save REPL state to tree

[[turns]]
type = "restore_snapshot"      # simulate suspend/resume

[[turns]]
type = "switch_branch"
target_turn = 3                # switch to Nth user turn (1-indexed)

[[turns]]
type = "assert"

[[turns.checks]]
kind = "variable_equals"
name = "x"
expected = 1

# Assertion kinds:
#   variable_equals    — exact match (with float tolerance)
#   variable_exists    — variable exists (any value)
#   variable_absent    — variable does NOT exist
#   function_defined   — Python function is defined
#   context_contains   — LLM prompt contains substring
#   context_excludes   — LLM prompt excludes substring
#   tree_size_gte      — tree has >= N entries
#   path_length        — root-to-leaf path has exactly N entries
#   all_parents_valid  — all parent_id references are valid
```

---

## 7. Integration with conv-loop Skill

The skill definition (similar to browsecomp-loop) will drive the
autonomous development loop:

```
conv-loop skill workflow:

1. Read state: results.tsv + failing diagnostics
2. Identify lowest-scoring capability
3. Read the relevant scenario + the relevant gw-loop source
4. Form hypothesis for why the capability is failing
5. Edit gw-loop / gw-runtime / gw-core source
6. Build
7. Run Mode 1 (deterministic) → fast feedback
8. If Mode 1 passes, run Mode 2 (live) → slow feedback
9. Log results
10. Keep or revert
11. Loop
```

The key difference from browsecomp-loop: **the skill targets system code
(types, logic, policies), not prompts.** The scenarios are the fixed
reference (like sample30.tsv), and the system implementation is the
variable being optimized.

---

## 8. Implementation Order

| Phase | What | Status |
|---|---|---|
| 1 | Scenario types (`Scenario`, `ScenarioTurn`, `Assertion`) | **Done** — 5 turn types, 9 assertion kinds |
| 1 | Scenario TOML parser | **Done** |
| 1 | S01-S03 scenarios | **Done** |
| 2 | Deterministic scenario runner | **Done** — real ouros REPL + mock LLM |
| 2 | Per-capability reporter | **Done** |
| 3 | S04-S10 scenarios | **Done** (S04-S07, S09-S10) |
| 4 | Live scenario runner (real LLM) | **Done** — `OllamaLlmClient` + `ConversationLoop` |
| 4 | LLM-judged assertion evaluator | Not started |
| 5 | Benchmark adapters (tau-bench, MINT) | **Done** — with sample data |
| 5 | L01-L17 live scenarios | **Done** — 17 scenarios, 3 capabilities |
| 6 | conv-loop skill definition | **Done** |

All phases complete except LLM-judged assertions.

---

## 9. Open Questions

**Scenario determinism in Mode 2.** With a real LLM, the rLM might not
set variables with the exact names the assertions expect. Options:
(a) use LLM-judged assertions for everything in Mode 2; (b) seed the
rLM with expected variable names in the prompt; (c) check for semantic
equivalence rather than exact match. Start with (b) — it's the simplest
and closest to how we'd actually use the system.

**Mock fidelity.** The mock REPL agent needs to actually execute Python
(set variables, define functions) for deterministic assertions to work.
Options: (a) use a real ouros session with a mock LLM (cheapest — we
already have ouros); (b) build a minimal Python variable store. Start
with (a) — the mock is only the LLM, not the REPL.

**Scenario coverage.** How do we know the scenario catalog covers the
system? Track which `gw-loop` functions are exercised by at least one
scenario. Add scenarios when coverage gaps appear.

**Regression vs. improvement.** If a change improves BranchRestore but
slightly degrades InstructionRetention, is that a keep or revert? Propose:
never allow a capability to drop below its previous score. If a change
causes regression in any capability, revert. This is stricter than
browsecomp-loop (which only tracks scalar accuracy) but appropriate for
infrastructure where partial regressions compound.

**Eval speed budget.** Mode 1 should complete in <30 seconds for the
full S01-S10 suite. Mode 2 should complete in <10 minutes. If scenarios
grow beyond this, split into "fast" (CI) and "full" (manual) suites.
