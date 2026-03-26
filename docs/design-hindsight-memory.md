# Design: Structured Agent Memory — Hindsight Integration

**Status:** In Progress — BrowseComp Phase A + gw-memory Phases 1-3 implemented
**Date:** 2026-03-26
**Paper:** [Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects](https://arxiv.org/abs/2512.12818) — Vectorize.io & Virginia Tech

---

## 0. Implementation Status

### BrowseComp Phase A (BC-1 + BC-2): Implemented (2026-03-26)

FactRegistry with structured fact accumulation and confidence-scored candidates.
Awaiting benchmark evaluation against the 40% (12/30) baseline.

**Files changed:**

| File | Change |
|---|---|
| `bench/browsecomp/fact_registry.py` | **New.** Standalone FactRegistry class — entity extraction, fact storage, candidate tracking with confidence scoring (reinforce/contradict/weaken) |
| `bench/browsecomp/ollama_client.py` | Imports FactRegistry, injects as `facts` in REPL namespace. Updated SYSTEM_PROMPT with FACT TRACKING docs and example workflow. Updated QUERY_TEMPLATE |
| `crates/gw-bench/src/main.rs` | Added `FACT_REGISTRY_BOOTSTRAP` constant (Python class embedded as string, executed via `agent.execute()` before rLM loop). Updated SYSTEM_PROMPT: added `facts` to TOOLS, added FACT TRACKING section, updated WORKFLOW (7 steps with propose/verify), updated EXAMPLE, updated RULES. Updated iteration nudges (halfway/last chance/final) to reference `facts.candidates()` and `facts.best_candidate()`. Added FactRegistry-first fallback: tries `facts.best_candidate()` before LLM fallback extraction. Bootstrap runs for both standard and conv_loop paths |

**Key design decisions:**
- Confidence deltas: reinforce +0.15, weaken -0.10, contradict -0.25 (tuned from Hindsight's symmetric ±0.1 to penalize contradictions more heavily — BrowseComp benefits from stickier correct answers)
- Entity extraction: regex-based (capitalized phrases, quoted strings, years) — no spaCy dependency, runs inside ouros sandbox
- Backward compatible: if agent ignores `facts`, pipeline works identically to before
- Dual implementation: standalone Python class (`fact_registry.py`) for `ollama_client.py`, embedded Python string constant for `gw-bench` ouros path

### gw-memory Phases 1-3: Implemented (2026-03-26)

**Phase 1 — Core schema + types:**
- Migration 008: `memory_kind` enum, `confidence`/`occurred_at`/`occurred_end`/`entities`
  columns, `memory_edges` table
- `gw-core`: `MemoryKind` (with `sqlx::Type` derive), `MemoryEdgeKind`
- `gw-memory`: `MemoryRecord` extended, `MemoryMeta` struct, `kind_filter` on `RecallOpts`

**Phase 2 — Event dispatch:**
- `HybridStore` accepts optional `DispatchFn`, dispatches `BeforeMemoryStore`/`AfterMemoryRecall`
- `EventData::Memory` extended with `meta: Option<Value>` for plugin-enriched metadata

**Phase 3 — `hindsight-retain` plugin:**

| File | What |
|---|---|
| `gw-engine/src/builtins/hindsight_retain.rs` | Plugin impl: `BeforeMemoryStore` handler (entity extraction via compiled regex NER, memory kind classification via content heuristics, default confidence for opinions). Registers `memory.extract_entities` host function. 8 unit tests |
| `gw-engine/src/builtins/mod.rs` | Builtins module |
| `gw-server/src/main.rs` | Plugin wired into `GreatWheelEngine` |

Deferred to async dispatch resolution (§6.2 Q7): LLM-powered fact extraction,
LLM entity resolution, causal edges, graph edge computation.

### gw-memory Phases 4-5: Not started

- `hindsight-recall` (Phase 4) — graph traversal, temporal parsing, optional reranking
- `hindsight-opinions` (Phase 5) — confidence evolution

See Section 3 for the full implementation plan.

---

## 1. Context

Greatwheel's `gw-memory` crate implements a tri-backend hybrid memory store
(Postgres + LanceDB + tantivy) with RRF fusion. Memories are flat key-value
records with JSONB values, scoped by org/user/agent/session. There is no
distinction between facts, beliefs, and observations, no graph structure
linking related memories, and no temporal awareness in retrieval.

The Hindsight paper demonstrates that structuring agent memory into typed
networks with graph edges and confidence-scored opinions yields a +44.6 point
improvement over full-context baselines on LongMemEval (39% → 83.6% with a
20B model). Their architecture achieves this through three operations:

- **Retain** — LLM-powered narrative fact extraction with entity resolution
- **Recall** — four-channel hybrid retrieval (semantic + BM25 + graph + temporal) fused via RRF
- **Reflect** — preference-conditioned response with evolving beliefs

We already have the Recall foundation (vector + BM25 + RRF). This design
adds the missing pieces: memory typing, graph edges, temporal retrieval,
opinion tracking, and structured fact extraction.

### 1.1 What We Already Have

| Hindsight concept | Greatwheel equivalent | Gap |
|---|---|---|
| Semantic retrieval (HNSW) | LanceDB vector search | None |
| BM25 keyword retrieval | tantivy full-text search | None |
| Reciprocal Rank Fusion | `fusion.rs` (k=60) | None |
| Cross-encoder reranker | — | Missing |
| Four memory networks | Flat JSONB values | No type distinction |
| Graph edges + spreading activation | — | No relationships |
| Temporal metadata + temporal retrieval | `created_at`/`updated_at` columns, unused in search | Not used in ranking |
| Entity resolution | — | Missing |
| Opinion confidence scores | — | Missing |
| Narrative fact extraction | — | Raw values stored as-is |
| Scoping (bank-level) | `MemoryScope` (Org/User/Agent/Session) | Equivalent |

### 1.2 Design Goals

1. **Typed memories** — distinguish facts, experiences, opinions, and observations
2. **Graph retrieval** — entity/causal/temporal edges with spreading activation as a new RRF channel
3. **Temporal retrieval** — time-aware queries as a new RRF channel
4. **Opinion evolution** — confidence-scored beliefs that update with evidence
5. **Narrative extraction** — LLM-powered fact extraction at write time
6. **Incremental** — each feature is independently useful and testable
7. **Backward compatible** — existing `store()`/`recall()` API continues to work

---

## 2. Design

### 2.1 Memory Types

Add a `MemoryKind` enum to `gw-core`:

```rust
/// Hindsight-inspired memory classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, sqlx::Type, Serialize, Deserialize)]
#[sqlx(type_name = "memory_kind", rename_all = "snake_case")]
pub enum MemoryKind {
    /// Objective facts about the external world.
    Fact,
    /// Agent's own biographical history (first-person).
    Experience,
    /// Subjective beliefs with confidence scores.
    Opinion,
    /// Preference-neutral entity summaries synthesized from facts.
    Observation,
}
```

**Postgres migration:**

```sql
CREATE TYPE memory_kind AS ENUM ('fact', 'experience', 'opinion', 'observation');

ALTER TABLE memories
    ADD COLUMN kind memory_kind NOT NULL DEFAULT 'fact',
    ADD COLUMN confidence FLOAT,          -- Only meaningful for 'opinion' kind
    ADD COLUMN occurred_at TIMESTAMPTZ,   -- When the fact happened (τs)
    ADD COLUMN occurred_end TIMESTAMPTZ,  -- End of occurrence interval (τe)
    ADD COLUMN entities TEXT[];           -- Canonical entity names extracted

CREATE INDEX idx_memories_kind ON memories(org_id, kind);
CREATE INDEX idx_memories_occurred ON memories(occurred_at) WHERE occurred_at IS NOT NULL;
CREATE INDEX idx_memories_entities ON memories USING GIN(entities) WHERE entities IS NOT NULL;
```

**Why these columns:**
- `kind` routes retrieval — opinion queries weight opinion memories higher
- `confidence` enables the Hindsight reinforcement rules
- `occurred_at`/`occurred_end` enable temporal retrieval (distinct from `created_at` which is when we learned the fact)
- `entities` enables entity-based graph traversal without a join

### 2.2 Memory Graph

A new `memory_edges` table captures relationships between memories:

```sql
CREATE TABLE memory_edges (
    from_id  UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id    UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL,  -- 'entity', 'temporal', 'semantic', 'causal'
    weight   FLOAT NOT NULL DEFAULT 1.0,
    PRIMARY KEY (from_id, to_id, kind)
);

CREATE INDEX idx_memory_edges_from ON memory_edges(from_id);
CREATE INDEX idx_memory_edges_to ON memory_edges(to_id);
```

**Edge types and weight computation** (following Hindsight):

| Edge type | Weight formula | Created when |
|---|---|---|
| `entity` | 1.0 (binary) | Two memories share a canonical entity in `entities[]` |
| `temporal` | `exp(-Δt / σ_t)` where σ_t is configurable (default 24h) | Two memories have `occurred_at` within a threshold |
| `semantic` | Cosine similarity (if ≥ θ_s, default 0.8) | Computed at insert time against recent memories |
| `causal` | 1.0 (binary) | LLM extraction identifies cause-effect relationship |

**Edge maintenance:** Edges are created at insert time. When a new memory is
stored, the retain pipeline:
1. Computes entity edges by querying `memories` for overlapping `entities[]`
2. Computes temporal edges by querying memories with nearby `occurred_at`
3. Computes semantic edges by vector search against existing memories (top-k, threshold)
4. Causal edges are extracted by the LLM during narrative extraction

### 2.3 Graph Retrieval Channel

Add `SearchMode::Graph` and integrate it into the hybrid recall path:

```rust
pub enum SearchMode {
    Vector,
    FullText,
    Hybrid { alpha: f32 },
    /// NEW: Four-channel retrieval (vector + BM25 + graph + temporal)
    Full {
        /// Weight balance between channels (all fed into RRF)
        graph_hops: usize,   // Max traversal depth (default 2)
        decay: f32,           // Activation decay per hop (default 0.5)
        temporal_range: Option<(DateTime<Utc>, DateTime<Utc>)>,
    },
}
```

**Spreading activation algorithm** (graph channel):

```
fn graph_retrieve(seeds: Vec<MemoryId>, hops: usize, decay: f32) -> Vec<ScoredKey>:
    frontier = {seed: 1.0 for seed in seeds}
    visited = {}
    for hop in 0..hops:
        next_frontier = {}
        for (node, activation) in frontier:
            if node in visited: continue
            visited[node] = activation
            for (neighbor, edge_weight) in edges(node):
                new_activation = activation * decay * edge_weight
                if new_activation > threshold:
                    next_frontier[neighbor] = max(next_frontier[neighbor], new_activation)
        frontier = next_frontier
    return visited sorted by activation
```

**Integration into `HybridStore::recall()`:**

When `SearchMode::Full` is used:
1. Run vector search → ranked list 1
2. Run BM25 search → ranked list 2
3. Seed graph search with top-k from vector results → ranked list 3
4. Run temporal filter/rank (if range provided) → ranked list 4
5. Fuse all lists via RRF (existing `fusion.rs`)
6. Return top-k

### 2.4 Temporal Retrieval Channel

The paper's temporal retrieval is more sophisticated than a simple recency
filter. It uses a **hybrid temporal parser** to normalize query expressions
into date ranges, then **interval intersection** to find matching memories,
scored by **temporal proximity**. Additionally, the temporal window
**constrains graph traversal** — spreading activation in Section 2.3 is
restricted to memories within the resolved time range.

#### 2.4.1 Hybrid Temporal Parser

A two-stage parser converts natural language time expressions into concrete
date ranges `[τ_start, τ_end]`:

**Stage 1: Rule-based analyzer** (handles the majority of queries at low latency)
- Uses date parsing libraries (e.g., `chrono`'s NaiveDate parsing, or a port of
  Python's `dateutil`/`dateparser`) to normalize explicit and relative expressions
- Handles: "yesterday", "last week", "in June 2024", "before March 5",
  "three days ago", "last weekend", "Q3 2025"
- Produces `Option<(DateTime<Utc>, DateTime<Utc>)>` — None if no temporal
  expression detected

**Stage 2: Seq2seq fallback** (for expressions the rule-based parser cannot resolve)
- Hindsight uses `google/flan-t5-small` (77M params) to convert remaining
  temporal expressions into date ranges
- Our equivalent: a lightweight LLM call via Ollama (e.g., qwen3.5:0.6b) with
  a constrained output format: `{"start": "2026-03-01", "end": "2026-03-07"}`
- Only invoked when rule-based parsing returns None but temporal keywords are
  detected in the query (heuristic: presence of time-related words like
  "when", "before", "after", "during", "last", "recent", month/day names)

```rust
pub struct TemporalRange {
    pub start: DateTime<Utc>,
    pub end: DateTime<Utc>,
}

pub fn parse_temporal(query: &str) -> Option<TemporalRange> {
    // Stage 1: rule-based
    if let Some(range) = rule_based_parse(query) {
        return Some(range);
    }
    // Stage 2: seq2seq fallback (only if temporal keywords detected)
    if has_temporal_keywords(query) {
        return llm_parse_temporal(query);
    }
    None
}
```

#### 2.4.2 Interval Intersection Matching

Once we have a query range `[τ_start, τ_end]`, we retrieve all memories whose
occurrence interval overlaps:

```
R_temp = { f ∈ V : [τ_s^f, τ_e^f] ∩ [τ_start, τ_end] ≠ ∅ }
```

In SQL (using the `occurred_at` / `occurred_end` columns from Section 2.1):

```sql
SELECT id, key, occurred_at, occurred_end FROM memories
WHERE org_id = $1
  AND occurred_at <= $2          -- fact started before query range ends
  AND COALESCE(occurred_end, occurred_at) >= $3  -- fact ended after query range starts
```

This is a proper interval overlap check, not just a point-in-range filter.
Memories with `occurred_end IS NULL` are treated as point events
(interval = `[occurred_at, occurred_at]`).

#### 2.4.3 Temporal Proximity Scoring

Each memory in `R_temp` is scored by how close its midpoint is to the query
range midpoint, normalized by the query range duration:

```
s_temp(Q, f) = 1 - |τ_mid^f - τ_mid^Q| / (Δτ / 2)
```

where:
- `τ_mid^f = (τ_s^f + τ_e^f) / 2` — midpoint of fact's occurrence interval
- `τ_mid^Q = (τ_start + τ_end) / 2` — midpoint of query range
- `Δτ = τ_end - τ_start` — query range duration

This produces scores in `[0, 1]` where 1.0 means the fact is centered
exactly in the query range. Facts at the edges of the range score ~0.0.

```rust
fn temporal_score(
    fact_start: DateTime<Utc>,
    fact_end: DateTime<Utc>,
    query_start: DateTime<Utc>,
    query_end: DateTime<Utc>,
) -> f32 {
    let fact_mid = fact_start + (fact_end - fact_start) / 2;
    let query_mid = query_start + (query_end - query_start) / 2;
    let delta = (query_end - query_start).num_seconds().max(1) as f32;
    let dist = (fact_mid - query_mid).num_seconds().abs() as f32;
    (1.0 - dist / (delta / 2.0)).max(0.0)
}
```

#### 2.4.4 Temporal-Constrained Graph Traversal

When a temporal range is resolved, graph traversal (Section 2.3) is
**restricted to `R_temp`** — the spreading activation algorithm only visits
nodes whose occurrence intervals overlap the query range. This prevents the
graph channel from pulling in temporally irrelevant memories that happen to
share entities with the seed set.

In `SearchMode::Full`:
1. Parse temporal range from query
2. If range found: filter candidate set to `R_temp` before graph traversal
3. Score `R_temp` members by temporal proximity → ranked list 4
4. Graph traversal seeds from vector top-k, but only visits nodes in `R_temp`
5. All four lists (vector, BM25, graph-within-temporal, temporal-proximity)
   enter RRF fusion

If no temporal range is detected, the temporal channel falls back to
**recency decay**:

```
s_recency(f) = exp(-(now - occurred_at) / σ_recency)
```

where `σ_recency` is configurable (default 7 days). This biases toward recent
memories without hard-filtering, and still enters RRF as a soft signal.

### 2.5 Opinion Evolution

Opinions are memories with `kind = 'opinion'` and a non-null `confidence` score.

**Reinforcement rules** (from Hindsight CARA):

```rust
const OPINION_ALPHA: f32 = 0.1;

fn update_confidence(current: f32, assessment: OpinionAssessment) -> f32 {
    match assessment {
        Reinforce   => (current + OPINION_ALPHA).min(1.0),
        Weaken      => (current - OPINION_ALPHA).max(0.0),
        Contradict  => (current - 2.0 * OPINION_ALPHA).max(0.0),
        Neutral     => current,
    }
}
```

**When updates happen:** During `retain()`, if a newly extracted fact relates
to an existing opinion (detected via entity overlap + semantic similarity),
the LLM assesses whether it reinforces, weakens, or contradicts, and the
confidence is updated in-place.

**Retrieval integration:** Opinion memories below a configurable confidence
threshold (default 0.3) are excluded from recall results unless explicitly
requested.

### 2.6 Narrative Fact Extraction (Retain Pipeline)

The core write-path change. Instead of storing raw values, conversations pass
through an LLM extraction step:

```
retain(transcript: &[ConversationTurn], ctx: &CallContext) -> Vec<MemoryRecord>
```

**Pipeline stages:**

```
┌─────────────┐     ┌──────────────┐     ┌─────────────────┐     ┌───────────┐
│ Conversation │ ──► │ LLM Extract  │ ──► │ Entity Resolve  │ ──► │ Store +   │
│ Transcript   │     │ 2-5 facts    │     │ (canonical IDs) │     │ Link      │
└─────────────┘     └──────────────┘     └─────────────────┘     └───────────┘
```

**Step 1: LLM Fact Extraction.** Prompt the LLM to produce structured facts:

```json
{
  "facts": [
    {
      "text": "Alice prefers dark-roast coffee over tea",
      "kind": "fact",
      "entities": ["alice"],
      "occurred_at": "2026-03-25T10:00:00Z",
      "causal_parent": null
    },
    {
      "text": "I recommended the Ethiopian blend based on Alice's preference",
      "kind": "experience",
      "entities": ["alice", "ethiopian_blend"],
      "occurred_at": "2026-03-25T10:01:00Z",
      "causal_parent": 0
    }
  ]
}
```

**Step 2: Entity Resolution.** For each entity mention:
1. Check `entities[]` column across existing memories for exact match
2. If no exact match, fuzzy match (Levenshtein distance ≤ 2 or Jaccard on tokens)
3. If no match, create new canonical entity

**Step 3: Store + Link.** For each extracted fact:
1. Store as a typed `MemoryRecord` with metadata columns populated
2. Compute and insert edges (entity, temporal, semantic, causal)

### 2.7 Cross-Encoder Reranker (Optional Enhancement)

Hindsight uses `ms-marco-MiniLM-L-6-v2` as a final reranker. We can add this
as an optional post-RRF step via Ollama or a dedicated ONNX runtime:

```rust
pub struct RecallOpts {
    pub top_k: usize,
    pub mode: SearchMode,
    pub scope: MemoryScope,
    pub rerank: bool,           // NEW: apply cross-encoder reranking
    pub token_budget: usize,    // NEW: max tokens in returned context
}
```

This is a lower-priority enhancement — RRF alone gets most of the benefit.

### 2.8 Plugin Integration

The plugin system (`gw-engine`, `gw-core/src/plugin.rs`) provides natural
extension points for Hindsight features. Rather than hard-wiring all new
memory capabilities into `gw-memory`, several components should be implemented
as plugins that compose with the existing `HybridStore` via lifecycle events
and the registration API.

#### 2.8.1 Plugin Architecture Mapping

| Hindsight component | Plugin extension point | Rationale |
|---|---|---|
| **Retain pipeline** (fact extraction) | `BeforeMemoryStore` event | Intercept raw values, extract typed facts + entities before persistence. Keeps extraction logic decoupled from store internals |
| **Graph retrieval channel** | `AfterMemoryRecall` event | Augment recall results with graph-traversed neighbors. Can modify the `EventPayload` to inject additional results into the returned set |
| **Temporal parser** | `AfterMemoryRecall` event + host function | Parse temporal expressions from queries; filter/re-score results by temporal proximity |
| **Opinion evolution** | `BeforeMemoryStore` event | Detect opinion-type memories, compare against existing opinions, apply confidence reinforcement rules before store commits |
| **Cross-encoder reranker** | `AfterMemoryRecall` event | Post-RRF reranking as a separate plugin — easy to enable/disable without touching core recall |
| **Entity resolution** | `BeforeMemoryStore` event | Resolve entity mentions to canonical IDs before persistence; maintain entity registry in `SharedState` |
| **Memory graph maintenance** | `BeforeMemoryStore` event | Compute and insert edges after entity resolution, before the store write completes |
| **Fact query host functions** | `register_host_fn()` | Expose `memory.facts_for_entity()`, `memory.temporal_query()`, `memory.graph_neighbors()` as Python-callable host functions |

#### 2.8.2 Plugin Design: `hindsight-retain`

A plugin implementing the Retain pipeline (Sections 2.1, 2.6):

```rust
pub struct HindsightRetainPlugin {
    llm_model: String,  // Model for fact extraction (can be small/fast)
}

impl Plugin for HindsightRetainPlugin {
    fn name(&self) -> &str { "hindsight-retain" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["memory:retain".into(), "host_fn:memory.facts".into()],
            requires: vec!["llm".into(), "memory".into()],
            priority: 50,  // After core memory, before recall augmentation
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        // Subscribe to BeforeMemoryStore — extract facts from raw values
        ctx.on(LifecycleEvent::BeforeMemoryStore, Arc::new(|payload| {
            // 1. Extract structured facts from payload.data (Memory { key, value })
            // 2. Classify into MemoryKind (fact/experience/opinion/observation)
            // 3. Extract entities via regex NER
            // 4. Resolve entities to canonical IDs (from SharedState)
            // 5. Compute temporal metadata (occurred_at, occurred_end)
            // 6. Modify payload with enriched memory record
            // 7. Compute and insert graph edges
            EventResult::Modified
        }));

        // Register host functions for Python agents
        ctx.register_host_fn("memory.facts_for_entity", Arc::new(|args, _kwargs| {
            // Query facts by entity name
            Ok(Value::Array(vec![]))
        }));

        ctx.register_host_fn("memory.entities", Arc::new(|_args, _kwargs| {
            // Return all known entities sorted by frequency
            Ok(Value::Array(vec![]))
        }));

        Ok(())
    }
}
```

#### 2.8.3 Plugin Design: `hindsight-recall`

A plugin augmenting recall with graph traversal, temporal filtering, and reranking (Sections 2.3, 2.4, 2.7):

```rust
pub struct HindsightRecallPlugin {
    graph_hops: usize,
    graph_decay: f32,
    rerank_model: Option<String>,
}

impl Plugin for HindsightRecallPlugin {
    fn name(&self) -> &str { "hindsight-recall" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "memory:graph_recall".into(),
                "memory:temporal_recall".into(),
                "host_fn:memory.graph_neighbors".into(),
                "host_fn:memory.temporal_query".into(),
            ],
            requires: vec!["memory".into()],
            priority: 60,  // After retain plugin
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        // Subscribe to AfterMemoryRecall — augment results
        ctx.on(LifecycleEvent::AfterMemoryRecall, Arc::new(|payload| {
            // 1. Parse temporal range from original query (hybrid parser)
            // 2. If temporal: filter results to R_temp, score by proximity
            // 3. Run spreading activation from top-k semantic seeds
            //    (constrained to R_temp if temporal range present)
            // 4. Fuse graph + temporal channels into existing RRF results
            // 5. Optional: cross-encoder reranking pass
            // 6. Modify payload with augmented result set
            EventResult::Modified
        }));

        // Host functions for direct agent access
        ctx.register_host_fn("memory.graph_neighbors", Arc::new(|args, kwargs| {
            // Spreading activation from a seed memory
            Ok(Value::Array(vec![]))
        }));

        ctx.register_host_fn("memory.temporal_query", Arc::new(|args, kwargs| {
            // Parse temporal expression, return matching memories
            Ok(Value::Array(vec![]))
        }));

        Ok(())
    }
}
```

#### 2.8.4 Plugin Design: `hindsight-opinions`

A plugin implementing opinion evolution (Section 2.5):

```rust
pub struct HindsightOpinionsPlugin;

impl Plugin for HindsightOpinionsPlugin {
    fn name(&self) -> &str { "hindsight-opinions" }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["memory:opinions".into()],
            requires: vec!["memory".into(), "llm".into(), "memory:retain".into()],
            priority: 55,  // After retain, before recall
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        // Subscribe to BeforeMemoryStore — detect opinion-type memories
        // and apply confidence reinforcement against existing opinions
        ctx.on(LifecycleEvent::BeforeMemoryStore, Arc::new(|payload| {
            // 1. Check if new memory is kind=Opinion or relates to existing opinions
            // 2. If relates: LLM assesses reinforce/weaken/contradict
            // 3. Update existing opinion confidence in-place
            // 4. Filter opinions below threshold from future recall
            EventResult::Modified
        }));

        Ok(())
    }
}
```

#### 2.8.5 Configuration

Each plugin gets its own TOML section under `[plugins.*]`:

```toml
[plugins.hindsight-retain]
enabled = true
extraction_model = "qwen3.5:1.7b"     # Small model for fact extraction
entity_resolution = "fuzzy"             # "exact", "fuzzy", or "llm"
max_facts_per_exchange = 5

[plugins.hindsight-recall]
enabled = true
graph_hops = 2
graph_decay = 0.5
temporal_parser = "rule-based"          # "rule-based", "llm-fallback", or "llm-only"
temporal_fallback_model = "qwen3.5:0.6b"
rerank = false                          # Enable cross-encoder reranking
rerank_model = "ms-marco-MiniLM-L-6-v2"
recency_sigma_days = 7

[plugins.hindsight-opinions]
enabled = true
alpha = 0.1                             # Confidence adjustment step
confidence_threshold = 0.3             # Below this, opinions excluded from recall
```

#### 2.8.6 What Stays in gw-memory vs. Plugins

The plugin system augments but does not replace `gw-memory`. Core storage
infrastructure remains in the crate; Hindsight-specific intelligence lives
in plugins:

| Component | Location | Why |
|---|---|---|
| `MemoryKind` enum | `gw-core` | Shared type needed by store + plugins |
| Schema migration (new columns) | `migrations/` | Core schema, not plugin-specific |
| `HybridStore` (Postgres + LanceDB + tantivy) | `gw-memory` | Unchanged — still the storage engine |
| RRF fusion (`fusion.rs`) | `gw-memory` | Core algorithm, plugins feed additional channels into it |
| `memory_edges` table | `migrations/` | Schema is core; edge computation is plugin logic |
| Fact extraction LLM prompts | `hindsight-retain` plugin | Decoupled — can iterate on prompts without touching storage |
| Entity resolution | `hindsight-retain` plugin | Strategy is configurable (exact/fuzzy/LLM) |
| Graph traversal | `hindsight-recall` plugin | Can be enabled/disabled per deployment |
| Temporal parser | `hindsight-recall` plugin | Parser strategy is configurable |
| Opinion reinforcement rules | `hindsight-opinions` plugin | Decoupled from storage; α is configurable |
| Cross-encoder reranker | `hindsight-recall` plugin | Optional; no dependency on specific model |

#### 2.8.7 Event Flow: Memory Store with Plugins

Complete event flow for a `store()` call with all Hindsight plugins enabled:

```
Agent calls memory.store(key, value)
    │
    ▼
BeforeMemoryStore event dispatched
    │
    ├─ [priority 50] hindsight-retain:
    │   1. LLM extracts 2-5 typed facts from value
    │   2. Regex NER extracts entities
    │   3. Entity resolution (fuzzy match against canonical registry)
    │   4. Sets kind, entities, occurred_at, occurred_end on payload
    │   5. Computes graph edges (entity, temporal, semantic)
    │   → EventResult::Modified (enriched payload)
    │
    ├─ [priority 55] hindsight-opinions:
    │   1. If kind=Opinion: check existing opinions for same entities
    │   2. LLM assesses reinforce/weaken/contradict
    │   3. Updates confidence on existing opinion
    │   → EventResult::Modified
    │
    ▼
HybridStore.store() executes with enriched record
    (Postgres + LanceDB + tantivy, writes new columns + edges)
```

#### 2.8.8 Event Flow: Memory Recall with Plugins

```
Agent calls memory.recall(query, opts)
    │
    ▼
HybridStore.recall() executes (vector + BM25 + RRF)
    │
    ▼
AfterMemoryRecall event dispatched (payload contains results)
    │
    ├─ [priority 60] hindsight-recall:
    │   1. Parse temporal range from query (hybrid parser)
    │   2. If temporal: interval intersection filter → R_temp
    │   3. Score R_temp by temporal proximity
    │   4. Spreading activation from top-k seeds
    │      (constrained to R_temp if temporal)
    │   5. RRF fuse graph + temporal channels with existing results
    │   6. Optional: cross-encoder reranking
    │   → EventResult::Modified (augmented results)
    │
    ▼
Augmented results returned to agent
```

---

## 3. Implementation Phases

The plugin system changes the implementation strategy. Core schema changes
still go into `gw-core` and `gw-memory`, but Hindsight-specific intelligence
(extraction, graph traversal, temporal parsing, opinion rules, reranking) is
implemented as three composable plugins. This lets deployments enable or
disable features via TOML config rather than compile-time flags.

**Dependency note:** Phases 1-2 are prerequisites for all plugins. The three
plugins (Phases 3-5) are independent of each other and can be implemented in
any order. Phase 3 depends on the `gw-engine` Phase 3 extension point
`register_memory_store()` being complete (currently deferred in the plugin
framework); until then, plugins use lifecycle events as the integration
mechanism.

### Phase 1: Core Schema + Types

**Effort:** Small — migration + enum, no plugin code yet
**Files:** `gw-core/src/lib.rs`, `migrations/`, `gw-memory/src/postgres.rs`

1. Add `MemoryKind` enum to `gw-core` (shared by store + all plugins)
2. Migration: add `kind`, `confidence`, `occurred_at`, `occurred_end`, `entities` columns to `memories`
3. Migration: create `memory_edges` table (schema is core; computation is plugin logic)
4. Update `PgMemoryStore` to read/write new columns
5. Update `MemoryRecord` to include typed metadata
6. Existing callers default to `kind = Fact`, `confidence = None`

**Testable outcome:** Memories carry type metadata; retrieval can filter by kind.
Edge table exists but is empty until the retain plugin populates it.

### Phase 2: Event Dispatch in HybridStore

**Effort:** Small — wire `BeforeMemoryStore` and `AfterMemoryRecall` events
**Files:** `gw-memory/src/hybrid.rs`, `gw-engine/src/engine.rs`

1. `HybridStore` receives an `Arc<EventDispatcher>` (optional, for backward compat)
2. `store()` dispatches `BeforeMemoryStore` before writing — plugins can enrich the payload
3. `recall()` dispatches `AfterMemoryRecall` after RRF — plugins can augment/re-score results
4. If no dispatcher (e.g., in tests or gw-bench), events are skipped silently

**Testable outcome:** Events fire during store/recall. Without plugins, behavior
is identical to current. With a no-op plugin subscribed, events propagate correctly.

### Phase 3: `hindsight-retain` Plugin

**Effort:** Medium — LLM prompt engineering, entity resolution, edge computation
**Files:** `crates/gw-engine/src/plugins/hindsight_retain.rs` (new)

1. Implement `Plugin` trait with `BeforeMemoryStore` handler
2. LLM fact extraction: prompt for 2-5 structured facts per exchange
3. Regex NER for entity extraction (reuses BrowseComp `_extract_entities` logic)
4. Entity resolution: exact match first, fuzzy (Levenshtein ≤ 2) fallback
5. Populate `kind`, `entities`, `occurred_at` on enriched payload
6. Compute graph edges (entity, temporal, semantic) and INSERT into `memory_edges`
7. Register `memory.facts_for_entity` and `memory.entities` host functions
8. Config: `extraction_model`, `entity_resolution`, `max_facts_per_exchange`

**Testable outcome:** Conversation transcripts produce typed, entity-tagged facts
with graph edges. Host functions expose entity queries to Python agents.

### Phase 4: `hindsight-recall` Plugin

**Effort:** Medium-large — graph traversal, temporal parsing, optional reranking
**Files:** `crates/gw-engine/src/plugins/hindsight_recall.rs` (new)

1. Implement `Plugin` trait with `AfterMemoryRecall` handler
2. Hybrid temporal parser (rule-based + LLM fallback, see Section 2.4.1)
3. Interval intersection filter on `occurred_at`/`occurred_end` (Section 2.4.2)
4. Temporal proximity scoring (Section 2.4.3)
5. Spreading activation graph traversal, constrained to `R_temp` if temporal (Section 2.4.4)
6. Fuse graph + temporal channels into existing RRF results
7. Optional cross-encoder reranking (config-gated, ONNX or Ollama)
8. Register `memory.graph_neighbors` and `memory.temporal_query` host functions
9. Config: `graph_hops`, `graph_decay`, `temporal_parser`, `rerank`, `recency_sigma_days`

**Testable outcome:** Multi-hop queries return graph-neighbor results. Temporal
queries ("what happened last week?") filter and rank by time. Graph traversal
respects temporal constraints.

### Phase 5: `hindsight-opinions` Plugin

**Effort:** Small — confidence update logic, LLM assessment
**Files:** `crates/gw-engine/src/plugins/hindsight_opinions.rs` (new)

1. Implement `Plugin` trait with `BeforeMemoryStore` handler (priority 55, after retain)
2. Detect opinion-type memories (kind=Opinion or entity overlap with existing opinions)
3. LLM assessment: reinforce/weaken/contradict/neutral
4. Apply confidence update rules (configurable α)
5. Threshold filtering: opinions below `confidence_threshold` excluded from `AfterMemoryRecall`
6. Config: `alpha`, `confidence_threshold`

**Requires:** `hindsight-retain` (declared via `requires: ["memory:retain"]` in manifest)

**Testable outcome:** Agent beliefs converge toward correct values over multiple
sessions as evidence accumulates. Low-confidence opinions are automatically
suppressed.

### Phase Summary

| Phase | Component | Type | Effort | Dependencies |
|---|---|---|---|---|
| 1 | Core schema + types | `gw-core`, `migrations/` | Small | None |
| 2 | Event dispatch in HybridStore | `gw-memory`, `gw-engine` | Small | Phase 1 |
| 3 | `hindsight-retain` plugin | `gw-engine/src/plugins/` | Medium | Phases 1-2 |
| 4 | `hindsight-recall` plugin | `gw-engine/src/plugins/` | Medium-large | Phases 1-2, optionally 3 |
| 5 | `hindsight-opinions` plugin | `gw-engine/src/plugins/` | Small | Phases 1-3 |

---

## 4. Evaluation Plan

### 4.1 Benchmarks

| Benchmark | What it tests | Target metric |
|---|---|---|
| **LongMemEval** | 500 questions across ~115K-1.5M tokens; info extraction, multi-session reasoning, temporal reasoning, knowledge updates, abstention | Overall accuracy (paper: 83.6% with 20B) |
| **LoCoMo** | 50 multi-session conversations, 304.9 avg turns, multimodal | Accuracy (paper: 85.67% with 120B) |
| **Internal multi-session** | Greatwheel conversation loop scenarios with memory queries | Accuracy on planted facts, temporal queries, opinion consistency |

### 4.2 Ablation Studies

Test each component independently against the baseline (`SearchMode::Hybrid`):

| Experiment | Baseline | +Change | Expected signal |
|---|---|---|---|
| A: Memory typing | Flat JSONB | +kind filtering in retrieval | Precision improvement on type-specific queries |
| B: Narrative extraction | Raw value storage | +LLM fact extraction | Higher recall, fewer irrelevant memories |
| C: Graph retrieval | Vector+BM25 RRF | +graph channel in RRF | Multi-hop accuracy improvement |
| D: Temporal channel | No time awareness | +temporal RRF channel | Temporal reasoning accuracy (paper: 85.7%) |
| E: Opinion tracking | Static beliefs | +confidence evolution | Belief consistency over 10+ sessions |
| F: Full pipeline | Flat + Hybrid | All of the above | Overall accuracy vs. paper numbers |

### 4.3 Internal Regression

Extend the existing `gw-loop` deterministic test suite with memory-focused scenarios:

- **S-MEM-01:** Store 3 facts across sessions, recall specific one by entity
- **S-MEM-02:** Store contradictory facts, verify knowledge update (most recent wins)
- **S-MEM-03:** Temporal query — "what did the agent learn yesterday?"
- **S-MEM-04:** Opinion formation and reinforcement across 5 turns
- **S-MEM-05:** Multi-hop — "what does Alice think about the project Bob started?"
- **S-MEM-06:** Entity resolution — "Al", "Alice", "alice_jones" resolve to same entity

---

## 5. Application to BrowseComp-Plus Benchmark

### 5.1 Current State

BrowseComp-Plus is an information retrieval benchmark where an rLM agent searches
a ~100K web document corpus to answer factual questions. Current best: **40% (12/30)**
with BM25 top-200 → ColBERT reranking → top-10.

**Failure breakdown:**
- **~60% retrieval misses** — gold document never found by any search
- **~30% extraction errors** — gold document retrieved but wrong answer extracted
- **~10% hedges** — model refuses to commit to an answer

The pipeline runs 12 rLM iterations: pre-search (5 queries + PRF) → iterative
search/read/extract loop → `FINAL("answer")`. The agent has access to `search()`,
`get_document()`, `llm_query()`, and `batch_llm_query()` as REPL tools.

86% of failures are retrieval-related. Dense vector search adds no lift (7-9/30
across 6 variants). ColBERT reranking is the only retrieval improvement that has
beaten the BM25 baseline.

### 5.2 Hindsight Concepts Applicable to BrowseComp

The BrowseComp setting differs from Hindsight's conversational memory scenario:
there is no multi-session history, no opinion evolution, and no temporal queries.
But several Hindsight ideas apply directly to the **intra-session fact management**
problem — how the agent organizes and reasons over evidence gathered during its
12-iteration research loop.

#### Concept 1: Structured Fact Accumulation (targets extraction errors) — IMPLEMENTED

**Problem:** The agent accumulates evidence as free-form Python variables
(`evidence`, `docs`, ad-hoc strings). By iteration 8, earlier findings are
scattered across variables with no structure, and the agent often forgets or
contradicts earlier extractions.

**Hindsight analog:** The Retain operation — narrative fact extraction producing
typed, entity-tagged facts.

**Proposed change:** Replace ad-hoc variable accumulation with a structured
`facts` registry maintained across iterations:

```python
# Injected into REPL namespace at session start
facts = FactRegistry()  # Stores typed facts with provenance

# Agent writes:
doc = get_document("doc_123")
extracted = llm_query(f"Extract facts relevant to: {question}\n\nDocument:\n{doc[:8000]}")

# Instead of: evidence.append(extracted)
# Agent writes:
facts.add(extracted, source="doc_123", kind="fact")  # Auto-extracts entities

# At any point:
facts.summary()        # Deduplicated, entity-grouped view
facts.for_entity("X")  # All facts mentioning entity X
facts.candidates()     # Candidate answers with supporting evidence count
```

**Implementation:** A Python class injected into the ouros REPL namespace via
`gw-bench`. The class:
1. Parses LLM extraction output into individual facts (split on newlines/bullets)
2. Extracts entity mentions (simple NER: capitalized phrases, quoted strings, numbers)
3. Deduplicates facts by entity + semantic overlap
4. Tracks provenance (which document, which iteration)
5. Generates a structured summary for the LLM's context window

**Expected impact:** Reduces extraction errors by keeping evidence organized.
The agent can call `facts.candidates()` to see which answers have the most
supporting evidence, rather than reconstructing this from scattered variables.

**Effort:** Small — Python class + prompt update, no Rust changes needed.

#### Concept 2: Entity-Bridged Search (targets retrieval misses)

**Problem:** 60% of failures are retrieval misses. The agent's search strategy is
sequential: it searches for terms from the query, reads documents, and sometimes
searches for entities found in those documents. But it has no systematic way to
follow entity chains across documents — the core strength of Hindsight's graph
retrieval.

**Hindsight analog:** Entity edges + spreading activation in the graph network.

**Proposed change:** Add an `entity_search()` tool that:
1. Takes an entity name discovered in a document
2. Searches the BM25 index for that entity
3. Extracts co-occurring entities from the top results
4. Searches for *those* entities (one hop)
5. Returns the union of results, ranked by entity co-occurrence with the query

```python
# Agent discovers "John Smith" in a document about the query topic
hits = entity_search("John Smith", hops=1)
# Returns documents mentioning John Smith AND documents mentioning
# entities that co-occur with John Smith (e.g., his company, his project)
```

**Implementation:** Python wrapper around existing `search()` in the REPL
namespace. No new index needed — it's BM25 search with automated entity
extraction and chained queries within a single tool call.

**Expected impact:** Medium — helps with multi-hop queries where the gold
document doesn't mention the query terms directly but is reachable through
an entity chain. This is exactly the failure mode where BM25 misses but
a human researcher would find the answer.

**Effort:** Small-medium — Python implementation + prompt update.

#### Concept 3: Confidence-Scored Candidates (targets hedges + extraction errors) — IMPLEMENTED

**Problem:** The agent sometimes has multiple candidate answers but picks the
wrong one, or hedges because it lacks confidence. There's no mechanism to track
how well-supported each candidate is.

**Hindsight analog:** Opinion confidence scores with reinforcement rules.

**Proposed change:** Extend the `FactRegistry` with candidate tracking:

```python
# Agent proposes a candidate answer
facts.propose("Marie Curie", confidence=0.6, evidence=["doc_42 mentions her award"])

# Later, finds confirming evidence
facts.reinforce("Marie Curie", evidence=["doc_89 confirms the award year"])
# confidence: 0.6 → 0.7

# Or contradicting evidence
facts.contradict("Marie Curie", reason="doc_91 says it was Pierre Curie")
# confidence: 0.6 → 0.4

# At FINAL() time, pick highest-confidence candidate
best = facts.best_candidate()  # Returns ("Pierre Curie", 0.7, [evidence...])
FINAL(best.answer)
```

**Implementation:** Extension of the FactRegistry class. The system prompt
instructs the agent to use `facts.propose()` / `facts.reinforce()` /
`facts.contradict()` as it gathers evidence. The `facts.best_candidate()`
method is used in the final iteration prompt and fallback extraction.

**Expected impact:** Reduces hedges (agent always has a ranked candidate to
submit) and extraction errors (confidence tracks evidence quality).

**Effort:** Small — Python class extension + prompt update.

#### Concept 4: Cross-Document Entity Graph for Retrieval (targets retrieval misses)

**Problem:** ColBERT reranking helps surface buried documents in BM25's top-200,
but if the gold document isn't in the top-200 at all, nothing helps. 60% of
failures are this case.

**Hindsight analog:** The full graph network with entity, semantic, and causal
edges — but applied to corpus documents rather than memory records.

**Proposed change:** Build a lightweight entity co-occurrence graph over the
100K corpus at index time:

```
Document "doc_42" mentions: {Marie Curie, Nobel Prize, 1903, Physics}
Document "doc_89" mentions: {Nobel Prize, Pierre Curie, Radioactivity}
→ Entity edge: doc_42 ↔ doc_89 (shared: "Nobel Prize")
→ Entity edge: implicit link via "Curie" family
```

At search time, after BM25 retrieves top-200:
1. Extract entities from top-10 BM25 results
2. Query the entity graph for documents sharing those entities
3. Add graph-retrieved documents to the ColBERT reranking pool
4. Rerank the expanded pool

**Implementation:**
- **Index time:** Extract named entities from all 100K docs (spaCy NER or
  simpler regex-based extraction). Store as a sparse entity-document matrix
  in tantivy (entity names as indexed terms) or a separate SQLite graph.
- **Search time:** New retrieval channel that expands the candidate set
  via entity co-occurrence before ColBERT reranking.

**Expected impact:** High — directly addresses the dominant failure mode.
If the gold document shares entities with any top-10 BM25 result (which is
likely for entity-heavy BrowseComp queries), graph expansion will pull it
into the reranking pool.

**Effort:** Large — requires corpus-wide NER + graph construction + new
retrieval path. But the NER can be approximate (regex for capitalized
phrases + known entity lists) since we only need it to create edges,
not for downstream accuracy.

#### Concept 5: Multi-Strategy Retrieval Fusion (targets retrieval misses)

**Problem:** We already use RRF to fuse BM25 + ColBERT. Hindsight shows that
adding more diverse retrieval channels to RRF improves recall.

**Hindsight analog:** Four-channel RRF (semantic + keyword + graph + temporal).

**Proposed change:** Add two new channels to the pre-search and in-loop
retrieval:

1. **Entity channel** — BM25 search on an entity-only index (extracted
   entities stored as a separate tantivy field). Catches documents where
   the entity appears in a non-prominent position that BM25 misses.

2. **Passage channel** — Instead of whole-document BM25, search on a
   passage-level index (documents split into 512-token chunks, each
   indexed separately). Catches answers buried in long documents where
   the document-level BM25 score is diluted.

Fusion: existing BM25 + ColBERT reranking + entity channel + passage channel,
all merged via RRF before presenting top-k to the agent.

**Expected impact:** Medium-high — more diverse retrieval channels mean the
union of results covers more of the document space. The passage-level index
is particularly promising for long documents where the answer appears in
a small section.

**Effort:** Medium — requires building entity and passage indices at index
time, but reuses existing tantivy infrastructure.

### 5.3 BrowseComp Experiment Plan

Ordered by expected impact / effort ratio:

| # | Experiment | Targets | Effort | Expected lift | Status |
|---|---|---|---|---|---|
| BC-1 | FactRegistry (structured accumulation) | Extraction errors (30%) | Small | +1-3 queries (reduce extraction errors by 30-50%) | **Implemented** |
| BC-2 | Confidence-scored candidates | Hedges (10%) + extraction | Small | +1-2 queries (eliminate hedges, better candidate selection) | **Implemented** |
| BC-3 | Entity-bridged search tool | Retrieval misses (60%) | Small-medium | +1-3 queries (helps multi-hop queries) | Planned |
| BC-4 | Passage-level BM25 index | Retrieval misses (60%) | Medium | +2-4 queries (catches buried answers in long docs) | Planned |
| BC-5 | Entity co-occurrence graph | Retrieval misses (60%) | Large | +2-5 queries (expands candidate pool for ColBERT) | Planned |
| BC-6 | Multi-strategy RRF (entity + passage channels) | Retrieval misses (60%) | Medium | +2-4 queries (diverse retrieval covers more docs) | Planned |

**Phase A (prompt-only, no index changes):** BC-1 + BC-2 — **IMPLEMENTED**
- FactRegistry Python class with entity extraction, fact storage, candidate confidence tracking
- System prompts updated in both `ollama_client.py` and `gw-bench/src/main.rs`
- FactRegistry-first fallback in gw-bench (tries `facts.best_candidate()` before LLM extraction)
- Iteration nudges reference `facts.candidates()` and `facts.best_candidate()`
- **Target: 43-47% (13-14/30)** — awaiting evaluation

**Phase B (new REPL tools):** BC-3
- Implement entity_search() as a REPL tool
- One-hop entity chain expansion via BM25
- **Target: 47-50% (14-15/30)**

**Phase C (index-time changes):** BC-4 + BC-5 + BC-6
- Build passage-level tantivy index (512-token chunks)
- Build entity co-occurrence graph (regex NER over 100K docs)
- Fuse all channels via RRF before ColBERT reranking
- **Target: 50-57% (15-17/30)**

### 5.4 FactRegistry Implementation

**File:** `bench/browsecomp/fact_registry.py` (standalone Python),
duplicated as `FACT_REGISTRY_BOOTSTRAP` constant in `gw-bench/src/main.rs`
(embedded string for ouros execution).

```python
class FactRegistry:
    """Structured evidence accumulator for research sessions."""

    # --- Fact management ---
    def add(self, text: str, source: str = "", kind: str = "fact") -> str:
        """Add extracted fact(s). Splits multi-line text into individual facts.
        Auto-extracts entities via regex NER. Returns confirmation string."""

    def for_entity(self, entity: str) -> list[str]:
        """All facts mentioning a specific entity (case-insensitive)."""

    def entities(self) -> list[str]:
        """All discovered entities sorted by mention frequency."""

    # --- Candidate management (Hindsight CARA-inspired) ---
    def propose(self, answer: str, confidence: float = 0.5, evidence: str = "") -> str:
        """Propose a candidate answer. If exists, updates to max confidence."""

    def reinforce(self, answer: str, evidence: str = "") -> str:
        """Increase confidence (+0.15, capped at 1.0). Auto-proposes if not found."""

    def contradict(self, answer: str, reason: str = "") -> str:
        """Decrease confidence (-0.25, floored at 0.0)."""

    def weaken(self, answer: str, reason: str = "") -> str:
        """Slightly decrease confidence (-0.10, floored at 0.0)."""

    def candidates(self) -> str:
        """Ranked candidate answers with confidence and evidence count."""

    def best_candidate(self) -> tuple[str, float] | None:
        """Highest-confidence candidate, or None."""

    # --- Summaries ---
    def summary(self) -> str:
        """Deduplicated, entity-grouped fact summary. Shows top 10 entities
        with up to 5 facts each, plus ungrouped facts and candidate ranking."""
```

**Confidence parameters** (tuned for BrowseComp):

| Operation | Delta | Rationale |
|---|---|---|
| `reinforce()` | +0.15 | Moderate boost — single confirming document is meaningful |
| `weaken()` | -0.10 | Mild — ambiguous evidence shouldn't kill a candidate |
| `contradict()` | -0.25 | Strong — direct contradiction should drop confidence fast |

**Entity extraction** (`_extract_entities()`):
- Quoted strings: `"Marie Curie"` → `Marie Curie`
- Capitalized phrases: `Nobel Prize in Physics` → `Nobel Prize in Physics`
- Proper nouns after lowercase context: `...awarded to Curie` → `Curie`
- Years: `1903`, `2024` (matches 1500-2029)
- Deduplicated by lowercase key, preserving original case

**Integration points:**
- **Python client** (`ollama_client.py`): `FactRegistry()` injected in `_build_repl_namespace()` return dict
- **Rust bench** (`gw-bench`): `FACT_REGISTRY_BOOTSTRAP` executed via `agent.execute()` after variable injection, before rLM loop
- **Fallback extraction**: gw-bench tries `facts.best_candidate()` before LLM fallback when max iterations reached
- **Iteration prompts**: halfway nudge says "check facts.candidates()"; last-chance prompt says "use facts.best_candidate()"; final prompt template includes `best = facts.best_candidate()`

### 5.5 Relationship to gw-memory Integration

The BrowseComp experiments (Section 5) and the gw-memory integration (Sections 2-4)
are complementary but independent:

| Aspect | gw-memory (Sections 2-4) | BrowseComp (Section 5) |
|---|---|---|
| Memory scope | Cross-session, persistent | Intra-session, ephemeral |
| Storage | Postgres + LanceDB + tantivy | Python dict in ouros REPL |
| Entity resolution | Canonical entity store | Simple regex NER |
| Graph | Persistent edge table | In-memory co-occurrence |
| Confidence | Persisted opinion evolution | Session-local candidate scoring |
| Retrieval | General-purpose hybrid recall | Corpus-specific multi-channel |

BrowseComp experiments serve as a **proving ground**: techniques that improve
benchmark accuracy (especially entity bridging and multi-strategy fusion) can
be promoted into the general `gw-memory` architecture. Conversely, `gw-memory`
features (like the graph retrieval channel) can be tested on BrowseComp before
deploying to production memory.

---

## 6. Open Questions

### 6.1 General

| # | Question | Options | Leaning |
|---|---|---|---|
| 1 | Where to run the extraction LLM? | (a) Same Ollama instance (b) Dedicated model (c) Async background job | (c) — extraction is not latency-sensitive |
| 2 | Edge computation at insert time vs. batch? | (a) Synchronous on insert (b) Background job (c) Hybrid | (c) — entity + causal edges on insert, semantic edges batched |
| 3 | Entity resolution strategy? | (a) String similarity only (b) LLM-powered (c) Embedding similarity | (a) for Phase 3, upgrade to (b) if accuracy is insufficient |
| 4 | Should graph edges cross scope boundaries? | (a) Never (b) Org-level edges visible to all (c) Configurable | (b) — facts are org-visible, opinions are agent-scoped |
| 5 | Opinion α value? | Fixed 0.1, configurable per-agent, or adaptive | Configurable per-agent via plugin config, default 0.1 |
| 6 | LongMemEval dataset — can we run it locally? | Need to check dataset availability and format | Investigate during Phase 1 |

### 6.2 Plugin-Specific

| # | Question | Options | Leaning |
|---|---|---|---|
| 7 | Sync vs. async event handlers? | (a) Sync only (current) (b) Async handlers (c) Sync handlers that spawn async tasks | (c) — retain plugin's LLM calls are async but handler signature is sync. Spawn via `tokio::spawn` from within handler, use `SharedState` to pass the runtime handle |
| 8 | Handler latency budget for `BeforeMemoryStore`? | (a) No limit (b) Configurable timeout (c) Hard 500ms cap | (b) — retain plugin with LLM extraction can take 1-2s; timeout should be per-plugin config with a default of 5s |
| 9 | How do recall-augmentation plugins access the original query? | (a) Embed in `EventData::Memory` (b) Thread-local (c) Add query field to `AfterMemoryRecall` data | (c) — `AfterMemoryRecall` needs the query text for temporal parsing and graph seeding. Requires adding a `query: String` field to the `EventData` variant |
| 10 | Plugin ordering for `BeforeMemoryStore`? | (a) Priority only (b) Explicit DAG (c) Capability-based ordering | (a) — manifest priority is sufficient. Retain at 50, opinions at 55 ensures correct order. DAG is over-engineering for 3 plugins |
| 11 | Should graph edges be writable from plugins directly or only via `BeforeMemoryStore` payload modification? | (a) Payload-only (b) Direct SQL from plugin (c) Dedicated `register_edges()` API | (a) for now — keep edge writes transactional with the memory write. Plugin enriches the payload; `HybridStore.store()` writes everything in one transaction |
| 12 | Plugin state persistence across restarts? | (a) Each plugin manages its own (b) Shared plugin_state table (c) `SharedState` survives restart via serde | (a) — entity registry can live in the `memories` table itself (entities column). No new persistence mechanism needed |

---

## 7. References

- Latimer et al. (2025). *Hindsight is 20/20: Building Agent Memory that Retains, Recalls, and Reflects.* arXiv:2512.12818
- Bae et al. (2024). *LongMemEval: Benchmarking Chat Assistants on Long-Term Interactive Memory.*
- Maharana et al. (2024). *LoCoMo: Long-Context Conversation Understanding with Multi-Session Multimodal Interactions.*
- Hong, Troynikov & Huber (2025). *The Dumb Zone: On Context Rot in Long-Running Agent Sessions.*
