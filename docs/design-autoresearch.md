# Design: Autoresearch Capabilities for Greatwheel

**Status:** Draft
**Date:** 2026-03-26
**Motivation:** [Automated Research Systems: A Survey](docs/autoresearch-survey.pdf) — internal survey of architectures and approaches across deep research, literature agents, full-pipeline science, and empirical optimisation systems

---

## 0. Executive Summary

Greatwheel's rLM + ouros architecture already implements the core execution
pattern shared by all major autoresearch systems: an iterative loop where an LLM
reasons, invokes tools, observes results, and repeats. The rLM variant is
arguably *more* flexible than standard tool-calling because agents write Python
code — they can compose tools, filter results inline, and maintain structured
state across iterations.

This document maps the four autoresearch categories identified in our survey to
Greatwheel's existing infrastructure, identifies gaps, and proposes a layered
implementation plan. The key insight is that most gaps are in *data access* and
*orchestration*, not in the core execution model.

**Three implementation layers:**

| Layer | Enables | Key deliverables | Dependencies |
|---|---|---|---|
| 1. External data access | Deep research, literature agents | Web search host function, academic API client, PDF ingestion | None (builds on existing host bridge) |
| 2. Multi-agent orchestration | Full-pipeline, multi-perspective | gw-bus implementation, episodes, persona decomposition | Layer 1 for useful sub-agents |
| 3. Generalised optimisation | Empirical optimisation at scale | Generic evaluator interface, Pareto ensemble, report synthesis | Layers 1-2 for full coverage |

---

## 1. Context: What We Have

### 1.1 Inventory of Existing Infrastructure

| Component | Crate | Status | Autoresearch role |
|---|---|---|---|
| rLM execution loop | gw-loop | Production (~1500 LOC) | Core agent reasoning — iterative plan-search-reason-verify |
| Hybrid memory | gw-memory | Production (1673 LOC) | BM25 + LanceDB vectors + Postgres, RRF fusion |
| Corpus search | gw-bench | Production (2949 LOC) | BM25 + optional ColBERT over fixed document corpora |
| LLM client | gw-llm | Production (~360 LOC) | Ollama + SGLang, multi-model, thinking support |
| ouros REPL | gw-runtime | Production (~476 LOC) | Sandboxed Python with persistent variables, snapshots |
| Session trees | gw-loop | Production | Branching, compaction, steering, Postgres persistence |
| FactRegistry | bench/ | Implemented (297 LOC) | Confidence-scored evidence accumulation (Hindsight Phase A) |
| GEPA optimisation | bench/ | In progress | ASI-driven prompt/config search over BrowseComp |
| Agent bus | gw-bus | Trait only | `call()` + `notify()` defined, zero implementation |
| Episodes | design doc | Designed | Compressed sub-agent summaries for context passing |
| Hindsight memory | design doc | Designed (Phase A done) | Typed memories, entity graphs, temporal retrieval |
| Scheduler | gw-scheduler | Trait only | Task queue + rate limiting defined, zero implementation |

### 1.2 How Our Architecture Maps to the Survey

The survey identified a common pattern across all autoresearch systems: an
iterative **ReAct loop** (plan → act → observe → reason → repeat). Every system
from OpenAI Deep Research to pi-autoresearch follows this pattern. The
differences are in:

1. **What the agent can act on** (web search, academic APIs, code execution, lab equipment)
2. **How the agent decomposes problems** (sub-queries, personas, evolution, dynamic tool selection)
3. **What persists between iterations** (context window, variables, archives, JSONL logs)
4. **How quality is controlled** (relevance scoring, novelty filtering, confidence estimation)

Greatwheel's rLM architecture is strong on (3) and (4) — Python variables
persist perfectly across iterations, and FactRegistry provides confidence-scored
evidence tracking. Our gaps are primarily in (1) — limited data access — and (2)
— single-agent decomposition only.

---

## 2. Layer 1: External Data Access

**Goal:** Give rLM agents access to the open web and academic literature, not
just pre-indexed corpora.

### 2.1 Web Search Host Function

**What:** A new host function `web_search(query, max_results=10)` that calls a
search API and returns structured results (title, URL, snippet).

**Why:** Every deep research system (OpenAI, Gemini, Perplexity, GPT-Researcher)
is built on iterative web search. Our rLM loop already handles the
reason-search-reason pattern; we just need to connect it to the web.

**Design:**

```python
# Agent-side API (available in ouros REPL namespace)
results = web_search("CRISPR gene therapy clinical trials 2025", max_results=10)
# Returns: [{"title": str, "url": str, "snippet": str, "source": str}, ...]

# Fetch full page content from a search result
content = web_fetch(url, max_chars=50000)
# Returns: {"url": str, "text": str, "title": str, "fetched_at": str}
```

**Host-side implementation:**

```rust
// New variant in the host function dispatcher (gw-bench or gw-runtime)
"web_search" => {
    let query = args[0].as_str()?;
    let max_results = args.get(1).and_then(|v| v.as_u64()).unwrap_or(10);
    let results = search_client.search(query, max_results as usize).await?;
    Ok(serde_json::to_value(results)?)
}

"web_fetch" => {
    let url = args[0].as_str()?;
    let max_chars = args.get(1).and_then(|v| v.as_u64()).unwrap_or(50000);
    let content = fetch_client.fetch(url, max_chars as usize).await?;
    Ok(serde_json::to_value(content)?)
}
```

**Search backend options (in preference order):**

| Backend | Cost | Quality | Notes |
|---|---|---|---|
| Brave Search API | $5/1K queries | Good | Privacy-focused, good snippet quality |
| Tavily Search API | $5/1K queries | Good | Purpose-built for AI agents, extracts clean text |
| Serper (Google) | $2/1K queries | Excellent | Google results via API |
| SearXNG (self-hosted) | Free | Variable | Meta-search, no API key needed |

Recommendation: Start with **Tavily** (best agent ergonomics — returns clean
extracted text, not raw HTML) and add SearXNG as a free fallback.

**Rate limiting:** Enforce via `ResourceLimits.max_search_calls` (already exists
in gw-core). Web searches are more expensive than corpus searches, so track
them separately: `max_web_search_calls` with a sensible default (50 per task).

**Files to modify:**
- `crates/gw-core/src/lib.rs` — add `max_web_search_calls` to ResourceLimits
- `crates/gw-runtime/src/lib.rs` — add `web_search` and `web_fetch` to host function dispatcher
- New file: `crates/gw-llm/src/web_search.rs` (or new crate `gw-web`) — search API client
- `crates/gw-bench/src/main.rs` — optional: wire web search into benchmark for hybrid corpus+web experiments

### 2.2 Academic API Client

**What:** Host functions for searching academic literature via Semantic Scholar,
CrossRef, and OpenAlex.

**Why:** Literature agents (PaperQA2, OpenScholar, Elicit) and full-pipeline
systems (AI Scientist's novelty checking) require structured access to the
academic corpus. These APIs are free and return rich metadata (authors, venue,
year, citation count, abstract, DOI).

**Design:**

```python
# Search for papers
papers = paper_search("attention mechanism transformer architecture", max_results=10)
# Returns: [{"paper_id": str, "title": str, "authors": [str], "year": int,
#            "venue": str, "citation_count": int, "abstract": str, "doi": str,
#            "url": str}, ...]

# Get full paper details (including references, citations)
details = paper_details(paper_id="649def34f8be52c8b66281af98ae884c09aef38b")
# Returns: full metadata + references list + citing papers list

# Search with filters
papers = paper_search("CRISPR delivery", year_range=(2023, 2026), min_citations=10)
```

**Backend priority:**
1. **Semantic Scholar** (primary) — 100 requests/sec unauthenticated, rich metadata, free
2. **OpenAlex** (fallback/enrichment) — fully open, good for citation graphs
3. **CrossRef** (DOI resolution) — canonical metadata source

**Implementation:** Model after PaperQA2's `clients/` module. Each backend is a
struct implementing a common `AcademicSearchClient` trait:

```rust
#[async_trait]
pub trait AcademicSearchClient: Send + Sync {
    async fn search(&self, query: &str, opts: PaperSearchOpts) -> Result<Vec<PaperMeta>>;
    async fn details(&self, id: &str) -> Result<PaperDetails>;
    async fn references(&self, id: &str) -> Result<Vec<PaperMeta>>;
    async fn citations(&self, id: &str) -> Result<Vec<PaperMeta>>;
}
```

**New crate:** `gw-academic` — keeps academic-specific logic (metadata clients,
retraction checking, citation graph traversal) separate from core memory.

### 2.3 PDF Ingestion

**What:** Ingest arbitrary PDF collections into the corpus search index (tantivy
BM25 + optional LanceDB vectors) so agents can search over user-supplied papers.

**Why:** PaperQA2's core value is answering questions from a user's PDF library.
With PDF ingestion, Greatwheel agents can search both the web and a local paper
corpus.

**Design:**

```python
# Agent-side: ingest a PDF into the searchable corpus
ingest_pdf("/path/to/paper.pdf", metadata={"authors": "Smith et al.", "year": 2024})

# Or ingest from URL
ingest_pdf_url("https://arxiv.org/pdf/2408.06292", metadata={...})

# Then search normally
results = search("attention mechanism scaling laws")
# Returns results from both pre-indexed corpus AND ingested PDFs
```

**Implementation approach:**
- Use `pdf-extract` or `lopdf` (Rust) for text extraction, or shell out to
  `pdftotext` (poppler) for reliability
- Chunk extracted text (1000 chars, 200 overlap — matching existing tantivy config)
- Index chunks into tantivy + optional LanceDB via existing `CorpusSearcher`
- Store metadata (title, authors, year, DOI) alongside chunks for citation generation

**Scope:** This is a convenience feature, not a core capability. Start with
`pdftotext` shelling and upgrade to native Rust extraction if needed.

---

## 3. Layer 2: Multi-Agent Orchestration

**Goal:** Enable multiple specialised agents to collaborate on research tasks,
with each agent's work accessible to others via episodes.

### 3.1 Implement gw-bus

**What:** A concrete implementation of the `AgentBus` trait for in-process
agent-to-agent communication.

**Why:** Every multi-agent autoresearch system (STORM's persona conversations, AI
Scientist's ideation→experimentation→review pipeline, Grok's 4-agent
architecture) requires agents to invoke and receive results from other agents.

**Design:**

The trait already exists:

```rust
pub trait AgentBus {
    async fn call(&self, ctx: &CallContext, agent: AgentId, task: Task)
        -> Result<serde_json::Value, ...>;
    async fn notify(&self, ctx: &CallContext, agent: AgentId, message: serde_json::Value)
        -> Result<(), ...>;
}
```

**Phase 1 — In-process implementation:**

```rust
pub struct InProcessBus {
    session_manager: Arc<SessionManager>,
}

impl AgentBus for InProcessBus {
    async fn call(&self, ctx: &CallContext, agent: AgentId, task: Task)
        -> Result<serde_json::Value, ...>
    {
        // 1. Look up AgentDef for target agent
        // 2. Create or retrieve ouros session for target agent
        // 3. Inject task as _task variable
        // 4. Run rLM loop to completion
        // 5. Extract result + generate episode summary
        // 6. Return result + episode to caller
    }
}
```

**Agent-side API:**

```python
# Synchronous call — blocks until sub-agent completes
result = agent.call("literature_reviewer",
    task={"query": "What are the latest CRISPR delivery mechanisms?"},
    context_episodes=[prior_episode])

# Fire-and-forget notification
agent.notify("monitor", {"event": "search_complete", "results_count": 47})

# Parallel dispatch (from design-episodes.md)
results = agent.call_parallel([
    ("domain_expert_1", {"perspective": "clinical", "query": q}),
    ("domain_expert_2", {"perspective": "regulatory", "query": q}),
    ("domain_expert_3", {"perspective": "economic", "query": q}),
])
```

**Recursion guard:** `max_call_depth` (default 3) from the episodes design. Track
depth in CallContext; reject calls that exceed the limit.

**Files to modify:**
- `crates/gw-bus/src/lib.rs` — add `InProcessBus` implementation
- `crates/gw-runtime/src/lib.rs` — add `agent.call()` and `agent.notify()` host functions
- `crates/gw-core/src/lib.rs` — add `call_depth` to CallContext

### 3.2 Implement Episodes

**What:** Compressed summaries of sub-agent work that flow as context between
agents. Follows the design in `docs/design-episodes.md`.

**Why:** STORM passes dialogue transcripts between persona agents. AI Scientist
passes experiment results to the writeup phase. PaperQA2 passes gathered evidence
to answer generation. All of these are forms of episodic context transfer.
Episodes are the general mechanism.

**Design (from existing design doc):**

```rust
pub struct Episode {
    pub id: EpisodeId,
    pub agent_id: AgentId,
    pub objective: String,        // What the agent was asked to do
    pub summary: String,          // LLM-generated summary of what happened
    pub artifacts: Vec<Artifact>, // Key outputs (answers, data, code)
    pub failures: Vec<String>,    // What didn't work (for avoiding repetition)
    pub token_cost: u64,          // Total tokens consumed
    pub duration_ms: u64,
}
```

**Episode generation:** When a sub-agent completes (via `agent.call()`), the bus:
1. Compresses the agent's session tree into a span summary
2. Calls a lightweight LLM (e.g. qwen3:4b) with: "Summarise this agent session
   into an episode: objective, key findings, artifacts, failures"
3. Stores the Episode in memory and returns it to the calling agent

**Episode injection:** When an agent receives `context_episodes`, they're
serialised and injected as `_prior_work` Python variable before the rLM loop
starts. The agent can read, filter, and build on prior findings.

**Connection to autoresearch patterns:**

| Pattern | How episodes enable it |
|---|---|
| STORM multi-perspective | Each persona agent produces an episode; orchestrator merges all episodes into an information table |
| AI Scientist pipeline | Ideation episode → experimentation episode → writeup episode → review episode, each building on the last |
| PaperQA2 evidence gathering | Each search-and-score cycle produces an episode; answer generation uses all episodes |
| pi-autoresearch ASI | Each experiment iteration's ASI is naturally an episode artifact |

### 3.3 Multi-Perspective Decomposition (STORM Pattern)

**What:** A decomposition agent that generates diverse research perspectives
before dispatching specialist sub-agents.

**Why:** STORM's most important architectural insight is that generating
sub-queries directly produces clustered, obvious questions. Generating *personas*
first ("Who would care about this topic? What would they ask?") produces diverse,
complementary coverage.

**Design:**

```
User query: "What are the implications of quantum computing for cryptography?"

Phase 1 — Persona generation (single LLM call):
  → Cryptographer (concerned about breaking RSA/ECC)
  → Post-quantum researcher (focused on lattice-based alternatives)
  → Policy analyst (national security implications)
  → Industry practitioner (migration timelines and costs)
  → Historian of cryptography (precedents from past transitions)

Phase 2 — Parallel sub-agent dispatch:
  For each persona:
    agent.call("research_interviewer", {
        "persona": persona_description,
        "topic": user_query,
        "max_turns": 3
    })
  → 5 episodes, each with persona-specific findings

Phase 3 — Synthesis:
  Merge all episodes into information table
  Generate outline from merged findings
  Write report section-by-section with citations
```

**Implementation as an agent type:**

```python
# system_prompt for the decomposition orchestrator
ORCHESTRATOR_PROMPT = """
You are a research orchestrator. Given a topic, you:
1. Generate 3-5 diverse expert personas who would investigate this differently
2. For each persona, dispatch a research sub-agent with that perspective
3. Merge findings from all sub-agents
4. Produce a comprehensive report covering all perspectives

Use agent.call_parallel() to dispatch persona agents concurrently.
Use _prior_work to access their episodes.
Use facts.add() to accumulate and deduplicate findings.
"""
```

This requires Layer 2 (bus + episodes) but no new infrastructure beyond that.
The orchestrator is just an agent with a specific system prompt that calls
other agents.

---

## 4. Layer 3: Generalised Optimisation

**Goal:** Extend the GEPA/ASI pattern from BrowseComp-specific to a general
empirical optimisation framework.

### 4.1 Generic Evaluator Interface

**What:** Abstract the GEPA evaluator beyond BrowseComp so any task with a
measurable metric can be optimised.

**Why:** pi-autoresearch works on any codebase with a benchmark script.
ShinkaEvolve works on any program with a fitness function. Our current GEPA
integration is BrowseComp-specific (`gepa_evaluator.py` wraps `gw-bench`). The
pattern is general; the implementation should be too.

**Design:**

```python
class Evaluator(Protocol):
    """Universal evaluator interface for GEPA optimisation."""

    def evaluate(self, candidate: dict, example: dict) -> tuple[float, dict]:
        """
        Args:
            candidate: The artifact being optimised (prompt, config, code, etc.)
            example: A single evaluation instance from the task distribution

        Returns:
            score: 0.0-1.0 scalar metric
            asi: Actionable Side Information dict — structured diagnostics
                 explaining *why* this score, not just *what* it was
        """
        ...
```

**Evaluator registry:**

```python
EVALUATORS = {
    "browsecomp": BrowseCompEvaluator,     # existing
    "conversation_loop": ConvLoopEvaluator, # gw-loop capability eval
    "report_quality": ReportQualityEvaluator, # LLM-judge on deep research output
    "custom": CustomScriptEvaluator,        # user provides eval.sh
}
```

**CustomScriptEvaluator** follows pi-autoresearch's pattern: the user provides a
benchmark script that outputs `METRIC name=value` lines. The evaluator parses
these and constructs ASI from stdout patterns:

```python
class CustomScriptEvaluator:
    def __init__(self, script_path: str, metric_name: str, best_direction: str):
        self.script = script_path
        self.metric = metric_name
        self.direction = best_direction  # "higher" or "lower"

    def evaluate(self, candidate: dict, example: dict) -> tuple[float, dict]:
        # 1. Write candidate artifact to temp file
        # 2. Run script with artifact path as argument
        # 3. Parse METRIC lines from stdout
        # 4. Construct ASI from script output (failures, warnings, timing)
        # 5. Return (normalised_score, asi_dict)
```

### 4.2 Pareto Ensemble with Query Routing

**What:** Maintain a frontier of complementary configurations and route incoming
queries to the best one. Phase 5 of `design-optimize-anything.md`.

**Why:** No single prompt/config is optimal for all query types. ShinkaEvolve's
archive maintains population diversity for the same reason. GEPA's Pareto search
already produces diverse candidates; the missing piece is runtime routing.

**Design:**

```
GEPA optimisation produces frontier:
  Config A: 80% on entity queries, 20% on temporal queries
  Config B: 30% on entity queries, 70% on temporal queries
  Config C: 60% on multi-hop queries, 40% on everything else

At runtime:
  1. Classify incoming query → type (entity / temporal / multi-hop / ...)
  2. Select frontier config with highest expected accuracy for that type
  3. Run rLM loop with selected config
  4. (Optional) Run top-2 configs, take higher-confidence answer
```

**Query classifier:** Lightweight LLM call (or regex heuristic) that maps a query
to a cluster. The clusters emerge from GEPA's per-query ASI — queries that fail
for the same reasons form natural clusters.

**Implementation:**

```rust
pub struct ParetoEnsemble {
    pub frontier: Vec<FrontierConfig>,      // Pareto-optimal configs
    pub cluster_model: QueryClassifier,     // Maps query → cluster
    pub cluster_config_map: HashMap<ClusterId, Vec<(ConfigId, f64)>>, // cluster → ranked configs
}

impl ParetoEnsemble {
    pub async fn select_config(&self, query: &str) -> &FrontierConfig {
        let cluster = self.cluster_model.classify(query).await;
        let (config_id, _score) = &self.cluster_config_map[&cluster][0];
        &self.frontier[*config_id]
    }
}
```

**Builds on:** GEPA Phase 4 (optimisation runs) + per-query ASI infrastructure
(already implemented). The frontier is a natural output of GEPA's Pareto search;
we just need to persist it and add a router.

### 4.3 Report Synthesis Agent

**What:** An agent that takes accumulated evidence (FactRegistry contents,
episodes, search results) and generates a structured, cited report.

**Why:** The gap between our current `FINAL(short_answer)` and a deep research
system's multi-page report is a synthesis layer. GPT-Researcher has a
`ReportGenerator`; STORM has section-by-section article writing; OpenAI Deep
Research produces structured reports. We need an equivalent.

**Design:**

```python
# System prompt for the report synthesis agent
REPORT_AGENT_PROMPT = """
You are a report synthesis agent. You receive:
- _prior_work: episodes from research sub-agents
- facts: a FactRegistry with accumulated evidence and confidence scores

Your job:
1. Organise findings into a logical outline (3-7 sections)
2. For each section, synthesise relevant facts with inline citations [source_id]
3. Flag areas of disagreement or low confidence
4. Produce a final report in markdown with a references section

Quality rules:
- Every factual claim must cite a source from the FactRegistry
- Acknowledge uncertainty — don't overstate weak evidence
- Prefer facts with high confidence scores
- Note when sources disagree
"""
```

**Citation tracking:** Extend FactRegistry to emit citations:

```python
# Current
facts.add("CRISPR delivery via lipid nanoparticles is effective", source="doc_42")

# Extended
facts.add("CRISPR delivery via lipid nanoparticles is effective",
          source="doc_42",
          citation={"authors": "Smith et al.", "year": 2024,
                    "title": "Advances in CRISPR Delivery"})

# At synthesis time
report_agent can call facts.with_citations() to get text + citation metadata
```

---

## 5. Composite Autoresearch Modes

With Layers 1-3 implemented, the system supports four distinct autoresearch modes
as compositions of the same primitives:

### 5.1 Deep Research Mode

```
User query
  → Orchestrator agent (STORM-style persona generation)
    → 3-5 parallel research sub-agents (web_search + web_fetch + facts)
      ← Episodes with per-perspective findings
  → Report synthesis agent (facts + episodes → structured report)
  ← Multi-page cited report
```

**Uses:** web_search (Layer 1), bus + episodes (Layer 2), report synthesis (Layer 3)

**Compared to GPT-Researcher:** Our version uses multi-perspective decomposition
instead of flat sub-queries, and FactRegistry confidence scoring instead of
cosine-similarity filtering.

### 5.2 Literature Review Mode

```
Research question
  → Literature search agent (paper_search + paper_details)
    → Ingest top-k papers (ingest_pdf)
    → For each paper: extract key claims into FactRegistry
    → Score relevance per claim (0-10, like PaperQA2)
  → Report synthesis agent (structured review with citations)
  ← Literature review with inline citations and consensus assessment
```

**Uses:** academic API + PDF ingestion (Layer 1), FactRegistry (existing),
report synthesis (Layer 3)

**Compared to PaperQA2:** Our version uses the rLM loop for dynamic tool
selection (search more vs. synthesise) and FactRegistry for confidence-scored
evidence, matching PaperQA2's RL-style environment approach.

### 5.3 Full-Pipeline Science Mode

```
Domain + seed ideas
  → Ideation agent (generate ideas, self-reflect)
    → Novelty check agent (paper_search loop, like AI Scientist)
      ← Novel idea episode
  → Experimentation agent (write + run code in ouros REPL)
    ← Experiment results episode
  → Writeup agent (LaTeX generation in ouros, like AI Scientist)
    → Citation agent (paper_search → BibTeX)
      ← Paper episode
  → Review agent (simulated peer review)
    ← Review scores + feedback episode
  → (Optional) Revision agent (revise based on review)
  ← Complete paper with code, results, review
```

**Uses:** All three layers. Requires academic API (Layer 1), full bus +
episodes pipeline (Layer 2), and the evaluator interface for review scoring
(Layer 3).

**Compared to AI Scientist:** Our version uses native ouros REPL instead of
Aider (simpler, no external tool dependency), episodes for phase-to-phase
context transfer (more structured than flat prompt injection), and FactRegistry
for citation tracking during writeup.

### 5.4 Empirical Optimisation Mode

```
Objective + benchmark script
  → GEPA optimisation loop:
      → Evaluate candidate (generic evaluator)
      ← Score + ASI (structured failure diagnostics)
      → Reflection LLM proposes variant based on ASI
      → Repeat for N iterations
  → Pareto frontier of complementary configs
  → Query router selects best config per query type
  ← Optimised system + routing policy
```

**Uses:** Generic evaluator (Layer 3), Pareto ensemble (Layer 3). Layers 1-2
optional but useful (e.g. web search for relevant papers during reflection).

**Compared to pi-autoresearch:** Our version uses GEPA's Pareto search instead
of sequential keep/discard, and ASI is already the same concept. We lack
pi-autoresearch's MAD-based confidence scoring and context-aware auto-resume,
which are worth adopting.

---

## 6. Implementation Plan

### 6.1 Phase 1: Web Search + Academic APIs (Layer 1)

**Estimated scope:** ~1 week

| Task | Detail |
|---|---|
| Add `web_search` host function | Tavily client in `gw-web` crate, dispatcher in gw-runtime |
| Add `web_fetch` host function | HTTP fetch with HTML-to-text extraction, char limit |
| Add `paper_search` host function | Semantic Scholar client in `gw-academic` crate |
| Add `paper_details` host function | S2 paper details + references/citations |
| Rate limiting | `max_web_search_calls` and `max_paper_search_calls` in ResourceLimits |
| Integration test | rLM agent that answers a factual question using web search |

**Deliverable:** An rLM agent that can search both a fixed corpus and the open
web, with academic paper search. This alone makes Greatwheel a functional deep
research system for single-agent use.

### 6.2 Phase 2: Bus + Episodes (Layer 2)

**Estimated scope:** ~2 weeks

| Task | Detail |
|---|---|
| Implement `InProcessBus` | In-process agent dispatch via SessionManager |
| Implement episode generation | LLM-summarised session compression on agent completion |
| Add `agent.call()` host function | Synchronous sub-agent invocation with episode return |
| Add `agent.call_parallel()` | Concurrent dispatch via tokio::join |
| Recursion guard | `max_call_depth` tracking in CallContext |
| Episode injection | `_prior_work` variable injection before rLM loop |
| Multi-perspective orchestrator | STORM-style persona generation agent template |
| Integration test | Orchestrator spawns 3 research agents, merges episodes |

**Deliverable:** Multi-agent research pipelines. An orchestrator agent can
dispatch specialist sub-agents and synthesise their findings.

### 6.3 Phase 3: Generalised Optimisation (Layer 3)

**Estimated scope:** ~2 weeks

| Task | Detail |
|---|---|
| Generic `Evaluator` protocol | Abstract evaluator with ASI support |
| `CustomScriptEvaluator` | pi-autoresearch-style script-based evaluation |
| Pareto ensemble persistence | Store frontier configs + cluster map |
| Query router | Classify → select config from frontier |
| Report synthesis agent | FactRegistry → structured cited report |
| Citation tracking in FactRegistry | Structured citation metadata on `facts.add()` |
| MAD confidence scoring | Adopt from pi-autoresearch for multi-run noise filtering |

**Deliverable:** General-purpose empirical optimisation + report generation.

### 6.4 Phase 4: Integration + Hardening

**Estimated scope:** ~1 week

| Task | Detail |
|---|---|
| Composite mode configs | Agent definitions for deep research, literature review, full-pipeline, empirical optimisation |
| Server wiring | Expose autoresearch modes via gw-server HTTP API |
| Tracing integration | OTel spans for web search, paper search, agent calls, episodes |
| Documentation | Usage guide for each autoresearch mode |

---

## 7. What Not to Build

| Temptation | Why not |
|---|---|
| **Own web crawler** | Search APIs (Tavily, Brave) are cheap, reliable, and return clean text. Crawling is an ops burden. |
| **External agent framework** (LangChain, AutoGen, CrewAI) | The rLM + ouros + session tree architecture *is* an agent framework. Adding another layer creates impedance mismatch and debugging opacity. |
| **Dense vector search for BrowseComp** | Six variants tested, zero lift. Entity-bridged search (Hindsight Phase B) is the correct path for the retrieval ceiling. |
| **Separate "deep research" product** | Deep research is a *mode* of the existing system (orchestrator prompt + web search tools + report synthesis), not a separate system. |
| **PDF rendering / LaTeX compilation in-agent** | Shell out to `pdflatex` or `pdftotext` from the host side. The agent writes LaTeX; the host compiles it. |

---

## 8. Risk Assessment

| Risk | Mitigation |
|---|---|
| **Search API costs at scale** | Rate limiting in ResourceLimits; SearXNG as free fallback; per-org cost tracking |
| **Sub-agent recursion explosion** | `max_call_depth` guard; per-task token budget; timeout per sub-agent |
| **Episode quality** | Use structured extraction prompts; include raw artifacts alongside LLM summary; allow agents to reject low-quality episodes |
| **Report hallucination** | Require every claim to cite a FactRegistry source; flag low-confidence claims; LLM-judge review pass |
| **GEPA overfitting to BrowseComp** | Generic evaluator interface forces abstraction; train/val splits; cross-benchmark validation |

---

## 9. Success Metrics

**Layer 1 (data access):**
- rLM agent with web search matches GPT-Researcher report quality on 10 test queries (human eval)
- Academic search enables novelty checking: given a known paper idea, system correctly identifies prior work >80% of the time

**Layer 2 (orchestration):**
- Multi-perspective decomposition produces ≥2x more diverse search queries than single-agent decomposition (measured by unique query terms)
- 3-agent research pipeline completes without human intervention on 10 test topics

**Layer 3 (optimisation):**
- Generic evaluator works on 3 distinct benchmarks (BrowseComp, conversation loop, custom script)
- Pareto ensemble improves BrowseComp accuracy by ≥3 queries over best single config
- Report synthesis agent produces cited reports where ≥90% of citations are verifiable

---

## 10. Relationship to Existing Design Documents

This document is a *composition layer* that connects existing designs:

| Existing design | Section here | Relationship |
|---|---|---|
| `design-optimize-anything.md` | §4.1, §4.2 | Generalises GEPA beyond BrowseComp |
| `design-hindsight-memory.md` | §2.3, §4.3 | FactRegistry (Phase A) powers evidence tracking; Phases B-F enable graph-based retrieval for deep research |
| `design-episodes.md` | §3.2, §3.3 | Episodes are the inter-agent context transfer mechanism |
| `design-conversation-loop.md` | §3.1 | Session trees provide the branching/compaction substrate for multi-agent work |
| `design-sglang-backend.md` | §6 (implicit) | Fast inference is prerequisite for running multi-agent pipelines in reasonable time |
| `design-browsecomp-conversation-loop.md` | §4.1 | BrowseComp evaluator becomes one instance of the generic evaluator |

No existing design needs to change. This document describes how they compose into
autoresearch capabilities.
