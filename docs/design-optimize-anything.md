# Design: GEPA optimize_anything for BrowseComp-Plus

**Status:** Draft
**Date:** 2026-03-19
**Motivation:** [optimize_anything: A Universal API for Optimizing any Text Parameter](https://gepa-ai.github.io/gepa/blog/2026/02/18/introducing-optimize-anything/) — GEPA

---

## 1. Context

Our BrowseComp-Plus benchmark evaluates the rLM agent architecture on 30 factoid
retrieval queries against a 100K web document corpus. After 40+ manually-iterated
experiment runs, we've reached 36.7% accuracy (11/30) with a reproducibility range
of 9-11/30 (mean ~10). The historical union of correct answers across all configs
is 40% (12/30).

Manual ablation identified the key levers — boosted BM25, PRF, doc-grounded
prompts, iteration count, retrieval breadth — and ruled out dead ends (vector
search, HyDE, majority voting, 16 turns). But the search has been one-variable-
at-a-time and prompt changes have been guided by intuition, not structured
diagnostics.

GEPA's `optimize_anything` API offers three ideas directly applicable here:

1. **Actionable Side Information (ASI):** structured per-query diagnostics that
   tell the optimizer *why* a candidate failed, not just *that* it failed
2. **Pareto-efficient search:** maintaining a frontier of candidates with
   complementary strengths rather than optimizing a single average score
3. **Multi-task mode:** treating query clusters as related tasks with cross-
   transfer of insights

This document describes how to integrate these ideas into the BrowseComp-Plus
experiment pipeline.

### 1.1 Where We Sit

| Property | GEPA case studies | BrowseComp-Plus (current) | BrowseComp-Plus (proposed) |
|---|---|---|---|
| Artifact | prompt / code / agent arch | system prompt + hardcoded params | prompt template + config struct |
| Evaluator | per-task scoring function | `evaluate_run.py` → scalar X/30 | per-query scoring + ASI |
| Search | GEPA Pareto engine | manual ablation (1 var at a time) | GEPA or equivalent Pareto search |
| Diagnostics | ASI (structured feedback) | results.tsv (scalar accuracy) | per-query failure classification |
| Ensemble | Pareto frontier = ensemble | planned (A4) but not built | frontier configs with query routing |

### 1.2 Most Relevant GEPA Case Studies

**Agent architecture evolution (ARC-AGI).** GEPA optimized an entire agent system
— code, sub-agent architecture, control flow, prompts — as a single text artifact.
Starting from a 10-line naive agent (32.5%), it evolved a 300+ line system reaching
89.5%. Our rLM system prompt + iteration logic is a similar "agent architecture as
text" artifact.

**Coding agent skills.** GEPA optimized natural-language instructions (skills) for
Claude Code, boosting resolve rates from 24% to 93% on one repo and cutting
resolution time by 47%. Our system prompt serves the same role — instructions that
shape agent behavior on a task distribution.

**Cloud scheduling (Can't Be Late).** Demonstrates generalization mode with
train/val splits on a distribution of scenarios. Our 30 queries could be split into
train (20) and val (10) to test generalization of prompt/config changes.

---

## 2. What We'd Optimize

### 2.1 Artifact 1: System Prompt + Iteration Prompts (highest ROI)

The primary optimization target is ~2KB of hand-tuned text across three functions:

- `SYSTEM_PROMPT` (main.rs:572-645) — tools, workflow, strategy, rules, examples
- `iteration_prompt()` (main.rs:647-688) — per-iteration nudges and hints
- `final_prompt()` (main.rs:690-704) — forced submission prompt

These have been the biggest lever historically. The progression from naive prompts
(3.3%) through DSPy-inspired rewrites (26.7%) to doc-grounded prompts (36.7%)
accounts for most of our accuracy gains.

The prompt has both structural elements (workflow steps, tool descriptions) and
stylistic elements (directive phrasing, emphasis, examples). Manual iteration
can't efficiently explore interactions between these — e.g., does a stronger
VERIFY directive help entity queries but hurt multi-hop ones?

**Optimization approach:** Generalization mode. Train on 20 queries, validate on
10 held-out queries. The prompt template is the artifact; GEPA proposes variants
guided by per-query ASI.

### 2.2 Artifact 2: Pipeline Configuration (joint parameter search)

The second target is the set of ~15 numeric parameters currently hardcoded across
main.rs and corpus.rs:

```
n_presearch_queries:    5       # extract_prompt → .take(5)
presearch_k:            5       # min(k, 5) per sub-query
prf_top_n:              5       # snippets to analyze for PRF terms
prf_term_count:         5       # distinctive terms to extract
prf_min_word_len:       4       # minimum term length
round2_keep_max:        15      # max docs to keep from round 1
round2_new_queries:     3       # new queries in refinement round
bm25_phrase_boost:      4.0     # exact phrase match weight
bm25_slop_boost:        2.0     # phrase with slop-2 weight
bm25_and_boost:         1.5     # all-terms-AND weight
bm25_or_boost:          1.0     # individual-terms-OR weight
max_iterations:         12      # rLM loop iterations
snippet_preview_len:    300     # chars shown per context snippet
iteration_nudge_start:  4       # when mid-loop hints kick in
query_timeout_secs:     180     # per-query time limit
```

Our ablations only tested these one at a time (k=5 vs k=10, 12 turns vs 16 turns).
Interactions are unexplored — e.g., more pre-search queries might work if paired
with a smaller snippet preview length (less context to overwhelm the model).

**Optimization approach:** This could be treated as a second text artifact
(serialized config), optimized jointly with the prompt. Or optimized separately
using GEPA's single-task mode on the full 30-query set, since config parameters
are less likely to overfit than prompt text.

### 2.3 Artifact 3: Pareto Ensemble (replaces planned A4 experiment)

Rather than hand-picking 2-3 diverse configs for our planned A4 ensemble
experiment, the Pareto frontier from optimization *is* the ensemble. Each frontier
point excels at a different query subset.

**Optimization approach:** Multi-objective search with per-cluster accuracy as
separate objectives. The frontier configs become the ensemble members; at inference
time, route each query to the best-suited config or run top-N and take the union.

---

## 3. Actionable Side Information (ASI)

ASI is the core unlock. Today each run collapses to a scalar (X/30 correct).
But we already capture rich per-query data in the trajectory JSONs — we just
don't feed it back into the optimization loop.

### 3.1 Failure Classification

Every query result gets tagged with a primary failure mode:

| Failure mode | Definition | Detection |
|---|---|---|
| `retrieval_miss` | Right document never retrieved | Ground-truth docid not in `retrieved_docids` |
| `extraction_error` | Right document retrieved, wrong fact extracted | Docid present but answer incorrect |
| `hedge` | Model equivocated or said "unable to determine" | Pattern match on final answer text |
| `timeout` | Hit turn/time limit before submitting | Status = timeout, no FINAL() called |
| `wrong_verify` | Verified an incorrect candidate answer | VERIFY search issued for wrong entity |
| `correct` | Correct answer | Evaluator confirms match |

Current data shows ~86% of failures are `retrieval_miss`. This tells the optimizer
that prompt changes targeting extraction or verification are unlikely to help —
the bottleneck is getting the right documents into scope.

### 3.2 Per-Query ASI Structure

```python
{
    "query_id": "572",
    "correct": false,
    "failure_mode": "retrieval_miss",
    "searches_issued": [
        "Kiki Gyan Ghanaian musician",
        "highlife funk keyboard Ghana",
        "Gyan album discography"
    ],
    "n_unique_searches": 8,
    "n_docs_read": 4,
    "retrieved_docids": ["doc_1823", "doc_9441", "doc_7732", "doc_5519"],
    "ground_truth_docid_retrieved": false,
    "turns_used": 12,
    "token_cost": 14891,
    "final_answer": "24 Hour Party People",
    "expected_answer": "Disco Train"
}
```

For the optimizer, the key signals are:
- **Which queries fail consistently** across prompt variants (hard queries vs prompt-sensitive queries)
- **What searches the model issued** — were they reasonable? Too broad? Missing key terms?
- **Whether retrieval or extraction failed** — guides whether to optimize search strategy vs. reading strategy
- **Token cost** — enables Pareto optimization against accuracy

### 3.3 Aggregated ASI for Prompt Optimization

Beyond per-query diagnostics, we can surface patterns to the GEPA proposer:

```python
{
    "accuracy": 10,
    "total": 30,
    "failure_distribution": {
        "retrieval_miss": 17,
        "extraction_error": 2,
        "timeout": 1,
        "correct": 10
    },
    "consistently_correct": [159, 175, 191, 464, 797],  # correct in >80% of runs
    "consistently_wrong": [330, 412, 668, 901, 1003],    # never correct
    "high_variance": [469, 572, 830, 853, 885],          # correct in some runs
    "avg_token_cost": 149481,
    "avg_unique_searches": 7.2,
    "insight": "17/20 failures are retrieval misses. The model's search queries "
               "are reasonable but the target documents lack keyword overlap with "
               "the query. High-variance queries (469, 572, 830, 853, 885) are "
               "the optimization opportunity — they're solvable but not reliably."
}
```

This is the "gradient" analogy from the GEPA paper — instead of telling the
proposer "you scored 10/30," we tell it "17 queries failed because the right
documents were never found; 5 queries are solvable but flaky; here are the
search patterns that worked vs. didn't."

---

## 4. Optimization Modes

### 4.1 Prompt Optimization (Generalization Mode)

```python
import gepa.optimize_anything as oa

# Split queries into train/val
train_queries = sample30[:20]
val_queries = sample30[20:]

def evaluate(candidate: dict, example: dict) -> tuple[float, dict]:
    """Run one query with the candidate prompt config."""
    prompt_text = candidate["system_prompt"]
    query = example["query"]
    expected = example["answer"]

    result = run_single_query(prompt_text, query, config=DEFAULT_CONFIG)
    correct = evaluate_answer(result["final_answer"], expected)

    return float(correct), {
        "query_id": example["id"],
        "failure_mode": classify_failure(result, expected),
        "searches": result["searches_issued"],
        "docs_read": result["retrieved_docids"],
        "turns_used": result["turns_used"],
        "tokens": result["total_tokens"],
        "final_answer": result["final_answer"],
        "expected": expected,
    }

result = oa.optimize_anything(
    seed_candidate={"system_prompt": CURRENT_SYSTEM_PROMPT},
    evaluator=evaluate,
    dataset=train_queries,
    valset=val_queries,
    objective=(
        "Optimize the system prompt for an rLM agent that answers factoid queries "
        "by searching a 100K web document corpus. The agent works in a Python REPL "
        "with search(), get_document(), and llm_query() tools. 86% of failures are "
        "retrieval misses — the agent's search queries don't find the right documents. "
        "Focus on improving search strategy and query decomposition."
    ),
    background=(
        "BM25 keyword search. The agent gets 12 iterations. Pre-search decomposes "
        "the query into 5 sub-queries. Key insight: the model reasons well once it "
        "has the right documents — the bottleneck is search, not extraction."
    ),
    config=oa.GEPAConfig(
        engine=oa.EngineConfig(max_metric_calls=200),
    ),
)
```

**Cost estimate:** 200 metric calls × 20 train queries × ~2 min/query = ~133 GPU-hours.
Reducible by evaluating on a representative 8-query subset first (queries spanning
the consistently-correct, high-variance, and consistently-wrong categories).

### 4.2 Joint Config + Prompt Search

For the combined prompt + config search, the candidate becomes a dict:

```python
seed = {
    "system_prompt": CURRENT_SYSTEM_PROMPT,
    "config": json.dumps({
        "n_presearch_queries": 5,
        "presearch_k": 5,
        "bm25_phrase_boost": 4.0,
        "bm25_slop_boost": 2.0,
        "bm25_and_boost": 1.5,
        "max_iterations": 12,
        # ... etc
    })
}
```

The evaluator parses the config, writes it to a TOML file, and invokes `gw-bench`
with `--config`. This is the ARC-AGI pattern from GEPA — optimizing the entire
agent system (code structure + parameters) as a single text artifact.

### 4.3 Pareto Ensemble Construction

Multi-objective optimization with per-cluster accuracy:

```python
def evaluate(candidate, example):
    result = run_single_query(candidate["system_prompt"], example["query"])
    correct = evaluate_answer(result["final_answer"], example["answer"])

    scores = {"accuracy": float(correct)}

    # Per-cluster scores for Pareto selection
    cluster = example.get("cluster", "general")
    scores[f"accuracy_{cluster}"] = float(correct)

    return scores, build_asi(result, example)
```

Queries would be tagged with clusters based on our failure analysis:
- `entity_lookup` — single-entity factoid (names, dates, places)
- `multi_hop` — requires combining facts from multiple documents
- `temporal` — requires reasoning about dates/chronology
- `rare_entity` — target entity has low corpus frequency

The Pareto frontier preserves configs that are best-in-class for any cluster,
even if their average accuracy is suboptimal.

---

## 5. Implementation Plan

### Phase 1: ASI Infrastructure

**Goal:** Machine-readable per-query diagnostics, independent of GEPA.

**Changes:**
- `bench/browsecomp/classify_failures.py` — new script that reads trajectory
  JSONs, tags each query with a failure mode, outputs structured ASI JSON
- Requires ground-truth docid mapping (from `browsecomp_plus_decrypted.jsonl`)
  to distinguish retrieval_miss from extraction_error
- Produces `asi_summary.json` per run with the aggregated structure from §3.3

**Validation:** Run on existing trajectory data from the f6d4868 best run and
its reproducibility runs. Confirm the ~86% retrieval_miss rate matches manual
analysis.

**Value even without GEPA:** Turns results.tsv from "X/30" into an analyzable
failure distribution. Makes future manual experiments more targeted.

### Phase 2: Config Extraction

**Goal:** All hardcoded parameters in a single, serializable config struct.

**Changes:**
- `crates/gw-bench/src/main.rs` — new `BenchConfig` struct, loaded from TOML
  via `--config <path>`. Falls back to current hardcoded defaults when absent.
- `bench/browsecomp/configs/` — directory of named config TOML files
  (`baseline.toml`, `best.toml`, etc.)
- System prompt loaded from a text file path in the config, rather than a Rust
  `const`

**Benefit:** Every run is reproducible from a single config file. Enables
automated parameter sweeps and GEPA integration.

### Phase 3: Evaluator Wrapper

**Goal:** A Python function matching the `optimize_anything` evaluator signature.

**Changes:**
- `bench/browsecomp/gepa_evaluator.py` — wraps the pipeline:
  1. Receives `(candidate_dict, example_dict)`
  2. Writes prompt to temp file, config to temp TOML
  3. Invokes `cargo run --release --bin gw-bench -- --config <toml> --query <single_query>`
     (requires adding single-query mode to gw-bench)
  4. Runs `classify_failures.py` on the output
  5. Returns `(score, asi_dict)`
- `crates/gw-bench/src/main.rs` — add `--single-query` flag that runs one query
  and exits (avoids 30-query overhead per metric call during optimization)

### Phase 4: Optimization Runs

**Goal:** Use GEPA (or equivalent search) to find better prompts and configs.

**Approach:**
1. Start with prompt-only optimization (§4.1) on 8-query representative subset
2. Validate best candidates on full 30-query set
3. Expand to joint config+prompt search (§4.2) if prompt-only shows gains
4. Extract Pareto frontier for ensemble (§4.3)

**Compute budget:** ~50-100 GPU-hours for initial prompt optimization on 8-query
subset. Scale to full 30-query evaluation for top candidates only.

### Phase 5: Ensemble Routing

**Goal:** Production-ready multi-config ensemble.

**Changes:**
- `bench/browsecomp/ensemble.py` — runs top-N Pareto frontier configs per query,
  takes union of answers (or confidence-weighted vote if answer confidence is
  available from trajectory analysis)
- Query clustering heuristic based on ASI patterns from Phase 1

---

## 6. Expected Impact

| Phase | Expected accuracy | Confidence | Cost |
|---|---|---|---|
| Current best | 36.7% (11/30), mean ~33% | Known | — |
| + ASI-guided manual iteration | ~37-40% (11-12/30) | Medium | Low (human time only) |
| + GEPA prompt optimization | ~40-47% (12-14/30) | Medium | ~50-100 GPU-hours |
| + Joint config search | ~43-50% (13-15/30) | Low-Medium | ~100-200 GPU-hours |
| + Pareto ensemble (3 configs) | ~47-53% (14-16/30) | Medium | 3x inference cost |

These estimates are conservative. The ARC-AGI case study showed a ~3x accuracy
improvement from agent architecture optimization, but our task is more retrieval-
bound — prompt changes can't fix queries where the target document simply lacks
keyword overlap with any reasonable search terms.

The 86% retrieval-miss rate is both the ceiling and the key insight: GEPA can
optimize search strategy (query decomposition, keyword selection, iterative
refinement) but can't conjure BM25 matches that don't exist in the index. The
planned A3 (larger model) and R2 (ColBERT reranker) experiments address the
retrieval ceiling directly and are complementary to this work.

---

## 7. Alternatives Considered

**DSPy optimizers.** DSPy's MIPRO and BootstrapFewShotWithRandomSearch could
optimize our prompts, but they lack ASI (feedback is scalar only) and Pareto
selection. GEPA subsumes DSPy's optimization approach while adding richer
diagnostics.

**Manual grid search.** We've effectively been doing this. It found the current
best but can't explore interactions efficiently. A 3-parameter grid at 5 levels
each = 125 runs × 58 min = 5 days. GEPA's directed search is more sample-
efficient.

**Bayesian optimization (Optuna).** Good for numeric parameters (Phase 2 config)
but can't optimize free-text prompts. Could complement GEPA for the config
search specifically. GEPA's blackbox case study showed it matches Optuna on
numeric optimization while also handling text.

**No optimization — just try a bigger model.** The A3 experiment (32B model) is
orthogonal and should happen regardless. But a bigger model with the same prompt
leaves accuracy on the table. The coding agent skills case study showed
optimized prompts improved even Claude Sonnet 4.5 from 94.8% to 100%.

---

## 8. Open Questions

1. **Train/val split strategy.** Which 20/10 split best represents the query
   distribution? Should we stratify by difficulty (consistently correct vs.
   high-variance vs. never correct) or by query type (entity vs. multi-hop)?

2. **Single-query evaluation noise.** Individual queries have high run-to-run
   variance. Should the evaluator run each query 2-3 times and report the
   majority result? This 2-3x cost but reduces noise that could mislead the
   optimizer.

3. **Prompt structure constraints.** Should we let GEPA modify the entire
   prompt freely, or constrain it to preserve structural elements (tool
   descriptions, code examples) while only optimizing strategy/directive
   sections?

4. **GEPA availability.** The blog post shows `pip install gepa` but the
   library may not yet be publicly available. If not, we can implement the
   core loop ourselves: LLM proposer → evaluator → ASI feedback → Pareto
   selection. The ASI infrastructure (Phase 1) and config extraction (Phase 2)
   are valuable regardless.

5. **Interaction with other planned experiments.** Should we optimize on the
   current pipeline first (boosted BM25, 9B model) or wait for A3 (larger
   model) / R2 (ColBERT reranker)? Optimizing the current pipeline first
   establishes a baseline; the optimized prompt may transfer to the improved
   pipeline.
