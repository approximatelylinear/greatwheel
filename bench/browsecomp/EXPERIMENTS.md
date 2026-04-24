# BrowseComp-Plus Experiment Designs

## Current Best: 40.0% fuzzy (12/30) — commit 1331262

Passage-level BM25 + doc-level boosted BM25 fused via RRF (no reranking).
Previous best: 40.0% exact (12/30, commit 7f431fc) with BM25 top-200 → ColBERT rerank.

## What We've Learned

### Things that help
- **ColBERT reranking** (Reason-ModernColBERT on BM25 top-200): **New best.** BM25 retrieves top-200, ColBERT reranks to top-10. Right documents were buried at ranks 50-200 in BM25; ColBERT surfaces them. Sweet spot is top-200 — top-50 is too narrow, top-500 adds noise.
- **Boosted BM25** (phrase match 4x, slop 2x, AND 1.5x, OR 1x): Biggest single contributor per ablation (+2-3 queries vs naive BM25)
- **Pseudo-relevance feedback (PRF)**: Extract distinctive terms from round-1 snippets, use as additional queries (~1-2 query lift)
- **Doc-grounded prompts**: Force get_document()+llm_query(), emphasize corpus-only answers
- **More pre-search queries (1→5)**: Decomposes multi-hop queries into independent sub-fact searches
- **k=10** (vs k=5): More retrieval breadth without overwhelming the model
- **12 turns** (vs 6-8): Enough iterations to explore, verify, and submit
- **think:false for utility calls**: Saves tokens on pre-search extraction and fallback
- **Search decomposition prompt**: "Decompose the query, search each part separately"
- **VERIFY directive**: "Search for the candidate answer by name to confirm"
- **llm_query() emphasis**: Sub-LLM calls are the model's most powerful analysis tool
- **Fallback extraction**: Recovers answers when model doesn't call FINAL in time

### Things that don't help (or hurt)
- **HyDE (hypothetical document embeddings)**: Ablated out — no benefit for BM25 queries
- **Vector search — all variants tested**: See "Vector Search Findings" below
- **More context documents (>25-30)**: Dilutes signal, overwhelms 9B model attention
- **More turns (16 vs 12)**: Model second-guesses itself, goes down rabbit holes
- **vector_search as REPL tool**: Model over-uses it, wastes turns on noisy semantic results
- **Python analysis emphasis**: Analysis isn't the bottleneck — retrieval is
- **LLM ranking of pre-search**: Fragile parsing, adds overhead, discards useful docs
- **Best-of-N majority voting**: Wrong answers agree more than correct ones (3x cost, same accuracy)
- **Search query coaching in prompt**: No measurable benefit over baseline
- **Wider pre-search (8+5 queries, full k)**: Too many context docs overwhelm model
- **Fuzzy BM25 matching** (edit distance 1, boost 0.5): Adds noise, regression to 4/30
- **Answer type classification**: One LLM call to classify expected type (person/place/date). Within variance, no clear benefit
- **ColBERT reranking of BM25 top-500**: Too wide a candidate pool — noise dilutes ColBERT precision (9/30 vs 12/30 at top-200)

### Vector Search Findings (mar16-mar19)

We ran an extensive set of vector search experiments. **None improved accuracy over BM25-only.**

| Experiment | What we tried | Result |
|---|---|---|
| Hybrid BM25+vector pre-search | Vector search during pre-search phase | 7/30, same as baseline, +53% tokens |
| Hybrid default search() in REPL | Agent's search() uses hybrid instead of BM25 | 7/30, no improvement |
| V1: Hybrid uses boosted BM25 | `search_hybrid()` calls `search_bm25_boosted()` | Tested as part of V1-V4 bundle |
| V2: nomic query prefix | Prepend `search_query:` to queries before embedding | Tested as part of V1-V4 bundle |
| V3: Weighted RRF (2x BM25) | BM25 results counted twice in RRF fusion | Tested as part of V1-V4 bundle |
| V4: LanceDB nprobes=50 | Increase IVF-PQ probes for better recall | Tested as part of V1-V4 bundle |
| V1-V4 bundle (unprefixed docs) | All four changes, docs without `search_document:` prefix | 7/30 — prefix mismatch hurts |
| V6: Re-embed with prefix | 100K docs re-embedded with `search_document:` prefix (37 min) | Done, table: `browsecomp_docs_prefixed` |
| V1-V4 bundle (prefixed docs) | All four changes, docs with proper prefix | **9/30** — still no improvement over BM25-only (9-11/30) |

**Conclusion:** Vector search with nomic-embed-text adds no value on BrowseComp, even with:
- Proper asymmetric prefixes (`search_query:`/`search_document:`)
- BM25-weighted RRF fusion
- Increased nprobes for better ANN recall
- Boosted BM25 in the hybrid path

Possible explanations:
1. BrowseComp queries are entity-heavy (names, dates, places) — BM25 keyword matching is naturally strong here
2. nomic-embed-text (768d) may lack the semantic discrimination needed for 100K docs
3. The RRF fusion adds noise — documents retrieved by vector but not BM25 tend to be irrelevant
4. The 9B model generates keyword-style search queries that don't benefit from semantic matching

**Recommendation:** Use boosted BM25 only for search. Vector search adds latency without accuracy gains. The `embed_queries()` helper and prefixed LanceDB table (`browsecomp_docs_prefixed`) remain available if we want to revisit with a different embedding model.

### Timing Profile (mar19)

Per-query timing breakdown (avg across 30 queries, V1-V4 prefixed run):
- **Total: 115s** per query
- **Pre-search: 12s** (10%) — LLM query decomposition + BM25 searches + PRF
- **rLM loop: 103s** (90%) — 12-iteration agent loop (dominated by LLM inference)
- **Embedding: ~20ms** per call (warm) — not a bottleneck
- **LanceDB ANN: ~5-8ms** per call (warm) — not a bottleneck

The bottleneck is LLM inference in the rLM loop, not search latency.

### ColBERT Reranking Findings (mar21)

ColBERT (Reason-ModernColBERT, 149M params) as a reranker on BM25 results is the first retrieval improvement to beat the BM25-only baseline.

**Architecture:** Rust native boosted BM25 retrieves top-N candidates → Python sidecar (`rerank_server.py`) scores them with ColBERT MaxSim → returns top-10 to the agent. BM25 search stays in Rust; only reranking uses Python.

| BM25 pool | ColBERT → top-10 | Notes |
|---|---|---|
| Top-50 | 7/30 | No improvement — too narrow, right docs not in pool |
| Top-100 | 10/30 | Ties previous best (11/30) |
| **Top-200** | **12/30** | **New all-time best (40%)** |
| Top-500 | 9/30 | Too wide — noise dilutes ColBERT precision |

**Why ColBERT works where vector search failed:**
- Dense vector search (nomic-embed-text) compresses documents into a single embedding — too lossy for entity-heavy queries
- ColBERT computes token-level MaxSim scores, preserving fine-grained keyword matching
- As a reranker (not first-stage retriever), it refines BM25's candidate set rather than replacing it
- The right documents *are* in BM25's top-200, just not ranked highly enough — ColBERT surfaces them

**Model note:** qwen3.5:9b is required for competitive results. qwen2.5:7b scores ~3x worse (3-4/30 vs 8-12/30) with identical code.

### Passage Index Findings (mar27)

Passage-level BM25 is the biggest single retrieval improvement in the project.

**Architecture:** 100K documents split into ~512-byte chunks with 100-byte overlap at sentence boundaries → 8.5M passages indexed in a separate tantivy index. At search time, both doc-level boosted BM25 and passage-level BM25 run in parallel, results fused via RRF.

| Configuration | Exact | Fuzzy | Notes |
|---|---|---|---|
| Doc-level BM25 only | 8/30 | — | Baseline (no passages, no rerank) |
| **Passage + doc RRF** | **10/30** | **12/30** | **New all-time best (fuzzy). +2 exact, +4 fuzzy over baseline** |
| Passage + doc RRF + ColBERT rerank | 7/30 | 8/30 | Reranking hurts — regression from passage-only |

**Why passages help:**
- Long documents where the answer is in a small section get diluted by doc-level BM25 — the passage index matches the relevant paragraph directly
- Q1106 (Illusion) and Q1144 (dangerous driving) are new solves: answers were buried in documents too long for doc-level BM25 to rank highly
- Passage-level snippets are more focused — the model gets the relevant paragraph instead of scanning a full document

**Why ColBERT reranking hurts passages:**
- 10 queries lost gold doc access after reranking (passage RRF had the gold doc, rerank dropped it)
- ColBERT's semantic similarity ≠ BM25's keyword relevance for entity-heavy BrowseComp queries
- The gold documents contain specific keywords that BM25 matches perfectly; ColBERT favors topically similar but wrong documents
- 5 previously correct answers lost, only 2 gained

**Conclusion:** For BrowseComp's entity-heavy queries, BM25 keyword matching (doc + passage) is the right strategy. Semantic reranking consistently hurts by diluting keyword precision. ColBERT reranking should not be used with passage-level retrieval.

### ColBERT First-Stage Retrieval + Multi-Backend Comparison (apr7-apr10)

We built the full ColBERT first-stage retrieval pipeline (not just reranking) and
compared four search backends head-to-head. Full details in `docs/design-colbert-production.md`.

**Infrastructure built:**
- Passage blob store (Lance, 122 GB, 1M passages × float16 token tensors)
- Candle-based Rust encoder (parity-tested: cos sim 1.000000 vs Python HF encoder)
- Searcher protocol + 4 backends: brute-force, LanceDB MV, Qdrant, Elasticsearch
- Docker compose for ES 8.18 + Qdrant 1.12
- Shared HTTP ColBERT encoder service

**Retrieval recall comparison** (100K docs, 2000 tokens/doc cap, sample30):

| Backend          | R@5   | R@10  | R@20  | R@50  | R@100 | R@200 | p50 ms |
|------------------|-------|-------|-------|-------|-------|-------|--------|
| Qdrant ColBERT   | 10/30 | 13/30 | 15/30 | 21/30 | 25/30 | 25/30 | 25,680 |
| LanceDB MV       | 10/30 | 13/30 | 15/30 | 21/30 | 25/30 | 25/30 | 46,076 |
| Tantivy BM25     | 4/30  | 8/30  | 8/30  | 9/30  | 11/30 | 12/30 | 266    |
| Tantivy+rerank   | 7/30  | 7/30  | 8/30  | 11/30 | 12/30 | 12/30 | 87,500 |
| ES rank_vectors  | 0/30  | 0/30  | 0/30  | 0/30  | 0/30  | 0/30  | 10,065 |

ColBERT retrieval reaches **25/30 R@200** vs BM25's 12/30. Qdrant and LanceDB MV agree
perfectly (both native MaxSim). ES 8.18 `rank_vectors` is unusable at scale (script_score
brute-force, timed out on 8/30 queries).

**End-to-end agent evaluation** (qwen3.5:9b, Qdrant ColBERT backend):

| Backend          | Retrieval R@200 | Agent accuracy | Delta vs BM25 |
|------------------|-----------------|----------------|---------------|
| BM25 (baseline)  | 12/30           | 9/30 (30%)     | —             |
| Qdrant ColBERT   | 25/30           | 7/30 (23%)     | **-2**        |

**Better retrieval did NOT translate to better agent accuracy.** The Qdrant-backed
agent scored worse despite retrieving 13 more relevant docs. Root causes:

1. **Query latency (26s/search × 12 calls > 180s timeout)** — the biggest factor.
   The agent ran out of time before completing its reasoning chain on most queries.
2. **Agent prompt was optimized for BM25** — pre-search decomposition and PRF
   heuristics aren't adapted to ColBERT's semantic result orderings.
3. **2000-token per-doc cap** (Qdrant hard limit) — truncates long docs, losing
   answers in later passages.
4. **3000-char snippet truncation** — same as BM25 path but prompts aren't tuned
   to work around it with ColBERT results.

**Conclusion:** Retrieval recall is necessary but not sufficient. The agent pipeline
needs co-adaptation: faster serving → co-tuned prompt → measure → iterate.

### End-to-End Agent Evaluations with ColBERT Backends (apr10-apr12)

Four agent runs on sample30, all using qwen3.5:9b with baseline config:

| Backend                         | Accuracy | Correct queries                              |
|---------------------------------|----------|----------------------------------------------|
| Tantivy BM25 (no rerank)        | 5/30     | 175, 191, 572, 643, 885                      |
| **Tantivy BM25 + blob rerank**  | **7/30** | 191, 469, 572, 797, 853, 894, 1144           |
| **Qdrant ColBERT native MV**    | **7/30** | 159, 464, 572, 643, 830, 885, 1030           |
| BM25s + blob rerank (no boosts) | 4/30     | 464, 469, 572, 797                           |

**Note:** The tantivy baseline reproduced at 5/30, not the historical 9/30.
LLM non-determinism and possible model updates contribute to run-to-run
variance. All runs in this comparison were done in the same session with
the same Ollama model to control for this.

**Key findings:**

**1. Blob rerank helps: +2 over baseline (5→7).** The rerank surfaces
docs buried at BM25 ranks 50-200 that the agent would otherwise never
see. Gains: q469, q797, q853, q894, q1144. Losses: q175, q643, q885
(ColBERT semantic similarity preferred wrong docs over BM25's keyword
matches for these entity-heavy queries).

**2. Qdrant ties at 7/30 but with a different correct set.** Only q572
is correct across all three approaches. Qdrant finds q159, q830, q1030
that no BM25-based approach finds (Weller-bound queries). BM25+rerank
finds q469, q797, q853, q894, q1144 that Qdrant misses.

**3. Union across approaches: 14/30 (47%).** If we could pick the right
backend per query, we'd solve 14 of 30. This is strong evidence for
ensemble/routing: different retrieval strategies have complementary
strengths on different query types.

**4. BM25 boosts matter.** BM25s (Python, no boosts) scored 4/30 vs
tantivy (Rust, phrase+slop+AND boosts) at 5/30. The correct sets barely
overlap. The boosted BM25 implementation isn't just cosmetically better
— it finds genuinely different documents. All future experiments should
use the Rust native backend to ensure boosted BM25.

**5. Latency was NOT the bottleneck this time.** The blob rerank runs
at ~200ms/query (vs Qdrant's ~26s). Both the tantivy+rerank and native
baseline runs completed all 30 queries well within timeout. The Qdrant
run also completed (26s/search × ~12 calls ≈ 312s > 180s timeout, but
the agent adapted by making fewer calls). The 7/30 results are genuine
accuracy, not timeout artifacts.

### Key insight
86% of failures are **retrieval** (right documents never found), not extraction. The 9B model reasons well once it has the right documents. Dense vector search doesn't help — but ColBERT reranking of a wider BM25 pool does, because the right documents are often in BM25's top-200 but ranked too low for the agent to see.

### Meng et al. 2026 — `Revisiting Text Ranking in Deep Research` (apr24)

Independent study on the **same BrowseComp-Plus dataset** we use. Full 830
queries × 2 open-source agents (gpt-oss-20b, GLM-4.7-Flash-30B) × 5
retrievers (BM25, SPLADE-v3, RepLLaMA, Qwen3-Embed-8B, ColBERTv2) × 3
re-rankers (monoT5-3B, RankLLaMA-7B, Rank1-7B). arXiv:2602.21456v1.

**Their best single-pipeline result**: gpt-oss-20b + BM25 (passage corpus)
+ monoT5-3B re-ranker at depth 50 = **recall 0.716, accuracy 0.689**.
Relative to no re-ranking, this is +16.2% recall, +20.5% accuracy — the
biggest single-lever gain in their whole study.

**What corroborates what we found:**
- Passage-level > document-level, especially on smaller-context agents
  (gpt-oss-20b 131K) — matches our +5/+6 passage lift.
- BM25 on passages BEATS all neural retrievers (raw): BM25 0.572 >
  SPLADE-v3 0.516 > ColBERTv2 0.521 > RepLLaMA 0.406 > Qwen3-Embed-8B
  0.417. Matches our Qwen3 Phase 2 regression.
- Reasoning-based re-ranker (Rank1-7B) shows no clear advantage over
  non-reasoning rerankers on keyword-style queries — matches our doubts
  about reasoning on agent-issued queries.
- Agent-issued queries are web-search style with quotation marks and
  exact-match tokens (their Table 5 examples: `"90+7" attendance 61700`,
  `Man United" "4-1" "90+4"`) — which is exactly why lexical retrieval
  dominates.

**What contradicts what we found:**
- Re-ranking helps them *consistently*; it *hurt* us. Gap: they test
  monoT5-3B / RankLLaMA-7B / Rank1-7B (T5- / Llama- cross-encoder
  rerankers trained on MS MARCO). We only tried **ColBERT-as-reranker**
  and it regressed -6 fuzzy on sample30. ColBERT is used as a *retriever*
  in their study, not a re-ranker — the two are different jobs. A
  T5-style cross-encoder reranker on top of our BM25+passage stack is
  an **untested, likely big lever** on our side.

**Novel method worth stealing — Q2Q (query-to-question)**. Neural rankers
are trained on MS MARCO natural-language questions, but agents emit
keyword-style web-search queries → training–inference mismatch hurts
neural rerankers badly. Their fix: a small LLM call reformulates the
agent's keyword query into an NL question (optionally conditioning on
the agent's recent reasoning trace). Keep the raw query for BM25, use
the NL version for the neural reranker. Gains (their Table 11, passage
corpus, gpt-oss-20b):
- SPLADE-v3: raw 0.516 → Q2Q(Q+R) 0.557 (+7.95% relative, p<0.05)
- Qwen3-Embed-8B: raw 0.417 → Q2Q(Q+R) 0.459 (+10.1% relative, p<0.05)
- SPLADE-v3 + Rank1(d=10): raw 0.580 → Q2Q(Q+R) 0.613 (+5.7%)
- BM25: Q2Q *hurts* (expected, BM25 wants keywords)

**Recipe details worth copying:**
- Passage size: ~250 words (spaCy `en_core_web_sm` sentence-bounded),
  avg 279 tokens. Our current 4096-byte chunks are ~1K tokens — roughly
  4x theirs.
- Title prepended to each passage when available.
- BM25 defaults (`k1=0.9, b=0.4`) are fine on passages but *worst* on
  documents. Better on documents: `k1=3.8, b=0.87` or `k1=10, b=1`
  (stronger length normalization).
- Deeper rerank depths help: d=10 → d=20 (+6.2% recall, +6.8% acc);
  d=20 → d=50 (+2.1% recall, +2.2% acc).
- Full-document reader tool *helps* on document corpus (truncated input
  context), *hurts* on passage corpus (redundant — passage retrieval
  already gives access to relevant segments).

**Blind spots (where our direction is orthogonal):**
- They do NOT test ensembling, multi-retriever union, or diversity-based
  set selection. Their whole contribution is *vertical* (better
  retriever → reranker → deeper depths). Our complementary-channel /
  coverage-pre-search finding (3-way union = 20/30 on sample30) is a
  different axis; the paper is silent on whether it's better to optimize
  one pipeline or compose several.

---

## Completed Experiments (mar16-mar27)

### Retrieval (Rust: gw-memory, gw-bench)
- [x] **Boosted BM25** — phrase match (4x), slop phrase (2x), AND'd terms (1.5x), OR'd terms (1x). **Biggest contributor.**
- [x] **PRF (pseudo-relevance feedback)** — distinctive snippet terms as additional queries. Worth ~1-2 queries.
- [x] **HyDE** — hypothetical answer passage as BM25 query. **Ablated out: no benefit.**
- [x] **Vector search (6 variants)** — all showed no improvement. See table above.
- [x] **Wider pre-search** — 8+5 queries overwhelm model with context.
- [x] **Fuzzy BM25 matching** — edit distance 1, boost 0.5. **Regression — noise hurts.**
- [x] **ColBERT reranking** — BM25 top-N → Reason-ModernColBERT rerank. **New best at top-200 (12/30).** See ColBERT findings above.

### Prompts (Rust: gw-bench)
- [x] **Doc-grounded prompts** — force document reading + verification. Matches best.
- [x] **Early-submit prompts** — nudge when evidence found. Slight regression.
- [x] **Search query coaching** — BM25 formulation advice. No benefit.
- [x] **Answer type classification** — classify expected answer type before REPL loop. Within variance, no clear benefit.

### Hindsight Integration (mar26-mar27)
- [x] **FactRegistry (Phase A)** — structured evidence tracking in REPL. Available but not prompt-mandated (bloated prompts regress 9B model).
- [x] **entity_search (Phase B)** — one-hop entity chain BM25 search. Available in namespace for multi-hop questions.
- [x] **Passage-level BM25 index (Phase C/BC-4)** — 8.5M passages (512-byte chunks, 100-byte overlap). **New all-time best: 12/30 fuzzy.**
- [x] **Passage + doc RRF (BC-6)** — fused via RRF. +2 exact, +4 fuzzy over doc-only.
- [x] **Passage + ColBERT rerank** — reranking hurts passages (7/30). ColBERT drops keyword-matched gold docs.
- [ ] **Entity co-occurrence graph (BC-5)** — deferred, see P4 in Planned Experiments.

### Infrastructure
- [x] **Per-query timing breakdown** — pre_search_ms, rlm_loop_ms, bridge component timers
- [x] **Re-embedded LanceDB index** — `browsecomp_docs_prefixed` with `search_document:` prefix (100K docs, 37 min)
- [x] **ColBERT rerank server** — `bench/browsecomp/rerank_server.py` + `colbert_reranker.py`. Python sidecar, Rust calls via HTTP.
- [x] **FINAL() extraction fix** — skip code blocks, require quoted strings (prevents variable names as answers)
- [x] **Passage index CLI** — `--build-passage-index` and `--passage-index` flags on gw-bench

### ColBERT Production Pipeline (apr7-apr10)
- [x] **Passage blob store** — Lance table, 122 GB, 1M passages × float16 token tensors. `build_passage_blob_store.py`, 96 min build.
- [x] **Blob reranker (Python)** — `blob_reranker.py`, fetches precomputed tokens from blob store, MaxSim via torch. 200ms/query.
- [x] **Blob reranker (Rust)** — `gw-memory/src/colbert_blobs.rs`, parity-tested (max diff 2.12e-06 vs Python).
- [x] **Candle ColBERT encoder (Rust)** — `gw-memory/src/colbert/candle_encoder.rs`, loads Reason-ModernColBERT via candle-transformers. Parity-tested: cos sim 1.000000 vs Python HF encoder.
- [x] **ColbertStore (Rust)** — `gw-memory/src/colbert/mod.rs`, composes encoder + first-stage retriever + blob reranker.
- [x] **Searcher protocol** — `searchers/base.py`, `EncoderClient`, uniform `search(query, k)` interface.
- [x] **4 search backends** — brute_force, lancedb_mv, elasticsearch, qdrant. All smoke-tested + full-build completed.
- [x] **BlobRerankWrapper** — decorator that adds blob-store MaxSim rerank on top of any Searcher.
- [x] **Shared encoder service** — `colbert_server.py`, HTTP API (encode_query, encode_doc_batch).
- [x] **retrieval_benchmark_v2.py** — unified multi-backend harness with `--searchers` flag.
- [x] **Docker compose for benchmarks** — `docker/docker-compose.bench.yml` (ES 8.18 + Qdrant 1.12).
- [x] **Qdrant search server** — `search_server_qdrant.py`, drop-in replacement for search_server.py.
- [x] **End-to-end agent eval** — 30 queries with Qdrant ColBERT backend. Result: 7/30 (regression from BM25 9/30).

---

## Planned Experiments

### BM25 Refinements

#### B1. Fuzzy term matching (5th signal) — DONE, no benefit
- **Result:** Regression to 4/30. Fuzzy matching adds noise that overwhelms the signal.

#### B2. Document field boosting (BM25f)
- **Where:** `crates/gw-memory/src/corpus.rs` — index schema + search
- **What:** Extract title/heading from web documents at index time. Add separate tantivy fields with higher boost.
- **Effort:** Large — requires corpus analysis + re-indexing
- **Expected impact:** Medium — titles are high-signal for entity matching
- **Status:** Deprioritized — ColBERT reranking addresses ranking quality more directly

#### B3. Query expansion via LLM
- **Where:** `crates/gw-bench/src/main.rs` — pre_search or bridge
- **What:** Before each BM25 search, expand the query with synonyms/related terms via a fast LLM call.
- **Effort:** Medium
- **Expected impact:** Low-medium — could widen BM25 recall, benefiting ColBERT reranking
- **Status:** Worth revisiting now that ColBERT reranking is in place

#### B4. BM25 parameter tuning on document corpus
- **What:** Meng et al. grid-search BM25 on BrowseComp-Plus documents and report the default (`k1=0.9, b=0.4`, following tantivy/BrowseCompPlus defaults) is the *worst* setting for docs — larger `b` (stronger length normalization) and larger `k1` help substantially. Their best: `k1=10, b=1` → acc 0.506 (vs 0.259 at default) OR `k1=3.8, b=0.87` → acc 0.513.
- **Where:** `crates/gw-memory/src/corpus.rs` — tantivy BM25 params (currently default). Passages are reportedly fine at default, so only affects the doc-level channel.
- **Effort:** Small — parameter change + rebuild doc index.
- **Expected impact:** Medium on the doc channel (half of our RRF fusion). Uncertain effect on end-to-end when passage channel dominates — our sample30 may already rely on passages more than docs.
- **Status:** Worth a probe. Low cost, bounded risk.

### Agent/Prompt Improvements

#### A1. Iterative search budget enforcement
- **What:** Track unique search queries per question. If < 5 by iteration 4, force more searching before allowing FINAL().
- **Where:** `crates/gw-bench/src/main.rs` — iteration prompts + bridge tracking
- **Effort:** Medium
- **Expected impact:** Low-medium — addresses under-searching

#### A2. Answer type classification — DONE, no benefit
- **Result:** Within variance (7/30 vs 8/30 baseline). No clear benefit.

#### A3. Larger model for main loop
- **What:** Use qwen2.5:32b or qwen3.5:32b for main rLM loop (keep 9B for utility calls).
- **Effort:** Small config change, large latency increase (~4x slower)
- **Expected impact:** High — better reasoning → better queries + extraction
- **Status:** Deprioritized per user preference

#### A4. Config-diverse ensemble with union scoring
- **What:** Run 2-3 diverse configurations per query, take union of correct answers.
- **Expected impact:** High — historical union across configs reaches ~40%
- **Cost:** 2-3x compute
- **Status:** Needs orchestration code

#### A5. Harness reliability fixes
- **What:** Two bugs identified in the rLM loop:
  1. Fallback extraction trajectory uses `role: "system"` — should be `"assistant"`
  2. Final iteration prompt sometimes sent without getting/recording an LLM response
- **Where:** `crates/gw-bench/src/main.rs` — `run_rlm_loop()` and `fallback_extract()`
- **Effort:** Small
- **Expected impact:** Low — fixes edge cases, may recover 1-2 answers

### Alternative Retrieval

#### R1. Different embedding model — Revisiting (was Deprioritized)
- **Original status:** Dense vector search doesn't help on this benchmark regardless of embedding model.
- **Update (apr16):** That conclusion was tested only with nomic-embed-text-v1.5 (~137M encoder). The BrowseComp-Plus paper shows Qwen3-Embedding-8B (a decoder-LLM-class embedder) clearly outperforming BM25 across multiple agents. Their indexing recipe also differs materially: full documents at max_length=4096 with EOS pooling and an instruction-style query prefix — a regime nomic was never tested under here. See R8 below.

#### R2. ColBERT reranker on BM25 results — DONE, new best
- **Result:** BM25 top-200 → Reason-ModernColBERT rerank → top-10 achieves **12/30 (40%)**, new all-time best.
- **Implementation:** Python sidecar (`rerank_server.py`) called via HTTP from Rust native backend.
- **Key finding:** Sweet spot is top-200. Top-50 too narrow (7/30), top-500 too noisy (9/30).
- **See:** ColBERT Reranking Findings section above for full details.

#### R2b. Reason-ModernColBERT as first-stage retriever — DONE
- **Result:** ColBERT first-stage retrieval reaches 25/30 R@200 (vs BM25 12/30). But end-to-end agent accuracy regressed to 7/30 (vs BM25 9/30). See "ColBERT First-Stage Retrieval" findings above.
- **Status:** Infrastructure is complete (4 backends, blob store, Rust encoder, benchmark harness). The bottleneck has shifted from retrieval recall to agent co-adaptation. See R4-R7 below.

#### R3. Learned sparse retrieval (SPLADE-style)
- **What:** Use a learned sparse model that outputs weighted term expansions. Combines BM25-style matching with learned term importance.
- **Effort:** Large — needs model + index infrastructure
- **Expected impact:** Medium-high — addresses vocabulary mismatch without the noise of dense vectors
- **Status:** Research needed

#### R4. BM25 + blob rerank (fast ColBERT rerank, no server) — DONE
- **Result:** 7/30 (+2 over tantivy baseline of 5/30). Gains q469, q797, q853, q894, q1144. Losses q175, q643, q885 (ColBERT's semantic similarity preferred wrong docs on entity-heavy queries). Same accuracy as Qdrant native MV (7/30) but with a **different correct set** — only q572 overlaps. Union of all approaches reaches 14/30.
- **Implementation:** `rerank_server_blobs.py` on port 8001 + `gw-bench --search-backend native --rerank-url http://localhost:8001`. Native tantivy with boosts provides top-200 candidates, blob reranker does MaxSim in ~200ms.
- **Key finding:** Blob rerank is a net positive and latency is NOT the bottleneck (200ms/query, well within 180s timeout). The correct set is complementary to Qdrant's — an ensemble/routing approach would reach 14/30 (47%).

#### R5. LanceDB MV as agent backend (no token cap, no server)
- **What:** LanceDB MV native MaxSim — same 25/30 R@200 as Qdrant, but embedded (no server latency), no per-doc token cap (Qdrant's 2000-token limit doesn't apply). p50 was 46s on the retrieval benchmark, but that's without any ANN index — building an IVF index on the lance table may bring it under 5s.
- **Where:** New `search_server_lancedb_mv.py` wrapping `LanceDbMvSearcher`
- **Effort:** Small — searcher already exists, just needs the HTTP wrapper
- **Expected impact:** Medium — isolates whether the 2000-token cap was causing the Qdrant regression. If LanceDB MV + full tokens gets >9/30, the cap was the bottleneck.
- **Status:** Needs lance IVF index build first, otherwise 46s/query will timeout the agent.

#### R6. ColBERT-tuned agent prompt (GEPA co-optimization)
- **What:** Run GEPA prompt optimization with the ColBERT backend (R4 or R5) instead of BM25. The current prompt was tuned for BM25 keyword results — GEPA will adapt the pre-search decomposition, PRF, and verify strategy to ColBERT's semantic result characteristics.
- **Where:** `run_gepa.py` — configure with ColBERT search server instead of BM25
- **Effort:** Medium — GEPA runs take ~4-8 hours per optimization cycle
- **Expected impact:** High — the prompt-retriever coupling is the most likely cause of the 7/30 regression. Co-optimization is the principled fix.
- **Status:** Blocked on R4 or R5 (need a fast ColBERT backend first, otherwise GEPA iterations will be prohibitively slow at 26s/search).

#### R8. Qwen3-Embedding (LLM-class single-vector dense)
- **What:** Mirror the BrowseComp-Plus paper's indexing recipe with Qwen3-Embedding-0.6B (smallest variant, fast to iterate). Full-doc encoding at max_length=4096, EOS pooling, L2-normalized, no doc prefix. Queries get the BC+ instruction prefix, max_length=512. Single-vector cosine retrieval via LanceDB.
- **Where:** `bench/browsecomp/qwen3_embed_server.py` (HTTP), `build_qwen3_index.py`, `searchers/qwen3_searcher.py`, `qwen3` backend in `retrieval_benchmark_v2.py`. See `DESIGN-qwen3-embed.md` for the full plan.
- **Hypothesis:** R1's "no benefit regardless of embedding model" was nomic-specific. A 0.6B decoder-LLM embedder with full-doc context and the right query prompt is a fundamentally different class of model. If 0.6B already lifts R@200 toward ColBERT's 25/30, escalate to 4B/8B; otherwise the chart's gain is unique to the 8B regime.
- **Effort:** Small-medium (infra is built; ~30 min for full corpus index build at 0.6B on a 4090).
- **Expected impact:** Unknown — that's what the experiment measures. Phase 1 is retrieval R@k, Phase 2 is end-to-end agent eval.
- **Status:** Built apr16. Phase 1 retrieval result (apr17):

**Phase 1 — Retrieval R@k on sample30:**

| Backend                | R@5  | R@10 | R@20 | R@50 | R@100 | R@200 | p50 ms |
|------------------------|------|------|------|------|-------|-------|--------|
| BM25 (tantivy boosted) | 4/30 | 8/30 | 8/30 | 9/30 | 11/30 | 12/30 | 266    |
| **Qwen3-Embed-0.6B**   | 5/30 | 10/30| 11/30| **21/30** | **22/30** | **24/30** | **54** |
| Qdrant ColBERT         | 10/30| 13/30| 15/30| 21/30| 25/30 | 25/30 | 25,680 |
| LanceDB MV (ColBERT)   | 10/30| 13/30| 15/30| 21/30| 25/30 | 25/30 | 46,076 |

**Hypothesis confirmed.** The smallest variant (0.6B) hits R@200 = 24/30 —
essentially matching ColBERT's ceiling — at ~475x lower latency than
Qdrant and ~850x lower than LanceDB MV. The prior "dense doesn't help"
conclusion was definitively nomic-specific. Critically, the 54ms latency
kills the main driver of ColBERT's end-to-end regression (timeout
pressure from 26s/search × 12 calls). Phase 2 (agent eval) now strongly
motivated — this is the first backend with both high recall AND low
enough latency to let the agent complete its reasoning chain.

**Per-query inspection (sanity check).** Ran `inspect_qwen3_hits.py` to
verify the aggregate numbers aren't an artifact of something weird
(identical scores, degenerate ranking, etc.). Rank distribution of the
24 finds: rank 1: 3, rank 2-5: 2, rank 6-10: 5, rank 11-50: 11, rank
51-200: 3. Sums exactly to 24 and reproduces every R@k number in the
table above.

The **R@10 → R@50 jump (10/30 → 21/30)** is the key quantitative story:
eleven queries place gold at ranks 11-50. The agent's default k=10 would
miss all of them; k=50 (or a reranker over the top-200) would catch
them. This is a different retrieval regime than BM25's flatter rank
distribution and should inform the agent's k setting during Phase 2.

Spot checks confirm semantic matching is real:
- *Best rank (q159, art restoration):* gold at rank 1; top-5 are all
  art / museum / restoration documents.
- *Mid rank (q763, "historic site near an airport with a specific
  runway length"):* gold at rank 21; top-5 are thematically adjacent
  (historic sites, airfields, national register lists) but not the
  exact gold — classic dense behavior of hitting the topic zone.
- *Miss (q747):* multi-hop query chaining Person A → state → tribe →
  college → VP from Maine → …. Qwen3 latches onto "Council of Three
  Fires" (the most prominent entity) and retrieves docs about the tribe
  itself, not about Person A. No single-vector encoder solves this in
  one shot — it's a query-decomposition problem.

The 6 misses (q1034, q1257, q469, q689, q747, q853) are all
long-chain multi-hop queries of this shape — same class of queries
that challenge every other backend. They are not close-but-wrong
retrievals.

**Artifacts for Phase 2:** The qwen3 index is at `data/qwen3-embed/`
(401 MB, 100195 docs), encoder service runs on port 8003,
`search_server_qwen3.py` is the drop-in agent-facing HTTP shim
(same contract as `search_server.py` / `search_server_qdrant.py`).

**Phase 2 — End-to-end agent eval on sample30 (apr17-18):**

| Backend                | Exact    | Fuzzy    | Avg tokens |
|------------------------|----------|----------|------------|
| BM25 baseline          | 9/30     | 11/30    | 134K       |
| Qdrant ColBERT         | 12/30    | 13/30    | 122K       |
| **Qwen3-Embed-0.6B**   | **3/30** | **4/30** | **63K**    |

Command: `target/release/gw-bench --search-backend http --search-url http://localhost:8000 --search-mode bm25 --model qwen3.5:9b --config bench/browsecomp/configs/baseline.toml --query bench/browsecomp/sample30.tsv --output-dir runs/qwen3-eval --k 10 --max-turns 12`
(scored via `quick_eval.py --fuzzy` against the same run dir; baseline
scores come from running `quick_eval.py` on `runs/native-bm25-baseline`
and `runs/qdrant-eval` for an apples-to-apples comparison).

**Same "retrieval recall ≠ agent accuracy" pattern as ColBERT, even
more extreme.** Phase 1 R@200=24/30, but Phase 2 fuzzy=4/30. Token
usage is *half* the BM25 run — the agent is terminating early, confident
in wrong answers. Correct: q175 (Cocomelon), q464 (115), q797
(Breadfast).

Failure-mode analysis (`runs/qwen3-eval/run_*.json`):

1. **"Topic zone but wrong doc" errors on queries Phase 1 solved.**
   q159: Phase 1 found gold docid 42101 at rank 1, but the agent
   answered "July 27, 2018" (vs truth "October 15, 2016") — it hit
   a topically-adjacent doc and confidently picked its date. q830
   (rank 1 in Phase 1): agent returned "Intervention" vs truth
   "Implicit Theories of Intelligence Predict Achievement" — found
   *a* psych paper, not *the* psych paper.
2. **Parsing bug** in the FINAL extractor: q625's recorded answer is
   literally `FINAL("Yoko Kanno")` — the agent's FINAL call wrapper
   was captured verbatim instead of unwrapped. Not qwen3-specific but
   surfaced by this run.
3. **Multi-hop over-compression.** q747: agent answered "Iowa" (the
   state Person A was born in, an intermediate entity in the query
   chain) instead of "Japan" (where Person A spent two years as a
   teenager, the actual final answer). The dense retriever surfaced
   topic-relevant docs but the single-vector compression plus the
   BM25-tuned decomposition prompt together lost the specificity
   needed to hop correctly.

**Root cause:** the agent's system prompt is BM25-tuned. Pre-search
decomposition, PRF, phrase/AND/OR boost intuitions, and the "verify
by name" directive all assume keyword matching. Qwen3-Embedding's
"finds the topic zone, not the exact doc" character (documented in
Phase 1 spot checks) is not what this prompt expects. The agent reads
a semantically-plausible snippet, treats it as evidence, and submits.

**This is exactly the failure mode R6 (GEPA co-optimization) was
designed to address.** Qwen3 now joins ColBERT as a retriever in need
of a co-adapted agent prompt. The R@200=24/30 ceiling is there to be
captured if the prompt uses the backend's characteristics (topic
clusters, wider k, reranking) rather than fighting them.

Next-step options (in preference order):
- **GEPA on qwen3 backend (R6).** Prompt-retriever co-optimization.
  Qwen3's 54ms latency makes this tractable where ColBERT's 26s/search
  made it prohibitive. This is the most principled fix.
- **k=50 + simple rerank (cheap agent-side co-adaptation).** Phase 1's
  R@10→R@50 spike (10→21) is a hint: the agent at k=10 sees gold only
  ⅓ of the time, but at k=50 it would see it ⅔ of the time. Bump
  `--k 10` to `--k 50` and see if giving the agent more candidates
  moves the needle.
- **Ensemble with BM25 / ColBERT (A4).** Each backend's correct set is
  small and partly disjoint; an ensemble with per-query routing could
  reach ~14/30 (the historical 3-way union) plus any qwen3-unique
  wins.
- **Fix the `FINAL(...)` unwrap regression** (unrelated to qwen3 but
  found here). Small prompt / parser change.

#### R9. Qwen3 at k=50 (agent-side cheap diagnostic)
- **What:** Re-run R8 Phase 2 with `--k 50` instead of `--k 10`. Phase 1 showed 11/30 queries have gold at rank 11-50; at k=10 the agent never sees them, at k=50 it would. This isolates the question: is the Phase 2 regression because the agent can't *see* the gold doc, or because it can't *recognize* it once seen?
- **Where:** `target/release/gw-bench --k 50 ...` pointing at the existing qwen3 search server. No code changes required.
- **Effort:** Tiny — ~90 min run, no new infra. Existing index/server/binary sufficient.
- **Expected impact:** Diagnostic, high information value. Two possible outcomes:
  - If k=50 lifts fuzzy from 4/30 toward BM25's 11/30, the problem is candidate visibility → a reranker over qwen3 top-50 is the fix (cheap + principled).
  - If k=50 doesn't help, the problem is agent recognition → GEPA co-optimization (R6) is necessary and cheap fixes won't cut it.
- **Risk:** Larger k may overwhelm the 9B model's attention (we've historically seen 25-30 docs is near the ceiling for context stuffing — noted in "Things that don't help"). Worth a look at context_snippet_chars behavior at k=50 before the run.
- **Status:** Ready to run. Should be scheduled before R6 since R6's verdict depends on this answer.

#### R7. Brute-force MaxSim recall ceiling
- **What:** Run the brute-force searcher (exact MaxSim, no approximation, no token cap) on sample30. Establishes the true recall ceiling — the number every HNSW or ANN backend is trying to approximate.
- **Where:** `retrieval_benchmark_v2.py --searchers brute_force`
- **Effort:** Small — searcher exists, ~5 min/query × 30 = ~2.5 hours
- **Expected impact:** Low directly, high indirectly — tells us whether the 25/30 from Qdrant/LanceDB MV is at the ceiling or leaving recall on the table (e.g. due to the 2000-token cap or HNSW approximation error).
- **Status:** Ready to run.

#### M1. monoT5-3B cross-encoder reranker on BM25+passage
- **What:** Replace / augment the current ColBERT-blob reranker with a T5-style cross-encoder reranker (monoT5-3B, or RankLLaMA-7B if GPU budget allows). Meng et al. (arxiv 2602.21456) report this as the single biggest lever on BrowseComp-Plus: BM25+passage alone = acc 0.572, +monoT5-3B at depth 50 = acc 0.689 (+20.5%).
- **Where:** New Python server `bench/browsecomp/rerank_server_monot5.py` mirroring the rerank_server_blobs.py HTTP contract. `gw-bench --rerank-url http://localhost:8002` already plumbed in.
- **Effort:** Medium — monoT5-3B is on HF, ~3-5 GB VRAM, ~100-300 ms/query at batch 32. One eval run at depths 10/20/50 takes ~2-3 hours.
- **Expected impact:** High. Our ColBERT-rerank regression (7/29 fuzzy vs 13/29 passage-only) is likely NOT a general "reranking hurts" result but specific to ColBERT-as-reranker + no Q2Q; Meng shows reranking helps consistently with T5-style cross-encoders. If their +20% relative holds on sample30, we'd go from ~13/29 to ~16/29.
- **Status:** Highest-EV next experiment on this branch. Pair with M2 below.

#### M2. Q2Q (query-to-question) reformulation for neural rerankers
- **What:** Small LLM call that translates the agent's keyword-style query into a natural-language question before handing it to a neural reranker (see Meng et al. §3.2.5). Two variants: Q (raw query only) and Q+R (query + recent reasoning trace). Keep the raw keyword query for BM25; use the NL version for the reranker only.
- **Where:** `crates/gw-bench/src/main.rs` — rerank dispatch currently sends the raw query to the rerank URL. Add a Q2Q LLM call using the same Ollama client we use for pre-search. Gate on a config flag `rerank_q2q_mode = "off" | "query" | "query_and_trace"`.
- **Effort:** Small-medium — ~50 LOC + one cheap LLM call per rerank invocation. Reformulator prompt is in the paper.
- **Expected impact:** Medium — Meng reports +7-10% relative accuracy for SPLADE/Qwen3-Embed and +5.7% for SPLADE+Rank1. Effect on monoT5 isn't in the paper but same mechanism should apply.
- **Status:** Blocked on M1 (need a T5-style reranker to benefit from Q2Q — BM25 is hurt by Q2Q, so we should NOT route BM25 through this).

### Passage Index Extensions (mar27 — builds on new best)

Current best: **passage + doc BM25 via RRF (12/30 fuzzy)**. These experiments build on that baseline.

#### P1. Wider passage retrieval pool
- **What:** Increase passage retrieval from k=10 to k=200, take union with doc-level top-200 before RRF. Currently both channels retrieve k results; widening the passage pool may surface more buried documents.
- **Where:** `crates/gw-memory/src/corpus.rs` — `search_with_passages()`, or make pool size configurable via CLI
- **Effort:** Small — parameter change
- **Expected impact:** Medium — more passage candidates means more chances to find buried answers. Risk: too many candidates could dilute precision (same pattern as ColBERT top-500).

#### P2. PRF on passage-level results
- **What:** Apply pseudo-relevance feedback to passage hits — extract distinctive terms from top passage snippets, use them as additional BM25 queries. Passage snippets are more focused than doc-level snippets, so PRF terms should be higher quality.
- **Where:** `crates/gw-bench/src/main.rs` — pre_search, adapted for passage results
- **Effort:** Medium — reuse existing PRF infrastructure with passage searcher
- **Expected impact:** Medium — passage PRF should produce better expansion terms than doc-level PRF

#### P3. Python client with passage index
- **What:** Add passage search support to the Python path (`ollama_client.py` / `search_server.py`). Currently passages only work on the Rust native path.
- **Where:** `bench/browsecomp/search_server.py` — add passage tantivy index, `bench/browsecomp/bm25s_searcher.py` — build passage-level bm25s index
- **Effort:** Medium — either wrap Rust passage searcher via HTTP, or build a pure-Python passage index with bm25s
- **Expected impact:** Low-medium — enables passage experiments on the faster/cheaper Python path (69K tokens vs 110K)

#### P4. Entity co-occurrence graph (BC-5)
- **What:** Regex NER over all 100K docs at index time → sparse entity-document matrix. At search time, extract entities from top-10 BM25 results, find other documents sharing those entities, add to retrieval pool.
- **Where:** `crates/gw-memory/src/corpus.rs` — new entity index, or SQLite adjacency list
- **Effort:** Large — requires corpus-wide NER + graph construction
- **Expected impact:** Medium-high — directly addresses retrieval misses where gold doc shares entities with a top-10 result but isn't in BM25's top-k itself

#### P5. Passage chunk size tuning
- **What:** Current chunks are 512 bytes with 100-byte overlap. Test different sizes: 256/50, 1024/200, 2048/256. Smaller chunks = more precise matching but more fragmentation. Larger chunks = more context but less focused.
- **Where:** `crates/gw-bench/src/main.rs` — `--build-passage-index` params
- **Effort:** Small — rebuild index with different params, re-run
- **Expected impact:** Low-medium — current 512/100 was a reasonable first guess, tuning may help

#### P6. Passage-aware pre-search
- **What:** Use passage index during pre-search (before the rLM loop). Currently pre-search uses doc-level BM25 only. Passage-level pre-search could inject more focused context snippets.
- **Where:** `crates/gw-bench/src/main.rs` — `pre_search()` function
- **Effort:** Small — swap `search_bm25_boosted` for `search_with_passages` in pre-search
- **Expected impact:** Medium — better pre-search context = better starting point for the rLM loop. The 12/30 run already benefited from passage search during the loop; extending to pre-search could compound the effect.

#### P7. Meng-style passage recipe (smaller chunks + title prepend)
- **What:** Rebuild the passage index following Meng et al. exactly: split on spaCy `en_core_web_sm` sentence boundaries into ~250-word passages (~1500 bytes, ~279 tokens), and prepend the document title to each passage when available. Our current 4096-byte chunks are ~4x theirs and split naively.
- **Where:** `crates/gw-memory/src/corpus.rs` passage builder — would need (a) sentence-boundary splitter (spaCy called from Python, or a Rust sentence splitter), (b) title extraction from the corpus JSONL (title field exists in `corpus_meta.jsonl`), (c) title-prepending at index time.
- **Effort:** Medium. Rust sentence splitter options: `rust-sent-segment`, `unicode-segmentation`, or a Python preprocessing pass emitting a passage-jsonl that Rust indexes. Pre-segmentation in Python is simplest; ~30 min rebuild after.
- **Expected impact:** Medium-high. Meng's BM25+passage (this recipe) = raw acc 0.572; our 4096-byte passage + doc RRF sits at ~0.43-0.48 fuzzy on sample30 (11-13/29). Some of that gap is likely the finer chunking + title anchor.
- **Status:** Worth doing before M1 — gives us a cleaner apples-to-apples baseline against the paper's 0.572 number.

#### P8. Hierarchical did not help — record for posterity
- **What:** Multi-chunk hierarchical index (512 + 1024 + 2048 + 4096 bytes) built apr21. 15M passages. On sample30 this regressed to 9/29 fuzzy vs 13/29 fuzzy for 4096-only. Smaller chunks have higher term density per byte and outrank the 4096s, but the narrower snippets underdetermine the answer phrase.
- **Status:** DONE, regression — keep the entry so we don't re-run it.

### Evidence Set Construction (set-level retrieval)

Thesis: for entity-heavy deep-search tasks, the core retrieval problem is not pointwise semantic matching but **evidence-set construction** — picking k docs that collectively cover the query subject to relevance. Two buckets:
- **(a) Amplification** — derive more anchored queries from anchored evidence (S7, S8; passage PRF and entity-hop also belong here).
- **(b) Composition** — pick the final k from a candidate pool (S1-S6).

Agent co-adaptation (R6/GEPA) remains a peer track, not subsumed — the R@200=24/30 → 7/30 gap was driven by latency and prompt-fit as much as by set quality.

Common design principle: widen along axes that preserve lexical anchoring (rare-term overlap, entity anchors), avoid widening along axes that relax it (pure semantic similarity, embedding-only MMR).

#### S1. MMR with lexical redundancy (composition)
- **What:** Rerank top-N with Max Marginal Relevance using rare-term Jaccard (not embedding cosine) as the redundancy term: `score(d) − λ·max_{d' in picked} jaccard_rare(d, d')`. Diversifies the final k while keeping a lexical anchor.
- **Where:** Post-processing wrapper `mmr_wrapper.py` over any Searcher, or as a stage in `search_server*.py`.
- **Effort:** Small — ~40 LOC wrapper + rare-term extraction (regex on capitalized spans).
- **Expected impact:** Medium — directly operationalizes set-level framing with a well-understood baseline. One knob (λ) to sweep.

#### S2. Soft-capped source-family dedup (composition)
- **What:** Cluster top-N candidates by MinHash on passage text or shared-rare-term Jaccard. Soft cap: ≤3 picks from any one cluster before moving on. Avoids budget drain on mirrors/near-duplicates while preserving aggregator list pages (sometimes the gold doc).
- **Where:** Same wrapper as S1.
- **Effort:** Small.
- **Expected impact:** Low-medium — targets a specific failure mode (near-duplicate drain) that hasn't been measured directly.

#### S3. Entity-cluster balance (composition)
- **What:** Extract rare capitalized spans from top-N via regex (no NER model). Cluster candidates by entity fingerprint. Require ≥2 distinct clusters in final k; do NOT force parity — the pivot entity should dominate.
- **Where:** Same wrapper as S1/S2.
- **Effort:** Small-medium.
- **Expected impact:** Low-medium — unclear whether entity diversity is load-bearing vs a proxy for source diversity already caught by S2.

#### S4. Score-band sampling (composition)
- **What:** Partition top-N into per-retriever score bands (high/mid/low). Sample k with a conservative mix like 7/2/1 for k=10. Per-retriever bands because BM25, ColBERT, and qwen3 scores live on different scales.
- **Where:** Same wrapper.
- **Effort:** Small (requires S9 normalization for cross-retriever bands).
- **Expected impact:** Low — start conservative. Widen only if retrospective analysis shows low-band docs ever contribute a gold doc.

#### S5. Query-coverage set selection (composition)
- **What:** Pre-search already decomposes queries into sub-facts. Extract target entities from each sub-fact; greedy set-cover over candidates, picking docs that each cover an uncovered sub-fact entity. Directly implements `coverage(S) s.t. relevance(S)`.
- **Where:** New stage in `pre_search` / bridge, consumes existing decomposition output.
- **Effort:** Medium.
- **Expected impact:** Medium-high — the most direct implementation of the thesis. Ceiling is decomposition quality.

#### S6. Complementary-channel union with overlap budget (composition)
- **What:** Build k=10 by pulling top-m from each of {BM25+boosts, blob rerank, qwen3, optionally ColBERT} with a capped overlap (e.g. ≤2 shared docids before backfilling from lower ranks). Current 3-channel union ceiling is 14/30 — this tests how much is reachable without an oracle.
- **Where:** New `union_searcher.py` wrapping multiple Searchers.
- **Effort:** Small.
- **Expected impact:** Medium — prior ensembling (best-of-N, majority voting) had limited gains, but those operated on *answers* (post-agent). This operates on *retrieval sets* (pre-agent) and is a cheaper, earlier intervention.
- **Caveat:** Ensembling hasn't historically paid off here. Worth trying because the operating layer is different, but temper expectations.

#### S7. Gold-support expansion from passages (amplification)
- **What:** When a top passage looks strong — confidence gate: rank-1/rank-5 score margin above threshold, OR doc-channel and passage-channel agree on the same docid — extract rare terms + capitalized spans, run a second-hop BM25 search, add top corroborating docs. PRF anchored to rare terms rather than distinctiveness scores.
- **Where:** `crates/gw-bench/src/main.rs` `pre_search` — new stage after round-1.
- **Effort:** Medium.
- **Expected impact:** Medium-high — gated expansion avoids compounding wrong anchors, which was the failure mode of prior unanchored PRF variants.

#### S8. Anchor-preserving expansion filter (amplification)
- **What:** Any doc added via any expansion path (PRF, entity-hop, query expansion) must contain ≥1 rare term from the *original* query. Enforces "widen along lexical-preserving axes." Lightweight filter, not a new retrieval stage.
- **Where:** Wrapper over existing PRF and any future expansion in `pre_search`.
- **Effort:** Tiny.
- **Expected impact:** Low-medium — mostly a safety rail; may recover lost precision from existing PRF.

#### S9. Cross-retriever score normalization (infrastructure)
- **What:** Normalize BM25 and dense scores to [0,1] via `(2/π)·arctan(bm25/k)` (or similar compressive transform) so that cross-retriever score bands (S4) and weighted union (S6) are well-defined. Currently we RRF everything, which discards score magnitude entirely.
- **Where:** Shared utility in `searchers/base.py`.
- **Effort:** Small.
- **Expected impact:** Unknown on its own — enables S4 and weighted variants of S6. Only worth doing if those show promise.
- **Status:** Pointer, not a standalone priority. Noted because the normalization has precedent (arctan-compressed BM25 alongside cosine scores).

#### S10. Oracle set scorer (diagnostic ceiling)
- **What:** For each query, generate several candidate k-sets (pointwise top-k, MMR, set-cover, union). Run a single LLM pass per set: "does this collectively support answering the query?" Pick the best. Measures the ceiling of set-level framing under current retrieval.
- **Where:** Offline harness, not in the agent loop.
- **Effort:** Medium.
- **Expected impact:** Diagnostic — if oracle lifts agent accuracy over pointwise top-k, set framing is validated. If not, pointwise is near the ceiling and set-level gains will be marginal. Either way it bounds the bucket.

#### S11. Failure-case ideal-set design (selector specification)
- **What:** Inverse of S10. Take 5-10 concrete failure cases. For each, hand-design the ideal evidence set that would have supported the correct answer (gold doc + k-1 corroborators). Then reverse-engineer: what feature function over the candidate pool would have ranked the ideal set above what we actually selected? Extract selection rules; if they generalize, they become the spec for S1/S2/S5/S6 heuristics (or a learned selector).
- **Where:** Offline notebook / scratch harness. Failure set drawn from the queries Qdrant/BM25/qwen3 all miss (complement of the 14/30 union). Gold docs known, so "ideal set" construction is semi-automatic: seed with gold doc, add corroborators that share rare terms / entity anchors.
- **Pre-check:** For each failure, verify the gold doc is in top-200 of at least one channel. If not, it's a recovery problem (bucket a), not composition (bucket b) — skip for this experiment.
- **Anti-bias protocol:** Write selection rules *before* looking at what any existing selector returned for that query, to avoid post-hoc rationalization.
- **Effort:** Medium — mostly analyst time, 30-60 min per case.
- **Expected impact:** High as a design tool (produces concrete requirements for S1/S5/S6), low as a direct accuracy lift (the selector has to be built separately). Pair with S10: S10 bounds the ceiling, S11 specifies how to reach it.
- **Caveat:** Small N. 5-10 cases is enough to derive rules but not enough to learn a parametric selector. If rules don't generalize across cases, that itself is a finding: per-query routing (A4 ensemble) may beat a universal selector.

#### S12. Entity-linker ColBERT as auxiliary signal
- **What:** Small ColBERT index built over a *curated vocabulary* — canonical entity strings extracted from the corpus (regex NER + light normalization for aliases/abbreviations), optionally joined with gw-kb topic labels. Used as an **entity-linking / normalization layer**, not as a primary retriever: given a query or a candidate doc, return the set of canonical entity IDs it mentions, with fuzzy tolerance for surface-form variation.
- **Why (thesis connection):** most set-construction signals currently assume we can tell whether a doc "contains" a query entity. Right now that's regex exact match — brittle for "FDR" vs "Franklin D. Roosevelt", "US" vs "United States", etc. Token-level MaxSim over canonical entity strings is a fuzzy-but-anchored match: it tolerates surface-form variation without widening to generic semantic similarity (which would relax lexical anchoring by design). Scale is tractable — ~500K entities × ~3 tokens ≈ 1.5M vectors, no PLAID/WARP needed.
- **Consumers:**
  - **S5 (query-coverage)**: match sub-fact entities from pre-search decomposition against candidate docs via canonical ID, not surface string.
  - **S7 (gold-support expansion)**: canonicalize entities in a strong passage before the second-hop search, so corroborator queries aren't spelling-dependent.
  - **S8 (anchor-preserving filter)**: the "≥1 rare term from original query" check becomes "≥1 canonical entity from original query," which is more robust.
- **Where:** New `entity_linker.py` (offline extraction + canonicalization + ColBERT index), consumed by the S5/S7/S8 wrappers.
- **Effort:** Medium — entity extraction + canonicalization is the bulk of the work; the ColBERT index itself is small and can reuse our existing candle encoder.
- **Expected impact:** Medium as a *robustness multiplier* for S5/S7/S8. Not a standalone accuracy lift — it makes existing signals more reliable rather than adding a new channel.
- **Non-goal:** Not a first-stage retriever, not a KG substitute. See Related Work for why the curated-vocabulary-as-first-stage framing doesn't match our bottleneck.

---

### gw-kb Topic Graph Eval (apr8-apr10)

We built a full gw-kb eval harness (HTTP serve → GwKbSearcher → ollama_client → quick_eval)
and ran three ablation variants against sample30 with qwen3.5:9b.

**Infrastructure built:**
- `gw-kb serve` axum subcommand (search/topic/topics/explore/healthz endpoints)
- `gw_kb_searcher.py` with ablation-gated `repl_extras()` (kb_topic, kb_topics, kb_explore)
- `--system-prompt-file` flag on ollama_client.py + three system prompt variants
- `run_gw_kb.sh` + `run_gw_kb_full_pipeline.sh` (sidecar orchestration)
- `--max-per-topic` fan-out cap on `gw-kb link` (bounds dense-corpus edge explosion)
- Lenient tantivy query parsing (apostrophes in queries caused 500 errors)

**KB state for eval:** 2374 sources, 42755 chunks, 40553 tagged, 4075 topics, 83694 edges (top-25/topic), 2687 synthesized summaries.

**Results:**

| Variant | Exact | Fuzzy | LLM Judge | Gold doc recall |
|---------|-------|-------|-----------|-----------------|
| search-only | 6/30 (20%) | 6/30 (20%) | 7/30 (23%) | 21/30 (70%) |
| +topic | 4/30 (13%) | 4/30 (13%) | 5/30 (17%) | 16/30 (53%) |
| full | 2/30 (7%) | 2/30 (7%) | 3/30 (10%) | 12/30 (40%) |
| *BM25s baseline* | *—* | *12/30 (40%)* | *—* | *—* |

**Topic tools hurt accuracy.** Each additional tool layer reduced both retrieval recall and answer rate. The agent entered explore/topic browsing loops instead of reading documents.

**Key diagnostic data (search-only variant):**

| Metric | Count |
|--------|-------|
| Gold doc found in search results | 21/30 |
| Gold doc found AND read via get_document | 4/30 |
| Gold doc found but NOT read | 17/30 |
| Correct answers | 6/30 |

**The conversion funnel is: find → read → extract → answer.** The biggest drop is find → read: the agent sees the gold doc in snippets but doesn't call `get_document()` on it in 17/30 cases. This is a prompt problem, not a retrieval problem.

**Root causes identified:**
1. **Apostrophe crash** — tantivy's strict query parser crashed on `'` (People's, President's). Fixed with `parse_query_lenient`. Affected 3 queries per variant.
2. **Topic tools are a turn sink** — in `full`, q853 spent 36 kb_topic + 17 kb_explore calls but only 4 searches. search-only solved q853 with 11 searches.
3. **Agent doesn't read documents** — 14/30 queries used get_document at all; only 4/30 read a gold doc. The snippet-only workflow loses information.
4. **Topic summaries contain answers but agent doesn't extract them** — q853's management-science summary mentioned "Richard Larson" explicitly, but the agent kept exploring instead of submitting.
5. **gw-kb hybrid search underperforms BM25s baseline** — topic-membership RRF signal may add noise at this corpus scale (4075 topics, many weakly clustered).

---

## gw-kb Experiments (apr10)

### KB1. Re-run search-only with lenient parse fix
- **What:** The apostrophe fix (`parse_query_lenient`) was applied after the eval runs. Re-run search-only to measure the isolated impact.
- **Where:** `bench/browsecomp/run_gw_kb.sh search-only`
- **Effort:** Tiny — just re-run
- **Expected impact:** +1-3 queries — 3 queries per variant had 500 errors from apostrophes (q237, q853, q893). q853 was answered correctly in search-only despite the errors; the other two may flip.

### KB2. Document-reading prompt
- **What:** The #1 conversion bottleneck is find → read (17/30 gold docs found but not read). Redesign the search-only prompt to mandate `get_document()` on the top hit of every search, with explicit `llm_query()` extraction. Model the prompt on the existing doc-grounded prompt from the Rust path (see "Things that help" above).
- **Where:** New `bench/browsecomp/prompts/gw_kb_doc_grounded.txt`
- **Effort:** Small — prompt change only
- **Expected impact:** High — if even half of the 17 "found but not read" queries convert, that's +4-8 correct answers, potentially beating the 12/30 baseline. The Rust path already proved doc-grounded prompts help.

### KB3. BM25-only search (disable vector + topic-membership signals)
- **What:** gw-kb hybrid search fuses BM25 + vector + topic-membership via RRF. The BM25s baseline uses pure BM25 and gets 12/30. Test whether the vector and topic-membership signals are adding noise. Modify `hybrid_search` to expose a `bm25_only` flag, or have the server endpoint accept a `signals` parameter.
- **Where:** `crates/gw-kb/src/search.rs` — add flag to skip vector/topic passes
- **Effort:** Small — conditional skip in hybrid_search
- **Expected impact:** Medium-high — if BM25-only gw-kb matches the BM25s baseline (12/30), the hybrid signals are the problem. If it's still lower, the tantivy index configuration differs (field boosts, tokenizer).

### KB4. Topic as late-stage verifier (not explorer)
- **What:** Restructure the prompt so topic tools are used ONLY after the agent has a candidate answer (turns 8+), to verify or disambiguate — not for initial exploration. Hard-cap: max 3 kb_topic + 1 kb_explore per query. Implement via a wrapper that raises after the budget is exhausted.
- **Where:** New prompt `bench/browsecomp/prompts/gw_kb_verify.txt` + budget enforcement in `gw_kb_searcher.py`
- **Effort:** Small-medium — prompt + ~15 LOC wrapper
- **Expected impact:** Medium — prevents the topic-browsing death spiral while preserving the case where summaries genuinely help (like q853's management-science summary mentioning Richard Larson). Addresses the core finding that topic summaries contain answers but the agent treats them as exploration rather than evidence.

### KB5. Topic-seeded query expansion
- **What:** Instead of exposing topic tools to the agent, use them server-side: before returning search results, run the query through `kb_explore`, read the top-3 topic summaries, extract named entities, and inject them as additional BM25 search terms. The agent never sees the topic tools — it just gets better search results.
- **Where:** New endpoint `POST /search_expanded` in `crates/gw-kb/src/server.rs`, or a pre-processing step in `gw_kb_searcher.py`
- **Effort:** Medium — ~50 LOC server-side, or ~30 LOC Python wrapper
- **Expected impact:** Medium — leverages the topic graph's unique contribution (cross-source synthesis) without the turn-sink problem. If management-science's summary mentions "Richard Larson", that name becomes a search term automatically.

### KB6. Increase turn budget for topic variants
- **What:** The current 12-turn budget was tuned for search-only. Topic variants need more turns to accommodate exploration + reading. Test 16 and 20 turns for the +topic and full variants.
- **Where:** `bench/browsecomp/run_gw_kb.sh` — `--max-turns` per variant
- **Effort:** Tiny — parameter change
- **Expected impact:** Low-medium — EXPERIMENTS.md notes that 16 turns hurt on the BM25 baseline ("model second-guesses itself"), but the topic variants are turn-starved (20-27 avg turns used vs 12-turn cap with sub-iterations). Worth testing whether the more-turns penalty is offset by more time to converge after exploration.

### KB7. Pre-inject topic summaries into system prompt
- **What:** Eliminate runtime topic browsing entirely. At query time, run `kb_explore(query, k=5)` server-side, fetch the top-5 topic summaries, and prepend them to the system prompt as "Background context." The agent gets topic knowledge for free (no turn cost) and focuses entirely on search + document reading.
- **Where:** Pre-processing in `gw_kb_searcher.py` or a new `--pre-inject-topics` flag
- **Effort:** Small-medium — ~30 LOC in searcher
- **Expected impact:** Medium-high — removes the turn-sink problem entirely while still surfacing topic knowledge. If topics help, they help as context; if they don't, the agent ignores them (like existing system prompt instructions). The risk is prompt length — 5 summaries × 300 words = ~1500 tokens of preamble, which may hurt the 9B model's attention.

### KB8. Tantivy index parity check
- **What:** Verify that gw-kb's tantivy index produces the same BM25 results as the BM25s baseline index for identical queries. The baseline uses bm25s (Python) with its own tokenizer and scoring. If the indexes differ substantially, that explains the 12/30 → 6/30 regression before any hybrid signal is involved.
- **Where:** Run 10 sample queries through both indexes, compare top-10 docid overlap
- **Effort:** Small — scripted comparison
- **Expected impact:** Diagnostic — if overlap is low, the fix is index configuration (tokenizer, stopwords, field boosts). If overlap is high, the hybrid signals are the confirmed culprit.

---

## Priority Ranking (updated apr24 post-Meng)

Six independent threads:
1. **Cross-encoder reranking (new, evidence from Meng et al.)** — the single biggest lever reported on BrowseComp-Plus (+20.5% acc with BM25+monoT5-3B at depth 50). Our prior ColBERT-rerank regression is likely specific to that reranker, not "reranking in general." M1 (monoT5 rerank) + M2 (Q2Q reformulation) together represent the most concrete unexplored direction with external evidence.
2. **Evidence set construction / coverage** — S5-lite (coverage pre-search) is now implemented and is a complementary channel: 12/29 fuzzy alone, 4-way union with passage runs = 20/30. S6 (channel union) is the remaining unbuilt composition experiment.
3. **Dense retrieval** — Qwen3-Embedding-0.6B Phase 1 hit R@200=24/30 at 54ms p50. Phase 2 (agent eval) regressed to 4/30 fuzzy; R9 at k=50 confirmed agent can't recognize gold even when seen. Only recoverable via co-adapted prompt (R6/GEPA).
4. **Ensemble/routing** — three-channel union = 14/30, four-channel (+ S5-lite) = 20/30. Needs a per-query pick-the-right-answer heuristic that avoids prior majority-vote failure (wrong answers agreed more).
5. **gw-kb conversion funnel** — 70% gold-doc recall but 20% accuracy. Agent finds docs but doesn't read them.
6. **Recipe/infra audit (new, from Meng)** — their passage recipe (250-word spaCy chunks + title prepend, P7) and their BM25 doc params (`k1=3.8, b=0.87`, B4) are cheap pre-pipeline tweaks with published numbers.

| Priority | Experiment | Expected Impact | Effort | Rationale |
|----------|-----------|----------------|--------|-----------|
| **1** | M1. monoT5-3B rerank on BM25+passage | High (+20% acc in paper) | Medium | Meng et al. report this as the single biggest lever on BrowseComp-Plus. Our ColBERT-rerank regression was reranker-specific, not a general result; a T5 cross-encoder is an untested, evidence-backed lever. |
| **2** | M2. Q2Q reformulation for neural rerankers | Medium (+5-10% rel) | Small-medium | Paper's Table 11/13: mitigates the training–inference query mismatch that hurt our neural-rerank attempts. Blocked by M1 (only applies when a neural reranker is in the loop). |
| **3** | P7. Meng-style passages (250 words, spaCy, title prepend) | Medium-high | Medium | Their 0.572 baseline = BM25 on this exact recipe. Our 4096-byte chunks are ~4x theirs. Direct recipe replication. |
| **4** | A4 / S6. Four-channel routing (pass-run1 + pass-run2 + S5-lite + S5-filter) | High | Small-medium | 4-way union reaches 20/30 fuzzy on sample30. Per-query pick-right heuristic is the last-mile to operationalize. |
| **5** | S1. MMR with lexical redundancy | Medium | Small | Classical diverse selection, reformulated to preserve lexical anchoring (rare-term Jaccard, not embedding cosine). Different diversity axis from S5. |
| **6** | S7. Gold-support expansion from passages | Medium-high | Medium | Confidence-gated rare-term PRF. Amplification bucket; compounds naturally with S6 composition. |
| **7** | B4. BM25 param tuning on document corpus (k1=3.8, b=0.87) | Low-medium | Tiny | Paper shows default = worst on docs. Doc channel is half of our RRF. Cheap probe. |
| **8** | KB2. Doc-reading prompt | High | Small | The #1 bottleneck in gw-kb: 17/30 gold docs found but never read. Doc-grounded prompts already proven on Rust path. |
| **9** | R6. GEPA co-optimization on qwen3 | High | Medium | Only way to recover the Qwen3 R@200=24/30 ceiling. Deferred pending cheaper wins. |
| ~~done~~ | ~~R8. Qwen3-Embedding (Phase 2: agent eval)~~ | | | **Done apr18 — fuzzy=4/30 (regression vs BM25 11/30). Retriever-agent coupling fails; R9/R6 are the diagnostics.** |
| **4** | S6. Complementary-channel union w/ overlap budget | Medium | Small | Cheapest test of the set-construction thesis. Operates on retrieval sets (pre-agent), unlike prior post-agent ensembling that failed. Targets the 14/30 union ceiling. |
| **5** | S1. MMR with lexical redundancy | Medium | Small | Classical diverse selection, reformulated to preserve lexical anchoring (rare-term Jaccard, not embedding cosine). Pairs well with S6. |
| **6** | S5. Query-coverage set selection | Medium-high | Medium | Most direct implementation of `coverage(S) s.t. relevance(S)`. Leverages existing pre-search decomposition. |
| **7** | S7. Gold-support expansion from passages | Medium-high | Medium | Confidence-gated rare-term PRF. Amplification bucket; compounds naturally with S6 composition. |
| **8** | KB2. Doc-reading prompt | High | Small | The #1 bottleneck: 17/30 gold docs found but never read. Doc-grounded prompts already proven on Rust path. |
| **10** | KB8. Tantivy index parity check | Diagnostic | Small | Must know if gw-kb's tantivy produces same results as BM25s before diagnosing further. |
| **11** | KB3. BM25-only search | Medium-High | Small | Isolates whether hybrid signals (vector + topic-membership) hurt. |
| **12** | S11. Failure-case ideal-set design | Design tool | Medium | Hand-designs ideal evidence sets from failures; reverse-engineers selector requirements. Specifies the heuristics S1/S5/S6 should implement. Pair with S10. |
| **13** | S10. Oracle set scorer | Diagnostic | Medium | Bounds the set-construction bucket. S11 specifies how to reach the ceiling; S10 measures where it is. |
| **14** | S2. Soft-capped source-family dedup | Low-medium | Small | Quick composition win if near-duplicate drain is real. |
| **15** | S8. Anchor-preserving expansion filter | Low-medium | Tiny | Safety rail over existing PRF; low risk. |
| **16** | S12. Entity-linker ColBERT (auxiliary) | Medium | Medium | Robustness multiplier for S5/S7/S8 — canonicalizes entity anchors so set-construction signals tolerate surface-form variation without relaxing lexical anchoring. Not a standalone retriever. |
| **17** | R7. Brute-force recall ceiling | Low→High | Small | 2.5-hour one-time run. Tells us if 25/30 is the ceiling or if the 2000-token cap is costing us. |
| **18** | R5. LanceDB MV agent eval | Medium | Small | Isolates Qdrant token-cap effect. No server, no token cap. |
| **19** | KB7. Pre-inject topic summaries | Medium-High | Small | Removes turn-sink, surfaces topic knowledge for free. |
| **20** | S3. Entity-cluster balance | Low-medium | Small-medium | May be subsumed by S2; worth running only if S2 underdelivers. |
| **21** | S4. Score-band sampling | Low | Small | Needs S9 normalization to work across retrievers. Conservative 7/2/1 start. |
| **22** | KB5. Topic-seeded query expansion | Medium | Medium | Server-side topic leverage. |
| **23** | B3. Query expansion + reranking | Medium | Medium | LLM query expansion widens BM25 recall for reranking. |
| **24** | R3. SPLADE | Medium-High | Large | May be redundant now that dense + ColBERT both work |
| **25** | R8b. Qwen3-Embedding 4B/8B escalation | Unknown | Medium | If Phase 2 0.6B validates, test whether the larger model closes more misses. |
| **26** | S9. Score normalization (arctan) | Enables others | Small | Infrastructure for S4 and weighted S6. Only worth doing if either shows promise. |
| ~~done~~ | ~~R9. Qwen3 at k=50 diagnostic~~ | | | **Done apr19 — 4/29 exact, same as k=10. Agent can't recognize gold even with 5x candidates. Visibility not the bottleneck.** |
| ~~done~~ | ~~S5-lite. Coverage pre-search~~ | | | **Done apr24 — 12/29 fuzzy on sample30, complementary channel (wins q1106, q625 no other config gets). 4-way union = 20/30.** |
| ~~done~~ | ~~R8. Qwen3-Embedding (Phase 1: R@k)~~ | | | **Done apr17 — R@200=24/30 (vs BM25 12/30, ColBERT 25/30) at 54ms p50. Hypothesis confirmed.** |
| ~~done~~ | ~~R4. BM25 + blob rerank~~ | | | **Done — 7/30 (+2 over 5/30 baseline). Complementary to Qdrant. Union=14/30.** |
| ~~done~~ | ~~KB eval harness~~ | | | **Done — 3 ablations, all below baseline** |
| ~~done~~ | ~~R2b. ColBERT first-stage~~ | | | **Done — 25/30 R@200, 7/30 agent accuracy** |
| ~~done~~ | ~~R2. ColBERT reranker~~ | | | **Done — 12/30 (40%), historical best** |
| ~~done~~ | ~~B1. Fuzzy matching~~ | | | **Done — regression** |
| ~~done~~ | ~~A2. Answer type~~ | | | **Done — no benefit** |
| ~~skip~~ | ~~A3. Larger model~~ | | | Deprioritized per user preference |
| ~~skip~~ | ~~R1. Different embeddings~~ | | | Dense vector search doesn't help |

---

## Related Work / Research Bookmarks

Not on the priority list, but worth knowing about so we don't re-discover them. These are potential research detours if the main threads stall; park them here.

### Papers directly relevant to BrowseComp-Plus

- **Meng, Ou, MacAvaney, Dalton — `Revisiting Text Ranking in Deep Research`** (arXiv:2602.21456v1, Feb 2026). Summarized inline in the findings section above. Key levers: BM25+passage+monoT5-3B (their best single pipeline, acc 0.689) and Q2Q reformulation (+5-10% on neural rankers). Planned experiments M1 and M2 were added based on this paper; P7 (passage recipe) and B4 (doc BM25 params) replicate their recipe details.

### Fast late-interaction retrieval (WARP family)

Both of these target the gap we hit with Qdrant MV (26s) and LanceDB MV (46s): a native ColBERT-class retriever that's agent-viable. We sidestepped the problem via blob-rerank (200ms over BM25 top-N) and qwen3-embed (54ms p50, single-vector). Parked because qwen3-embed already occupies the high-recall + low-latency slot and BM25+set-construction is the current research thesis.

- **WARP** (SIGIR'25, `jlscheerer/xtr-warp`) — first-stage engine built on Google's XTR (T5-based) + ColBERTv2/PLAID. Three techniques:
  - **Centroid path**: k-means over corpus tokens (~√N centroids); each doc token stored as `(centroid_id, 2-bit residual)`. Query tokens retrieve candidates from nearest centroids' member sets.
  - **Implicit decompression**: MaxSim score rewritten as `q · centroid[id] + q · residual` — first term precomputed per query, second is a dot product against the 2-bit residual. Never materializes full token vectors.
  - **WARP_SELECT**: imputes similarities for unvisited centroids from a global prior instead of computing against all centroids. Avoids `O(Q · num_centroids)` per query.
  - **Reported**: 3x faster than PLAID, 41x faster than XTR reference.

- **Witchcraft** (`dropbox/witchcraft`) — Rust reimplementation of WARP backed by a single SQLite file. 21ms p95 on NFCorpus (M2 Max), ~2x faster than WARP itself. Hybrid BM25+semantic in one store via SQLite FTS. Single-file deployment, no server. File layout maps cleanly to the algorithm: `packops.rs` (residual packing), `rans64.rs` (entropy-coded residuals), `fast_ops.rs` / `fused_matmul.rs` (SIMD inner products), `merger.rs` (score aggregation). Encoder lives in `quantized_t5.rs` / `openvino_t5.rs` / `embedder.rs` and is the only T5/XTR-specific surface.

### Porting Reason-ModernColBERT into a WARP-style index

*Infrastructure (moderate, ~1-2 weeks):*
- Swap encoder: replace Witchcraft's `quantized_t5.rs` with a candle ModernBERT encoder. `crates/gw-memory/src/colbert/candle_encoder.rs` is already parity-tested and is the natural starting point.
- Verify output dim (standard ColBERT = 128; should match). If different, rebuild rANS codebook widths.
- k-means over our existing blob-store token tensors (122 GB, already computed) → centroids. ~1 hour CPU.
- Requantize residuals, pack into Witchcraft's SQLite format.
- The encoder-agnostic kernels (`packops`, `fast_ops`, `fused_matmul`, `merger`, `rans64`) work unchanged.

*Quality (unknown, the real risk):*
- WARP's approximations were calibrated around XTR's training objective, which deliberately makes individual token matches self-contained (each token is a meaningful retrieval signal on its own). Standard ColBERT — including Reason-ModernColBERT — doesn't have that property; it relies on MaxSim over *all* query tokens.
- PLAID on ColBERTv2 shows residual compression loss is small, so the centroid + decompression half should transfer cleanly. WARP_SELECT's imputation half is the open question — centroid-skipping may lose more recall on ColBERT than on XTR.
- Only way to know is to build and measure.

**Bookmark status:** not on the critical path. Revisit if (a) the set-construction thesis plateaus below 18-20/30, and (b) we have reason to believe an additional ColBERT-quality channel (beyond blob-rerank) would unlock union-ceiling gains that qwen3-embed can't.

### ColBERT over a curated vocabulary (as first-stage)

An alternative route to fast late-interaction: sidestep the scale problem entirely by encoding only a *curated token set* — topic labels from gw-kb, or canonical entity strings from NER — rather than every token in every doc. This collapses the index by 3-5 orders of magnitude (gw-kb has ~4075 topics; corpus-wide entity extraction would yield ~500K canonical entities) and makes brute-force MaxSim tractable without PLAID/WARP tricks.

**Why this doesn't fit as a first-stage retriever for BrowseComp:**

- **Regime shift**: this is no longer full-text late interaction, it's **entity/topic linking** that resolves to documents via graph edges (gw-kb's topic-doc edges, or co-occurrence edges from NER). Different technique with different failure modes, not a drop-in replacement for BM25 or qwen3-embed.
- **Empirical evidence**: gw-kb's existing topic path already underperforms BM25s on BrowseComp (2-6/30 vs 12/30 baseline). The documented bottleneck wasn't fuzzy topic matching — it was the agent not reading documents and topic summaries not being in the retrieval path the right way. Better topic matching doesn't fix either.
- **Late interaction degrades on short strings**: MaxSim's advantage is token-level matching over long doc contexts. Matching a query against a 5-8 token topic label is close to single-vector cosine on strings — sentence-transformers do this cheaply.
- **Still need passage-level retrieval downstream**: BrowseComp answers live in passages, not topic summaries. A topic router finds the cluster, then has to dump member docs back into BM25 anyway.

**Why it isn't a KG substitute either:**

You'd get soft entity resolution (good — "FDR" ↔ "Franklin D. Roosevelt") and topic routing (okay — gw-kb already has this). You'd not get typed relations, multi-hop traversal via typed edges, or structural inference. The hardest BrowseComp failures (q747's multi-hop chain) specifically need relational composition, which a fuzzy entity matcher doesn't provide.

**Where the idea does earn its keep:** as an *auxiliary entity-linker* feeding the set-construction experiments (S5/S7/S8), not as a standalone retriever. That version is captured as **S12** in the Evidence Set Construction section.

**Bookmark status:** the curated-vocabulary first-stage framing is parked here. The useful slot (auxiliary entity linker) is on the priority list.
