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

### Key insight
86% of failures are **retrieval** (right documents never found), not extraction. The 9B model reasons well once it has the right documents. Dense vector search doesn't help — but ColBERT reranking of a wider BM25 pool does, because the right documents are often in BM25's top-200 but ranked too low for the agent to see.

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

#### R1. Different embedding model — Deprioritized
- **Status:** Dense vector search doesn't help on this benchmark regardless of embedding model.

#### R2. ColBERT reranker on BM25 results — DONE, new best
- **Result:** BM25 top-200 → Reason-ModernColBERT rerank → top-10 achieves **12/30 (40%)**, new all-time best.
- **Implementation:** Python sidecar (`rerank_server.py`) called via HTTP from Rust native backend.
- **Key finding:** Sweet spot is top-200. Top-50 too narrow (7/30), top-500 too noisy (9/30).
- **See:** ColBERT Reranking Findings section above for full details.

#### R2b. Reason-ModernColBERT as first-stage retriever
- **What:** Build a multi-vector ColBERT index over 100K docs. Use ColBERT as the primary retriever (not just reranker), potentially finding documents BM25 misses entirely.
- **Effort:** Large — requires multi-vector index build (hours), RAGatouille/NextPlaid integration
- **Expected impact:** High — addresses the fundamental retrieval gap (docs not in BM25 top-200 at all)
- **Status:** Design doc at `DESIGN-reason-colbert.md`. R2 (reranker) results suggest ColBERT quality is good — the question is whether first-stage retrieval finds different docs than BM25.

#### R3. Learned sparse retrieval (SPLADE-style)
- **What:** Use a learned sparse model that outputs weighted term expansions. Combines BM25-style matching with learned term importance.
- **Effort:** Large — needs model + index infrastructure
- **Expected impact:** Medium-high — addresses vocabulary mismatch without the noise of dense vectors
- **Status:** Research needed

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

---

## Priority Ranking (updated mar21)

| Priority | Experiment | Expected Impact | Effort | Rationale |
|----------|-----------|----------------|--------|-----------|
| **1** | R2b. ColBERT first-stage retriever | High | Large | Only way to find docs BM25 misses entirely; reranker validated ColBERT quality |
| **2** | A5. Harness reliability fixes | Low | Small | Quick wins — may recover 1-2 answers from edge cases |
| **3** | A4. Ensemble | High | Medium | Guaranteed from known correct set diversity |
| **4** | B3. Query expansion + reranking | Medium | Medium | LLM query expansion widens BM25 recall, ColBERT reranking cleans it up |
| **5** | A1. Search budget | Low-Medium | Medium | Addresses under-searching |
| **6** | R3. SPLADE | Medium-High | Large | Best-of-both-worlds retrieval |
| ~~done~~ | ~~R2. ColBERT reranker~~ | ~~Medium-High~~ | ~~Medium~~ | **Done — 12/30 (40%), new best** |
| ~~done~~ | ~~B1. Fuzzy matching~~ | ~~Low~~ | ~~Small~~ | **Done — regression, noise hurts** |
| ~~done~~ | ~~A2. Answer type~~ | ~~Low-Medium~~ | ~~Low~~ | **Done — no benefit** |
| ~~skip~~ | ~~A3. Larger model~~ | ~~High~~ | ~~Small~~ | Deprioritized per user preference |
| ~~skip~~ | ~~R1. Different embeddings~~ | ~~Unknown~~ | ~~Medium~~ | Dense vector search doesn't help |
| ~~skip~~ | ~~B2. Field boosting~~ | ~~Medium~~ | ~~Large~~ | ColBERT reranking addresses ranking more directly |
