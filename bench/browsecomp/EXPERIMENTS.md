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

#### R2b. Reason-ModernColBERT as first-stage retriever — DONE
- **Result:** ColBERT first-stage retrieval reaches 25/30 R@200 (vs BM25 12/30). But end-to-end agent accuracy regressed to 7/30 (vs BM25 9/30). See "ColBERT First-Stage Retrieval" findings above.
- **Status:** Infrastructure is complete (4 backends, blob store, Rust encoder, benchmark harness). The bottleneck has shifted from retrieval recall to agent co-adaptation. See R4-R7 below.

#### R3. Learned sparse retrieval (SPLADE-style)
- **What:** Use a learned sparse model that outputs weighted term expansions. Combines BM25-style matching with learned term importance.
- **Effort:** Large — needs model + index infrastructure
- **Expected impact:** Medium-high — addresses vocabulary mismatch without the noise of dense vectors
- **Status:** Research needed

#### R4. BM25 + blob rerank (fast ColBERT rerank, no server)
- **What:** BM25 first stage (fast, <1s) → blob store MaxSim rerank (200ms) → top-K. Uses the existing passage blob store — no Qdrant, no ES, no HNSW. This is the "Option 1" from `design-colbert-production.md`.
- **Where:** `search_server.py` — add a `--blob-store` flag, wrap BM25 candidates with `BlobRerankWrapper`
- **Effort:** Small — the wrapper and blob store already exist
- **Expected impact:** High — fixes the latency problem (200ms vs 26s), stays within the agent's 180s timeout. R@200 is capped at BM25's 12/30 but head precision improves (R@5: 4→7 measured). The key test: does faster rerank give the agent enough time to reason, recovering the 9/30 baseline while adding rerank quality?
- **Status:** Ready to run. This is the highest-priority experiment.

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

#### R7. Brute-force MaxSim recall ceiling
- **What:** Run the brute-force searcher (exact MaxSim, no approximation, no token cap) on sample30. Establishes the true recall ceiling — the number every HNSW or ANN backend is trying to approximate.
- **Where:** `retrieval_benchmark_v2.py --searchers brute_force`
- **Effort:** Small — searcher exists, ~5 min/query × 30 = ~2.5 hours
- **Expected impact:** Low directly, high indirectly — tells us whether the 25/30 from Qdrant/LanceDB MV is at the ceiling or leaving recall on the table (e.g. due to the 2000-token cap or HNSW approximation error).
- **Status:** Ready to run.

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

## Priority Ranking (updated apr10)

The bottleneck has shifted from **retrieval recall** (solved: 25/30 R@200)
to **agent co-adaptation** (end-to-end accuracy regressed with ColBERT).
Priorities are reordered accordingly.

| Priority | Experiment | Expected Impact | Effort | Rationale |
|----------|-----------|----------------|--------|-----------|
| **1** | R4. BM25 + blob rerank | High | Small | Fastest path to ColBERT rerank under the agent's timeout budget (200ms vs 26s). All infrastructure exists. |
| **2** | R7. Brute-force recall ceiling | Low→High | Small | 2.5-hour one-time run. Tells us if 25/30 is the ceiling or if the 2000-token cap is costing us. |
| **3** | R6. GEPA co-optimization | High | Medium | The prompt-retriever coupling is the most likely cause of the 7/30 regression. Needs R4 first. |
| **4** | R5. LanceDB MV agent eval | Medium | Small | Isolates the Qdrant token-cap effect. No server, no token cap. Needs lance IVF index. |
| **5** | A5. Harness reliability fixes | Low | Small | Quick wins — may recover 1-2 answers from edge cases |
| **6** | A4. Ensemble | High | Medium | Guaranteed from known correct set diversity |
| **7** | B3. Query expansion + reranking | Medium | Medium | LLM query expansion widens BM25 recall, ColBERT reranking cleans it up |
| **8** | R3. SPLADE | Medium-High | Large | Best-of-both-worlds retrieval — may be redundant now that ColBERT retrieval works |
| ~~done~~ | ~~R2b. ColBERT first-stage~~ | | | **Done — 25/30 R@200, but 7/30 agent accuracy (regression)** |
| ~~done~~ | ~~R2. ColBERT reranker~~ | | | **Done — 12/30 (40%), previous best** |
| ~~done~~ | ~~B1. Fuzzy matching~~ | | | **Done — regression, noise hurts** |
| ~~done~~ | ~~A2. Answer type~~ | | | **Done — no benefit** |
| ~~skip~~ | ~~A3. Larger model~~ | | | Deprioritized per user preference |
| ~~skip~~ | ~~R1. Different embeddings~~ | | | Dense vector search doesn't help |
| ~~skip~~ | ~~B2. Field boosting~~ | | | ColBERT reranking addresses ranking more directly |
