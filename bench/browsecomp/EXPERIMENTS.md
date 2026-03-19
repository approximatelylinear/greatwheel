# BrowseComp-Plus Experiment Designs

## Current Best: 36.7% (11/30) — commit f6d4868

Boosted BM25 (phrase+AND+OR) + PRF + doc-grounded prompts.
Reproducibility range: 9-11/30 across runs (mean ~10).

## What We've Learned

### Things that help
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

### Key insight
86% of failures are **retrieval** (right documents never found), not extraction. The 9B model reasons well once it has the right documents. However, vector search doesn't help with retrieval on this benchmark — the missing documents lack keyword overlap AND semantic overlap with the queries.

---

## Completed Experiments (mar16-mar19)

### Retrieval (Rust: gw-memory, gw-bench)
- [x] **Boosted BM25** — phrase match (4x), slop phrase (2x), AND'd terms (1.5x), OR'd terms (1x). **Biggest contributor.**
- [x] **PRF (pseudo-relevance feedback)** — distinctive snippet terms as additional queries. Worth ~1-2 queries.
- [x] **HyDE** — hypothetical answer passage as BM25 query. **Ablated out: no benefit.**
- [x] **Vector search (6 variants)** — all showed no improvement. See table above.
- [x] **Wider pre-search** — 8+5 queries overwhelm model with context.

### Prompts (Rust: gw-bench)
- [x] **Doc-grounded prompts** — force document reading + verification. Matches best.
- [x] **Early-submit prompts** — nudge when evidence found. Slight regression.
- [x] **Search query coaching** — BM25 formulation advice. No benefit.

### Infrastructure
- [x] **Per-query timing breakdown** — pre_search_ms, rlm_loop_ms, bridge component timers
- [x] **Re-embedded LanceDB index** — `browsecomp_docs_prefixed` with `search_document:` prefix (100K docs, 37 min)

---

## Planned Experiments

### BM25 Refinements

#### B1. Fuzzy term matching (5th signal)
- **Where:** `crates/gw-memory/src/corpus.rs` — `search_bm25_boosted()`
- **What:** Add FuzzyTermQuery with edit distance 1 (boost 0.5) as 5th signal. Catches typos and morphological variants.
- **Effort:** Small — tantivy has `FuzzyTermQuery`
- **Expected impact:** Low — may help with name misspellings
- **Status:** Ready to implement

#### B2. Document field boosting (BM25f)
- **Where:** `crates/gw-memory/src/corpus.rs` — index schema + search
- **What:** Extract title/heading from web documents at index time. Add separate tantivy fields with higher boost.
- **Effort:** Large — requires corpus analysis + re-indexing
- **Expected impact:** Medium — titles are high-signal for entity matching
- **Status:** Needs corpus structure analysis

#### B3. Query expansion via LLM
- **Where:** `crates/gw-bench/src/main.rs` — pre_search or bridge
- **What:** Before each BM25 search, expand the query with synonyms/related terms via a fast LLM call. Different from HyDE — this targets keyword expansion, not passage generation.
- **Effort:** Medium
- **Expected impact:** Low-medium — addresses vocabulary mismatch

### Agent/Prompt Improvements

#### A1. Iterative search budget enforcement
- **What:** Track unique search queries per question. If < 5 by iteration 4, force more searching before allowing FINAL().
- **Where:** `crates/gw-bench/src/main.rs` — iteration prompts + bridge tracking
- **Effort:** Medium
- **Expected impact:** Low-medium — addresses under-searching

#### A2. Answer type classification
- **What:** Classify expected answer type (person, place, date, number, etc.) before REPL loop. Inject into prompt.
- **Effort:** Low — one LLM call
- **Expected impact:** Low-medium — focuses extraction

#### A3. Larger model for main loop
- **What:** Use qwen2.5:32b or qwen3.5:32b for main rLM loop (keep 9B for utility calls).
- **Effort:** Small config change, large latency increase (~4x slower)
- **Expected impact:** High — better reasoning → better queries + extraction
- **Status:** Requires GPU memory assessment

#### A4. Config-diverse ensemble with union scoring
- **What:** Run 2-3 diverse configurations per query, take union of correct answers.
- **Expected impact:** High — historical union across configs reaches ~40%
- **Cost:** 2-3x compute
- **Status:** Needs orchestration code

### Alternative Retrieval

#### R1. Different embedding model
- **What:** Try a stronger embedding model (e.g., bge-large, e5-mistral) instead of nomic-embed-text
- **Effort:** Medium — need to re-embed 100K docs
- **Expected impact:** Unknown — vector search may benefit from better embeddings, but BrowseComp's entity-heavy nature favors BM25
- **Status:** Deprioritized given vector search findings

#### R2. ColBERT reranker on BM25 results
- **What:** After boosted BM25 retrieves top-k candidates, rerank them with a ColBERT cross-encoder before feeding to the agent. ColBERT computes fine-grained token-level similarity between query and document — much more precise than BM25 scores for ranking.
- **Where:** Two implementation paths:
  1. **Python sidecar (fastest to test):** Add a reranking step in `lancedb_searcher.py` or a new `reranker.py` that the Rust bridge calls via HTTP. LanceDB's Python client has built-in support: `tbl.search(query, query_type="fts").rerank(reranker=ColbertReranker()).to_list()`
  2. **Rust-native (production path):** Run the ColBERT model via ONNX runtime in Rust, or call Ollama/a local model server for reranking scores.
- **Models:** `colbert-ir/colbertv2.0` (default) or `answerdotai/answerai-colbert-small-v1` (lighter). Both auto-download on first use via HuggingFace.
- **Integration with current pipeline:**
  - Pre-search: BM25 retrieves k=20-30 candidates → ColBERT reranks → top 10 go into `context`
  - REPL search: Each `search()` call retrieves k=20 → ColBERT reranks → top 10 returned to agent
  - This is different from vector search (which failed) because ColBERT scores query-document **pairs** rather than comparing independent embeddings
- **LanceDB Python API:**
  ```python
  from lancedb.rerankers import ColbertReranker
  reranker = ColbertReranker()  # or model_name="answerdotai/answerai-colbert-small-v1"
  results = tbl.search("query", query_type="fts").rerank(reranker=reranker).to_list()
  ```
  Also works with hybrid search: `query_type="hybrid"` (with `return_score="relevance"`)
- **Effort:** Medium — Python sidecar is straightforward; Rust-native is larger
- **Expected impact:** Medium-high — cross-encoders consistently outperform bi-encoders and BM25 for precision. Our failure analysis showed the right document is often in the top-20 BM25 results but not top-10 — reranking could surface it.
- **Latency:** ColBERT reranking of 20-30 documents per query adds ~100-500ms per search call (GPU) or ~1-3s (CPU). Pre-search adds one reranking pass; REPL searches add per-call latency.
- **Risk:** Python sidecar adds HTTP round-trip overhead. CPU-only reranking may be too slow for REPL search calls. Could limit reranking to pre-search only.
- **Status:** Ready to prototype (Python sidecar path)

#### R3. Learned sparse retrieval (SPLADE-style)
- **What:** Use a learned sparse model that outputs weighted term expansions. Combines BM25-style matching with learned term importance.
- **Effort:** Large — needs model + index infrastructure
- **Expected impact:** Medium-high — addresses vocabulary mismatch without the noise of dense vectors
- **Status:** Research needed

---

## Priority Ranking

| Priority | Experiment | Expected Impact | Effort | Rationale |
|----------|-----------|----------------|--------|-----------|
| **1** | A3. Larger model | High | Small | Biggest ceiling lift, bottleneck is LLM reasoning |
| **2** | R2. ColBERT reranker | Medium-High | Medium | Cross-encoders beat BM25 for precision; right docs may be in top-20 but not top-10 |
| **3** | A4. Ensemble | High | Medium | Guaranteed from known correct set diversity |
| **4** | B2. Field boosting | Medium | Large | Titles are high-signal for entity queries |
| **5** | B1. Fuzzy matching | Low | Small | Easy marginal gain |
| **6** | A2. Answer type | Low-Medium | Low | Cheap, may focus extraction |
| **7** | A1. Search budget | Low-Medium | Medium | Addresses under-searching |
| **8** | B3. Query expansion | Low-Medium | Medium | Vocabulary mismatch |
| **9** | R3. SPLADE | Medium-High | Large | Best-of-both-worlds retrieval |
| **10** | R1. Different embeddings | Unknown | Medium | Deprioritized — vector search doesn't help |
