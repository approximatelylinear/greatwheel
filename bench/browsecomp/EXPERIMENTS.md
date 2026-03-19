# BrowseComp-Plus Experiment Designs

## Current Best: 36.7% (11/30) — commit f6d4868

Boosted BM25 (phrase+AND+OR) + PRF + doc-grounded prompts.
Reproducibility range: 9-11/30 across runs (mean ~10).

## What We've Learned

### Things that help
- **Boosted BM25** (phrase match 4x, slop 2x, AND 1.5x, OR 1x): Biggest single contributor per ablation
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
- **Vector search in pre-search or REPL**: Consistently no benefit across multiple experiments
- **More context documents (>25-30)**: Dilutes signal, overwhelms 9B model attention
- **More turns (16 vs 12)**: Model second-guesses itself, goes down rabbit holes
- **vector_search as REPL tool**: Model over-uses it, wastes turns on noisy semantic results
- **Python analysis emphasis**: Analysis isn't the bottleneck — retrieval is
- **LLM ranking of pre-search**: Fragile parsing, adds overhead, discards useful docs
- **Best-of-N majority voting**: Wrong answers agree more than correct ones (3x cost, same accuracy)
- **RRF equal weighting (BM25+vector)**: Vector results dilute BM25's keyword precision
- **Search query coaching in prompt**: No measurable benefit over baseline
- **Wider pre-search (8+5 queries, full k)**: Too many context docs overwhelm model

### Key insight
86% of failures are **retrieval** (right documents never found), not extraction. The 9B model reasons well once it has the right documents. Improvements should focus on getting better documents into context.

---

## Planned: Vector Search Improvements (no model changes)

### V1. Fix search_hybrid to use boosted BM25
- **Where:** `crates/gw-memory/src/corpus.rs` — `search_hybrid()` calls `search_bm25()`, should call `search_bm25_boosted()`
- **Effort:** 1 line
- **Expected impact:** Medium — compounds boosted BM25 with vector signal in hybrid mode
- **Status:** Ready to implement

### V2. nomic-embed-text task prefixes (query side)
- **Where:** `crates/gw-llm/src/lib.rs` embed() or call sites in gw-bench
- **What:** Prepend `search_query: ` to query text before embedding. nomic-embed-text is trained with asymmetric prefixes (`search_query:` for queries, `search_document:` for docs). Even without re-embedding docs, the query prefix alone may improve retrieval.
- **Effort:** Small — add prefix string at embedding call sites for search queries
- **Expected impact:** Medium-high — known 5-15% recall improvement for nomic models
- **Status:** Ready to implement
- **Risk:** Documents were embedded without `search_document:` prefix; partial prefix may help or hurt

### V3. Weighted RRF (BM25-favored fusion)
- **Where:** `crates/gw-memory/src/fusion.rs` + `crates/gw-memory/src/corpus.rs`
- **What:** Give BM25 results higher weight in RRF. Options:
  - Double BM25 results in RRF input (count each BM25 hit twice)
  - Add per-list weight parameter to `reciprocal_rank_fusion()`
  - Use asymmetric k constants (lower k for BM25 = steeper rank decay = more weight)
- **Effort:** Small
- **Expected impact:** Medium — BM25 is demonstrably stronger, shouldn't be diluted by equal-weight fusion
- **Status:** Ready to implement

### V4. LanceDB nprobes tuning
- **Where:** `crates/gw-memory/src/corpus.rs` — `search_vector()` LanceDB query
- **What:** Increase nprobes for IVF-PQ search. Default is often low (10-20). Try 50-100 for better recall at slight latency cost.
- **Effort:** Small — add `.nprobes(N)` to vector search query builder
- **Expected impact:** Low-medium — depends on current default and index geometry
- **Status:** Ready to implement

### V5. Query reformulation before vector embedding
- **Where:** `crates/gw-bench/src/main.rs` — bridge `search_with_mode()` for "hybrid"/"vector"
- **What:** Before embedding a keyword query for vector search, reformulate as natural language. The agent generates keyword queries like `"convent Michigan 1932"` that embed poorly. Reformulate to `"A convent in Michigan established around 1932"` for better semantic matching.
- **Options:**
  - Fast LLM call (adds latency but high quality)
  - Template-based expansion (no latency: prepend "Find documents about" or expand abbreviations)
  - Cache reformulations per query session to avoid repeated LLM calls
- **Effort:** Medium
- **Expected impact:** Medium — better query-doc alignment in embedding space
- **Status:** Needs design

### V6. Re-embed documents with search_document prefix
- **Where:** LanceDB index rebuild pipeline (bench/browsecomp/lancedb_searcher.py or equivalent)
- **What:** Re-embed all 100K documents with `search_document: ` prefix for nomic-embed-text. Combined with V2 (query prefix), gives full asymmetric retrieval benefit.
- **Effort:** Large — full re-indexing, ~hours with Ollama embedding at 100K docs
- **Expected impact:** High — completes the nomic task-prefix story
- **Status:** Blocked on V2 results (only worth rebuilding if query prefix alone shows promise)

### V7. Multi-vector query fusion
- **Where:** `crates/gw-memory/src/corpus.rs` — new method
- **What:** For a single search query, generate multiple embedding vectors (original query, expanded query, hypothetical answer) and fuse their vector search results with RRF.
- **Effort:** Medium
- **Expected impact:** Low-medium — HyDE already showed no benefit for BM25; may help for vector
- **Status:** Deprioritized (HyDE ablation suggests limited value)

---

## Planned: BM25 Refinements

### B1. Fuzzy term matching (5th signal)
- **Where:** `crates/gw-memory/src/corpus.rs` — `search_bm25_boosted()`
- **What:** Add FuzzyTermQuery with edit distance 1 (boost 0.5) as 5th signal. Catches typos and morphological variants (e.g., "Gyan" matching "Gyans").
- **Effort:** Small — tantivy has `FuzzyTermQuery`
- **Expected impact:** Low — may help with name misspellings in rare cases
- **Status:** Ready to implement

### B2. Document field boosting (BM25f)
- **Where:** `crates/gw-memory/src/corpus.rs` — index schema + search
- **What:** Extract title/heading from web documents at index time. Add separate tantivy fields with higher boost. BrowseComp docs are web pages — `<title>` and `<h1>` are high-signal.
- **Effort:** Large — requires corpus analysis + re-indexing
- **Expected impact:** Medium — titles are high-signal for entity matching
- **Status:** Needs corpus structure analysis

---

## Planned: Agent/Prompt Improvements

### A1. Iterative search budget enforcement
- **What:** Track unique search queries per question. If < 5 by iteration 4, force more searching before allowing FINAL().
- **Where:** `crates/gw-bench/src/main.rs` — iteration prompts + bridge tracking
- **Effort:** Medium
- **Expected impact:** Low-medium — addresses under-searching

### A2. Answer type classification
- **What:** Classify expected answer type (person, place, date, number, etc.) before REPL loop. Inject into prompt.
- **Effort:** Low — one LLM call
- **Expected impact:** Low-medium — focuses extraction

### A3. Larger model for main loop
- **What:** Use qwen2.5:32b or qwen3.5:32b for main rLM loop (keep 9B for utility calls).
- **Effort:** Small config change, large latency increase
- **Expected impact:** High — better reasoning → better queries + extraction
- **Status:** Requires GPU memory assessment

### A4. Config-diverse ensemble with union scoring
- **What:** Run 2-3 diverse configurations per query, take union of correct answers.
- **Expected impact:** High — historical union across configs reaches ~40%
- **Cost:** 2-3x compute
- **Status:** Needs orchestration code

---

## Priority Ranking

| Priority | Experiment | Expected Impact | Effort | Rationale |
|----------|-----------|----------------|--------|-----------|
| **1** | V1. Hybrid uses boosted BM25 | Medium | Trivial | Free improvement, 1 line |
| **2** | V2. nomic query prefix | Medium-High | Small | Known technique, easy to test |
| **3** | V3. Weighted RRF | Medium | Small | BM25 shouldn't be diluted |
| **4** | V4. LanceDB nprobes | Low-Medium | Small | Easy to test |
| **5** | V5. Query reformulation | Medium | Medium | Better embedding alignment |
| **6** | B1. Fuzzy matching | Low | Small | Marginal but easy |
| **7** | A4. Ensemble | High | Medium | Guaranteed from known sets |
| **8** | A3. Larger model | High | Small | Biggest ceiling lift |
| **9** | V6. Re-embed with prefix | High | Large | Only after V2 validates |
| **10** | B2. Field boosting | Medium | Large | Requires re-indexing |
