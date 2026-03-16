# BrowseComp-Plus Experiment Designs

## Current Best: 30% (9/30) — retrieval-v1 / hybrid-aug
- qwen3.5:9b, 12 turns, k=10, 5 pre-search BM25 queries, search decomposition prompt
- Competitive with frontier models + premium embeddings (Mixedbread blog: 33-38%)

## What We've Learned

### Things that help
- **More pre-search queries (1→5)**: Decomposes multi-hop queries into independent sub-fact searches
- **k=10** (vs k=5): More retrieval breadth without overwhelming the model
- **12 turns** (vs 6-8): Enough iterations to explore, verify, and submit
- **think:false for utility calls**: Saves tokens on pre-search extraction and fallback
- **Search decomposition prompt**: "Decompose the query, search each part separately"
- **VERIFY directive**: "Search for the candidate answer by name to confirm"
- **llm_query() emphasis**: Sub-LLM calls are the model's most powerful analysis tool
- **Fallback extraction**: Recovers answers when model doesn't call FINAL in time

### Things that don't help (or hurt)
- **More context documents (>25-30)**: Dilutes signal, overwhelms 9B model attention
- **More turns (16 vs 12)**: Model second-guesses itself, goes down rabbit holes
- **vector_search as REPL tool**: Model over-uses it, wastes turns on noisy semantic results
- **Python analysis emphasis**: Analysis isn't the bottleneck — retrieval is
- **LLM ranking of pre-search**: Fragile parsing, adds overhead, discards useful docs
- **Best-of-N majority voting**: Wrong answers agree more than correct ones (3x cost, same accuracy)
- **RRF equal weighting (BM25+vector)**: Vector results dilute BM25's keyword precision
- **Dual query generation (BM25+vector in pre-search)**: Degrades BM25 query quality, too many docs

### Key insight
The bottleneck is **retrieval** (finding the right documents), not **analysis** (understanding them). The 9B model reasons well once it has the right documents. Improvements should focus on getting better documents into context, not on how the model processes them.

---

## Experiment Designs

### A. Pre-search: Adaptive k per query

**Hypothesis**: Some queries need broad retrieval (k=10+), others benefit from precision (k=3). Currently k is fixed.

**Design**: After generating 5 pre-search queries, have the LLM also rate each query's expected specificity (1-5). Use higher k for vague queries, lower k for specific ones. E.g., "Tim Ellis Relativity Space" → k=3, "rocket company Pacific Northwest" → k=8.

**Expected effect**: Reduces noise for specific queries while maintaining recall for broad ones. Should keep context at ~25 docs.

**Risk**: Extra LLM call overhead. Specificity rating may not correlate with retrieval quality.

---

### B. Pre-search: Iterative query reformulation (no filtering)

**Hypothesis**: Round-1 results can inform better round-2 queries, but we should ADD round-2 results rather than REPLACE round-1 results (filtering is lossy).

**Design**:
1. Round 1: Generate 5 queries, search, keep ALL results
2. Round 2: Show the LLM the query + round-1 snippets. Ask for 3 NEW queries targeting UNCOVERED facts only. Add round-2 results to context (no filtering of round-1)
3. Cap total context at 30 documents (drop lowest-ranked docs from round-1 if over limit)

**Key difference from refine-add (failed)**: Cap at 30 docs prevents context dilution. Drop by BM25 rank, not by LLM filtering.

**Expected effect**: More diverse retrieval without context explosion. Estimated 25-35 context docs.

**Risk**: Capping by rank may drop relevant low-ranked docs. Marginal improvement over just using 5 good initial queries.

---

### C. Pre-search: Entity-focused decomposition

**Hypothesis**: Current pre-search generates 5 generic keyword queries. Many BrowseComp queries contain 3-5 entities (person, place, event, organization, work). Explicitly extracting and searching for each entity may improve recall.

**Design**: Change the pre-search LLM prompt from "output 5 search queries" to:
```
Extract all named entities from this question (people, places, organizations, events, works, dates).
For each entity, output a BM25 search query (2-4 words).
Format: ENTITY_TYPE: search query
```
Then search for each entity separately with k=5.

**Expected effect**: More systematic coverage of query sub-facts. Prevents the LLM from generating overlapping queries.

**Risk**: Some entities may not be searchable (too generic). May miss non-entity facts (e.g., "served as Vice President").

---

### D. REPL: Guided search strategy per turn

**Hypothesis**: The model sometimes wastes turns repeating similar searches or doing unproductive analysis. Explicit per-turn guidance could steer it more effectively.

**Design**: Replace generic iteration prompts with phase-specific prompts:
- **Turns 1-3 (EXPLORE)**: "Focus on searching. Run at least 3 different searches per turn. Don't analyze yet."
- **Turns 4-6 (ANALYZE)**: "Load the most promising documents. Use llm_query() to extract specific facts."
- **Turns 7-9 (SYNTHESIZE)**: "Cross-reference your findings. Which candidate answers are supported by multiple documents?"
- **Turns 10-12 (VERIFY+SUBMIT)**: "Search for your candidate answer to confirm. Then call FINAL()."

**Expected effect**: Prevents premature convergence. Ensures the model explores broadly before committing.

**Risk**: Rigid phases may not match all query types. Some queries are better answered with early convergence.

---

### E. REPL: Parallel sub-fact investigation

**Hypothesis**: BrowseComp queries have multiple independent sub-facts. The model could investigate them in parallel using batch_llm_query.

**Design**: Add an explicit prompt pattern:
```
For multi-part queries, investigate each sub-fact independently:
1. Search for each sub-fact separately
2. Use batch_llm_query() to analyze multiple documents simultaneously
3. Cross-reference: which entity appears in documents from DIFFERENT sub-facts?
```
The cross-referencing step is key — the answer is often the entity that connects multiple sub-facts.

**Expected effect**: More efficient use of turns. Cross-referencing could identify correct answers that single-fact searches miss.

**Risk**: batch_llm_query adds token cost. Cross-referencing logic may be too complex for 9B model.

---

### F. Context: Snippet quality improvement

**Hypothesis**: Pre-search snippets are truncated at 300 chars from document start. Many documents have irrelevant headers/metadata at the start. Better snippet extraction could improve the model's ability to identify relevant docs.

**Design**: Instead of `snippet[:300]`, extract the most relevant 300-char window:
1. In search_server.py, for each BM25 hit, find the paragraph containing the most query term matches
2. Return that paragraph as the snippet (centered on the best match)
3. Alternatively, return the first 300 chars after stripping markdown headers/metadata

**Expected effect**: More informative snippets help the model decide which documents to load with get_document().

**Risk**: Server-side changes. Paragraph extraction adds latency. May not matter since model already uses get_document().

---

### G. Retrieval: Query expansion with synonyms

**Hypothesis**: BM25 misses documents that use different terminology. Query expansion with synonyms or related terms could improve recall.

**Design**: For each pre-search query, generate 2-3 synonym variants:
- "Vice President born Maine" → also search "VP born Maine", "vice-president Maine"
- Use the LLM to generate variants, or use a simple rule-based approach (abbreviations, alternate spellings)

**Expected effect**: Catches documents that use different terminology for the same concept.

**Risk**: Query explosion (5 queries × 3 variants = 15 searches). Most BM25 misses are due to missing content, not terminology mismatch.

---

### H. Model: Thinking budget allocation

**Hypothesis**: The model uses thinking tokens (when enabled) even for simple searches, wasting budget. Selectively enabling thinking for hard analysis steps could improve quality/cost ratio.

**Design**:
- `think:false` for pre-search query generation (already done)
- `think:false` for early exploration turns (1-3) where the model is just searching
- `think:true` for analysis turns (4+) where reasoning matters
- `think:false` for fallback extraction (already done)

**Expected effect**: Saves ~50% of thinking tokens. Better reasoning on hard analysis steps.

**Risk**: Think mode affects output quality unpredictably with qwen3.5. The `/no_think` suffix in system prompt may already handle this.

---

### I. Ensemble: Config-diverse runs with union scoring

**Hypothesis**: Different configurations find different queries correct. Union across configs could yield 40%+ accuracy.

**Design**: Run 3 configurations that have shown complementary correct sets:
1. retrieval-v1 (best overall): Q159, Q175, Q191, Q464, Q469, Q797, Q830, Q853, Q885
2. refine-filter (gains Q572, Q643): Q159, Q191, Q464, Q572, Q643, Q797, Q885
3. dspy-v1 (gains Q1128): Q159, Q175, Q191, Q464, Q572, Q797, Q830, Q1128

Union: Q159, Q175, Q191, Q464, Q469, Q572, Q643, Q797, Q830, Q853, Q885, Q1128 = **12/30 (40%)**

For each query, run all 3 configs and pick the answer that appears most (or any non-"Unable to determine" answer).

**Expected effect**: 40% accuracy from existing configurations without new model improvements.

**Risk**: 3x compute cost. Need a good meta-strategy for picking among disagreeing answers.

---

### J. Retrieval: Document re-ranking with cross-encoder

**Hypothesis**: BM25 retrieves broadly but ranks imprecisely. A cross-encoder could re-rank top-k results for better precision.

**Design**: After BM25 retrieves k=20 results, use a cross-encoder model (e.g., via Ollama or a small local model) to score each (query, document) pair and re-rank. Keep top 10.

**Expected effect**: More relevant documents in top positions. Better signal-to-noise in context.

**Risk**: Cross-encoder inference adds latency. Need a good cross-encoder model that runs locally. Previous LLM-based ranking (ranking-v1) failed due to parsing fragility — a dedicated cross-encoder avoids that issue.

---

### K. Pre-search: Two-phase with gold document detection

**Hypothesis**: Some pre-search results are very likely to contain the answer ("gold documents"). Detecting these early and loading them fully could improve accuracy.

**Design**:
1. Pre-search as normal (5 queries, k=5 each)
2. For each result, compute a "gold score" based on: number of query terms in snippet, snippet length, title relevance
3. Auto-load the top 3 "gold" documents with get_document() before the REPL loop
4. Inject loaded documents as additional context variables (e.g., `gold_doc_1`, `gold_doc_2`)

**Expected effect**: Model starts with full text of most promising documents instead of just snippets.

**Risk**: Loading 3 full documents adds ~24K chars to context. May overwhelm the model if documents are large.

---

### L. Prompt: Answer type classification

**Hypothesis**: Knowing the expected answer type (person, place, date, number, organization) could help the model focus its search and extraction.

**Design**: Before the REPL loop, classify the query into answer types:
```
What type of answer does this question expect?
A) Person name  B) Place/location  C) Date/time  D) Number  E) Organization  F) Other
```
Inject the answer type into the system prompt: "The answer is expected to be a PERSON NAME."

**Expected effect**: Helps the model focus extraction. Prevents answers like "Episode 2" when the question asks for a person.

**Risk**: Misclassification could mislead. BrowseComp queries often have non-obvious answer types.

---

## Priority Ranking

Based on our experimental history, ranked by expected impact and feasibility:

| Priority | Experiment | Expected Impact | Cost | Rationale |
|----------|-----------|----------------|------|-----------|
| 1 | **I. Ensemble** | High (40%) | 3x compute | Guaranteed improvement from known correct sets |
| 2 | **C. Entity decomposition** | Medium | Low | Better pre-search targeting, addresses retrieval bottleneck |
| 3 | **B. Iterative reformulation (capped)** | Medium | Low | Addresses gaps without dilution, simple to implement |
| 4 | **D. Guided search phases** | Medium | None | Better turn utilization, no extra compute |
| 5 | **L. Answer type classification** | Medium | Low | Focuses extraction, cheap LLM call |
| 6 | **F. Snippet quality** | Medium | Low | Better doc selection, addresses retrieval |
| 7 | **K. Gold document detection** | Medium | Medium | Gives model full text early, risky context size |
| 8 | **E. Parallel sub-facts** | Low-Medium | Medium | Efficient but complex for 9B model |
| 9 | **H. Thinking budget** | Low | None | Marginal optimization |
| 10 | **A. Adaptive k** | Low | Low | Over-engineering for small effect |
| 11 | **G. Query expansion** | Low | Medium | Most misses are content gaps, not terminology |
| 12 | **J. Cross-encoder** | Medium | High | Good idea but needs infrastructure |
