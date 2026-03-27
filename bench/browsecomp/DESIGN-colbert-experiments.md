# Design: ColBERT Deep-Dive Experiments

**Status:** Draft
**Date:** 2026-03-27
**Goal:** Understand why ColBERT reranking hurts and find a configuration where it helps.

---

## 1. Current Observations

| Configuration | Exact | Fuzzy | Delta vs baseline |
|---|---|---|---|
| Doc BM25 only (baseline) | 8/30 | — | — |
| Doc BM25 top-200 → ColBERT rerank (mar21) | 12/30 | — | +4 exact |
| Passage+doc RRF (no rerank) | 10/30 | 12/30 | +2/+4 |
| Passage+doc RRF → ColBERT rerank | 7/30 | 8/30 | -1/-4 from baseline |

ColBERT reranking helped in one setting (doc-only, +4) but hurt in another
(passage+doc, -3 to -4). The passage+rerank run lost 5 previously correct
answers while gaining only 2.

---

## 2. Hypotheses for Why ColBERT Hurts

### H1: Document truncation kills long-doc answers
ColBERT truncates to **2048 chars / 512 tokens**. BrowseComp docs average
~2400 chars but many are 10K+. If the answer is past the 2048-char mark,
ColBERT literally can't see it — it will rank the doc lower than a
topically-similar doc where it can see matching tokens in the first 2048 chars.

**Test:** Measure how many gold docs have the answer string past char 2048.

### H2: Semantic similarity ≠ keyword relevance for entity queries
BrowseComp queries are entity-heavy. BM25 matches "Neville Gruzman architect
Sydney" literally. ColBERT may rank documents about "Australian architecture"
or "Sydney architects" higher even if they don't mention Gruzman. MaxSim finds
many token-level matches for common terms, drowning out the rare entity match.

**Test:** Compare BM25 rank vs ColBERT rank for the gold doc across all 30
queries. If ColBERT consistently ranks gold docs lower, this confirms the
hypothesis.

### H3: Passage snippets are too short for ColBERT
When passage+doc RRF feeds ColBERT, many candidates are passage-surfaced docs
with focused 512-byte snippets. ColBERT may score these poorly because it has
less text to match against (MaxSim sum is lower with fewer tokens).

**Test:** Compare ColBERT scores for the same docid when fed a passage snippet
vs full doc text.

### H4: Reranking pool is wrong for passages
The rerank pool is top-200, calibrated for doc-level BM25. Passage+doc RRF may
already have high-quality top-10 — reranking a wider pool introduces noise.

**Test:** Try ColBERT rerank on passage+doc top-50 and top-20 instead of 200.

### H5: ColBERT query encoding mismatches BrowseComp query style
BrowseComp queries are very long (multi-paragraph descriptions with constraints).
ColBERT truncates queries to 128 tokens. The distinctive keywords may be past
the truncation point.

**Test:** Measure average query token count. If >128, try extracting query
keywords first, then encoding the shortened version.

---

## 3. Diagnostic Experiments (no code changes, analysis only)

### D1: Gold doc truncation analysis
```python
# For each query: where does the answer string first appear in the gold doc?
# If answer_position > 2048 chars in many cases, ColBERT can't see it.
for query in sample30:
    gold_doc = get_gold_doc(query)
    answer = get_answer(query)
    pos = gold_doc.text.find(answer)
    print(f"{query.id}: answer at char {pos}, doc length {len(gold_doc.text)}")
```

### D2: BM25 vs ColBERT gold doc rank comparison
```python
# For each query: what rank does BM25 vs ColBERT give the gold doc?
for query in sample30:
    bm25_results = search_bm25(query, k=200)
    colbert_results = rerank_colbert(query, bm25_results)
    bm25_rank = find_rank(gold_docid, bm25_results)
    colbert_rank = find_rank(gold_docid, colbert_results)
    print(f"{query.id}: BM25 rank={bm25_rank}, ColBERT rank={colbert_rank}")
```

### D3: Query length analysis
```python
# How many tokens do BrowseComp queries produce? Are we truncating?
for query in sample30:
    tokens = tokenizer.encode(query.text)
    print(f"{query.id}: {len(tokens)} tokens {'[TRUNCATED]' if len(tokens) > 128 else ''}")
```

---

## 4. Code Experiments (ordered by expected insight)

### C1: Increase ColBERT document token limit
- **What:** Increase from 512 to 2048 tokens (or remove limit entirely)
- **Where:** `colbert_reranker.py` line 29 (`max_length`), line 79 (char truncation)
- **Effort:** Small — parameter change in reranker
- **Rationale:** If H1 is correct, this is the fix. 512 tokens ≈ 350-400 words,
  but many BrowseComp docs are 2000+ words.

### C2: Smaller rerank pool with passages
- **What:** Try ColBERT rerank on passage+doc top-20/50 instead of top-200
- **Where:** `main.rs` line 474 — change `max(k, 200)` threshold
- **Effort:** Small — parameter change
- **Rationale:** Tests H4. The passage index already provides high-quality
  candidates — ColBERT should refine, not reshuffle.

### C3: Query keyword extraction before ColBERT encoding
- **What:** Use LLM to extract 5-10 key terms from the long BrowseComp query,
  then encode the shortened query for ColBERT. Keep full query for BM25.
- **Where:** `rerank_server.py` or `colbert_reranker.py` — pre-process query
- **Effort:** Medium — add LLM call before encoding
- **Rationale:** Tests H5. If queries are >128 tokens, distinctive terms may
  be lost. Extraction focuses ColBERT on what matters.

### C4: ColBERT as first-stage retriever (LanceDB multi-vector)
- **What:** Build the full ColBERT index (requires `build_colbert_index.py`,
  2-4 hours on GPU), then use `search_colbert()` as a retrieval channel
  alongside BM25 in RRF (not as a reranker).
- **Where:** Already implemented in `corpus.rs::search_hybrid_colbert()`
- **Effort:** Large (index build time), small (code already exists)
- **Rationale:** LightOn used ColBERT as first-stage, not reranker. As a
  reranker, ColBERT can only reshuffle what BM25 found. As a retriever, it
  can find documents BM25 missed entirely. This addresses the core retrieval
  miss failure mode.

### C5: ColBERT + Passage RRF (3-channel fusion)
- **What:** Three retrieval channels in RRF: doc BM25 + passage BM25 + ColBERT
  first-stage. No reranking — pure fusion.
- **Where:** New search method in `corpus.rs`, or compose existing methods
- **Effort:** Medium (needs ColBERT index from C4)
- **Rationale:** Each channel finds different documents. BM25 catches keywords,
  passages catch buried answers, ColBERT catches reasoning-intensive queries.
  RRF ensures a doc found by any channel gets boosted.

### C6: Score-weighted RRF instead of rank-based
- **What:** Instead of standard RRF (rank-based), weight by normalized score:
  `score_rrf(d) = α * bm25_norm(d) + β * colbert_norm(d)`. This preserves
  score magnitude — a high-confidence BM25 match won't be displaced by a
  mediocre ColBERT match.
- **Where:** `fusion.rs` — new fusion function
- **Effort:** Small
- **Rationale:** Standard RRF treats all result lists equally. But BM25 may
  have high confidence on entity matches while ColBERT's scores are spread
  more evenly. Score-weighting respects this.

### C7: Passage-level ColBERT reranking
- **What:** Instead of reranking full documents, rerank passage-level snippets.
  ColBERT sees the focused 512-byte passage text, not the first 2048 chars of
  the full document.
- **Where:** Modify rerank flow to send passage text instead of full doc text
- **Effort:** Medium
- **Rationale:** Tests H3. Passage text is exactly the relevant section — no
  truncation problem, no dilution from irrelevant doc content.

---

## 5. Recommended Experiment Order

**Phase 1: Diagnostics** (no code, 30 min)
Run D1, D2, D3 to understand the data before changing code.

**Phase 2: Quick parameter sweeps** (small code changes)
1. C1 (increase token limit) — directly tests H1
2. C2 (smaller rerank pool) — directly tests H4
3. C6 (score-weighted fusion) — tests whether rank-based RRF is the issue

**Phase 3: Architecture changes** (require ColBERT index build)
4. C4 (ColBERT first-stage) — the big bet, tests the reranker-vs-retriever hypothesis
5. C5 (3-channel RRF) — combines all retrieval strategies
6. C7 (passage-level reranking) — alternative to full-doc reranking

**Phase 4: Query preprocessing**
7. C3 (query keyword extraction) — addresses query truncation

---

## 6. Success Criteria

- **Minimum:** ColBERT doesn't hurt (matches passage-only 12/30 fuzzy)
- **Target:** ColBERT adds +2-3 queries beyond passage-only (14-15/30)
- **Stretch:** Matches LightOn's published numbers at smaller model size

If diagnostics show H1 is the dominant issue (answer past truncation), C1
alone may recover most of the lost accuracy.
