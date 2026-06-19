# Design: Reason-ModernColBERT for BrowseComp-Plus Retrieval

**Status:** Proposed
**Date:** 2026-03-19
**Supersedes:** R2 (ColBERT reranker) in EXPERIMENTS.md

---

## Motivation

Our retrieval pipeline tops out at 36.7% (11/30) on BrowseComp-Plus with boosted BM25.
86% of failures are retrieval misses — the right documents never reach the agent.
Dense retrieval (nomic-embed-text, 6 variants) added zero lift.

LightOn published BrowseComp-Plus results using Reason-ModernColBERT (149M params):

| LLM | Retriever | Accuracy | Search calls |
|-----|-----------|----------|--------------|
| GPT-5 | Reason-ModernColBERT (custom scaffold) | 87.59% | 13.27 |
| GPT-5 | Reason-ModernColBERT (standard) | 79.52% | 19.31 |
| GPT-5 | Mixedbread Search | 78.41% | 44.67 |

Two things stand out:
1. The retriever delta is real but modest (~1 point between Reason-ModernColBERT and
   Mixedbread at standard scaffold). The custom scaffold matters more (+8 points).
2. We already have a comparable scaffold (get_document + llm_query + iterative search).

The open question is: **how much of the retrieval gap can we close with a better retriever
while keeping our small local LLM?** This is worth isolating before studying model size effects.

### Why late interaction beats dense on this task

Our nomic-embed-text experiments failed not because of model quality but because of
**architecture**. Dense retrieval compresses a document into a single vector — too lossy
for reasoning-intensive queries where the answer signal is a specific phrase buried in
a long document. ColBERT's late interaction computes token-level MaxSim scores, preserving
fine-grained matching. LightOn's own ablation shows +7.3 NDCG@10 for late interaction vs
dense on the same training data.

BrowseComp queries are entity-heavy, which favors BM25 keyword matching. But our failure
analysis shows the missing documents often *don't share keywords* with the query at all —
they require reasoning about what kind of document would contain the answer. This is exactly
where Reason-ModernColBERT was trained to excel (via ReasonIR's synthetic reasoning queries).

---

## Goal

Measure the isolated effect of Reason-ModernColBERT on retrieval quality with our existing
qwen3.5:9b agent, then establish a retrieval quality baseline for future model-size studies.

**Primary metric:** Accuracy on sample30 (comparable to current 36.7% baseline)
**Secondary metric:** Retrieval recall@k — does the gold document appear in top-k results
more often? This isolates retriever quality from LLM reasoning ability.

---

## Design

### Architecture: first-stage retriever, not reranker

The original R2 plan proposed ColBERT as a reranker on BM25 results. LightOn's results
suggest using Reason-ModernColBERT as the **primary retriever** instead:

- BM25 reranking limits recall to what BM25 found — it can't surface documents BM25 missed
- Our failure mode is retrieval misses, not ranking errors within the BM25 result set
- LightOn used it as a first-stage retriever, not a reranker

**Proposed retrieval modes to test:**

| Mode | Description | Expected benefit |
|------|-------------|-----------------|
| A. ColBERT-only | Reason-ModernColBERT as sole retriever | Tests raw ColBERT retrieval quality |
| B. BM25 + ColBERT fusion | RRF fusion of BM25 and ColBERT results | Best-of-both: keywords + reasoning |
| C. BM25 → ColBERT rerank | BM25 top-50 reranked by ColBERT | Preserves BM25 strengths, ColBERT refines |

Mode B is the most promising — BM25 catches entity/keyword queries, ColBERT catches
reasoning-intensive ones. Mode A isolates ColBERT's contribution. Mode C is the safest
fallback (closest to current pipeline).

### Indexing

Reason-ModernColBERT is a multi-vector model: each document produces N token embeddings
(128-dim each), not a single vector. This requires a multi-vector index.

**Options:**

1. **PyLate + NextPlaid** (LightOn's own stack) — purpose-built for ColBERT indexing.
   NextPlaid is a local-first multi-vector database. Likely what they used for BrowseComp.
2. **PyLate + custom FAISS** — store all token embeddings, retrieve via MaxSim aggregation.
3. **ColBERT v2 / PLAID index** — the original ColBERT indexing engine (Stanford).
4. **RAGatouille** — high-level wrapper around ColBERT with simple indexing API.

Recommendation: start with **RAGatouille** for fastest iteration, consider **NextPlaid**
if we need performance at scale. RAGatouille handles index creation, search, and model
loading in a few lines of code and supports custom ColBERT models.

### Corpus indexing cost

- 100K documents at ~8192 max tokens each
- 149M model, 128-dim token embeddings
- Estimate: 1-3 hours on a single GPU (24GB) for full corpus encoding
- Index size: ~10-50 GB depending on average document length (multi-vector indexes are large)
- This is a one-time cost; the index is reused across all experiments

### Integration into search_server.py

```
search_server.py
├── POST /call/search
│   ├── mode=bm25        (current default)
│   ├── mode=colbert      (new: Reason-ModernColBERT)
│   ├── mode=hybrid       (new: BM25 + ColBERT RRF fusion)
│   └── mode=rerank       (new: BM25 top-50 → ColBERT rerank)
└── POST /call/get_document  (unchanged)
```

The search server already supports mode switching. Adding ColBERT requires:
1. Load the Reason-ModernColBERT index at startup
2. Implement `search_colbert(query, k)` using RAGatouille or PyLate
3. Implement `search_hybrid(query, k)` using RRF fusion (we already have RRF code)
4. Implement `search_rerank(query, k)` calling BM25 top-50 then ColBERT reranking

---

## Experiment Plan

### Phase 0: Retrieval recall measurement (new)

Before changing anything, measure **retrieval recall** on the current BM25 pipeline.
For each sample30 query, check whether the gold document appears in BM25's top-10/20/50.
This gives us a ceiling: if the gold doc isn't in top-50, no reranking can help.

**Requires:** A mapping from sample30 queries to gold document IDs. If this doesn't exist,
we can approximate by checking which docids the agent retrieved in successful runs.

### Phase 1: Index + smoke test (1-2 days)

1. Install PyLate/RAGatouille and download Reason-ModernColBERT (cc-by-nc for research)
2. Encode the 100K BrowseComp-Plus corpus into a ColBERT index
3. Run 5 sample queries manually, inspect retrieval quality
4. Compare top-10 results against BM25 top-10 for the same queries

**Go/no-go gate:** Do ColBERT results contain documents that BM25 misses for at least
2 of 5 test queries? If yes, proceed. If no, the index or model may need debugging.

### Phase 2: Integration + sample30 eval (1-2 days)

1. Wire ColBERT search into search_server.py (all three modes)
2. Run sample30 with each mode (A, B, C) using current qwen3.5:9b agent
3. Record accuracy, retrieval recall@10, search calls, tokens

**Expected outcomes:**
- Mode A (ColBERT-only): Likely lower than BM25 on entity queries, higher on reasoning queries
- Mode B (hybrid): Best overall — addresses both failure modes
- Mode C (rerank): Modest improvement — limited by BM25 recall ceiling

### Phase 3: Retrieval quality analysis (1 day)

For each sample30 query, compare retrieval sets across modes:
- Which queries gained/lost documents vs BM25-only?
- For failures: is the gold document in the full corpus but missed by all retrievers?
- Categorize queries by type (entity lookup vs reasoning) and measure per-type accuracy

This analysis feeds into the model-size study: it separates "retriever found the doc but
LLM couldn't reason over it" from "retriever missed the doc entirely."

### Phase 4: Model-size interaction study (future)

Hold the best retrieval mode fixed. Sweep LLM size:
- qwen3.5:9b (current)
- qwen2.5:32b or qwen3.5:32b
- API model (Claude / GPT-5) for ceiling comparison

Plot accuracy vs model size × retriever type. This answers: "does a better retriever
reduce the need for a larger LLM?" If the curves converge at small model sizes, retrieval
dominates. If they diverge, LLM reasoning is the limiting factor regardless of retrieval.

---

## Licensing Path

For research/benchmarking: use the cc-by-nc-4.0 model directly from HuggingFace.

For production (Apache 2.0):
1. Clone the [ReasonIR data generation code](https://github.com/facebookresearch/ReasonIR/tree/main/synthetic_data_generation)
2. Regenerate training triplets using our own LLM calls (~100K samples)
3. Fine-tune `lightonai/GTE-ModernColBERT-v1` (Apache 2.0 base) using
   [LightOn's training script](https://gist.github.com/NohTow/d563244596548bf387f19fcd790664d3)
4. Training: 3 epochs, batch 256, lr 1e-5, bf16 — ~2 hours on multi-GPU

The data generation step is the bottleneck: it requires LLM API calls for synthetic
query generation. Cost depends on the LLM used (the original code targets GPT-4 batch API;
we could substitute with a local model at quality risk).

**Decision point:** Only pursue Apache 2.0 reproduction if Phase 2 results justify it.

---

## Hardware Requirements

| Component | VRAM | Notes |
|-----------|------|-------|
| qwen3.5:9b (agent LLM) | ~6-7 GB | Current, via Ollama |
| Reason-ModernColBERT (inference) | ~400 MB | 149M params, 128-dim |
| ColBERT index in memory | ~2-8 GB | Depends on index format |
| **Total** | **~10-16 GB** | Fits on 24GB GPU |

Indexing the corpus (one-time) may need more memory depending on batch size.
Inference adds ~100-500ms per search call for ColBERT scoring.

---

## Risks

1. **Multi-vector index size.** ColBERT stores per-token embeddings — index could be
   30-50 GB for 100K long documents. May need quantization or disk-based index.

2. **Query length limit.** Reason-ModernColBERT caps queries at 128 tokens. Our agent's
   reformulated queries may exceed this. Mitigation: truncate or extract key terms.

3. **BrowseComp ≠ BRIGHT.** The model was trained on ReasonIR (derived from BRIGHT).
   BrowseComp queries may have different characteristics. LightOn validated on BrowseComp
   but with GPT-5 — we don't know how it performs with a weaker agent LLM.

4. **GPU memory contention.** Concurrent Ollama inference + ColBERT scoring on the same
   GPU. Mitigation: ColBERT is small enough that memory contention is unlikely, but
   may need to serialize access if OOM occurs.

5. **Indexing time.** 100K documents × 8K tokens × 149M model could take hours. Plan
   for overnight indexing run.

---

## Dependencies

```
# New Python dependencies (add to bench/browsecomp/pyproject.toml)
ragatouille >= 0.3      # or: pylate + nextplaid
torch >= 2.0            # ColBERT inference
```

---

## Success Criteria

| Outcome | Interpretation | Next step |
|---------|---------------|-----------|
| Mode B accuracy ≥ 40% (12+/30) | Retriever improvement is real with small LLM | Proceed to Phase 3-4, consider Apache 2.0 path |
| Retrieval recall@10 improves but accuracy flat | ColBERT helps retrieval, LLM is the bottleneck | Proceed to Phase 4 (model-size study) |
| No improvement across all modes | BrowseComp failures are corpus gaps, not ranking | Deprioritize retrieval, focus on ensemble (A4) |
