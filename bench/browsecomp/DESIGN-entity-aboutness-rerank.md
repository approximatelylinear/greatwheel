# Design: Entity-Aware MiniLM Aboutness Reranking

**Status:** Proposed
**Date:** 2026-04-25

---

## Motivation

Our hybrid retriever (BM25 + dense + RRF) is structurally bad at distinguishing
**aboutness** from **mentions**.

Concrete failure mode: query "who was Charlemagne", corpus contains articles on
many Frankish kings. Almost every article mentions Charlemagne in passing. A
long, well-written biography of Pepin the Short — high topical overlap, several
Charlemagne mentions — outranks the actual Charlemagne article, which may be
shorter and have less aggregate Frankish-kings vocabulary.

Why each retriever fails:

- **BM25** rewards term frequency relative to length. Long articles in the same
  topic that mention the entity outscore short focal articles.
- **Single-vector dense** collapses a document to one point dominated by its
  primary subject. Any document in the same topic cluster lands near the query;
  the focal article is not preferentially closer.
- **RRF fusion amplifies the failure.** Both retrievers agree the Pepin article
  is "good" (it's topical and contains the entity), so RRF promotes it further.

This is the textbook *focal-entity ranking* / *salience* problem in IR. It is
distinct from the recall failure mode that Reason-ModernColBERT addresses
(see `DESIGN-reason-colbert.md`) — here the gold doc is *in* the candidate
set, but ranked below tangentially-relevant peers.

## Hypothesis

A small cross-encoder reranker (MiniLM-class, ~22-33M params) can be steered
toward focal aboutness without fine-tuning by manipulating its inputs and
scoring formula:

1. **Entity-header injection.** Prepend a synthesized aboutness header derived
   from per-doc entity salience to the passage MiniLM cross-attends over.
2. **Contrastive sibling-entity scoring.** Score the candidate against the
   focal query *and* against decoy queries built from sibling entities; rank
   by the gap. A passage that scores high for "who was Charlemagne" and also
   high for "who was Pepin" is a generic Frankish-kings article, not focal.

The conjecture: these two interventions together close most of the gap to a
purpose-built aboutness model, at MiniLM latency (1-10ms/pair on CPU).

## Why MiniLM and not Qwen3-Reranker

Qwen3-Reranker-0.6B is instruction-tuneable and would let us prompt for
aboutness directly — but it costs 50-200ms/pair. MiniLM is 10-50x faster, which
matters because the contrastive scoring formula multiplies pair count by
(1 + n_siblings). At n_siblings=3 we run 4x as many reranker calls per
candidate, so per-pair cost dominates.

If the input-engineering tricks fail, escalation path is Qwen3-Reranker with
an explicit "primarily about" instruction (no contrastive scoring needed).

## Design

### Components

| File                                              | Role                                                                       |
|---------------------------------------------------|----------------------------------------------------------------------------|
| `entity_extractor.py`                             | Per-doc primary entities + salience scores (offline).                      |
| `entity_header.py`                                | Renders aboutness header from entity record at index time.                 |
| `sibling_index.py`                                | Maps entity → k nearest sibling entities (offline, by type/category).      |
| `searchers/aboutness_reranker.py`                 | `Reranker` impl: MiniLM with header injection + contrastive scoring.       |
| `retrieval_benchmark_v2.py` (`+aboutness` flag)   | Wires reranker on top of any existing first-stage searcher.                |

The reranker is composable: it sits on top of *any* first-stage searcher
(BM25, qwen3-embed, ColBERT, hybrid). The benchmark harness already supports
chaining; we add `--rerank aboutness` as a post-stage flag.

### Per-doc entity record (built at index time)

For each document, store:

```json
{
  "doc_id": "wiki_12345",
  "primary_entities": ["Charlemagne", "Carolingian dynasty"],
  "mentioned_entities": ["Pepin the Short", "Frankish kingdom", "Aachen", ...],
  "title": "Charlemagne",
  "lede": "Charlemagne (c. 748 – 814 AD), also known as Charles the Great..."
}
```

`primary_entities` is the salience signal. Generation: one LLM pass per doc at
ingest, prompt `"List up to 5 entities or topics this document is primarily
about (not merely mentions). Return JSON array."` This is the
GUM-SAGE-style operationalization — graded entity salience as one extra LLM
call per doc, cached forever.

For the BrowseComp-Plus corpus (~100K docs) on a local Ollama qwen3.5:9b at
~20 docs/min, this is ~80 hours of compute — a one-time overnight run, or
parallelize across workers. Using a smaller extraction model (qwen2.5:3b)
should bring it under 8 hours at acceptable quality.

### Entity-header injection (input-engineering trick)

What MiniLM normally sees: `(query, passage_text)`.

What MiniLM sees with header injection:

```
Primary subject: Pepin the Short.
Other entities mentioned: Charlemagne, Frankish kingdom, Carolingian dynasty.
---
[original passage text]
```

The header is short (typically <40 tokens) and high-signal. When the query is
"Charlemagne", the cross-attention sees "Primary subject: Pepin" and steers
the relevance score down. For the actual Charlemagne article, the header reads
"Primary subject: Charlemagne" and reinforces.

This is a hack — MiniLM was trained on raw MS MARCO passages, not annotated
ones — but cross-encoders cross-attend over whatever you give them. The header
costs ~zero (string concat at index time, stored alongside the passage).

### Contrastive sibling-entity scoring

For each candidate, compute:

```python
score_focal = MiniLM(query_focal, passage_with_header)
score_decoys = [MiniLM(query_decoy_i, passage_with_header) for i in siblings]
aboutness = score_focal - max(score_decoys)
```

`query_focal` is the original query rewritten around the linked entity:
`"who was {entity}"` if the original query is a who/what lookup, else the
original query verbatim.

`query_decoy_i` is the same template instantiated with a sibling entity:
`"who was Pepin"`, `"who was Charles Martel"`, etc.

Sibling selection (offline, in `sibling_index.py`):

- For each entity in our index, find k=3 nearest sibling entities by
  Wikidata type/category match (e.g. other "person" entities of subclass
  "monarch" sharing a parent class).
- Fallback when Wikidata isn't available: use embedding-space neighbors of the
  entity surface form within the same broad cluster.
- Cache as `entity_id → [sibling_id, ...]`.

The aboutness score is a *gap*, not a sum. A passage with `score_focal=0.9,
max(decoys)=0.2` (gap 0.7) is genuinely about the focal entity. A passage with
`score_focal=0.85, max(decoys)=0.80` (gap 0.05) is generically about the topic.

### Final ranking formula

```python
# After first-stage retrieval (top 50)
final_score = (
    1.0 * aboutness +
    0.3 * score_focal +              # absolute relevance, not just gap
    0.5 * primary_topic_match +      # 1.0 if query entity in primary_entities, else 0
    0.2 * title_contains_entity      # 1.0 if entity in title, else 0
)
```

Weights are starting points; tune via grid search on the focal eval set
(below). The `primary_topic_match` and `title_contains_entity` signals are
nearly free and provide a strong structural prior alongside the MiniLM gap.

## Integration into existing pipeline

`retrieval_benchmark_v2.py` already supports `--searchers` for first-stage
selection. Add:

```
--rerank aboutness          # apply this design's reranker on top-50
--rerank-config k=3,sibling_source=wikidata,model=ms-marco-MiniLM-L-6-v2
```

The reranker is searcher-agnostic. We test it on top of:

1. tantivy (BM25) — current production baseline
2. qwen3-embed (single-vector dense) — see `DESIGN-qwen3-embed.md`
3. ColBERT — see `DESIGN-reason-colbert.md`
4. tantivy + qwen3-embed RRF hybrid

This isolates whether the aboutness fix is additive across first-stage
retrievers or only helps where the first-stage was already aboutness-blind.

## Evaluation

### Standard retrieval metrics will not show this

R@k and NDCG against BrowseComp-Plus gold judgments will be roughly flat or
mildly improved. The Pepin article is "relevant" by topical-overlap labels.
We need a focused eval set.

### Focal-eval-30: a hand-curated aboutness probe

Build a 30-50 query set where the corpus contains both:

- a focal document whose primary subject is the query entity, and
- 3+ sibling documents that mention the query entity but are about something
  related (sibling kings, sibling companies, sibling concepts).

Hand-label the focal doc per query. Metrics:

- **Focal@1**: is the focal doc at rank 1?
- **Focal@3**: is the focal doc in top-3?
- **Mention-leak rate**: fraction of top-3 spots taken by mention-only docs.

This eval set is cheap to build (an afternoon of curation against the
BrowseComp-Plus corpus) and is the only thing that will reveal the fix.

### End-to-end agent eval

Run sample30 with each retrieval mode (with/without aboutness reranker)
through the existing qwen3.5:9b agent. Hypothesis: aboutness reranker
improves accuracy on entity-lookup queries (a subset of sample30) without
regressing on reasoning-intensive queries.

## Run plan

### Phase 0 — Build focal-eval-30 (1 day)

Curate the focal eval set against BrowseComp-Plus. Without this, no later
phase has a metric.

### Phase 1 — Per-doc entity records (1-2 days)

1. Implement `entity_extractor.py` against Ollama (qwen2.5:3b primary, qwen3.5:9b
   for spot-checks).
2. Run on a 1000-doc sample, validate `primary_entities` lists by hand against
   ~30 docs.
3. Run on full corpus (background, overnight).

**Decision gate:** does manual inspection on 30 docs show the LLM correctly
identifying the primary subject vs incidental mentions? If <80%, escalate to
qwen3.5:9b and rerun on the sample before committing to full-corpus extraction.

### Phase 2 — Sibling index (0.5 day)

1. Implement Wikidata-based sibling lookup for entities present in
   `primary_entities` across the corpus.
2. Build entity → siblings cache. Spot-check 20 entities; siblings should be
   things a human would recognize as "same category."

### Phase 3 — Reranker implementation + focal eval (1-2 days)

1. Implement `aboutness_reranker.py` with header injection and contrastive
   scoring.
2. Wire into `retrieval_benchmark_v2.py` via `--rerank aboutness`.
3. Run on focal-eval-30 with each first-stage searcher.

**Decision gate (the critical one):** does the reranker lift Focal@1 by ≥20
percentage points over the no-rerank baseline on focal-eval-30? If yes, the
input-engineering hypothesis is confirmed and we proceed. If no, ablate:

- Run with header injection only (no contrastive). If this alone moves the
  metric, the contrastive trick is unnecessary complexity.
- Run with contrastive only (no header). If this alone moves the metric, the
  header is unnecessary.
- If neither moves it, the MiniLM ceiling is the bottleneck — escalate to
  Qwen3-Reranker-0.6B with an explicit aboutness instruction.

### Phase 4 — End-to-end sample30 eval (1 day)

Run sample30 with `--searchers tantivy --rerank aboutness` and
`--searchers qwen3 --rerank aboutness` against the no-rerank baselines.
Measure agent accuracy and per-query category breakdown (entity-lookup vs
reasoning).

### Phase 5 — Tuning (optional, 1-2 days)

Grid search over:
- `n_siblings ∈ {2, 3, 5}`
- final-score weights
- header verbosity (entities-only vs entities + 1-line summary)

Only worth doing if Phase 3-4 results are promising.

## Risks

1. **Entity extraction quality dominates everything.** If the LLM is bad at
   identifying primary subjects on our corpus, the header injection becomes
   noise. Mitigation: validate on a hand-labeled sample before full extraction.

2. **Sibling-entity selection is hard for non-Wikipedia corpora.** Wikidata
   coverage is excellent for historical/biographical content, weaker for
   industry-specific or technical content. Mitigation: embedding-space
   fallback; accept that contrastive scoring degrades to noise where siblings
   are bad and rely on header injection alone.

3. **MiniLM may not respond to header injection.** It was trained on raw
   passages; the synthesized header is out-of-distribution. The cross-attention
   might just ignore it. Mitigation: ablation in Phase 3 will reveal this; if
   so, escalate to Qwen3-Reranker which is instruction-tuned and *will* respond
   to a "Primary subject: X" cue.

4. **Aboutness improvements may not translate to BrowseComp-Plus accuracy.**
   BrowseComp queries are reasoning-heavy, not pure entity lookups. The
   aboutness fix may move focal-eval-30 dramatically while moving sample30
   only modestly. This is fine — focal-eval-30 is the metric for this design;
   sample30 lift is a secondary outcome.

5. **Latency budget.** At n_siblings=3 we run 4x MiniLM calls per candidate.
   At top-50 candidates and ~5ms/pair on CPU, that's ~1s per query — acceptable
   for interactive use, expensive for batch eval. Mitigation: batch all
   (query, passage) pairs across siblings into one MiniLM forward pass.

## Dependencies

```toml
# bench/browsecomp/pyproject.toml additions
sentence-transformers = ">=3.0"  # ms-marco-MiniLM-L-6-v2 reranker
```

Wikidata sibling lookup uses the public SPARQL endpoint (no auth, rate
limited — cache aggressively).

## Success Criteria

| Outcome                                                       | Interpretation                                       | Next step                                                       |
|---------------------------------------------------------------|------------------------------------------------------|-----------------------------------------------------------------|
| Focal@1 lifts ≥20pp over no-rerank baseline                   | Aboutness reranking works as designed                | Phase 4-5; consider productionizing                             |
| Focal@1 lift comes mainly from header injection, not contrast | Simpler design wins                                  | Drop sibling index, ship header-only reranker                   |
| Focal@1 lift comes mainly from contrast, not header           | Contrastive scoring is the real signal               | Drop header, ship contrast-only; revisit header for Qwen3       |
| Focal@1 flat with MiniLM                                      | MiniLM ceiling is the bottleneck                     | Escalate to Qwen3-Reranker-0.6B with aboutness instruction      |
| Focal@1 lifts but sample30 accuracy flat                      | Aboutness fix doesn't address BrowseComp's failures  | Ship for citation use case; don't claim sample30 win            |
