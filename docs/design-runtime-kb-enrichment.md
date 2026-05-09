# Design: runtime KB enrichment

**Status:** Draft 2026-05-07. Sits alongside `design-kb.md` and
`design-kb-entities.md`. Describes a shift from eager
ingest-time enrichment to lazy load-time enrichment, plus the
shared statistics substrate that drives wiki-style framing,
candidate selection, and entity-card UX.

## 1. Why

`gw-kb` today does most of its expensive work at ingest:
classification, topic linking, entity linking, summary
synthesis. That cost is paid once per doc — but it is paid
**whether or not the doc is ever read**. For a wiki-style
corpus the access distribution is bimodal: a small head of
hot pages plus a long tail that is read zero or one times.
Most ingest-time LLM work on the tail is wasted.

A second pressure: the wiki UX wants per-document framing
that ingest can't easily produce in isolation —

- An autopopulated **subject card** for the 1–3 entities or
  topics a document is "about".
- A ranked list of **important other entities** in the doc, a
  handful of which deserve their own card on click.
- Per-doc **relationships** between entities, labeled with a
  semantic predicate (not just "same paragraph").

These all depend on signals that are corpus-relative
(co-occurrence, document frequency, importance) and on
LLM enrichment that is heavy enough we don't want to pay it
for every doc.

The proposal: move enrichment to the load path, gate the
LLM by UX surface area, and back the whole thing with a
deterministic statistics substrate that is cheap to maintain.

## 2. Economic shape

Lazy by default, with a popularity warmer:

- **Ingest** stays cheap and deterministic — clean, chunk,
  NER, embeddings, structural relationship candidates,
  index updates. No LLM.
- **First load** of a doc computes the structural answer
  synchronously, returns immediately, and fires LLM
  enrichment in the background. Results are cached
  permanently keyed by stable input hashes.
- **Subsequent loads** read from cache.
- **A warmer task** scans access logs and pre-enriches
  top-N docs offline, reusing the same code path.

LLM cost scales with **views of unique content**, not with
ingestion volume.

## 3. The statistics substrate

Three small, deterministic stores fed incrementally on
ingest. Everything downstream — aboutness, candidate
selection, importance — reads from these.

### Mention table

Per `(doc, entity)` row recording what the structural pass
already found:

```sql
CREATE TABLE kb_doc_entity_mentions (
    source_id        UUID NOT NULL REFERENCES kb_sources(source_id),
    entity_id        UUID NOT NULL REFERENCES kb_entities(entity_id),
    mention_count    INT  NOT NULL,
    section_spread   INT  NOT NULL,   -- distinct sections containing the entity
    in_title         BOOL NOT NULL,
    in_intro         BOOL NOT NULL,
    in_heading       BOOL NOT NULL,
    coref_chain_len  INT  NOT NULL,
    PRIMARY KEY (source_id, entity_id)
);
```

Same shape mirrors for topics. Costs nothing — these are
counts the existing entity extractor already produces.

### Co-occurrence counts

Sparse pairwise table at the granularities that matter:

```sql
CREATE TABLE kb_entity_cooccurrence (
    entity_a       UUID NOT NULL,
    entity_b       UUID NOT NULL,        -- ordered: a < b
    sentence_count INT  NOT NULL DEFAULT 0,
    paragraph_count INT NOT NULL DEFAULT 0,
    doc_count      INT  NOT NULL DEFAULT 0,
    PRIMARY KEY (entity_a, entity_b)
);

CREATE TABLE kb_entity_marginals (
    entity_id      UUID PRIMARY KEY,
    sentence_count INT  NOT NULL DEFAULT 0,
    paragraph_count INT NOT NULL DEFAULT 0,
    doc_count      INT  NOT NULL DEFAULT 0
);
```

Bound the matrix by keeping only top-K co-occurrents per
entity (K ≈ 200) plus a global threshold; everything below
collapses to "no signal". A nightly compaction job is
sufficient.

### Document frequency

Trivially derivable from the marginals (`doc_count`); kept as
a column for clarity.

PMI is then a query-time arithmetic over these tables:

$$
\text{PMI}(a,b) = \log \frac{P(a,b)}{P(a)\,P(b)}
$$

with `P` estimated from sentence-, paragraph-, or doc-level
counts depending on the call site.

## 4. Aboutness — what is the doc about?

A doc has 1–3 subject entities/topics. They drive the
framing of the wiki page and the auto-populated subject
cards.

The signals are positional and very strong:

- **Title/heading mention** (binary)
- **Intro-paragraph mention** (binary)
- **Coref chain length** — the subject has the longest
  pronoun chain
- **Section spread** — appears across many sections, not just
  one
- **TF-IDF**: `mention_count × log(N / doc_frequency)`

A linear combination produces an aboutness score per
`(doc, entity)`. Take entities above a threshold *and* with
a sharp gap to the next entity. Cap at 3. Calibrate weights
on a small held-out set.

For Wikipedia content this is mostly free: page title and
infobox already declare the subject. The heuristic ranker
is reserved for non-wiki ingestion.

Topics use the same scorer with topic IDs.

## 5. Co-occurrence as candidate selection

PMI from the substrate enters the relationship pipeline in
three places:

1. **Disambiguation.** "Apple" near "Tim Cook" vs. near
   "orchard" resolves to different canonical entities by
   matching against the cooccurrence neighborhoods of
   candidate canonical entities.
2. **Pruning before the LLM.** Candidate pairs with PMI
   near zero or negative are coincidence; drop them before
   sending to the labeler.
3. **Boosting structurally weak signals.** Two entities a
   paragraph apart with high corpus PMI become candidates
   even though their local co-occurrence is thin.

This is on top of the deterministic gates already
recommended in the architectural sketch — co-occurrence
threshold, embedding similarity, entity-type compatibility,
syntactic distance. PMI is the **corpus-level** gate; the
others are **document-level**.

## 6. Per-doc entity importance

Importance is `(doc, entity) → score`, not per-entity.
The same entity is central in one doc and incidental in
another.

Components:

| Signal | Source |
|---|---|
| Mention count, spread, position weighting | mention table |
| Coref chain length | NER+coref pass |
| Heading / intro / conclusion bonus | structural tree |
| TF-IDF (`mention × log(N/df)`) | mention + DF tables |
| PMI to the doc's subject entities | cooccurrence table |
| Local-graph centrality (PageRank on the doc's entity-cooccurrence subgraph) | computed per-doc |
| Subject-prior (how often this entity is the subject of *some* doc) | aboutness output, aggregated |

The two underrated signals:

- **Section spread** cleanly separates "topic that runs
  through the doc" from "topic that's mentioned once in a
  tangent" — the best filter against frequent-but-incidental
  entities.
- **PMI-to-subject** surfaces "the things this doc connects
  its subject to" — exactly what an entity card next to the
  subject card should show.

UX gates fall out as percentile cuts on the score:

- **Top 1–3** → subject cards, autopopulated, summary
  precomputed at first load.
- **Top 4–~12** → important neighbors. Listed inline with
  affordance to expand; card materialized lazily on click
  and cached.
- **Tail** → inline link only. No card unless explicitly
  requested.

## 7. Load-time pipeline

`fetch_wiki_doc` (already the entry point in
`crates/gw-kb/src/wiki.rs`) becomes the trigger for the
whole flow. Four tiers:

### 7.1 Synchronous, no LLM (~tens of ms)

- Pull mentions, structural tree, co-occurrences from substrate
- Compute aboutness → 1–3 subjects
- Compute importance ranking → top neighbors
- Resolve enclosing context (sentence → paragraph → list
  item → section) for every shown entity
- Build structural relationship candidates among
  `{subjects} ∪ {top neighbors}`, gated by deterministic
  filters and corpus PMI
- Render skeleton: doc body, subject placeholders, ranked
  entity list

The page is interactive at this point.

### 7.2 Background, batched LLM (a few hundred ms each)

- **Subject summaries** — 1–3 entities, one batched call
- **Relation labeling** — single call per context, taking
  the full pruned candidate set and emitting a JSON list
  of triples. Constrained decoding onto a closed relation
  vocabulary; the "retry on generic labels" pattern goes
  away when the output space is bounded.
- Stream results into the page over SSE / agent events as
  each completes.

### 7.3 Lazy on click (cached after first compute)

- Neighbor card → summary + that entity's relation set in
  this doc.

### 7.4 Never (until accessed)

- Tail entities. They exist in the substrate but no
  per-doc LLM artefacts are produced.

## 8. LLM workload sizing

Bounded by UX surface area, not document length. A 200-entity
doc and a 20-entity doc cost roughly the same at first load,
because both render ~3 subject cards and ~10 candidate
neighbors. The expensive quadratic candidate set never
reaches the model — corpus PMI plus structural gates collapse
it before the call.

Model selection follows from the task shape:

- **Relation labeling** is closer to classification than
  generation. A 0.5–1B model with logit biasing onto a
  fixed relation vocabulary (or a dedicated RE model in the
  REBEL family) runs sub-100ms. The current 7B-class model
  is doing more work than the task requires.
- **Subject summaries** are short and genuinely generative;
  a mid-sized model is fine, scoped to ≤80 tokens per
  summary, batched per doc.

## 9. Caching and invalidation

Cache keys must capture every input that could change the
output:

- `relation_label`: `(entity_a, entity_b, context_hash, schema_version, model_id)`
- `summary`: `(entity_id, context_hash, schema_version, model_id)`
- `aboutness`: `(source_id, substrate_version)`
- `importance_ranking`: `(source_id, substrate_version)`

`schema_version` advances when the relation vocabulary or
summary spec changes. `substrate_version` advances when the
co-occurrence/marginals tables are recompacted in a way
that materially shifts scores. `model_id` is bumped on model
swap. Bake all of these into keys from day one — retrofitting
is painful.

A request-coalescing layer above the cache deduplicates
concurrent first-loads of the same doc; one compute job
serves all waiters.

## 10. Migration from current gw-kb

The pieces already in place:

- `linking.rs` — `link_entities`, `link_topic_entities`,
  `spread_from_seeds`, `neighbors_of`. Today these run at
  ingest; under this design they shift to load-time triggers
  fed from the substrate.
- `wiki.rs::fetch_wiki_doc` — already the load-time entry
  point. It currently returns structural data; it grows the
  responsibility of orchestrating the four-tier pipeline.
- `kb_entities`, `kb_entity_links`, `kb_topic_entity_links`,
  `kb_chunk_entity_links` — the entity graph this design
  layers on top of.

What's new:

- `kb_doc_entity_mentions`, `kb_entity_cooccurrence`,
  `kb_entity_marginals` tables (Section 3).
- An aboutness scorer and an importance scorer
  (`crates/gw-kb/src/aboutness.rs`,
  `crates/gw-kb/src/importance.rs`), both pure functions
  over the substrate plus the structural tree.
- A relation-extraction call shape that takes a context and
  a candidate set and returns triples (replaces the per-pair
  loop).
- A load-time orchestrator in `wiki.rs` that fans out the
  four tiers and writes the cache.
- A popularity-driven warmer task (background job in
  `gw-server`) that reuses the same orchestrator.

Migration order:

1. Add the substrate tables and start populating them on
   ingest. Keep the existing eager pipeline running; the
   substrate is purely additive.
2. Implement the aboutness and importance scorers as
   read-only computations. Validate offline against a
   sample set.
3. Move the existing `linking.rs` calls behind the
   load-time orchestrator. Leave a feature flag so eager
   ingestion can be re-enabled per source.
4. Land the batched relation-extraction call shape. Retire
   per-pair labeling.
5. Add the warmer.
6. Once the lazy path is healthy, stop running the eager
   path on new ingests by default.

## 11. Risks and open questions

- **Cold-start.** Aboutness, importance, and PMI all need
  corpus statistics. A Wikipedia-only ingest gives them on
  day one. For mixed corpora the first few hundred docs
  have noisy scores; cache per-doc and recompute lazily on
  re-view when the substrate has materially shifted.
- **First-view latency on cold pages** is worse than
  today's eager system unless the streaming UX is solid.
  The page must be useful with structural-only data; the
  LLM tier upgrades but does not gate.
- **Hot-doc thundering herd.** A trending doc on a cold
  cache could fire many concurrent enrichment jobs. The
  request-coalescing layer is not optional.
- **Cache invalidation correctness** depends on getting
  the version keys right at the start. The cost of a missed
  dimension is silent stale answers.
- **Section-spread metric** depends on the structural tree
  being trustworthy. HTML cleaning quality bounds importance
  quality.
- **Subject-prior is recursive** — it depends on the
  aboutness output of other docs. Compute it as an offline
  aggregate on a lag, not synchronously.
- **Aboutness for non-wiki content** is the hardest case.
  News articles, papers, and blog posts have weaker title
  conventions than wiki pages. The heuristic ranker may
  need a small classifier head trained on a labeled set.
