# Design: extending gw-kb with typed entities

**Status:** Drafted 2026-04-26 alongside the literature-assistant
v1 demo. The demo currently runs an in-memory pipeline (arXiv →
embed → PCA → cloud). This doc plans the path to v2, where the
cloud is rendered from a persistent entity graph in `gw-kb`.

## 1. Why

The literature MVP has three real limitations that all trace to
the same root cause — there is no persistent representation of the
entities it discovers:

1. **No memory between turns.** Click a point, agent re-searches
   arXiv, builds the detail view from a fresh fetch. We re-pay
   the API + embedding cost on every interaction.
2. **No entities, just papers.** The MVP plots one point per paper.
   "Authors", "concepts", "methods", "datasets" — the things a
   researcher actually navigates by — don't exist as nodes.
3. **No cross-corpus connection.** Papers found via topic A and
   papers found via topic B never share a node. Two queries that
   should overlap ("retrieval-augmented generation" and "RAG")
   produce two unrelated clouds.

`gw-kb` already solves these problems for **topics**:

- Persistent storage in Postgres (`kb_topics`, `kb_chunks`,
  `kb_topic_links`).
- An ingest pipeline (`extract` → `chunk` → `classify` → `organize`
  → `link`) that turns raw documents into a typed topic graph.
- A typed graph from `linking.rs` — co-occurrence + cosine-similarity
  edges with merged confidence.
- Three host functions (`kb_search`, `kb_explore`, `kb_topic`) that
  agents already use.

Adding **entities** as a sibling node type — using the same
infrastructure — turns the literature assistant from a one-shot
search-and-cluster tool into a real knowledge browser.

## 2. The new node type

### Schema

```sql
CREATE TABLE kb_entities (
    entity_id   UUID PRIMARY KEY,
    label       TEXT NOT NULL,           -- canonical display name
    slug        TEXT NOT NULL UNIQUE,    -- lowercase-hyphenated, stable
    kind        TEXT NOT NULL,           -- "author" | "concept" | "method" | "dataset" | "venue" | ...
    aliases     TEXT[] NOT NULL DEFAULT '{}',
    mentions    INT NOT NULL DEFAULT 0,
    vector      BYTEA,                   -- f32[] of the canonical embedding
    summary     TEXT,                    -- LLM-synthesized, optional
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX kb_entities_kind ON kb_entities(kind);
CREATE INDEX kb_entities_slug ON kb_entities(slug);
```

Modelled directly on `kb_topics`. `vector` is a serialized `Vec<f32>`
(same encoding as topic vectors) so we can do linear-scan cosine
similarity in memory.

### Edges

Three new edge tables, all symmetric like `kb_topic_links`:

```sql
CREATE TABLE kb_entity_links (
    entity_id_a UUID NOT NULL REFERENCES kb_entities(entity_id),
    entity_id_b UUID NOT NULL REFERENCES kb_entities(entity_id),
    confidence  REAL NOT NULL,
    relation    TEXT NOT NULL DEFAULT 'related',
    PRIMARY KEY (entity_id_a, entity_id_b),
    CHECK (entity_id_a < entity_id_b)
);

CREATE TABLE kb_topic_entity_links (
    topic_id    UUID NOT NULL REFERENCES kb_topics(topic_id),
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id),
    confidence  REAL NOT NULL,
    PRIMARY KEY (topic_id, entity_id)
);

CREATE TABLE kb_chunk_entity_links (
    chunk_id    UUID NOT NULL REFERENCES kb_chunks(chunk_id),
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id),
    role        TEXT,                    -- "subject" | "object" | "mention"
    PRIMARY KEY (chunk_id, entity_id)
);
```

`kb_chunk_entity_links` is the source of truth for "this entity
appears in this passage." `kb_topic_entity_links` is derivable from
chunk membership (any chunk that belongs to topic T and mentions
entity E contributes evidence). `kb_entity_links` is computed by
extending `linking.rs` with co-mention + cosine.

### Why same infrastructure

- **Vector storage:** identical bytes encoding to topics, so
  `bytes_to_vec` / `cosine` from `topics.rs` reuse directly.
- **Linker:** `linking.rs::link()` already computes co-occurrence
  (Jaccard over chunk membership) + cosine. New entrypoints
  `link_entities()` and `link_topic_entities()` follow the same
  pattern, sharing helpers.
- **Search:** entity vectors live in the same table-scan-friendly
  shape, so `kb_explore`'s spreading-activation walk extends to
  cross-type traversal (topic → entity → topic) without needing a
  new data layer.

## 3. Extraction

This is the hard part. Two viable strategies; we'll likely run both.

### 3.1 LLM-prompted (high quality, expensive)

Per chunk, prompt: *"List named entities from {Author, Concept,
Method, Dataset, Venue}. Output JSON with `label`, `kind`,
`canonical_form`, `confidence`."* The existing `llm_parse.rs` module
has the JSON-output discipline; add an `extract_entities()` helper
following the same pattern.

Cost: ~one LLM call per chunk. For a 200-paper corpus chunked
into ~600 abstracts, that's 600 calls — acceptable as a one-shot
ingest. Embeddings + UMAP in the existing pipeline already pay
similar costs.

### 3.2 spaCy + rules (fast, lower ceiling)

For people / orgs / dates / locations, spaCy's NER is competitive
and free. The existing `extract.rs` already calls Python via
`pyo3` (trafilatura, pymupdf4llm) so the runtime is in place.
Add a step that runs spaCy NER and feeds typed mentions to the
LLM step for verification + canonicalisation.

### 3.3 Canonicalisation

Same entity, many mentions: "Lewis et al.", "P. Lewis", "Patrick
Lewis" all → entity `lewis-patrick`. Approach:

1. **Embed each surface form** (the cheap part — one vector per
   mention).
2. **Cluster mentions by cosine ≥ 0.9** (per-kind threshold; tunable).
3. **Pick the longest surface form as canonical label**, the others
   as `aliases`.

This is a separate step in the pipeline, between extraction and
linking. Borrows directly from `merge.rs`'s topic-merge logic.

## 4. New host functions

Two additions, parallel to the existing topic ones:

```python
kb_entities(kind: str | None = None, k: int = 50) -> list[dict]
# All entities, optionally filtered by kind. Returns
# {entity_id, label, kind, mentions, slug, summary}.
# Used to populate the entity cloud at session start.

kb_entity(slug: str) -> dict | None
# Full entity record + linked topics + linked entities + sample
# chunks. Returns None if not found.
```

Plus `kb_explore` extends to take a starting entity slug and walk
the cross-type graph (entity → topic → entity → ...). The signature
stays the same; the implementation gets a `seed_kind` parameter
internally.

## 5. The literature demo, v2

Same UI, different data source:

| | v1 (current) | v2 (gw-kb-backed) |
|---|---|---|
| Source | live arXiv search | pre-ingested KB |
| Nodes | papers | entities (authors, concepts, methods, datasets) |
| Coords | PCA over abstracts | UMAP / PCA over entity vectors |
| Memory | none | persistent (Postgres) |
| Click | re-fetch from arXiv | `kb_entity(slug)` |
| Cross-corpus | no | yes — entities aggregate across queries |

The frontend `EntityCloud` widget shape stays identical:
`{points: [{id, label, x, y, kind}], highlight}`. The host
function plumbing changes, the rendering doesn't.

## 6. Implementation order

1. **Schema migration** for `kb_entities` and the three link tables.
2. **`kb-extract` extension** — add `extract_entities` to the
   ingest pipeline, behind a feature flag so existing topic-only
   ingests stay unchanged.
3. **`linking.rs` extension** — add `link_entities` and
   `link_topic_entities` functions. Reuse the cosine + co-occurrence
   helpers verbatim.
4. **Plugin host fns** — add `kb_entities` and `kb_entity` to
   `gw-kb/src/plugin.rs`. Same `kb.read` capability gate as the
   existing fns.
5. **Demo migration** — replace `arxiv_search` / `embed_papers` /
   `project_2d` in `literature_assistant.rs` with a thin wrapper
   that calls `kb_entities` + `project_2d` on the entity vectors.
6. **Ingest a paper corpus** — one-shot CLI run that ingests, e.g.,
   the most-cited 500 NeurIPS papers via the existing arXiv source.
   Lives as a `cargo run -p gw-kb --bin ingest_arxiv` invocation.

Each step is independently shippable; the demo can run on whatever
state the KB is in.

## 7. Open questions

- **Kind taxonomy stability.** Five kinds (`author / concept /
  method / dataset / venue`) is good for ML papers; biology or
  history might want different kinds. Make `kind` a free-form
  string with a recommended set, not an enum.
- **Cross-kind embeddings.** Author embeddings come from the
  contexts they appear in; concept embeddings come from definitions;
  these may live in different parts of the embedding space. Worth
  an experiment: do cross-kind cosine distances make sense, or
  should we keep separate per-kind embedding spaces?
- **Edge confidence calibration.** Topic edges are calibrated from
  Jaccard + cosine; entity edges might want different weights
  (co-mention is a noisier signal than topic co-occurrence because
  the same paper mentions many entities). Tune empirically.
- **Author disambiguation.** "P. Lewis" is at least three different
  researchers. Embedding-based clustering catches surface-form
  collapse but not name collisions. Punt to v3 — orcid integration
  is the right answer.
- **Synonym resolution at query time.** "RAG" and "retrieval-
  augmented generation" should resolve to the same entity. The
  canonicalisation step handles this at ingest, but not at query
  time. Likely needs an `kb_entity_search(label)` host fn that
  does fuzzy + alias matching.

## 8. What this doc is NOT

Not a near-term commitment. The literature MVP works without any of
this; the gw-kb extension is the long-term path that lets the demo
graduate from "search-and-cluster" to "browse the knowledge graph
the agent has built up."

When we do start, schema migration + linking are the steady,
predictable part. Extraction quality is where time will go — likely
several iterations on the LLM prompt + canonicalisation thresholds
before the cloud looks honest.
