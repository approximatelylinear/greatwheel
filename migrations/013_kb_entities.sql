-- Phase B Step 1: typed entities as a sibling node type to topics.
--
-- See `docs/design-kb-entities.md` for the full design. This migration
-- adds four tables that mirror the topic schema:
--
--   kb_entities             — author / concept / method / dataset / venue / ...
--   kb_entity_links         — symmetric edges between entities (co-mention + cosine)
--   kb_topic_entity_links   — bridge between the topic graph and the entity graph
--   kb_chunk_entity_links   — source-of-truth: "this entity appears in this passage"
--
-- The topic graph (kb_topics + kb_topic_chunks + kb_topic_links from
-- migration 009) is unchanged. Entities sit alongside it; the linker
-- (extending crates/gw-kb/src/linking.rs in a follow-up) walks both
-- types to support cross-kind spreading activation.

-- ─── Entities ────────────────────────────────────────────────────────

CREATE TABLE kb_entities (
    entity_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label       TEXT NOT NULL,                              -- canonical display name
    slug        TEXT NOT NULL UNIQUE,                       -- lowercase-hyphenated, stable
    kind        TEXT NOT NULL,                              -- "author" | "concept" | "method" | "dataset" | "venue" | ...
    aliases     TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],    -- alternate surface forms collapsed by canonicalisation
    mentions    INT NOT NULL DEFAULT 0,                     -- running count maintained by ingest
    vector      BYTEA,                                      -- f32[] of the canonical embedding (same encoding as kb_topics.vector)
    summary     TEXT,                                       -- LLM-synthesised, optional
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_kb_entities_kind ON kb_entities (kind);
CREATE INDEX idx_kb_entities_label ON kb_entities (lower(label));

-- ─── Entity ↔ entity edges ───────────────────────────────────────────
--
-- Symmetric: the CHECK constraint enforces (a < b) so each undirected
-- edge has exactly one row. Reverse lookup is covered by a second
-- index on the b side.

CREATE TABLE kb_entity_links (
    entity_id_a UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    entity_id_b UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    relation    TEXT NOT NULL DEFAULT 'related',            -- free-form (e.g. "co_author", "uses_dataset", "introduces", "extends")
    confidence  REAL NOT NULL DEFAULT 0.5,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entity_id_a, entity_id_b),
    CHECK (entity_id_a < entity_id_b)
);

CREATE INDEX idx_kb_entity_links_b ON kb_entity_links (entity_id_b);

-- ─── Topic ↔ entity bridge ───────────────────────────────────────────
--
-- Derivable in principle from chunk membership (any chunk in topic T
-- mentioning entity E contributes evidence) but materialised here so
-- spreading-activation walks don't have to fan out through chunks
-- on every hop.

CREATE TABLE kb_topic_entity_links (
    topic_id    UUID NOT NULL REFERENCES kb_topics(topic_id)   ON DELETE CASCADE,
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    confidence  REAL NOT NULL DEFAULT 0.5,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (topic_id, entity_id)
);

CREATE INDEX idx_kb_topic_entity_links_entity ON kb_topic_entity_links (entity_id);

-- ─── Chunk ↔ entity (source of truth) ────────────────────────────────
--
-- One row per (chunk, entity) mention. `role` is optional and lets
-- callers distinguish e.g. an author who *wrote* a paper from one
-- merely *cited* in it. `kb_chunks.entities TEXT[]` from migration 010
-- stays as-is for now — it's the raw extractor output; this table is
-- the canonicalised, FK-checked form.

CREATE TABLE kb_chunk_entity_links (
    chunk_id    UUID NOT NULL REFERENCES kb_chunks(chunk_id)    ON DELETE CASCADE,
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    role        TEXT,                                          -- "subject" | "object" | "mention" | NULL
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (chunk_id, entity_id)
);

CREATE INDEX idx_kb_chunk_entity_links_entity ON kb_chunk_entity_links (entity_id);
