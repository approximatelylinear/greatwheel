-- Per-occurrence entity mentions with character spans into chunk content.
--
-- See `docs/design-kb-wiki-provenance.md` §3.1 for the design. This is
-- milestone 1: highlights on normalized markdown. Provenance back to
-- original-source coordinates (PDF bbox, HTML DOM) lands in a separate
-- table during M2.
--
-- The pre-existing kb_chunk_entity_links table (migration 013) was
-- (chunk_id, entity_id) → one row, which threw away per-occurrence
-- position. This migration:
--   1. Creates kb_entity_mentions — one row per occurrence with spans
--      into chunk.content (UTF-8 byte offsets).
--   2. Migrates existing kb_chunk_entity_links rows as span-less
--      mentions (NULL norm_start/norm_end/surface). They render as
--      sidebar entries without inline highlights until a backfill pass
--      re-extracts spans.
--   3. Drops the old table and recreates it as a view over the
--      mentions table so existing readers (wiki.rs, plugin.rs,
--      linking.rs) need no code changes.

-- ─── kb_entity_mentions ─────────────────────────────────────────────

CREATE TABLE kb_entity_mentions (
    mention_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id    UUID NOT NULL REFERENCES kb_chunks(chunk_id)    ON DELETE CASCADE,
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    -- UTF-8 byte offsets into kb_chunks.content. NULL on legacy rows
    -- migrated from kb_chunk_entity_links before spans existed; those
    -- rows are visible to the sidebar but not to inline highlighting.
    -- After a re-extraction backfill (`gw_kb backfill-mentions`), all
    -- rows should have non-NULL spans.
    norm_start  INT,
    norm_end    INT,
    surface     TEXT,
    role        TEXT,                              -- carried over from kb_chunk_entity_links
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (norm_start IS NULL AND norm_end IS NULL AND surface IS NULL)
        OR
        (norm_start IS NOT NULL AND norm_end IS NOT NULL AND surface IS NOT NULL
         AND norm_end > norm_start AND norm_start >= 0)
    )
);

-- Read by frontend highlight overlay: "give me all mentions in this
-- chunk, sorted by position". Composite covers the common WHERE +
-- ORDER BY pattern.
CREATE INDEX idx_kb_entity_mentions_chunk_pos
    ON kb_entity_mentions (chunk_id, norm_start);

-- Read by entity-detail card: "where does this entity appear?"
CREATE INDEX idx_kb_entity_mentions_entity
    ON kb_entity_mentions (entity_id);

-- ─── Migrate existing kb_chunk_entity_links data ────────────────────
--
-- Preserve the chunk → entity edges that the linker has already
-- canonicalised. Spans land NULL; backfill repopulates them later.

INSERT INTO kb_entity_mentions (chunk_id, entity_id, role, created_at)
SELECT chunk_id, entity_id, role, created_at
FROM kb_chunk_entity_links;

-- ─── Replace kb_chunk_entity_links with a view ──────────────────────
--
-- Existing readers (wiki.rs:427, plugin.rs:605, linking.rs:318,
-- linking.rs:501) only project chunk_id / entity_id / role, so a
-- DISTINCT projection over the mentions table is observationally
-- equivalent to the old table. The view is read-only — the single
-- writer (entities.rs::insert_chunk_entity_link) is being replaced
-- with direct mentions-table writes in the same patch series.

DROP TABLE kb_chunk_entity_links;

CREATE VIEW kb_chunk_entity_links AS
SELECT chunk_id,
       entity_id,
       -- role: pick any non-NULL role for the (chunk, entity) pair.
       -- The old PK was (chunk_id, entity_id) so there was at most
       -- one role per pair; the new schema allows multiple mentions
       -- with potentially different roles. We surface a representative.
       (array_agg(role) FILTER (WHERE role IS NOT NULL))[1] AS role,
       MIN(created_at)                                       AS created_at
FROM kb_entity_mentions
GROUP BY chunk_id, entity_id;

-- The old idx_kb_chunk_entity_links_entity index dies with the table.
-- Equivalent coverage is provided by idx_kb_entity_mentions_entity
-- above, which the view's GROUP BY will use.
