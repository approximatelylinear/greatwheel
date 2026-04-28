-- Phase B / Spine step 1: entry-level entity attribution + relations.
--
-- See `docs/design-semantic-spine.md` §3.1 + §3.2 for the full design.
-- Two new tables, both owned by gw-loop (which owns session_entries).
-- The link's entity side references kb_entities, but the entry side is
-- the local citizen — so cross-crate ownership stays clean.
--
-- The spine widget projects over these on read; nothing here is the
-- spine itself.

-- ─── Per-entry entity attribution ───────────────────────────────────
--
-- One row per (entry, entity) mention. Mirrors kb_chunk_entity_links
-- but with session-aware fields: role describes what the entry was
-- *doing* with the entity, status flags whether the user/agent has
-- promoted it from "mentioned in passing" to "load-bearing for the
-- workspace".
--
-- span_start / span_end let a future render pass highlight the
-- mention in-message without re-running extraction. Optional because
-- a single entry may produce a synthesised mention with no exact
-- surface form (e.g. agent narrating about an entity not literally
-- named in the prose).

CREATE TABLE session_entry_entities (
    entry_id     UUID NOT NULL REFERENCES session_entries(id)   ON DELETE CASCADE,
    entity_id    UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    surface      TEXT NOT NULL,                                 -- text span as it appeared
    role         TEXT NOT NULL DEFAULT 'referenced',            -- "introduced" | "referenced" | "decided" | "compared" | ...
    status       TEXT NOT NULL DEFAULT 'mentioned',             -- "mentioned" | "committed"
    confidence   REAL NOT NULL DEFAULT 0.5,
    span_start   INT,                                           -- char offset within entry text
    span_end     INT,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    -- COALESCE so two rows for the same (entry, entity) at different
    -- spans are distinct, but two rows with NULL span at the same
    -- (entry, entity) collapse into one.
    PRIMARY KEY (entry_id, entity_id, COALESCE(span_start, -1))
);

CREATE INDEX idx_session_entry_entities_entity ON session_entry_entities (entity_id);
CREATE INDEX idx_session_entry_entities_committed
    ON session_entry_entities (entry_id) WHERE status = 'committed';

-- ─── Per-entry typed relations between entities ─────────────────────
--
-- Distinct from kb_entity_links: those are global, symmetric,
-- co-mention/cosine-driven; these are per-entry, typed, sometimes
-- directional, with a surface span so the sidebar can show the user
-- *what* asserted the relation. A future job can promote frequently-
-- repeated assertions back into kb_entity_links — see design doc §3.2
-- "Promotion to the global graph".

CREATE TABLE session_entry_relations (
    relation_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entry_id     UUID NOT NULL REFERENCES session_entries(id)   ON DELETE CASCADE,
    subject_id   UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    object_id    UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    predicate    TEXT NOT NULL,                                 -- "compared_with" | "tradeoff_in" | "composes" | "outperforms" | "is_a" | "uses" | "evaluated_on" | ...
    directed     BOOL NOT NULL DEFAULT TRUE,                    -- false for symmetric predicates like "compared_with"
    surface      TEXT NOT NULL,                                 -- the span that asserted it
    confidence   REAL NOT NULL DEFAULT 0.5,
    span_start   INT,
    span_end     INT,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_entry_relations_entry   ON session_entry_relations (entry_id);
CREATE INDEX idx_session_entry_relations_subject ON session_entry_relations (subject_id);
CREATE INDEX idx_session_entry_relations_object  ON session_entry_relations (object_id);
CREATE INDEX idx_session_entry_relations_predicate ON session_entry_relations (predicate);
