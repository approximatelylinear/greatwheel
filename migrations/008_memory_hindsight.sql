-- Hindsight-inspired structured memory: typed facts, temporal metadata, entities, and graph edges.
-- See docs/design-hindsight-memory.md for details.

-- Memory kind enum
CREATE TYPE memory_kind AS ENUM ('fact', 'experience', 'opinion', 'observation');

-- Add structured columns to memories table
ALTER TABLE memories
    ADD COLUMN kind memory_kind NOT NULL DEFAULT 'fact',
    ADD COLUMN confidence FLOAT,
    ADD COLUMN occurred_at TIMESTAMPTZ,
    ADD COLUMN occurred_end TIMESTAMPTZ,
    ADD COLUMN entities TEXT[];

-- Indexes for new columns
CREATE INDEX idx_memories_kind ON memories(org_id, kind);
CREATE INDEX idx_memories_occurred ON memories(occurred_at) WHERE occurred_at IS NOT NULL;
CREATE INDEX idx_memories_entities ON memories USING GIN(entities) WHERE entities IS NOT NULL;

-- Memory graph: edges between related memories
CREATE TABLE memory_edges (
    from_id  UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    to_id    UUID NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    kind     TEXT NOT NULL,   -- 'entity', 'temporal', 'semantic', 'causal'
    weight   FLOAT NOT NULL DEFAULT 1.0,
    PRIMARY KEY (from_id, to_id, kind)
);

CREATE INDEX idx_memory_edges_from ON memory_edges(from_id);
CREATE INDEX idx_memory_edges_to ON memory_edges(to_id);
