-- Phase 2 organize layer: chunk tagging + topic vectors.
--
-- New columns:
--   kb_chunks.tagged_at  — set when organize has assigned topics to a chunk
--   kb_chunks.entities   — named entities extracted by the tagger
--   kb_topics.vector     — topic embedding (raw float32 bytes)

ALTER TABLE kb_chunks
    ADD COLUMN tagged_at TIMESTAMPTZ,
    ADD COLUMN entities  TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[];

CREATE INDEX idx_kb_chunks_tagged ON kb_chunks (tagged_at) WHERE tagged_at IS NULL;

ALTER TABLE kb_topics
    ADD COLUMN vector BYTEA;
