-- Phase 2 refinement: store the topic's label embedding separately from
-- the running label+content mean.
--
--   kb_topics.label_vector — embed(label), set once at creation, never updated.
--                            Used for matching new tag labels to existing topics.
--   kb_topics.vector       — mean of (label + member chunk embeddings),
--                            used for spreading activation, query→topic discovery.

ALTER TABLE kb_topics
    ADD COLUMN label_vector BYTEA;
