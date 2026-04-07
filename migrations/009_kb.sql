-- Knowledge base ingestor schema (gw-kb).
-- See docs/design-kb.md §2 for the data model.

-- Sources: ingested documents (URL or file).
CREATE TABLE kb_sources (
    source_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url           TEXT,
    file_path     TEXT,
    title         TEXT NOT NULL,
    author        TEXT,
    published_at  TIMESTAMPTZ,
    ingested_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_hash  BYTEA NOT NULL,
    source_format TEXT NOT NULL,
    extractor     TEXT NOT NULL,
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb
);

CREATE UNIQUE INDEX idx_kb_sources_url
    ON kb_sources (url) WHERE url IS NOT NULL;
CREATE UNIQUE INDEX idx_kb_sources_path
    ON kb_sources (file_path) WHERE file_path IS NOT NULL;
CREATE INDEX idx_kb_sources_hash ON kb_sources (content_hash);

-- Chunks: passages from a source. Atomic retrieval unit.
CREATE TABLE kb_chunks (
    chunk_id     UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id    UUID NOT NULL REFERENCES kb_sources(source_id) ON DELETE CASCADE,
    ordinal      INT NOT NULL,
    content      TEXT NOT NULL,
    char_offset  INT NOT NULL,
    char_length  INT NOT NULL,
    heading_path TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_kb_chunks_source ON kb_chunks (source_id, ordinal);
CREATE INDEX idx_kb_chunks_tsv ON kb_chunks USING GIN(tsv);

-- Domains: coarse groupings of topics. Populated by the organize pipeline.
CREATE TABLE kb_domains (
    domain_id   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label       TEXT NOT NULL,
    slug        TEXT NOT NULL UNIQUE,
    description TEXT,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Topics: clusters of related chunks. The unit of understanding.
CREATE TABLE kb_topics (
    topic_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id   UUID REFERENCES kb_domains(domain_id) ON DELETE SET NULL,
    label       TEXT NOT NULL,
    slug        TEXT NOT NULL UNIQUE,
    summary     TEXT,
    summary_at  TIMESTAMPTZ,
    chunk_count INT NOT NULL DEFAULT 0,
    first_seen  TIMESTAMPTZ NOT NULL DEFAULT now(),
    last_seen   TIMESTAMPTZ NOT NULL DEFAULT now(),
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_kb_topics_domain ON kb_topics (domain_id);

-- Topic membership: chunks belong to topics (soft clustering).
CREATE TABLE kb_topic_chunks (
    topic_id  UUID NOT NULL REFERENCES kb_topics(topic_id) ON DELETE CASCADE,
    chunk_id  UUID NOT NULL REFERENCES kb_chunks(chunk_id) ON DELETE CASCADE,
    relevance REAL NOT NULL DEFAULT 1.0,
    PRIMARY KEY (topic_id, chunk_id)
);

CREATE INDEX idx_kb_topic_chunks_chunk ON kb_topic_chunks (chunk_id);

-- Topic links: typed edges between topics. Used for spreading activation.
CREATE TYPE kb_link_kind AS ENUM ('related', 'builds_on', 'contradicts', 'subtopic_of');

CREATE TABLE kb_topic_links (
    from_topic_id UUID NOT NULL REFERENCES kb_topics(topic_id) ON DELETE CASCADE,
    to_topic_id   UUID NOT NULL REFERENCES kb_topics(topic_id) ON DELETE CASCADE,
    kind          kb_link_kind NOT NULL DEFAULT 'related',
    confidence    REAL NOT NULL DEFAULT 0.5,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (from_topic_id, to_topic_id)
);

CREATE INDEX idx_kb_topic_links_to ON kb_topic_links (to_topic_id);
