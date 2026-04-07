-- Feed subscriptions for gw-kb.
--
-- Each row is a registered RSS/Atom feed. `gw-kb feed sync` fetches the
-- feed, parses entries, and calls the ingest pipeline for any entry whose
-- link isn't already in `kb_sources`. Sources ingested via a feed get
-- their `feed_id` stamped so you can trace a source back to its origin.

CREATE TABLE kb_feeds (
    feed_id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name               TEXT NOT NULL,
    slug               TEXT NOT NULL UNIQUE,
    url                TEXT NOT NULL UNIQUE,
    feed_format        TEXT,                -- 'rss' | 'atom' | NULL (auto)
    last_synced_at     TIMESTAMPTZ,
    last_entry_seen_at TIMESTAMPTZ,
    created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
    metadata           JSONB NOT NULL DEFAULT '{}'::jsonb
);

ALTER TABLE kb_sources
    ADD COLUMN feed_id UUID REFERENCES kb_feeds(feed_id) ON DELETE SET NULL;

CREATE INDEX idx_kb_sources_feed ON kb_sources(feed_id);
