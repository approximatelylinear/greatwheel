-- Session tree entries for the conversation loop.
CREATE TABLE session_entries (
    id          UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(id),
    parent_id   UUID REFERENCES session_entries(id),
    entry_type  TEXT NOT NULL,
    content     JSONB NOT NULL DEFAULT '{}',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_entries_session_id ON session_entries(session_id);
CREATE INDEX idx_session_entries_parent_id ON session_entries(parent_id);

ALTER TABLE sessions ADD COLUMN active_leaf_id UUID REFERENCES session_entries(id);
