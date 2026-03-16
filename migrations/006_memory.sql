CREATE TABLE memories (
    id           UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id       UUID NOT NULL REFERENCES orgs(id),
    user_id      UUID,
    agent_id     UUID,
    session_id   UUID,
    key          TEXT NOT NULL,
    value        JSONB NOT NULL,
    text_content TEXT NOT NULL,
    tsv          TSVECTOR GENERATED ALWAYS AS (to_tsvector('english', text_content)) STORED,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, key)
);

CREATE INDEX idx_memories_tsv ON memories USING GIN(tsv);
CREATE INDEX idx_memories_org ON memories(org_id);
CREATE INDEX idx_memories_org_user ON memories(org_id, user_id);
CREATE INDEX idx_memories_org_agent ON memories(org_id, agent_id);
CREATE INDEX idx_memories_org_session ON memories(org_id, session_id);
