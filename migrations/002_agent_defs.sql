CREATE TABLE agent_defs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,
    description TEXT,
    system_prompt TEXT NOT NULL,
    source_type TEXT NOT NULL,
    source_ref TEXT,
    tool_permissions JSONB NOT NULL,
    model_config JSONB NOT NULL,
    resource_limits JSONB NOT NULL,
    current_version INTEGER NOT NULL DEFAULT 1,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name)
);

CREATE TABLE agent_versions (
    id BIGSERIAL PRIMARY KEY,
    agent_id UUID NOT NULL REFERENCES agent_defs(id),
    version INTEGER NOT NULL,
    source_snapshot TEXT NOT NULL,
    trigger TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (agent_id, version)
);
