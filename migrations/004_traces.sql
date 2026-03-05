CREATE TABLE traces (
    id BIGSERIAL PRIMARY KEY,
    trace_id TEXT NOT NULL,
    span_id TEXT NOT NULL UNIQUE,
    parent_span_id TEXT,
    operation_name TEXT NOT NULL,
    org_id UUID NOT NULL REFERENCES orgs(id),
    agent_id UUID REFERENCES agent_defs(id),
    session_id UUID REFERENCES sessions(id),
    model TEXT,
    provider TEXT,
    input_messages JSONB,
    output_messages JSONB,
    input_tokens INTEGER,
    output_tokens INTEGER,
    duration_ms BIGINT NOT NULL,
    status TEXT NOT NULL,
    attributes JSONB,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_traces_org_agent ON traces(org_id, agent_id, created_at);
CREATE INDEX idx_traces_trace_id ON traces(trace_id);
