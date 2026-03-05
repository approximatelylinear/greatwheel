CREATE TABLE org_secrets (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id UUID NOT NULL REFERENCES orgs(id),
    name TEXT NOT NULL,
    encrypted_value BYTEA NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (org_id, name)
);

CREATE TABLE rate_limit_counters (
    org_id UUID NOT NULL REFERENCES orgs(id),
    user_id UUID REFERENCES users(id),
    period_start TIMESTAMPTZ NOT NULL,
    tokens_used BIGINT NOT NULL DEFAULT 0,
    PRIMARY KEY (org_id, COALESCE(user_id, '00000000-0000-0000-0000-000000000000'), period_start)
);
