use chrono::{DateTime, Utc};
use gw_core::{AgentId, OrgId, SessionId};
use serde::{Deserialize, Serialize};

/// A recorded trace span for OTel GenAI instrumentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub org_id: OrgId,
    pub agent_id: Option<AgentId>,
    pub session_id: Option<SessionId>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub duration_ms: i64,
    pub status: String,
    pub attributes: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

/// Records OTel GenAI traces to Postgres (and optionally OTLP).
pub struct TraceRecorder {
    // TODO: sqlx pool, OTLP exporter
}

impl TraceRecorder {
    pub fn new() -> Self {
        Self {}
    }
}
