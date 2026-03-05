use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use uuid::Uuid;

/// Newtype wrappers for domain IDs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OrgId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UserId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AgentId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub Uuid);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TaskId(pub Uuid);

/// Opaque session key — the only credential an agent receives.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionKey(pub String);

/// A task envelope representing inbound work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Task {
    pub id: TaskId,
    pub org_id: OrgId,
    pub user_id: UserId,
    pub channel: String,
    pub payload: String,
    pub context: Option<serde_json::Value>,
    pub parent_task: Option<TaskId>,
}

/// Agent definition — what an agent is and how it runs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentDef {
    pub id: AgentId,
    pub org_id: OrgId,
    pub name: String,
    pub description: String,
    pub system_prompt: String,
    pub source: AgentSource,
    pub tools: ToolPermissions,
    pub model: ModelConfig,
    pub limits: ResourceLimits,
    pub current_version: u32,
}

/// Where the agent's Python source comes from.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AgentSource {
    Bare,
    File(String),
    Inline(String),
    Git { repo: String, path: String, rev: String },
}

/// Allowed/denied host function sets for an agent or user.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ToolPermissions {
    pub allowed: HashSet<String>,
    pub denied: HashSet<String>,
}

/// Model configuration for an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConfig {
    pub model: String,
    pub temperature: Option<f32>,
    pub max_tokens: Option<u32>,
}

/// Resource limits for an agent session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceLimits {
    pub max_llm_calls: Option<u32>,
    pub max_execution_seconds: Option<u64>,
}

/// Context passed to every host function call.
#[derive(Debug, Clone)]
pub struct CallContext {
    pub org_id: OrgId,
    pub user_id: UserId,
    pub session_id: SessionId,
    pub agent_id: AgentId,
    pub task_id: TaskId,
    pub session_key: SessionKey,
    pub permissions: ToolPermissions,
}

/// Rate limit configuration (soft + hard token budgets).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub soft_token_limit: Option<u64>,
    pub hard_token_limit: Option<u64>,
}
