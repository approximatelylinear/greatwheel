use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::SessionId;

/// Unique identifier for a session tree entry.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntryId(pub Uuid);

impl EntryId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for EntryId {
    fn default() -> Self {
        Self::new()
    }
}

/// A single entry in the session tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: EntryId,
    pub session_id: SessionId,
    pub parent_id: Option<EntryId>,
    pub entry_type: EntryType,
    pub created_at: DateTime<Utc>,
}

/// The kind of content stored in a session entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryType {
    UserMessage(String),
    AssistantMessage {
        content: String,
        model: Option<String>,
    },
    /// User-visible narration the agent produced this turn — the
    /// resolved `FINAL("...")` string or any other prose channelled
    /// to the chat. Distinct from `AssistantMessage` (which holds the
    /// LLM's raw response, mostly Python source for tool-using
    /// agents) so the spine extractor and any downstream "what the
    /// user actually saw" consumer can read prose without the code
    /// noise. Persisted alongside the raw assistant entry; the
    /// frontend's chat row anchors to this entry's id via
    /// `TEXT_MESSAGE_START.entry_id`.
    AssistantNarration {
        content: String,
    },
    CodeExecution {
        code: String,
        stdout: String,
        result: serde_json::Value,
    },
    HostCall {
        function: String,
        args: serde_json::Value,
        result: serde_json::Value,
    },
    ReplSnapshot(ReplSnapshotData),
    Compaction {
        summary: String,
        first_kept_id: EntryId,
        snapshot: Box<ReplSnapshotData>,
    },
    BranchSummary(String),
    System(String),
}

/// Serialized REPL state for snapshots and compaction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReplSnapshotData {
    pub variables: serde_json::Value,
    pub definitions: Vec<String>,
    pub raw_bytes: Option<Vec<u8>>,
}

/// Tracks the active state of a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: SessionId,
    pub active_leaf: Option<EntryId>,
    pub model: Option<String>,
}
