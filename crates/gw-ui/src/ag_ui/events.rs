//! AG-UI event envelope types.
//!
//! Outbound events (`AgUiEvent`) are serialised to SSE `data:` frames.
//! Inbound events currently arrive as typed JSON on dedicated endpoints
//! — we carry the inbound shape per endpoint rather than as a single
//! tagged enum, because axum's routing already disambiguates.

use serde::{Deserialize, Serialize};

/// Outbound AG-UI event. Variant names follow AG-UI's SCREAMING_SNAKE_CASE
/// convention; the minimal set below covers step 3 (chat + widgets).
/// Extra variants (`THINKING_*`, `TOOL_CALL_*`, etc.) will be added in
/// later steps as we need them.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "SCREAMING_SNAKE_CASE")]
pub enum AgUiEvent {
    /// Start of an assistant-authored text message. Subsequent
    /// `TextMessageContent` / `TextMessageEnd` events with the same
    /// `message_id` belong to this message.
    TextMessageStart { message_id: String },
    /// Delta of assistant-authored text. Today emitted once per
    /// message with the full content; forward-compatible with a
    /// future stream of many deltas between START/END.
    TextMessageContent { message_id: String, delta: String },
    /// End of an assistant-authored text message.
    TextMessageEnd { message_id: String },
    /// A new run has started (maps from `LoopEvent::TurnStarted`).
    RunStarted {
        #[serde(skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
    },
    /// The current run has finished (maps from `LoopEvent::TurnComplete`).
    RunFinished {
        #[serde(skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
    },
    /// The current run failed (maps from `LoopEvent::TurnError`). Carries
    /// a human-readable message so clients can surface the failure.
    RunError {
        message: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        run_id: Option<String>,
    },
    /// The agent is asking the user for input.
    InputRequest { prompt: String },
    /// Full canonical state snapshot. Emitted once per session on SSE
    /// subscribe so clients can initialise their state store without
    /// a separate fetch. Body shape: `state.widgets`,
    /// `state.widgetOrder`, `state.canvasSlot`, `state.pressed`,
    /// `state.focusedScope`.
    StateSnapshot {
        surface_id: String,
        state: serde_json::Value,
    },
    /// Vanilla AG-UI JSON-Patch delta against the canonical state
    /// (RFC 6902). The only widget-state update channel — widget
    /// creation arrives as `{op: "add", path: "/widgets/<id>", value: Widget}`
    /// plus `{op: "add", path: "/widgetOrder/-", value: "<id>"}`.
    StateDelta {
        surface_id: String,
        patches: Vec<serde_json::Value>,
    },
    /// A host function is about to run. `tool_call_id` correlates
    /// this event with the matching `_ARGS` / `_END` events.
    ToolCallStart {
        tool_call_id: String,
        tool_name: String,
    },
    /// Arguments for a dispatched tool call. Currently emitted once
    /// per call with the full args payload; reserved for streaming
    /// later. `delta` is the AG-UI-canonical field name for the args
    /// chunk.
    ToolCallArgs {
        tool_call_id: String,
        delta: serde_json::Value,
    },
    /// Tool call completed. Carries the result and error (if any) as
    /// an AG-UI extension on the standard TOOL_CALL_END shape —
    /// spec-strict clients can ignore the extra fields and treat this
    /// as a plain END marker.
    ToolCallEnd {
        tool_call_id: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        result: Option<serde_json::Value>,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Diagnostic: a code block the agent just ran, with its stdout and
    /// terminal-ness. Greatwheel-specific (not in AG-UI's standard
    /// vocabulary); named with a `DEBUG_` prefix so a spec-strict
    /// client can drop it safely.
    DebugCodeExec {
        code: String,
        stdout: String,
        is_final: bool,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
}

/// Body of `POST /sessions/:id/messages`.
#[derive(Debug, Clone, Deserialize)]
pub struct PostMessageBody {
    pub content: String,
}
