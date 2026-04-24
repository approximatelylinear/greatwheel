use serde::{Deserialize, Serialize};

use crate::session_tree::EntryId;
use crate::ui::{Widget, WidgetEvent, WidgetId};

/// Events that drive the conversation loop state machine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LoopEvent {
    /// User sends a message.
    UserMessage(String),
    /// LLM produces a follow-up (e.g., after code execution).
    FollowUp(String),
    /// Start of an assistant-authored text message. All subsequent
    /// `TextMessageDelta` and `TextMessageEnd` events carrying the
    /// same `message_id` belong to this message. Maps to AG-UI
    /// `TEXT_MESSAGE_START`.
    TextMessageStart { message_id: String },
    /// A delta (partial text) of an in-flight message. Today the
    /// conversation loop emits one of these per message carrying the
    /// full text (LLM output isn't plumbed as a live token stream yet
    /// because the rLM loop needs the full output to parse for code
    /// blocks before deciding what's user-visible). When we add real
    /// streaming, nothing else on the wire needs to change — clients
    /// simply see many of these between a START/END pair.
    TextMessageDelta {
        message_id: String,
        delta: String,
        model: Option<String>,
    },
    /// End of an assistant-authored text message. Maps to AG-UI
    /// `TEXT_MESSAGE_END`.
    TextMessageEnd { message_id: String },
    /// LLM requests input from the user.
    InputRequest(String),
    /// A plugin-registered host function is about to be dispatched.
    /// Outbound-only — channels map this to AG-UI `TOOL_CALL_START`.
    /// `tool_call_id` correlates start / args / complete across the
    /// three event types. The built-in conversation primitives
    /// (`send_message`, `ask_user`, `compact_session`) do NOT produce
    /// tool-call events — they have their own dedicated projections.
    HostCallStarted {
        tool_call_id: String,
        function: String,
    },
    /// Arguments for a dispatched host function. Emitted once between
    /// `HostCallStarted` and `HostCallCompleted` with the full
    /// positional + keyword args. Maps to AG-UI `TOOL_CALL_ARGS`.
    HostCallArgs {
        tool_call_id: String,
        args: serde_json::Value,
    },
    /// A host call has been resolved. Carries the outcome (error or
    /// value) for the tool-call correlation identified by
    /// `tool_call_id`. Maps to AG-UI `TOOL_CALL_END`.
    HostCallCompleted {
        tool_call_id: String,
        function: String,
        result: serde_json::Value,
        #[serde(skip_serializing_if = "Option::is_none")]
        error: Option<String>,
    },
    /// Switch the active branch to a different entry.
    SwitchBranch(EntryId),
    /// Trigger context compaction.
    Compact,
    /// A new turn is starting (user input has arrived, the agent is
    /// about to run). Outbound-only — channels map this to AG-UI
    /// `RUN_STARTED`. Fired before `handle_turn` so clients can show
    /// an immediate "thinking" indicator without waiting for the
    /// first real event.
    TurnStarted,
    /// The current turn is complete.
    TurnComplete,
    /// The current turn failed. Carries the human-readable error so
    /// channels can surface it to the user. Outbound-only; channels
    /// map this to AG-UI `RUN_ERROR`. The loop still propagates the
    /// underlying `LoopError` upward — this variant is for telemetry,
    /// not control flow.
    TurnError { message: String },
    /// End the session.
    SessionEnd,
    /// Agent emitted a widget to the session's UI surface. Outbound.
    WidgetEmitted(Widget),
    /// Agent replaced an existing widget. The old widget transitions
    /// to `WidgetState::Superseded`. Outbound.
    WidgetSuperseded { old: WidgetId, new: Widget },
    /// User interacted with a widget. Inbound; will be routed into
    /// `handle_turn` the same way `UserMessage` is once step 4 lands.
    WidgetInteraction(WidgetEvent),
    /// Fired by the conversation loop after each code block executes.
    /// Outbound / diagnostic — channels can surface this to dev tools
    /// so prompt authors can see exactly what the agent wrote and what
    /// came out. Not fed back into the loop.
    CodeExecuted {
        code: String,
        stdout: String,
        is_final: bool,
        error: Option<String>,
    },
}
