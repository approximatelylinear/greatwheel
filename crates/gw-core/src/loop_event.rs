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
    /// LLM response with content and optional model tag.
    Response {
        content: String,
        model: Option<String>,
    },
    /// LLM requests input from the user.
    InputRequest(String),
    /// A host call has been resolved.
    HostCallCompleted {
        function: String,
        result: serde_json::Value,
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
