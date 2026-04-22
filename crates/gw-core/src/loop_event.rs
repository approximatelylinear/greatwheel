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
    /// The current turn is complete.
    TurnComplete,
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
}
