//! Generative UX types.
//!
//! Widgets are session-scoped UI elements emitted by agents and rendered by
//! a frontend. The crucial invariant is that widgets are **not** stored
//! inside `SessionEntry`: the session tree records what was *said*, a
//! separate surface store records what was *shown* and its current state.
//! `Widget::origin_entry` is a back-reference for rendering — never a
//! containment relationship.
//!
//! See `docs/design-gw-ui.md` for the full design.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::{session_tree::EntryId, SessionId};

/// Identifier for a UI surface (typically one per session in v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct UiSurfaceId(pub Uuid);

impl UiSurfaceId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for UiSurfaceId {
    fn default() -> Self {
        Self::new()
    }
}

/// Identifier for a widget instance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct WidgetId(pub Uuid);

impl WidgetId {
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }
}

impl Default for WidgetId {
    fn default() -> Self {
        Self::new()
    }
}

/// Which rendering mechanism the frontend should use for a widget.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetKind {
    /// Declarative A2UI component tree, rendered by the A2UI renderer.
    A2ui,
    /// MCP-UI resource, rendered in a sandboxed iframe.
    McpUi,
    /// Host-specific widget; the string is a renderer key the frontend
    /// resolves against its custom catalog.
    Custom(String),
}

/// Lifecycle state of a widget.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WidgetState {
    /// Live, accepting input.
    Active,
    /// User interacted; `Widget::resolution` carries the terminal value.
    Resolved,
    /// Timed out or agent-ended without resolution.
    Expired,
    /// Replaced by a newer widget via `ui.supersede_widget`.
    Superseded,
}

/// Widget content.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum WidgetPayload {
    /// Inline payload — A2UI component tree or inline MCP-UI HTML.
    Inline(serde_json::Value),
    /// External reference — used for MCP-UI resources served elsewhere.
    Reference { uri: String, csp: Option<String> },
}

/// A widget emitted by an agent.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Widget {
    pub id: WidgetId,
    pub surface_id: UiSurfaceId,
    pub session_id: SessionId,
    /// Session entry this widget was emitted alongside. Lets the
    /// frontend place the widget with its originating message during
    /// history replay without embedding it into the entry.
    pub origin_entry: Option<EntryId>,
    pub kind: WidgetKind,
    pub state: WidgetState,
    pub payload: WidgetPayload,
    /// Widget this one supersedes, if any.
    pub supersedes: Option<WidgetId>,
    pub created_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    /// Terminal value when `state == Resolved`.
    pub resolution: Option<serde_json::Value>,
    /// If `true`, the adapter does **not** auto-resolve this widget on
    /// user interaction — clicks become pure events and the widget
    /// stays `Active` until the agent explicitly resolves / expires /
    /// supersedes it. Useful for persistent "tool palette" widgets
    /// (chapter pickers, model selectors) that are meant to be used
    /// many times. Defaults to `false` for backward compatibility with
    /// existing form-style widgets.
    #[serde(default)]
    pub multi_use: bool,
    /// If `true`, the frontend attaches this widget to the nearest
    /// assistant chat message (typically for follow-up question
    /// buttons) rather than rendering it in the general chat scroll
    /// tail. Pure frontend convention; backend treats it as any other
    /// widget. Defaults to `false`.
    #[serde(default)]
    pub follow_up: bool,
}

/// A user interaction with a widget. Produced by the frontend, consumed by
/// the conversation loop as a turn-starting event (see
/// `LoopEvent::WidgetInteraction`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WidgetEvent {
    pub widget_id: WidgetId,
    pub surface_id: UiSurfaceId,
    /// Action name — e.g., "submit", "select", or a widget-specific verb.
    pub action: String,
    pub data: serde_json::Value,
}

impl WidgetEvent {
    /// Project this interaction onto a single-line user-facing text string,
    /// suitable for appending to an LLM context as the user's next turn.
    /// The format is stable enough for an LLM to reliably extract fields
    /// but not versioned as a wire protocol — if you need that, serialise
    /// the struct directly.
    pub fn to_user_message(&self) -> String {
        let data_str =
            serde_json::to_string(&self.data).unwrap_or_else(|_| "<unserialisable>".to_string());
        format!(
            "[widget-event] widget={} action={} data={}",
            self.widget_id.0, self.action, data_str
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn widget_event_formats_as_single_line_user_message() {
        let ev = WidgetEvent {
            widget_id: WidgetId(Uuid::nil()),
            surface_id: UiSurfaceId(Uuid::nil()),
            action: "submit".into(),
            data: serde_json::json!({ "choice": "approve" }),
        };
        let msg = ev.to_user_message();
        assert!(msg.starts_with("[widget-event] widget="));
        assert!(msg.contains("action=submit"));
        assert!(msg.contains(r#"data={"choice":"approve"}"#));
        assert!(!msg.contains('\n'), "must be single-line: {msg}");
    }
}
