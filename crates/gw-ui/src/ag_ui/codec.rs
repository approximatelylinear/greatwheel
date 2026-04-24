//! Mapping between gw-core `LoopEvent` and AG-UI outbound events.
//!
//! Widget-state updates go through `state::notification_to_patches`
//! and arrive on the wire as `STATE_DELTA` (RFC 6902 JSON-Patch). This
//! module handles the non-widget events: assistant text, run
//! lifecycle (started / finished / error), input requests, and the
//! greatwheel-specific `DEBUG_CODE_EXEC` extension.

use gw_core::LoopEvent;
use uuid::Uuid;

use super::events::AgUiEvent;

/// Project a `LoopEvent` onto an AG-UI outbound event. Returns `None` for
/// events that do not belong on the wire (inbound-only, internal
/// transitions, or widget lifecycle events that are handled via the
/// `UiSurfaceStore` notification path).
pub fn loop_event_to_ag_ui(event: &LoopEvent) -> Option<AgUiEvent> {
    match event {
        LoopEvent::Response { content, .. } => Some(AgUiEvent::TextMessageContent {
            message_id: Uuid::new_v4().to_string(),
            delta: content.clone(),
        }),
        LoopEvent::TurnStarted => Some(AgUiEvent::RunStarted { run_id: None }),
        LoopEvent::TurnComplete => Some(AgUiEvent::RunFinished { run_id: None }),
        LoopEvent::TurnError { message } => Some(AgUiEvent::RunError {
            message: message.clone(),
            run_id: None,
        }),
        LoopEvent::InputRequest(prompt) => Some(AgUiEvent::InputRequest {
            prompt: prompt.clone(),
        }),
        LoopEvent::CodeExecuted {
            code,
            stdout,
            is_final,
            error,
        } => Some(AgUiEvent::DebugCodeExec {
            code: code.clone(),
            stdout: stdout.clone(),
            is_final: *is_final,
            error: error.clone(),
        }),
        // Widget lifecycle: projected via the store notification path
        // (`state::notification_to_patches`), not through LoopEvent.
        LoopEvent::WidgetEmitted(_) | LoopEvent::WidgetSuperseded { .. } => None,
        // Not projected outbound: inbound-only variants and internal
        // state transitions that don't need frontend awareness.
        LoopEvent::UserMessage(_)
        | LoopEvent::FollowUp(_)
        | LoopEvent::HostCallCompleted { .. }
        | LoopEvent::SwitchBranch(_)
        | LoopEvent::Compact
        | LoopEvent::SessionEnd
        | LoopEvent::WidgetInteraction(_) => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_maps_to_text_message_content() {
        let ev = LoopEvent::Response {
            content: "hello".into(),
            model: None,
        };
        let ag = loop_event_to_ag_ui(&ev).unwrap();
        match ag {
            AgUiEvent::TextMessageContent { delta, .. } => assert_eq!(delta, "hello"),
            other => panic!("expected TextMessageContent, got {:?}", other),
        }
    }

    #[test]
    fn turn_started_maps_to_run_started() {
        let ev = LoopEvent::TurnStarted;
        assert!(matches!(
            loop_event_to_ag_ui(&ev),
            Some(AgUiEvent::RunStarted { .. })
        ));
    }

    #[test]
    fn turn_complete_maps_to_run_finished() {
        let ev = LoopEvent::TurnComplete;
        assert!(matches!(
            loop_event_to_ag_ui(&ev),
            Some(AgUiEvent::RunFinished { .. })
        ));
    }

    #[test]
    fn turn_error_maps_to_run_error_with_message() {
        let ev = LoopEvent::TurnError {
            message: "boom".into(),
        };
        match loop_event_to_ag_ui(&ev).unwrap() {
            AgUiEvent::RunError { message, .. } => assert_eq!(message, "boom"),
            other => panic!("expected RunError, got {:?}", other),
        }
    }

    #[test]
    fn user_message_has_no_outbound_projection() {
        let ev = LoopEvent::UserMessage("hi".into());
        assert!(loop_event_to_ag_ui(&ev).is_none());
    }

    #[test]
    fn widget_events_go_via_store_not_loop_event() {
        use chrono::Utc;
        use gw_core::{
            SessionId, UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState,
        };

        let w = Widget {
            id: WidgetId::new(),
            surface_id: UiSurfaceId::new(),
            session_id: SessionId(Uuid::new_v4()),
            origin_entry: None,
            kind: WidgetKind::A2ui,
            state: WidgetState::Active,
            payload: WidgetPayload::Inline(serde_json::json!({"type": "Button"})),
            supersedes: None,
            created_at: Utc::now(),
            resolved_at: None,
            resolution: None,
            multi_use: false,
            follow_up: false,
            scope: None,
        };
        assert!(loop_event_to_ag_ui(&LoopEvent::WidgetEmitted(w.clone())).is_none());
        assert!(loop_event_to_ag_ui(&LoopEvent::WidgetSuperseded {
            old: WidgetId::new(),
            new: w,
        })
        .is_none());
    }

    #[test]
    fn code_executed_maps_to_debug_code_exec() {
        let ev = LoopEvent::CodeExecuted {
            code: "FINAL(\"hi\")".into(),
            stdout: "".into(),
            is_final: true,
            error: None,
        };
        let ag = loop_event_to_ag_ui(&ev).unwrap();
        match ag {
            AgUiEvent::DebugCodeExec { code, is_final, .. } => {
                assert_eq!(code, "FINAL(\"hi\")");
                assert!(is_final);
            }
            other => panic!("expected DebugCodeExec, got {:?}", other),
        }
    }
}
