//! Mapping between gw-core `LoopEvent` and AG-UI outbound events.
//!
//! Widget-state updates go through `state::notification_to_patches`
//! and arrive on the wire as `STATE_DELTA` (RFC 6902 JSON-Patch). This
//! module handles the non-widget events: assistant text, run
//! lifecycle (started / finished / error), input requests, and the
//! greatwheel-specific `DEBUG_CODE_EXEC` extension.

use gw_core::LoopEvent;

use super::events::AgUiEvent;

/// Project a `LoopEvent` onto an AG-UI outbound event. Returns `None` for
/// events that do not belong on the wire (inbound-only, internal
/// transitions, or widget lifecycle events that are handled via the
/// `UiSurfaceStore` notification path).
pub fn loop_event_to_ag_ui(event: &LoopEvent) -> Option<AgUiEvent> {
    match event {
        LoopEvent::TextMessageStart { message_id } => Some(AgUiEvent::TextMessageStart {
            message_id: message_id.clone(),
        }),
        LoopEvent::TextMessageDelta {
            message_id, delta, ..
        } => Some(AgUiEvent::TextMessageContent {
            message_id: message_id.clone(),
            delta: delta.clone(),
        }),
        LoopEvent::TextMessageEnd { message_id } => Some(AgUiEvent::TextMessageEnd {
            message_id: message_id.clone(),
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
        LoopEvent::HostCallStarted {
            tool_call_id,
            function,
        } => Some(AgUiEvent::ToolCallStart {
            tool_call_id: tool_call_id.clone(),
            tool_name: function.clone(),
        }),
        LoopEvent::HostCallArgs { tool_call_id, args } => Some(AgUiEvent::ToolCallArgs {
            tool_call_id: tool_call_id.clone(),
            delta: args.clone(),
        }),
        LoopEvent::HostCallCompleted {
            tool_call_id,
            result,
            error,
            ..
        } => Some(AgUiEvent::ToolCallEnd {
            tool_call_id: tool_call_id.clone(),
            result: if error.is_some() {
                None
            } else {
                Some(result.clone())
            },
            error: error.clone(),
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
        // Spine events also land here for now — Issue #3 will route
        // them through STATE_DELTA against the spine widget surface.
        LoopEvent::UserMessage(_)
        | LoopEvent::FollowUp(_)
        | LoopEvent::SwitchBranch(_)
        | LoopEvent::Compact
        | LoopEvent::SessionEnd
        | LoopEvent::WidgetInteraction(_)
        | LoopEvent::SpineEntryExtracted { .. }
        | LoopEvent::SpineSegmentsUpdated { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    #[test]
    fn text_message_trilogy_maps_to_ag_ui() {
        let id = "msg-1".to_string();

        let start = loop_event_to_ag_ui(&LoopEvent::TextMessageStart {
            message_id: id.clone(),
        })
        .unwrap();
        assert!(matches!(start, AgUiEvent::TextMessageStart { .. }));

        let delta = loop_event_to_ag_ui(&LoopEvent::TextMessageDelta {
            message_id: id.clone(),
            delta: "hello".into(),
            model: None,
        })
        .unwrap();
        match delta {
            AgUiEvent::TextMessageContent {
                message_id,
                delta: d,
            } => {
                assert_eq!(message_id, id);
                assert_eq!(d, "hello");
            }
            other => panic!("expected TextMessageContent, got {other:?}"),
        }

        let end = loop_event_to_ag_ui(&LoopEvent::TextMessageEnd {
            message_id: id.clone(),
        })
        .unwrap();
        assert!(matches!(end, AgUiEvent::TextMessageEnd { .. }));
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
    fn host_call_events_map_to_tool_call_trilogy() {
        let id = "tc-1".to_string();

        let start = loop_event_to_ag_ui(&LoopEvent::HostCallStarted {
            tool_call_id: id.clone(),
            function: "emit_widget".into(),
        })
        .unwrap();
        match start {
            AgUiEvent::ToolCallStart {
                tool_call_id,
                tool_name,
            } => {
                assert_eq!(tool_call_id, id);
                assert_eq!(tool_name, "emit_widget");
            }
            other => panic!("expected ToolCallStart, got {other:?}"),
        }

        let args = loop_event_to_ag_ui(&LoopEvent::HostCallArgs {
            tool_call_id: id.clone(),
            args: serde_json::json!({"args": [], "kwargs": {"kind": "a2ui"}}),
        })
        .unwrap();
        assert!(matches!(args, AgUiEvent::ToolCallArgs { .. }));

        let end_ok = loop_event_to_ag_ui(&LoopEvent::HostCallCompleted {
            tool_call_id: id.clone(),
            function: "emit_widget".into(),
            result: serde_json::json!({"widget_id": "abc"}),
            error: None,
        })
        .unwrap();
        match end_ok {
            AgUiEvent::ToolCallEnd { result, error, .. } => {
                assert!(result.is_some());
                assert!(error.is_none());
            }
            other => panic!("expected ToolCallEnd, got {other:?}"),
        }

        let end_err = loop_event_to_ag_ui(&LoopEvent::HostCallCompleted {
            tool_call_id: id,
            function: "emit_widget".into(),
            result: serde_json::Value::Null,
            error: Some("boom".into()),
        })
        .unwrap();
        match end_err {
            AgUiEvent::ToolCallEnd { result, error, .. } => {
                assert!(result.is_none());
                assert_eq!(error.as_deref(), Some("boom"));
            }
            other => panic!("expected ToolCallEnd with error, got {other:?}"),
        }
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
