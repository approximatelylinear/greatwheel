//! Mapping between gw-core `LoopEvent` / `UiNotification` and AG-UI events.
//!
//! Only outbound mapping is needed for step 3. Inbound mapping is handled
//! per-endpoint in `adapter.rs` (each endpoint deserialises its own
//! shape into a `LoopEvent`).

use gw_core::{LoopEvent, SessionId};
use uuid::Uuid;

use crate::surface::{UiNotification, UiSurfaceStore};

use super::events::AgUiEvent;

/// Project a `LoopEvent` onto an AG-UI outbound event. Returns `None` for
/// events that do not belong on the wire (internal transitions, inbound-
/// only variants, etc.).
pub fn loop_event_to_ag_ui(event: &LoopEvent) -> Option<AgUiEvent> {
    match event {
        LoopEvent::Response { content, .. } => Some(AgUiEvent::TextMessageContent {
            message_id: Uuid::new_v4().to_string(),
            delta: content.clone(),
        }),
        LoopEvent::TurnComplete => Some(AgUiEvent::RunFinished { run_id: None }),
        LoopEvent::InputRequest(prompt) => Some(AgUiEvent::InputRequest {
            prompt: prompt.clone(),
        }),
        LoopEvent::WidgetEmitted(w) => Some(AgUiEvent::UiEvent {
            surface_id: w.surface_id.0.to_string(),
            widget: serde_json::to_value(w).ok()?,
        }),
        LoopEvent::WidgetSuperseded { old, new } => Some(AgUiEvent::StateDelta {
            surface_id: new.surface_id.0.to_string(),
            patch: serde_json::json!({
                "kind": "supersede",
                "old": old.0,
                "new": new,
            }),
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

/// Project a store `UiNotification` onto `(session_id, AgUiEvent)`. Needs
/// the store for `Resolved` / `Expired` notifications because those only
/// carry a `WidgetId` — we look up the widget to recover `session_id` and
/// `surface_id`.
pub async fn notification_to_ag_ui(
    store: &UiSurfaceStore,
    notif: UiNotification,
) -> Option<(SessionId, AgUiEvent)> {
    match notif {
        UiNotification::Emitted(w) => {
            let session_id = w.session_id;
            let surface_id = w.surface_id.0.to_string();
            let widget = serde_json::to_value(&w).ok()?;
            Some((session_id, AgUiEvent::UiEvent { surface_id, widget }))
        }
        UiNotification::Superseded { old, new } => {
            let session_id = new.session_id;
            let surface_id = new.surface_id.0.to_string();
            Some((
                session_id,
                AgUiEvent::StateDelta {
                    surface_id,
                    patch: serde_json::json!({
                        "kind": "supersede",
                        "old": old.0,
                        "new": new,
                    }),
                },
            ))
        }
        UiNotification::Resolved { id, data } => {
            let w = store.get_widget(id).await?;
            Some((
                w.session_id,
                AgUiEvent::StateDelta {
                    surface_id: w.surface_id.0.to_string(),
                    patch: serde_json::json!({
                        "kind": "resolve",
                        "widget_id": id.0,
                        "data": data,
                    }),
                },
            ))
        }
        UiNotification::Expired { id } => {
            let w = store.get_widget(id).await?;
            Some((
                w.session_id,
                AgUiEvent::StateDelta {
                    surface_id: w.surface_id.0.to_string(),
                    patch: serde_json::json!({
                        "kind": "expire",
                        "widget_id": id.0,
                    }),
                },
            ))
        }
        UiNotification::Pinned { id } => {
            let w = store.get_widget(id).await?;
            Some((
                w.session_id,
                AgUiEvent::StateDelta {
                    surface_id: w.surface_id.0.to_string(),
                    patch: serde_json::json!({
                        "kind": "pin",
                        "widget_id": id.0,
                    }),
                },
            ))
        }
        UiNotification::ButtonHighlighted {
            widget_id,
            button_id,
        } => {
            let w = store.get_widget(widget_id).await?;
            Some((
                w.session_id,
                AgUiEvent::StateDelta {
                    surface_id: w.surface_id.0.to_string(),
                    patch: serde_json::json!({
                        "kind": "highlight",
                        "widget_id": widget_id.0,
                        "button_id": button_id,
                    }),
                },
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use gw_core::{
        SessionId, UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState,
    };

    fn sample_widget() -> Widget {
        Widget {
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
        }
    }

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
    fn turn_complete_maps_to_run_finished() {
        let ev = LoopEvent::TurnComplete;
        assert!(matches!(
            loop_event_to_ag_ui(&ev),
            Some(AgUiEvent::RunFinished { .. })
        ));
    }

    #[test]
    fn widget_emitted_maps_to_ui_event() {
        let w = sample_widget();
        let expected_surface = w.surface_id.0.to_string();
        let ev = LoopEvent::WidgetEmitted(w);
        let ag = loop_event_to_ag_ui(&ev).unwrap();
        match ag {
            AgUiEvent::UiEvent { surface_id, .. } => assert_eq!(surface_id, expected_surface),
            other => panic!("expected UiEvent, got {:?}", other),
        }
    }

    #[test]
    fn widget_superseded_maps_to_state_delta() {
        let old_id = WidgetId::new();
        let new = sample_widget();
        let ev = LoopEvent::WidgetSuperseded {
            old: old_id,
            new: new.clone(),
        };
        let ag = loop_event_to_ag_ui(&ev).unwrap();
        match ag {
            AgUiEvent::StateDelta { patch, .. } => {
                assert_eq!(patch["kind"], "supersede");
                assert_eq!(patch["old"], serde_json::json!(old_id.0));
            }
            other => panic!("expected StateDelta, got {:?}", other),
        }
    }

    #[test]
    fn user_message_has_no_outbound_projection() {
        let ev = LoopEvent::UserMessage("hi".into());
        assert!(loop_event_to_ag_ui(&ev).is_none());
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

    #[tokio::test]
    async fn resolved_notification_looks_up_session() {
        let store = UiSurfaceStore::new();
        let w = sample_widget();
        let id = w.id;
        let session_id = w.session_id;
        store.emit(w).await.unwrap();
        store
            .resolve(id, serde_json::json!({"choice": "ok"}))
            .await
            .unwrap();

        let notif = UiNotification::Resolved {
            id,
            data: serde_json::json!({"choice": "ok"}),
        };
        let (got_sid, ev) = notification_to_ag_ui(&store, notif).await.unwrap();
        assert_eq!(got_sid, session_id);
        match ev {
            AgUiEvent::StateDelta { patch, .. } => {
                assert_eq!(patch["kind"], "resolve");
                assert_eq!(patch["widget_id"], serde_json::json!(id.0));
            }
            other => panic!("expected StateDelta, got {:?}", other),
        }
    }
}
