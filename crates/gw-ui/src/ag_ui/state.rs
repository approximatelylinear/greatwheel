//! Canonical state shape and JSON-Patch translation for the AG-UI
//! vanilla wire protocol.
//!
//! json-render (and any standard AG-UI client) expects `STATE_DELTA`
//! events to carry RFC 6902 JSON-Patch arrays against a shared state
//! object. This module builds that object from a surface snapshot and
//! translates our internal `UiNotification`s into the corresponding
//! patch ops. See `docs/design-json-render-migration.md` §3.
//!
//! Dual-emission phase: the adapter still emits the legacy
//! domain-shaped `UI_PATCH` event alongside the new `STATE_DELTA` so
//! the current frontend keeps working. Phase 3 removes UI_PATCH.

use std::collections::HashMap;

use serde_json::{json, Value};

use crate::surface::{UiNotification, UiSurfaceSnapshot, UiSurfaceStore};

/// Serialize a surface snapshot into the canonical state shape
/// described in the migration design doc §3. Emitted as the body of
/// `STATE_SNAPSHOT` on SSE subscribe; all subsequent `STATE_DELTA`
/// patches are JSON-Pointer writes against this shape.
///
/// `focused_scope` is the adapter's per-session focus map (populated
/// from button-click `data.scope` / `data.section` in
/// `post_widget_event`). Passed in rather than stored on `UiSurface`
/// because focus is a wire-layer concern; the core surface store
/// doesn't track it.
pub fn canonical_state(snap: &UiSurfaceSnapshot, focused_scope: &HashMap<String, Value>) -> Value {
    let mut widgets = serde_json::Map::new();
    let mut pinned = serde_json::Map::new();
    for w in &snap.widgets {
        widgets.insert(
            w.id.0.to_string(),
            serde_json::to_value(w).unwrap_or(Value::Null),
        );
    }
    if let Some(id) = snap.surface.canvas_slot {
        pinned.insert(id.0.to_string(), Value::Bool(true));
    }
    if let Some(id) = snap.surface.canvas_aux_slot {
        pinned.insert(id.0.to_string(), Value::Bool(true));
    }
    let order: Vec<Value> = snap
        .surface
        .widget_order
        .iter()
        .map(|id| Value::String(id.0.to_string()))
        .collect();
    let focus_map: serde_json::Map<String, Value> = focused_scope
        .iter()
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();

    json!({
        "widgets": Value::Object(widgets),
        "widgetOrder": Value::Array(order),
        "canvasSlot": snap.surface.canvas_slot.map(|id| id.0.to_string()),
        "canvasAuxSlot": snap.surface.canvas_aux_slot.map(|id| id.0.to_string()),
        "pinnedIds": Value::Object(pinned),
        "pressed": {},
        "focusedScope": Value::Object(focus_map),
    })
}

/// Translate one `UiNotification` into the set of JSON-Patch ops a
/// client would apply against the canonical state. Looks up widgets
/// via the store when a notification only carries an id.
///
/// Returns `None` when the notification references a widget the store
/// has already forgotten (shouldn't happen in practice; guards against
/// out-of-order shutdowns).
pub async fn notification_to_patches(
    store: &UiSurfaceStore,
    notif: &UiNotification,
) -> Option<Vec<Value>> {
    match notif {
        UiNotification::Emitted(w) => Some(vec![
            json!({"op": "add", "path": format!("/widgets/{}", w.id.0), "value": w}),
            json!({"op": "add", "path": "/widgetOrder/-", "value": w.id.0.to_string()}),
        ]),
        UiNotification::Superseded { old, new } => Some(vec![
            json!({"op": "replace", "path": format!("/widgets/{}/state", old.0), "value": "Superseded"}),
            json!({"op": "add", "path": format!("/widgets/{}", new.id.0), "value": new}),
            json!({"op": "add", "path": "/widgetOrder/-", "value": new.id.0.to_string()}),
        ]),
        UiNotification::Resolved { id, data } => Some(vec![
            json!({"op": "replace", "path": format!("/widgets/{}/state", id.0), "value": "Resolved"}),
            json!({"op": "replace", "path": format!("/widgets/{}/resolution", id.0), "value": data}),
        ]),
        UiNotification::Expired { id } => Some(vec![
            json!({"op": "replace", "path": format!("/widgets/{}/state", id.0), "value": "Expired"}),
        ]),
        UiNotification::Pinned { id } => Some(vec![
            json!({"op": "replace", "path": "/canvasSlot", "value": id.0.to_string()}),
            json!({"op": "add", "path": format!("/pinnedIds/{}", id.0), "value": true}),
        ]),
        UiNotification::AuxPinned { id } => Some(vec![
            json!({"op": "replace", "path": "/canvasAuxSlot", "value": id.0.to_string()}),
            json!({"op": "add", "path": format!("/pinnedIds/{}", id.0), "value": true}),
        ]),
        UiNotification::ButtonHighlighted {
            widget_id,
            button_id,
        } => {
            // Touch the store just to validate the widget still exists
            // before emitting the patch — consistent with the legacy
            // adapter path.
            let _ = store.get_widget(*widget_id).await?;
            // Nested shape `{widget_id: {button_id: true}}` so json-render
            // `{$state: "/pressed/<widget>/<button>"}` bindings resolve
            // to a plain boolean. Replacing the inner map (rather than
            // adding a key) auto-clears any previously-pressed button
            // for the same widget.
            Some(vec![json!({
                "op": "replace",
                "path": format!("/pressed/{}", widget_id.0),
                "value": { button_id: true },
            })])
        }
    }
}

/// Return the session id a notification belongs to, looking up via
/// the store when the notification doesn't carry the widget record
/// directly. Mirrors the lookup in `notification_to_ag_ui`.
pub async fn notification_session(
    store: &UiSurfaceStore,
    notif: &UiNotification,
) -> Option<gw_core::SessionId> {
    match notif {
        UiNotification::Emitted(w) => Some(w.session_id),
        UiNotification::Superseded { new, .. } => Some(new.session_id),
        UiNotification::Resolved { id, .. }
        | UiNotification::Expired { id }
        | UiNotification::Pinned { id }
        | UiNotification::AuxPinned { id } => store.get_widget(*id).await.map(|w| w.session_id),
        UiNotification::ButtonHighlighted { widget_id, .. } => {
            store.get_widget(*widget_id).await.map(|w| w.session_id)
        }
    }
}

/// Return the surface id a notification belongs to. Used to label the
/// outgoing `STATE_DELTA` event.
pub async fn notification_surface(
    store: &UiSurfaceStore,
    notif: &UiNotification,
) -> Option<String> {
    match notif {
        UiNotification::Emitted(w) => Some(w.surface_id.0.to_string()),
        UiNotification::Superseded { new, .. } => Some(new.surface_id.0.to_string()),
        UiNotification::Resolved { id, .. }
        | UiNotification::Expired { id }
        | UiNotification::Pinned { id }
        | UiNotification::AuxPinned { id } => store
            .get_widget(*id)
            .await
            .map(|w| w.surface_id.0.to_string()),
        UiNotification::ButtonHighlighted { widget_id, .. } => store
            .get_widget(*widget_id)
            .await
            .map(|w| w.surface_id.0.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use gw_core::{
        SessionId, UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState,
    };
    use uuid::Uuid;

    use crate::surface::UiSurface;

    fn sample_widget(session_id: SessionId, surface_id: UiSurfaceId) -> Widget {
        Widget {
            id: WidgetId::new(),
            surface_id,
            session_id,
            origin_entry: None,
            kind: WidgetKind::A2ui,
            state: WidgetState::Active,
            payload: WidgetPayload::Inline(json!({"type": "Button"})),
            supersedes: None,
            created_at: Utc::now(),
            resolved_at: None,
            resolution: None,
            multi_use: false,
            follow_up: false,
            scope: None,
        }
    }

    #[test]
    fn canonical_state_basic_shape() {
        let sid = SessionId(Uuid::new_v4());
        let surf = UiSurfaceId::new();
        let w = sample_widget(sid, surf);
        let snap = UiSurfaceSnapshot {
            surface: UiSurface {
                id: surf,
                session_id: sid,
                widget_order: vec![w.id],
                canvas_slot: Some(w.id),
                canvas_aux_slot: None,
            },
            widgets: vec![w.clone()],
        };
        let state = canonical_state(&snap, &HashMap::new());
        assert_eq!(state["widgets"][w.id.0.to_string()]["id"], json!(w.id.0));
        assert_eq!(state["widgetOrder"][0], json!(w.id.0.to_string()));
        assert_eq!(state["canvasSlot"], json!(w.id.0.to_string()));
        assert_eq!(state["canvasAuxSlot"], Value::Null);
        assert_eq!(state["pinnedIds"][w.id.0.to_string()], json!(true));
        assert!(state["pressed"].is_object());
        assert!(state["focusedScope"].is_object());
    }

    #[test]
    fn canonical_state_includes_focused_scope() {
        let sid = SessionId(Uuid::new_v4());
        let surf = UiSurfaceId::new();
        let snap = UiSurfaceSnapshot {
            surface: UiSurface {
                id: surf,
                session_id: sid,
                widget_order: vec![],
                canvas_slot: None,
                canvas_aux_slot: None,
            },
            widgets: vec![],
        };
        let mut focus = HashMap::new();
        focus.insert("section".into(), json!(4));
        let state = canonical_state(&snap, &focus);
        assert_eq!(state["focusedScope"]["section"], json!(4));
    }

    #[tokio::test]
    async fn emitted_produces_expected_patch_ops() {
        let store = UiSurfaceStore::new();
        let sid = SessionId(Uuid::new_v4());
        let surf = UiSurfaceId::new();
        let w = sample_widget(sid, surf);
        let id = w.id;

        let patches = notification_to_patches(&store, &UiNotification::Emitted(w.clone()))
            .await
            .unwrap();
        assert_eq!(patches.len(), 2);
        assert_eq!(patches[0]["op"], "add");
        assert_eq!(patches[0]["path"], format!("/widgets/{}", id.0));
        assert_eq!(patches[1]["op"], "add");
        assert_eq!(patches[1]["path"], "/widgetOrder/-");
    }

    #[tokio::test]
    async fn highlighted_produces_pressed_write() {
        let store = UiSurfaceStore::new();
        let sid = SessionId(Uuid::new_v4());
        let surf = UiSurfaceId::new();
        let w = sample_widget(sid, surf);
        let id = w.id;
        store.emit(w).await.unwrap();

        let patches = notification_to_patches(
            &store,
            &UiNotification::ButtonHighlighted {
                widget_id: id,
                button_id: "ch-5".into(),
            },
        )
        .await
        .unwrap();
        assert_eq!(patches.len(), 1);
        assert_eq!(patches[0]["op"], "replace");
        assert_eq!(patches[0]["path"], format!("/pressed/{}", id.0));
        assert_eq!(patches[0]["value"], json!({"ch-5": true}));
    }
}
