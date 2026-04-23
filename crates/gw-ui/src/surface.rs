//! Session-scoped UI surface store.
//!
//! One `UiSurface` per session in v1. Widgets are owned here, *not* in the
//! session tree — the tree records what was said, this records what was
//! shown and its current state. See `docs/design-gw-ui.md` §6.

use std::collections::HashMap;

use chrono::Utc;
use gw_core::{SessionId, UiSurfaceId, Widget, WidgetId, WidgetState};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use thiserror::Error;
use tokio::sync::{broadcast, RwLock};

/// The rendering surface attached to a session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiSurface {
    pub id: UiSurfaceId,
    pub session_id: SessionId,
    /// Widgets in insertion order.
    pub widget_order: Vec<WidgetId>,
    /// Widget pinned to the canvas slot, if any.
    pub canvas_slot: Option<WidgetId>,
}

/// A snapshot of a surface and its widgets — returned to the frontend
/// on reconnect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UiSurfaceSnapshot {
    pub surface: UiSurface,
    pub widgets: Vec<Widget>,
}

#[derive(Debug, Error)]
pub enum UiError {
    #[error("widget {0:?} not found")]
    WidgetNotFound(WidgetId),
    #[error("surface for session {0:?} not found")]
    SurfaceNotFound(SessionId),
    #[error("widget {widget_id:?} is in state {current:?}, cannot transition")]
    InvalidTransition {
        widget_id: WidgetId,
        current: WidgetState,
    },
}

/// Broadcast notifications sent whenever the store mutates. Downstream
/// subscribers (primarily the AG-UI channel adapter) translate these
/// into outbound events.
#[derive(Debug, Clone)]
pub enum UiNotification {
    Emitted(Widget),
    Superseded {
        old: WidgetId,
        new: Widget,
    },
    Resolved {
        id: WidgetId,
        data: Value,
    },
    Expired {
        id: WidgetId,
    },
    /// A widget was moved into the surface's `canvas_slot`.
    Pinned {
        id: WidgetId,
    },
    /// The agent declared that a specific button within a widget is
    /// the "currently focused" one. Transient UI hint; not persisted
    /// in the store — just broadcast for the frontend to mirror in
    /// its local pressed-state map.
    ButtonHighlighted {
        widget_id: WidgetId,
        button_id: String,
    },
}

/// In-memory widget store. Concurrent-safe via an interior `RwLock`.
pub struct UiSurfaceStore {
    inner: RwLock<Inner>,
    tx: broadcast::Sender<UiNotification>,
}

#[derive(Default)]
struct Inner {
    surfaces: HashMap<SessionId, UiSurface>,
    widgets: HashMap<WidgetId, Widget>,
}

impl UiSurfaceStore {
    pub fn new() -> Self {
        let (tx, _) = broadcast::channel(256);
        Self {
            inner: RwLock::new(Inner::default()),
            tx,
        }
    }

    /// Subscribe to store notifications. Each subscriber gets its own
    /// receiver; slow subscribers will see `Lagged` errors rather than
    /// block the store.
    pub fn subscribe(&self) -> broadcast::Receiver<UiNotification> {
        self.tx.subscribe()
    }

    /// Insert a new widget, auto-creating the surface for its session
    /// if none exists.
    pub async fn emit(&self, widget: Widget) -> Result<(), UiError> {
        {
            let mut inner = self.inner.write().await;
            let surface = inner
                .surfaces
                .entry(widget.session_id)
                .or_insert_with(|| UiSurface {
                    id: widget.surface_id,
                    session_id: widget.session_id,
                    widget_order: Vec::new(),
                    canvas_slot: None,
                });
            surface.widget_order.push(widget.id);
            inner.widgets.insert(widget.id, widget.clone());
        }
        let _ = self.tx.send(UiNotification::Emitted(widget));
        Ok(())
    }

    /// Replace `old` with `new`. `old` transitions to `Superseded` and
    /// must currently be `Active`. `new` is inserted on the same surface
    /// as an ordinary emit.
    pub async fn supersede(&self, old: WidgetId, new: Widget) -> Result<(), UiError> {
        {
            let mut inner = self.inner.write().await;

            // Transition the old widget in a scoped borrow.
            {
                let old_widget = inner
                    .widgets
                    .get_mut(&old)
                    .ok_or(UiError::WidgetNotFound(old))?;
                if old_widget.state != WidgetState::Active {
                    return Err(UiError::InvalidTransition {
                        widget_id: old,
                        current: old_widget.state,
                    });
                }
                old_widget.state = WidgetState::Superseded;
                old_widget.resolved_at = Some(Utc::now());
            }

            // Insert the new widget, creating the surface if needed.
            {
                let surface = inner
                    .surfaces
                    .entry(new.session_id)
                    .or_insert_with(|| UiSurface {
                        id: new.surface_id,
                        session_id: new.session_id,
                        widget_order: Vec::new(),
                        canvas_slot: None,
                    });
                surface.widget_order.push(new.id);
            }
            inner.widgets.insert(new.id, new.clone());
        }
        let _ = self.tx.send(UiNotification::Superseded { old, new });
        Ok(())
    }

    /// Resolve an active widget with a terminal value (user interaction
    /// outcome or agent-driven close).
    pub async fn resolve(&self, id: WidgetId, data: Value) -> Result<(), UiError> {
        {
            let mut inner = self.inner.write().await;
            let widget = inner
                .widgets
                .get_mut(&id)
                .ok_or(UiError::WidgetNotFound(id))?;
            if widget.state != WidgetState::Active {
                return Err(UiError::InvalidTransition {
                    widget_id: id,
                    current: widget.state,
                });
            }
            widget.state = WidgetState::Resolved;
            widget.resolution = Some(data.clone());
            widget.resolved_at = Some(Utc::now());
        }
        let _ = self.tx.send(UiNotification::Resolved { id, data });
        Ok(())
    }

    /// Mark a widget expired. Idempotent: returns `Ok(false)` if the
    /// widget was already in a terminal state.
    pub async fn expire(&self, id: WidgetId) -> Result<bool, UiError> {
        let transitioned = {
            let mut inner = self.inner.write().await;
            let widget = inner
                .widgets
                .get_mut(&id)
                .ok_or(UiError::WidgetNotFound(id))?;
            if widget.state != WidgetState::Active {
                false
            } else {
                widget.state = WidgetState::Expired;
                widget.resolved_at = Some(Utc::now());
                true
            }
        };
        if transitioned {
            let _ = self.tx.send(UiNotification::Expired { id });
        }
        Ok(transitioned)
    }

    /// Pin a widget into its surface's canvas slot. The widget must
    /// exist; state is not restricted (you can pin a terminal widget
    /// to show its final state on the canvas).
    pub async fn pin_to_canvas(&self, id: WidgetId) -> Result<(), UiError> {
        {
            let mut inner = self.inner.write().await;
            let session_id = inner
                .widgets
                .get(&id)
                .ok_or(UiError::WidgetNotFound(id))?
                .session_id;
            let surface = inner
                .surfaces
                .get_mut(&session_id)
                .ok_or(UiError::SurfaceNotFound(session_id))?;
            surface.canvas_slot = Some(id);
        }
        let _ = self.tx.send(UiNotification::Pinned { id });
        Ok(())
    }

    /// Broadcast an agent-declared highlight for a specific button
    /// inside a widget. Transient: no store mutation, frontend applies
    /// to its local pressed-state map. The widget doesn't have to
    /// exist — we forward the hint regardless.
    pub fn highlight_button(&self, widget_id: WidgetId, button_id: String) {
        let _ = self.tx.send(UiNotification::ButtonHighlighted {
            widget_id,
            button_id,
        });
    }

    /// Snapshot the full surface for a session, widgets in insertion
    /// order. Used by the AG-UI adapter on frontend reconnect.
    pub async fn snapshot(&self, session: SessionId) -> Result<UiSurfaceSnapshot, UiError> {
        let inner = self.inner.read().await;
        let surface = inner
            .surfaces
            .get(&session)
            .cloned()
            .ok_or(UiError::SurfaceNotFound(session))?;
        let widgets = surface
            .widget_order
            .iter()
            .filter_map(|id| inner.widgets.get(id).cloned())
            .collect();
        Ok(UiSurfaceSnapshot { surface, widgets })
    }

    /// Direct widget lookup — mainly for tests and diagnostics.
    pub async fn get_widget(&self, id: WidgetId) -> Option<Widget> {
        self.inner.read().await.widgets.get(&id).cloned()
    }
}

impl Default for UiSurfaceStore {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use gw_core::{WidgetKind, WidgetPayload};
    use uuid::Uuid;

    fn active_widget(session: SessionId, surface: UiSurfaceId) -> Widget {
        Widget {
            id: WidgetId::new(),
            surface_id: surface,
            session_id: session,
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

    #[tokio::test]
    async fn emit_creates_surface_and_inserts_widget() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;

        store.emit(w).await.unwrap();

        let snapshot = store.snapshot(session).await.unwrap();
        assert_eq!(snapshot.widgets.len(), 1);
        assert_eq!(snapshot.widgets[0].id, id);
        assert_eq!(snapshot.surface.widget_order, vec![id]);
    }

    #[tokio::test]
    async fn emit_twice_preserves_order() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w1 = active_widget(session, surface);
        let w2 = active_widget(session, surface);
        let (id1, id2) = (w1.id, w2.id);

        store.emit(w1).await.unwrap();
        store.emit(w2).await.unwrap();

        let snapshot = store.snapshot(session).await.unwrap();
        assert_eq!(snapshot.surface.widget_order, vec![id1, id2]);
    }

    #[tokio::test]
    async fn supersede_marks_old_and_inserts_new() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let old = active_widget(session, surface);
        let old_id = old.id;
        store.emit(old).await.unwrap();

        let new = active_widget(session, surface);
        let new_id = new.id;
        store.supersede(old_id, new).await.unwrap();

        let old_w = store.get_widget(old_id).await.unwrap();
        assert_eq!(old_w.state, WidgetState::Superseded);
        assert!(old_w.resolved_at.is_some());

        let new_w = store.get_widget(new_id).await.unwrap();
        assert_eq!(new_w.state, WidgetState::Active);

        let snapshot = store.snapshot(session).await.unwrap();
        assert_eq!(snapshot.surface.widget_order, vec![old_id, new_id]);
    }

    #[tokio::test]
    async fn supersede_fails_on_non_active() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();
        store
            .resolve(id, serde_json::json!({"choice": "yes"}))
            .await
            .unwrap();

        let new = active_widget(session, surface);
        let err = store.supersede(id, new).await.unwrap_err();
        assert!(matches!(
            err,
            UiError::InvalidTransition {
                current: WidgetState::Resolved,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn resolve_sets_resolution() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();

        let data = serde_json::json!({"choice": "b"});
        store.resolve(id, data.clone()).await.unwrap();

        let got = store.get_widget(id).await.unwrap();
        assert_eq!(got.state, WidgetState::Resolved);
        assert_eq!(got.resolution.as_ref().unwrap(), &data);
        assert!(got.resolved_at.is_some());
    }

    #[tokio::test]
    async fn resolve_fails_on_non_active() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();
        store.expire(id).await.unwrap();

        let err = store.resolve(id, serde_json::json!({})).await.unwrap_err();
        assert!(matches!(
            err,
            UiError::InvalidTransition {
                current: WidgetState::Expired,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn expire_is_idempotent() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();

        assert!(store.expire(id).await.unwrap());
        assert!(!store.expire(id).await.unwrap());

        let got = store.get_widget(id).await.unwrap();
        assert_eq!(got.state, WidgetState::Expired);
    }

    #[tokio::test]
    async fn snapshot_unknown_session_errors() {
        let store = UiSurfaceStore::new();
        let err = store.snapshot(SessionId(Uuid::new_v4())).await.unwrap_err();
        assert!(matches!(err, UiError::SurfaceNotFound(_)));
    }

    #[tokio::test]
    async fn pin_to_canvas_updates_slot_and_broadcasts() {
        let store = UiSurfaceStore::new();
        let mut rx = store.subscribe();

        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();
        // discard the emit notification
        let _ = rx.recv().await.unwrap();

        store.pin_to_canvas(id).await.unwrap();

        let snapshot = store.snapshot(session).await.unwrap();
        assert_eq!(snapshot.surface.canvas_slot, Some(id));

        match rx.recv().await.unwrap() {
            UiNotification::Pinned { id: got } => assert_eq!(got, id),
            other => panic!("expected Pinned, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn pin_to_canvas_unknown_widget_errors() {
        let store = UiSurfaceStore::new();
        let err = store.pin_to_canvas(WidgetId::new()).await.unwrap_err();
        assert!(matches!(err, UiError::WidgetNotFound(_)));
    }

    #[tokio::test]
    async fn emit_broadcasts_notification() {
        let store = UiSurfaceStore::new();
        let mut rx = store.subscribe();

        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();

        let notif = rx.recv().await.unwrap();
        match notif {
            UiNotification::Emitted(w) => assert_eq!(w.id, id),
            other => panic!("expected Emitted, got {:?}", other),
        }
    }
}
