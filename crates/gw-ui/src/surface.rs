//! Session-scoped UI surface store.
//!
//! One `UiSurface` per session in v1. Widgets are owned here, *not* in the
//! session tree — the tree records what was said, this records what was
//! shown and its current state. See `docs/design-gw-ui.md` §6.

use std::collections::HashMap;

use chrono::Utc;
use gw_core::{SessionId, UiSurfaceId, Widget, WidgetId, WidgetPayload, WidgetState};
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
    /// Widget pinned to the primary canvas slot (top), if any.
    pub canvas_slot: Option<WidgetId>,
    /// Widget pinned below the primary canvas slot (auxiliary /
    /// contextual), if any. Typical use: primary = navigation picker,
    /// auxiliary = information scoped to the current selection.
    #[serde(default)]
    pub canvas_aux_slot: Option<WidgetId>,
    /// Widget pinned to the dedicated wiki pane — a long-form,
    /// addressable doc view (e.g. a `KbDocWiki` rendering a single KB
    /// source). Lives next to the canvas, not inside it, so chat /
    /// canvas / wiki can coexist.
    #[serde(default)]
    pub wiki_slot: Option<WidgetId>,
    /// Pin history for the canvas slot, in chronological pin order.
    /// Every successful `pin_to_canvas` truncates entries after the
    /// cursor and appends; `nav_slot` moves the cursor without
    /// truncating. See `docs/design-slot-nav.md`.
    #[serde(default)]
    pub canvas_history: Vec<WidgetId>,
    #[serde(default)]
    pub canvas_cursor: Option<usize>,
    #[serde(default)]
    pub canvas_aux_history: Vec<WidgetId>,
    #[serde(default)]
    pub canvas_aux_cursor: Option<usize>,
    #[serde(default)]
    pub wiki_history: Vec<WidgetId>,
    #[serde(default)]
    pub wiki_cursor: Option<usize>,
}

impl UiSurface {
    /// Construct an empty surface with no pinned widgets and no history.
    /// The four UiSurface literal sites that used to spell out the
    /// defaults call this instead so adding nav-history fields doesn't
    /// require touching every callsite.
    pub fn new(id: UiSurfaceId, session_id: SessionId) -> Self {
        Self {
            id,
            session_id,
            widget_order: Vec::new(),
            canvas_slot: None,
            canvas_aux_slot: None,
            wiki_slot: None,
            canvas_history: Vec::new(),
            canvas_cursor: None,
            canvas_aux_history: Vec::new(),
            canvas_aux_cursor: None,
            wiki_history: Vec::new(),
            wiki_cursor: None,
        }
    }
}

/// Identifies one of the three pinned slots on a `UiSurface`. Used by
/// `nav_slot` and by the per-slot `can_back` / `can_forward` derivation
/// in the AG-UI codec.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SlotKind {
    Canvas,
    Aux,
    Wiki,
}

/// Direction for `nav_slot`. `Back` moves the cursor one step earlier
/// in the slot's pin history; `Forward` one step later. Both no-op at
/// the boundary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NavDirection {
    Back,
    Forward,
}

/// Booleans the frontend needs to enable/disable the per-slot
/// back/forward chevrons. Derived from `(history, cursor)` of one
/// slot.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct SlotNavState {
    pub can_back: bool,
    pub can_forward: bool,
}

impl SlotNavState {
    pub fn from_history(history: &[WidgetId], cursor: Option<usize>) -> Self {
        match cursor {
            Some(c) => Self {
                can_back: c > 0,
                can_forward: c + 1 < history.len(),
            },
            None => Self::default(),
        }
    }
}

/// Browser-style "navigate to a new entry" update of a slot's pin
/// history. If the same widget is already at the cursor, no-op
/// (idempotent re-pin). Otherwise truncate everything past the cursor
/// and append, leaving the cursor at the new end.
///
/// Lifted into a helper because all three `pin_to_*` methods need
/// exactly this update; keeps the call sites a single line and the
/// invariants tested in one place.
fn update_pin_history(history: &mut Vec<WidgetId>, cursor: &mut Option<usize>, id: WidgetId) {
    if let Some(c) = *cursor {
        if history.get(c).copied() == Some(id) {
            return;
        }
        history.truncate(c + 1);
    } else {
        history.clear();
    }
    history.push(id);
    *cursor = Some(history.len() - 1);
}

/// Which slot a widget should be re-pinned to when restored from
/// history. Routed by the payload's inline `type` discriminator,
/// matching the mapping the demo agents use at emit time
/// (cf. `crates/gw-ui/examples/literature_assistant.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreSlot {
    Canvas,
    Aux,
    Wiki,
}

/// Pick the slot a widget should be restored into, by payload kind.
/// Wiki-style widgets (`KbDocWiki` / `KbClusterWiki`) go to the wiki
/// drawer; `EntityCloud` to the primary canvas; anything else to the
/// auxiliary slot below the canvas. Unknown / non-inline payloads fall
/// through to Aux — least disruptive default.
pub fn pick_slot_for_widget(widget: &Widget) -> RestoreSlot {
    let WidgetPayload::Inline(inner) = &widget.payload else {
        return RestoreSlot::Aux;
    };
    match inner.get("type").and_then(|v| v.as_str()) {
        Some("KbDocWiki") | Some("KbClusterWiki") => RestoreSlot::Wiki,
        Some("EntityCloud") => RestoreSlot::Canvas,
        _ => RestoreSlot::Aux,
    }
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
    /// A widget was moved into the surface's primary `canvas_slot`.
    Pinned {
        id: WidgetId,
    },
    /// A widget was moved into the surface's `canvas_aux_slot` (the
    /// auxiliary region below the primary canvas pin).
    AuxPinned {
        id: WidgetId,
    },
    /// A widget was moved into the surface's `wiki_slot` (the
    /// dedicated long-form doc pane).
    WikiPinned {
        id: WidgetId,
    },
    /// The surface's `wiki_slot` was cleared (close-wiki). Carries the
    /// session and surface id so adapters can route the resulting
    /// STATE_DELTA without a widget lookup (the slot is empty).
    WikiUnpinned {
        session_id: SessionId,
        surface_id: UiSurfaceId,
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
                .or_insert_with(|| UiSurface::new(widget.surface_id, widget.session_id));
            surface.widget_order.push(widget.id);
            inner.widgets.insert(widget.id, widget.clone());
        }
        let _ = self.tx.send(UiNotification::Emitted(widget));
        Ok(())
    }

    /// Replace `old` with `new`. `old` transitions to `Superseded` and
    /// must currently be `Active`. `new` is inserted on the same surface
    /// as an ordinary emit.
    ///
    /// **Slot follow-through:** if `old` currently occupies a slot
    /// (canvas / aux / wiki), the slot repoints to `new` and the slot's
    /// pin history advances — same effect as calling the matching
    /// `pin_to_*` on `new`. This is what makes "update the chart" work:
    /// the agent supersedes the pinned cloud and the canvas follows the
    /// replacement automatically, instead of stranding the slot on the
    /// now-Superseded old widget (which would render a "Replaced" banner
    /// over stale content). Without it, callers had to remember a
    /// separate `pin_to_*` after every supersede.
    pub async fn supersede(&self, old: WidgetId, new: Widget) -> Result<(), UiError> {
        let new_id = new.id;
        let mut slots_followed: Vec<SlotKind> = Vec::new();
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
                    .or_insert_with(|| UiSurface::new(new.surface_id, new.session_id));
                surface.widget_order.push(new.id);

                // Slot follow-through: any slot pointing at `old` moves
                // to `new`, advancing that slot's pin history.
                if surface.canvas_slot == Some(old) {
                    surface.canvas_slot = Some(new_id);
                    update_pin_history(
                        &mut surface.canvas_history,
                        &mut surface.canvas_cursor,
                        new_id,
                    );
                    slots_followed.push(SlotKind::Canvas);
                }
                if surface.canvas_aux_slot == Some(old) {
                    surface.canvas_aux_slot = Some(new_id);
                    update_pin_history(
                        &mut surface.canvas_aux_history,
                        &mut surface.canvas_aux_cursor,
                        new_id,
                    );
                    slots_followed.push(SlotKind::Aux);
                }
                if surface.wiki_slot == Some(old) {
                    surface.wiki_slot = Some(new_id);
                    update_pin_history(
                        &mut surface.wiki_history,
                        &mut surface.wiki_cursor,
                        new_id,
                    );
                    slots_followed.push(SlotKind::Wiki);
                }
            }
            inner.widgets.insert(new.id, new.clone());
        }
        let _ = self.tx.send(UiNotification::Superseded { old, new });
        // Emit a pin notification per followed slot so the codec mirrors
        // the new slot pointer (+ nav state) to the frontend.
        for slot in slots_followed {
            let notif = match slot {
                SlotKind::Canvas => UiNotification::Pinned { id: new_id },
                SlotKind::Aux => UiNotification::AuxPinned { id: new_id },
                SlotKind::Wiki => UiNotification::WikiPinned { id: new_id },
            };
            let _ = self.tx.send(notif);
        }
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

    /// Pin a widget into its surface's auxiliary (below) canvas slot.
    /// Same shape as `pin_to_canvas`; writes `canvas_aux_slot` and
    /// extends `canvas_aux_history` (browser-style: truncate forward,
    /// then append, unless the same id is already at the cursor).
    pub async fn pin_below_canvas(&self, id: WidgetId) -> Result<(), UiError> {
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
            surface.canvas_aux_slot = Some(id);
            update_pin_history(
                &mut surface.canvas_aux_history,
                &mut surface.canvas_aux_cursor,
                id,
            );
        }
        let _ = self.tx.send(UiNotification::AuxPinned { id });
        Ok(())
    }

    /// Pin a widget into its surface's wiki slot. Same shape as
    /// `pin_to_canvas`; writes `wiki_slot` and extends `wiki_history`.
    /// The wiki pane is the long-form addressable doc view (e.g.
    /// KbDocWiki) and sits alongside the canvas rather than replacing
    /// it.
    pub async fn pin_to_wiki(&self, id: WidgetId) -> Result<(), UiError> {
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
            surface.wiki_slot = Some(id);
            update_pin_history(&mut surface.wiki_history, &mut surface.wiki_cursor, id);
        }
        let _ = self.tx.send(UiNotification::WikiPinned { id });
        Ok(())
    }

    /// Clear a session's `wiki_slot`. Idempotent: returns `Ok(())`
    /// even when the slot was already empty (so the close handler
    /// doesn't have to track state). No-op when the surface itself
    /// doesn't exist.
    pub async fn clear_wiki_slot(&self, session_id: SessionId) -> Result<(), UiError> {
        let cleared_surface = {
            let mut inner = self.inner.write().await;
            match inner.surfaces.get_mut(&session_id) {
                Some(surface) if surface.wiki_slot.is_some() => {
                    surface.wiki_slot = None;
                    Some(surface.id)
                }
                _ => None,
            }
        };
        if let Some(surface_id) = cleared_surface {
            let _ = self.tx.send(UiNotification::WikiUnpinned {
                session_id,
                surface_id,
            });
        }
        Ok(())
    }

    /// Pin a widget into its surface's canvas slot. The widget must
    /// exist; state is not restricted (you can pin a terminal widget
    /// to show its final state on the canvas). Also extends
    /// `canvas_history` for back/forward nav (see
    /// `docs/design-slot-nav.md`).
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
            update_pin_history(&mut surface.canvas_history, &mut surface.canvas_cursor, id);
        }
        let _ = self.tx.send(UiNotification::Pinned { id });
        Ok(())
    }

    /// Step the cursor for one slot's pin history without truncating
    /// the forward stack. The slot pointer follows the cursor — so
    /// after `nav_slot(_, Canvas, Back)` the canvas renders whatever
    /// widget the new cursor position points at. At a history boundary
    /// (back from index 0, forward from the end) this is a silent
    /// no-op so the frontend's disabled-state and a stray click don't
    /// disagree.
    ///
    /// Bypasses `pin_to_*` deliberately — the pin methods truncate
    /// forward, which is exactly what nav must *not* do.
    pub async fn nav_slot(
        &self,
        session_id: SessionId,
        slot: SlotKind,
        direction: NavDirection,
    ) -> Result<(), UiError> {
        let (widget_id, new_cursor) = {
            let inner = self.inner.read().await;
            let surface = inner
                .surfaces
                .get(&session_id)
                .ok_or(UiError::SurfaceNotFound(session_id))?;
            let (history, cursor) = match slot {
                SlotKind::Canvas => (&surface.canvas_history, surface.canvas_cursor),
                SlotKind::Aux => (&surface.canvas_aux_history, surface.canvas_aux_cursor),
                SlotKind::Wiki => (&surface.wiki_history, surface.wiki_cursor),
            };
            let Some(c) = cursor else {
                return Ok(());
            };
            let next = match direction {
                NavDirection::Back => c.checked_sub(1),
                NavDirection::Forward => (c + 1 < history.len()).then_some(c + 1),
            };
            match next {
                Some(n) => (history[n], n),
                None => return Ok(()),
            }
        };
        {
            let mut inner = self.inner.write().await;
            let surface = inner
                .surfaces
                .get_mut(&session_id)
                .ok_or(UiError::SurfaceNotFound(session_id))?;
            match slot {
                SlotKind::Canvas => {
                    surface.canvas_slot = Some(widget_id);
                    surface.canvas_cursor = Some(new_cursor);
                }
                SlotKind::Aux => {
                    surface.canvas_aux_slot = Some(widget_id);
                    surface.canvas_aux_cursor = Some(new_cursor);
                }
                SlotKind::Wiki => {
                    surface.wiki_slot = Some(widget_id);
                    surface.wiki_cursor = Some(new_cursor);
                }
            }
        }
        let notif = match slot {
            SlotKind::Canvas => UiNotification::Pinned { id: widget_id },
            SlotKind::Aux => UiNotification::AuxPinned { id: widget_id },
            SlotKind::Wiki => UiNotification::WikiPinned { id: widget_id },
        };
        let _ = self.tx.send(notif);
        Ok(())
    }

    /// Re-pin a previously emitted widget into its appropriate slot,
    /// routed by payload kind. Pure UI op: the widget's lifecycle
    /// state (Active / Superseded / Resolved) is left unchanged — only
    /// the slot pointer moves. Superseded is the typical input here,
    /// and it staying Superseded keeps supersede chains coherent.
    pub async fn restore_widget(
        &self,
        session_id: SessionId,
        widget_id: WidgetId,
    ) -> Result<(), UiError> {
        let widget = {
            let inner = self.inner.read().await;
            inner
                .widgets
                .get(&widget_id)
                .cloned()
                .ok_or(UiError::WidgetNotFound(widget_id))?
        };
        if widget.session_id != session_id {
            return Err(UiError::WidgetNotFound(widget_id));
        }
        match pick_slot_for_widget(&widget) {
            RestoreSlot::Canvas => self.pin_to_canvas(widget_id).await,
            RestoreSlot::Aux => self.pin_below_canvas(widget_id).await,
            RestoreSlot::Wiki => self.pin_to_wiki(widget_id).await,
        }
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

    /// Return the primary `canvas_slot` widget id for a session, if
    /// any. Used by host functions that want to act on "the pinned
    /// widget" without the agent having to remember its id across
    /// turns (REPL assignments don't persist; `set_variable` does,
    /// but that's only called at agent construction).
    pub async fn primary_pin(&self, session_id: SessionId) -> Option<WidgetId> {
        self.inner
            .read()
            .await
            .surfaces
            .get(&session_id)
            .and_then(|s| s.canvas_slot)
    }

    /// Return the `wiki_slot` widget id for a session, if any.
    pub async fn wiki_pin(&self, session_id: SessionId) -> Option<WidgetId> {
        self.inner
            .read()
            .await
            .surfaces
            .get(&session_id)
            .and_then(|s| s.wiki_slot)
    }

    /// Direct widget lookup — mainly for tests and diagnostics.
    pub async fn get_widget(&self, id: WidgetId) -> Option<Widget> {
        self.inner.read().await.widgets.get(&id).cloned()
    }

    /// Per-slot back/forward enabled-state for the codec to project
    /// onto `/canvasNav` / `/canvasAuxNav` / `/wikiNav`. Returns the
    /// "empty" state (both false) when the session has no surface or
    /// the slot has never been pinned.
    pub async fn slot_nav_state(&self, session_id: SessionId, slot: SlotKind) -> SlotNavState {
        let inner = self.inner.read().await;
        let Some(surface) = inner.surfaces.get(&session_id) else {
            return SlotNavState::default();
        };
        let (history, cursor) = match slot {
            SlotKind::Canvas => (&surface.canvas_history, surface.canvas_cursor),
            SlotKind::Aux => (&surface.canvas_aux_history, surface.canvas_aux_cursor),
            SlotKind::Wiki => (&surface.wiki_history, surface.wiki_cursor),
        };
        SlotNavState::from_history(history, cursor)
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
            scope: None,
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

    fn widget_with_type(session: SessionId, surface: UiSurfaceId, ty: &str) -> Widget {
        Widget {
            payload: WidgetPayload::Inline(serde_json::json!({"type": ty})),
            ..active_widget(session, surface)
        }
    }

    #[tokio::test]
    async fn restore_widget_routes_by_kind() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();

        let cloud = widget_with_type(session, surface, "EntityCloud");
        let doc_wiki = widget_with_type(session, surface, "KbDocWiki");
        let cluster_wiki = widget_with_type(session, surface, "KbClusterWiki");
        let other = widget_with_type(session, surface, "PaperAbstract");
        let (cloud_id, doc_id, cluster_id, other_id) =
            (cloud.id, doc_wiki.id, cluster_wiki.id, other.id);

        store.emit(cloud).await.unwrap();
        store.emit(doc_wiki).await.unwrap();
        store.emit(cluster_wiki).await.unwrap();
        store.emit(other).await.unwrap();

        store.restore_widget(session, cloud_id).await.unwrap();
        store.restore_widget(session, other_id).await.unwrap();
        store.restore_widget(session, doc_id).await.unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_slot, Some(cloud_id));
        assert_eq!(snap.surface.canvas_aux_slot, Some(other_id));
        assert_eq!(snap.surface.wiki_slot, Some(doc_id));

        // KbClusterWiki also routes to wiki, replacing the prior wiki pin.
        store.restore_widget(session, cluster_id).await.unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.wiki_slot, Some(cluster_id));
    }

    #[tokio::test]
    async fn restore_widget_preserves_lifecycle_state() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();

        let old = widget_with_type(session, surface, "EntityCloud");
        let old_id = old.id;
        store.emit(old).await.unwrap();
        let new = widget_with_type(session, surface, "EntityCloud");
        store.supersede(old_id, new).await.unwrap();

        assert_eq!(
            store.get_widget(old_id).await.unwrap().state,
            WidgetState::Superseded
        );
        store.restore_widget(session, old_id).await.unwrap();
        // Slot moves; widget state does not.
        assert_eq!(
            store.get_widget(old_id).await.unwrap().state,
            WidgetState::Superseded
        );
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_slot, Some(old_id));
    }

    #[tokio::test]
    async fn restore_widget_rejects_cross_session() {
        let store = UiSurfaceStore::new();
        let session_a = SessionId(Uuid::new_v4());
        let session_b = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = widget_with_type(session_a, surface, "EntityCloud");
        let id = w.id;
        store.emit(w).await.unwrap();

        let err = store.restore_widget(session_b, id).await.unwrap_err();
        assert!(matches!(err, UiError::WidgetNotFound(_)));
    }

    #[tokio::test]
    async fn supersede_follows_pinned_slot() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let old = active_widget(session, surface);
        let old_id = old.id;
        store.emit(old).await.unwrap();
        store.pin_to_canvas(old_id).await.unwrap();

        let new = active_widget(session, surface);
        let new_id = new.id;
        store.supersede(old_id, new).await.unwrap();

        let snap = store.snapshot(session).await.unwrap();
        // Canvas slot follows the replacement, not stranded on old.
        assert_eq!(snap.surface.canvas_slot, Some(new_id));
        // History advanced; back returns to the superseded widget.
        assert_eq!(snap.surface.canvas_history, vec![old_id, new_id]);
        assert_eq!(snap.surface.canvas_cursor, Some(1));
        // Old widget is Superseded but the slot no longer points at it.
        assert_eq!(
            store.get_widget(old_id).await.unwrap().state,
            WidgetState::Superseded
        );
    }

    #[tokio::test]
    async fn supersede_does_not_touch_unpinned_slots() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let old = active_widget(session, surface);
        let old_id = old.id;
        store.emit(old).await.unwrap();
        // old is NOT pinned anywhere.
        let new = active_widget(session, surface);
        store.supersede(old_id, new).await.unwrap();

        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_slot, None);
        assert_eq!(snap.surface.canvas_aux_slot, None);
        assert_eq!(snap.surface.wiki_slot, None);
        assert!(snap.surface.canvas_history.is_empty());
    }

    #[tokio::test]
    async fn pin_to_canvas_appends_to_history() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w1 = active_widget(session, surface);
        let w2 = active_widget(session, surface);
        let (id1, id2) = (w1.id, w2.id);
        store.emit(w1).await.unwrap();
        store.emit(w2).await.unwrap();

        store.pin_to_canvas(id1).await.unwrap();
        store.pin_to_canvas(id2).await.unwrap();

        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_history, vec![id1, id2]);
        assert_eq!(snap.surface.canvas_cursor, Some(1));
        assert_eq!(snap.surface.canvas_slot, Some(id2));
    }

    #[tokio::test]
    async fn pin_dedupes_on_equal_cursor() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();

        store.pin_to_canvas(id).await.unwrap();
        store.pin_to_canvas(id).await.unwrap();
        store.pin_to_canvas(id).await.unwrap();

        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_history, vec![id]);
        assert_eq!(snap.surface.canvas_cursor, Some(0));
    }

    #[tokio::test]
    async fn nav_back_then_forward_walks_history() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w1 = active_widget(session, surface);
        let w2 = active_widget(session, surface);
        let w3 = active_widget(session, surface);
        let (id1, id2, id3) = (w1.id, w2.id, w3.id);
        store.emit(w1).await.unwrap();
        store.emit(w2).await.unwrap();
        store.emit(w3).await.unwrap();
        store.pin_to_canvas(id1).await.unwrap();
        store.pin_to_canvas(id2).await.unwrap();
        store.pin_to_canvas(id3).await.unwrap();

        store
            .nav_slot(session, SlotKind::Canvas, NavDirection::Back)
            .await
            .unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_slot, Some(id2));
        assert_eq!(snap.surface.canvas_cursor, Some(1));
        assert_eq!(snap.surface.canvas_history, vec![id1, id2, id3]);

        store
            .nav_slot(session, SlotKind::Canvas, NavDirection::Back)
            .await
            .unwrap();
        store
            .nav_slot(session, SlotKind::Canvas, NavDirection::Forward)
            .await
            .unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_slot, Some(id2));
        assert_eq!(snap.surface.canvas_cursor, Some(1));
    }

    #[tokio::test]
    async fn nav_at_boundary_is_noop() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w = active_widget(session, surface);
        let id = w.id;
        store.emit(w).await.unwrap();
        store.pin_to_canvas(id).await.unwrap();

        // back from cursor=0 → no-op
        store
            .nav_slot(session, SlotKind::Canvas, NavDirection::Back)
            .await
            .unwrap();
        // forward from cursor=end → no-op
        store
            .nav_slot(session, SlotKind::Canvas, NavDirection::Forward)
            .await
            .unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_cursor, Some(0));
        assert_eq!(snap.surface.canvas_slot, Some(id));
    }

    #[tokio::test]
    async fn pin_after_back_truncates_forward() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w1 = active_widget(session, surface);
        let w2 = active_widget(session, surface);
        let w3 = active_widget(session, surface);
        let (id1, id2, id3) = (w1.id, w2.id, w3.id);
        store.emit(w1).await.unwrap();
        store.emit(w2).await.unwrap();
        store.emit(w3).await.unwrap();
        store.pin_to_canvas(id1).await.unwrap();
        store.pin_to_canvas(id2).await.unwrap();
        store
            .nav_slot(session, SlotKind::Canvas, NavDirection::Back)
            .await
            .unwrap();
        // cursor=0, history=[id1, id2]. Now pin id3 → truncate forward, append.
        store.pin_to_canvas(id3).await.unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.canvas_history, vec![id1, id3]);
        assert_eq!(snap.surface.canvas_cursor, Some(1));
        assert_eq!(snap.surface.canvas_slot, Some(id3));
    }

    #[tokio::test]
    async fn close_wiki_keeps_history() {
        let store = UiSurfaceStore::new();
        let session = SessionId(Uuid::new_v4());
        let surface = UiSurfaceId::new();
        let w1 = active_widget(session, surface);
        let w2 = active_widget(session, surface);
        let (id1, id2) = (w1.id, w2.id);
        store.emit(w1).await.unwrap();
        store.emit(w2).await.unwrap();
        store.pin_to_wiki(id1).await.unwrap();
        store.pin_to_wiki(id2).await.unwrap();

        store.clear_wiki_slot(session).await.unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.wiki_slot, None);
        assert_eq!(snap.surface.wiki_history, vec![id1, id2]);
        assert_eq!(snap.surface.wiki_cursor, Some(1));

        // back from a closed wiki re-pins the previous widget
        store
            .nav_slot(session, SlotKind::Wiki, NavDirection::Back)
            .await
            .unwrap();
        let snap = store.snapshot(session).await.unwrap();
        assert_eq!(snap.surface.wiki_slot, Some(id1));
        assert_eq!(snap.surface.wiki_cursor, Some(0));
    }

    #[tokio::test]
    async fn nav_unknown_session_errors() {
        let store = UiSurfaceStore::new();
        let err = store
            .nav_slot(
                SessionId(Uuid::new_v4()),
                SlotKind::Canvas,
                NavDirection::Back,
            )
            .await
            .unwrap_err();
        assert!(matches!(err, UiError::SurfaceNotFound(_)));
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
