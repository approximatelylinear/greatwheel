//! AG-UI adapter: one per gw instance, serves many sessions.
//!
//! Responsibilities:
//!   - Maintain per-session inbound channels (`EventSender` equivalents)
//!     so HTTP POSTs can inject `LoopEvent`s into the conversation loop.
//!   - Maintain per-session outbound SSE broadcasters.
//!   - Subscribe to `UiSurfaceStore` notifications and fan them out to
//!     the right session's SSE broadcaster.
//!   - Expose an axum `Router` mountable by `gw-server`.
//!
//! The `gw_core::ChannelAdapter` trait is not implemented here. Its
//! current shape (`handle_outbound(&self, &LoopEvent)`) is not
//! session-aware, which clashes with AG-UI's one-adapter-many-sessions
//! model. We surface richer methods (`register_session`, `dispatch`,
//! `router`) instead; the trait impl can follow once the trait itself
//! grows a session dimension.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::sse::{Event, KeepAlive, Sse},
    routing::{get, post},
    Json, Router,
};
use futures::Stream;
use gw_core::{
    LoopEvent, SessionId, SpineSegmentSnapshot, UiSurfaceId, Widget, WidgetEvent, WidgetId,
    WidgetKind, WidgetPayload, WidgetState,
};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::surface::{UiSurfaceSnapshot, UiSurfaceStore};

/// Assemble the SemanticSpine widget payload from a session's
/// current (live) segments. Custom A2UI component shape; the
/// frontend's json-render catalog needs a matching `SemanticSpine`
/// entry to render it. See `docs/design-semantic-spine.md` §5.3.
fn build_spine_payload(segments: &[SpineSegmentSnapshot]) -> serde_json::Value {
    let segs: Vec<serde_json::Value> = segments
        .iter()
        .map(|s| {
            serde_json::json!({
                "id": s.segment_id.to_string(),
                "label": s.label,
                "kind": s.kind,
                "entry_first": s.entry_first.0.to_string(),
                "entry_last": s.entry_last.0.to_string(),
                "entity_count": s.entity_ids.len(),
                "entity_ids": s.entity_ids
                    .iter()
                    .map(|id| id.to_string())
                    .collect::<Vec<_>>(),
                "summary": s.summary,
            })
        })
        .collect();
    serde_json::json!({
        "type": "SemanticSpine",
        "segments": segs,
    })
}

use super::codec::loop_event_to_ag_ui;
use super::events::{AgUiEvent, PostMessageBody};
use super::state::{
    canonical_state, notification_session, notification_surface, notification_to_patches,
};

/// Inbound sink type: one per registered session. The frontend's HTTP
/// POSTs are translated to `LoopEvent`s and pushed through this channel
/// to reach the conversation loop.
pub type InboundSender = mpsc::UnboundedSender<LoopEvent>;

/// Shared adapter state — held behind an `Arc` and cloned into each
/// axum request via `State`.
pub struct AgUiState {
    store: Arc<UiSurfaceStore>,
    /// Per-session outbound SSE broadcasters.
    sessions: Mutex<HashMap<SessionId, broadcast::Sender<AgUiEvent>>>,
    /// Per-session inbound sinks.
    inbound: Mutex<HashMap<SessionId, InboundSender>>,
    /// Per-session focused-scope map (what the user has navigated to,
    /// keyed by scope kind). Updated when a widget event carries
    /// `data.scope = {kind, key}` (or, for back-compat,
    /// `data.section = N`). Emitted into STATE_SNAPSHOT and pushed as
    /// STATE_DELTA patches on change. Purely AG-UI-layer state; not
    /// persisted anywhere else. See
    /// `docs/design-json-render-migration.md` §3.1.
    focused_scope: Mutex<HashMap<SessionId, HashMap<String, serde_json::Value>>>,
    /// App-wide branding. Server-set once at construction; included in
    /// every STATE_SNAPSHOT under `/branding`. Frontend reads
    /// `{$state: "/branding/title"}` etc. so each demo
    /// (Frankenstein / data-explorer / future) can label itself
    /// instead of all sharing a hardcoded "Frankenstein" header.
    branding: std::sync::Mutex<Option<Branding>>,
    /// Per-session spine widget registry. The first
    /// `SpineSegmentsUpdated` for a session emits a fresh
    /// `SemanticSpine` widget; subsequent updates supersede it on
    /// the same surface. Map persists `(surface_id, widget_id)` so
    /// we can pin the supersede chain to one surface — moving the
    /// spine across surfaces would confuse focus tracking.
    spine_widgets: Mutex<HashMap<SessionId, (UiSurfaceId, WidgetId)>>,
}

/// App-wide branding shown in the frontend header. `layout` is an
/// optional hint the frontend uses to pick a top-level grid:
///
///   - None / `"chat-primary"` (default) — chat fills, canvas is a
///     narrow right rail. Good for chat demos where the canvas
///     holds compact navigation (Frankenstein chapter picker).
///   - `"canvas-primary"` — canvas fills, chat is a narrow left
///     rail. Good for data / dashboard demos where the canvas holds
///     wide content (DataTable in the SQL explorer).
#[derive(Debug, Clone)]
pub struct Branding {
    pub title: String,
    pub subtitle: String,
    pub layout: Option<String>,
    /// Pre-session welcome copy + suggested starter prompts shown by
    /// the frontend's empty-state landing page. None falls back to a
    /// generic prompt; each example sets its own via `set_welcome`.
    pub welcome: Option<Welcome>,
}

/// Landing-page copy for an example. Shipped alongside `Branding`
/// inside `/branding/welcome` and consumed by `ChatPane` when there
/// are no messages yet.
#[derive(Debug, Clone)]
pub struct Welcome {
    pub heading: String,
    pub body: String,
    pub suggestions: Vec<String>,
}

/// Public-facing adapter. Construct once per gw instance and share.
pub struct AgUiAdapter {
    state: Arc<AgUiState>,
    store_task: tokio::task::JoinHandle<()>,
}

impl AgUiAdapter {
    pub fn new(store: &Arc<UiSurfaceStore>) -> Self {
        // Subscribe synchronously before spawning the forwarder so we
        // don't miss notifications emitted in the window between
        // construction and the task's first poll.
        let mut rx = store.subscribe();

        let state = Arc::new(AgUiState {
            store: store.clone(),
            sessions: Mutex::new(HashMap::new()),
            inbound: Mutex::new(HashMap::new()),
            focused_scope: Mutex::new(HashMap::new()),
            branding: std::sync::Mutex::new(None),
            spine_widgets: Mutex::new(HashMap::new()),
        });

        let st = state.clone();
        let store_task = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(notif) => {
                        // Every UiSurfaceStore notification projects to
                        // a single STATE_DELTA (JSON-Patch array) against
                        // the canonical state shape.
                        let Some(session_id) = notification_session(&st.store, &notif).await else {
                            continue;
                        };
                        let surface_id = notification_surface(&st.store, &notif)
                            .await
                            .unwrap_or_default();
                        let Some(patches) = notification_to_patches(&st.store, &notif).await else {
                            continue;
                        };
                        let sessions = st.sessions.lock().await;
                        if let Some(tx) = sessions.get(&session_id) {
                            let _ = tx.send(AgUiEvent::StateDelta {
                                surface_id,
                                patches,
                            });
                        }
                    }
                    Err(broadcast::error::RecvError::Lagged(n)) => {
                        warn!(
                            lagged = n,
                            "ag-ui store subscriber lagged; some notifications dropped"
                        );
                    }
                    Err(broadcast::error::RecvError::Closed) => break,
                }
            }
        });

        Self { state, store_task }
    }

    /// Register a session with the adapter. `event_tx` is the channel
    /// the adapter pushes inbound `LoopEvent`s through (the conversation
    /// loop's inbound sender).
    pub async fn register_session(&self, session_id: SessionId, event_tx: InboundSender) {
        self.state.inbound.lock().await.insert(session_id, event_tx);
        let mut sessions = self.state.sessions.lock().await;
        sessions
            .entry(session_id)
            .or_insert_with(|| broadcast::channel::<AgUiEvent>(256).0);
    }

    /// Set the app-wide branding (title + subtitle) that appears in
    /// the frontend header. Called once at startup by each example.
    /// Surfaced in every subsequent STATE_SNAPSHOT. Layout defaults
    /// to None (chat-primary); use `set_layout` to override.
    pub fn set_branding(&self, title: impl Into<String>, subtitle: impl Into<String>) {
        let mut b = self.state.branding.lock().expect("branding mutex poisoned");
        let prev_layout = b.as_ref().and_then(|b| b.layout.clone());
        let prev_welcome = b.as_ref().and_then(|b| b.welcome.clone());
        *b = Some(Branding {
            title: title.into(),
            subtitle: subtitle.into(),
            layout: prev_layout,
            welcome: prev_welcome,
        });
    }

    /// Set the empty-state landing copy for this demo. Heading + body
    /// run above the input box; each `suggestions` string becomes a
    /// click-to-submit chip. Call after `set_branding`.
    pub fn set_welcome(
        &self,
        heading: impl Into<String>,
        body: impl Into<String>,
        suggestions: impl IntoIterator<Item = impl Into<String>>,
    ) {
        let mut b = self.state.branding.lock().expect("branding mutex poisoned");
        let welcome = Welcome {
            heading: heading.into(),
            body: body.into(),
            suggestions: suggestions.into_iter().map(Into::into).collect(),
        };
        if let Some(existing) = b.as_mut() {
            existing.welcome = Some(welcome);
        } else {
            *b = Some(Branding {
                title: String::new(),
                subtitle: String::new(),
                layout: None,
                welcome: Some(welcome),
            });
        }
    }

    /// Override the layout hint. Call after `set_branding`. See
    /// `Branding` for the recognised values.
    pub fn set_layout(&self, layout: impl Into<String>) {
        let mut b = self.state.branding.lock().expect("branding mutex poisoned");
        let layout = layout.into();
        if let Some(existing) = b.as_mut() {
            existing.layout = Some(layout);
        } else {
            *b = Some(Branding {
                title: String::new(),
                subtitle: String::new(),
                layout: Some(layout),
                welcome: None,
            });
        }
    }

    pub async fn unregister_session(&self, session_id: SessionId) {
        self.state.inbound.lock().await.remove(&session_id);
        self.state.sessions.lock().await.remove(&session_id);
        self.state.focused_scope.lock().await.remove(&session_id);
    }

    /// Dispatch a `LoopEvent` to a specific session's SSE stream. Called
    /// by the conversation loop for outbound events (Response,
    /// TurnComplete, etc.). Events without an outbound projection are
    /// silently dropped.
    pub async fn dispatch(&self, session_id: SessionId, event: &LoopEvent) {
        // Side-effect: spine segment updates project into the widget
        // store so the existing notification → STATE_DELTA path
        // delivers the new segments to the frontend with no
        // additional plumbing. The first SpineSegmentsUpdated for
        // a session creates the SemanticSpine widget; subsequent
        // updates supersede it on the same surface so the spine
        // pane keeps a stable widget id to focus on.
        if let LoopEvent::SpineSegmentsUpdated { segments, .. } = event {
            self.update_spine_widget(session_id, segments).await;
        }

        let Some(ag) = loop_event_to_ag_ui(event) else {
            return;
        };
        let sessions = self.state.sessions.lock().await;
        if let Some(tx) = sessions.get(&session_id) {
            let _ = tx.send(ag);
        }
    }

    /// Project a `SpineSegmentsUpdated` event into a SemanticSpine
    /// widget. Lazily creates the widget on the first update for a
    /// session; supersedes thereafter so the widget id stays stable
    /// for the chain (frontend can hold a focusedSegmentId across
    /// updates without losing context).
    async fn update_spine_widget(
        &self,
        session_id: SessionId,
        segments: &[SpineSegmentSnapshot],
    ) {
        let payload = build_spine_payload(segments);
        let mut spine = self.state.spine_widgets.lock().await;
        match spine.get(&session_id).copied() {
            Some((surface_id, old_id)) => {
                let new_widget = Widget {
                    id: WidgetId::new(),
                    surface_id,
                    session_id,
                    origin_entry: None,
                    kind: WidgetKind::A2ui,
                    state: WidgetState::Active,
                    payload: WidgetPayload::Inline(payload),
                    supersedes: Some(old_id),
                    created_at: chrono::Utc::now(),
                    resolved_at: None,
                    resolution: None,
                    multi_use: true,
                    follow_up: false,
                    scope: None,
                };
                let new_id = new_widget.id;
                if let Err(e) = self.state.store.supersede(old_id, new_widget).await {
                    warn!(error = %e, "spine widget supersede failed");
                    return;
                }
                spine.insert(session_id, (surface_id, new_id));
            }
            None => {
                let surface_id = UiSurfaceId::new();
                let widget = Widget {
                    id: WidgetId::new(),
                    surface_id,
                    session_id,
                    origin_entry: None,
                    kind: WidgetKind::A2ui,
                    state: WidgetState::Active,
                    payload: WidgetPayload::Inline(payload),
                    supersedes: None,
                    created_at: chrono::Utc::now(),
                    resolved_at: None,
                    resolution: None,
                    multi_use: true,
                    follow_up: false,
                    scope: None,
                };
                let widget_id = widget.id;
                if let Err(e) = self.state.store.emit(widget).await {
                    warn!(error = %e, "spine widget emit failed");
                    return;
                }
                spine.insert(session_id, (surface_id, widget_id));
            }
        }
    }

    /// Subscribe directly to a session's AG-UI event stream. Used by
    /// tests and by the SSE endpoint; callers outside the adapter
    /// should prefer the HTTP endpoint.
    pub async fn subscribe_session(&self, session_id: SessionId) -> broadcast::Receiver<AgUiEvent> {
        let mut sessions = self.state.sessions.lock().await;
        let tx = sessions
            .entry(session_id)
            .or_insert_with(|| broadcast::channel::<AgUiEvent>(256).0);
        tx.subscribe()
    }

    /// axum `Router` exposing the AG-UI endpoints. Mount under whatever
    /// prefix you like in `gw-server`.
    pub fn router(&self) -> Router {
        Router::new()
            .route("/sessions/{session_id}/stream", get(sse_handler))
            .route("/sessions/{session_id}/messages", post(post_message))
            .route(
                "/sessions/{session_id}/widget-events",
                post(post_widget_event),
            )
            .route("/sessions/{session_id}/surface", get(get_surface))
            .with_state(self.state.clone())
    }
}

impl Drop for AgUiAdapter {
    fn drop(&mut self) {
        self.store_task.abort();
    }
}

fn parse_sid(s: &str) -> Result<SessionId, (StatusCode, String)> {
    s.parse::<Uuid>()
        .map(SessionId)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid session_id: {e}")))
}

async fn sse_handler(
    Path(session_id): Path<String>,
    State(state): State<Arc<AgUiState>>,
) -> Result<Sse<impl Stream<Item = Result<Event, axum::Error>>>, (StatusCode, String)> {
    let sid = parse_sid(&session_id)?;
    let tx = {
        let mut sessions = state.sessions.lock().await;
        sessions
            .entry(sid)
            .or_insert_with(|| broadcast::channel::<AgUiEvent>(256).0)
            .clone()
    };
    let rx = tx.subscribe();
    debug!(session_id = %sid.0, "ag-ui sse subscribed");

    // Always build a STATE_SNAPSHOT so a vanilla AG-UI client can
    // render without a separate /surface fetch — and so that
    // session-level state like branding always reaches the client,
    // even on empty sessions before the first widget is emitted.
    // (Previously this fell through to None on Err, which meant
    // tabs that subscribed early — before the agent ran the first
    // turn — never received branding and were stuck on the default
    // header / layout.)
    let snap = state
        .store
        .snapshot(sid)
        .await
        .unwrap_or_else(|_| empty_snapshot(sid));
    let focus = state
        .focused_scope
        .lock()
        .await
        .get(&sid)
        .cloned()
        .unwrap_or_default();
    let branding = state
        .branding
        .lock()
        .expect("branding mutex poisoned")
        .clone();
    let snapshot_event = AgUiEvent::StateSnapshot {
        surface_id: snap.surface.id.0.to_string(),
        state: canonical_state(&snap, &focus, branding.as_ref()),
    };
    let snapshot_stream = futures::stream::iter(std::iter::once(snapshot_event).map(|ev| {
        Event::default()
            .json_data(&ev)
            .map_err(|_| axum::Error::new("serialise STATE_SNAPSHOT"))
    }));

    let live_stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(ev) => Event::default().json_data(&ev).ok().map(Ok),
        Err(_lagged) => None,
    });

    let stream = snapshot_stream.chain(live_stream);
    Ok(Sse::new(stream).keep_alive(KeepAlive::default()))
}

async fn post_message(
    Path(session_id): Path<String>,
    State(state): State<Arc<AgUiState>>,
    Json(body): Json<PostMessageBody>,
) -> Result<StatusCode, (StatusCode, String)> {
    let sid = parse_sid(&session_id)?;
    let inbound = state.inbound.lock().await;
    let tx = inbound.get(&sid).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "session not registered with ag-ui adapter".to_string(),
        )
    })?;
    tx.send(LoopEvent::UserMessage(body.content)).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "session inbound channel closed".to_string(),
        )
    })?;
    Ok(StatusCode::ACCEPTED)
}

async fn post_widget_event(
    Path(session_id): Path<String>,
    State(state): State<Arc<AgUiState>>,
    Json(body): Json<WidgetEvent>,
) -> Result<StatusCode, (StatusCode, String)> {
    let sid = parse_sid(&session_id)?;

    // Resolve the widget in the store first — unless it's `multi_use`,
    // in which case clicks are pure events and the widget stays
    // `Active` for further interaction. Best-effort: a widget that is
    // already terminal (superseded, expired, or resolved by another
    // interaction in flight) should still forward the interaction to
    // the agent, but we log the anomaly.
    let multi_use = state
        .store
        .get_widget(body.widget_id)
        .await
        .map(|w| w.multi_use)
        .unwrap_or(false);
    if !multi_use {
        if let Err(e) = state.store.resolve(body.widget_id, body.data.clone()).await {
            warn!(
                widget_id = ?body.widget_id,
                error = %e,
                "ag-ui widget event for non-active widget; forwarding interaction anyway"
            );
        }
    }

    // Infer focused-scope updates from the click payload. Either
    // `data.scope = {kind, key}` (preferred) or `data.section = N`
    // (back-compat while the existing Frankenstein prompts still use
    // the old convention). Emits a STATE_DELTA and updates the
    // adapter's per-session focus map so subsequent STATE_SNAPSHOTs
    // include the current focus.
    if let Some((kind, key)) = extract_scope_update(&body.data) {
        let mut map = state.focused_scope.lock().await;
        let entry = map.entry(sid).or_default();
        let changed = entry.get(&kind) != Some(&key);
        if changed {
            entry.insert(kind.clone(), key.clone());
            let patch = serde_json::json!({
                "op": "replace",
                "path": format!("/focusedScope/{}", kind),
                "value": key,
            });
            drop(map);
            let sessions = state.sessions.lock().await;
            if let Some(tx) = sessions.get(&sid) {
                let _ = tx.send(AgUiEvent::StateDelta {
                    surface_id: body.surface_id.0.to_string(),
                    patches: vec![patch],
                });
            }
        }
    }

    let inbound = state.inbound.lock().await;
    let tx = inbound.get(&sid).ok_or_else(|| {
        (
            StatusCode::NOT_FOUND,
            "session not registered with ag-ui adapter".to_string(),
        )
    })?;
    tx.send(LoopEvent::WidgetInteraction(body)).map_err(|_| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            "session inbound channel closed".to_string(),
        )
    })?;
    Ok(StatusCode::ACCEPTED)
}

/// Inspect a WidgetEvent's `data` payload for a focused-scope update.
/// Accepts two shapes:
///   - `{"scope": {"kind": "...", "key": <any-json>}}` — canonical form.
///   - `{"section": N}` — Frankenstein-demo back-compat; treated as
///     `kind="section"`, `key=N`.
fn extract_scope_update(data: &serde_json::Value) -> Option<(String, serde_json::Value)> {
    if let Some(scope) = data.get("scope").and_then(|v| v.as_object()) {
        let kind = scope.get("kind")?.as_str()?.to_string();
        let key = scope.get("key")?.clone();
        return Some((kind, key));
    }
    if let Some(section) = data.get("section") {
        if !section.is_null() {
            return Some(("section".into(), section.clone()));
        }
    }
    None
}

async fn get_surface(
    Path(session_id): Path<String>,
    State(state): State<Arc<AgUiState>>,
) -> Result<Json<UiSurfaceSnapshot>, (StatusCode, String)> {
    let sid = parse_sid(&session_id)?;
    match state.store.snapshot(sid).await {
        Ok(snap) => Ok(Json(snap)),
        Err(e) => {
            // Treat missing surface as an empty one rather than 404 — a
            // fresh session that has never emitted a widget is normal.
            debug!(error = %e, "ag-ui surface snapshot: returning empty");
            Ok(Json(empty_snapshot(sid)))
        }
    }
}

fn empty_snapshot(session_id: SessionId) -> UiSurfaceSnapshot {
    use crate::surface::UiSurface;
    use gw_core::UiSurfaceId;
    UiSurfaceSnapshot {
        surface: UiSurface {
            id: UiSurfaceId::new(),
            session_id,
            widget_order: Vec::new(),
            canvas_slot: None,
            canvas_aux_slot: None,
        },
        widgets: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;
    use gw_core::{UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState};
    use serde_json::json;

    #[test]
    fn extract_scope_update_reads_canonical_shape() {
        let data = json!({"scope": {"kind": "section", "key": 4}});
        let (kind, key) = extract_scope_update(&data).unwrap();
        assert_eq!(kind, "section");
        assert_eq!(key, json!(4));
    }

    #[test]
    fn extract_scope_update_back_compat_section() {
        let data = json!({"section": 7});
        let (kind, key) = extract_scope_update(&data).unwrap();
        assert_eq!(kind, "section");
        assert_eq!(key, json!(7));
    }

    #[test]
    fn extract_scope_update_none_when_absent() {
        assert!(extract_scope_update(&json!({})).is_none());
        assert!(extract_scope_update(&json!({"foo": "bar"})).is_none());
        assert!(extract_scope_update(&json!({"section": null})).is_none());
    }

    #[test]
    fn extract_scope_update_rejects_partial_scope() {
        assert!(extract_scope_update(&json!({"scope": {"kind": "section"}})).is_none());
        assert!(extract_scope_update(&json!({"scope": {"key": 4}})).is_none());
    }

    fn active_widget(session: SessionId, surface: UiSurfaceId) -> Widget {
        Widget {
            id: WidgetId::new(),
            surface_id: surface,
            session_id: session,
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

    #[tokio::test]
    async fn register_and_dispatch_reaches_subscriber() {
        let store = Arc::new(UiSurfaceStore::new());
        let adapter = AgUiAdapter::new(&store);

        let sid = SessionId(Uuid::new_v4());
        let (tx, _rx) = mpsc::unbounded_channel();
        adapter.register_session(sid, tx).await;

        let mut sub = adapter.subscribe_session(sid).await;
        let message_id = "msg-1".to_string();
        adapter
            .dispatch(
                sid,
                &LoopEvent::TextMessageDelta {
                    message_id: message_id.clone(),
                    delta: "hi".into(),
                    model: None,
                },
            )
            .await;

        let ev = tokio::time::timeout(std::time::Duration::from_millis(100), sub.recv())
            .await
            .unwrap()
            .unwrap();
        match ev {
            AgUiEvent::TextMessageContent { delta, .. } => assert_eq!(delta, "hi"),
            other => panic!("expected TextMessageContent, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn store_emit_reaches_session_subscriber() {
        let store = Arc::new(UiSurfaceStore::new());
        let adapter = AgUiAdapter::new(&store);

        let sid = SessionId(Uuid::new_v4());
        let (tx, _rx) = mpsc::unbounded_channel();
        adapter.register_session(sid, tx).await;

        let mut sub = adapter.subscribe_session(sid).await;

        let w = active_widget(sid, UiSurfaceId::new());
        store.emit(w).await.unwrap();

        // Phase 3c: single vanilla STATE_DELTA per notification.
        let ev = tokio::time::timeout(std::time::Duration::from_millis(200), sub.recv())
            .await
            .unwrap()
            .unwrap();
        match ev {
            AgUiEvent::StateDelta { patches, .. } => {
                assert!(!patches.is_empty());
                assert_eq!(patches[0]["op"], "add");
            }
            other => panic!("expected STATE_DELTA, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn unregister_removes_session() {
        let store = Arc::new(UiSurfaceStore::new());
        let adapter = AgUiAdapter::new(&store);

        let sid = SessionId(Uuid::new_v4());
        let (tx, _rx) = mpsc::unbounded_channel();
        adapter.register_session(sid, tx).await;
        adapter.unregister_session(sid).await;

        assert!(adapter.state.inbound.lock().await.get(&sid).is_none());
        assert!(adapter.state.sessions.lock().await.get(&sid).is_none());
    }
}
