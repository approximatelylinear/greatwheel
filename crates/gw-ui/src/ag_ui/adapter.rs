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
use gw_core::{LoopEvent, SessionId, WidgetEvent};
use tokio::sync::{broadcast, mpsc, Mutex};
use tokio_stream::{wrappers::BroadcastStream, StreamExt};
use tracing::{debug, warn};
use uuid::Uuid;

use crate::surface::{UiSurfaceSnapshot, UiSurfaceStore};

use super::codec::{loop_event_to_ag_ui, notification_to_ag_ui};
use super::events::{AgUiEvent, PostMessageBody};

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
        });

        let st = state.clone();
        let store_task = tokio::spawn(async move {
            loop {
                match rx.recv().await {
                    Ok(notif) => {
                        let Some((session_id, ev)) = notification_to_ag_ui(&st.store, notif).await
                        else {
                            continue;
                        };
                        let sessions = st.sessions.lock().await;
                        if let Some(tx) = sessions.get(&session_id) {
                            let _ = tx.send(ev);
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

    pub async fn unregister_session(&self, session_id: SessionId) {
        self.state.inbound.lock().await.remove(&session_id);
        self.state.sessions.lock().await.remove(&session_id);
    }

    /// Dispatch a `LoopEvent` to a specific session's SSE stream. Called
    /// by the conversation loop for outbound events (Response,
    /// TurnComplete, etc.). Events without an outbound projection are
    /// silently dropped.
    pub async fn dispatch(&self, session_id: SessionId, event: &LoopEvent) {
        let Some(ag) = loop_event_to_ag_ui(event) else {
            return;
        };
        let sessions = self.state.sessions.lock().await;
        if let Some(tx) = sessions.get(&session_id) {
            let _ = tx.send(ag);
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

    let stream = BroadcastStream::new(rx).filter_map(|msg| match msg {
        Ok(ev) => Event::default().json_data(&ev).ok().map(Ok),
        Err(_lagged) => None,
    });

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
        adapter
            .dispatch(
                sid,
                &LoopEvent::Response {
                    content: "hi".into(),
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

        let ev = tokio::time::timeout(std::time::Duration::from_millis(200), sub.recv())
            .await
            .unwrap()
            .unwrap();
        assert!(matches!(ev, AgUiEvent::UiEvent { .. }));
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
