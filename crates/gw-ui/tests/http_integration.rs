//! HTTP integration tests for the AG-UI adapter.
//!
//! Binds axum on an ephemeral port and drives the endpoints with
//! `reqwest` to smoke-test the real wire path. SSE parsing is
//! intentionally out of scope here — the internal broadcast flow is
//! already covered by `adapter.rs` unit tests; these tests verify
//! routing, argument extraction, and 404 behaviour.

use std::sync::Arc;

use chrono::Utc;
use gw_core::{
    LoopEvent, SessionId, UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState,
};
use gw_ui::{AgUiAdapter, UiSurfaceStore};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use uuid::Uuid;

struct Server {
    adapter: Arc<AgUiAdapter>,
    store: Arc<UiSurfaceStore>,
    base: String,
}

async fn start_server() -> Server {
    let store = Arc::new(UiSurfaceStore::new());
    let adapter = Arc::new(AgUiAdapter::new(&store));

    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = adapter.router();

    tokio::spawn(async move {
        axum::serve(listener, app).await.unwrap();
    });

    Server {
        adapter,
        store,
        base: format!("http://{addr}"),
    }
}

fn active_widget(session: SessionId, surface: UiSurfaceId) -> Widget {
    Widget {
        id: WidgetId::new(),
        surface_id: surface,
        session_id: session,
        origin_entry: None,
        kind: WidgetKind::A2ui,
        state: WidgetState::Active,
        payload: WidgetPayload::Inline(serde_json::json!({ "type": "Button" })),
        supersedes: None,
        created_at: Utc::now(),
        resolved_at: None,
        resolution: None,
        multi_use: false,
            follow_up: false,
    }
}

#[tokio::test]
async fn post_message_routes_inbound_to_session() {
    let server = start_server().await;
    let sid = SessionId(Uuid::new_v4());
    let (tx, mut rx) = mpsc::unbounded_channel::<LoopEvent>();
    server.adapter.register_session(sid, tx).await;

    let resp = reqwest::Client::new()
        .post(format!("{}/sessions/{}/messages", server.base, sid.0))
        .json(&serde_json::json!({ "content": "hello world" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
        .await
        .unwrap()
        .unwrap();
    match event {
        LoopEvent::UserMessage(content) => assert_eq!(content, "hello world"),
        other => panic!("expected UserMessage, got {:?}", other),
    }
}

#[tokio::test]
async fn post_message_unknown_session_returns_404() {
    let server = start_server().await;
    let unknown = Uuid::new_v4();

    let resp = reqwest::Client::new()
        .post(format!("{}/sessions/{unknown}/messages", server.base))
        .json(&serde_json::json!({ "content": "hi" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::NOT_FOUND);
}

#[tokio::test]
async fn post_message_bad_session_id_returns_400() {
    let server = start_server().await;

    let resp = reqwest::Client::new()
        .post(format!("{}/sessions/not-a-uuid/messages", server.base))
        .json(&serde_json::json!({ "content": "hi" }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn get_surface_returns_empty_for_fresh_session() {
    let server = start_server().await;
    let sid = Uuid::new_v4();

    let resp = reqwest::Client::new()
        .get(format!("{}/sessions/{sid}/surface", server.base))
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::OK);
    let body: serde_json::Value = resp.json().await.unwrap();
    assert_eq!(body["widgets"].as_array().unwrap().len(), 0);
    assert_eq!(
        body["surface"]["session_id"].as_str().unwrap(),
        sid.to_string()
    );
}

#[tokio::test]
async fn post_widget_event_resolves_and_forwards() {
    let server = start_server().await;
    let sid = SessionId(Uuid::new_v4());
    let surface = UiSurfaceId::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<LoopEvent>();
    server.adapter.register_session(sid, tx).await;

    // Emit a widget directly into the store so it's Active.
    let widget = active_widget(sid, surface);
    let widget_id = widget.id;
    server.store.emit(widget).await.unwrap();

    let body = serde_json::json!({
        "widget_id": widget_id.0,
        "surface_id": surface.0,
        "action": "submit",
        "data": { "choice": "yes" },
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/sessions/{}/widget-events", server.base, sid.0))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    // LoopEvent::WidgetInteraction should have been forwarded.
    let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
        .await
        .unwrap()
        .unwrap();
    match event {
        LoopEvent::WidgetInteraction(ev) => {
            assert_eq!(ev.widget_id, widget_id);
            assert_eq!(ev.action, "submit");
            assert_eq!(ev.data, serde_json::json!({ "choice": "yes" }));
        }
        other => panic!("expected WidgetInteraction, got {:?}", other),
    }

    // Widget should have transitioned to Resolved with the posted data.
    let stored = server.store.get_widget(widget_id).await.unwrap();
    assert_eq!(stored.state, WidgetState::Resolved);
    assert_eq!(
        stored.resolution.unwrap(),
        serde_json::json!({ "choice": "yes" })
    );
}

#[tokio::test]
async fn post_widget_event_does_not_resolve_multi_use_widget() {
    let server = start_server().await;
    let sid = SessionId(Uuid::new_v4());
    let surface = UiSurfaceId::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<LoopEvent>();
    server.adapter.register_session(sid, tx).await;

    let mut widget = active_widget(sid, surface);
    widget.multi_use = true;
    let widget_id = widget.id;
    server.store.emit(widget).await.unwrap();

    let body = serde_json::json!({
        "widget_id": widget_id.0,
        "surface_id": surface.0,
        "action": "submit",
        "data": { "choice": "a" },
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/sessions/{}/widget-events", server.base, sid.0))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    // Interaction forwarded.
    let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(event, LoopEvent::WidgetInteraction(_)));

    // Widget stays Active — no resolution, no state change.
    let stored = server.store.get_widget(widget_id).await.unwrap();
    assert_eq!(stored.state, WidgetState::Active);
    assert!(stored.resolution.is_none());

    // Further clicks keep going through without resolving.
    let resp2 = reqwest::Client::new()
        .post(format!("{}/sessions/{}/widget-events", server.base, sid.0))
        .json(&serde_json::json!({
            "widget_id": widget_id.0,
            "surface_id": surface.0,
            "action": "submit",
            "data": { "choice": "b" },
        }))
        .send()
        .await
        .unwrap();
    assert_eq!(resp2.status(), reqwest::StatusCode::ACCEPTED);
    let ev2 = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(ev2, LoopEvent::WidgetInteraction(_)));
    assert_eq!(
        server.store.get_widget(widget_id).await.unwrap().state,
        WidgetState::Active
    );
}

#[tokio::test]
async fn post_widget_event_forwards_even_when_resolve_fails() {
    // A second click on an already-resolved widget should still reach
    // the agent as a WidgetInteraction — the store resolve just logs.
    let server = start_server().await;
    let sid = SessionId(Uuid::new_v4());
    let surface = UiSurfaceId::new();
    let (tx, mut rx) = mpsc::unbounded_channel::<LoopEvent>();
    server.adapter.register_session(sid, tx).await;

    let widget = active_widget(sid, surface);
    let widget_id = widget.id;
    server.store.emit(widget).await.unwrap();
    // Pre-resolve so the adapter's resolve call will fail with InvalidTransition.
    server
        .store
        .resolve(widget_id, serde_json::json!({ "choice": "first" }))
        .await
        .unwrap();
    // Drain the inbound channel of any prior state (there shouldn't be any,
    // but be explicit).
    while rx.try_recv().is_ok() {}

    let body = serde_json::json!({
        "widget_id": widget_id.0,
        "surface_id": surface.0,
        "action": "submit",
        "data": { "choice": "second" },
    });
    let resp = reqwest::Client::new()
        .post(format!("{}/sessions/{}/widget-events", server.base, sid.0))
        .json(&body)
        .send()
        .await
        .unwrap();
    assert_eq!(resp.status(), reqwest::StatusCode::ACCEPTED);

    let event = tokio::time::timeout(std::time::Duration::from_millis(200), rx.recv())
        .await
        .unwrap()
        .unwrap();
    assert!(matches!(event, LoopEvent::WidgetInteraction(_)));
}
