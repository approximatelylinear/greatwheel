//! Standalone AG-UI adapter server for frontend smoke-testing.
//!
//! Binds 127.0.0.1:8787, creates one fixed session (UUID printed at
//! startup), and echoes each user message back. Also emits a sample
//! A2UI "Yes / No" button widget on the first user turn so the
//! frontend has something interactive to render. Interacting with the
//! widget is logged server-side.
//!
//! Run:
//!     cargo run -p gw-ui --example echo_server
//!
//! The frontend should point at `http://127.0.0.1:8787` and use the
//! session UUID printed on startup.

use std::sync::Arc;

use chrono::Utc;
use gw_core::{
    LoopEvent, SessionId, UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState,
};
use gw_ui::{AgUiAdapter, UiSurfaceStore};
use serde_json::json;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt::init();

    let store = Arc::new(UiSurfaceStore::new());
    let adapter = Arc::new(AgUiAdapter::new(&store));

    // One fixed session for the whole run.
    let session_id = SessionId(Uuid::new_v4());
    let surface_id = UiSurfaceId::new();
    let (inbound_tx, mut inbound_rx) = mpsc::unbounded_channel::<LoopEvent>();
    adapter.register_session(session_id, inbound_tx).await;

    // Echo loop — receives inbound events, mirrors back a response + a
    // sample widget on the first turn.
    let adapter_for_loop = adapter.clone();
    let store_for_loop = store.clone();
    tokio::spawn(async move {
        let mut turn = 0u32;
        while let Some(ev) = inbound_rx.recv().await {
            match ev {
                LoopEvent::UserMessage(content) => {
                    turn += 1;
                    tracing::info!(turn, ?content, "user message");
                    // Simulated streaming response.
                    let reply = format!("echo #{turn}: {content}");
                    adapter_for_loop
                        .dispatch(
                            session_id,
                            &LoopEvent::Response {
                                content: reply,
                                model: None,
                            },
                        )
                        .await;
                    // On the first turn, emit an A2UI widget so the
                    // frontend has something interactive to render.
                    if turn == 1 {
                        let widget = sample_widget(session_id, surface_id);
                        store_for_loop.emit(widget).await.ok();
                    }
                    adapter_for_loop
                        .dispatch(session_id, &LoopEvent::TurnComplete)
                        .await;
                }
                LoopEvent::WidgetInteraction(ev) => {
                    tracing::info!(?ev, "widget interaction");
                    adapter_for_loop
                        .dispatch(
                            session_id,
                            &LoopEvent::Response {
                                content: format!(
                                    "received widget event: action={} data={}",
                                    ev.action, ev.data
                                ),
                                model: None,
                            },
                        )
                        .await;
                    adapter_for_loop
                        .dispatch(session_id, &LoopEvent::TurnComplete)
                        .await;
                }
                other => tracing::debug!(?other, "ignored"),
            }
        }
    });

    let app = adapter.router().layer(CorsLayer::permissive());
    let listener = TcpListener::bind("127.0.0.1:8787").await?;
    println!("ag-ui echo server listening on http://127.0.0.1:8787");
    println!("session_id: {}", session_id.0);
    println!(
        "export this for the frontend:\n    VITE_SESSION_ID={}",
        session_id.0
    );
    axum::serve(listener, app).await?;
    Ok(())
}

fn sample_widget(session_id: SessionId, surface_id: UiSurfaceId) -> Widget {
    Widget {
        id: WidgetId::new(),
        surface_id,
        session_id,
        origin_entry: None,
        kind: WidgetKind::A2ui,
        state: WidgetState::Active,
        payload: WidgetPayload::Inline(json!({
            "type": "Column",
            "children": [
                { "type": "Text", "text": "Approve this action?" },
                { "type": "Row", "children": [
                    { "type": "Button", "id": "yes", "label": "Yes",
                      "action": "submit", "data": { "choice": "yes" } },
                    { "type": "Button", "id": "no", "label": "No",
                      "action": "submit", "data": { "choice": "no" } }
                ]}
            ]
        })),
        supersedes: None,
        created_at: Utc::now(),
        resolved_at: None,
        resolution: None,
        multi_use: false,
            follow_up: false,
    }
}
