//! AG-UI server backed by the full rLM loop + Ollama (qwen3.5:9b).
//!
//! Wires `GreatWheelEngine + UiPlugin → HostFnRouter` into a
//! `ConversationBridge`, constructs a `ReplAgent` + `ConversationLoop`,
//! and bridges the loop's event stream into the AG-UI adapter so a
//! browser can drive it.
//!
//! Prerequisites:
//!     ollama pull qwen3.5:9b        # one-time
//!     ollama serve                  # (or make sure the daemon is running)
//!
//! Run:
//!     cargo run -p gw-ui --example llm_server
//!
//! Copy the printed session UUID, then in another terminal:
//!     cd frontend && VITE_SESSION_ID=<uuid> npm run dev
//! or open http://localhost:5173/?session=<uuid>.

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{LoopEvent, SessionId};
use gw_engine::GreatWheelEngine;
use gw_llm::OllamaClient;
use gw_loop::bridge::{new_ask_handle, ConversationBridge};
use gw_loop::{ConversationLoop, LoopConfig, OllamaLlmClient, SnapshotPolicy};
use gw_runtime::ReplAgent;
use gw_ui::{AgUiAdapter, UiPlugin, UiSurfaceStore};
use ouros::Object;
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

const MODEL: &str = "qwen3.5:9b";
const OLLAMA_URL: &str = "http://localhost:11434";
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";

const SYSTEM_PROMPT: &str = r#"You are a helpful assistant running inside a Python REPL sandbox.

For each user turn, write exactly one Python code block. In it, you may optionally call `emit_widget(...)` to show the user an interactive widget, then call `FINAL("text")` to submit your reply for this turn.

Host functions (all keyword arguments):
  - emit_widget(session_id, kind, payload)
      kind="a2ui"  -> payload is an A2UI component tree (see example)
      kind="mcp-ui" -> pass payload_uri (string) + optional payload_csp
  - supersede_widget(old_widget_id, session_id, kind, payload)  — replace an active widget
  - resolve_widget(widget_id, data)                             — agent-driven close
  - pin_to_canvas(widget_id)                                    — move to canvas slot
  - FINAL(text)                                                 — submit your reply

Your session id is available as the Python variable `gw_session_id`. Pass it as session_id= when emitting widgets.

A2UI components you can use in payloads:
  - {"type": "Column", "children": [...]}
  - {"type": "Row", "children": [...]}
  - {"type": "Text", "text": "..."}
  - {"type": "Button", "id": "<id>", "label": "...", "action": "submit", "data": {...}}

Example — plain reply:
```python
FINAL("Hi! How can I help?")
```

Example — reply with a picker:
```python
emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={
        "type": "Column",
        "children": [
            {"type": "Text", "text": "Which one?"},
            {"type": "Row", "children": [
                {"type": "Button", "id": "a", "label": "Option A", "action": "submit", "data": {"choice": "a"}},
                {"type": "Button", "id": "b", "label": "Option B", "action": "submit", "data": {"choice": "b"}},
            ]},
        ],
    },
)
FINAL("Pick one above.")
```

When the user clicks a button you'll receive a message like `[widget-event] widget=<uuid> action=submit data={"choice":"a"}` — react accordingly.

Always call FINAL() exactly once per turn."#;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gw_ui=debug".into()),
        )
        .init();

    // ── LLM ──────────────────────────────────────────────────────────
    let ollama = OllamaClient::new(
        OLLAMA_URL.into(),
        OLLAMA_URL.into(),
        MODEL.into(),
        EMBEDDING_MODEL.into(),
    );
    let loop_llm: Box<dyn gw_loop::LlmClient> =
        Box::new(OllamaLlmClient::new(ollama).with_think(Some(false)));

    // ── Plugins: UiPlugin owns the UiSurfaceStore; pull it back out
    // of SharedState so the AG-UI adapter can share it.
    let engine = GreatWheelEngine::new()
        .add_plugin(UiPlugin)
        .init(&HashMap::new())?;
    let plugin_router = engine.host_fn_router_arc();
    let store: Arc<UiSurfaceStore> = engine
        .registry
        .shared()
        .get::<Arc<UiSurfaceStore>>()
        .cloned()
        .ok_or("UiPlugin did not provide UiSurfaceStore")?;

    // ── AG-UI adapter ────────────────────────────────────────────────
    let adapter = Arc::new(AgUiAdapter::new(&store));
    let session_id = SessionId(Uuid::new_v4());

    // ── Channels + tap ───────────────────────────────────────────────
    // `tap_tx` is the single sender the adapter (for inbound POSTs)
    // and the conversation loop (for self-sent events) both write to.
    // A task reads from tap_rx, dispatches outbound-kind events to the
    // AG-UI adapter, then forwards every event to the loop's actual
    // receiver so internal events (Compact, SessionEnd) still reach
    // the loop's match arms.
    let (tap_tx, mut tap_rx) = mpsc::unbounded_channel::<LoopEvent>();
    let (loop_tx, loop_rx) = mpsc::unbounded_channel::<LoopEvent>();

    adapter.register_session(session_id, tap_tx.clone()).await;

    let adapter_for_tap = adapter.clone();
    tokio::spawn(async move {
        while let Some(ev) = tap_rx.recv().await {
            adapter_for_tap.dispatch(session_id, &ev).await;
            if loop_tx.send(ev).is_err() {
                break;
            }
        }
    });

    // ── ReplAgent + ConversationBridge ───────────────────────────────
    let ask_handle = new_ask_handle();
    let conv_bridge = ConversationBridge::with_plugin_router(
        tap_tx.clone(),
        ask_handle,
        None,
        Some(plugin_router),
    );

    let external_fns = vec![
        "FINAL".to_string(),
        "emit_widget".to_string(),
        "supersede_widget".to_string(),
        "resolve_widget".to_string(),
        "pin_to_canvas".to_string(),
        "emit_mcp_resource".to_string(),
    ];
    let mut repl = ReplAgent::new(external_fns, Box::new(conv_bridge));

    // Make the session id visible to the agent as a Python variable.
    repl.set_variable("gw_session_id", Object::String(session_id.0.to_string()))
        .ok();

    // ── ConversationLoop ─────────────────────────────────────────────
    let config = LoopConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        recency_window: 20,
        max_iterations: 4,
        include_code_output: true,
        repl_output_max_chars: 2000,
        strip_think_tags: true,
        answer_validator: None,
        iteration_callback: None,
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 0,
            before_compaction: false,
        },
        compaction_keep_count: 0,
        auto_compact_after_turns: None,
    };
    let mut conv_loop = ConversationLoop::new(session_id, repl, loop_llm, config, tap_tx);

    // ConversationLoop is !Sync (ouros internals + Box<dyn HostBridge>),
    // so we can't `tokio::spawn` it. Run on a dedicated OS thread with
    // its own multi-threaded runtime — the bridge's `block_in_place` +
    // `block_on` pattern needs multi-threaded. The axum server on the
    // outer #[tokio::main] runtime keeps serving HTTP independently;
    // both runtimes talk via the `tap_tx`/`tap_rx` mpsc channel.
    std::thread::Builder::new()
        .name("gw-loop".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(2)
                .build()
                .expect("failed to build loop runtime");
            rt.block_on(async move {
                if let Err(e) = conv_loop.run(loop_rx).await {
                    tracing::error!(error = %e, "conversation loop exited");
                }
            });
        })?;

    // ── HTTP server ──────────────────────────────────────────────────
    let app = adapter.router().layer(CorsLayer::permissive());
    let listener = TcpListener::bind("127.0.0.1:8787").await?;
    println!("ag-ui llm server listening on http://127.0.0.1:8787");
    println!("model: {MODEL}");
    println!("session_id: {}", session_id.0);
    println!(
        "point the frontend at it:\n    VITE_SESSION_ID={} npm --prefix frontend run dev\n\
         or open http://localhost:5173/?session={}",
        session_id.0, session_id.0,
    );

    axum::serve(listener, app).await?;
    Ok(())
}
