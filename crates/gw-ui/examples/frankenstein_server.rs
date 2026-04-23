//! "Explore Frankenstein" — a Frankenstein-themed agent backed by the
//! full rLM loop + Ollama + gw-ui. Demonstrates the whole generative-UX
//! stack: host-function-driven content access, an A2UI chapter picker
//! pinned to the canvas, interactive widget → turn round-trip, and
//! free-form chat grounded in the text.
//!
//! The corpus (Gutenberg eBook #84) is baked into the binary via
//! `include_str!` as `data/frankenstein.json`.
//!
//! Prerequisites:
//!     ollama pull qwen3.5:9b
//!     ollama serve
//!
//! Run:
//!     cargo run -p gw-ui --example frankenstein_server
//!
//! Then point the frontend at the printed session UUID and ask the
//! agent about the novel.

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{LoopEvent, Plugin, PluginContext, PluginError, PluginManifest, SessionId};
use gw_engine::GreatWheelEngine;
use gw_llm::OllamaClient;
use gw_loop::bridge::{new_ask_handle, ConversationBridge};
use gw_loop::{ConversationLoop, LoopConfig, OllamaLlmClient, SnapshotPolicy};
use gw_runtime::ReplAgent;
use gw_ui::{AgUiAdapter, UiPlugin, UiSurfaceStore};
use ouros::Object;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use uuid::Uuid;

const MODEL: &str = "qwen3.5:9b";
const OLLAMA_URL: &str = "http://localhost:11434";
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";

const FRANKENSTEIN_JSON: &str = include_str!("data/frankenstein.json");

const SYSTEM_PROMPT: &str = r###"You are a literary guide to Mary Shelley's *Frankenstein; or, The Modern Prometheus* (1818). You help the user explore the novel.

The full text is accessible through host functions:
  - list_sections() -> list of {"index": int, "title": str}
  - get_section(index=N) -> {"index", "title", "body"}  (N is 1..=28; letters 1-4 then chapters 1-24, so Chapter N is at index N+4)

UI host functions (keyword args):
  - emit_widget(session_id, kind, payload) -> {"widget_id": "<uuid>"}   kind must be the literal string "a2ui"
  - supersede_widget(old_widget_id, session_id, kind, payload)
  - pin_to_canvas(widget_id)
  - FINAL(text)

Your session id is the Python variable `gw_session_id`.

**Grounding rule:** whenever you discuss a specific section, call `get_section(index=N)` first and base your answer only on the returned `body`. Do not rely on memory for plot details.

**Behaviour**

1. First user turn (greeting): show a chapter picker pinned to the canvas. Group buttons in rows of 7.

   ```python
   sections = list_sections()
   buttons = [
       {"type": "Button", "id": f"sec-{s['index']}", "label": s["title"],
        "action": "submit", "data": {"section": s["index"]}}
       for s in sections
   ]
   rows = [{"type": "Row", "children": buttons[i:i+7]} for i in range(0, len(buttons), 7)]
   result = emit_widget(
       session_id=gw_session_id,
       kind="a2ui",
       payload={"type": "Column", "children": [
           {"type": "Text", "text": "Frankenstein — pick a section to explore:"},
           *rows,
       ]},
   )
   pin_to_canvas(widget_id=result["widget_id"])
   FINAL("Welcome. Tap a section above, or ask me any question about the novel.")
   ```

2. Widget button click (`[widget-event] ... data={"section": N}`): call `get_section(index=N)`, then summarise from the returned body in 4-6 sentences plus 1-2 themes. Start your reply with `## {section["title"]}` so the user can see which section you loaded.

3. Free-text question: call `list_sections()`, then `get_section()` on the relevant section(s), then answer grounded in the returned bodies. Quote briefly when useful.

4. Always end each turn with exactly one `FINAL(text)` call.
"###;

// ── Frankenstein corpus + plugin ─────────────────────────────────────

#[derive(Debug, Deserialize, Serialize)]
struct Section {
    index: usize,
    title: String,
    body: String,
}

#[derive(Debug, Deserialize, Serialize)]
struct Corpus {
    title: String,
    author: String,
    source: String,
    sections: Vec<Section>,
}

struct FrankensteinPlugin {
    corpus: Arc<Corpus>,
}

impl FrankensteinPlugin {
    fn load() -> Result<Self, Box<dyn std::error::Error>> {
        let corpus: Corpus = serde_json::from_str(FRANKENSTEIN_JSON)?;
        Ok(Self {
            corpus: Arc::new(corpus),
        })
    }
}

impl Plugin for FrankensteinPlugin {
    fn name(&self) -> &str {
        "frankenstein"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "frankenstein".into(),
                "host_fn:list_sections".into(),
                "host_fn:get_section".into(),
            ],
            requires: vec![],
            priority: 60,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let corpus = self.corpus.clone();
        ctx.register_host_fn_sync("list_sections", None, move |_args, _kwargs| {
            let list: Vec<Value> = corpus
                .sections
                .iter()
                .map(|s| json!({ "index": s.index, "title": s.title }))
                .collect();
            Ok(Value::Array(list))
        });

        let corpus = self.corpus.clone();
        ctx.register_host_fn_sync("get_section", None, move |_args, kwargs| {
            let idx = kwargs
                .get("index")
                .and_then(|v| v.as_u64())
                .ok_or_else(|| PluginError::HostFunction("index required (integer)".into()))?
                as usize;
            let section = corpus
                .sections
                .iter()
                .find(|s| s.index == idx)
                .ok_or_else(|| PluginError::HostFunction(format!("no section {idx}")))?;
            Ok(json!({
                "index": section.index,
                "title": section.title,
                "body": section.body,
            }))
        });

        Ok(())
    }
}

// ── Server ───────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gw_ui=debug".into()),
        )
        .init();

    // LLM
    let ollama = OllamaClient::new(
        OLLAMA_URL.into(),
        OLLAMA_URL.into(),
        MODEL.into(),
        EMBEDDING_MODEL.into(),
    );
    let loop_llm: Box<dyn gw_loop::LlmClient> = Box::new(OllamaLlmClient::new(ollama));

    // Plugins: UiPlugin + FrankensteinPlugin
    let frank = FrankensteinPlugin::load()?;
    tracing::info!(
        title = %frank.corpus.title,
        sections = frank.corpus.sections.len(),
        "frankenstein corpus loaded"
    );

    let engine = GreatWheelEngine::new()
        .add_plugin(UiPlugin)
        .add_plugin(frank)
        .init(&HashMap::new())?;
    let plugin_router = engine.host_fn_router_arc();
    let store: Arc<UiSurfaceStore> = engine
        .registry
        .shared()
        .get::<Arc<UiSurfaceStore>>()
        .cloned()
        .ok_or("UiPlugin did not provide UiSurfaceStore")?;

    // AG-UI
    let adapter = Arc::new(AgUiAdapter::new(&store));
    let session_id = SessionId(Uuid::new_v4());

    // Channels + tap (same pattern as llm_server)
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

    // Bridge + repl
    let ask_handle = new_ask_handle();
    let conv_bridge = ConversationBridge::with_plugin_router(
        tap_tx.clone(),
        ask_handle,
        None,
        Some(plugin_router),
    );

    let external_fns = vec![
        "FINAL".into(),
        "emit_widget".into(),
        "supersede_widget".into(),
        "resolve_widget".into(),
        "pin_to_canvas".into(),
        "emit_mcp_resource".into(),
        "list_sections".into(),
        "get_section".into(),
    ];
    let mut repl = ReplAgent::new(external_fns, Box::new(conv_bridge));
    repl.set_variable("gw_session_id", Object::String(session_id.0.to_string()))
        .ok();

    // Loop
    let config = LoopConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        recency_window: 30,
        max_iterations: 4,
        include_code_output: true,
        repl_output_max_chars: 4000,
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

    // HTTP
    let app = adapter.router().layer(CorsLayer::permissive());
    let listener = TcpListener::bind("127.0.0.1:8787").await?;
    println!("frankenstein server listening on http://127.0.0.1:8787");
    println!("model: {MODEL}");
    println!("session_id: {}", session_id.0);
    println!("open http://localhost:5173/?session={}", session_id.0);

    axum::serve(listener, app).await?;
    Ok(())
}
