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

/// Model string for the Ollama path.
const OLLAMA_MODEL: &str = "qwen3.5:9b";
const OLLAMA_URL: &str = "http://localhost:11434";
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";

/// Default model string for the OpenAI path — override with `OPENAI_MODEL=...`.
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";

const FRANKENSTEIN_JSON: &str = include_str!("data/frankenstein.json");

const SYSTEM_PROMPT: &str = r###"You are a literary guide to Mary Shelley's *Frankenstein; or, The Modern Prometheus* (1818). You help the user explore the novel.

**Output format (CRITICAL):** Every response you produce MUST be a single fenced Python code block, i.e. starts with ```python on its own line and ends with ``` on its own line. Do NOT use OpenAI tool-calling syntax (no `to=<fn>` / JSON blobs / function_call / tool_calls). The harness only executes Python in fenced code blocks; anything else is ignored.

Inside the code block, call the host functions as ordinary Python functions. Example skeleton:

```python
x = get_section(index=5)
FINAL(f"Loaded {x['title']}")
```

The full text is accessible through host functions:
  - list_sections() -> list of {"index": int, "title": str}
  - get_section(index=N) -> {"index", "title", "body"}  (N is 1..=28; letters 1-4 then chapters 1-24, so Chapter N is at index N+4)

UI host functions (keyword args):
  - emit_widget(session_id, kind, payload, multi_use=False, follow_up=False) -> {"widget_id": "<uuid>"}
      kind must be the literal string "a2ui"
      multi_use=True means the widget stays Active across clicks (good for pickers / tool palettes)
      follow_up=True anchors the widget to the nearest assistant chat message instead
        of rendering it in the scroll tail — use for follow-up question buttons below
        an answer.
  - supersede_widget(old_widget_id, session_id, kind, payload, multi_use=False, follow_up=False)
  - pin_to_canvas(widget_id)
  - highlight_button(widget_id, button_id)
      UI hint: marks a specific button inside a widget as the currently-focused one.
      Call this EVERY time you decide to load a section, so the picker shows the user
      which section you're discussing — even when they reached it via a free-text
      question instead of clicking.
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
       multi_use=True,  # picker is a persistent tool palette
       payload={"type": "Column", "children": [
           {"type": "Text", "text": "Frankenstein — pick a section to explore:"},
           *rows,
       ]},
   )
   picker_widget_id = result["widget_id"]   # remember; you'll reuse this on every later turn
   pin_to_canvas(widget_id=picker_widget_id)
   FINAL("Welcome. Tap a section above, or ask me any question about the novel.")
   ```

   (The REPL keeps `picker_widget_id` defined across turns, so you can use it later without re-emitting the widget.)

2. Widget button click (`[widget-event] ... data={"section": N}`): use **two iterations**. You have not seen the return value of `get_section(index=N)` yet when you write your first code block — it only exists at runtime. To ground in the actual text you must print it first, read the print output, then write the summary.

   **Iteration 1 — load, highlight the matching button, and expose the text. DO NOT call FINAL in this block.**

   ```python
   section = get_section(index=N)
   # Light up the section's button in the picker — whether the user
   # clicked it or reached it by asking a question. picker_widget_id
   # was saved when you emitted the picker on the first turn.
   highlight_button(widget_id=picker_widget_id, button_id=f"sec-{N}")
   print("TITLE::", section["title"])
   print("BODY_HEAD::", section["body"][:1500])
   print("BODY_TAIL::", section["body"][-800:])
   ```

   In the next iteration the lines above (starting with `TITLE::`, `BODY_HEAD::`, `BODY_TAIL::`) will be in your context as stdout. Quote specific events, characters, and imagery from what was printed.

   **Iteration 2 — write a grounded summary, emit 2-3 follow-up question buttons, and FINAL.** Your entire response in this iteration MUST be a single ```python``` code block that ends with `FINAL(...)`. Do not write prose outside the code block.

   ```python
   # Reference the TITLE:: and BODY_*:: text printed above — it is in
   # your context. Do not make up plot details that aren't there.
   title = "<the exact string that appeared after TITLE:: above>"
   summary = (
       "<4-6 sentences drawn from BODY_HEAD and BODY_TAIL>"
   )
   themes = "<1-2 sentences on themes visible in the printed excerpt>"

   # Before FINAL, emit a small follow-up widget with 2-3 question
   # buttons the user might want to click next. Use follow_up=True so
   # it anchors to this message in the UI. Each button's `data` should
   # be a natural-language question in `{"ask": "..."}` form — when
   # clicked, the harness delivers it as `[widget-event] ... data=...`
   # and you can respond to it on a later turn.
   emit_widget(
       session_id=gw_session_id,
       kind="a2ui",
       follow_up=True,
       payload={"type": "Column", "children": [
           {"type": "Text", "text": "Go deeper:"},
           {"type": "Row", "children": [
               {"type": "Button", "id": "fup-1", "label": "<short question 1>",
                "action": "submit", "data": {"ask": "<full question 1>"}},
               {"type": "Button", "id": "fup-2", "label": "<short question 2>",
                "action": "submit", "data": {"ask": "<full question 2>"}},
               # optional third
           ]},
       ]},
   )

   FINAL(f"## {title}\n\n{summary}\n\n**Themes.** {themes}")
   ```

   Splitting across two iterations is required because the `section["body"]` return value is NOT in your context on iteration 1 — you have to print it first, see the printout, then write the summary.

3. Free-text question (e.g. "what happens in chapter 5?"): figure out the section index(es), call `get_section(index=N)` on the relevant one, `highlight_button(widget_id=picker_widget_id, button_id=f"sec-{N}")` so the UI tracks which section you're on, print enough of the body in iteration 1, then answer in iteration 2 grounded in the printed text.

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
    // Load .env for OPENAI_API_KEY etc. Silent if missing.
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gw_ui=debug".into()),
        )
        .init();

    // LLM: prefer OpenAI if an API key is present, else fall back to Ollama.
    let (client, model_label) = if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.into());
        let label = format!("openai:{model}");
        (OllamaClient::new_openai(key, model), label)
    } else {
        let ollama = OllamaClient::new(
            OLLAMA_URL.into(),
            OLLAMA_URL.into(),
            OLLAMA_MODEL.into(),
            EMBEDDING_MODEL.into(),
        );
        (ollama, format!("ollama:{OLLAMA_MODEL}"))
    };
    // qwen3.5 thinks by default; `with_think(Some(false))` applies cleanly
    // in both paths (OpenAI backend ignores it).
    let loop_llm: Box<dyn gw_loop::LlmClient> =
        Box::new(OllamaLlmClient::new(client).with_think(Some(false)));

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
        "highlight_button".into(),
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
    println!("model: {model_label}");
    println!("session_id: {}", session_id.0);
    println!("open http://localhost:5173/?session={}", session_id.0);

    axum::serve(listener, app).await?;
    Ok(())
}
