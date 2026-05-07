//! BrowseComp experiment dashboard — runnable Past-mode demo.
//!
//! Wires `BenchPlugin` (read-only host fns over `runs/`) and `UiPlugin`
//! (widget emit / pin) into a `gw-loop` conversation. The agent walks
//! the user through the Past flow described in
//! `docs/design-experiment-dashboard.md` §3b: list runs → select →
//! drill into a query → compare → annotate → cross-run difficulty
//! matrix and cost trend.
//!
//! Run from workspace root:
//!     cargo run -p gw-bench-ui --example bench_dashboard
//!
//! Then either point the existing frontend at the printed session UUID
//! (default port 8787 matches the other gw-ui demos) or set
//! `VITE_API_BASE` if you customised it.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

use gw_bench_ui::{BenchPaths, BenchPlugin, BenchStore};
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
use tower_http::trace::TraceLayer;
use uuid::Uuid;

const OLLAMA_URL: &str = "http://localhost:11434";
const OLLAMA_MODEL: &str = "qwen3.5:9b";
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";
const DEFAULT_PORT: u16 = 8787;

const SYSTEM_PROMPT: &str = r###"You are a research assistant for BrowseComp retrieval experiments. The user analyses runs in `runs/*` (each a directory of per-query JSONs) plus the narrative in `bench/browsecomp/EXPERIMENTS.md`. Load data through the `bench.*` host functions and present results by emitting widgets — keep chat replies short, the data lives in the canvas.

**Output format (CRITICAL):** Every response MUST be a single fenced Python code block — starts with ```python on its own line, ends with ``` on its own line. Do NOT use OpenAI tool-calling syntax. The harness only executes Python in fenced code blocks.

**Python dialect (ouros sandbox):** one module per `import` statement (`import json` then `import os`, never `import a, b`); `from x import a, b` is fine; no starred or relative imports.

# Data host functions (read-only)

  - list_experiments() -> [{"slug", "n_queries", "exact", "fuzzy", "p50_ms", "mtime", "tags"}]
  - load_run(slug=...) -> {"slug", "metadata", "queries": [{"query_id", "status", "exact", "fuzzy", "iterations", "total_ms", "total_tokens", "termination_reason"}], "aggregate": {"n_queries", "exact", "fuzzy", "p50_ms", "p95_ms", "total_input_tokens", "total_output_tokens"}, "notes", "tags"}
  - load_query(slug=..., query_id=...) -> {"question", "gold_answer", "agent_answer", "exact", "fuzzy", "trace", "trajectory", "retrieved_docids", "timing", "usage", "iterations", "termination_reason"}
  - load_config(slug=...) -> {"slug", "toml_path", "contents"}
  - compare(slug_a=..., slug_b=...) -> {"slug_a", "slug_b", "per_query": [{"query_id", "a", "b", "winner"}], "deltas": {"exact_a", "exact_b", "fuzzy_a", "fuzzy_b", "p50_ms_a", "p50_ms_b", "total_tokens_a", "total_tokens_b"}}
  - gold(query_id=...) -> {"query_id", "query", "answer"} or null
  - difficulty_matrix(slugs=[...]) -> {"queries": [qid], "runs": [slug], "cells": [[ "exact"|"fuzzy"|"wrong"|"error"|"missing", ... ]]}
  - query_history(query_id=..., slugs=[...]?) -> {"query_id", "question", "gold_answer", "entries": [{"slug", "status", "exact", "fuzzy", "agent_answer", "retrieved_docids", "iterations", "total_ms"}]}
  - cost_summary(window_days=N?) -> [{"slug", "model", "input_tokens", "output_tokens", "est_usd", "mtime"}]

# Annotation host function (read-write)

  - read_annotations(slug=...) -> {"notes": [{"ts", "text"}], "tags": [str]}
  - write_annotation(slug=..., text=..., tags=[...]?) -> {"notes": [...], "tags": [...]}

# UI host functions

  - emit_widget(session_id, kind, payload, multi_use=False, follow_up=False, scope=None) -> {"widget_id"}
  - supersede_widget(old_widget_id, session_id, kind, payload, ...)
  - pin_to_canvas(widget_id)
  - pin_below_canvas(widget_id)
  - FINAL("text") — terminates the turn with a chat narration.

# Catalog widgets you can emit

Generic: Column / Row / Text / Heading / Link / Button / Card / DataTable.

Bench-specific:
  - Code: {"type": "Code", "content": str, "language": str?, "diff": bool?, "title": str?}
      Monospace block. Set `diff=True` to colour `+` / `-` / `@@` lines as a unified diff.
  - Markdown: {"type": "Markdown", "content": str}
      GitHub-flavoured markdown. Use for EXPERIMENTS.md excerpts or hypothesis narrative.
  - DifficultyMatrix: {"type": "DifficultyMatrix", "queries": [str], "runs": [str], "cells": [[str]]}
      Cells must be one of "exact"/"fuzzy"/"wrong"/"error"/"missing". Click a cell → `{"slug", "query_id", "cell"}`.
  - CostTrend: {"type": "CostTrend", "rows": [<cost_summary row>], "metric": "tokens"|"usd"}
      Inline sparkline.

# Turn 1 — session start (no user message yet, or first user prompt)

Single iteration. Call `list_experiments()`, build an `ExperimentList` DataTable (columns: slug, n_queries, exact, fuzzy, p50_ms, mtime), pin it to the canvas with `multi_use=True`, FINAL with a one-paragraph welcome.

```python
rows = list_experiments()
table_rows = [
    [r["slug"], r["n_queries"], r["exact"], r["fuzzy"], r.get("p50_ms"), r["mtime"][:10]]
    for r in rows
]
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    multi_use=True,
    payload={"type": "Column", "children": [
        {"type": "Heading", "text": f"BrowseComp · {len(rows)} experiments", "level": 2},
        {"type": "DataTable",
         "columns": ["slug", "n", "exact", "fuzzy", "p50_ms", "date"],
         "rows": table_rows,
         "rowKey": "slug"},
    ]},
)
pin_to_canvas(widget_id=result["widget_id"])
FINAL("Click a row to drill in, ask 'compare X vs Y', 'difficulty matrix of A,B,C', 'cost trend', or 'history of <query_id>'. Annotate with 'note <slug>: <text>'.")
```

# Turn 2+ — drill into one run (user clicks a row, or asks "show <slug>")

Use **two iterations**: load in iteration 1 (no FINAL), emit + narrate in iteration 2.

```python
slug = "<slug from row click data or user question>"
detail = load_run(slug=slug)
ann = read_annotations(slug=slug)
print("LOAD_OK")
```

Then, in iteration 2, build a Column with: a Heading, a Text summary line, a PerQueryTable (DataTable with rowKey="query_id"), and an AnnotationsPanel (Column with notes + tag Row). Pin it to the canvas aux slot via `pin_below_canvas`. FINAL with a one-line summary referencing real numbers.

Per-query-table columns (recommended): query_id, status, exact, fuzzy, iters, total_ms, total_tokens, termination_reason. The DataTable's row click delivers `{"rowId": <query_id>, "row": {...}}` — use that to drill into a single query.

# Drill into one query (user clicks a row in the per-query table)

```python
slug = "<slug>"
qid = "<query_id from row click>"
q = load_query(slug=slug, query_id=qid)
print("Q_OK")
```

Iteration 2: emit a Column containing a Heading (the question), a Text (gold + agent answer side-by-side), a Code block with the trajectory or trace summarised, a small DataTable of `retrieved_docids` (truncated to top 20). Pin via `pin_below_canvas`. FINAL one sentence on what happened (correct? wrong retrieval? hit max_turns?).

If the trace is long, format only the top 12 trace entries as JSON and pass `Code(content=..., language="json")`.

# Compare two runs (user asks "compare A vs B")

Iteration 1: call `compare(slug_a=A, slug_b=B)` and `load_config(slug=A)` / `load_config(slug=B)`; print "CMP_OK".

Iteration 2: emit a Column with:
  - A `ComparisonTable` DataTable: columns ["query_id", "A", "B", "winner"]; cells like ("✓"/"✗"/"·") for exact-correct on each side.
  - A `ConfigDiff` `Code` block with `diff=True`. You compute the unified diff yourself in Python — `import difflib; diff_text = "\n".join(difflib.unified_diff(a_text.splitlines(), b_text.splitlines(), fromfile=A, tofile=B, lineterm=""))`. Skip the diff entirely (just a Text node "configs identical") if both contents are equal or empty.

Pin to canvas aux. FINAL: "A leads N/30 exact; B leads M/30; agreed on K." referencing real `deltas`.

# Difficulty matrix

User asks "difficulty matrix of A, B, C, …" or "matrix of recent 5". One iteration is enough (no follow-up emit needed):

```python
slugs = [...]  # parsed from user message, or first 5 from list_experiments
m = difficulty_matrix(slugs=slugs)
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={"type": "DifficultyMatrix",
             "queries": m["queries"],
             "runs": m["runs"],
             "cells": m["cells"]},
)
pin_below_canvas(widget_id=result["widget_id"])
FINAL("Rows = queries, columns = runs (left → right in order given). Click any cell to drill into that (run, query).")
```

Cell clicks deliver `{"slug", "query_id", "cell"}` — handle them like a per-query row click.

# Query history

User asks "history of <query_id>" or clicks a recurring row across runs:

```python
qid = "<query_id>"
h = query_history(query_id=qid)
table_rows = [
    [e["slug"], "exact" if e["exact"] else ("fuzzy" if e["fuzzy"] else e["status"]),
     (e["agent_answer"] or "")[:80], e["iterations"], e.get("total_ms")]
    for e in h["entries"]
]
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={"type": "Column", "children": [
        {"type": "Heading", "text": f"Query {qid}", "level": 3},
        {"type": "Text", "text": h["question"]},
        {"type": "Text", "text": f"Gold: {h['gold_answer']}"},
        {"type": "DataTable",
         "columns": ["run", "result", "answer", "iters", "ms"],
         "rows": table_rows,
         "rowKey": "run"},
    ]},
)
pin_below_canvas(widget_id=result["widget_id"])
FINAL("…")
```

# Cost trend

```python
rows = cost_summary()
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={"type": "CostTrend", "rows": rows, "metric": "tokens"},
)
pin_below_canvas(widget_id=result["widget_id"])
FINAL("…")
```

If the user says "in dollars" or "USD", set `"metric": "usd"`. `est_usd` is null for local Ollama runs — those points get omitted from the line.

# Annotate a run

User says `note <slug>: <text>` or `tag <slug> with <tag>`:

```python
slug = "<slug>"
res = write_annotation(slug=slug, text="<text>", tags=["<tag>", ...])
# Re-emit the AnnotationsPanel for that run to confirm.
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={"type": "Column", "children": [
        {"type": "Heading", "text": f"Notes · {slug}", "level": 3},
        *[{"type": "Text", "text": f"[{n['ts'][:10]}] {n['text']}"} for n in res["notes"]],
        {"type": "Row", "children": [
            {"type": "Text", "text": "tags:"},
            *[{"type": "Text", "text": t} for t in res["tags"]],
        ]},
    ]},
)
pin_below_canvas(widget_id=result["widget_id"])
FINAL(f"Noted on {slug}.")
```

# General rules

  - Always pin large data widgets via `pin_below_canvas` (the canvas aux slot replaces on each new question). The persistent ExperimentList stays on the main canvas via `pin_to_canvas` from turn 1.
  - Chat is a record of *questions* and short narrations — never restate tables in chat.
  - Never set `scope=` on top-level answer widgets. `scope` is for drill-downs anchored to a specific (run, query) — fine to use when emitting a query detail with `scope={"kind": "query", "key": <query_id>}`.
  - When a user message references a slug, match it loosely (case-insensitive substring) against `list_experiments()` results. If ambiguous, pin a short clarification with Card per match and FINAL("which one?"); do not guess.
  - When `load_run` / `load_query` raises (slug missing), surface the error in chat with a one-line Text and ask the user to clarify.
"###;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gw_ui=debug,gw_bench_ui=debug".into()),
        )
        .init();

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
    let loop_llm: Box<dyn gw_loop::LlmClient> =
        Box::new(OllamaLlmClient::new(client).with_think(Some(false)));

    // Walk the workspace root from CARGO_MANIFEST_DIR up two levels.
    let workspace_root = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .ok_or("could not derive workspace root from CARGO_MANIFEST_DIR")?
        .to_path_buf();
    let paths = BenchPaths::from_workspace_root(&workspace_root);
    println!("workspace root: {}", workspace_root.display());
    println!("runs:           {}", paths.runs_dir.display());
    println!("gold:           {}", paths.gold_jsonl.display());
    let store = Arc::new(BenchStore::new(paths));

    // Smoke check on startup so a misconfigured runs/ surfaces fast.
    match store.list_experiments() {
        Ok(rows) => println!("found {} experiments", rows.len()),
        Err(e) => eprintln!("warning: list_experiments failed: {e}"),
    }

    let engine = GreatWheelEngine::new()
        .add_plugin(UiPlugin)
        .add_plugin(BenchPlugin::new(store.clone()))
        .init(&HashMap::new())?;
    let plugin_router = engine.host_fn_router_arc();
    let ui_store: Arc<UiSurfaceStore> = engine
        .registry
        .shared()
        .get::<Arc<UiSurfaceStore>>()
        .cloned()
        .ok_or("UiPlugin did not provide UiSurfaceStore")?;

    let adapter = Arc::new(AgUiAdapter::new(&ui_store));
    adapter.set_branding("BrowseComp", "experiment dashboard");
    adapter.set_layout("canvas-primary");
    adapter.set_welcome(
        "Past-mode analysis of BrowseComp retrieval experiments.",
        "Pick a run from the table, drill into a query, compare two runs side-by-side, or ask for the cross-run difficulty matrix and cost trend. Notes you write are saved as a sidecar JSON next to the run.",
        [
            "Show apr19-baseline-native",
            "Compare apr19-passages-4096 vs apr19-gepa-v3-native",
            "Difficulty matrix of recent 5",
            "Cost trend",
        ],
    );
    let session_id = SessionId(Uuid::new_v4());

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
        "pin_below_canvas".into(),
        "highlight_button".into(),
        "list_experiments".into(),
        "load_run".into(),
        "load_query".into(),
        "load_config".into(),
        "compare".into(),
        "gold".into(),
        "read_annotations".into(),
        "write_annotation".into(),
        "difficulty_matrix".into(),
        "query_history".into(),
        "cost_summary".into(),
    ];
    let mut repl = ReplAgent::new(external_fns, Box::new(conv_bridge));
    repl.set_variable("gw_session_id", Object::String(session_id.0.to_string()))
        .ok();

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

    let port: u16 = std::env::var("BENCH_DASHBOARD_PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_PORT);
    let app = adapter
        .router()
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());
    let bind = format!("127.0.0.1:{port}");
    let listener = TcpListener::bind(&bind).await?;
    println!("bench dashboard listening on http://{bind}");
    println!("model:      {model_label}");
    println!("session_id: {}", session_id.0);
    println!("open http://localhost:5173/?session={}", session_id.0);
    if port != 8787 {
        println!(
            "(non-default port — set VITE_API_BASE=http://127.0.0.1:{port} when running vite)"
        );
    }

    axum::serve(listener, app).await?;
    Ok(())
}
