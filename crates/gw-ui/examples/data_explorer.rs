//! "Data explorer" — an agent that writes SQL, runs it against a
//! bundled SQLite bookshop dataset, and renders interactive tables.
//! Demonstrates agent-computed state: every number on screen is a
//! direct function of data the agent queried.
//!
//! Schema is created in-memory at startup from `data/bookshop_seed.sql`
//! (no external file dependency). The agent uses two host functions:
//!
//!   - describe_schema()  — returns table/column info for grounding.
//!   - run_sql(sql, ...)  — executes a SELECT statement.
//!
//! Two new widget types — `DataTable` and `QueryCard` — are added to
//! the json-render catalog on the frontend.
//!
//! Prerequisites:
//!     ollama pull qwen3.5:9b           (or set OPENAI_API_KEY)
//!     ollama serve
//!
//! Run:
//!     cargo run -p gw-ui --example data_explorer
//!
//! See `docs/design-demo-sql-explorer.md` for the full design.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use gw_core::{LoopEvent, Plugin, PluginContext, PluginError, PluginManifest, SessionId};
use gw_engine::GreatWheelEngine;
use gw_llm::OllamaClient;
use gw_loop::bridge::{new_ask_handle, ConversationBridge};
use gw_loop::{ConversationLoop, LoopConfig, OllamaLlmClient, SnapshotPolicy};
use gw_runtime::ReplAgent;
use gw_ui::{AgUiAdapter, UiPlugin, UiSurfaceStore};
use ouros::Object;
use rusqlite::{types::ValueRef, Connection};
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

const OLLAMA_MODEL: &str = "qwen3.5:9b";
const OLLAMA_URL: &str = "http://localhost:11434";
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";

const SCHEMA_SQL: &str = r#"
CREATE TABLE books (
    id INTEGER PRIMARY KEY,
    title TEXT NOT NULL,
    author TEXT NOT NULL,
    genre TEXT NOT NULL,
    price_cents INTEGER NOT NULL,
    year_published INTEGER NOT NULL
);

CREATE TABLE customers (
    id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    country TEXT NOT NULL,
    signed_up DATE NOT NULL
);

CREATE TABLE orders (
    id INTEGER PRIMARY KEY,
    customer_id INTEGER NOT NULL,
    book_id INTEGER NOT NULL,
    quantity INTEGER NOT NULL,
    ordered_at DATETIME NOT NULL,
    FOREIGN KEY (customer_id) REFERENCES customers(id),
    FOREIGN KEY (book_id) REFERENCES books(id)
);

CREATE TABLE reviews (
    id INTEGER PRIMARY KEY,
    book_id INTEGER NOT NULL,
    customer_id INTEGER NOT NULL,
    rating INTEGER NOT NULL,
    body TEXT,
    created_at DATETIME NOT NULL,
    FOREIGN KEY (book_id) REFERENCES books(id),
    FOREIGN KEY (customer_id) REFERENCES customers(id)
);
"#;

const SEED_SQL: &str = include_str!("data/bookshop_seed.sql");

const SYSTEM_PROMPT: &str = r###"You are a data analyst exploring a small bookshop database. The user asks questions in plain language; you write SQL, run it, and render the results as interactive widgets.

**Output format (CRITICAL):** Every response you produce MUST be a single fenced Python code block, i.e. starts with ```python on its own line and ends with ``` on its own line. Do NOT use OpenAI tool-calling syntax (no `to=<fn>` / JSON blobs / function_call / tool_calls). The harness only executes Python in fenced code blocks; anything else is ignored.

The database has four tables (more in `describe_schema()`):
  - books        (id, title, author, genre, price_cents, year_published)
  - customers    (id, name, country, signed_up)
  - orders       (id, customer_id, book_id, quantity, ordered_at)
  - reviews      (id, book_id, customer_id, rating, body, created_at)

`price_cents` is in cents (USD); divide by 100 for dollars. Dates are ISO strings. Today is 2026-04-25.

Data host functions:
  - describe_schema() -> {"tables": [{"name", "columns": [{"name", "type"}], "row_count"}]}
  - run_sql(sql, limit=100) -> {"columns": [str], "rows": [[any]], "truncated": bool, "row_count": int}
      Read-only. Anything that doesn't start with SELECT (or WITH followed by SELECT) is rejected.

UI host functions (keyword args, same as the Frankenstein demo):
  - emit_widget(session_id, kind, payload, multi_use=False, follow_up=False, scope=None) -> {"widget_id": "<uuid>"}
  - supersede_widget(old_widget_id, session_id, kind, payload, multi_use=False, follow_up=False, scope=None)
  - pin_to_canvas(widget_id)
  - pin_below_canvas(widget_id)
  - highlight_button(session_id, button_id)
  - FINAL("text") — terminates the turn with the given text as the assistant message.

The frontend's json-render catalog includes two SQL-specific widget types beyond the generic Column/Row/Text/Button/Card primitives:

  - DataTable: payload {"type": "DataTable", "columns": [str], "rows": [[any]], "rowKey": optional column name}.
    Each row becomes clickable; clicks emit `{"action": "select", "data": {"rowId": <value of rowKey or row index>, "row": {<col>: <value>, ...}}}`.

  - QueryCard: payload {"type": "QueryCard", "sql": str, "summary": optional str, "error": optional str}.
    Read-only — shows the SQL you wrote so the user can audit it.

You can also wrap things in Column/Row/Text/Button/Card as in any A2UI widget tree.

# Turn patterns

## Turn 1 (session start, when no widgets exist yet)

Single iteration is fine. Call describe_schema(), emit a `SchemaExplorer` widget — a Column with a Text header "Bookshop · 4 tables" and one Card per table (id "tbl-books" / "tbl-customers" / "tbl-orders" / "tbl-reviews", title=table name, subtitle=row count), pin it to the canvas with multi_use=True, and FINAL with a one-paragraph welcome ("Ask about sales, ratings, or any cross-table question…"). DO NOT include scope on the schema widget — it's persistent.

```python
schema = describe_schema()
table_cards = []
for t in schema["tables"]:
    table_cards.append({
        "type": "Card",
        "id": f"tbl-{t['name']}",
        "title": t["name"],
        "subtitle": f"{t['row_count']} rows · {len(t['columns'])} cols",
        "action": "select",
        "data": {"table": t["name"]},
    })
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    multi_use=True,
    payload={"type": "Column", "children": [
        {"type": "Text", "text": "Bookshop · 4 tables"},
        {"type": "Column", "children": table_cards},
    ]},
)
pin_to_canvas(widget_id=result["widget_id"])
FINAL("Ask me about sales, customers, ratings, or any cross-table question. I'll write the SQL, run it, and explain what I find.")
```

## Turn 2+ (user asked a question in chat)

Use **two iterations**. Iteration 1 writes and runs the SQL but does not call FINAL. Iteration 2 reads the result and emits widgets + FINAL. This is required because `run_sql`'s return value is NOT in your context until the next iteration.

**Iteration 1 — write SQL, run it, print:**

```python
sql = """
SELECT b.genre, ROUND(SUM(o.quantity * b.price_cents) / 100.0, 2) AS revenue_usd
FROM orders o JOIN books b ON b.id = o.book_id
GROUP BY b.genre
ORDER BY revenue_usd DESC
"""
result = run_sql(sql, limit=20)
print("SQL_OK")  # so iteration 2 knows to proceed
```

(The harness will surface stdout to iteration 2's context as part of the agent's recent history. Do not print large result blobs.)

**Iteration 2 — emit DataTable + QueryCard, narrate:**

```python
sql = """
SELECT b.genre, ROUND(SUM(o.quantity * b.price_cents) / 100.0, 2) AS revenue_usd
FROM orders o JOIN books b ON b.id = o.book_id
GROUP BY b.genre
ORDER BY revenue_usd DESC
"""
data = run_sql(sql, limit=20)

# 1. Emit the SQL transparently.
emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={"type": "QueryCard", "sql": sql.strip(),
             "summary": "Revenue by genre, top to bottom"},
)

# 2. Emit the result table. Use a meaningful rowKey if a column makes
#    sense as a record id; otherwise omit it.
emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    payload={
        "type": "DataTable",
        "columns": data["columns"],
        "rows": data["rows"],
        "rowKey": "genre",
    },
)

# 3. Emit 2-3 follow-up question buttons (follow_up=True) — short,
#    contextually-relevant drill-downs.
emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    follow_up=True,
    payload={"type": "Column", "children": [
        {"type": "Text", "text": "Drill in:"},
        {"type": "Row", "children": [
            {"type": "Button", "id": "fup-1", "label": "Show monthly trend",
             "action": "submit", "data": {"ask": "Plot monthly revenue across all genres for the past year."}},
            {"type": "Button", "id": "fup-2", "label": "Top 5 books in Fiction",
             "action": "submit", "data": {"ask": "What are the top 5 books in Fiction by revenue?"}},
            {"type": "Button", "id": "fup-3", "label": "Compare to ratings",
             "action": "submit", "data": {"ask": "Average rating by genre — does revenue track quality?"}},
        ]},
    ]},
)

FINAL("Fiction leads at $X, then Science at $Y — driven mostly by Compaction's January launch. The long tail (Travel, Memoir) sits well below.")
```

## Row click drill-down

When the user clicks a DataTable row, you get a WidgetInteraction with `data = {"rowId": <key>, "row": {...}}`. Treat this as a request to drill into that record. Use **two iterations** if you need fresh data; otherwise one is fine.

If the row has a natural section/scope (e.g., a book id), pass `scope={"kind": "book", "key": <id>}` on emitted drill-down widgets. Then put them in `pin_below_canvas` so they appear in the aux canvas slot. They'll auto-hide when the user navigates to a different book.

## Filter Card click

When the user clicks a Card representing a filter (e.g., a genre bucket from a previous query), re-run the original query with that filter applied. Supersede the previous DataTable rather than emitting a new one. Use `supersede_widget` and pass the previous widget_id (track it in your iteration context).

# General rules

  - Always call describe_schema() at session start, never repeatedly within a turn.
  - SQL must be a single SELECT (or WITH ... SELECT). No INSERT/UPDATE/DELETE/DDL.
  - Use parameterised values directly in the SQL when literal — there is no parameter binding from Python.
  - Cap LIMIT at 100 unless the user explicitly asks for more.
  - When a query returns 0 rows, say so plainly. Don't fabricate.
  - When a query errors, the error appears in run_sql's return as `{"error": "<msg>", "columns": [], "rows": []}`. Read it, fix your SQL, try again on the same iteration if the fix is obvious; otherwise emit a QueryCard with the error and ask the user for clarification.
"###;

// ── SQL plugin ──────────────────────────────────────────────────────

struct SqlPlugin {
    conn: Arc<Mutex<Connection>>,
}

impl SqlPlugin {
    fn open() -> Result<Self, PluginError> {
        let conn = Connection::open_in_memory()
            .map_err(|e| PluginError::HostFunction(format!("open sqlite: {e}")))?;
        conn.execute_batch(SCHEMA_SQL)
            .map_err(|e| PluginError::HostFunction(format!("schema: {e}")))?;
        conn.execute_batch(SEED_SQL)
            .map_err(|e| PluginError::HostFunction(format!("seed: {e}")))?;
        Ok(Self {
            conn: Arc::new(Mutex::new(conn)),
        })
    }
}

impl Plugin for SqlPlugin {
    fn name(&self) -> &str {
        "sql"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "sql".into(),
                "host_fn:sql.describe_schema".into(),
                "host_fn:sql.run_sql".into(),
            ],
            requires: vec![],
            priority: 0,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let conn = self.conn.clone();
        ctx.register_host_fn_sync("describe_schema", None, move |_args, _kwargs| {
            let conn = conn
                .lock()
                .map_err(|_| PluginError::HostFunction("sqlite mutex poisoned".into()))?;
            describe_schema(&conn)
        });

        let conn = self.conn.clone();
        ctx.register_host_fn_sync("run_sql", None, move |args, kwargs| {
            let sql = args
                .first()
                .and_then(|v| v.as_str())
                .or_else(|| kwargs.get("sql").and_then(|v| v.as_str()))
                .ok_or_else(|| PluginError::HostFunction("sql required (string)".into()))?
                .trim()
                .to_string();
            let limit = kwargs
                .get("limit")
                .and_then(|v| v.as_u64())
                .unwrap_or(100)
                .min(500) as usize;
            let conn = conn
                .lock()
                .map_err(|_| PluginError::HostFunction("sqlite mutex poisoned".into()))?;
            Ok(run_sql(&conn, &sql, limit))
        });

        Ok(())
    }
}

fn describe_schema(conn: &Connection) -> Result<Value, PluginError> {
    let mut tables_q = conn
        .prepare("SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name")
        .map_err(|e| PluginError::HostFunction(format!("schema list: {e}")))?;
    let table_names: Vec<String> = tables_q
        .query_map([], |row| row.get::<_, String>(0))
        .map_err(|e| PluginError::HostFunction(format!("schema rows: {e}")))?
        .filter_map(Result::ok)
        .collect();

    let mut tables = Vec::new();
    for name in &table_names {
        let mut col_q = conn
            .prepare(&format!("PRAGMA table_info(\"{name}\")"))
            .map_err(|e| PluginError::HostFunction(format!("table_info {name}: {e}")))?;
        let columns: Vec<Value> = col_q
            .query_map([], |row| {
                Ok(json!({
                    "name": row.get::<_, String>(1)?,
                    "type": row.get::<_, String>(2)?,
                }))
            })
            .map_err(|e| PluginError::HostFunction(format!("col rows {name}: {e}")))?
            .filter_map(Result::ok)
            .collect();

        let row_count: i64 = conn
            .query_row(&format!("SELECT COUNT(*) FROM \"{name}\""), [], |row| {
                row.get(0)
            })
            .unwrap_or(0);

        tables.push(json!({
            "name": name,
            "columns": columns,
            "row_count": row_count,
        }));
    }
    Ok(json!({ "tables": tables }))
}

fn run_sql(conn: &Connection, sql: &str, limit: usize) -> Value {
    // Read-only guard: only SELECT or WITH ... SELECT.
    let head = sql
        .split_whitespace()
        .next()
        .unwrap_or("")
        .to_uppercase();
    if head != "SELECT" && head != "WITH" {
        return json!({
            "columns": [],
            "rows": [],
            "truncated": false,
            "row_count": 0,
            "error": format!("only SELECT / WITH queries are allowed, got '{}'", head),
        });
    }

    let mut stmt = match conn.prepare(sql) {
        Ok(s) => s,
        Err(e) => {
            return json!({
                "columns": [],
                "rows": [],
                "truncated": false,
                "row_count": 0,
                "error": e.to_string(),
            })
        }
    };
    let columns: Vec<String> = stmt.column_names().into_iter().map(String::from).collect();
    let col_count = columns.len();

    let mut rows_q = match stmt.query([]) {
        Ok(r) => r,
        Err(e) => {
            return json!({
                "columns": columns,
                "rows": [],
                "truncated": false,
                "row_count": 0,
                "error": e.to_string(),
            })
        }
    };

    let mut rows = Vec::new();
    let mut total = 0usize;
    let mut truncated = false;
    while let Ok(Some(row)) = rows_q.next() {
        total += 1;
        if rows.len() >= limit {
            truncated = true;
            continue;
        }
        let mut out_row = Vec::with_capacity(col_count);
        for i in 0..col_count {
            let v = row.get_ref(i).ok();
            out_row.push(value_ref_to_json(v));
        }
        rows.push(Value::Array(out_row));
    }

    json!({
        "columns": columns,
        "rows": rows,
        "truncated": truncated,
        "row_count": total,
    })
}

fn value_ref_to_json(v: Option<ValueRef<'_>>) -> Value {
    match v {
        None | Some(ValueRef::Null) => Value::Null,
        Some(ValueRef::Integer(i)) => Value::Number(i.into()),
        Some(ValueRef::Real(f)) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        Some(ValueRef::Text(t)) => match std::str::from_utf8(t) {
            Ok(s) => Value::String(s.into()),
            Err(_) => Value::Null,
        },
        Some(ValueRef::Blob(_)) => Value::String("<blob>".into()),
    }
}

// ── Server ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gw_ui=debug".into()),
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

    let sql = SqlPlugin::open()?;

    let engine = GreatWheelEngine::new()
        .add_plugin(UiPlugin)
        .add_plugin(sql)
        .init(&HashMap::new())?;
    let plugin_router = engine.host_fn_router_arc();
    let store: Arc<UiSurfaceStore> = engine
        .registry
        .shared()
        .get::<Arc<UiSurfaceStore>>()
        .cloned()
        .ok_or("UiPlugin did not provide UiSurfaceStore")?;

    let adapter = Arc::new(AgUiAdapter::new(&store));
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
        "describe_schema".into(),
        "run_sql".into(),
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

    let app = adapter
        .router()
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());
    let listener = TcpListener::bind("127.0.0.1:8787").await?;
    println!("data explorer listening on http://127.0.0.1:8787");
    println!("model: {model_label}");
    println!("session_id: {}", session_id.0);
    println!("open http://localhost:5173/?session={}", session_id.0);

    axum::serve(listener, app).await?;
    Ok(())
}
