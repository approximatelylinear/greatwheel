//! End-to-end agent integration test for the gw-kb plugin.
//!
//! Exercises the full Phase 1 + Phase 2 stack:
//!
//!   agent Python code
//!     → ouros Runner (via `run_agent`)
//!     → HostBridge::call (sync trait)
//!     → ConversationBridge with plugin router
//!     → HostFnRouter::dispatch
//!     → Async handler (`block_in_place` + `block_on` bridge)
//!     → gw-kb operations (hybrid_search, spread_from_seeds, etc.)
//!     → PostgreSQL + LanceDB + tantivy + embedder
//!     → JSON returned to agent
//!     → agent keeps running, calls FINAL
//!
//! This test is **gated on the presence of a live dev environment**:
//! `DATABASE_URL` must be set, the KB tables must be populated, and the
//! LanceDB / tantivy data directories must exist. It is skipped with a
//! clear message if any of those preconditions are missing so CI without
//! KB data doesn't fail.
//!
//! Run with:
//!
//! ```bash
//! DATABASE_URL=postgres://gw:gw@localhost:5432/greatwheel \
//! PYO3_PYTHON=$HOME/Code/greatwheel/crates/gw-kb/python/.venv/bin/python \
//! GW_KB_PYTHON_PATH=$HOME/Code/greatwheel/crates/gw-kb/python \
//! cargo test -p gw-kb --test agent_integration -- --nocapture
//! ```

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use gw_engine::GreatWheelEngine;
use gw_kb::embed::Embedder;
use gw_kb::index::{KbLanceStore, KbTantivyStore};
use gw_kb::ingest::KbStores;
use gw_kb::plugin::KbPlugin;
use gw_llm::OllamaClient;
use gw_runtime::{run_agent, HostBridge};
use ouros::Object;
use serde_json::Value;
use sqlx::postgres::PgPoolOptions;

const EMBEDDING_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";
const EMBEDDING_DIM: i32 = 768;

/// Workspace-root-relative path. `cargo test` runs from the crate
/// directory, so we navigate up from `CARGO_MANIFEST_DIR`.
fn workspace_path(rel: &str) -> std::path::PathBuf {
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest.parent().and_then(|p| p.parent()).unwrap_or(manifest);
    workspace.join(rel)
}

/// Check that a dev KB environment is present. Returns `Some((pg_url,
/// lance_path, tantivy_path))` if ready, `None` if the test should be
/// skipped.
fn preconditions_met() -> Option<(String, std::path::PathBuf, std::path::PathBuf)> {
    let Ok(pg_url) = std::env::var("DATABASE_URL") else {
        eprintln!("skip: DATABASE_URL not set");
        return None;
    };
    let lance = workspace_path("data/kb-lancedb");
    if !lance.exists() {
        eprintln!("skip: {} does not exist", lance.display());
        return None;
    }
    let tantivy = workspace_path("data/kb-tantivy");
    if !tantivy.exists() {
        eprintln!("skip: {} does not exist", tantivy.display());
        return None;
    }
    if std::env::var("PYO3_PYTHON").is_err() {
        eprintln!("skip: PYO3_PYTHON not set (needed for sentence-transformers)");
        return None;
    }
    if std::env::var("GW_KB_PYTHON_PATH").is_err() {
        eprintln!("skip: GW_KB_PYTHON_PATH not set");
        return None;
    }
    Some((pg_url, lance, tantivy))
}

async fn build_stores(
    pg_url: &str,
    lance_path: &Path,
    tantivy_path: &Path,
) -> KbStores {
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(pg_url)
        .await
        .expect("connect postgres");
    let lance = Arc::new(
        KbLanceStore::open(lance_path.to_str().expect("lance path utf8"), EMBEDDING_DIM)
            .await
            .expect("open lance store"),
    );
    let tantivy = Arc::new(KbTantivyStore::open(tantivy_path).expect("open tantivy"));
    let embedder = Arc::new(Embedder::new(EMBEDDING_MODEL.to_string()));
    let llm = Arc::new(OllamaClient::new(
        "http://localhost:11434".to_string(),
        "http://localhost:11434".to_string(),
        "qwen3.5:9b".to_string(),
        "unused".to_string(),
    ));
    KbStores {
        pg: pool,
        lance,
        tantivy,
        embedder,
        llm,
    }
}

/// Build an engine with just the KbPlugin and return the router it
/// produces. Isolates the test from hindsight and any other plugins.
fn engine_with_kb(
    stores: KbStores,
) -> Arc<gw_engine::HostFnRouter> {
    let engine = GreatWheelEngine::new()
        .add_plugin(KbPlugin::new(stores))
        .init(&HashMap::new())
        .expect("engine init");
    engine.host_fn_router_arc()
}

/// Minimal HostBridge that dispatches KB calls through the plugin
/// router and intercepts `FINAL(value)` to capture the agent's final
/// result. `run_agent` (unlike ReplAgent::execute) does not special-
/// case FINAL — it just forwards every call to the bridge — so the
/// interception has to live here.
struct RouterBridge {
    router: Arc<gw_engine::HostFnRouter>,
    final_value: Option<Value>,
}

impl HostBridge for RouterBridge {
    fn call(
        &mut self,
        function: &str,
        args: Vec<Value>,
        kwargs: HashMap<String, Value>,
    ) -> Result<Object, gw_runtime::AgentError> {
        if function == "FINAL" {
            // Capture the argument as the final agent value. Return it
            // to the runner so the Python expression evaluates to the
            // same value and execution continues cleanly.
            let val = args.into_iter().next().unwrap_or(Value::Null);
            self.final_value = Some(val.clone());
            return Ok(gw_runtime::json_to_object(val));
        }
        match self.router.dispatch(function, args, kwargs) {
            Some(Ok(val)) => Ok(gw_runtime::json_to_object(val)),
            Some(Err(e)) => Err(gw_runtime::AgentError::HostFunction {
                function: function.to_string(),
                message: e.to_string(),
            }),
            None => Err(gw_runtime::AgentError::UnknownFunction(function.to_string())),
        }
    }
}

/// Run a Python agent script and return its `FINAL(...)` value.
fn run(router: Arc<gw_engine::HostFnRouter>, code: &str) -> Value {
    // External function names ouros needs to know about: the plugin
    // router's function names plus FINAL for turn termination.
    let mut external: Vec<String> = router
        .function_names()
        .into_iter()
        .map(|s| s.to_string())
        .collect();
    external.push("FINAL".into());

    let mut bridge = RouterBridge {
        router,
        final_value: None,
    };
    let _result = run_agent(code, vec![], external, &mut bridge).expect("agent ran");
    bridge.final_value.unwrap_or(Value::Null)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn kb_plugin_end_to_end() {
    let Some((pg_url, lance_path, tantivy_path)) = preconditions_met() else {
        return; // Skipped.
    };

    let stores = build_stores(&pg_url, &lance_path, &tantivy_path).await;
    let router = engine_with_kb(stores);

    // Sanity: the router knows about all four kb_* functions.
    let fn_names = router.function_names();
    for expected in ["kb_search", "kb_explore", "kb_topic", "kb_topics"] {
        assert!(
            fn_names.contains(&expected),
            "router missing {}; got {:?}",
            expected,
            fn_names
        );
    }

    // ─── Test 1: kb_topics returns the top hubs ──────────────────────────
    let code = r#"
topics = kb_topics(limit=5)
FINAL(topics)
"#;
    let val = run(Arc::clone(&router), code);
    let arr = val.as_array().expect("topics is an array");
    assert!(
        !arr.is_empty(),
        "expected at least one topic from kb_topics"
    );
    let first = &arr[0];
    assert!(first.get("label").is_some(), "topic has label field");
    assert!(first.get("slug").is_some(), "topic has slug field");
    assert!(
        first.get("chunk_count").is_some(),
        "topic has chunk_count field"
    );
    println!(
        "kb_topics returned {} topics; first label = {:?}",
        arr.len(),
        first.get("label")
    );

    // ─── Test 2: kb_topic fetches one topic by slug ──────────────────────
    // Use the first topic's slug from the previous call.
    let first_slug = first
        .get("slug")
        .and_then(|v| v.as_str())
        .expect("slug is a string")
        .to_string();
    let code = format!(
        r#"
topic = kb_topic(slug={slug:?})
FINAL(topic)
"#,
        slug = first_slug
    );
    let val = run(Arc::clone(&router), &code);
    assert!(val.is_object(), "kb_topic returned an object");
    assert_eq!(
        val.get("slug").and_then(|v| v.as_str()),
        Some(first_slug.as_str()),
        "returned topic has the requested slug"
    );
    let neighbors = val.get("neighbors").and_then(|v| v.as_array());
    assert!(neighbors.is_some(), "kb_topic returned a neighbors field");
    println!(
        "kb_topic({}) has {} neighbors",
        first_slug,
        neighbors.unwrap().len()
    );

    // ─── Test 3: kb_topic for a missing slug returns null ────────────────
    let code = r#"
missing = kb_topic(slug="this-slug-does-not-exist-zzz")
FINAL(missing)
"#;
    let val = run(Arc::clone(&router), code);
    assert!(val.is_null(), "kb_topic for unknown slug returns null");

    // ─── Test 4: kb_search returns ranked hits ───────────────────────────
    let code = r#"
hits = kb_search(query="information retrieval", k=3)
FINAL(hits)
"#;
    let val = run(Arc::clone(&router), code);
    let arr = val.as_array().expect("kb_search returned an array");
    assert!(!arr.is_empty(), "expected at least one search hit");
    let first = &arr[0];
    for field in [
        "chunk_id",
        "source_id",
        "source_title",
        "content",
        "score",
    ] {
        assert!(
            first.get(field).is_some(),
            "search hit missing field: {}",
            field
        );
    }
    println!(
        "kb_search returned {} hits; top source = {:?}",
        arr.len(),
        first.get("source_title")
    );

    // ─── Test 5: kb_explore walks the graph ──────────────────────────────
    let code = r#"
activated = kb_explore(query="large language models", k=5)
FINAL(activated)
"#;
    let val = run(Arc::clone(&router), &code);
    let arr = val.as_array().expect("kb_explore returned an array");
    println!("kb_explore returned {} activated topics", arr.len());
    if !arr.is_empty() {
        let first = &arr[0];
        for field in ["topic_id", "label", "slug", "chunk_count", "score"] {
            assert!(
                first.get(field).is_some(),
                "activated topic missing field: {}",
                field
            );
        }
    }

    // ─── Test 6: exercise multiple host calls in one script ──────────────
    let code = r#"
# This mirrors a realistic agent pattern: search, then drill into a topic.
hits = kb_search(query="efficient inference", k=3)
if len(hits) == 0:
    FINAL({"error": "no hits"})
top = hits[0]
topics = kb_topics(limit=3)
FINAL({
    "top_hit_source": top["source_title"],
    "top_hit_score": top["score"],
    "topic_count": len(topics),
})
"#;
    let val = run(Arc::clone(&router), code);
    assert!(val.is_object(), "multi-call script returned an object");
    assert!(val.get("error").is_none(), "multi-call script had an error: {:?}", val);
    assert!(val.get("top_hit_source").is_some());
    assert!(val.get("topic_count").is_some());
    println!("multi-call script returned: {}", val);
}

