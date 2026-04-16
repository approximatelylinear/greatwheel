//! gw-kb Plugin — registers read-only knowledge base host functions
//! that agents can call from inside ouros sessions.
//!
//! See `docs/design-kb-agent-integration.md` for the full design. This
//! module is Phase 2: the plugin itself. Phase 1 (in gw-engine/gw-loop/
//! gw-server) wired the router into ConversationBridge so the functions
//! registered here are actually reachable from an agent.
//!
//! ### Exposed functions
//!
//! All four are registered under the `kb.read` capability.
//!
//! Names use underscores (not dots) because ouros parses `kb.search(x)`
//! as `AttrCall(Name("kb"), "search", ...)` — looking up `kb` as a bare
//! identifier first — and there is no `kb` in the module namespace. A
//! dotted namespace could be built as a Python preamble shim, but
//! that's cosmetic and we YAGNI'd it.
//!
//! - `kb_search(query: str, k: int = 5) -> list[dict]`
//!   Hybrid search (BM25 + vector + topic membership) over the KB.
//!   Returns ranked chunks with source attribution.
//!
//! - `kb_explore(query: str, k: int = 15, seeds: int = 3, hops: int = 3,
//!   decay: float = 0.5) -> list[dict]`
//!   Spreading-activation discovery: picks seed topics nearest to the
//!   query, walks the typed graph, returns activated topics ranked by
//!   accumulated score.
//!
//! - `kb_topic(slug: str) -> dict | None`
//!   Fetch a topic's metadata, synthesized summary, and typed
//!   neighbor graph. Returns None if the slug doesn't exist.
//!
//! - `kb_topics(limit: int = 50) -> list[dict]`
//!   List topics ordered by chunk count, with per-topic stats.
//!
//! ### Construction
//!
//! The plugin takes a pre-built `KbStores`. `gw-server` constructs
//! the stores at startup (sharing its PgPool) and hands them to
//! `KbPlugin::new`. This keeps lifetime management at the server layer
//! and makes the plugin trivially injectable for tests.

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{Plugin, PluginContext, PluginError, PluginManifest};
use serde_json::{json, Value};
use tracing::debug;

use crate::ingest::KbStores;
use crate::linking::{
    nearest_topics_to_query, neighbors_of, spread_from_seeds, EdgeDirection, SpreadOpts,
};
use crate::search::hybrid_search;
use crate::synthesize::fetch_summary;
use crate::topics::{fetch_topic_by_slug, list_topic_summaries};

/// Capability string declared by every read-only KB host function.
pub const KB_READ_CAPABILITY: &str = "kb.read";

/// The KB plugin. Constructed by `gw-server` with a pre-built
/// `KbStores`; registered on the engine alongside other builtins.
pub struct KbPlugin {
    stores: Arc<KbStores>,
}

impl KbPlugin {
    pub fn new(stores: KbStores) -> Self {
        Self {
            stores: Arc::new(stores),
        }
    }
}

impl Plugin for KbPlugin {
    fn name(&self) -> &str {
        "kb"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "host_fn:kb_search".into(),
                "host_fn:kb_explore".into(),
                "host_fn:kb_topic".into(),
                "host_fn:kb_topics".into(),
            ],
            requires: vec![],
            priority: 50,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        // kb.search(query: str, k: int = 5) -> list[dict]
        let stores = Arc::clone(&self.stores);
        ctx.register_host_fn_async(
            "kb_search",
            Some(KB_READ_CAPABILITY),
            move |args, kwargs| {
                let stores = Arc::clone(&stores);
                async move {
                    let query = get_required_str(&args, &kwargs, 0, "query")?;
                    let k = get_optional_usize(&args, &kwargs, 1, "k").unwrap_or(5);
                    let hits = hybrid_search(&stores, &query, k)
                        .await
                        .map_err(|e| PluginError::HostFunction(format!("kb.search: {e}")))?;
                    // SearchHit derives Serialize, so this is direct.
                    serde_json::to_value(hits)
                        .map_err(|e| PluginError::HostFunction(format!("kb.search serialize: {e}")))
                }
            },
        );

        // kb.explore(query: str, k: int = 15, hops: int = 3,
        //            decay: float = 0.5, seeds: int = 3) -> list[dict]
        let stores = Arc::clone(&self.stores);
        ctx.register_host_fn_async(
            "kb_explore",
            Some(KB_READ_CAPABILITY),
            move |args, kwargs| {
                let stores = Arc::clone(&stores);
                async move {
                    let query = get_required_str(&args, &kwargs, 0, "query")?;
                    let k = get_optional_usize(&args, &kwargs, 1, "k").unwrap_or(15);
                    let seeds_k = get_optional_usize(&args, &kwargs, 2, "seeds").unwrap_or(3);
                    let hops = get_optional_usize(&args, &kwargs, 3, "hops").unwrap_or(3);
                    let decay = get_optional_f32(&args, &kwargs, 4, "decay").unwrap_or(0.5);

                    // 1. Embed query
                    let query_vec = stores
                        .embedder
                        .embed_one(&query)
                        .map_err(|e| PluginError::HostFunction(format!("kb.explore embed: {e}")))?;
                    // 2. Seed topics
                    let seed_topics = nearest_topics_to_query(&stores.pg, &query_vec, seeds_k)
                        .await
                        .map_err(|e| PluginError::HostFunction(format!("kb.explore seeds: {e}")))?;
                    if seed_topics.is_empty() {
                        return Ok(Value::Array(vec![]));
                    }
                    // 3. Spreading activation
                    let activated = spread_from_seeds(
                        &stores.pg,
                        &seed_topics,
                        SpreadOpts {
                            max_hops: hops,
                            decay,
                            limit: k,
                        },
                    )
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("kb.explore spread: {e}")))?;

                    // Shape the result as plain dicts. ActivatedTopic isn't
                    // Serialize-derived on purpose (score type), so assemble
                    // by hand for a stable wire shape.
                    let out: Vec<Value> = activated
                        .into_iter()
                        .map(|t| {
                            json!({
                                "topic_id": t.topic_id.to_string(),
                                "label": t.label,
                                "slug": t.slug,
                                "chunk_count": t.chunk_count,
                                "score": t.score,
                            })
                        })
                        .collect();
                    Ok(Value::Array(out))
                }
            },
        );

        // kb.topic(slug: str) -> dict | None
        let stores = Arc::clone(&self.stores);
        ctx.register_host_fn_async("kb_topic", Some(KB_READ_CAPABILITY), move |args, kwargs| {
            let stores = Arc::clone(&stores);
            async move {
                let slug = get_required_str(&args, &kwargs, 0, "slug")?;

                // Topic lookup is the only function that returns Null for
                // "not found" (everything else returns an empty list).
                let topic = match fetch_topic_by_slug(&stores.pg, &slug).await {
                    Ok(t) => t,
                    Err(_) => return Ok(Value::Null),
                };

                let summary_pair = fetch_summary(&stores.pg, topic.topic_id)
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("kb.topic summary: {e}")))?;
                let neigh = neighbors_of(&stores.pg, topic.topic_id, 25)
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("kb.topic neighbors: {e}")))?;

                let neighbors_json: Vec<Value> = neigh
                    .into_iter()
                    .map(|n| {
                        let dir = match n.direction {
                            EdgeDirection::OutgoingFrom => "outgoing",
                            EdgeDirection::IncomingTo => "incoming",
                            EdgeDirection::Symmetric => "symmetric",
                        };
                        json!({
                            "topic_id": n.topic_id.to_string(),
                            "label": n.label,
                            "slug": n.slug,
                            "chunk_count": n.chunk_count,
                            "confidence": n.confidence,
                            "kind": n.kind,
                            "direction": dir,
                        })
                    })
                    .collect();

                let (summary, summary_at) = match summary_pair {
                    Some((s, t)) => (Value::String(s), Value::String(t.to_rfc3339())),
                    None => (Value::Null, Value::Null),
                };

                Ok(json!({
                    "topic_id": topic.topic_id.to_string(),
                    "label": topic.label,
                    "slug": topic.slug,
                    "chunk_count": topic.chunk_count,
                    "first_seen": topic.first_seen.to_rfc3339(),
                    "last_seen": topic.last_seen.to_rfc3339(),
                    "summary": summary,
                    "summary_at": summary_at,
                    "neighbors": neighbors_json,
                }))
            }
        });

        // kb.topics(limit: int = 50) -> list[dict]
        let stores = Arc::clone(&self.stores);
        ctx.register_host_fn_async(
            "kb_topics",
            Some(KB_READ_CAPABILITY),
            move |args, kwargs| {
                let stores = Arc::clone(&stores);
                async move {
                    let limit = get_optional_i64(&args, &kwargs, 0, "limit").unwrap_or(50);
                    let rows = list_topic_summaries(&stores.pg, limit)
                        .await
                        .map_err(|e| PluginError::HostFunction(format!("kb.topics: {e}")))?;
                    let out: Vec<Value> = rows
                        .into_iter()
                        .map(|t| {
                            json!({
                                "topic_id": t.topic_id.to_string(),
                                "label": t.label,
                                "slug": t.slug,
                                "chunk_count": t.chunk_count,
                                "source_count": t.source_count,
                                "last_seen": t.last_seen.to_rfc3339(),
                            })
                        })
                        .collect();
                    Ok(Value::Array(out))
                }
            },
        );

        debug!("kb plugin registered 4 host functions");
        Ok(())
    }
}

// ─── Arg-extraction helpers ─────────────────────────────────────────────────
//
// Host functions receive (args: Vec<Value>, kwargs: HashMap<String, Value>).
// Callers can pass arguments positionally or by name; these helpers check
// positional first, then kwargs, mimicking Python's resolution order.

fn get_required_str(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    pos: usize,
    name: &str,
) -> Result<String, PluginError> {
    get_str(args, kwargs, pos, name)
        .ok_or_else(|| PluginError::HostFunction(format!("missing required arg '{name}'")))
}

fn get_str(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    pos: usize,
    name: &str,
) -> Option<String> {
    args.get(pos)
        .and_then(|v| v.as_str())
        .or_else(|| kwargs.get(name).and_then(|v| v.as_str()))
        .map(|s| s.to_string())
}

fn get_optional_usize(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    pos: usize,
    name: &str,
) -> Option<usize> {
    args.get(pos)
        .and_then(|v| v.as_u64())
        .or_else(|| kwargs.get(name).and_then(|v| v.as_u64()))
        .map(|n| n as usize)
}

fn get_optional_i64(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    pos: usize,
    name: &str,
) -> Option<i64> {
    args.get(pos)
        .and_then(|v| v.as_i64())
        .or_else(|| kwargs.get(name).and_then(|v| v.as_i64()))
}

fn get_optional_f32(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    pos: usize,
    name: &str,
) -> Option<f32> {
    args.get(pos)
        .and_then(|v| v.as_f64())
        .or_else(|| kwargs.get(name).and_then(|v| v.as_f64()))
        .map(|f| f as f32)
}
