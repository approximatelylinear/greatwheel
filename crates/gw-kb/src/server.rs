//! HTTP server exposing the gw-kb read-only API on localhost.
//!
//! Mirrors the four `KbPlugin` host functions (search, topic, topics,
//! explore) as JSON endpoints so external processes — notably the
//! BrowseComp-Plus eval harness in `bench/browsecomp/` — can drive the
//! KB without going through the agent loop.
//!
//! All endpoints are POST with JSON request bodies (except `/healthz`).
//! There's no auth: bind to 127.0.0.1 only.
//!
//! `/search` extends the standard `SearchHit` shape with a `docid` field
//! pulled from `kb_sources.metadata->>'docid'`. BrowseComp ingests stash
//! the corpus docid there at ingest time; the eval harness needs it to
//! map hits back to gold qrels.

use std::collections::HashMap;
use std::sync::Arc;

use axum::{
    extract::State,
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sqlx::Row;
use tracing::{error, info};
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;
use crate::linking::{
    nearest_topics_to_query, neighbors_of, spread_from_seeds, EdgeDirection, SpreadOpts,
};
use crate::search::hybrid_search;
use crate::synthesize::fetch_summary;
use crate::topics::{fetch_topic_by_slug, list_topic_summaries};

#[derive(Clone)]
struct AppState {
    stores: Arc<KbStores>,
}

/// Run the gw-kb HTTP server until SIGINT.
pub async fn run(stores: Arc<KbStores>, host: &str, port: u16) -> Result<(), KbError> {
    let state = AppState { stores };

    let app = Router::new()
        .route("/healthz", get(healthz))
        .route("/search", post(search_handler))
        .route("/topic", post(topic_handler))
        .route("/topics", post(topics_handler))
        .route("/explore", post(explore_handler))
        .with_state(state);

    let addr: std::net::SocketAddr = format!("{host}:{port}")
        .parse()
        .map_err(|e| KbError::Other(format!("invalid bind addr {host}:{port}: {e}")))?;

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|e| KbError::Other(format!("bind {addr}: {e}")))?;
    info!(%addr, "gw-kb serve listening");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .map_err(|e| KbError::Other(format!("server error: {e}")))?;
    Ok(())
}

async fn shutdown_signal() {
    let _ = tokio::signal::ctrl_c().await;
    info!("shutdown signal received");
}

// ─── /healthz ───────────────────────────────────────────────────────────

async fn healthz(State(state): State<AppState>) -> impl IntoResponse {
    let topics: i64 = sqlx::query_scalar("SELECT count(*) FROM kb_topics")
        .fetch_one(&state.stores.pg)
        .await
        .unwrap_or(-1);
    let sources: i64 = sqlx::query_scalar("SELECT count(*) FROM kb_sources")
        .fetch_one(&state.stores.pg)
        .await
        .unwrap_or(-1);
    Json(json!({"ok": true, "topics": topics, "sources": sources}))
}

// ─── /search ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct SearchRequest {
    query: String,
    #[serde(default = "default_search_k")]
    k: usize,
}

fn default_search_k() -> usize {
    5
}

#[derive(Serialize)]
struct WireSearchHit {
    chunk_id: Uuid,
    source_id: Uuid,
    source_title: String,
    source_url: Option<String>,
    heading_path: Vec<String>,
    content: String,
    score: f32,
    /// `kb_sources.metadata->>'docid'` if present. BrowseComp eval needs
    /// this to map hits back to gold qrels; other corpora leave it null.
    docid: Option<String>,
}

async fn search_handler(
    State(state): State<AppState>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<Vec<WireSearchHit>>, ApiError> {
    let hits = hybrid_search(&state.stores, &req.query, req.k).await?;
    if hits.is_empty() {
        return Ok(Json(vec![]));
    }
    let source_ids: Vec<Uuid> = hits.iter().map(|h| h.source_id).collect();
    let docids = fetch_docids(&state.stores.pg, &source_ids).await?;
    let out = hits
        .into_iter()
        .map(|h| WireSearchHit {
            docid: docids.get(&h.source_id).cloned(),
            chunk_id: h.chunk_id,
            source_id: h.source_id,
            source_title: h.source_title,
            source_url: h.source_url,
            heading_path: h.heading_path,
            content: h.content,
            score: h.score,
        })
        .collect();
    Ok(Json(out))
}

/// Batch lookup of `metadata->>'docid'` for a set of sources.
async fn fetch_docids(
    pool: &sqlx::PgPool,
    source_ids: &[Uuid],
) -> Result<HashMap<Uuid, String>, KbError> {
    let rows = sqlx::query(
        "SELECT source_id, metadata->>'docid' AS docid FROM kb_sources WHERE source_id = ANY($1)",
    )
    .bind(source_ids)
    .fetch_all(pool)
    .await?;
    let mut out = HashMap::new();
    for row in rows {
        let sid: Uuid = row.get("source_id");
        let docid: Option<String> = row.get("docid");
        if let Some(d) = docid {
            out.insert(sid, d);
        }
    }
    Ok(out)
}

// ─── /topic ─────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TopicRequest {
    slug: String,
}

async fn topic_handler(
    State(state): State<AppState>,
    Json(req): Json<TopicRequest>,
) -> Result<Json<Value>, ApiError> {
    let topic = match fetch_topic_by_slug(&state.stores.pg, &req.slug).await {
        Ok(t) => t,
        Err(_) => return Ok(Json(Value::Null)),
    };
    let summary_pair = fetch_summary(&state.stores.pg, topic.topic_id).await?;
    let neigh = neighbors_of(&state.stores.pg, topic.topic_id, 25).await?;

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

    Ok(Json(json!({
        "topic_id": topic.topic_id.to_string(),
        "label": topic.label,
        "slug": topic.slug,
        "chunk_count": topic.chunk_count,
        "first_seen": topic.first_seen.to_rfc3339(),
        "last_seen": topic.last_seen.to_rfc3339(),
        "summary": summary,
        "summary_at": summary_at,
        "neighbors": neighbors_json,
    })))
}

// ─── /topics ────────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct TopicsRequest {
    #[serde(default = "default_topics_limit")]
    limit: i64,
}

fn default_topics_limit() -> i64 {
    50
}

async fn topics_handler(
    State(state): State<AppState>,
    Json(req): Json<TopicsRequest>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let rows = list_topic_summaries(&state.stores.pg, req.limit).await?;
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
    Ok(Json(out))
}

// ─── /explore ───────────────────────────────────────────────────────────

#[derive(Deserialize)]
struct ExploreRequest {
    query: String,
    #[serde(default = "default_explore_k")]
    k: usize,
    #[serde(default = "default_explore_seeds")]
    seeds: usize,
    #[serde(default = "default_explore_hops")]
    hops: usize,
    #[serde(default = "default_explore_decay")]
    decay: f32,
}

fn default_explore_k() -> usize {
    15
}
fn default_explore_seeds() -> usize {
    3
}
fn default_explore_hops() -> usize {
    3
}
fn default_explore_decay() -> f32 {
    0.5
}

async fn explore_handler(
    State(state): State<AppState>,
    Json(req): Json<ExploreRequest>,
) -> Result<Json<Vec<Value>>, ApiError> {
    let query_vec = state.stores.embedder.embed_one(&req.query)?;
    let seed_topics = nearest_topics_to_query(&state.stores.pg, &query_vec, req.seeds).await?;
    if seed_topics.is_empty() {
        return Ok(Json(vec![]));
    }
    let activated = spread_from_seeds(
        &state.stores.pg,
        &seed_topics,
        SpreadOpts {
            max_hops: req.hops,
            decay: req.decay,
            limit: req.k,
        },
    )
    .await?;
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
    Ok(Json(out))
}

// ─── Error wrapping ─────────────────────────────────────────────────────

struct ApiError(KbError);

impl From<KbError> for ApiError {
    fn from(e: KbError) -> Self {
        ApiError(e)
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        error!(error = %self.0, "request failed");
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": self.0.to_string()})),
        )
            .into_response()
    }
}
