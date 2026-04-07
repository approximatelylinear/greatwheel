//! Hybrid search over KB chunks: BM25 (tantivy) + vector (LanceDB),
//! fused with reciprocal rank fusion. Final results are hydrated from
//! Postgres so callers get full chunk + source context.

use gw_memory::fusion::reciprocal_rank_fusion;
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;

const RRF_K: usize = 60;
/// How many candidates to pull from each backend before fusion. Larger
/// pools improve recall at the cost of more Postgres hydration work.
const PER_BACKEND_K: usize = 50;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SearchHit {
    pub chunk_id: Uuid,
    pub source_id: Uuid,
    pub source_title: String,
    pub source_url: Option<String>,
    pub heading_path: Vec<String>,
    pub content: String,
    pub score: f32,
}

/// Hybrid search over the knowledge base.
pub async fn hybrid_search(
    stores: &KbStores,
    query: &str,
    k: usize,
) -> Result<Vec<SearchHit>, KbError> {
    // 1. Tantivy (BM25) — sync, in-process
    let bm25_hits = stores.tantivy.search(query, PER_BACKEND_K)?;

    // 2. LanceDB (vector) — embed query via sentence-transformers
    let query_vec = stores.embedder.embed_one(query)?;
    let vector_hits = stores.lance.search(query_vec, PER_BACKEND_K).await?;

    // 3. RRF fusion
    let fused = reciprocal_rank_fusion(&[bm25_hits, vector_hits], RRF_K);

    // 4. Take top k chunk_ids and hydrate from Postgres
    let top_ids: Vec<Uuid> = fused
        .iter()
        .take(k)
        .filter_map(|(id, _)| Uuid::parse_str(id).ok())
        .collect();

    if top_ids.is_empty() {
        return Ok(vec![]);
    }

    let rows = hydrate_chunks(&stores.pg, &top_ids).await?;

    // Re-attach scores in fused order
    let mut out = Vec::with_capacity(rows.len());
    for (id_str, score) in fused.iter().take(k) {
        let id = match Uuid::parse_str(id_str) {
            Ok(u) => u,
            Err(_) => continue,
        };
        if let Some(row) = rows.iter().find(|r| r.chunk_id == id) {
            out.push(SearchHit {
                chunk_id: row.chunk_id,
                source_id: row.source_id,
                source_title: row.source_title.clone(),
                source_url: row.source_url.clone(),
                heading_path: row.heading_path.clone(),
                content: row.content.clone(),
                score: *score,
            });
        }
    }
    Ok(out)
}

#[derive(sqlx::FromRow)]
struct HydratedRow {
    chunk_id: Uuid,
    source_id: Uuid,
    source_title: String,
    source_url: Option<String>,
    heading_path: Vec<String>,
    content: String,
}

async fn hydrate_chunks(pool: &PgPool, ids: &[Uuid]) -> Result<Vec<HydratedRow>, KbError> {
    let rows: Vec<HydratedRow> = sqlx::query_as(
        r#"
        SELECT c.chunk_id,
               c.source_id,
               s.title       AS source_title,
               s.url         AS source_url,
               c.heading_path,
               c.content
        FROM kb_chunks c
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE c.chunk_id = ANY($1)
        "#,
    )
    .bind(ids)
    .fetch_all(pool)
    .await?;
    Ok(rows)
}
