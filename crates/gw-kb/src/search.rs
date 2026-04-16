//! Hybrid search over KB chunks: BM25 (tantivy) + vector (LanceDB),
//! fused with reciprocal rank fusion. Final results are hydrated from
//! Postgres so callers get full chunk + source context.

use gw_memory::fusion::{reciprocal_rank_fusion, ScoredKey};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;
use crate::linking::nearest_topics_to_query;

const RRF_K: usize = 60;
/// How many candidates to pull from each backend before fusion. Larger
/// pools improve recall at the cost of more Postgres hydration work.
const PER_BACKEND_K: usize = 50;

/// How many topics nearest to the query to use for the topic-membership
/// ranking signal. A small number keeps the topic boost tightly focused.
const TOPIC_SEED_K: usize = 5;

/// Per-topic chunk budget when building the topic-membership list —
/// we take at most this many highest-relevance chunks per topic so
/// one very large topic doesn't swamp the fused ranking.
const CHUNKS_PER_TOPIC: i64 = 20;

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
///
/// Three signals get RRF-fused:
///
///   1. **BM25** (tantivy) — keyword term matches.
///   2. **Vector** (LanceDB) — semantic similarity of chunk content to
///      the query embedding.
///   3. **Topic membership** — chunks that belong to topics whose
///      content vector is closest to the query embedding. This captures
///      "chunks about the same conceptual topic" even when they don't
///      share vocabulary or have high direct cosine.
///
/// The topic signal is especially useful for conceptual queries like
/// "efficient inference for large language models": a chunk that's part
/// of a topic called "Large Language Models" or "Efficiency" will get a
/// boost even if the chunk itself doesn't contain those exact words.
pub async fn hybrid_search(
    stores: &KbStores,
    query: &str,
    k: usize,
) -> Result<Vec<SearchHit>, KbError> {
    // 1. Tantivy (BM25) — sync, in-process
    let bm25_hits = stores.tantivy.search(query, PER_BACKEND_K)?;

    // 2. LanceDB (vector) — embed query via sentence-transformers
    let query_vec = stores.embedder.embed_one(query)?;
    let vector_hits = stores
        .lance
        .search(query_vec.clone(), PER_BACKEND_K)
        .await?;

    // 3. Topic membership — find chunks belonging to topics nearest to
    //    the query, ordered by topic relevance × per-topic chunk relevance.
    let topic_hits = topic_membership_hits(&stores.pg, &query_vec).await?;

    // 4. RRF fusion across all three signals
    let fused = reciprocal_rank_fusion(&[bm25_hits, vector_hits, topic_hits], RRF_K);

    // 5. Take top k chunk_ids and hydrate from Postgres
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

/// Build the "topic membership" ranking list. Finds the TOPIC_SEED_K
/// topics closest to the query, then walks their member chunks in
/// descending per-topic relevance. Topics with higher query similarity
/// contribute their chunks earlier in the list, so RRF gives those
/// chunks a stronger boost.
async fn topic_membership_hits(
    pool: &PgPool,
    query_vec: &[f32],
) -> Result<Vec<ScoredKey>, KbError> {
    let seed_topics = nearest_topics_to_query(pool, query_vec, TOPIC_SEED_K).await?;
    if seed_topics.is_empty() {
        return Ok(Vec::new());
    }

    let mut out: Vec<ScoredKey> = Vec::new();
    for (topic_id, topic_sim) in seed_topics {
        let rows: Vec<(Uuid, f32)> = sqlx::query_as(
            r#"
            SELECT chunk_id, relevance
            FROM kb_topic_chunks
            WHERE topic_id = $1
            ORDER BY relevance DESC
            LIMIT $2
            "#,
        )
        .bind(topic_id)
        .bind(CHUNKS_PER_TOPIC)
        .fetch_all(pool)
        .await?;

        for (chunk_id, relevance) in rows {
            out.push(ScoredKey {
                key: chunk_id.to_string(),
                // Score is topic_sim × chunk_relevance. RRF doesn't use the
                // actual score value (it uses rank position), but emitting
                // in descending order matters, and combining two signals
                // into the score is a cheap way to sort the contributions
                // consistently.
                score: topic_sim * relevance,
            });
        }
    }

    // Sort the full list by the combined score so chunks from closer
    // topics end up at the top of the ranking RRF sees.
    out.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
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
