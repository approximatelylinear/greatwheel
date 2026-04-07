//! Phase 2 part 2: build the typed topic graph.
//!
//! Two signal sources merged into a single confidence score per edge:
//!
//!   1. **Co-occurrence** — pairs of topics sharing ≥ `min_shared_chunks`
//!      members. Confidence = Jaccard similarity (intersection / union).
//!      Cheap, no embeddings, no LLM. Free baseline of "these topics
//!      tend to appear together".
//!
//!   2. **Embedding proximity** — pairs whose content-aware `vector`
//!      cosine is ≥ `min_cosine`. Captures semantic similarity even
//!      between topics that never co-occur in any chunk.
//!
//! Edges are symmetric `related` for now (no LLM classifier yet) and
//! stored once per pair with the lower topic_id as `from_topic_id`.
//! When both signals fire for a pair, the higher confidence wins.
//!
//! `link()` is a full rebuild: it truncates `kb_topic_links` and
//! recomputes from current topic state. At our scale (hundreds of topics)
//! this is cheap and avoids any incremental-update bookkeeping.

use std::collections::HashMap;

use sqlx::PgPool;
use tracing::{debug, info};
use uuid::Uuid;

use crate::error::KbError;
use crate::topics::{bytes_to_vec, cosine, load_all_topic_states};

#[derive(Debug, Clone, Copy)]
pub struct LinkOpts {
    /// Co-occurrence: minimum shared chunks for a pair to be considered.
    pub min_shared_chunks: i64,
    /// Embedding: minimum cosine on the content `vector` to emit an edge.
    pub min_cosine: f32,
    /// Drop any edge whose final confidence is below this floor.
    pub min_confidence: f32,
}

impl Default for LinkOpts {
    fn default() -> Self {
        Self {
            min_shared_chunks: 2,
            min_cosine: 0.65,
            min_confidence: 0.20,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LinkReport {
    pub topics_seen: usize,
    pub cooccurrence_pairs: usize,
    pub embedding_pairs: usize,
    pub edges_written: usize,
}

/// Rebuild the topic link graph from scratch.
pub async fn link(pool: &PgPool, opts: LinkOpts) -> Result<LinkReport, KbError> {
    let mut report = LinkReport::default();

    // 1. Wipe existing edges (full rebuild)
    sqlx::query("TRUNCATE kb_topic_links").execute(pool).await?;

    // 2. Load topic state
    let topics = load_all_topic_states(pool).await?;
    report.topics_seen = topics.len();
    info!(topics = topics.len(), "loaded topics for linking");
    if topics.len() < 2 {
        return Ok(report);
    }

    // For Jaccard we need each topic's total chunk count. The TopicState
    // already carries it.
    let chunk_counts: HashMap<Uuid, usize> =
        topics.iter().map(|t| (t.topic_id, t.chunk_count)).collect();

    // Edge map: keep the highest confidence per (lo, hi) pair.
    let mut edges: HashMap<(Uuid, Uuid), f32> = HashMap::new();

    // 3. Co-occurrence pass — single SQL query, no per-pair work in Rust.
    let cooc_rows: Vec<(Uuid, Uuid, i64)> = sqlx::query_as(
        r#"
        SELECT a.topic_id AS from_id,
               b.topic_id AS to_id,
               count(*)   AS shared
        FROM kb_topic_chunks a
        JOIN kb_topic_chunks b
          ON a.chunk_id = b.chunk_id
         AND a.topic_id < b.topic_id
        GROUP BY a.topic_id, b.topic_id
        HAVING count(*) >= $1
        "#,
    )
    .bind(opts.min_shared_chunks)
    .fetch_all(pool)
    .await?;
    report.cooccurrence_pairs = cooc_rows.len();
    info!(pairs = report.cooccurrence_pairs, "co-occurrence candidates");

    for (from_id, to_id, shared) in cooc_rows {
        let count_a = *chunk_counts.get(&from_id).unwrap_or(&0);
        let count_b = *chunk_counts.get(&to_id).unwrap_or(&0);
        let union = count_a + count_b - (shared as usize);
        let jaccard = if union == 0 {
            0.0
        } else {
            (shared as f32) / (union as f32)
        };
        let key = ordered_pair(from_id, to_id);
        edges
            .entry(key)
            .and_modify(|c| {
                if jaccard > *c {
                    *c = jaccard;
                }
            })
            .or_insert(jaccard);
    }

    // 4. Embedding pass — all-pairs cosine on the content `vector`.
    //    O(n²) but n ≈ low hundreds, dim 768 → trivial.
    let mut embed_count = 0usize;
    for i in 0..topics.len() {
        for j in (i + 1)..topics.len() {
            let sim = cosine(&topics[i].vector, &topics[j].vector);
            if sim < opts.min_cosine {
                continue;
            }
            embed_count += 1;
            let key = ordered_pair(topics[i].topic_id, topics[j].topic_id);
            edges
                .entry(key)
                .and_modify(|c| {
                    if sim > *c {
                        *c = sim;
                    }
                })
                .or_insert(sim);
        }
    }
    report.embedding_pairs = embed_count;
    info!(pairs = embed_count, "embedding candidates");

    // 5. Insert all edges that pass the confidence floor.
    let mut written = 0usize;
    for ((from_id, to_id), confidence) in &edges {
        if *confidence < opts.min_confidence {
            continue;
        }
        sqlx::query(
            r#"
            INSERT INTO kb_topic_links (from_topic_id, to_topic_id, kind, confidence)
            VALUES ($1, $2, 'related', $3)
            ON CONFLICT (from_topic_id, to_topic_id) DO UPDATE
              SET confidence = EXCLUDED.confidence
            "#,
        )
        .bind(from_id)
        .bind(to_id)
        .bind(*confidence)
        .execute(pool)
        .await?;
        written += 1;
    }
    report.edges_written = written;

    info!(
        edges = written,
        cooccurrence = report.cooccurrence_pairs,
        embedding = report.embedding_pairs,
        "link rebuild complete"
    );
    Ok(report)
}

/// Order a pair `(a, b)` so the smaller UUID is first. Used as a HashMap
/// key so each undirected pair has exactly one entry.
fn ordered_pair(a: Uuid, b: Uuid) -> (Uuid, Uuid) {
    if a < b {
        (a, b)
    } else {
        (b, a)
    }
}

// ---------- Spreading activation traversal ----------

/// Result of spreading activation: a topic ID with its accumulated score.
#[derive(Debug, Clone)]
pub struct ActivatedTopic {
    pub topic_id: Uuid,
    pub label: String,
    pub slug: String,
    pub chunk_count: i32,
    pub score: f32,
}

#[derive(Debug, Clone, Copy)]
pub struct SpreadOpts {
    pub max_hops: usize,
    pub decay: f32,
    pub limit: usize,
}

impl Default for SpreadOpts {
    fn default() -> Self {
        Self {
            max_hops: 3,
            decay: 0.5,
            limit: 20,
        }
    }
}

/// Spreading activation from `seeds` over the symmetric topic link graph.
///
/// We treat each row of `kb_topic_links` as bidirectional (the storage
/// convention is "lower UUID first" but the graph is undirected for now).
/// For each visited topic, we keep the highest accumulated score across
/// all paths from any seed.
pub async fn spread_from_seeds(
    pool: &PgPool,
    seeds: &[(Uuid, f32)],
    opts: SpreadOpts,
) -> Result<Vec<ActivatedTopic>, KbError> {
    if seeds.is_empty() || opts.max_hops == 0 {
        return Ok(Vec::new());
    }

    // Recursive CTE walks the graph in both directions. We compute a
    // "best score" per topic (max across all paths) and exclude the
    // seeds themselves from the results.
    let seed_ids: Vec<Uuid> = seeds.iter().map(|(id, _)| *id).collect();
    let seed_scores: Vec<f32> = seeds.iter().map(|(_, s)| *s).collect();

    let rows: Vec<(Uuid, String, String, i32, f32)> = sqlx::query_as(
        r#"
        WITH RECURSIVE
        all_edges AS (
            SELECT from_topic_id AS a, to_topic_id AS b, confidence
            FROM kb_topic_links
            UNION ALL
            SELECT to_topic_id, from_topic_id, confidence
            FROM kb_topic_links
        ),
        seeds(topic_id, score) AS (
            SELECT t.id, t.score
            FROM unnest($1::uuid[], $2::real[]) AS t(id, score)
        ),
        spread AS (
            SELECT e.b AS topic_id,
                   s.score * e.confidence * $3::real AS score,
                   1 AS hop
            FROM seeds s
            JOIN all_edges e ON e.a = s.topic_id
            UNION ALL
            SELECT e.b,
                   sp.score * e.confidence * $3::real,
                   sp.hop + 1
            FROM spread sp
            JOIN all_edges e ON e.a = sp.topic_id
            WHERE sp.hop < $4
        ),
        ranked AS (
            SELECT topic_id, max(score) AS score
            FROM spread
            WHERE topic_id <> ALL($1::uuid[])
            GROUP BY topic_id
        )
        SELECT t.topic_id, t.label, t.slug, t.chunk_count, r.score
        FROM ranked r
        JOIN kb_topics t USING (topic_id)
        ORDER BY r.score DESC
        LIMIT $5
        "#,
    )
    .bind(&seed_ids)
    .bind(&seed_scores)
    .bind(opts.decay)
    .bind(opts.max_hops as i32)
    .bind(opts.limit as i64)
    .fetch_all(pool)
    .await?;

    debug!(rows = rows.len(), "spreading activation returned");

    Ok(rows
        .into_iter()
        .map(|(id, label, slug, cc, score)| ActivatedTopic {
            topic_id: id,
            label,
            slug,
            chunk_count: cc,
            score,
        })
        .collect())
}

/// Direct neighbors of one topic, ordered by edge confidence.
pub async fn neighbors_of(
    pool: &PgPool,
    topic_id: Uuid,
    limit: i64,
) -> Result<Vec<ActivatedTopic>, KbError> {
    let rows: Vec<(Uuid, String, String, i32, f32)> = sqlx::query_as(
        r#"
        WITH n AS (
            SELECT to_topic_id AS topic_id, confidence
            FROM kb_topic_links WHERE from_topic_id = $1
            UNION ALL
            SELECT from_topic_id, confidence
            FROM kb_topic_links WHERE to_topic_id = $1
        )
        SELECT t.topic_id, t.label, t.slug, t.chunk_count, n.confidence
        FROM n JOIN kb_topics t USING (topic_id)
        ORDER BY n.confidence DESC
        LIMIT $2
        "#,
    )
    .bind(topic_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, label, slug, cc, score)| ActivatedTopic {
            topic_id: id,
            label,
            slug,
            chunk_count: cc,
            score,
        })
        .collect())
}

/// Find the top-k topics nearest to a query embedding via cosine on the
/// content `vector`. Used as the seed-selection step for `gw-kb explore`.
pub async fn nearest_topics_to_query(
    pool: &PgPool,
    query_vec: &[f32],
    k: usize,
) -> Result<Vec<(Uuid, f32)>, KbError> {
    // Linear scan over all topics — fine at hundreds of topics. If this
    // grows we can promote topic vectors into LanceDB for ANN.
    let rows: Vec<(Uuid, Option<Vec<u8>>)> =
        sqlx::query_as("SELECT topic_id, vector FROM kb_topics")
            .fetch_all(pool)
            .await?;

    let mut scored: Vec<(Uuid, f32)> = rows
        .into_iter()
        .filter_map(|(id, vec_bytes)| {
            vec_bytes.map(|b| {
                let v = bytes_to_vec(&b);
                (id, cosine(query_vec, &v))
            })
        })
        .collect();
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}
