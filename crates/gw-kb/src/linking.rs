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
    /// Optional per-topic fan-out cap. After candidates are collected,
    /// each topic keeps at most its top-K strongest edges (by
    /// confidence). The final edge set is the union across topics, so
    /// the total can exceed `topics × K / 2` only when both endpoints
    /// of an edge keep it. Bounds blow-up on dense corpora where
    /// every topic correlates weakly with every other.
    pub max_per_topic: Option<usize>,
}

impl Default for LinkOpts {
    fn default() -> Self {
        Self {
            min_shared_chunks: 2,
            min_cosine: 0.65,
            min_confidence: 0.20,
            max_per_topic: None,
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
    info!(
        pairs = report.cooccurrence_pairs,
        "co-occurrence candidates"
    );

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

    // 5. Optional per-topic fan-out cap. For each topic keep its top-K
    //    edges by confidence; an edge survives if at least one endpoint
    //    keeps it. Bounds runaway edge counts on dense corpora.
    if let Some(k) = opts.max_per_topic {
        #[allow(clippy::type_complexity)]
        let mut by_topic: HashMap<Uuid, Vec<((Uuid, Uuid), f32)>> = HashMap::new();
        for (&(a, b), &conf) in edges.iter() {
            if conf < opts.min_confidence {
                continue;
            }
            by_topic.entry(a).or_default().push(((a, b), conf));
            by_topic.entry(b).or_default().push(((a, b), conf));
        }
        let mut keep: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
        for list in by_topic.values_mut() {
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (key, _) in list.iter().take(k) {
                keep.insert(*key);
            }
        }
        edges.retain(|key, _| keep.contains(key));
        info!(
            kept = edges.len(),
            max_per_topic = k,
            "applied per-topic fan-out cap"
        );
    }

    // 6. Insert all edges that pass the confidence floor.
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

// ---------- Entity link graph (Phase B step 4) ----------
//
// Mirror of the topic linker against kb_entities + kb_chunk_entity_links.
// Two signal sources merged into a single confidence per pair:
//
//   1. Co-occurrence: Jaccard over chunk-membership sets. Cheap; one
//      SQL self-join.
//   2. Embedding proximity: cosine over kb_entities.vector. O(n²) in
//      the entity count but n is small (low thousands at our scale).
//
// Symmetric edges only — kb_entity_links has CHECK (a < b). Edge kind
// is free-form `relation TEXT` (defaults to "related"); a future LLM
// pass could classify e.g. "co_author" / "uses_dataset" / "extends".

#[derive(Debug, Clone, Copy)]
pub struct LinkEntitiesOpts {
    /// Co-occurrence: minimum shared chunks for a pair to be considered.
    /// Default 1 — a single co-mention is meaningful for entities like
    /// authors, where many real edges have only one shared paper.
    pub min_shared_chunks: i64,
    /// Embedding: minimum cosine on `kb_entities.vector`.
    pub min_cosine: f32,
    pub min_confidence: f32,
    pub max_per_entity: Option<usize>,
}

impl Default for LinkEntitiesOpts {
    fn default() -> Self {
        Self {
            min_shared_chunks: 1,
            min_cosine: 0.70,
            min_confidence: 0.10,
            max_per_entity: Some(20),
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LinkEntitiesReport {
    pub entities_seen: usize,
    pub cooccurrence_pairs: usize,
    pub embedding_pairs: usize,
    pub edges_written: usize,
}

/// Rebuild the entity link graph from scratch. Truncates
/// `kb_entity_links` and re-derives every edge from current state.
pub async fn link_entities(
    pool: &PgPool,
    opts: LinkEntitiesOpts,
) -> Result<LinkEntitiesReport, KbError> {
    use crate::topics::bytes_to_vec;

    let mut report = LinkEntitiesReport::default();
    sqlx::query("TRUNCATE kb_entity_links")
        .execute(pool)
        .await?;

    // 1. Load entity vectors + per-entity chunk counts in one go.
    //    `mentions` is maintained by `canonicalize_and_persist` as
    //    "distinct chunks mentioning this entity" — exactly what we
    //    need for Jaccard's |A| / |B|.
    let rows: Vec<(Uuid, i32, Option<Vec<u8>>)> =
        sqlx::query_as("SELECT entity_id, mentions, vector FROM kb_entities")
            .fetch_all(pool)
            .await?;
    report.entities_seen = rows.len();
    info!(entities = rows.len(), "loaded entities for linking");
    if rows.len() < 2 {
        return Ok(report);
    }
    let entity_ids: Vec<Uuid> = rows.iter().map(|r| r.0).collect();
    let chunk_counts: HashMap<Uuid, i32> = rows.iter().map(|r| (r.0, r.1)).collect();
    let vectors: HashMap<Uuid, Vec<f32>> = rows
        .iter()
        .filter_map(|r| {
            r.2.as_ref()
                .map(|b| (r.0, bytes_to_vec(b)))
                .filter(|(_, v)| !v.is_empty())
        })
        .collect();

    let mut edges: HashMap<(Uuid, Uuid), f32> = HashMap::new();

    // 2. Co-occurrence — single SQL query, no per-pair work in Rust.
    //    `entity_id_a < entity_id_b` keeps each undirected pair once.
    let cooc_rows: Vec<(Uuid, Uuid, i64)> = sqlx::query_as(
        r#"
        SELECT a.entity_id AS id_a,
               b.entity_id AS id_b,
               count(*)    AS shared
        FROM kb_chunk_entity_links a
        JOIN kb_chunk_entity_links b
          ON a.chunk_id = b.chunk_id
         AND a.entity_id < b.entity_id
        GROUP BY a.entity_id, b.entity_id
        HAVING count(*) >= $1
        "#,
    )
    .bind(opts.min_shared_chunks)
    .fetch_all(pool)
    .await?;
    report.cooccurrence_pairs = cooc_rows.len();
    info!(pairs = report.cooccurrence_pairs, "entity co-occurrence");

    for (a, b, shared) in cooc_rows {
        let ca = *chunk_counts.get(&a).unwrap_or(&0) as i64;
        let cb = *chunk_counts.get(&b).unwrap_or(&0) as i64;
        let union = (ca + cb - shared).max(1);
        let jaccard = (shared as f32) / (union as f32);
        let key = (a, b);
        edges
            .entry(key)
            .and_modify(|c| {
                if jaccard > *c {
                    *c = jaccard;
                }
            })
            .or_insert(jaccard);
    }

    // 3. Embedding pass — all-pairs cosine on kb_entities.vector.
    let mut embed_count = 0usize;
    for i in 0..entity_ids.len() {
        let Some(vi) = vectors.get(&entity_ids[i]) else {
            continue;
        };
        for j in (i + 1)..entity_ids.len() {
            let Some(vj) = vectors.get(&entity_ids[j]) else {
                continue;
            };
            if vi.len() != vj.len() {
                continue;
            }
            let sim = cosine(vi, vj);
            if sim < opts.min_cosine {
                continue;
            }
            embed_count += 1;
            let key = ordered_pair(entity_ids[i], entity_ids[j]);
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
    info!(pairs = embed_count, "entity embedding candidates");

    // 4. Optional per-entity fan-out cap. Keeps hub entities (e.g. a
    //    prolific author) from each owning hundreds of edges — the
    //    UI gets unreadable past ~25 neighbours per node.
    if let Some(k) = opts.max_per_entity {
        #[allow(clippy::type_complexity)]
        let mut by_entity: HashMap<Uuid, Vec<((Uuid, Uuid), f32)>> = HashMap::new();
        for (&(a, b), &conf) in edges.iter() {
            if conf < opts.min_confidence {
                continue;
            }
            by_entity.entry(a).or_default().push(((a, b), conf));
            by_entity.entry(b).or_default().push(((a, b), conf));
        }
        let mut keep: std::collections::HashSet<(Uuid, Uuid)> = std::collections::HashSet::new();
        for list in by_entity.values_mut() {
            list.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            for (key, _) in list.iter().take(k) {
                keep.insert(*key);
            }
        }
        edges.retain(|key, _| keep.contains(key));
        info!(kept = edges.len(), max = k, "applied entity fan-out cap");
    }

    // 5. Insert edges. Default `relation = 'related'` — a future LLM
    //    classifier can promote specific edges to richer relations.
    let mut written = 0usize;
    for ((a, b), confidence) in &edges {
        if *confidence < opts.min_confidence {
            continue;
        }
        sqlx::query(
            r#"
            INSERT INTO kb_entity_links (entity_id_a, entity_id_b, relation, confidence)
            VALUES ($1, $2, 'related', $3)
            ON CONFLICT (entity_id_a, entity_id_b) DO UPDATE
              SET confidence = EXCLUDED.confidence
            "#,
        )
        .bind(a)
        .bind(b)
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
        "entity link rebuild complete"
    );
    Ok(report)
}

// ---------- Topic ↔ entity bridge (Phase B step 4) ----------
//
// Materialises `kb_topic_entity_links` from chunk-level co-occurrence
// (any chunk that's a member of topic T and mentions entity E
// contributes evidence). Confidence = Jaccard over the chunk sets.
// No embedding pass: topic vectors live in label/content space and
// entity vectors in name/canonical-form space — direct cosine across
// the two doesn't have a meaningful interpretation (see
// design-kb-entities.md §7 "Cross-kind embeddings"). When/if we
// build a unified embedding space, we can revisit.

#[derive(Debug, Clone, Copy)]
pub struct LinkTopicEntitiesOpts {
    pub min_shared_chunks: i64,
    pub min_confidence: f32,
}

impl Default for LinkTopicEntitiesOpts {
    fn default() -> Self {
        Self {
            min_shared_chunks: 1,
            min_confidence: 0.10,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct LinkTopicEntitiesReport {
    pub pairs_considered: usize,
    pub edges_written: usize,
}

/// Rebuild `kb_topic_entity_links` from current chunk membership.
pub async fn link_topic_entities(
    pool: &PgPool,
    opts: LinkTopicEntitiesOpts,
) -> Result<LinkTopicEntitiesReport, KbError> {
    let mut report = LinkTopicEntitiesReport::default();
    sqlx::query("TRUNCATE kb_topic_entity_links")
        .execute(pool)
        .await?;

    // |A| (topic chunk counts) and |B| (entity chunk counts), one
    // hashmap each. Both columns already maintained on insert/update.
    let topic_counts: HashMap<Uuid, i32> =
        sqlx::query_as::<_, (Uuid, i32)>("SELECT topic_id, chunk_count FROM kb_topics")
            .fetch_all(pool)
            .await?
            .into_iter()
            .collect();
    let entity_counts: HashMap<Uuid, i32> =
        sqlx::query_as::<_, (Uuid, i32)>("SELECT entity_id, mentions FROM kb_entities")
            .fetch_all(pool)
            .await?
            .into_iter()
            .collect();

    // Pairwise shared-chunk counts.
    let pair_rows: Vec<(Uuid, Uuid, i64)> = sqlx::query_as(
        r#"
        SELECT tc.topic_id  AS topic_id,
               ce.entity_id AS entity_id,
               count(*)     AS shared
        FROM kb_topic_chunks tc
        JOIN kb_chunk_entity_links ce
          ON tc.chunk_id = ce.chunk_id
        GROUP BY tc.topic_id, ce.entity_id
        HAVING count(*) >= $1
        "#,
    )
    .bind(opts.min_shared_chunks)
    .fetch_all(pool)
    .await?;
    report.pairs_considered = pair_rows.len();
    info!(pairs = report.pairs_considered, "topic-entity candidates");

    let mut written = 0usize;
    for (topic_id, entity_id, shared) in pair_rows {
        let ct = *topic_counts.get(&topic_id).unwrap_or(&0) as i64;
        let ce = *entity_counts.get(&entity_id).unwrap_or(&0) as i64;
        let union = (ct + ce - shared).max(1);
        let confidence = (shared as f32) / (union as f32);
        if confidence < opts.min_confidence {
            continue;
        }
        sqlx::query(
            r#"
            INSERT INTO kb_topic_entity_links (topic_id, entity_id, confidence)
            VALUES ($1, $2, $3)
            ON CONFLICT (topic_id, entity_id) DO UPDATE
              SET confidence = EXCLUDED.confidence
            "#,
        )
        .bind(topic_id)
        .bind(entity_id)
        .bind(confidence)
        .execute(pool)
        .await?;
        written += 1;
    }
    report.edges_written = written;

    info!(
        edges = written,
        "topic-entity link rebuild complete"
    );
    Ok(report)
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

/// A direct neighbor of a topic in the link graph, with the edge details
/// needed for display. `direction` describes the relationship from the
/// query topic's perspective for directional kinds:
///
/// - `OutgoingFrom` — the query topic is the `from_topic_id` of the edge.
///   For `subtopic_of`/`builds_on`, this means the query topic is the
///   more specific / dependent one.
/// - `IncomingTo` — the query topic is the `to_topic_id` of the edge.
///   The neighbor is the child / dependent.
/// - `Symmetric` — for `related`/`contradicts`, direction is meaningless.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EdgeDirection {
    OutgoingFrom,
    IncomingTo,
    Symmetric,
}

#[derive(Debug, Clone)]
pub struct LinkedNeighbor {
    pub topic_id: Uuid,
    pub label: String,
    pub slug: String,
    pub chunk_count: i32,
    pub confidence: f32,
    pub kind: String,
    pub direction: EdgeDirection,
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

/// Direct neighbors of one topic, ordered by edge confidence. Returns the
/// edge `kind` and the direction (from the perspective of the query topic)
/// so callers can group by relationship type.
pub async fn neighbors_of(
    pool: &PgPool,
    topic_id: Uuid,
    limit: i64,
) -> Result<Vec<LinkedNeighbor>, KbError> {
    // Two unions: rows where the query topic is the `from` side, and rows
    // where it's the `to` side. We carry a `dir_outgoing` boolean to remember
    // which side it was on (matters for directional kinds).
    let rows: Vec<(Uuid, String, String, i32, f32, String, bool)> = sqlx::query_as(
        r#"
        WITH n AS (
            SELECT to_topic_id   AS topic_id,
                   confidence,
                   kind::text    AS kind,
                   true          AS dir_outgoing
            FROM kb_topic_links
            WHERE from_topic_id = $1
            UNION ALL
            SELECT from_topic_id AS topic_id,
                   confidence,
                   kind::text    AS kind,
                   false         AS dir_outgoing
            FROM kb_topic_links
            WHERE to_topic_id = $1
        )
        SELECT t.topic_id, t.label, t.slug, t.chunk_count,
               n.confidence, n.kind, n.dir_outgoing
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
        .map(|(id, label, slug, cc, conf, kind, dir_outgoing)| {
            let direction = match kind.as_str() {
                "subtopic_of" | "builds_on" => {
                    if dir_outgoing {
                        EdgeDirection::OutgoingFrom
                    } else {
                        EdgeDirection::IncomingTo
                    }
                }
                _ => EdgeDirection::Symmetric,
            };
            LinkedNeighbor {
                topic_id: id,
                label,
                slug,
                chunk_count: cc,
                confidence: conf,
                kind,
                direction,
            }
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
