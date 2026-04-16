//! Graph retrieval channel for Hindsight memory.
//!
//! Implements spreading activation over the memory_edges graph
//! (design-hindsight-memory.md §2.3).

use std::collections::{HashMap, HashSet};

use sqlx::PgPool;
use uuid::Uuid;

use crate::fusion::ScoredKey;

/// Run spreading activation from seed memories through the memory graph.
///
/// Starting from `seeds` (typically the top-k results from vector/BM25 search),
/// traverses `memory_edges` up to `max_hops` steps, decaying activation by
/// `decay * edge_weight` at each hop.
///
/// If `temporal_filter` is provided, only nodes whose `id` is in the set are
/// visited (temporal-constrained graph traversal, §2.4.4).
///
/// Returns scored keys sorted by activation (highest first).
///
/// TODO(perf): This issues N+1 queries (1 per hop + seed/key resolution).
/// For large graphs, collapse into a single recursive CTE:
/// ```sql
/// WITH RECURSIVE spread AS (
///     SELECT to_id, weight, 1 AS hop FROM memory_edges WHERE from_id = ANY($1)
///     UNION ALL
///     SELECT e.to_id, s.weight * $2 * e.weight, s.hop + 1
///     FROM spread s JOIN memory_edges e ON e.from_id = s.to_id
///     WHERE s.hop < $3
/// )
/// SELECT DISTINCT ON (to_id) to_id, weight FROM spread ORDER BY to_id, weight DESC
/// ```
pub async fn spreading_activation(
    pool: &PgPool,
    org_id: &Uuid,
    seeds: &[ScoredKey],
    max_hops: usize,
    decay: f32,
    temporal_filter: Option<&HashSet<Uuid>>,
) -> Result<Vec<ScoredKey>, sqlx::Error> {
    if seeds.is_empty() || max_hops == 0 {
        return Ok(Vec::new());
    }

    // Resolve seed keys to memory IDs
    let seed_keys: Vec<String> = seeds.iter().map(|s| s.key.clone()).collect();
    let seed_rows: Vec<(Uuid, String)> =
        sqlx::query_as("SELECT id, key FROM memories WHERE org_id = $1 AND key = ANY($2)")
            .bind(org_id)
            .bind(&seed_keys)
            .fetch_all(pool)
            .await?;

    // Initialize frontier with seed activations
    let seed_score_map: HashMap<&str, f32> =
        seeds.iter().map(|s| (s.key.as_str(), s.score)).collect();
    let mut activations: HashMap<Uuid, f32> = HashMap::new();
    for (id, key) in &seed_rows {
        let base = seed_score_map.get(key.as_str()).copied().unwrap_or(1.0);
        activations.insert(*id, base);
    }

    let seed_ids: HashSet<Uuid> = activations.keys().copied().collect();
    let mut frontier: Vec<Uuid> = activations.keys().copied().collect();

    // Spread activation for each hop.
    // Nodes can be revisited if a later hop finds a higher-weight path —
    // we only add to next_frontier when activation actually improves.
    for _hop in 0..max_hops {
        if frontier.is_empty() {
            break;
        }

        // Fetch edges from frontier nodes
        let edges: Vec<(Uuid, Uuid, f32)> = sqlx::query_as(
            "SELECT from_id, to_id, weight FROM memory_edges WHERE from_id = ANY($1)",
        )
        .bind(&frontier)
        .fetch_all(pool)
        .await?;

        let mut next_frontier = Vec::new();
        let mut frontier_set = HashSet::new();

        for (from_id, to_id, weight) in edges {
            // Don't traverse back into seed nodes
            if seed_ids.contains(&to_id) {
                continue;
            }

            // Apply temporal filter if present
            if let Some(filter) = temporal_filter {
                if !filter.contains(&to_id) {
                    continue;
                }
            }

            let parent_activation = activations.get(&from_id).copied().unwrap_or(0.0);
            let new_activation = parent_activation * decay * weight;

            if new_activation < 0.001 {
                continue; // Below threshold, stop propagating
            }

            let entry = activations.entry(to_id).or_insert(0.0);
            if new_activation > *entry {
                *entry = new_activation;
                // Only re-expand if activation improved
                if frontier_set.insert(to_id) {
                    next_frontier.push(to_id);
                }
            }
        }

        frontier = next_frontier;
    }

    // Remove seeds from results (caller already has them)
    let mut discovered: Vec<(Uuid, f32)> = activations
        .into_iter()
        .filter(|(id, _)| !seed_ids.contains(id))
        .collect();
    discovered.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    // Resolve IDs back to keys
    if discovered.is_empty() {
        return Ok(Vec::new());
    }

    let discovered_ids: Vec<Uuid> = discovered.iter().map(|(id, _)| *id).collect();
    let key_rows: Vec<(Uuid, String)> =
        sqlx::query_as("SELECT id, key FROM memories WHERE id = ANY($1)")
            .bind(&discovered_ids)
            .fetch_all(pool)
            .await?;

    let id_to_key: HashMap<Uuid, String> = key_rows.into_iter().collect();

    Ok(discovered
        .into_iter()
        .filter_map(|(id, score)| {
            id_to_key.get(&id).map(|key| ScoredKey {
                key: key.clone(),
                score,
            })
        })
        .collect())
}

/// Fetch memory IDs that overlap a temporal range.
///
/// Used to build the `R_temp` set for temporal-constrained graph traversal.
pub async fn fetch_temporal_set(
    pool: &PgPool,
    org_id: &Uuid,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
) -> Result<HashSet<Uuid>, sqlx::Error> {
    let rows: Vec<(Uuid,)> = sqlx::query_as(
        r#"
        SELECT id FROM memories
        WHERE org_id = $1
          AND occurred_at <= $2
          AND COALESCE(occurred_end, occurred_at) >= $3
        "#,
    )
    .bind(org_id)
    .bind(end)
    .bind(start)
    .fetch_all(pool)
    .await?;

    Ok(rows.into_iter().map(|(id,)| id).collect())
}

/// Score memories by temporal proximity to a query range.
///
/// Queries the database for memories overlapping the range and scores each
/// by how close its midpoint is to the query range midpoint.
pub async fn temporal_score_memories(
    pool: &PgPool,
    org_id: &Uuid,
    start: chrono::DateTime<chrono::Utc>,
    end: chrono::DateTime<chrono::Utc>,
    limit: usize,
) -> Result<Vec<ScoredKey>, sqlx::Error> {
    use crate::temporal::{temporal_proximity_score, TemporalRange};

    let rows: Vec<(
        String,
        chrono::DateTime<chrono::Utc>,
        Option<chrono::DateTime<chrono::Utc>>,
    )> = sqlx::query_as(
        r#"
            SELECT key, occurred_at, occurred_end FROM memories
            WHERE org_id = $1
              AND occurred_at <= $2
              AND COALESCE(occurred_end, occurred_at) >= $3
            "#,
    )
    .bind(org_id)
    .bind(end)
    .bind(start)
    .fetch_all(pool)
    .await?;

    let query_range = TemporalRange { start, end };
    let mut scored: Vec<ScoredKey> = rows
        .into_iter()
        .map(|(key, occ_start, occ_end)| {
            let fact_end = occ_end.unwrap_or(occ_start);
            let score = temporal_proximity_score(occ_start, fact_end, &query_range);
            ScoredKey { key, score }
        })
        .collect();

    scored.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(limit);
    Ok(scored)
}
