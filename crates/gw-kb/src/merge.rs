//! Topic merging — collapses near-duplicate topics into a single canonical
//! topic. Detects candidates by all-pairs `label_vector` cosine, then either
//! auto-merges (high confidence) or asks the LLM to confirm (medium
//! confidence).
//!
//! ### Pipeline per merge
//!
//! For each accepted (winner, loser) pair:
//!
//!   1. Move membership rows from loser → winner
//!      (`kb_topic_chunks`, with ON CONFLICT to dedupe overlapping chunks).
//!   2. Recompute the winner's content vector as a chunk-count-weighted
//!      mean of the two topic vectors. This is an algebraic approximation
//!      that avoids re-embedding member chunks. Acceptable because:
//!        - Both topics already have very similar labels (that's why we're
//!          merging them), so their vectors are already close together.
//!        - The next `gw-kb link` rebuild uses these vectors only for
//!          relative ranking, not absolute positions.
//!   3. Refresh the winner's `chunk_count` from the actual row count.
//!   4. Surgically delete the loser's edges from `kb_topic_links` so the
//!      graph doesn't reference a deleted topic. The user re-runs
//!      `gw-kb link` afterwards to rebuild any edges the merge would have
//!      created.
//!   5. Delete the loser from `kb_topics` (CASCADE removes any leftover
//!      membership rows).
//!
//! ### Why we delete edges instead of rewriting them
//!
//! After a merge, the loser's edges should logically transfer to the
//! winner. Rewriting them in place is hairy: the lower-UUID-first
//! convention may flip, the directional convention may flip, and we have
//! to dedupe against any edge the winner already has on the same
//! neighbour. Re-running `link` is one short SQL query and gets it right
//! by construction. So `merge` just removes broken edges and prints a
//! note telling the user to re-link.

use std::collections::HashSet;

use gw_llm::{Message, OllamaClient};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;
use crate::topics::{cosine, load_all_topic_states, vec_to_bytes, TopicState};

const REP_CHUNK_CHARS: usize = 200;
const SYSTEM_PROMPT: &str = "You decide whether two topics should be merged \
into one. Output JSON only — no prose, no markdown fences.";

#[derive(Debug, Clone, Copy)]
pub struct MergeOpts {
    /// Pairs with `cosine ≥ auto_threshold` merge without LLM confirmation.
    pub auto_threshold: f32,
    /// Pairs in `[ask_threshold, auto_threshold)` are sent to the LLM
    /// for a yes/no decision. Pairs below `ask_threshold` are ignored.
    pub ask_threshold: f32,
    /// Maximum number of candidate pairs to consider (highest cosine first).
    pub limit: Option<usize>,
    /// Don't actually persist anything — just report what would happen.
    pub dry_run: bool,
}

impl Default for MergeOpts {
    fn default() -> Self {
        Self {
            auto_threshold: 0.92,
            ask_threshold: 0.85,
            limit: None,
            dry_run: false,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct MergeReport {
    pub topics_seen: usize,
    pub candidates_considered: usize,
    pub auto_merges: usize,
    pub llm_confirmed: usize,
    pub llm_rejected: usize,
    pub llm_failures: usize,
    pub merges_executed: usize,
    pub examples: Vec<MergeExample>,
}

#[derive(Debug, Clone)]
pub struct MergeExample {
    pub winner: String,
    pub loser: String,
    pub cosine: f32,
    pub auto: bool,
}

#[derive(Debug, Default, Deserialize, Serialize)]
struct ConfirmOutput {
    #[serde(default)]
    merge: bool,
}

/// Find and execute topic merges.
pub async fn merge_topics(stores: &KbStores, opts: MergeOpts) -> Result<MergeReport, KbError> {
    let mut report = MergeReport::default();

    // 1. Load all topics into memory.
    let topics = load_all_topic_states(&stores.pg).await?;
    report.topics_seen = topics.len();
    info!(topics = topics.len(), "loaded topics for merge analysis");
    if topics.len() < 2 {
        return Ok(report);
    }

    // 2. Compute all-pairs label-vector cosine and collect candidates above
    //    the ask_threshold. Sorted descending.
    let mut candidates: Vec<(usize, usize, f32)> = Vec::new();
    for i in 0..topics.len() {
        for j in (i + 1)..topics.len() {
            let sim = cosine(&topics[i].label_vector, &topics[j].label_vector);
            if sim >= opts.ask_threshold {
                candidates.push((i, j, sim));
            }
        }
    }
    candidates.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap_or(std::cmp::Ordering::Equal));
    if let Some(limit) = opts.limit {
        candidates.truncate(limit);
    }
    info!(
        candidates = candidates.len(),
        "candidate pairs above ask threshold"
    );
    report.candidates_considered = candidates.len();

    // 3. Load representative chunk excerpts for any topic that might need
    //    LLM confirmation. We could lazy-load, but a single SQL query is
    //    cheap and saves coordinating async borrows in the loop.
    let snapshots = load_topic_excerpts(&stores.pg).await?;

    // 4. Walk candidates in cosine-descending order. Maintain a set of
    //    topics that have already been removed in this run so we never
    //    process a pair touching a deleted topic.
    let mut removed: HashSet<Uuid> = HashSet::new();

    for (i, j, sim) in &candidates {
        let t_i = &topics[*i];
        let t_j = &topics[*j];
        if removed.contains(&t_i.topic_id) || removed.contains(&t_j.topic_id) {
            continue;
        }

        // Pick winner = larger chunk count, ties broken by topic_id ordering
        // (deterministic).
        let (winner, loser) = if t_i.chunk_count > t_j.chunk_count
            || (t_i.chunk_count == t_j.chunk_count && t_i.topic_id < t_j.topic_id)
        {
            (t_i, t_j)
        } else {
            (t_j, t_i)
        };

        let auto = *sim >= opts.auto_threshold;
        let merge: bool;
        if auto {
            merge = true;
            report.auto_merges += 1;
        } else {
            // LLM confirmation required.
            let snap_a = snapshots
                .iter()
                .find(|s| s.topic_id == winner.topic_id)
                .map(|s| s.excerpt.as_str())
                .unwrap_or("");
            let snap_b = snapshots
                .iter()
                .find(|s| s.topic_id == loser.topic_id)
                .map(|s| s.excerpt.as_str())
                .unwrap_or("");
            match confirm_with_llm(&stores.llm, &winner.label, snap_a, &loser.label, snap_b).await {
                Ok(true) => {
                    merge = true;
                    report.llm_confirmed += 1;
                }
                Ok(false) => {
                    merge = false;
                    report.llm_rejected += 1;
                }
                Err(e) => {
                    warn!(err = %e, "merge confirmation failed, skipping pair");
                    merge = false;
                    report.llm_failures += 1;
                }
            }
        }

        if !merge {
            continue;
        }

        info!(
            winner = %winner.label,
            loser = %loser.label,
            cosine = sim,
            auto,
            "merging"
        );
        report.examples.push(MergeExample {
            winner: winner.label.clone(),
            loser: loser.label.clone(),
            cosine: *sim,
            auto,
        });

        if !opts.dry_run {
            execute_merge(&stores.pg, winner, loser).await?;
        }
        removed.insert(loser.topic_id);
        report.merges_executed += 1;
    }

    info!(
        merges = report.merges_executed,
        auto = report.auto_merges,
        confirmed = report.llm_confirmed,
        rejected = report.llm_rejected,
        failures = report.llm_failures,
        "merge complete"
    );

    if !opts.dry_run && report.merges_executed > 0 {
        info!("topic graph stale — run `gw-kb link` and `gw-kb classify` to rebuild edges");
    }

    Ok(report)
}

/// Execute one merge in a transaction.
///
/// See module docs for the algebraic vector recomputation rationale.
async fn execute_merge(
    pool: &PgPool,
    winner: &TopicState,
    loser: &TopicState,
) -> Result<(), KbError> {
    let mut tx = pool.begin().await?;

    // 1. Move chunk memberships, deduping any chunks that already belong
    //    to the winner. Take the higher relevance when both have it.
    sqlx::query(
        r#"
        INSERT INTO kb_topic_chunks (topic_id, chunk_id, relevance)
        SELECT $1, chunk_id, relevance
        FROM kb_topic_chunks
        WHERE topic_id = $2
        ON CONFLICT (topic_id, chunk_id) DO UPDATE
          SET relevance = GREATEST(kb_topic_chunks.relevance, EXCLUDED.relevance)
        "#,
    )
    .bind(winner.topic_id)
    .bind(loser.topic_id)
    .execute(&mut *tx)
    .await?;

    // 2. Recompute winner's content vector via weighted mean of the two
    //    topic vectors. The weight is `chunk_count + 1` because each topic
    //    vector is the mean of (label + chunk_count member embeddings).
    //    Overlapping chunks would technically get double-counted here, but
    //    in practice duplicate topics rarely share members and the
    //    approximation is well within the noise floor of cosine retrieval.
    let new_vec = weighted_mean(
        &winner.vector,
        winner.chunk_count + 1,
        &loser.vector,
        loser.chunk_count + 1,
    );

    // 3. Refresh chunk_count from the truth table (post-merge, post-dedupe).
    let new_count: i64 =
        sqlx::query_scalar("SELECT count(*) FROM kb_topic_chunks WHERE topic_id = $1")
            .bind(winner.topic_id)
            .fetch_one(&mut *tx)
            .await?;

    sqlx::query(
        r#"
        UPDATE kb_topics
        SET vector = $1,
            chunk_count = $2,
            last_seen = now(),
            updated_at = now()
        WHERE topic_id = $3
        "#,
    )
    .bind(vec_to_bytes(&new_vec))
    .bind(new_count as i32)
    .bind(winner.topic_id)
    .execute(&mut *tx)
    .await?;

    // 4. Surgical edge cleanup — see module docs.
    sqlx::query("DELETE FROM kb_topic_links WHERE from_topic_id = $1 OR to_topic_id = $1")
        .bind(loser.topic_id)
        .execute(&mut *tx)
        .await?;

    // 5. Drop the loser. CASCADE removes any leftover kb_topic_chunks rows.
    sqlx::query("DELETE FROM kb_topics WHERE topic_id = $1")
        .bind(loser.topic_id)
        .execute(&mut *tx)
        .await?;

    tx.commit().await?;

    debug!(
        winner = %winner.label,
        loser = %loser.label,
        new_count,
        "merge committed"
    );
    Ok(())
}

fn weighted_mean(a: &[f32], wa: usize, b: &[f32], wb: usize) -> Vec<f32> {
    debug_assert_eq!(a.len(), b.len());
    let denom = (wa + wb) as f32;
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x * wa as f32 + y * wb as f32) / denom)
        .collect()
}

#[derive(Debug, Clone)]
struct TopicExcerpt {
    topic_id: Uuid,
    excerpt: String,
}

async fn load_topic_excerpts(pool: &PgPool) -> Result<Vec<TopicExcerpt>, KbError> {
    let rows: Vec<(Uuid, String)> = sqlx::query_as(
        r#"
        SELECT DISTINCT ON (tc.topic_id) tc.topic_id, c.content
        FROM kb_topic_chunks tc
        JOIN kb_chunks c USING (chunk_id)
        ORDER BY tc.topic_id, tc.relevance DESC
        "#,
    )
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(topic_id, content)| TopicExcerpt {
            topic_id,
            excerpt: content
                .chars()
                .take(REP_CHUNK_CHARS)
                .collect::<String>()
                .replace('\n', " "),
        })
        .collect())
}

async fn confirm_with_llm(
    llm: &OllamaClient,
    label_a: &str,
    excerpt_a: &str,
    label_b: &str,
    excerpt_b: &str,
) -> Result<bool, KbError> {
    let prompt = format!(
        "Should these two topics be merged into a single topic?

Topic A: {label_a}
Sample passage from A:
\"\"\"
{excerpt_a}
\"\"\"

Topic B: {label_b}
Sample passage from B:
\"\"\"
{excerpt_b}
\"\"\"

DEFAULT TO `false`. Only respond `true` if the two labels are clearly two \
ways of writing the SAME concept — abbreviations, alternate spellings, \
case differences, or near-synonymous phrasings of the same underlying thing.

A topic that is a SPECIFIC INSTANCE OR SUBTYPE of another topic is NOT a \
merge — it's a subtype relationship and they must remain separate. A topic \
that is a SPECIFIC PART OR ASPECT of another topic is also NOT a merge.

SHOULD merge (same underlying concept):
  - \"I-205\" + \"Interstate 205 (Oregon–Washington)\"  → abbreviation
  - \"centroid pruning\" + \"Centroid Pruning\"  → casing
  - \"ΛCDM framework\" + \"Lambda-CDM model\"  → alternate writing
  - \"Pope Leo III\" + \"Pope Leo\"  → same person

Should NOT merge (different concepts even if related):
  - \"Amtrak trains\" + \"Passenger rail\"  → specific operator vs general category
  - \"Gothic choir\" + \"Gothic architecture\"  → part vs whole
  - \"ColBERTv2\" + \"Late interaction\"  → instance vs technique
  - \"Big Bang Nucleosynthesis\" + \"Standard Cosmology\"  → topic vs framework
  - \"Pre-trained language models\" + \"Neural Models\"  → subtype vs supertype
  - \"Coronation of Charlemagne\" + \"Carolingian Empire\"  → event vs entity

Output JSON only:
{{\"merge\": true}} or {{\"merge\": false}}"
    );

    let messages = vec![
        Message {
            role: "system".into(),
            content: SYSTEM_PROMPT.into(),
        },
        Message {
            role: "user".into(),
            content: prompt,
        },
    ];
    let resp = llm
        .chat_with_options(&messages, None, Some(false))
        .await
        .map_err(|e| KbError::Other(format!("llm chat: {e}")))?;

    let parsed: ConfirmOutput = crate::llm_parse::extract_json(&resp.content)?;
    Ok(parsed.merge)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_mean_basic() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        // Equal weights → midpoint
        let m = weighted_mean(&a, 1, &b, 1);
        assert!((m[0] - 0.5).abs() < 1e-6);
        assert!((m[1] - 0.5).abs() < 1e-6);
        // 3:1 weight → much closer to a
        let m = weighted_mean(&a, 3, &b, 1);
        assert!((m[0] - 0.75).abs() < 1e-6);
        assert!((m[1] - 0.25).abs() < 1e-6);
    }
}
