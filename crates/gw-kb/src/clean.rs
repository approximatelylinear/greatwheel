//! Outlier chunk removal — post-hoc topic cleanup.
//!
//! The tagger occasionally puts a chunk into a topic where it doesn't
//! belong (e.g. a census intro paragraph tagged with "Information
//! Retrieval" because it mentions "data collection"). Running the
//! tagger again is expensive; removing the stray membership is cheap.
//!
//! ### Approach
//!
//! For each topic with `chunk_count >= min_chunks`:
//!   1. Re-embed all its member chunks' contents in batches (via the
//!      sentence-transformers Embedder — same path as ingest).
//!   2. Compute cosine between each chunk embedding and the topic's
//!      stored `vector` (the running label+members mean).
//!   3. Any chunk whose cosine is below `threshold` is flagged as an
//!      outlier. In non-dry-run mode we delete its row from
//!      `kb_topic_chunks` and refresh the topic's `chunk_count` plus
//!      the topic vector (recomputed as a weighted reduction of the
//!      removed outlier).
//!
//! Notes:
//!
//! - We batch the embedding calls per topic so one `embed_texts`
//!   request covers the whole topic rather than one call per chunk.
//! - Topic vectors are updated incrementally: removing a chunk that
//!   contributed to the mean is the inverse operation of adding it.
//!   See `remove_chunk_from_vector`.
//! - The topic link graph is left alone — it was computed from the
//!   topic vectors and will naturally pick up the cleanup the next
//!   time you run `gw-kb link`.

use sqlx::PgPool;
use tracing::{debug, info, warn};
use uuid::Uuid;

use crate::error::KbError;
use crate::ingest::KbStores;
use crate::topics::{bytes_to_vec, cosine, vec_to_bytes};

#[derive(Debug, Clone, Copy)]
pub struct CleanOpts {
    /// Only topics with chunk_count ≥ this are cleaned.
    pub min_chunks: i32,
    /// Membership cosine threshold. Chunks below this are outliers.
    pub threshold: f32,
    /// Don't persist changes; just print what would happen.
    pub dry_run: bool,
    /// If set, only clean the topic with this id.
    pub only_topic: Option<Uuid>,
}

impl Default for CleanOpts {
    fn default() -> Self {
        Self {
            min_chunks: 5,
            threshold: 0.55,
            dry_run: false,
            only_topic: None,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct CleanReport {
    pub topics_considered: usize,
    pub chunks_scored: usize,
    pub outliers_found: usize,
    pub outliers_removed: usize,
    pub examples: Vec<CleanExample>,
}

#[derive(Debug, Clone)]
pub struct CleanExample {
    pub topic_label: String,
    pub source_title: String,
    pub heading: String,
    pub cosine: f32,
    pub content_preview: String,
}

/// Walk topics and remove (or report) chunk memberships whose content
/// cosine to the topic vector is below `threshold`.
pub async fn clean_outliers(stores: &KbStores, opts: CleanOpts) -> Result<CleanReport, KbError> {
    let mut report = CleanReport::default();

    let topics = load_topics(&stores.pg, &opts).await?;
    report.topics_considered = topics.len();
    info!(topics = topics.len(), threshold = opts.threshold, "loaded topics for cleanup");

    for topic in topics {
        let members = load_members(&stores.pg, topic.topic_id).await?;
        if members.is_empty() {
            continue;
        }

        // Batch-embed the member chunks. sentence-transformers handles
        // the batching internally when given a single call.
        let texts: Vec<String> = members.iter().map(|m| m.content.clone()).collect();
        let vectors = match stores.embedder.embed_texts(&texts) {
            Ok(v) => v,
            Err(e) => {
                warn!(topic = %topic.label, err = %e, "failed to embed topic members, skipping");
                continue;
            }
        };
        if vectors.len() != members.len() {
            warn!(topic = %topic.label, "embed returned wrong count, skipping");
            continue;
        }

        // Score each member against the topic vector. Flag outliers.
        let mut outliers: Vec<(usize, f32)> = Vec::new();
        for (i, v) in vectors.iter().enumerate() {
            let sim = cosine(v, &topic.vector);
            if sim < opts.threshold {
                outliers.push((i, sim));
            }
            report.chunks_scored += 1;
        }

        if outliers.is_empty() {
            debug!(topic = %topic.label, "no outliers");
            continue;
        }

        // Record examples for the report (cap at a few per topic to
        // keep the output readable).
        for (i, sim) in outliers.iter().take(3) {
            let m = &members[*i];
            report.examples.push(CleanExample {
                topic_label: topic.label.clone(),
                source_title: m.source_title.clone(),
                heading: m.heading_path.last().cloned().unwrap_or_default(),
                cosine: *sim,
                content_preview: m
                    .content
                    .chars()
                    .take(120)
                    .collect::<String>()
                    .replace('\n', " "),
            });
        }
        report.outliers_found += outliers.len();

        if opts.dry_run {
            debug!(
                topic = %topic.label,
                outliers = outliers.len(),
                "dry-run: would remove"
            );
            continue;
        }

        // Execute the removal in a transaction per topic: delete the
        // outlier rows, then recompute the topic vector by folding
        // each outlier's contribution back out.
        let mut tx = stores.pg.begin().await?;
        let mut new_vec = topic.vector.clone();
        let mut current_total = (topic.chunk_count + 1) as f32; // label + chunks
        for (i, _) in &outliers {
            let m = &members[*i];
            remove_chunk_from_vector(&mut new_vec, current_total, &vectors[*i]);
            current_total -= 1.0;
            sqlx::query(
                "DELETE FROM kb_topic_chunks WHERE topic_id = $1 AND chunk_id = $2",
            )
            .bind(topic.topic_id)
            .bind(m.chunk_id)
            .execute(&mut *tx)
            .await?;
            report.outliers_removed += 1;
        }

        // Refresh chunk_count and vector on the topic.
        let new_count: i64 = sqlx::query_scalar(
            "SELECT count(*) FROM kb_topic_chunks WHERE topic_id = $1",
        )
        .bind(topic.topic_id)
        .fetch_one(&mut *tx)
        .await?;
        sqlx::query(
            r#"
            UPDATE kb_topics
            SET vector = $1,
                chunk_count = $2,
                updated_at = now()
            WHERE topic_id = $3
            "#,
        )
        .bind(vec_to_bytes(&new_vec))
        .bind(new_count as i32)
        .bind(topic.topic_id)
        .execute(&mut *tx)
        .await?;

        tx.commit().await?;
    }

    info!(
        scored = report.chunks_scored,
        found = report.outliers_found,
        removed = report.outliers_removed,
        "outlier cleanup complete"
    );
    if report.outliers_removed > 0 && !opts.dry_run {
        info!("topic graph may need refresh — consider running `gw-kb link` and `gw-kb synthesize --stale-only`");
    }
    Ok(report)
}

/// Fold a single contributor out of a running mean.
///
/// If `vec` was computed as `sum / total`, removing one contributor `v`
/// gives us `(sum - v) / (total - 1)`, which we can compute in place.
/// `total` is the number of contributors BEFORE removal.
pub fn remove_chunk_from_vector(vec: &mut Vec<f32>, total: f32, v: &[f32]) {
    debug_assert_eq!(vec.len(), v.len());
    if total <= 1.0 {
        return; // can't reduce below the label-only vector
    }
    let new_total = total - 1.0;
    for (x, y) in vec.iter_mut().zip(v.iter()) {
        let sum = *x * total;
        *x = (sum - y) / new_total;
    }
}

#[derive(Debug, Clone)]
struct TopicRow {
    topic_id: Uuid,
    label: String,
    chunk_count: i32,
    vector: Vec<f32>,
}

async fn load_topics(pool: &PgPool, opts: &CleanOpts) -> Result<Vec<TopicRow>, KbError> {
    let min_chunks = opts.min_chunks;
    let only = opts.only_topic;
    let rows: Vec<(Uuid, String, i32, Option<Vec<u8>>)> = sqlx::query_as(
        r#"
        SELECT topic_id, label, chunk_count, vector
        FROM kb_topics
        WHERE chunk_count >= $1
          AND ($2::uuid IS NULL OR topic_id = $2::uuid)
          AND vector IS NOT NULL
        ORDER BY chunk_count DESC
        "#,
    )
    .bind(min_chunks)
    .bind(only)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .filter_map(|(id, label, cc, vec_bytes)| {
            vec_bytes.map(|b| TopicRow {
                topic_id: id,
                label,
                chunk_count: cc,
                vector: bytes_to_vec(&b),
            })
        })
        .collect())
}

#[derive(Debug, Clone)]
struct MemberChunk {
    chunk_id: Uuid,
    source_title: String,
    heading_path: Vec<String>,
    content: String,
}

async fn load_members(pool: &PgPool, topic_id: Uuid) -> Result<Vec<MemberChunk>, KbError> {
    let rows: Vec<(Uuid, String, Vec<String>, String)> = sqlx::query_as(
        r#"
        SELECT c.chunk_id, s.title, c.heading_path, c.content
        FROM kb_topic_chunks tc
        JOIN kb_chunks c USING (chunk_id)
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE tc.topic_id = $1
        ORDER BY c.source_id, c.ordinal
        "#,
    )
    .bind(topic_id)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(chunk_id, source_title, heading_path, content)| MemberChunk {
            chunk_id,
            source_title,
            heading_path,
            content,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remove_chunk_from_vector_inverse_of_add() {
        // Start with a "mean of (1,0) and (0,1)" = (0.5, 0.5), total=2
        let mut v = vec![0.5, 0.5];
        // Remove (0,1). New mean = (1.0, 0.0), total=1
        remove_chunk_from_vector(&mut v, 2.0, &[0.0, 1.0]);
        assert!((v[0] - 1.0).abs() < 1e-6);
        assert!((v[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn remove_chunk_from_vector_noop_at_total_1() {
        // Can't reduce below the label-only vector.
        let mut v = vec![1.0, 0.0];
        let orig = v.clone();
        remove_chunk_from_vector(&mut v, 1.0, &[0.0, 1.0]);
        assert_eq!(v, orig);
    }
}
