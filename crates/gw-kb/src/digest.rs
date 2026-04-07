//! Delta digest — "what's new in the KB since time X".
//!
//! Produces a summary of sources ingested, topics created, and topics
//! that grew since a given timestamp. Designed to be run regularly
//! (probably after `gw-kb feed sync`) to surface what the knowledge
//! base has learned recently.

use chrono::{DateTime, Utc};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;

#[derive(Debug, Clone)]
pub struct DigestReport {
    pub since: DateTime<Utc>,
    pub new_sources: Vec<NewSource>,
    pub new_topics: Vec<NewTopic>,
    pub grown_topics: Vec<GrownTopic>,
}

#[derive(Debug, Clone)]
pub struct NewSource {
    pub source_id: Uuid,
    pub title: String,
    pub url: Option<String>,
    pub source_format: String,
    pub ingested_at: DateTime<Utc>,
    pub feed_name: Option<String>,
    pub chunk_count: i64,
}

#[derive(Debug, Clone)]
pub struct NewTopic {
    pub slug: String,
    pub label: String,
    pub chunk_count: i32,
    pub source_count: i64,
    pub created_at: DateTime<Utc>,
}

/// A pre-existing topic that picked up new chunks since the cutoff.
#[derive(Debug, Clone)]
pub struct GrownTopic {
    pub slug: String,
    pub label: String,
    pub total_chunks: i32,
    pub new_chunks_in_window: i64,
}

/// Produce a delta digest of everything added since `since`.
pub async fn build_digest(pool: &PgPool, since: DateTime<Utc>) -> Result<DigestReport, KbError> {
    let new_sources = fetch_new_sources(pool, since).await?;
    let new_topics = fetch_new_topics(pool, since).await?;
    let grown_topics = fetch_grown_topics(pool, since).await?;
    Ok(DigestReport {
        since,
        new_sources,
        new_topics,
        grown_topics,
    })
}

async fn fetch_new_sources(
    pool: &PgPool,
    since: DateTime<Utc>,
) -> Result<Vec<NewSource>, KbError> {
    let rows: Vec<(
        Uuid,
        String,
        Option<String>,
        String,
        DateTime<Utc>,
        Option<String>,
        i64,
    )> = sqlx::query_as(
        r#"
        SELECT s.source_id,
               s.title,
               s.url,
               s.source_format,
               s.ingested_at,
               f.name                                   AS feed_name,
               COALESCE(cc.chunk_count, 0)              AS chunk_count
        FROM kb_sources s
        LEFT JOIN kb_feeds f ON f.feed_id = s.feed_id
        LEFT JOIN (
            SELECT source_id, COUNT(*) AS chunk_count
            FROM kb_chunks
            GROUP BY source_id
        ) cc USING (source_id)
        WHERE s.ingested_at >= $1
        ORDER BY s.ingested_at DESC
        "#,
    )
    .bind(since)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(
            |(source_id, title, url, source_format, ingested_at, feed_name, chunk_count)| {
                NewSource {
                    source_id,
                    title,
                    url,
                    source_format,
                    ingested_at,
                    feed_name,
                    chunk_count,
                }
            },
        )
        .collect())
}

async fn fetch_new_topics(
    pool: &PgPool,
    since: DateTime<Utc>,
) -> Result<Vec<NewTopic>, KbError> {
    let rows: Vec<(String, String, i32, i64, DateTime<Utc>)> = sqlx::query_as(
        r#"
        SELECT t.slug,
               t.label,
               t.chunk_count,
               COALESCE(src.source_count, 0) AS source_count,
               t.created_at
        FROM kb_topics t
        LEFT JOIN (
            SELECT tc.topic_id, COUNT(DISTINCT c.source_id) AS source_count
            FROM kb_topic_chunks tc
            JOIN kb_chunks c USING (chunk_id)
            GROUP BY tc.topic_id
        ) src USING (topic_id)
        WHERE t.created_at >= $1
        ORDER BY t.chunk_count DESC, t.label ASC
        "#,
    )
    .bind(since)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(slug, label, chunk_count, source_count, created_at)| NewTopic {
            slug,
            label,
            chunk_count,
            source_count,
            created_at,
        })
        .collect())
}

/// Topics that existed before the cutoff but gained chunks inside the
/// window. We approximate "new chunks in window" as the count of member
/// chunks whose underlying source was ingested in the window.
async fn fetch_grown_topics(
    pool: &PgPool,
    since: DateTime<Utc>,
) -> Result<Vec<GrownTopic>, KbError> {
    let rows: Vec<(String, String, i32, i64)> = sqlx::query_as(
        r#"
        SELECT t.slug,
               t.label,
               t.chunk_count,
               COUNT(*) AS new_in_window
        FROM kb_topics t
        JOIN kb_topic_chunks tc USING (topic_id)
        JOIN kb_chunks c USING (chunk_id)
        JOIN kb_sources s USING (source_id)
        WHERE t.created_at < $1
          AND s.ingested_at >= $1
        GROUP BY t.topic_id, t.slug, t.label, t.chunk_count
        HAVING COUNT(*) > 0
        ORDER BY COUNT(*) DESC, t.label ASC
        LIMIT 25
        "#,
    )
    .bind(since)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(slug, label, total_chunks, new_chunks_in_window)| GrownTopic {
            slug,
            label,
            total_chunks,
            new_chunks_in_window,
        })
        .collect())
}
