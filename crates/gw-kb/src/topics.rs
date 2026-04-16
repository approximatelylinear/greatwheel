//! Topic CRUD and vector helpers.
//!
//! Topic vectors are stored as raw float32 bytes in Postgres
//! (`kb_topics.vector` BYTEA). For our scale (hundreds of topics),
//! we load all of them into memory at organize time and do linear-scan
//! similarity search. LanceDB ANN for topics is a future optimization.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;

/// sqlx tuple for `load_all_topic_states` rows.
type TopicStateRow = (Uuid, String, String, i32, Option<Vec<u8>>, Option<Vec<u8>>);

/// sqlx tuple for `fetch_topic_by_slug` rows.
type TopicRow = (
    Uuid,
    String,
    String,
    i32,
    DateTime<Utc>,
    DateTime<Utc>,
    DateTime<Utc>,
    DateTime<Utc>,
);

/// sqlx tuple for `list_chunks_for_topic` rows.
type ChunkRow = (Uuid, String, Option<String>, Vec<String>, String, f32);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Topic {
    pub topic_id: Uuid,
    pub label: String,
    pub slug: String,
    pub chunk_count: i32,
    pub first_seen: DateTime<Utc>,
    pub last_seen: DateTime<Utc>,
    pub created_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
}

/// Topic state held in memory during an organize run.
///
/// Two vectors are kept per topic:
///
///  - `label_vector` — `embed(label)`, frozen at topic creation. Used for
///    *matching* new tag labels against existing topics. Pure label-to-label
///    similarity; never diluted by accumulated content.
///  - `vector` — running mean of (label embedding + member chunk embeddings).
///    Used for downstream tasks like spreading activation, query→topic
///    discovery, and topic-to-topic similarity.
#[derive(Debug, Clone)]
pub struct TopicState {
    pub topic_id: Uuid,
    pub label: String,
    pub slug: String,
    pub chunk_count: usize,
    pub label_vector: Vec<f32>,
    pub vector: Vec<f32>,
    /// Has the in-memory state been changed since the last DB flush?
    pub dirty: bool,
    /// Has the topic been newly created in this organize run (vs loaded from DB)?
    pub created_in_run: bool,
}

impl TopicState {
    /// Incrementally fold a new chunk into the running content vector.
    ///
    /// `vector` is the mean of (label embedding) + `chunk_count` chunk
    /// embeddings, so the total contributor count is `chunk_count + 1`.
    /// After adding the new chunk, the contributor count becomes
    /// `chunk_count + 2`.
    ///
    /// `label_vector` is intentionally NOT modified here.
    pub fn update_vector_with_member(&mut self, chunk_vec: &[f32]) {
        debug_assert_eq!(self.vector.len(), chunk_vec.len());
        let n = (self.chunk_count + 1) as f32; // current contributors (label + members)
        let new_n = n + 1.0;
        for (out, c) in self.vector.iter_mut().zip(chunk_vec.iter()) {
            *out = (*out * n + *c) / new_n;
        }
        self.chunk_count += 1;
        self.dirty = true;
    }
}

/// Convert a Vec<f32> to little-endian bytes for BYTEA storage.
pub fn vec_to_bytes(v: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(v.len() * 4);
    for f in v {
        out.extend_from_slice(&f.to_le_bytes());
    }
    out
}

/// Convert little-endian bytes back to Vec<f32>.
pub fn bytes_to_vec(b: &[u8]) -> Vec<f32> {
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// Cosine similarity between two same-dimension vectors.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

/// Convert a label to a URL-safe slug. Caller is responsible for
/// disambiguating collisions.
pub fn slugify(label: &str) -> String {
    let mut out = String::with_capacity(label.len());
    let mut prev_dash = false;
    for ch in label.chars() {
        if ch.is_ascii_alphanumeric() {
            out.push(ch.to_ascii_lowercase());
            prev_dash = false;
        } else if !prev_dash && !out.is_empty() {
            out.push('-');
            prev_dash = true;
        }
    }
    while out.ends_with('-') {
        out.pop();
    }
    if out.is_empty() {
        out.push_str("topic");
    }
    out
}

/// Load all existing topics from Postgres into in-memory state.
pub async fn load_all_topic_states(pool: &PgPool) -> Result<Vec<TopicState>, KbError> {
    let rows: Vec<TopicStateRow> = sqlx::query_as(
        "SELECT topic_id, label, slug, chunk_count, label_vector, vector FROM kb_topics",
    )
    .fetch_all(pool)
    .await?;

    let mut out = Vec::with_capacity(rows.len());
    for (topic_id, label, slug, chunk_count, label_bytes, vector_bytes) in rows {
        let (Some(lb), Some(vb)) = (label_bytes, vector_bytes) else {
            continue; // skip topics without vectors (shouldn't happen post-organize)
        };
        out.push(TopicState {
            topic_id,
            label,
            slug,
            chunk_count: chunk_count as usize,
            label_vector: bytes_to_vec(&lb),
            vector: bytes_to_vec(&vb),
            dirty: false,
            created_in_run: false,
        });
    }
    Ok(out)
}

/// Insert a brand-new topic. Caller has verified the slug is unique.
/// Both `label_vector` and `vector` are required at creation time —
/// `label_vector` is the pure label embedding, `vector` is the initial
/// content-aware mean (typically `mean(label_vector, first_chunk_vec)`).
pub async fn insert_topic(
    pool: &PgPool,
    label: &str,
    slug: &str,
    label_vector: &[f32],
    vector: &[f32],
    chunk_count: i32,
) -> Result<Uuid, KbError> {
    let label_bytes = vec_to_bytes(label_vector);
    let vector_bytes = vec_to_bytes(vector);
    let id: Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO kb_topics (label, slug, chunk_count, label_vector, vector)
        VALUES ($1, $2, $3, $4, $5)
        RETURNING topic_id
        "#,
    )
    .bind(label)
    .bind(slug)
    .bind(chunk_count)
    .bind(&label_bytes)
    .bind(&vector_bytes)
    .fetch_one(pool)
    .await?;
    Ok(id)
}

/// Update a topic's running content vector and chunk_count.
/// `label_vector` is never updated by this function.
pub async fn update_topic_vector(
    pool: &PgPool,
    topic_id: Uuid,
    vector: &[f32],
    chunk_count: i32,
) -> Result<(), KbError> {
    let bytes = vec_to_bytes(vector);
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
    .bind(&bytes)
    .bind(chunk_count)
    .bind(topic_id)
    .execute(pool)
    .await?;
    Ok(())
}

/// Insert a chunk-topic membership row. Idempotent on (topic_id, chunk_id).
pub async fn insert_topic_chunk(
    pool: &PgPool,
    topic_id: Uuid,
    chunk_id: Uuid,
    relevance: f32,
) -> Result<(), KbError> {
    sqlx::query(
        r#"
        INSERT INTO kb_topic_chunks (topic_id, chunk_id, relevance)
        VALUES ($1, $2, $3)
        ON CONFLICT (topic_id, chunk_id) DO NOTHING
        "#,
    )
    .bind(topic_id)
    .bind(chunk_id)
    .bind(relevance)
    .execute(pool)
    .await?;
    Ok(())
}

/// Mark a chunk as tagged and store its extracted entities.
pub async fn mark_chunk_tagged(
    pool: &PgPool,
    chunk_id: Uuid,
    entities: &[String],
) -> Result<(), KbError> {
    sqlx::query("UPDATE kb_chunks SET tagged_at = now(), entities = $2 WHERE chunk_id = $1")
        .bind(chunk_id)
        .bind(entities)
        .execute(pool)
        .await?;
    Ok(())
}

/// Summary row for `gw-kb topics`.
#[derive(Debug, Clone)]
pub struct TopicSummary {
    pub topic_id: Uuid,
    pub label: String,
    pub slug: String,
    pub chunk_count: i32,
    pub source_count: i64,
    pub last_seen: DateTime<Utc>,
}

pub async fn list_topic_summaries(pool: &PgPool, limit: i64) -> Result<Vec<TopicSummary>, KbError> {
    let rows: Vec<(Uuid, String, String, i32, i64, DateTime<Utc>)> = sqlx::query_as(
        r#"
        SELECT t.topic_id,
               t.label,
               t.slug,
               t.chunk_count,
               COALESCE(s.source_count, 0) AS source_count,
               t.last_seen
        FROM kb_topics t
        LEFT JOIN (
            SELECT tc.topic_id, COUNT(DISTINCT c.source_id) AS source_count
            FROM kb_topic_chunks tc
            JOIN kb_chunks c USING (chunk_id)
            GROUP BY tc.topic_id
        ) s USING (topic_id)
        ORDER BY t.chunk_count DESC, t.label ASC
        LIMIT $1
        "#,
    )
    .bind(limit)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, label, slug, cc, sc, ls)| TopicSummary {
            topic_id: id,
            label,
            slug,
            chunk_count: cc,
            source_count: sc,
            last_seen: ls,
        })
        .collect())
}

pub async fn fetch_topic_by_slug(pool: &PgPool, slug: &str) -> Result<Topic, KbError> {
    let row: TopicRow = sqlx::query_as(
        r#"
            SELECT topic_id, label, slug, chunk_count, first_seen, last_seen, created_at, updated_at
            FROM kb_topics
            WHERE slug = $1
            "#,
    )
    .bind(slug)
    .fetch_optional(pool)
    .await?
    .ok_or_else(|| KbError::Other(format!("no topic with slug '{slug}'")))?;
    Ok(Topic {
        topic_id: row.0,
        label: row.1,
        slug: row.2,
        chunk_count: row.3,
        first_seen: row.4,
        last_seen: row.5,
        created_at: row.6,
        updated_at: row.7,
    })
}

#[derive(Debug, Clone)]
pub struct TopicChunkRow {
    pub chunk_id: Uuid,
    pub source_title: String,
    pub source_url: Option<String>,
    pub heading_path: Vec<String>,
    pub content: String,
    pub relevance: f32,
}

pub async fn list_chunks_for_topic(
    pool: &PgPool,
    topic_id: Uuid,
    limit: i64,
) -> Result<Vec<TopicChunkRow>, KbError> {
    let rows: Vec<ChunkRow> = sqlx::query_as(
        r#"
        SELECT c.chunk_id,
               s.title,
               s.url,
               c.heading_path,
               c.content,
               tc.relevance
        FROM kb_topic_chunks tc
        JOIN kb_chunks c USING (chunk_id)
        JOIN kb_sources s ON s.source_id = c.source_id
        WHERE tc.topic_id = $1
        ORDER BY tc.relevance DESC
        LIMIT $2
        "#,
    )
    .bind(topic_id)
    .bind(limit)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(id, title, url, path, content, rel)| TopicChunkRow {
            chunk_id: id,
            source_title: title,
            source_url: url,
            heading_path: path,
            content,
            relevance: rel,
        })
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vec_roundtrip() {
        let v = vec![0.1, -0.2, 1e-5, 42.0];
        let b = vec_to_bytes(&v);
        let back = bytes_to_vec(&b);
        assert_eq!(v, back);
    }

    #[test]
    fn cosine_basic() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        assert!((cosine(&a, &b) - 1.0).abs() < 1e-6);
        assert!(cosine(&a, &c).abs() < 1e-6);
    }

    #[test]
    fn slug_basic() {
        assert_eq!(slugify("Carolingian Octagon"), "carolingian-octagon");
        assert_eq!(slugify("KV-cache compression!"), "kv-cache-compression");
        assert_eq!(slugify("   "), "topic");
        assert_eq!(slugify("---test---"), "test");
    }

    #[test]
    fn vector_update_label_dominates_initially() {
        let label_vec = vec![1.0, 0.0];
        // Topic created from label embedding only (chunk_count=0, n=1)
        let mut t = TopicState {
            topic_id: Uuid::nil(),
            label: "test".into(),
            slug: "test".into(),
            chunk_count: 0,
            label_vector: label_vec.clone(),
            vector: label_vec.clone(),
            dirty: false,
            created_in_run: true,
        };
        // Add a chunk vector orthogonal to the label
        t.update_vector_with_member(&[0.0, 1.0]);
        // After 1 chunk: mean of label + 1 chunk = (0.5, 0.5)
        assert!((t.vector[0] - 0.5).abs() < 1e-6);
        assert!((t.vector[1] - 0.5).abs() < 1e-6);
        assert_eq!(t.chunk_count, 1);
        // Add another orthogonal chunk
        t.update_vector_with_member(&[0.0, 1.0]);
        // 3 contributors: (1,0), (0,1), (0,1) → mean (1/3, 2/3)
        assert!((t.vector[0] - 1.0 / 3.0).abs() < 1e-6);
        assert!((t.vector[1] - 2.0 / 3.0).abs() < 1e-6);
        // label_vector is never touched
        assert_eq!(t.label_vector, label_vec);
    }
}
