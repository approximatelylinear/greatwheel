//! Source: an ingested document (URL or local file).

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;
use crate::extract::Extracted;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub source_id: Uuid,
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub title: String,
    pub author: Option<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub ingested_at: DateTime<Utc>,
    pub updated_at: DateTime<Utc>,
    pub content_hash: Vec<u8>,
    pub source_format: String,
    pub extractor: String,
    pub metadata: serde_json::Value,
}

/// SHA-256 of the extracted markdown body. Used for deduplication and
/// detecting upstream content changes.
pub fn compute_content_hash(markdown: &str) -> Vec<u8> {
    let mut hasher = Sha256::new();
    hasher.update(markdown.as_bytes());
    hasher.finalize().to_vec()
}

/// Outcome of an upsert: did we insert a new source, update an existing one,
/// or skip because the content hash already matched?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UpsertOutcome {
    Inserted,
    Updated,
    Unchanged,
}

#[derive(Debug, Clone)]
pub struct UpsertResult {
    pub source: Source,
    pub outcome: UpsertOutcome,
}

/// Insert or update a source row based on the extracted document.
///
/// Lookup key precedence: `url` if present, otherwise `file_path`.
pub async fn upsert_source(
    pool: &PgPool,
    url: Option<&str>,
    file_path: Option<&str>,
    extracted: &Extracted,
) -> Result<UpsertResult, KbError> {
    let content_hash = compute_content_hash(&extracted.markdown);
    let title = extracted
        .title
        .clone()
        .unwrap_or_else(|| derive_title_fallback(url, file_path));

    // Check for an existing row by url or path
    let existing: Option<(Uuid, Vec<u8>)> = if let Some(u) = url {
        sqlx::query_as("SELECT source_id, content_hash FROM kb_sources WHERE url = $1")
            .bind(u)
            .fetch_optional(pool)
            .await?
    } else if let Some(p) = file_path {
        sqlx::query_as("SELECT source_id, content_hash FROM kb_sources WHERE file_path = $1")
            .bind(p)
            .fetch_optional(pool)
            .await?
    } else {
        return Err(KbError::Other(
            "upsert_source requires either url or file_path".into(),
        ));
    };

    if let Some((existing_id, existing_hash)) = existing {
        if existing_hash == content_hash {
            // No change — skip update
            let source = fetch_source(pool, existing_id).await?;
            return Ok(UpsertResult {
                source,
                outcome: UpsertOutcome::Unchanged,
            });
        }

        // Content changed — update
        sqlx::query(
            r#"
            UPDATE kb_sources
            SET title = $2,
                author = $3,
                published_at = $4,
                content_hash = $5,
                source_format = $6,
                extractor = $7,
                updated_at = now()
            WHERE source_id = $1
            "#,
        )
        .bind(existing_id)
        .bind(&title)
        .bind(extracted.author.as_ref())
        .bind(extracted.published_at)
        .bind(&content_hash)
        .bind(&extracted.source_format)
        .bind(&extracted.extractor)
        .execute(pool)
        .await?;

        let source = fetch_source(pool, existing_id).await?;
        return Ok(UpsertResult {
            source,
            outcome: UpsertOutcome::Updated,
        });
    }

    // Insert new source
    let source_id: Uuid = sqlx::query_scalar(
        r#"
        INSERT INTO kb_sources
            (url, file_path, title, author, published_at,
             content_hash, source_format, extractor)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        RETURNING source_id
        "#,
    )
    .bind(url)
    .bind(file_path)
    .bind(&title)
    .bind(extracted.author.as_ref())
    .bind(extracted.published_at)
    .bind(&content_hash)
    .bind(&extracted.source_format)
    .bind(&extracted.extractor)
    .fetch_one(pool)
    .await?;

    let source = fetch_source(pool, source_id).await?;
    Ok(UpsertResult {
        source,
        outcome: UpsertOutcome::Inserted,
    })
}

pub async fn fetch_source(pool: &PgPool, source_id: Uuid) -> Result<Source, KbError> {
    let row: Source = sqlx::query_as::<_, SourceRow>(
        r#"
        SELECT source_id, url, file_path, title, author, published_at,
               ingested_at, updated_at, content_hash, source_format,
               extractor, metadata
        FROM kb_sources
        WHERE source_id = $1
        "#,
    )
    .bind(source_id)
    .fetch_one(pool)
    .await?
    .into();
    Ok(row)
}

/// A summary row for `kb-list` output: just enough to identify a source
/// and show how big it is.
#[derive(Debug, Clone)]
pub struct SourceSummary {
    pub source_id: Uuid,
    pub title: String,
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub source_format: String,
    pub ingested_at: DateTime<Utc>,
    pub published_at: Option<DateTime<Utc>>,
    pub chunk_count: i64,
}

/// List all sources, newest first, with chunk counts.
pub async fn list_sources(pool: &PgPool, limit: i64) -> Result<Vec<SourceSummary>, KbError> {
    let rows: Vec<(Uuid, String, Option<String>, Option<String>, String, DateTime<Utc>, Option<DateTime<Utc>>, i64)> =
        sqlx::query_as(
            r#"
            SELECT s.source_id,
                   s.title,
                   s.url,
                   s.file_path,
                   s.source_format,
                   s.ingested_at,
                   s.published_at,
                   COALESCE(c.chunk_count, 0) AS chunk_count
            FROM kb_sources s
            LEFT JOIN (
                SELECT source_id, COUNT(*) AS chunk_count
                FROM kb_chunks
                GROUP BY source_id
            ) c USING (source_id)
            ORDER BY s.ingested_at DESC
            LIMIT $1
            "#,
        )
        .bind(limit)
        .fetch_all(pool)
        .await?;

    Ok(rows
        .into_iter()
        .map(|(id, title, url, fp, fmt, ing, pub_at, cc)| SourceSummary {
            source_id: id,
            title,
            url,
            file_path: fp,
            source_format: fmt,
            ingested_at: ing,
            published_at: pub_at,
            chunk_count: cc,
        })
        .collect())
}

/// Resolve a user-supplied source identifier to a full UUID.
///
/// Accepts either a full UUID or any unique prefix of length >= 4.
/// Returns an error if the prefix matches multiple sources or none.
pub async fn resolve_source_id(pool: &PgPool, id_or_prefix: &str) -> Result<Uuid, KbError> {
    if let Ok(uuid) = Uuid::parse_str(id_or_prefix) {
        return Ok(uuid);
    }
    let prefix = id_or_prefix.trim();
    if prefix.len() < 4 {
        return Err(KbError::Other(
            "source id prefix must be at least 4 characters".into(),
        ));
    }
    let pattern = format!("{}%", prefix);
    let matches: Vec<(Uuid, String)> = sqlx::query_as(
        "SELECT source_id, title FROM kb_sources WHERE source_id::text LIKE $1 LIMIT 5",
    )
    .bind(&pattern)
    .fetch_all(pool)
    .await?;

    match matches.len() {
        0 => Err(KbError::Other(format!("no source matches '{prefix}'"))),
        1 => Ok(matches[0].0),
        _ => {
            let lines: Vec<String> = matches
                .iter()
                .map(|(id, title)| format!("  {} {}", id, title))
                .collect();
            Err(KbError::Other(format!(
                "ambiguous source prefix '{prefix}', matches:\n{}",
                lines.join("\n")
            )))
        }
    }
}

/// A chunk row for `kb source --chunks` output.
#[derive(Debug, Clone)]
pub struct ChunkSummary {
    pub chunk_id: Uuid,
    pub ordinal: i32,
    pub char_offset: i32,
    pub char_length: i32,
    pub heading_path: Vec<String>,
    pub content: String,
}

/// Fetch all chunks for a source in ordinal order.
pub async fn list_chunks_for_source(
    pool: &PgPool,
    source_id: Uuid,
) -> Result<Vec<ChunkSummary>, KbError> {
    let rows: Vec<(Uuid, i32, i32, i32, Vec<String>, String)> = sqlx::query_as(
        r#"
        SELECT chunk_id, ordinal, char_offset, char_length, heading_path, content
        FROM kb_chunks
        WHERE source_id = $1
        ORDER BY ordinal
        "#,
    )
    .bind(source_id)
    .fetch_all(pool)
    .await?;

    Ok(rows
        .into_iter()
        .map(|(id, ord, off, len, path, content)| ChunkSummary {
            chunk_id: id,
            ordinal: ord,
            char_offset: off,
            char_length: len,
            heading_path: path,
            content,
        })
        .collect())
}

/// Delete all chunks for a source. Used when re-ingesting changed content
/// before re-chunking.
pub async fn delete_chunks_for_source(pool: &PgPool, source_id: Uuid) -> Result<u64, KbError> {
    let result = sqlx::query("DELETE FROM kb_chunks WHERE source_id = $1")
        .bind(source_id)
        .execute(pool)
        .await?;
    Ok(result.rows_affected())
}

fn derive_title_fallback(url: Option<&str>, file_path: Option<&str>) -> String {
    if let Some(u) = url {
        return u.to_string();
    }
    if let Some(p) = file_path {
        return std::path::Path::new(p)
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or(p)
            .to_string();
    }
    "untitled".to_string()
}

// sqlx FromRow shim — kb_sources columns don't all map cleanly to Source
// because metadata is JSON and content_hash is bytea
#[derive(sqlx::FromRow)]
struct SourceRow {
    source_id: Uuid,
    url: Option<String>,
    file_path: Option<String>,
    title: String,
    author: Option<String>,
    published_at: Option<DateTime<Utc>>,
    ingested_at: DateTime<Utc>,
    updated_at: DateTime<Utc>,
    content_hash: Vec<u8>,
    source_format: String,
    extractor: String,
    metadata: serde_json::Value,
}

impl From<SourceRow> for Source {
    fn from(r: SourceRow) -> Self {
        Source {
            source_id: r.source_id,
            url: r.url,
            file_path: r.file_path,
            title: r.title,
            author: r.author,
            published_at: r.published_at,
            ingested_at: r.ingested_at,
            updated_at: r.updated_at,
            content_hash: r.content_hash,
            source_format: r.source_format,
            extractor: r.extractor,
            metadata: r.metadata,
        }
    }
}
