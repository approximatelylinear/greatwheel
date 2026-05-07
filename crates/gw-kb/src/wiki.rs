//! Wiki-style view of a single KB source.
//!
//! Aggregates everything the frontend's `KbDocWiki` widget needs into
//! one read-only payload: source meta (infobox), TOC + sections grouped
//! by `heading_path`, mentioned entities, and referenced topics.
//!
//! Pure read; no schema changes. Sections are stitched from `kb_chunks`
//! rather than the original extracted markdown because we only persist
//! chunks today.

use serde::Serialize;
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::KbError;
use crate::source::{fetch_source, list_chunks_for_source, resolve_source_id};

#[derive(Debug, Clone, Serialize)]
pub struct WikiDoc {
    pub source: WikiSource,
    pub toc: Vec<WikiTocEntry>,
    pub sections: Vec<WikiSection>,
    pub entities: Vec<WikiEntityRef>,
    pub topics: Vec<WikiTopicRef>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiSource {
    pub source_id: String,
    pub title: String,
    pub author: Option<String>,
    pub url: Option<String>,
    pub file_path: Option<String>,
    pub source_format: String,
    pub published_at: Option<String>,
    pub ingested_at: String,
    pub metadata: serde_json::Value,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiTocEntry {
    pub anchor: String,
    pub label: String,
    pub depth: usize,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiSection {
    pub anchor: String,
    pub heading_path: Vec<String>,
    pub markdown: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiEntityRef {
    pub entity_id: String,
    pub label: String,
    pub slug: String,
    pub kind: String,
    pub mentions_in_doc: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiTopicRef {
    pub topic_id: String,
    pub label: String,
    pub slug: String,
    pub chunks_in_doc: i64,
}

/// Resolve an opaque user-supplied source reference into a concrete
/// `source_id`. Tries (in order):
///
/// 1. UUID or unique UUID prefix (>=4 chars) — see `resolve_source_id`.
/// 2. Full source URL (starts with `http`) — exact match on
///    `kb_sources.url`.
/// 3. arXiv id (e.g. `2504.13684`) — matches `metadata->>'arxiv_id'`,
///    falling back to a `https://arxiv.org/abs/<id>` URL match. The
///    literature_assistant ingester writes both, but older rows may
///    only have one.
///
/// Returns the resolved `source_id` or a `KbError::Other` describing
/// why no source matched.
pub async fn resolve_source_for_wiki(pool: &PgPool, query: &str) -> Result<Uuid, KbError> {
    let q = query.trim();
    if q.is_empty() {
        return Err(KbError::Other("source reference is empty".into()));
    }

    // (1) UUID / UUID-prefix path. Reuse the existing helper, which
    // also rejects ambiguous prefixes with a useful error.
    if Uuid::parse_str(q).is_ok() || (q.len() >= 4 && q.chars().all(|c| c.is_ascii_hexdigit() || c == '-')) {
        if let Ok(id) = resolve_source_id(pool, q).await {
            return Ok(id);
        }
        // Fall through to the URL/arxiv paths only when this looked
        // like a UUID prefix but didn't match. A real UUID that hit
        // resolve_source_id and errored (ambiguous) propagates below.
    }

    // (2) URL match. Tried before arxiv id so a fully-qualified URL
    // wins over the lighter-weight regex check.
    if q.starts_with("http://") || q.starts_with("https://") {
        if let Some((id,)) =
            sqlx::query_as::<_, (Uuid,)>("SELECT source_id FROM kb_sources WHERE url = $1")
                .bind(q)
                .fetch_optional(pool)
                .await?
        {
            return Ok(id);
        }
    }

    // (3) arXiv id. Hit the metadata column first; if unset (older
    // ingest), fall back to the canonical URL pattern.
    let arxiv_url = format!("https://arxiv.org/abs/{q}");
    if let Some((id,)) = sqlx::query_as::<_, (Uuid,)>(
        r#"
        SELECT source_id FROM kb_sources
        WHERE metadata->>'arxiv_id' = $1
           OR url = $2
        LIMIT 1
        "#,
    )
    .bind(q)
    .bind(&arxiv_url)
    .fetch_optional(pool)
    .await?
    {
        return Ok(id);
    }

    Err(KbError::Other(format!(
        "no kb source matches '{q}' (tried uuid/prefix, url, arxiv_id)"
    )))
}

/// Build the full wiki payload for a source. Each query is independent;
/// run them in parallel.
pub async fn fetch_wiki_doc(pool: &PgPool, source_id: Uuid) -> Result<WikiDoc, KbError> {
    let (source, chunks, entities, topics) = tokio::try_join!(
        fetch_source(pool, source_id),
        list_chunks_for_source(pool, source_id),
        fetch_doc_entities(pool, source_id),
        fetch_doc_topics(pool, source_id),
    )?;

    let (toc, sections) = build_toc_and_sections(&chunks);

    Ok(WikiDoc {
        source: WikiSource {
            source_id: source.source_id.to_string(),
            title: source.title,
            author: source.author,
            url: source.url,
            file_path: source.file_path,
            source_format: source.source_format,
            published_at: source.published_at.map(|t| t.to_rfc3339()),
            ingested_at: source.ingested_at.to_rfc3339(),
            metadata: source.metadata,
        },
        toc,
        sections,
        entities,
        topics,
    })
}

/// Group chunks (already in ordinal order) by their `heading_path` and
/// stitch their content into one section per path. Anchor IDs are
/// stable across calls — `sec-<index>` keyed on first-occurrence order.
fn build_toc_and_sections(
    chunks: &[crate::source::ChunkSummary],
) -> (Vec<WikiTocEntry>, Vec<WikiSection>) {
    let mut sections: Vec<WikiSection> = Vec::new();
    let mut toc: Vec<WikiTocEntry> = Vec::new();
    // Track which path we're currently appending to so contiguous
    // chunks under the same heading merge into one section.
    let mut current_path: Option<Vec<String>> = None;

    for chunk in chunks {
        let same_section = current_path
            .as_ref()
            .is_some_and(|p| p == &chunk.heading_path);
        if same_section {
            if let Some(last) = sections.last_mut() {
                last.markdown.push_str("\n\n");
                last.markdown.push_str(&chunk.content);
            }
            continue;
        }

        let anchor = format!("sec-{}", sections.len());
        let label = chunk
            .heading_path
            .last()
            .cloned()
            .unwrap_or_else(|| "(intro)".to_string());
        toc.push(WikiTocEntry {
            anchor: anchor.clone(),
            label,
            depth: chunk.heading_path.len().saturating_sub(1),
        });
        sections.push(WikiSection {
            anchor,
            heading_path: chunk.heading_path.clone(),
            markdown: chunk.content.clone(),
        });
        current_path = Some(chunk.heading_path.clone());
    }

    (toc, sections)
}

async fn fetch_doc_entities(
    pool: &PgPool,
    source_id: Uuid,
) -> Result<Vec<WikiEntityRef>, KbError> {
    type Row = (Uuid, String, String, String, i64);
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT e.entity_id, e.label, e.slug, e.kind, COUNT(*) AS mentions
        FROM kb_chunk_entity_links l
        JOIN kb_chunks c  ON c.chunk_id = l.chunk_id
        JOIN kb_entities e ON e.entity_id = l.entity_id
        WHERE c.source_id = $1
        GROUP BY e.entity_id, e.label, e.slug, e.kind
        ORDER BY mentions DESC, e.label ASC
        LIMIT 200
        "#,
    )
    .bind(source_id)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(entity_id, label, slug, kind, mentions)| WikiEntityRef {
            entity_id: entity_id.to_string(),
            label,
            slug,
            kind,
            mentions_in_doc: mentions,
        })
        .collect())
}

async fn fetch_doc_topics(pool: &PgPool, source_id: Uuid) -> Result<Vec<WikiTopicRef>, KbError> {
    type Row = (Uuid, String, String, i64);
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT t.topic_id, t.label, t.slug, COUNT(*) AS chunks_in_doc
        FROM kb_topic_chunks tc
        JOIN kb_chunks c ON c.chunk_id = tc.chunk_id
        JOIN kb_topics t ON t.topic_id = tc.topic_id
        WHERE c.source_id = $1
        GROUP BY t.topic_id, t.label, t.slug
        ORDER BY chunks_in_doc DESC, t.label ASC
        LIMIT 50
        "#,
    )
    .bind(source_id)
    .fetch_all(pool)
    .await?;
    Ok(rows
        .into_iter()
        .map(|(topic_id, label, slug, count)| WikiTopicRef {
            topic_id: topic_id.to_string(),
            label,
            slug,
            chunks_in_doc: count,
        })
        .collect())
}

#[allow(unused_imports)]
#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ChunkSummary;

    fn chunk(ord: i32, path: &[&str], content: &str) -> ChunkSummary {
        ChunkSummary {
            chunk_id: Uuid::new_v4(),
            ordinal: ord,
            char_offset: 0,
            char_length: content.len() as i32,
            heading_path: path.iter().map(|s| s.to_string()).collect(),
            content: content.to_string(),
        }
    }

    #[test]
    fn contiguous_chunks_under_same_heading_merge() {
        let chunks = vec![
            chunk(0, &["Intro"], "first paragraph"),
            chunk(1, &["Intro"], "second paragraph"),
            chunk(2, &["Body"], "body para"),
        ];
        let (toc, sections) = build_toc_and_sections(&chunks);
        assert_eq!(sections.len(), 2);
        assert_eq!(toc.len(), 2);
        assert_eq!(toc[0].label, "Intro");
        assert_eq!(toc[1].label, "Body");
        assert!(sections[0].markdown.contains("first paragraph"));
        assert!(sections[0].markdown.contains("second paragraph"));
    }

    #[test]
    fn empty_heading_path_labels_as_intro() {
        let chunks = vec![chunk(0, &[], "preamble")];
        let (toc, sections) = build_toc_and_sections(&chunks);
        assert_eq!(toc[0].label, "(intro)");
        assert_eq!(toc[0].depth, 0);
        assert_eq!(sections[0].heading_path, Vec::<String>::new());
    }

    #[test]
    fn anchors_are_stable_indexed() {
        let chunks = vec![
            chunk(0, &["A"], "a"),
            chunk(1, &["B"], "b"),
            chunk(2, &["C"], "c"),
        ];
        let (toc, sections) = build_toc_and_sections(&chunks);
        assert_eq!(toc[0].anchor, "sec-0");
        assert_eq!(toc[1].anchor, "sec-1");
        assert_eq!(toc[2].anchor, "sec-2");
        assert_eq!(sections[2].anchor, "sec-2");
    }
}
