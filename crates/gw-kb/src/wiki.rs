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
    /// Entity mentions whose surfaces appear inside this section's
    /// `markdown`. Offsets are byte positions into `markdown` (the
    /// section-local frame, after chunks have been concatenated), not
    /// into the originating chunk. Sorted by `norm_start` ascending.
    /// Empty when extraction hasn't produced spans for these chunks yet.
    #[serde(default)]
    pub mentions: Vec<WikiMention>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiMention {
    pub entity_id: String,
    pub label: String,
    pub slug: String,
    pub kind: String,
    /// UTF-8 byte offset into the containing `WikiSection.markdown`.
    pub norm_start: usize,
    /// Exclusive. `markdown[norm_start..norm_end]` is the matched text.
    pub norm_end: usize,
    /// The exact matched surface string, redundant with the markdown
    /// slice. Kept on the wire so the frontend can verify a render
    /// hasn't drifted (markdown vs. plain-text differences, escaping).
    pub surface: String,
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

// ─── Cluster wiki ───────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct WikiCluster {
    pub title: String,
    pub summary: Option<String>,
    pub sources: Vec<WikiClusterSource>,
    /// Entities mentioned in at least two of the cluster's sources,
    /// ordered by total mentions desc and capped.
    pub shared_entities: Vec<WikiClusterEntity>,
    /// Topics that appear in at least two of the cluster's sources,
    /// ordered by aggregate chunk count desc and capped.
    pub shared_topics: Vec<WikiClusterTopic>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiClusterSource {
    pub source_id: String,
    pub title: String,
    pub author: Option<String>,
    pub url: Option<String>,
    pub arxiv_id: Option<String>,
    pub source_format: String,
    pub published_at: Option<String>,
    pub ingested_at: String,
    /// First chunk's markdown, truncated — a TLDR for the card.
    pub intro: Option<String>,
    pub top_entities: Vec<WikiEntityRef>,
    pub top_topics: Vec<WikiTopicRef>,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiClusterEntity {
    pub entity_id: String,
    pub label: String,
    pub slug: String,
    pub kind: String,
    /// Total mentions across the cluster's sources.
    pub total_mentions: i64,
    /// How many of the cluster's sources mention this entity.
    pub source_count: i64,
}

#[derive(Debug, Clone, Serialize)]
pub struct WikiClusterTopic {
    pub topic_id: String,
    pub label: String,
    pub slug: String,
    pub total_chunks: i64,
    pub source_count: i64,
}

/// Top-N cap for per-source `top_entities` / `top_topics` cards. The
/// rails on the wiki page show shared-across-sources; the per-card
/// list is meant to be a quick fingerprint, not exhaustive.
const PER_SOURCE_ENTITY_LIMIT: usize = 8;
const PER_SOURCE_TOPIC_LIMIT: usize = 5;
const INTRO_CHAR_LIMIT: usize = 360;

/// Build the cluster wiki payload for N sources. Each source's
/// per-doc query runs in parallel; cross-cutting entity/topic rollups
/// are computed in-memory from the same data — no second roundtrip.
pub async fn fetch_wiki_cluster(
    pool: &PgPool,
    source_ids: &[Uuid],
    title: Option<String>,
) -> Result<WikiCluster, KbError> {
    if source_ids.is_empty() {
        return Err(KbError::Other("cluster has no sources".into()));
    }

    let mut futs = Vec::with_capacity(source_ids.len());
    for sid in source_ids {
        futs.push(fetch_one_cluster_source(pool, *sid));
    }
    let sources: Vec<WikiClusterSource> = futures::future::try_join_all(futs).await?;

    let shared_entities = aggregate_shared_entities(&sources);
    let shared_topics = aggregate_shared_topics(&sources);

    let resolved_title = title.unwrap_or_else(|| {
        if sources.len() == 1 {
            sources[0].title.clone()
        } else {
            format!("Cluster of {} papers", sources.len())
        }
    });

    Ok(WikiCluster {
        title: resolved_title,
        summary: None,
        sources,
        shared_entities,
        shared_topics,
    })
}

async fn fetch_one_cluster_source(
    pool: &PgPool,
    source_id: Uuid,
) -> Result<WikiClusterSource, KbError> {
    let (source, chunks, entities, topics) = tokio::try_join!(
        fetch_source(pool, source_id),
        list_chunks_for_source(pool, source_id),
        fetch_doc_entities(pool, source_id),
        fetch_doc_topics(pool, source_id),
    )?;

    let intro = chunks.first().map(|c| truncate_intro(&c.content));
    let arxiv_id = source
        .metadata
        .get("arxiv_id")
        .and_then(|v| v.as_str())
        .map(|s| s.to_string());

    Ok(WikiClusterSource {
        source_id: source.source_id.to_string(),
        title: source.title,
        author: source.author,
        url: source.url,
        arxiv_id,
        source_format: source.source_format,
        published_at: source.published_at.map(|t| t.to_rfc3339()),
        ingested_at: source.ingested_at.to_rfc3339(),
        intro,
        top_entities: entities.into_iter().take(PER_SOURCE_ENTITY_LIMIT).collect(),
        top_topics: topics.into_iter().take(PER_SOURCE_TOPIC_LIMIT).collect(),
    })
}

fn truncate_intro(s: &str) -> String {
    let trimmed = s.trim();
    if trimmed.chars().count() <= INTRO_CHAR_LIMIT {
        return trimmed.to_string();
    }
    let cut: String = trimmed.chars().take(INTRO_CHAR_LIMIT).collect();
    // Trim back to the last space so we don't slice a word in half.
    let end = cut.rfind(char::is_whitespace).unwrap_or(cut.len());
    let mut out = cut[..end].to_string();
    out.push('…');
    out
}

fn aggregate_shared_entities(sources: &[WikiClusterSource]) -> Vec<WikiClusterEntity> {
    let mut acc: std::collections::HashMap<String, WikiClusterEntity> =
        std::collections::HashMap::new();
    for s in sources {
        for e in &s.top_entities {
            let entry = acc
                .entry(e.entity_id.clone())
                .or_insert_with(|| WikiClusterEntity {
                    entity_id: e.entity_id.clone(),
                    label: e.label.clone(),
                    slug: e.slug.clone(),
                    kind: e.kind.clone(),
                    total_mentions: 0,
                    source_count: 0,
                });
            entry.total_mentions += e.mentions_in_doc;
            entry.source_count += 1;
        }
    }
    let mut shared: Vec<WikiClusterEntity> =
        acc.into_values().filter(|e| e.source_count >= 2).collect();
    shared.sort_by(|a, b| {
        b.source_count
            .cmp(&a.source_count)
            .then(b.total_mentions.cmp(&a.total_mentions))
            .then(a.label.cmp(&b.label))
    });
    shared.truncate(30);
    shared
}

fn aggregate_shared_topics(sources: &[WikiClusterSource]) -> Vec<WikiClusterTopic> {
    let mut acc: std::collections::HashMap<String, WikiClusterTopic> =
        std::collections::HashMap::new();
    for s in sources {
        for t in &s.top_topics {
            let entry = acc
                .entry(t.topic_id.clone())
                .or_insert_with(|| WikiClusterTopic {
                    topic_id: t.topic_id.clone(),
                    label: t.label.clone(),
                    slug: t.slug.clone(),
                    total_chunks: 0,
                    source_count: 0,
                });
            entry.total_chunks += t.chunks_in_doc;
            entry.source_count += 1;
        }
    }
    let mut shared: Vec<WikiClusterTopic> =
        acc.into_values().filter(|t| t.source_count >= 2).collect();
    shared.sort_by(|a, b| {
        b.source_count
            .cmp(&a.source_count)
            .then(b.total_chunks.cmp(&a.total_chunks))
            .then(a.label.cmp(&b.label))
    });
    shared.truncate(20);
    shared
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

    let (toc, mut sections, chunk_offsets) = build_toc_and_sections(&chunks);
    // Mentions are loaded once per source and then sliced into the
    // section-local frame using the offsets we recorded during section
    // assembly. One DB roundtrip regardless of section count.
    populate_section_mentions(pool, source_id, &mut sections, &chunk_offsets).await?;

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

/// Where a chunk's content lands inside the assembled section markdown.
/// `section_idx` indexes into the `sections` vec; `section_offset` is
/// the byte position in `sections[section_idx].markdown` where this
/// chunk's content starts. Used by `populate_section_mentions` to
/// translate chunk-relative mention offsets into section-relative ones.
#[derive(Debug, Clone)]
struct ChunkPlacement {
    chunk_id: Uuid,
    section_idx: usize,
    section_offset: usize,
}

/// Group chunks (already in ordinal order) by their `heading_path` and
/// stitch their content into one section per path. Anchor IDs are
/// stable across calls — `sec-<index>` keyed on first-occurrence order.
///
/// Returns the TOC, sections (mentions empty — fill via
/// `populate_section_mentions`), and a placement map so callers can
/// translate chunk-relative offsets into section-relative ones.
fn build_toc_and_sections(
    chunks: &[crate::source::ChunkSummary],
) -> (Vec<WikiTocEntry>, Vec<WikiSection>, Vec<ChunkPlacement>) {
    const SEPARATOR: &str = "\n\n";
    let mut sections: Vec<WikiSection> = Vec::new();
    let mut toc: Vec<WikiTocEntry> = Vec::new();
    let mut placements: Vec<ChunkPlacement> = Vec::with_capacity(chunks.len());
    // Track which path we're currently appending to so contiguous
    // chunks under the same heading merge into one section.
    let mut current_path: Option<Vec<String>> = None;

    for chunk in chunks {
        let same_section = current_path
            .as_ref()
            .is_some_and(|p| p == &chunk.heading_path);
        if same_section {
            if let Some(last) = sections.last_mut() {
                let section_offset = last.markdown.len() + SEPARATOR.len();
                last.markdown.push_str(SEPARATOR);
                last.markdown.push_str(&chunk.content);
                placements.push(ChunkPlacement {
                    chunk_id: chunk.chunk_id,
                    section_idx: sections.len() - 1,
                    section_offset,
                });
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
        let section_idx = sections.len();
        sections.push(WikiSection {
            anchor,
            heading_path: chunk.heading_path.clone(),
            markdown: chunk.content.clone(),
            mentions: Vec::new(),
        });
        placements.push(ChunkPlacement {
            chunk_id: chunk.chunk_id,
            section_idx,
            section_offset: 0,
        });
        current_path = Some(chunk.heading_path.clone());
    }

    (toc, sections, placements)
}

/// Fetch all entity mentions for this source's chunks in one query,
/// then distribute them into the section-local frames defined by
/// `placements`. Mentions with NULL spans (legacy data migrated from
/// `kb_chunk_entity_links`) are silently dropped — the sidebar
/// (`fetch_doc_entities`) still surfaces their entities, just without
/// inline highlights.
async fn populate_section_mentions(
    pool: &PgPool,
    source_id: Uuid,
    sections: &mut [WikiSection],
    placements: &[ChunkPlacement],
) -> Result<(), KbError> {
    type Row = (Uuid, Uuid, String, String, String, i32, i32, String);
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT m.chunk_id,
               m.entity_id,
               e.label,
               e.slug,
               e.kind,
               m.norm_start,
               m.norm_end,
               m.surface
        FROM kb_entity_mentions m
        JOIN kb_chunks   c ON c.chunk_id  = m.chunk_id
        JOIN kb_entities e ON e.entity_id = m.entity_id
        WHERE c.source_id = $1
          AND m.norm_start IS NOT NULL
          AND m.norm_end   IS NOT NULL
          AND m.surface    IS NOT NULL
        "#,
    )
    .bind(source_id)
    .fetch_all(pool)
    .await?;

    // Build a chunk_id → placement lookup once.
    let by_chunk: std::collections::HashMap<Uuid, &ChunkPlacement> =
        placements.iter().map(|p| (p.chunk_id, p)).collect();

    for (chunk_id, entity_id, label, slug, kind, norm_start, norm_end, surface) in rows {
        let Some(placement) = by_chunk.get(&chunk_id) else {
            // Chunk in mentions but not in placements means it was
            // filtered before reaching the section builder — shouldn't
            // happen, but if it does, dropping is safe.
            continue;
        };
        let section = &mut sections[placement.section_idx];
        let section_start = placement.section_offset + norm_start as usize;
        let section_end = placement.section_offset + norm_end as usize;
        // Defensive: drop mentions whose translated span falls outside
        // the section markdown. Indicates a chunk content drift since
        // extraction — re-running the backfill is the fix.
        if section_end > section.markdown.len() {
            continue;
        }
        section.mentions.push(WikiMention {
            entity_id: entity_id.to_string(),
            label,
            slug,
            kind,
            norm_start: section_start,
            norm_end: section_end,
            surface,
        });
    }

    for section in sections.iter_mut() {
        section
            .mentions
            .sort_by_key(|m| (m.norm_start, m.norm_end));
    }
    Ok(())
}

async fn fetch_doc_entities(
    pool: &PgPool,
    source_id: Uuid,
) -> Result<Vec<WikiEntityRef>, KbError> {
    type Row = (Uuid, String, String, String, i64);
    // Counts mention rows rather than chunk-entity pairs: post-migration
    // 018, an entity can have multiple mentions per chunk (one per
    // occurrence), and the sidebar wants the true mention count to
    // sort by. Legacy span-less rows still contribute one count each,
    // which matches the old (chunk_id, entity_id) semantic.
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT e.entity_id, e.label, e.slug, e.kind, COUNT(m.mention_id) AS mentions
        FROM kb_entity_mentions m
        JOIN kb_chunks   c ON c.chunk_id  = m.chunk_id
        JOIN kb_entities e ON e.entity_id = m.entity_id
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
        let (toc, sections, _placements) = build_toc_and_sections(&chunks);
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
        let (toc, sections, _) = build_toc_and_sections(&chunks);
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
        let (toc, sections, _) = build_toc_and_sections(&chunks);
        assert_eq!(toc[0].anchor, "sec-0");
        assert_eq!(toc[1].anchor, "sec-1");
        assert_eq!(toc[2].anchor, "sec-2");
        assert_eq!(sections[2].anchor, "sec-2");
    }

    #[test]
    fn placement_offsets_align_with_assembled_markdown() {
        let chunks = vec![
            chunk(0, &["Intro"], "first paragraph"),
            chunk(1, &["Intro"], "second paragraph"),
            chunk(2, &["Body"], "body para"),
        ];
        let (_, sections, placements) = build_toc_and_sections(&chunks);
        assert_eq!(placements.len(), 3);
        // Chunk 0 starts at offset 0 of section 0.
        assert_eq!(placements[0].section_idx, 0);
        assert_eq!(placements[0].section_offset, 0);
        // Chunk 1 is appended after "\n\n" — offset = len("first paragraph") + 2.
        assert_eq!(placements[1].section_idx, 0);
        assert_eq!(placements[1].section_offset, "first paragraph".len() + 2);
        // Chunk 2 opens a new section (Body).
        assert_eq!(placements[2].section_idx, 1);
        assert_eq!(placements[2].section_offset, 0);
        // Verify the recorded offset actually points at the chunk content
        // inside the assembled markdown.
        let md = &sections[0].markdown;
        let off = placements[1].section_offset;
        assert_eq!(&md[off..off + "second paragraph".len()], "second paragraph");
    }
}
