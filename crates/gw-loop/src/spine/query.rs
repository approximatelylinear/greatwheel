//! Read-side queries for spine UI surfaces.
//!
//! `fetch_segment_detail` is the read-only DB join that powers the
//! Phase-B EntitySidebar: given a segment id, return the segment plus
//! its entries (with previews), entities (with global metadata), and
//! typed relations within the entry range. No LLM, no canonicalisation
//! — just a server-side join.
//!
//! Lives in `gw-loop` because it joins across `session_segments` and
//! `session_entries` (gw-loop-owned tables), `kb_entities` and
//! `session_entry_relations` (entity side). The HTTP handler that
//! exposes this lives in the binary that owns the AG-UI router; this
//! module just provides the typed result.

use chrono::{DateTime, Utc};
use serde::Serialize;
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::LoopError;

const PREVIEW_CHARS: usize = 240;

#[derive(Debug, Clone, Serialize)]
pub struct SegmentDetail {
    pub segment: SegmentSummary,
    pub entries: Vec<EntrySummary>,
    pub entities: Vec<EntityCard>,
    pub relations: Vec<RelationRow>,
}

#[derive(Debug, Clone, Serialize)]
pub struct SegmentSummary {
    pub id: Uuid,
    pub session_id: Uuid,
    pub label: String,
    pub kind: String,
    pub entry_first: Uuid,
    pub entry_last: Uuid,
    pub entity_count: usize,
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntrySummary {
    pub id: Uuid,
    /// "user" | "assistant" — derived from `entry_type_tag`.
    pub role: String,
    pub created_at: DateTime<Utc>,
    /// First N characters of the entry text (whichever variant of
    /// `EntryType` it was). Lets the sidebar render a recognisable
    /// "first message in this segment" preview without paging the
    /// whole content blob.
    pub preview: String,
    /// Whether `preview` was cut short. The UI can render a "…"
    /// affordance.
    pub preview_truncated: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct EntityCard {
    pub entity_id: Uuid,
    pub label: String,
    pub slug: String,
    pub kind: String,
    /// How many entries within this segment mention the entity.
    pub mentions_in_segment: i64,
    /// `kb_entities.mentions` — how many chunks across the whole
    /// corpus mention this entity. Lets the UI flag a "first time
    /// in this conversation but seen often elsewhere" entity.
    pub global_mentions: i32,
    pub aliases: Vec<String>,
    pub summary: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RelationRow {
    pub id: Uuid,
    pub entry_id: Uuid,
    pub subject_id: Uuid,
    pub subject_label: String,
    pub object_id: Uuid,
    pub object_label: String,
    pub predicate: String,
    pub directed: bool,
    pub surface: String,
    pub confidence: f32,
}

/// One read of all the data the EntitySidebar needs for a focused
/// segment. Returns `None` if the segment doesn't exist or has been
/// invalidated by re-segmentation. Errors only on actual DB failures.
pub async fn fetch_segment_detail(
    pool: &PgPool,
    segment_id: Uuid,
) -> Result<Option<SegmentDetail>, LoopError> {
    type SegmentRow = (
        Uuid,
        Uuid,
        String,
        String,
        Uuid,
        Uuid,
        Vec<Uuid>,
        Option<String>,
        DateTime<Utc>,
    );
    let seg_row: Option<SegmentRow> = sqlx::query_as(
        r#"
        SELECT segment_id, session_id, label, kind, entry_first,
               entry_last, entity_ids, summary, created_at
        FROM session_segments
        WHERE segment_id = $1 AND invalidated_at IS NULL
        "#,
    )
    .bind(segment_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("fetch segment: {e}")))?;
    let Some((id, session_id, label, kind, entry_first, entry_last, entity_ids, summary, created_at)) =
        seg_row
    else {
        return Ok(None);
    };

    let segment = SegmentSummary {
        id,
        session_id,
        label,
        kind,
        entry_first,
        entry_last,
        entity_count: entity_ids.len(),
        summary,
        created_at,
    };

    // Entries in [first, last] inclusive, ordered by created_at. Use
    // the bracket entries' timestamps as bounds — same shape the
    // segmenter walked.
    type EntryRow = (Uuid, String, serde_json::Value, DateTime<Utc>);
    let entry_rows: Vec<EntryRow> = sqlx::query_as(
        r#"
        WITH bounds AS (
            SELECT
                (SELECT created_at FROM session_entries WHERE id = $2) AS first_at,
                (SELECT created_at FROM session_entries WHERE id = $3) AS last_at
        )
        SELECT se.id, se.entry_type, se.content, se.created_at
        FROM session_entries se, bounds b
        WHERE se.session_id = $1
          AND se.entry_type IN ('user_message', 'assistant_message')
          AND se.created_at BETWEEN b.first_at AND b.last_at
        ORDER BY se.created_at, se.id
        "#,
    )
    .bind(segment.session_id)
    .bind(segment.entry_first)
    .bind(segment.entry_last)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("fetch entries: {e}")))?;
    let entries: Vec<EntrySummary> = entry_rows
        .into_iter()
        .map(|(id, entry_type, content, created_at)| {
            let role = match entry_type.as_str() {
                "user_message" => "user".to_string(),
                "assistant_message" => "assistant".to_string(),
                other => other.to_string(),
            };
            let text = entry_text(&content);
            let truncated = text.chars().count() > PREVIEW_CHARS;
            let preview: String = text.chars().take(PREVIEW_CHARS).collect();
            EntrySummary {
                id,
                role,
                created_at,
                preview,
                preview_truncated: truncated,
            }
        })
        .collect();

    // Entities in this segment, ranked by within-segment mention count.
    // The JOIN into kb_entities gives the labels + aliases the UI
    // needs to render cards. The within-segment count is computed in
    // the same query so we don't fan out a per-entity follow-up.
    type EntityRow = (
        Uuid,
        String,
        String,
        String,
        Vec<String>,
        i32,
        Option<String>,
        i64,
    );
    let entity_rows: Vec<EntityRow> = sqlx::query_as(
        r#"
        WITH bounds AS (
            SELECT
                (SELECT created_at FROM session_entries WHERE id = $2) AS first_at,
                (SELECT created_at FROM session_entries WHERE id = $3) AS last_at
        ),
        seg_entries AS (
            SELECT se.id
            FROM session_entries se, bounds b
            WHERE se.session_id = $1
              AND se.entry_type IN ('user_message', 'assistant_message')
              AND se.created_at BETWEEN b.first_at AND b.last_at
        )
        SELECT e.entity_id, e.label, e.slug, e.kind,
               e.aliases, e.mentions, e.summary,
               COUNT(DISTINCT ee.entry_id) AS mentions_in_segment
        FROM session_entry_entities ee
        JOIN kb_entities e ON e.entity_id = ee.entity_id
        WHERE ee.entry_id IN (SELECT id FROM seg_entries)
        GROUP BY e.entity_id, e.label, e.slug, e.kind, e.aliases,
                 e.mentions, e.summary
        ORDER BY mentions_in_segment DESC, e.label
        "#,
    )
    .bind(segment.session_id)
    .bind(segment.entry_first)
    .bind(segment.entry_last)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("fetch entities: {e}")))?;
    let entities: Vec<EntityCard> = entity_rows
        .into_iter()
        .map(
            |(
                entity_id,
                label,
                slug,
                kind,
                aliases,
                global_mentions,
                summary,
                mentions_in_segment,
            )| EntityCard {
                entity_id,
                label,
                slug,
                kind,
                aliases,
                global_mentions,
                summary,
                mentions_in_segment,
            },
        )
        .collect();

    // Relations asserted in this segment's entries. Subject + object
    // labels are joined in directly so the UI can render
    // `subject ←predicate→ object` rows without a second round-trip.
    type RelationRowRaw = (
        Uuid,
        Uuid,
        Uuid,
        String,
        Uuid,
        String,
        String,
        bool,
        String,
        f32,
    );
    let relation_rows: Vec<RelationRowRaw> = sqlx::query_as(
        r#"
        WITH bounds AS (
            SELECT
                (SELECT created_at FROM session_entries WHERE id = $2) AS first_at,
                (SELECT created_at FROM session_entries WHERE id = $3) AS last_at
        ),
        seg_entries AS (
            SELECT se.id
            FROM session_entries se, bounds b
            WHERE se.session_id = $1
              AND se.entry_type IN ('user_message', 'assistant_message')
              AND se.created_at BETWEEN b.first_at AND b.last_at
        )
        SELECT r.relation_id, r.entry_id, r.subject_id, s.label,
               r.object_id, o.label, r.predicate, r.directed,
               r.surface, r.confidence
        FROM session_entry_relations r
        JOIN kb_entities s ON s.entity_id = r.subject_id
        JOIN kb_entities o ON o.entity_id = r.object_id
        WHERE r.entry_id IN (SELECT id FROM seg_entries)
        ORDER BY r.extracted_at, r.relation_id
        "#,
    )
    .bind(segment.session_id)
    .bind(segment.entry_first)
    .bind(segment.entry_last)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("fetch relations: {e}")))?;
    let relations: Vec<RelationRow> = relation_rows
        .into_iter()
        .map(
            |(
                id,
                entry_id,
                subject_id,
                subject_label,
                object_id,
                object_label,
                predicate,
                directed,
                surface,
                confidence,
            )| RelationRow {
                id,
                entry_id,
                subject_id,
                subject_label,
                object_id,
                object_label,
                predicate,
                directed,
                surface,
                confidence,
            },
        )
        .collect();

    Ok(Some(SegmentDetail {
        segment,
        entries,
        entities,
        relations,
    }))
}

/// Pull the prose text out of a `session_entries.content` JSONB blob.
/// Mirrors the variants `EntryType::UserMessage(s)` and
/// `EntryType::AssistantMessage { content, .. }` which serialise as
/// `{"UserMessage": "…"}` and `{"AssistantMessage": {"content":
/// "…"}}` respectively. Other variants return empty.
fn entry_text(content: &serde_json::Value) -> String {
    if let Some(s) = content.get("UserMessage").and_then(|v| v.as_str()) {
        return s.to_string();
    }
    if let Some(s) = content
        .get("AssistantMessage")
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
    {
        return s.to_string();
    }
    String::new()
}
