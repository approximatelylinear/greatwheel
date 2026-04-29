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

use std::collections::HashMap;

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
    /// When non-null, the user has committed this segment to the
    /// workspace (Issue #5). The frontend uses it to render the
    /// "Saved" toggle state and the workspace list filters on it.
    pub committed_at: Option<DateTime<Utc>>,
    /// True when the segment has been superseded by a later
    /// resegment pass (`session_segments.invalidated_at IS NOT NULL`).
    /// We let committed segments stick around even after invalidation
    /// so the user's curated list survives churn — the workspace
    /// view tags these "(superseded)" so the user can still revisit
    /// the historical context.
    pub invalidated: bool,
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
        Option<DateTime<Utc>>,
        Option<DateTime<Utc>>,
    );
    let seg_row: Option<SegmentRow> = sqlx::query_as(
        r#"
        SELECT segment_id, session_id, label, kind, entry_first,
               entry_last, entity_ids, summary, created_at,
               committed_at, invalidated_at
        FROM session_segments
        WHERE segment_id = $1
        "#,
    )
    .bind(segment_id)
    .fetch_optional(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("fetch segment: {e}")))?;
    let Some((
        id,
        session_id,
        label,
        kind,
        entry_first,
        entry_last,
        entity_ids,
        summary,
        created_at,
        committed_at,
        invalidated_at,
    )) = seg_row
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
        committed_at,
        invalidated: invalidated_at.is_some(),
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
          AND se.entry_type IN ('user_message', 'assistant_message', 'assistant_narration')
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
                "assistant_message" | "assistant_narration" => "assistant".to_string(),
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
              AND se.entry_type IN ('user_message', 'assistant_message', 'assistant_narration')
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
              AND se.entry_type IN ('user_message', 'assistant_message', 'assistant_narration')
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

// ─── Workspace (Issue #5) ──────────────────────────────────────────

/// One committed segment as it appears in the workspace listing.
/// Lighter than `SegmentDetail` — no entries / relations payload, just
/// what the workspace card needs to render and link.
#[derive(Debug, Clone, Serialize)]
pub struct WorkspaceItem {
    pub segment_id: Uuid,
    pub label: String,
    pub kind: String,
    pub entry_first: Uuid,
    pub entry_last: Uuid,
    pub entity_count: usize,
    pub top_entities: Vec<String>,
    pub summary: Option<String>,
    pub committed_at: DateTime<Utc>,
    pub invalidated: bool,
}

/// Toggle a segment's commit state. Idempotent in both directions:
/// `commit` on an already-committed segment is a no-op, ditto
/// `uncommit` on an already-uncommitted one. Returns the resulting
/// `committed_at` value (None when uncommitted).
pub async fn set_segment_commit(
    pool: &PgPool,
    segment_id: Uuid,
    committed: bool,
) -> Result<Option<DateTime<Utc>>, LoopError> {
    let row: Option<(Option<DateTime<Utc>>,)> = if committed {
        sqlx::query_as(
            "UPDATE session_segments \
             SET committed_at = COALESCE(committed_at, now()) \
             WHERE segment_id = $1 \
             RETURNING committed_at",
        )
        .bind(segment_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| LoopError::Spine(format!("commit segment: {e}")))?
    } else {
        sqlx::query_as(
            "UPDATE session_segments SET committed_at = NULL \
             WHERE segment_id = $1 \
             RETURNING committed_at",
        )
        .bind(segment_id)
        .fetch_optional(pool)
        .await
        .map_err(|e| LoopError::Spine(format!("uncommit segment: {e}")))?
    };
    Ok(row.and_then(|(t,)| t))
}

/// All committed segments for `session_id`, newest-commit first.
/// Includes invalidated-but-still-committed rows — the workspace
/// shows them tagged so the user's curated list survives resegment
/// churn. Top entity labels are joined in for compact card rendering.
pub async fn list_workspace(
    pool: &PgPool,
    session_id: Uuid,
) -> Result<Vec<WorkspaceItem>, LoopError> {
    type Row = (
        Uuid,
        String,
        String,
        Uuid,
        Uuid,
        Vec<Uuid>,
        Option<String>,
        DateTime<Utc>,
        bool,
    );
    let rows: Vec<Row> = sqlx::query_as(
        r#"
        SELECT segment_id, label, kind, entry_first, entry_last,
               entity_ids, summary, committed_at,
               (invalidated_at IS NOT NULL) AS invalidated
        FROM session_segments
        WHERE session_id = $1 AND committed_at IS NOT NULL
        ORDER BY committed_at DESC
        "#,
    )
    .bind(session_id)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("list workspace: {e}")))?;

    if rows.is_empty() {
        return Ok(Vec::new());
    }

    // Collect all top entity ids across the workspace (cap per
    // segment) and resolve their labels in one shot, so the listing
    // doesn't fan out to N queries.
    const TOP_PER_SEGMENT: usize = 5;
    let mut all_ids: Vec<Uuid> = Vec::new();
    for r in &rows {
        for id in r.5.iter().take(TOP_PER_SEGMENT) {
            all_ids.push(*id);
        }
    }
    type LabelRow = (Uuid, String);
    let label_rows: Vec<LabelRow> = sqlx::query_as(
        "SELECT entity_id, label FROM kb_entities WHERE entity_id = ANY($1)",
    )
    .bind(&all_ids)
    .fetch_all(pool)
    .await
    .map_err(|e| LoopError::Spine(format!("workspace labels: {e}")))?;
    let labels: HashMap<Uuid, String> = label_rows.into_iter().collect();

    Ok(rows
        .into_iter()
        .map(|(seg_id, label, kind, first, last, entity_ids, summary, committed_at, invalidated)| {
            let top_entities: Vec<String> = entity_ids
                .iter()
                .take(TOP_PER_SEGMENT)
                .filter_map(|id| labels.get(id).cloned())
                .collect();
            WorkspaceItem {
                segment_id: seg_id,
                label,
                kind,
                entry_first: first,
                entry_last: last,
                entity_count: entity_ids.len(),
                top_entities,
                summary,
                committed_at,
                invalidated,
            }
        })
        .collect())
}

/// Pull the prose text out of a `session_entries.content` JSONB blob.
/// Mirrors the variants `EntryType::UserMessage(s)`,
/// `EntryType::AssistantMessage { content, .. }`, and
/// `EntryType::AssistantNarration { content }` which serialise as
/// `{"UserMessage": "…"}`, `{"AssistantMessage": {"content": "…"}}`,
/// and `{"AssistantNarration": {"content": "…"}}` respectively.
/// Other variants return empty.
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
    if let Some(s) = content
        .get("AssistantNarration")
        .and_then(|v| v.get("content"))
        .and_then(|v| v.as_str())
    {
        return s.to_string();
    }
    String::new()
}
