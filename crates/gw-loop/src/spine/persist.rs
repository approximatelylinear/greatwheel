//! Persistence: write `EntryExtraction` rows to
//! `session_entry_entities` and `session_entry_relations`.
//!
//! Both tables are idempotent on their natural keys — the entity
//! table's PK is `(entry_id, entity_id, COALESCE(span_start, -1))`,
//! so a re-run of extraction over the same entry collapses identical
//! mentions cleanly. The relation table has a UUID primary key and
//! no natural unique constraint, so re-running CAN create duplicate
//! relation rows. That's by design — relations are sparse and a
//! second run is rare; we'll add a deduper if it becomes a problem.

use sqlx::PgPool;

use super::types::EntryExtraction;
use crate::error::LoopError;

#[derive(Debug, Clone, Default)]
pub struct EntryPersistReport {
    pub entity_links_written: usize,
    pub relations_written: usize,
}

/// Persist one entry's extraction. Idempotent on
/// `session_entry_entities`'s PK; relations are append-only (the
/// table has no natural unique key — a future canonicalisation pass
/// can dedupe across `(entry_id, subject_id, object_id, predicate)`
/// if needed).
pub async fn persist_entry_extraction(
    pool: &PgPool,
    extraction: &EntryExtraction,
) -> Result<EntryPersistReport, LoopError> {
    let mut report = EntryPersistReport::default();
    if extraction.entities.is_empty() && extraction.relations.is_empty() {
        return Ok(report);
    }

    for link in &extraction.entities {
        let r = sqlx::query(
            r#"
            INSERT INTO session_entry_entities
                (entry_id, entity_id, surface, role, status, confidence,
                 span_start, span_end)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            ON CONFLICT (entry_id, entity_id, COALESCE(span_start, -1))
              DO NOTHING
            "#,
        )
        .bind(link.entry_id.0)
        .bind(link.entity_id)
        .bind(&link.surface)
        .bind(&link.role)
        .bind(&link.status)
        .bind(link.confidence)
        .bind(link.span_start)
        .bind(link.span_end)
        .execute(pool)
        .await
        .map_err(|e| LoopError::Spine(format!("persist entry-entity: {e}")))?;
        if r.rows_affected() > 0 {
            report.entity_links_written += 1;
        }
    }

    for rel in &extraction.relations {
        sqlx::query(
            r#"
            INSERT INTO session_entry_relations
                (entry_id, subject_id, object_id, predicate, directed,
                 surface, confidence, span_start, span_end)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
            "#,
        )
        .bind(rel.entry_id.0)
        .bind(rel.subject_id)
        .bind(rel.object_id)
        .bind(&rel.predicate)
        .bind(rel.directed)
        .bind(&rel.surface)
        .bind(rel.confidence)
        .bind(rel.span_start)
        .bind(rel.span_end)
        .execute(pool)
        .await
        .map_err(|e| LoopError::Spine(format!("persist entry-relation: {e}")))?;
        report.relations_written += 1;
    }

    Ok(report)
}
