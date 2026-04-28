//! Persistence: write `EntryExtraction` rows to
//! `session_entry_entities` and `session_entry_relations`.
//!
//! Stubbed for the skeleton commit; populated in the next step
//! alongside the joint extractor.

use sqlx::PgPool;

use super::types::EntryExtraction;
use crate::error::LoopError;

/// Persist one entry's extraction. Idempotent on the natural keys of
/// each table — re-running for the same entry+entity at the same
/// span replaces nothing (PRIMARY KEY conflict is silently ignored).
pub async fn persist_entry_extraction(
    _pool: &PgPool,
    extraction: &EntryExtraction,
) -> Result<EntryPersistReport, LoopError> {
    // Empty extraction — nothing to write. Lets callers fire the
    // pipeline for every entry without guarding on entry type.
    if extraction.entities.is_empty() && extraction.relations.is_empty() {
        return Ok(EntryPersistReport::default());
    }

    // TODO(spine #1 step C): row writes live here.
    Ok(EntryPersistReport::default())
}

#[derive(Debug, Clone, Default)]
pub struct EntryPersistReport {
    pub entity_links_written: usize,
    pub relations_written: usize,
}
