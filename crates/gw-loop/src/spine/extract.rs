//! Joint entity + relation extraction for one session entry.
//!
//! This module is the LLM-touching half of the spine pipeline. The
//! entry's text → one chat call → JSON `{entities, relations}` →
//! canonicalised entity ids → typed edges. The persistence half lives
//! in `super::persist`.
//!
//! ### Why a separate prompt from `gw-kb::entities::extract_entities_for_chunk`
//!
//! `gw-kb` extracts entities only — that's enough for document
//! ingestion where relations between entities are inferred globally
//! from co-mention statistics by `linking.rs`. The spine wants
//! relations *as asserted in this turn* ("BM25 vs ColBERT" really
//! means `compared_with`, not just "co-mentioned"). One LLM call that
//! returns both halves keeps the entity context in working memory so
//! the relation extraction can reference the same labels — splitting
//! into two calls would either double the cost or force the second
//! call to re-extract entities.

use std::sync::Arc;

use gw_core::{EntryId, EntryType, SessionEntry};
use gw_kb::ingest::KbStores;

use super::types::EntryExtraction;
use crate::error::LoopError;

/// Owner of the LLM client + KB stores needed to extract one entry.
/// Cheap to clone (everything inside is `Arc`-shared); the
/// `ConversationLoop` holds a single `SpineExtractor` and clones it
/// into each `tokio::spawn` task.
#[derive(Clone)]
pub struct SpineExtractor {
    pub(crate) stores: Arc<KbStores>,
}

impl SpineExtractor {
    pub fn new(stores: Arc<KbStores>) -> Self {
        Self { stores }
    }

    /// Run extraction over one entry. Returns an empty
    /// `EntryExtraction` for entry types that aren't user-facing
    /// prose (code execution, host calls, repl snapshots,
    /// compactions). The persistence step is a no-op on empty
    /// extractions, so callers can fire this for every entry without
    /// guarding.
    pub async fn extract_entry(
        &self,
        entry: &SessionEntry,
    ) -> Result<EntryExtraction, LoopError> {
        let _text = match &entry.entry_type {
            EntryType::UserMessage(s) => s.as_str(),
            EntryType::AssistantMessage { content, .. } => content.as_str(),
            // Skip everything else — code, host-fn results, snapshots,
            // compaction summaries, branch summaries, system messages.
            // These are either non-prose or synthesised for the agent's
            // own bookkeeping; running an extractor over them adds
            // noise to the entity graph.
            _ => return Ok(EntryExtraction::default()),
        };

        // TODO(spine #1 step B): joint LLM extraction lives here.
        // Stub returns an empty extraction so the public surface
        // compiles and tests can mock out the LLM call. The next
        // commit fills this in with the joint entities+relations
        // prompt and parser.
        let _entry_id: EntryId = entry.id;
        // Field touch-up: keep `stores` from being dead-code-flagged
        // until step B reads from it (LLM client + canonicalisation).
        // Cheap reference; compiles to nothing.
        let _ = self.stores.pg.size();
        Ok(EntryExtraction::default())
    }
}
