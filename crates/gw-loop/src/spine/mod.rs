//! Semantic spine — entry-level entity attribution + relations.
//!
//! See `docs/design-semantic-spine.md`. This module is the backend
//! half of Issue #1: extract typed entities and typed relations from
//! each `UserMessage` / `AssistantMessage` entry, canonicalise the
//! entities against `gw-kb`'s shared entity table, and write
//! per-entry rows to `session_entry_entities` and
//! `session_entry_relations`.
//!
//! Entry shape vs. chunk shape:
//!   - `gw-kb` extracts entities from *chunks* (passages of an
//!     ingested document) and writes `kb_chunk_entity_links`.
//!   - The spine extracts from *session entries* (conversation
//!     turns) and writes `session_entry_entities` +
//!     `session_entry_relations`. The link's *entity* side
//!     references the same `kb_entities` row, so a method named
//!     "BM25" mentioned in chat resolves to the same node a paper
//!     ingest would land on.
//!
//! The data flow:
//!
//! ```text
//!   entry text
//!     └─► joint LLM extract { entities, relations }
//!         └─► canonicalise entities → kb_entities row IDs
//!             └─► drop relations whose endpoints didn't survive
//!                 canonicalisation
//!                 └─► persist session_entry_entities rows
//!                     └─► persist session_entry_relations rows
//! ```
//!
//! All of this runs off the chat path via `tokio::spawn` from
//! `ConversationLoop::flush_tree`. Per-entry failures are logged and
//! skipped; the spine catches up on the next entry.

pub mod extract;
pub mod persist;
pub mod resegment;
pub mod segment;
pub mod types;

pub use extract::SpineExtractor;
pub use resegment::{resegment, ResegmentOpts, ResegmentReport, SegmentSnapshot};
pub use segment::{
    segment, ProposedSegment, SegmentEntry, SegmentOpts, DEFAULT_MAX_ENTRIES_PER_SEGMENT,
    DEFAULT_MIN_SHARED_ENTITIES,
};
pub use types::{
    EntryEntityLink, EntryExtraction, EntryRelation, RecommendedPredicate,
    RECOMMENDED_PREDICATES,
};
