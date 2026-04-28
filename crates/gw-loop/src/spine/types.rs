//! Public types for spine entry extraction.
//!
//! The wire shape produced by the extractor and consumed by the
//! persistence layer. Mirrors the SQL columns in migration 014 so the
//! mapping is direct.

use gw_core::EntryId;
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Recommended predicates the extractor prompt steers the LLM toward.
/// `predicate` is free-form `TEXT` in `session_entry_relations`, but
/// the prompt asks the model to pick from this set so we don't drown
/// in synonym variants. Out-of-vocabulary predicates are kept (we
/// don't filter the way we do for `kb_entities.kind`) — relations are
/// per-session and a stray `compared_against` instead of
/// `compared_with` is recoverable later by a one-time canonicalisation
/// pass; see design-semantic-spine.md §8.
pub const RECOMMENDED_PREDICATES: &[RecommendedPredicate] = &[
    RecommendedPredicate {
        name: "compared_with",
        directed: false,
        gloss: "two methods/datasets being compared head-to-head",
    },
    RecommendedPredicate {
        name: "tradeoff_in",
        directed: false,
        gloss: "two qualities trading against each other (recall vs precision)",
    },
    RecommendedPredicate {
        name: "composes",
        directed: true,
        gloss: "subject feeds into object (ColBERT → cross-encoder rerank)",
    },
    RecommendedPredicate {
        name: "outperforms",
        directed: true,
        gloss: "subject beats object on some metric",
    },
    RecommendedPredicate {
        name: "is_a",
        directed: true,
        gloss: "subject is a kind/instance of object",
    },
    RecommendedPredicate {
        name: "uses",
        directed: true,
        gloss: "subject incorporates / depends on object",
    },
    RecommendedPredicate {
        name: "evaluated_on",
        directed: true,
        gloss: "subject (a method) is evaluated on object (a dataset/benchmark)",
    },
];

#[derive(Debug, Clone, Copy)]
pub struct RecommendedPredicate {
    pub name: &'static str,
    pub directed: bool,
    pub gloss: &'static str,
}

/// One canonicalised entity mention, ready to be written as a
/// `session_entry_entities` row. Produced by `SpineExtractor` after
/// the joint LLM extraction + canonicalisation pass.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryEntityLink {
    pub entry_id: EntryId,
    pub entity_id: Uuid,
    pub surface: String,
    /// "introduced" | "referenced" | "decided" | "compared" | ...
    /// Free-form; the prompt steers toward those four.
    pub role: String,
    /// "mentioned" | "committed". Always "mentioned" at extraction
    /// time; promotion to "committed" is a separate user/agent action.
    pub status: String,
    pub confidence: f32,
    pub span_start: Option<i32>,
    pub span_end: Option<i32>,
}

/// One canonicalised relation between two entities, ready to be
/// written as a `session_entry_relations` row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EntryRelation {
    pub entry_id: EntryId,
    pub subject_id: Uuid,
    pub object_id: Uuid,
    pub predicate: String,
    pub directed: bool,
    pub surface: String,
    pub confidence: f32,
    pub span_start: Option<i32>,
    pub span_end: Option<i32>,
}

/// Aggregate output of one entry's extraction pass. Returned by
/// `SpineExtractor::extract_entry` and consumed by the persistence
/// layer.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EntryExtraction {
    pub entities: Vec<EntryEntityLink>,
    pub relations: Vec<EntryRelation>,
}

// ─── Pre-canonicalisation shapes (LLM output) ───────────────────────
//
// What the joint extractor in `extract::raw_extract_entry` returns
// before we canonicalise entity labels into `kb_entities` ids. The
// relation's subject/object are still strings here because the LLM
// only ever names entities, never assigns ids. Step C resolves them.

/// One entity mention from the LLM, before canonicalisation. Includes
/// the spine-specific fields (`surface`, `role`, `span_*`) that
/// `gw_kb::entities::RawEntity` doesn't carry — chat-turn extraction
/// needs to know *what the user/agent was doing with* the entity, and
/// the in-message highlight pass needs the span.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawEntryEntity {
    /// Surface form as it appeared in the entry text (or the LLM's
    /// light normalisation of it).
    pub label: String,
    /// "author" | "concept" | "method" | "dataset" | "venue" | ...
    /// Same recommended taxonomy as `gw_kb::entities::RECOMMENDED_KINDS`;
    /// out-of-vocabulary kinds are filtered post-parse.
    pub kind: String,
    /// LLM's best guess at a stable form (defaults to `label` when
    /// blank). Used for cross-mention canonicalisation.
    pub canonical_form: String,
    /// "introduced" | "referenced" | "decided" | "compared" | ...
    /// Free-form. Defaults to "referenced" when missing.
    pub role: String,
    /// What this mention got asserted as in the chat. Always
    /// "mentioned" at extraction time; the `commit_entity` host fn
    /// promotes to "committed".
    pub status: String,
    pub confidence: f32,
    pub span_start: Option<i32>,
    pub span_end: Option<i32>,
}

/// One relation between two named entities, before label →
/// entity_id resolution. Relations whose `subject` or `object` don't
/// resolve to a canonicalised entity in the same extraction get
/// dropped — better to lose the relation than to point at the wrong
/// entity (design-semantic-spine.md §4.1 step 3).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RawRelation {
    pub subject: String,
    pub object: String,
    pub predicate: String,
    pub directed: bool,
    pub surface: String,
    pub confidence: f32,
    pub span_start: Option<i32>,
    pub span_end: Option<i32>,
}

/// Aggregate of one joint LLM call: entity mentions + asserted
/// relations between them, both pre-canonicalisation.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct RawJointExtraction {
    pub entities: Vec<RawEntryEntity>,
    pub relations: Vec<RawRelation>,
}
