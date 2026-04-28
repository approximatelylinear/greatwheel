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
