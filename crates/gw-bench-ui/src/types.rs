//! Run-JSON deserialization and dashboard API return shapes.
//!
//! The `Run*` types mirror the schema written by `gw-bench`
//! (`crates/gw-bench/src/main.rs::RunRecord`). They are pure data —
//! all `Deserialize` for ingest and `Serialize` so the plugin can
//! re-emit them through host functions.
//!
//! The `Experiment*` / `Query*` / `Comparison*` types are the public
//! API the agent sees. They aggregate over the raw run shapes and add
//! computed fields (exact / fuzzy correctness, p50 latency, winner
//! classification).

use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

// -------------------------------------------------------------------------- //
// Raw run-JSON schema (mirrors `gw-bench/src/main.rs`).
// -------------------------------------------------------------------------- //

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunRecord {
    pub metadata: RunMetadata,
    #[serde(default)]
    pub query_id: Option<String>,
    #[serde(default)]
    pub tool_call_counts: HashMap<String, u32>,
    pub usage: UsageInfo,
    #[serde(default)]
    pub timing: Option<TimingInfo>,
    pub status: String,
    pub termination_reason: String,
    pub iterations_used: u32,
    #[serde(default)]
    pub retrieved_docids: Vec<String>,
    #[serde(default)]
    pub result: Vec<ResultEntry>,
    #[serde(default)]
    pub trajectory: Vec<TrajectoryMessage>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RunMetadata {
    pub model: String,
    pub llm_backend: String,
    pub llm_url: String,
    pub searcher: String,
    pub max_turns: u32,
    pub k: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageInfo {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimingInfo {
    pub total_ms: u64,
    pub bm25_ms: u64,
    pub embed_ms: u64,
    pub vector_ms: u64,
    pub llm_query_ms: u64,
    pub get_doc_ms: u64,
    pub root_llm_ms: u64,
    pub other_ms: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResultEntry {
    #[serde(rename = "type")]
    pub entry_type: String,
    #[serde(default)]
    pub tool_name: Option<String>,
    #[serde(default)]
    pub arguments: Option<serde_json::Value>,
    #[serde(default)]
    pub output: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMessage {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub code_blocks: Vec<String>,
    #[serde(default)]
    pub repl_output: Option<String>,
}

// -------------------------------------------------------------------------- //
// Gold answers.
// -------------------------------------------------------------------------- //

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoldEntry {
    pub query_id: String,
    pub query: String,
    pub answer: String,
}

/// JSONL row in `vendor/BrowseComp-Plus/data/browsecomp_plus_decrypted.jsonl`.
#[derive(Debug, Deserialize)]
pub(crate) struct RawGoldRow {
    pub query_id: serde_json::Value,
    pub query: String,
    pub answer: String,
}

// -------------------------------------------------------------------------- //
// Dashboard API return shapes.
// -------------------------------------------------------------------------- //

#[derive(Debug, Clone, Serialize)]
pub struct ExperimentRow {
    pub slug: String,
    pub n_queries: usize,
    pub exact: usize,
    pub fuzzy: usize,
    pub p50_ms: Option<u64>,
    pub mtime: DateTime<Utc>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct RunDetail {
    pub slug: String,
    pub metadata: Option<RunMetadata>,
    pub queries: Vec<QuerySummary>,
    pub aggregate: Aggregate,
    pub notes: Vec<Note>,
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QuerySummary {
    pub query_id: String,
    pub status: String,
    pub exact: bool,
    pub fuzzy: bool,
    pub iterations: u32,
    pub total_ms: Option<u64>,
    pub total_tokens: u32,
    pub termination_reason: String,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct Aggregate {
    pub n_queries: usize,
    pub exact: usize,
    pub fuzzy: usize,
    pub p50_ms: Option<u64>,
    pub p95_ms: Option<u64>,
    pub total_input_tokens: u64,
    pub total_output_tokens: u64,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryDetail {
    pub slug: String,
    pub query_id: String,
    pub question: String,
    pub gold_answer: String,
    pub agent_answer: String,
    pub exact: bool,
    pub fuzzy: bool,
    pub trace: Vec<ResultEntry>,
    pub trajectory: Vec<TrajectoryMessage>,
    pub retrieved_docids: Vec<String>,
    pub timing: Option<TimingInfo>,
    pub usage: UsageInfo,
    pub iterations: u32,
    pub termination_reason: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ConfigInfo {
    pub slug: String,
    pub toml_path: Option<String>,
    pub contents: Option<String>,
}

/// Annotation entry. Stored in the sidecar `runs/<slug>/notes.json`
/// and merged into `RunDetail` / `ExperimentRow` on load.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Note {
    pub ts: DateTime<Utc>,
    pub text: String,
}

/// Sidecar shape for `runs/<slug>/notes.json`. Both fields default to
/// empty so a partial / freshly-created file deserializes cleanly.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Annotations {
    #[serde(default)]
    pub notes: Vec<Note>,
    #[serde(default)]
    pub tags: Vec<String>,
}

#[derive(Debug, Clone, Serialize)]
pub struct Comparison {
    pub slug_a: String,
    pub slug_b: String,
    pub per_query: Vec<ComparisonRow>,
    pub deltas: ComparisonDeltas,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonRow {
    pub query_id: String,
    pub a: Option<QuerySummary>,
    pub b: Option<QuerySummary>,
    pub winner: Winner,
}

/// Outcome of a single query's a/b comparison. `BothCorrect` /
/// `BothWrong` / `Tie` distinguish ties so the UI can color them.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Winner {
    A,
    B,
    BothCorrect,
    BothWrong,
    OnlyA,
    OnlyB,
    Tie,
}

// -------------------------------------------------------------------------- //
// Difficulty matrix / query history / cost summary.
// -------------------------------------------------------------------------- //

#[derive(Debug, Clone, Serialize)]
pub struct DifficultyMatrix {
    /// Sorted union of `query_id`s appearing in any of the requested runs.
    pub queries: Vec<String>,
    /// Run slugs in input order.
    pub runs: Vec<String>,
    /// `cells[q][r]` is the cell at query `queries[q]`, run `runs[r]`.
    pub cells: Vec<Vec<MatrixCell>>,
}

/// One cell in `DifficultyMatrix`. `Error` covers runs whose status
/// was `"error"` or whose termination reason indicates a timeout —
/// i.e. the run touched the query but couldn't produce an answer.
/// `Missing` means the run did not attempt the query at all.
#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum MatrixCell {
    Exact,
    Fuzzy,
    Wrong,
    Error,
    Missing,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryHistory {
    pub query_id: String,
    pub question: String,
    pub gold_answer: String,
    pub entries: Vec<QueryHistoryEntry>,
}

#[derive(Debug, Clone, Serialize)]
pub struct QueryHistoryEntry {
    pub slug: String,
    /// `"completed"` / `"error"` / `"missing"`.
    pub status: String,
    pub exact: bool,
    pub fuzzy: bool,
    pub agent_answer: String,
    pub retrieved_docids: Vec<String>,
    pub iterations: u32,
    pub total_ms: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
pub struct CostRow {
    pub slug: String,
    pub model: String,
    pub input_tokens: u64,
    pub output_tokens: u64,
    /// Placeholder until per-model pricing config is wired through —
    /// `None` for local Ollama models, set for known OpenAI models.
    pub est_usd: Option<f64>,
    pub mtime: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize)]
pub struct ComparisonDeltas {
    pub exact_a: usize,
    pub exact_b: usize,
    pub fuzzy_a: usize,
    pub fuzzy_b: usize,
    pub p50_ms_a: Option<u64>,
    pub p50_ms_b: Option<u64>,
    pub total_tokens_a: u64,
    pub total_tokens_b: u64,
}
