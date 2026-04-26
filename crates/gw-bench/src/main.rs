//! gw-bench: Greatwheel BrowseComp-Plus benchmark using rLM-style REPL agents.
//!
//! Architecture:
//! 1. Rust host drives the rLM loop (system prompt → LLM → extract code → execute → repeat)
//! 2. Code blocks execute inside ouros ReplSession with persistent state
//! 3. External functions (search, get_document, llm_query) pause ouros and get resolved by Rust
//! 4. FINAL(answer) signals completion

mod conv_loop_runner;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use chrono::Utc;
use clap::Parser;
use ouros::Object;
use serde::{Deserialize, Serialize};
use tracing::{info, info_span, warn};

use gw_llm::{Message, OllamaClient};
use gw_memory::corpus::CorpusSearcher;
use gw_runtime::{
    extract_code_blocks, extract_final_answer, json_to_object, AgentError, HostBridge, ReplAgent,
};

// -------------------------------------------------------------------------- //
// Types matching BrowseComp-Plus output format
// -------------------------------------------------------------------------- //

#[derive(Debug, Clone, Serialize)]
struct TimingInfo {
    total_ms: u64,
    bm25_ms: u64,
    embed_ms: u64,
    vector_ms: u64,
    llm_query_ms: u64,
    get_doc_ms: u64,
    root_llm_ms: u64,
    other_ms: u64,
}

#[derive(Debug, Clone, Serialize)]
struct RunRecord {
    metadata: RunMetadata,
    query_id: Option<String>,
    tool_call_counts: HashMap<String, u32>,
    usage: UsageInfo,
    #[serde(skip_serializing_if = "Option::is_none")]
    timing: Option<TimingInfo>,
    status: String,
    /// Why the rLM loop terminated: "final_called", "max_turns", "timeout",
    /// "refusal_rejected", "llm_error", "max_turns_fallback"
    termination_reason: String,
    /// Number of rLM iterations actually used
    iterations_used: u32,
    retrieved_docids: Vec<String>,
    result: Vec<ResultEntry>,
    /// Full conversation trajectory (system + user + assistant messages)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    trajectory: Vec<TrajectoryMessage>,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct TrajectoryMessage {
    role: String,
    content: String,
    /// For assistant messages: code blocks extracted
    #[serde(skip_serializing_if = "Vec::is_empty")]
    code_blocks: Vec<String>,
    /// For assistant messages: REPL execution output
    #[serde(skip_serializing_if = "Option::is_none")]
    repl_output: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
struct RunMetadata {
    model: String,
    llm_backend: String,
    llm_url: String,
    searcher: String,
    max_turns: u32,
    k: u32,
}

#[derive(Debug, Clone, Serialize)]
struct UsageInfo {
    input_tokens: u32,
    output_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Clone, Serialize)]
pub(crate) struct ResultEntry {
    #[serde(rename = "type")]
    entry_type: String,
    tool_name: Option<String>,
    arguments: Option<serde_json::Value>,
    output: Option<serde_json::Value>,
}

// -------------------------------------------------------------------------- //
// Search types
// -------------------------------------------------------------------------- //

#[derive(Debug, Serialize, Deserialize)]
struct SearchHit {
    docid: String,
    #[allow(dead_code)]
    score: Option<f64>,
    snippet: Option<String>,
}

// -------------------------------------------------------------------------- //
// Benchmark configuration (GEPA Phase 2)
// -------------------------------------------------------------------------- //

/// All tunable parameters for the benchmark pipeline.
/// Loaded from TOML via `--config`, falls back to hardcoded defaults.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub(crate) struct BenchConfig {
    // --- Prompts ---
    /// Path to system prompt text file. If empty, uses the built-in SYSTEM_PROMPT.
    pub system_prompt_path: String,

    // --- Pre-search ---
    /// Number of initial LLM-generated search queries
    pub n_presearch_queries: usize,
    /// Max results per pre-search sub-query
    pub presearch_k: usize,
    /// Number of top snippets to analyze for PRF terms
    pub prf_top_n: usize,
    /// Number of distinctive PRF terms to extract
    pub prf_term_count: usize,
    /// Minimum word length for PRF term extraction
    pub prf_min_word_len: usize,
    /// Max word length for PRF term extraction
    pub prf_max_word_len: usize,
    /// Max results per PRF query
    pub prf_k: usize,
    /// Max documents to keep from round 1 refinement
    pub round2_keep_max: usize,
    /// Number of new queries in refinement round
    pub round2_new_queries: usize,
    /// Max results per round-2 sub-query
    pub round2_k: usize,
    /// Max round-1 results shown to LLM for refinement
    pub round1_preview_limit: usize,
    /// Chars per snippet in refinement preview
    pub round1_preview_chars: usize,
    /// Chars per snippet in context injection
    pub context_snippet_chars: usize,

    // --- BM25 boost weights ---
    pub bm25_phrase_boost: f32,
    pub bm25_slop_boost: f32,
    pub bm25_slop: u32,
    pub bm25_and_boost: f32,
    pub bm25_or_boost: f32,

    // --- rLM loop ---
    /// Per-query timeout in seconds
    pub query_timeout_secs: u64,
    /// Max LLM calls within the REPL
    pub max_llm_calls: u32,
    /// Max search calls within the REPL
    pub max_search_calls: u32,
    /// Bridge timeout in seconds
    pub bridge_timeout_secs: u64,

    // --- Output truncation ---
    /// Max chars from REPL code block execution
    pub repl_output_max_chars: usize,
    /// Max chars in fallback extraction history
    pub fallback_history_max_chars: usize,
    /// Max chars per message in fallback history
    pub fallback_msg_max_chars: usize,

    // --- Iteration prompts ---
    /// Iteration at which the early reminder kicks in
    pub nudge_reminder_start: usize,
    /// Max answer length before it's rejected as a refusal
    pub max_answer_length: usize,

    // --- S5-lite: coverage pre-search ---
    /// When true, pre-search round-1 takes top-N from each sub-query in
    /// round-robin (rank r across all queries before rank r+1). Enforces
    /// sub-fact coverage in the top-k.
    pub presearch_coverage: bool,
    /// Per-sub-query depth when presearch_coverage is true. Total budget is
    /// roughly n_presearch_queries × presearch_coverage_per_query.
    pub presearch_coverage_per_query: usize,
    /// When true AND presearch_coverage is true, skip PRF + round-2
    /// refinement entirely (pure S5-lite). When false, coverage feeds into
    /// the standard PRF + round-2 pipeline (coverage + refine hybrid).
    pub presearch_coverage_skip_refine: bool,
}

impl Default for BenchConfig {
    fn default() -> Self {
        Self {
            system_prompt_path: String::new(),
            n_presearch_queries: 5,
            presearch_k: 5,
            prf_top_n: 5,
            prf_term_count: 5,
            prf_min_word_len: 4,
            prf_max_word_len: 20,
            prf_k: 3,
            round2_keep_max: 15,
            round2_new_queries: 3,
            round2_k: 5,
            round1_preview_limit: 25,
            round1_preview_chars: 200,
            context_snippet_chars: 300,
            bm25_phrase_boost: 4.0,
            bm25_slop_boost: 2.0,
            bm25_slop: 2,
            bm25_and_boost: 1.5,
            bm25_or_boost: 1.0,
            query_timeout_secs: 180,
            max_llm_calls: 25,
            max_search_calls: 20,
            bridge_timeout_secs: 150,
            repl_output_max_chars: 8000,
            presearch_coverage: false,
            presearch_coverage_per_query: 2,
            presearch_coverage_skip_refine: true,
            fallback_history_max_chars: 15000,
            fallback_msg_max_chars: 2000,
            nudge_reminder_start: 3,
            max_answer_length: 100,
        }
    }
}

impl BenchConfig {
    fn load(path: &str) -> Result<Self, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("Failed to read config {path}: {e}"))?;
        toml::from_str(&content).map_err(|e| format!("Failed to parse config {path}: {e}"))
    }

    pub(crate) fn system_prompt(&self) -> String {
        if self.system_prompt_path.is_empty() {
            return SYSTEM_PROMPT.to_string();
        }
        match std::fs::read_to_string(&self.system_prompt_path) {
            Ok(content) => content,
            Err(e) => {
                warn!(path = self.system_prompt_path.as_str(), error = %e, "Failed to load system prompt, using default");
                SYSTEM_PROMPT.to_string()
            }
        }
    }
}

// -------------------------------------------------------------------------- //
// BrowseComp host bridge
// -------------------------------------------------------------------------- //

/// Search backend: HTTP (Python search server) or Native (in-process Rust).
#[derive(Clone)]
enum SearchBackend {
    Http {
        url: String,
        client: reqwest::Client,
    },
    Native {
        searcher: Arc<CorpusSearcher>,
    },
}

struct BrowseCompBridge {
    backend: SearchBackend,
    k: u32,
    search_mode: String, // mode string sent to HTTP backend ("bm25", "rerank", etc.)
    rerank_url: Option<String>, // Optional rerank server URL (for native backend + ColBERT reranking)
    colbert_encode_url: Option<String>, // Optional ColBERT encode server URL (for multi-vector search)
    llm: OllamaClient,
    model: String,
    /// Runtime handle for async operations. Uses a dedicated runtime when
    /// running inside ConversationLoop (to avoid nested block_on).
    rt: tokio::runtime::Handle,
    /// Dedicated runtime owned by the bridge, used when `use_dedicated_rt` is true.
    _dedicated_rt: Option<tokio::runtime::Runtime>,
    // Tracking
    tool_counts: HashMap<String, u32>,
    all_docids: Arc<std::sync::Mutex<Vec<String>>>,
    total_input_tokens: u32,
    total_output_tokens: u32,
    // Limits
    max_llm_calls: u32,
    max_search_calls: u32,
    start_time: std::time::Instant,
    timeout_secs: u64,
    // Timing (cumulative milliseconds)
    timing_bm25_ms: u64,
    timing_embed_ms: u64,
    timing_vector_ms: u64,
    timing_llm_ms: u64,
    timing_get_doc_ms: u64,
}

impl BrowseCompBridge {
    fn new(
        backend: SearchBackend,
        k: u32,
        search_mode: String,
        rerank_url: Option<String>,
        colbert_encode_url: Option<String>,
        llm: OllamaClient,
        model: String,
        rt: tokio::runtime::Handle,
        docid_tracker: Arc<std::sync::Mutex<Vec<String>>>,
    ) -> Self {
        Self {
            backend,
            k,
            search_mode,
            rerank_url,
            colbert_encode_url,
            _dedicated_rt: None,
            llm,
            model,
            rt,
            tool_counts: HashMap::new(),
            all_docids: docid_tracker,
            total_input_tokens: 0,
            total_output_tokens: 0,
            max_llm_calls: 25,    // overridden by BenchConfig when used
            max_search_calls: 20, // overridden by BenchConfig when used
            start_time: std::time::Instant::now(),
            timeout_secs: 150, // overridden by BenchConfig when used
            timing_bm25_ms: 0,
            timing_embed_ms: 0,
            timing_vector_ms: 0,
            timing_llm_ms: 0,
            timing_get_doc_ms: 0,
        }
    }

    /// Switch the bridge to use a dedicated tokio runtime for its async operations.
    /// Required when running inside ConversationLoop (which has its own runtime).
    fn use_dedicated_runtime(&mut self) {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .expect("failed to create dedicated runtime for bridge");
        self.rt = rt.handle().clone();
        self._dedicated_rt = Some(rt);
    }

    fn is_timed_out(&self) -> bool {
        self.start_time.elapsed().as_secs() > self.timeout_secs
    }

    #[tracing::instrument(name = "host_function", skip(self), fields(function = "search", gw.mode = mode, gw.hits))]
    fn search_with_mode(&mut self, query: &str, mode: &str) -> Result<Object, AgentError> {
        if self.is_timed_out() {
            return Ok(Object::List(vec![]));
        }
        let count = self.tool_counts.entry("search".into()).or_insert(0);
        *count += 1;
        if *count > self.max_search_calls {
            return Ok(Object::List(vec![]));
        }

        let k = self.k as usize;
        let query_str = query.to_string();
        let mode_str = mode.to_string();

        let hits: Vec<SearchHit> = match &self.backend {
            SearchBackend::Http { url, client } => {
                let url = format!("{url}/call/search");
                let client = client.clone();
                self.rt.block_on(async {
                    let resp = client
                        .post(&url)
                        .json(&serde_json::json!({ "query": query_str, "k": k, "mode": mode_str }))
                        .send()
                        .await
                        .map_err(|e| AgentError::HostFunction {
                            function: "search".into(),
                            message: format!("{e}"),
                        })?;

                    if !resp.status().is_success() {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(AgentError::HostFunction {
                            function: "search".into(),
                            message: format!("HTTP {body}"),
                        });
                    }

                    resp.json().await.map_err(|e| AgentError::HostFunction {
                        function: "search".into(),
                        message: format!("{e}"),
                    })
                })?
            }
            SearchBackend::Native { searcher } => {
                let searcher = searcher.clone();
                let llm = self.llm.clone();
                let rerank_url = self.rerank_url.clone();
                let (corpus_hits, embed_ms, vector_ms, bm25_ms) = self.rt.block_on(async {
                    let mut embed_ms = 0u64;
                    let mut vector_ms = 0u64;
                    let mut bm25_ms = 0u64;
                    let corpus_hits = match mode_str.as_str() {
                        "vector" => {
                            let t0 = std::time::Instant::now();
                            let vecs = llm.embed(std::slice::from_ref(&query_str)).await.map_err(|e| {
                                AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("embed error: {e}"),
                                }
                            })?;
                            embed_ms = t0.elapsed().as_millis() as u64;
                            let vec = vecs.into_iter().next().unwrap_or_default();
                            let t1 = std::time::Instant::now();
                            let hits = searcher.search_vector(vec, k).await.map_err(|e| {
                                AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("{e}"),
                                }
                            })?;
                            vector_ms = t1.elapsed().as_millis() as u64;
                            hits
                        }
                        "hybrid" => {
                            let t0 = std::time::Instant::now();
                            let vecs = llm.embed(std::slice::from_ref(&query_str)).await.map_err(|e| {
                                AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("embed error: {e}"),
                                }
                            })?;
                            embed_ms = t0.elapsed().as_millis() as u64;
                            let vec = vecs.into_iter().next().unwrap_or_default();
                            let t1 = std::time::Instant::now();
                            let hits = searcher.search_hybrid(&query_str, vec, k).await.map_err(|e| {
                                AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("{e}"),
                                }
                            })?;
                            // hybrid internally does BM25 + vector, but we measure the whole thing
                            vector_ms = t1.elapsed().as_millis() as u64;
                            hits
                        }
                        "colbert" => {
                            // ColBERT-only: encode query tokens, then LanceDB MaxSim search
                            // Widen retrieval when reranking (reranker refines ANN approximation)
                            let retrieve_k = if rerank_url.is_some() { std::cmp::max(k, 200) } else { k };
                            let colbert_url = self.colbert_encode_url.clone();
                            let token_vecs = encode_colbert_query(&colbert_url, &query_str).await
                                .map_err(|e| AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("ColBERT encode error: {e}"),
                                })?;
                            let t1 = std::time::Instant::now();
                            let hits = searcher.search_colbert(&token_vecs, retrieve_k, &query_str).await
                                .map_err(|e| AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("ColBERT search error: {e}"),
                                })?;
                            vector_ms = t1.elapsed().as_millis() as u64;
                            hits
                        }
                        "hybrid_colbert" => {
                            // BM25 + ColBERT fusion via RRF
                            let colbert_url = self.colbert_encode_url.clone();
                            let token_vecs = encode_colbert_query(&colbert_url, &query_str).await
                                .map_err(|e| AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("ColBERT encode error: {e}"),
                                })?;
                            let t1 = std::time::Instant::now();
                            let hits = searcher.search_hybrid_colbert(&query_str, &token_vecs, k).await
                                .map_err(|e| AgentError::HostFunction {
                                    function: "search".into(),
                                    message: format!("ColBERT hybrid search error: {e}"),
                                })?;
                            // Hybrid has both BM25 and vector time mixed
                            bm25_ms = t1.elapsed().as_millis() as u64;
                            hits
                        }
                        _ => {
                            // Default to boosted BM25 — retrieve more if reranking
                            let retrieve_k = if rerank_url.is_some() { std::cmp::max(k, 200) } else { k };
                            let t0 = std::time::Instant::now();
                            // Use passage+doc RRF when passage index is available
                            let hits = if searcher.has_passage_index() {
                                searcher.search_with_passages(&query_str, retrieve_k).map_err(|e| {
                                    AgentError::HostFunction {
                                        function: "search".into(),
                                        message: format!("{e}"),
                                    }
                                })?
                            } else {
                                searcher.search_bm25_boosted(&query_str, retrieve_k).map_err(|e| {
                                    AgentError::HostFunction {
                                        function: "search".into(),
                                        message: format!("{e}"),
                                    }
                                })?
                            };
                            bm25_ms = t0.elapsed().as_millis() as u64;
                            hits
                        }
                    };
                    let mut hits: Vec<SearchHit> = corpus_hits
                        .into_iter()
                        .map(|h| SearchHit {
                            docid: h.docid,
                            score: Some(h.score as f64),
                            snippet: Some(h.text),
                        })
                        .collect();

                    // If rerank URL is set, send candidates to ColBERT for reranking
                    if let Some(ref url) = rerank_url {
                        let rerank_endpoint = format!("{url}/rerank");

                        // For ColBERT retrieval mode with passage index:
                        // expand each docid into ALL its passages so the reranker
                        // can pick the most relevant passage (not just BM25's best)
                        let docs_json: Vec<serde_json::Value> = if mode_str == "colbert" && searcher.has_passage_index() {
                            let mut passage_docs = Vec::new();
                            for h in hits.iter().take(20) { // top-20 docs, expand to passages
                                let passages = searcher.all_passages_for_docid(&h.docid);
                                if passages.is_empty() {
                                    // Fallback: use the snippet
                                    passage_docs.push(serde_json::json!({
                                        "docid": h.docid,
                                        "text": h.snippet.as_deref().unwrap_or(""),
                                    }));
                                } else {
                                    for passage in passages {
                                        passage_docs.push(serde_json::json!({
                                            "docid": h.docid,
                                            "text": passage,
                                        }));
                                    }
                                }
                            }
                            tracing::info!(n_passages = passage_docs.len(), n_docs = 20, "Expanded docs to passages for reranking");
                            passage_docs
                        } else {
                            hits.iter().map(|h| {
                                serde_json::json!({
                                    "docid": h.docid,
                                    "text": h.snippet.as_deref().unwrap_or(""),
                                })
                            }).collect()
                        };
                        let client = reqwest::Client::new();
                        match client.post(&rerank_endpoint)
                            .json(&serde_json::json!({
                                "query": query_str,
                                "documents": docs_json,
                                "k": k,
                            }))
                            .send()
                            .await
                        {
                            Ok(resp) if resp.status().is_success() => {
                                if let Ok(reranked) = resp.json::<Vec<serde_json::Value>>().await {
                                    hits = reranked.iter().map(|r| SearchHit {
                                        docid: r["docid"].as_str().unwrap_or("").to_string(),
                                        score: r["score"].as_f64(),
                                        snippet: r["text"].as_str().map(|s| s.to_string()),
                                    }).collect();
                                }
                            }
                            Ok(resp) => {
                                tracing::warn!(status = %resp.status(), "Rerank server error, using BM25 order");
                                hits.truncate(k);
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "Rerank server unreachable, using BM25 order");
                                hits.truncate(k);
                            }
                        }
                    }

                    Ok::<_, AgentError>((hits, embed_ms, vector_ms, bm25_ms))
                })?;
                self.timing_embed_ms += embed_ms;
                self.timing_vector_ms += vector_ms;
                self.timing_bm25_ms += bm25_ms;
                corpus_hits
            }
        };

        tracing::Span::current().record("gw.hits", hits.len() as u64);
        for hit in &hits {
            if let Ok(mut docids) = self.all_docids.lock() {
                docids.push(hit.docid.clone());
            }
        }

        let results: Vec<serde_json::Value> = hits
            .iter()
            .map(|h| {
                serde_json::json!({
                    "docid": h.docid,
                    "snippet": h.snippet.as_deref().unwrap_or(""),
                })
            })
            .collect();

        Ok(json_to_object(serde_json::Value::Array(results)))
    }

    fn search(&mut self, query: &str) -> Result<Object, AgentError> {
        let mode = self.search_mode.clone();
        self.search_with_mode(query, &mode)
    }

    fn vector_search(&mut self, query: &str) -> Result<Object, AgentError> {
        self.search_with_mode(query, "vector")
    }

    #[tracing::instrument(name = "host_function", skip(self), fields(function = "get_document"))]
    fn get_document(&mut self, docid: &str) -> Result<Object, AgentError> {
        let t0 = std::time::Instant::now();
        if self.is_timed_out() {
            return Ok(Object::String("[timeout]".into()));
        }
        *self.tool_counts.entry("get_document".into()).or_insert(0) += 1;

        let docid = docid.to_string();

        let text: String = match &self.backend {
            SearchBackend::Http { url, client } => {
                let url = format!("{url}/call/get_document");
                let client = client.clone();
                self.rt.block_on(async {
                    let resp = client
                        .post(&url)
                        .json(&serde_json::json!({ "docid": docid }))
                        .send()
                        .await
                        .map_err(|e| AgentError::HostFunction {
                            function: "get_document".into(),
                            message: format!("{e}"),
                        })?;

                    if !resp.status().is_success() {
                        let body = resp.text().await.unwrap_or_default();
                        return Err(AgentError::HostFunction {
                            function: "get_document".into(),
                            message: format!("HTTP {body}"),
                        });
                    }

                    resp.text().await.map_err(|e| AgentError::HostFunction {
                        function: "get_document".into(),
                        message: format!("{e}"),
                    })
                })?
            }
            SearchBackend::Native { searcher } => searcher
                .get_document(&docid)
                .map_err(|e| AgentError::HostFunction {
                    function: "get_document".into(),
                    message: format!("{e}"),
                })?
                .unwrap_or_else(|| format!("[document '{docid}' not found]")),
        };

        self.timing_get_doc_ms += t0.elapsed().as_millis() as u64;
        Ok(Object::String(text))
    }

    #[tracing::instrument(name = "host_function", skip(self, prompt), fields(function = "llm_query", gw.input_tokens, gw.output_tokens))]
    fn llm_query(&mut self, prompt: &str) -> Result<Object, AgentError> {
        let t0 = std::time::Instant::now();
        if self.is_timed_out() {
            return Ok(Object::String(
                "[timeout — provide your best answer now]".into(),
            ));
        }
        let count = self.tool_counts.entry("llm_query".into()).or_insert(0);
        *count += 1;
        if *count > self.max_llm_calls {
            return Ok(Object::String(
                "[llm_query limit reached — provide your best answer now]".into(),
            ));
        }

        let messages = vec![Message {
            role: "user".into(),
            content: prompt.to_string(),
        }];
        let llm = self.llm.clone();
        let model = self.model.clone();

        let resp = self.rt.block_on(async {
            llm.chat(&messages, Some(&model))
                .await
                .map_err(|e| AgentError::HostFunction {
                    function: "llm_query".into(),
                    message: format!("{e}"),
                })
        })?;

        self.total_input_tokens += resp.input_tokens.unwrap_or(0);
        self.total_output_tokens += resp.output_tokens.unwrap_or(0);
        tracing::Span::current().record("gw.input_tokens", resp.input_tokens.unwrap_or(0) as u64);
        tracing::Span::current().record("gw.output_tokens", resp.output_tokens.unwrap_or(0) as u64);

        self.timing_llm_ms += t0.elapsed().as_millis() as u64;
        Ok(Object::String(resp.content))
    }

    /// Batch LLM query — runs multiple prompts in parallel and returns a list of responses.
    #[tracing::instrument(name = "host_function", skip(self, prompts), fields(function = "batch_llm_query", gw.batch_size = prompts.len()))]
    fn batch_llm_query(&mut self, prompts: Vec<String>) -> Result<Object, AgentError> {
        if self.is_timed_out() {
            return Ok(Object::List(vec![Object::String("[timeout]".into())]));
        }
        let n = prompts.len();
        let count = self.tool_counts.entry("llm_query".into()).or_insert(0);
        *count += n as u32;

        let llm = self.llm.clone();
        let model = self.model.clone();

        let results: Vec<gw_llm::CompletionResponse> = self.rt.block_on(async {
            let mut handles = Vec::with_capacity(n);
            for prompt in prompts {
                let llm = llm.clone();
                let model = model.clone();
                handles.push(tokio::spawn(async move {
                    let messages = vec![Message {
                        role: "user".into(),
                        content: prompt,
                    }];
                    llm.chat(&messages, Some(&model)).await
                }));
            }

            let mut responses = Vec::with_capacity(n);
            for handle in handles {
                match handle.await {
                    Ok(Ok(resp)) => responses.push(resp),
                    Ok(Err(e)) => responses.push(gw_llm::CompletionResponse {
                        content: format!("[error: {e}]"),
                        model: None,
                        input_tokens: None,
                        output_tokens: None,
                    }),
                    Err(e) => responses.push(gw_llm::CompletionResponse {
                        content: format!("[error: {e}]"),
                        model: None,
                        input_tokens: None,
                        output_tokens: None,
                    }),
                }
            }
            responses
        });

        for resp in &results {
            self.total_input_tokens += resp.input_tokens.unwrap_or(0);
            self.total_output_tokens += resp.output_tokens.unwrap_or(0);
        }

        let response_strings: Vec<Object> = results
            .into_iter()
            .map(|r| Object::String(r.content))
            .collect();

        Ok(Object::List(response_strings))
    }
}

impl HostBridge for BrowseCompBridge {
    fn call(
        &mut self,
        function: &str,
        args: Vec<serde_json::Value>,
        kwargs: HashMap<String, serde_json::Value>,
    ) -> Result<Object, AgentError> {
        match function {
            "search" => {
                let query = args
                    .first()
                    .and_then(|v| v.as_str())
                    .or_else(|| kwargs.get("query").and_then(|v| v.as_str()))
                    .ok_or_else(|| AgentError::HostFunction {
                        function: "search".into(),
                        message: "missing 'query' argument".into(),
                    })?;
                self.search(query)
            }
            "vector_search" => {
                let query = args
                    .first()
                    .and_then(|v| v.as_str())
                    .or_else(|| kwargs.get("query").and_then(|v| v.as_str()))
                    .ok_or_else(|| AgentError::HostFunction {
                        function: "vector_search".into(),
                        message: "missing 'query' argument".into(),
                    })?;
                self.vector_search(query)
            }
            "get_document" => {
                let docid = args
                    .first()
                    .and_then(|v| v.as_str())
                    .or_else(|| kwargs.get("docid").and_then(|v| v.as_str()))
                    .ok_or_else(|| AgentError::HostFunction {
                        function: "get_document".into(),
                        message: "missing 'docid' argument".into(),
                    })?;
                self.get_document(docid)
            }
            "llm_query" => {
                let prompt = args
                    .first()
                    .and_then(|v| v.as_str())
                    .or_else(|| kwargs.get("prompt").and_then(|v| v.as_str()))
                    .ok_or_else(|| AgentError::HostFunction {
                        function: "llm_query".into(),
                        message: "missing 'prompt' argument".into(),
                    })?;
                self.llm_query(prompt)
            }
            "batch_llm_query" => {
                let prompts: Vec<String> = args
                    .first()
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str().map(|s| s.to_string()))
                            .collect()
                    })
                    .unwrap_or_default();
                if prompts.is_empty() {
                    return Err(AgentError::HostFunction {
                        function: "batch_llm_query".into(),
                        message: "missing list of prompts".into(),
                    });
                }
                self.batch_llm_query(prompts)
            }
            _ => Err(AgentError::UnknownFunction(function.into())),
        }
    }
}

// -------------------------------------------------------------------------- //
// FactRegistry — injected into ouros REPL before the rLM loop starts.
// Single source of truth: bench/browsecomp/fact_registry.py
// Embedded at compile time via include_str! to avoid duplication.
// -------------------------------------------------------------------------- //

const FACT_REGISTRY_BOOTSTRAP: &str = include_str!("../../../bench/browsecomp/fact_registry.py");

const ENTITY_SEARCH_BOOTSTRAP: &str = include_str!("../../../bench/browsecomp/entity_search.py");

// -------------------------------------------------------------------------- //
// rLM System Prompt & Iteration Prompts
// -------------------------------------------------------------------------- //

const SYSTEM_PROMPT: &str = r#"You are tasked with answering a query by searching and analyzing a corpus of ~100K web documents. You work interactively in a REPL environment that persists state across code blocks.

TOOLS AVAILABLE:
- `context` — pre-loaded list of {docid, snippet} dicts from initial searches
- `question` — the query string you need to answer
- `answer_type` — expected answer type (person, place, date, number, title, etc.)
- `search(query)` — BM25 keyword search. Use 2-5 specific nouns/names/numbers. Returns list of {docid, snippet} dicts
- `get_document(docid)` — retrieve full document text. ALWAYS READ FULL DOCUMENTS before answering.
- `llm_query(prompt)` — sub-LLM analysis (~10K char context). YOUR MOST POWERFUL TOOL for extracting specific facts.
- `batch_llm_query([p1, p2, ...])` — parallel LLM queries (use to analyze multiple docs at once)
- `facts` — FactRegistry for structured evidence tracking (see FACT TRACKING below)
- `entity_search(entity, hops=1, k=5)` — search for an entity AND entities that co-occur with it (one-hop chain). Returns deduplicated list of {docid, score, snippet} dicts. Use when a direct search misses but you know a related entity.
- `print()` — view output to continue reasoning
- `FINAL("answer")` — submit your final answer. Answer must be a precise name, number, date, or short phrase.

FACT TRACKING (use `facts` to organize evidence and track candidates):
  facts.add(text, source="docid")          — Record extracted facts (auto-extracts entities)
  facts.propose("answer", confidence=0.7, evidence="reason")  — Propose a candidate answer
  facts.reinforce("answer", evidence="new supporting evidence")  — Strengthen a candidate (+0.15)
  facts.contradict("answer", reason="why wrong")  — Weaken a candidate (-0.25)
  facts.summary()          — View all facts grouped by entity
  facts.candidates()       — View ranked candidates by confidence
  facts.best_candidate()   — Get (answer, confidence) of top candidate
  facts.for_entity("Name") — All facts mentioning an entity
  facts.entities()         — All discovered entities sorted by frequency

Write Python code in ```repl``` blocks. Variables persist between blocks.

CRITICAL: The answer MUST come from the documents, not from your own knowledge. You are searching a specific corpus — the answer is IN the documents. Read them carefully.

WORKFLOW (follow this order):
1. EXPLORE — examine context snippets. Identify 3+ promising documents.
2. READ — use get_document() to load the full text of promising documents. Snippets are NOT enough.
3. EXTRACT — use llm_query() to extract specific facts, then store them:
   extracted = llm_query(f"Question: {question}\n\nRead this document and extract any facts relevant to answering the question. Quote exact names, dates, and numbers.\n\nDocument:\n{doc[:8000]}")
   facts.add(extracted, source=docid)
4. PROPOSE — when you find a plausible answer, register it:
   facts.propose("candidate answer", confidence=0.6, evidence="found in docid")
5. SEARCH MORE — if you haven't found the answer, search with COMPLETELY DIFFERENT keywords. Try:
   - Entities from facts.entities() — discovered names, dates, places
   - Synonyms and alternate phrasings
   - Dates, numbers, locations mentioned in the query
6. VERIFY — search for your candidate answer BY NAME. Reinforce or contradict:
   facts.reinforce("candidate", evidence="confirmed in doc X")
   facts.contradict("candidate", reason="doc Y says otherwise")
7. SUBMIT — call FINAL() with your best candidate:
   best = facts.best_candidate()
   FINAL(best[0] if best else "your answer")

SEARCH STRATEGY:
- BM25 search matches keywords. Use 2-5 distinctive nouns, names, or numbers.
- NEVER repeat a similar search. Each search must target different information.
- When stuck, search for entities DISCOVERED in documents, not just from the query.
- For MULTI-HOP questions: use entity_search("discovered entity") to automatically follow entity chains. It searches for the entity AND co-occurring entities in one call.

COMMON MISTAKES TO AVOID:
- Don't guess from snippets alone — READ FULL DOCUMENTS with get_document().
- Don't rely on your own knowledge — the answer must come from the corpus.
- Don't submit without verifying — search for your candidate answer to confirm.
- Don't pick a well-known answer when the document mentions a different, specific one.

EXAMPLE:
```repl
# Step 1: Scan context for promising documents
for i, h in enumerate(context[:8]):
    print(f"{i}: {h['docid']} — {h['snippet'][:150]}")
```
```repl
# Step 2: Read top documents and extract facts
docs = [get_document(context[i]["docid"]) for i in [0, 2, 4]]
docids = [context[i]["docid"] for i in [0, 2, 4]]
prompts = [f"Question: {question}\n\nExtract ALL facts relevant to this question. Quote exact names, dates, numbers.\n\nDocument:\n{d[:8000]}" for d in docs]
evidence = batch_llm_query(prompts)
for i, e in enumerate(evidence):
    facts.add(e, source=docids[i])
    print(f"=== Doc {i} ===\n{e}\n")
# Propose candidate if we found one
facts.propose("discovered answer", confidence=0.6, evidence=f"found in {docids[0]}")
```
```repl
# Step 3: Search for entities discovered in the documents
top_entities = facts.entities()[:3]
for ent in top_entities:
    hits2 = search(ent)
    for h in hits2[:2]:
        doc = get_document(h["docid"])
        extracted = llm_query(f"Question: {question}\n\nExtract relevant facts.\n\nDocument:\n{doc[:8000]}")
        facts.add(extracted, source=h["docid"])
print(facts.summary())
```
```repl
# Step 4: Verify and submit best candidate
print(facts.candidates())
best = facts.best_candidate()
if best:
    FINAL(best[0])
```

RULES:
- NEVER answer "Unable to determine". Always give your BEST GUESS from the documents.
- NEVER repeat a search query. Each search must use different keywords.
- ALWAYS read at least 2 full documents before submitting an answer.
- ALWAYS use facts.add() to store extracted evidence — don't lose findings in ad-hoc variables.
- ALWAYS use facts.propose() when you find a plausible answer. Use facts.reinforce()/contradict() as evidence accumulates.
- You MUST call FINAL("answer") before running out of iterations. A wrong answer is better than no answer.
- After iteration 4, you should have a candidate. Use facts.best_candidate() and call FINAL().

/no_think"#;

pub(crate) fn iteration_prompt(
    query: &str,
    iteration: usize,
    max_iterations: usize,
    variables_info: &str,
) -> String {
    let counter = format!("[Iteration {}/{}]", iteration + 1, max_iterations);

    if iteration == 0 {
        format!(
            "{counter} EXPLORE FIRST — you have not seen your context yet.\n\n\
             Query: \"{query}\"\n\n\
             Start by examining the `context` variable (pre-loaded search results). \
             Pick the 2-3 most promising documents and load them with get_document(). \
             Use llm_query() to extract relevant facts. Do NOT submit a final answer yet.\n\n\
             Write a ```repl``` code block:"
        )
    } else {
        let vars_section = if variables_info.is_empty() {
            String::new()
        } else {
            format!("\n\nVariables in scope:\n{variables_info}\n")
        };
        let nudge = if iteration >= max_iterations - 2 {
            "\n⚠️ LAST CHANCE — you MUST call FINAL() in this code block. \
             Check facts.best_candidate() and submit it. Do NOT search again. \
             Do NOT say unable to determine. Submit NOW:\n\
             best = facts.best_candidate()\n\
             FINAL(best[0] if best else \"your best guess\")"
        } else if iteration >= max_iterations / 2 {
            "\n⚠️ HALFWAY — check facts.candidates(). If you have a candidate, call FINAL() now. \
             If not, do ONE more targeted search, then FINAL() in the next iteration. \
             Do NOT waste iterations — submit as soon as you have any plausible answer."
        } else if iteration >= 3 {
            "\nREMINDER: Each iteration is expensive. Check facts.candidates(). \
             If you have a plausible answer, call FINAL() now. A good guess beats no answer."
        } else {
            ""
        };
        format!(
            "{counter} Continue investigating: \"{query}\"\n\
             {vars_section}\n\
             What is your current best candidate answer? If you have one, call FINAL(\"answer\"). \
             Otherwise, do ONE focused search or document read, then submit.{nudge}\n\n\
             Write a ```repl``` code block:"
        )
    }
}

pub(crate) fn final_prompt(query: &str, variables_info: &str) -> String {
    let vars_section = if variables_info.is_empty() {
        String::new()
    } else {
        format!("\n\nVariables in scope:\n{variables_info}\n")
    };
    format!(
        "[FINAL ITERATION] You MUST call FINAL() NOW. No more searching.\n\n\
         Query: \"{query}\"\n\
         {vars_section}\n\
         Write ONLY this:\n\
         ```repl\n\
         best = facts.best_candidate()\n\
         FINAL(best[0] if best else \"your best guess\")\n\
         ```\n\
         If facts has no candidates, replace with a specific name, number, date, or short phrase. \
         NEVER say unable to determine. Give your BEST GUESS."
    )
}

/// Build a summary of REPL variables (DSPy-style metadata preview).
/// Shows type, length, and a short preview — NOT full content.
pub(crate) fn build_variables_info(agent: &ReplAgent) -> String {
    let var_names = [
        "context",
        "evidence",
        "answer",
        "doc",
        "doc1",
        "doc2",
        "doc3",
        "docs",
        "results",
        "hits",
        "response",
        "info",
        "data",
        "text",
        "combined",
        "analysis",
        "findings",
        "candidates",
        "best",
    ];
    let mut lines = Vec::new();
    for name in &var_names {
        if let Some(val) = agent.get_variable(name) {
            let json_val = gw_runtime::object_to_json(&val);
            let (type_str, length, preview) = describe_value(&json_val);
            lines.push(format!("  {name}: {type_str}(len={length}) = {preview}"));
        }
    }
    lines.join("\n")
}

fn describe_value(val: &serde_json::Value) -> (&'static str, usize, String) {
    match val {
        serde_json::Value::String(s) => {
            let preview: String = s.chars().take(80).collect();
            let suffix = if s.len() > 80 { "..." } else { "" };
            ("str", s.len(), format!("\"{preview}{suffix}\""))
        }
        serde_json::Value::Array(arr) => {
            let preview = if arr.is_empty() {
                "[]".to_string()
            } else {
                let first = serde_json::to_string(&arr[0]).unwrap_or_default();
                let first_short: String = first.chars().take(60).collect();
                format!("[{first_short}..., ...]")
            };
            ("list", arr.len(), preview)
        }
        serde_json::Value::Number(n) => ("num", 1, n.to_string()),
        serde_json::Value::Bool(b) => ("bool", 1, b.to_string()),
        serde_json::Value::Null => ("null", 0, "None".to_string()),
        serde_json::Value::Object(m) => {
            let keys: Vec<&String> = m.keys().take(5).collect();
            (
                "dict",
                m.len(),
                format!(
                    "{{{}}}",
                    keys.iter()
                        .map(|k| k.as_str())
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
            )
        }
    }
}

/// Fallback extraction: if max_iterations reached without FINAL, ask LLM to extract answer
/// from the accumulated conversation history (DSPy-style extract_sig).
pub(crate) fn fallback_extract(
    llm: &OllamaClient,
    model: &str,
    query: &str,
    messages: &[Message],
    rt: &tokio::runtime::Handle,
) -> (Option<String>, u32, u32) {
    // Build a condensed history for the extraction prompt
    let mut history = String::new();
    for msg in messages.iter().skip(1) {
        // skip system prompt
        let role = &msg.role;
        let content_preview: String = msg.content.chars().take(2000).collect();
        history.push_str(&format!("[{role}]: {content_preview}\n\n"));
    }
    // Cap total history
    let history: String = history.chars().take(15000).collect();

    let extract_prompt = format!(
        "Extract the answer from this research session.\n\n\
         Query: \"{query}\"\n\n\
         Research history:\n{history}\n\n\
         Reply with ONLY the answer — a specific name, number, date, or short phrase.\n\
         Rules:\n\
         - ONE answer only, no explanation\n\
         - NEVER say \"unable to determine\", \"not found\", \"insufficient\", or similar\n\
         - If unsure, pick the MOST LIKELY candidate mentioned in the research\n\
         - The answer is a short factual phrase (1-5 words)\n\n/no_think"
    );

    let msgs = vec![Message {
        role: "user".into(),
        content: extract_prompt,
    }];

    match rt.block_on(async { llm.chat_with_options(&msgs, Some(model), Some(false)).await }) {
        Ok(resp) => {
            let answer = strip_think_tags(resp.content.trim());
            let input = resp.input_tokens.unwrap_or(0);
            let output = resp.output_tokens.unwrap_or(0);
            if answer.is_empty() {
                (None, input, output)
            } else {
                (Some(answer), input, output)
            }
        }
        Err(e) => {
            warn!(error = %e, "Fallback extraction failed");
            (None, 0, 0)
        }
    }
}

// -------------------------------------------------------------------------- //
// rLM Loop
// -------------------------------------------------------------------------- //

/// Strip `<think>...</think>` blocks from LLM output (qwen3.5 thinking mode).
pub(crate) fn strip_think_tags(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(start) = rest.find("<think>") {
        result.push_str(&rest[..start]);
        if let Some(end) = rest[start..].find("</think>") {
            rest = &rest[start + end + "</think>".len()..];
        } else {
            // Unclosed <think> — skip everything after it
            return result.trim().to_string();
        }
    }
    result.push_str(rest);
    result.trim().to_string()
}

/// Check if an answer is a refusal/hedge that should be rejected.
pub(crate) fn is_refusal_answer(answer: &str) -> bool {
    let lower = answer.to_lowercase();
    // Reject if answer is too long — real answers are short
    // Note: threshold configurable via BenchConfig.max_answer_length (default 100)
    if answer.len() > 100 {
        return true;
    }
    let refusal_phrases = [
        "unable to",
        "not found",
        "insufficient",
        "cannot determine",
        "could not find",
        "could not determine",
        "no exact match",
        "no specific",
        "no matching",
        "not enough information",
        "based on the research",
        "based on the provided",
        "n/a",
        "best candidate based on",
    ];
    refusal_phrases.iter().any(|phrase| lower.contains(phrase))
}

/// Encode a query into ColBERT token vectors via the encode server.
async fn encode_colbert_query(url: &Option<String>, query: &str) -> Result<Vec<Vec<f32>>, String> {
    let url = url
        .as_ref()
        .ok_or_else(|| "--colbert-encode-url required for ColBERT search".to_string())?;
    let client = reqwest::Client::new();
    let resp = client
        .post(format!("{url}/encode"))
        .json(&serde_json::json!({"text": query}))
        .send()
        .await
        .map_err(|e| format!("ColBERT encode request failed: {e}"))?;

    if !resp.status().is_success() {
        let body = resp.text().await.unwrap_or_default();
        return Err(format!("ColBERT encode server error: {body}"));
    }

    let body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| format!("ColBERT encode parse error: {e}"))?;

    let tokens = body["tokens"]
        .as_array()
        .ok_or_else(|| "ColBERT response missing 'tokens' array".to_string())?;

    let token_vecs: Vec<Vec<f32>> = tokens
        .iter()
        .map(|t| {
            t.as_array()
                .unwrap_or(&vec![])
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .collect();

    Ok(token_vecs)
}

/// Run a BM25 search via the backend, returning SearchHit results.
fn backend_search(
    backend: &SearchBackend,
    query: &str,
    k: usize,
    rt: &tokio::runtime::Handle,
) -> Vec<SearchHit> {
    match backend {
        SearchBackend::Http { url, client } => {
            let url = format!("{url}/call/search");
            let client = client.clone();
            let query = query.to_string();
            let resp = rt.block_on(async {
                client
                    .post(&url)
                    .json(&serde_json::json!({ "query": query, "k": k, "mode": "bm25" }))
                    .send()
                    .await
            });
            if let Ok(resp) = resp {
                rt.block_on(async { resp.json::<Vec<SearchHit>>().await })
                    .unwrap_or_default()
            } else {
                vec![]
            }
        }
        SearchBackend::Native { searcher } => match searcher.search_bm25_boosted(query, k) {
            Ok(hits) => hits
                .into_iter()
                .map(|h| SearchHit {
                    docid: h.docid,
                    score: Some(h.score as f64),
                    snippet: Some(h.text),
                })
                .collect(),
            Err(e) => {
                tracing::warn!(error = %e, "Native BM25 search failed");
                vec![]
            }
        },
    }
}

/// Classify the expected answer type for a query.
/// Returns (answer_type_string, input_tokens, output_tokens).
fn classify_answer_type(
    llm: &OllamaClient,
    model: &str,
    query: &str,
    rt: &tokio::runtime::Handle,
) -> (String, u32, u32) {
    let prompt = format!(
        "What type of answer does this question expect? Reply with ONLY one of: person, place, date, number, title, organization, event, other\n\nQuestion: {query}\n\nAnswer type:"
    );
    let messages = vec![Message {
        role: "user".into(),
        content: prompt,
    }];
    match rt.block_on(async {
        llm.chat_with_options(&messages, Some(model), Some(false))
            .await
    }) {
        Ok(resp) => {
            let answer = strip_think_tags(resp.content.trim()).to_lowercase();
            // Extract just the type word
            let atype = answer
                .split_whitespace()
                .next()
                .unwrap_or("other")
                .to_string();
            let input = resp.input_tokens.unwrap_or(0);
            let output = resp.output_tokens.unwrap_or(0);
            info!(answer_type = %atype, "Classified answer type");
            (atype, input, output)
        }
        Err(e) => {
            warn!(error = %e, "Answer type classification failed");
            ("other".into(), 0, 0)
        }
    }
}

/// Use the LLM to extract search queries from the question, then pre-search.
/// Returns (context_json_for_ouros, text_summary_for_prompt, input_tokens, output_tokens)
#[tracing::instrument(name = "rlm.pre_search", skip(llm, rt, backend), fields(gw.hits, gw.queries))]
fn pre_search(
    llm: &OllamaClient,
    model: &str,
    query: &str,
    backend: &SearchBackend,
    k: u32,
    rt: &tokio::runtime::Handle,
    bench_config: &BenchConfig,
) -> (Vec<serde_json::Value>, String, u32, u32) {
    // Ask LLM to extract diverse keyword queries — one per sub-fact in the question
    let n_q = bench_config.n_presearch_queries;
    let extract_prompt = format!(
        "Given this question, output {n_q} SHORT BM25 keyword search queries (2-5 words each).\n\n\
         CRITICAL: Keep queries SHORT (2-5 words). BM25 matches keywords, not sentences.\n\
         Each query should target a DIFFERENT fact or entity from the question.\n\
         Include: specific names, places, dates, awards, organizations, works.\n\n\
         Output ONLY the queries, one per line. No numbering or explanation.\n\n\
         Question: {query}\n\nSearch queries:"
    );

    let messages = vec![Message {
        role: "user".into(),
        content: extract_prompt,
    }];

    let resp = match rt.block_on(async {
        llm.chat_with_options(&messages, Some(model), Some(false))
            .await
    }) {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Pre-search LLM call failed");
            return (vec![], String::new(), 0, 0);
        }
    };

    let input_tokens = resp.input_tokens.unwrap_or(0);
    let output_tokens = resp.output_tokens.unwrap_or(0);
    let content = strip_think_tags(&resp.content);

    let mut all_hits: Vec<serde_json::Value> = Vec::new();
    let mut seen_docids = std::collections::HashSet::new();
    let mut context_text = String::new();

    let queries: Vec<&str> = content
        .lines()
        .map(|l| {
            l.trim()
                .trim_start_matches(|c: char| c.is_numeric() || c == '.' || c == '-' || c == ')')
        })
        .map(|l| l.trim().trim_matches('"'))
        .filter(|l| !l.is_empty() && l.len() < 100)
        .take(bench_config.n_presearch_queries)
        .collect();

    info!(queries = ?queries, "Pre-search queries");
    let mut total_queries = queries.len();

    // S5-lite: coverage pre-search — take top-N from each sub-query in
    // round-robin (rank r across all queries before rank r+1), skip PRF +
    // round-2. Forces sub-fact coverage in the top-k.
    if bench_config.presearch_coverage {
        let per_q = bench_config.presearch_coverage_per_query.max(1);
        let mut per_query_hits: Vec<Vec<SearchHit>> = Vec::with_capacity(queries.len());
        for search_query in &queries {
            per_query_hits.push(backend_search(backend, search_query, per_q, rt));
        }
        let preview_chars = bench_config.context_snippet_chars;
        'rr: for rank in 0..per_q {
            for hits in &per_query_hits {
                let Some(hit) = hits.get(rank) else { continue };
                if !seen_docids.insert(hit.docid.clone()) {
                    continue;
                }
                let snippet = hit.snippet.as_deref().unwrap_or("");
                let preview: String = snippet.chars().take(preview_chars).collect();
                context_text.push_str(&format!("docid={}: {}\n", hit.docid, preview));
                all_hits.push(serde_json::json!({
                    "docid": hit.docid,
                    "snippet": snippet,
                }));
                if all_hits.len() >= queries.len() * per_q {
                    break 'rr;
                }
            }
        }
        info!(
            n_hits = all_hits.len(),
            n_queries = queries.len(),
            skip_refine = bench_config.presearch_coverage_skip_refine,
            "Coverage pre-search round-1 complete (round-robin)"
        );
        if bench_config.presearch_coverage_skip_refine {
            let span = tracing::Span::current();
            span.record("gw.hits", all_hits.len() as u64);
            span.record("gw.queries", total_queries as u64);
            return (all_hits, context_text, input_tokens, output_tokens);
        }
        // else: fall through to PRF + round-2 refinement over the
        // coverage-anchored seed set (hybrid S5-lite + refine).
    }

    for search_query in &queries {
        let hits = backend_search(backend, search_query, bench_config.presearch_k, rt);
        for hit in hits {
            if seen_docids.insert(hit.docid.clone()) {
                let snippet = hit.snippet.as_deref().unwrap_or("");
                let preview: String = snippet.chars().take(bench_config.context_snippet_chars).collect();
                context_text.push_str(&format!("docid={}: {}\n", hit.docid, preview));
                all_hits.push(serde_json::json!({
                    "docid": hit.docid,
                    "snippet": snippet,
                }));
            }
        }
    }

    info!(
        n_hits = all_hits.len(),
        n_queries = queries.len(),
        "Pre-search round 1 complete"
    );

    // --- Pseudo-relevance feedback: extract distinctive terms from top snippets ---
    let prf_terms = {
        let mut term_freq: std::collections::HashMap<String, usize> =
            std::collections::HashMap::new();
        for h in all_hits.iter().take(bench_config.prf_top_n) {
            let snippet = h["snippet"].as_str().unwrap_or("");
            for word in snippet.split_whitespace() {
                let clean: String = word
                    .to_lowercase()
                    .chars()
                    .filter(|c| c.is_alphanumeric())
                    .collect();
                if clean.len() >= 4 && clean.len() <= 20 {
                    *term_freq.entry(clean).or_insert(0) += 1;
                }
            }
        }
        // Remove terms already in the query
        let query_lower = query.to_lowercase();
        let query_words: std::collections::HashSet<String> = query_lower
            .split_whitespace()
            .map(|w| w.chars().filter(|c| c.is_alphanumeric()).collect())
            .collect();
        term_freq.retain(|t, _| !query_words.contains(t));
        // Remove very common words
        let stopwords = [
            "this", "that", "with", "from", "have", "been", "were", "they", "their", "about",
            "would", "which", "could", "other", "more", "some", "also", "into", "than", "them",
            "only", "over", "said", "will", "when", "what", "there", "after", "before", "first",
            "most", "very", "just", "like", "each", "where", "does", "many",
        ];
        let stop_set: std::collections::HashSet<&str> = stopwords.iter().cloned().collect();
        term_freq.retain(|t, _| !stop_set.contains(t.as_str()));
        // Sort by freq, take top N (configurable via prf_term_count)
        let mut sorted: Vec<(String, usize)> = term_freq.into_iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(&a.1));
        sorted
            .into_iter()
            .take(bench_config.prf_term_count)
            .map(|(t, _)| t)
            .collect::<Vec<String>>()
    };

    if !prf_terms.is_empty() {
        info!(prf_terms = ?prf_terms, "PRF terms extracted");
        // Search with PRF terms as a combined query
        let prf_query = prf_terms.join(" ");
        let prf_hits = backend_search(backend, &prf_query, bench_config.prf_k, rt);
        for hit in prf_hits {
            if seen_docids.insert(hit.docid.clone()) {
                let snippet = hit.snippet.as_deref().unwrap_or("");
                let preview: String = snippet.chars().take(300).collect();
                context_text.push_str(&format!("docid={}: {}\n", hit.docid, preview));
                all_hits.push(serde_json::json!({
                    "docid": hit.docid,
                    "snippet": snippet,
                }));
            }
        }
        total_queries += 1;
    }

    info!(n_hits = all_hits.len(), "Pre-search after HyDE + PRF");

    // --- Round 2: Refinement ---
    // Show the LLM round-1 results. Ask it to:
    // 1. Pick the most relevant docids from round 1
    // 2. Generate 3 new queries for missing information
    let round1_preview: String = all_hits
        .iter()
        .enumerate()
        .take(25)
        .map(|(i, h)| {
            let docid = h["docid"].as_str().unwrap_or("?");
            let snippet = h["snippet"].as_str().unwrap_or("");
            let preview: String = snippet.chars().take(200).collect();
            format!("  {}. [{}] {}", i + 1, docid, preview)
        })
        .collect::<Vec<_>>()
        .join("\n");

    let prf_hint = if !prf_terms.is_empty() {
        format!(
            "\n\nFrequent distinctive terms from top results: {}. Consider incorporating these into your new queries.",
            prf_terms.join(", ")
        )
    } else {
        String::new()
    };

    let refine_prompt = format!(
        "Question: {query}\n\n\
         Here are {n} search results. Select the MOST RELEVANT documents and generate new searches.\n\n\
         Results:\n{round1_preview}\n\n\
         First, list the numbers of the TOP 10 most relevant results (comma-separated).\n\
         Then, output 3 NEW keyword search queries (2-5 words) for facts NOT covered above.{prf_hint}\n\n\
         Format:\n\
         KEEP: 1, 3, 5, 7, ...\n\
         SEARCH: query one\n\
         SEARCH: query two\n\
         SEARCH: query three",
        n = all_hits.len().min(25),
    );

    let refine_messages = vec![Message {
        role: "user".into(),
        content: refine_prompt,
    }];

    let mut total_input = input_tokens;
    let mut total_output = output_tokens;

    if let Ok(resp2) = rt.block_on(async {
        llm.chat_with_options(&refine_messages, Some(model), Some(false))
            .await
    }) {
        total_input += resp2.input_tokens.unwrap_or(0);
        total_output += resp2.output_tokens.unwrap_or(0);
        let content2 = strip_think_tags(&resp2.content);

        // Parse KEEP line to filter round-1 results
        let mut keep_set = std::collections::BTreeSet::new();
        let mut new_query_set = std::collections::LinkedList::new();
        let mut seen_queries = std::collections::HashSet::new();

        for line in content2.lines() {
            let trimmed = line.trim();
            if trimmed.to_lowercase().starts_with("keep:") {
                let nums_part = trimmed.split_once(':').map(|x| x.1).unwrap_or("");
                for num_str in nums_part.split(',') {
                    if let Ok(n) = num_str.trim().parse::<usize>() {
                        if n >= 1 && n <= all_hits.len() {
                            keep_set.insert(n - 1); // 1-indexed to 0-indexed
                        }
                    }
                }
            } else if trimmed.to_lowercase().starts_with("search:") {
                let q = trimmed
                    .split_once(':')
                    .map(|x| x.1)
                    .unwrap_or("")
                    .trim()
                    .to_string();
                if !q.is_empty() && q.len() < 100 && seen_queries.insert(q.clone()) {
                    new_query_set.push_back(q);
                }
            }
        }

        let keep_indices: Vec<usize> = keep_set.into_iter().collect();
        let new_queries: Vec<String> = new_query_set
            .into_iter()
            .take(bench_config.round2_new_queries)
            .collect();

        info!(keep = ?keep_indices, new_queries = ?new_queries, "Pre-search round 2 parsed");

        // If we got valid KEEP indices, filter the context
        if !keep_indices.is_empty() && keep_indices.len() <= bench_config.round2_keep_max {
            let filtered: Vec<serde_json::Value> = keep_indices
                .iter()
                .filter_map(|&i| all_hits.get(i).cloned())
                .collect();

            // Rebuild context text from filtered hits
            context_text.clear();
            all_hits = Vec::new();
            seen_docids.clear();

            for h in &filtered {
                let docid = h["docid"].as_str().unwrap_or("?");
                let snippet = h["snippet"].as_str().unwrap_or("");
                if seen_docids.insert(docid.to_string()) {
                    let preview: String = snippet.chars().take(300).collect();
                    context_text.push_str(&format!("docid={}: {}\n", docid, preview));
                    all_hits.push(h.clone());
                }
            }

            info!(n_kept = all_hits.len(), "Filtered round-1 results");
        }

        // Run new searches from round 2
        let r2_count = new_queries.len().min(bench_config.round2_new_queries);
        total_queries += r2_count;
        for search_query in new_queries.iter().take(r2_count) {
            let hits = backend_search(backend, search_query, bench_config.round2_k, rt);
            for hit in hits {
                if seen_docids.insert(hit.docid.clone()) {
                    let snippet = hit.snippet.as_deref().unwrap_or("");
                    let preview: String = snippet.chars().take(300).collect();
                    context_text.push_str(&format!("docid={}: {}\n", hit.docid, preview));
                    all_hits.push(serde_json::json!({
                        "docid": hit.docid,
                        "snippet": snippet,
                    }));
                }
            }
        }

        info!(
            n_hits = all_hits.len(),
            "Pre-search round 2 complete (filtered + new)"
        );
    }

    let span = tracing::Span::current();
    span.record("gw.hits", all_hits.len() as u64);
    span.record("gw.queries", total_queries as u64);

    (all_hits, context_text, total_input, total_output)
}

pub(crate) struct RlmLoopResult {
    pub(crate) status: String,
    pub(crate) termination_reason: String,
    pub(crate) iterations_used: u32,
    pub(crate) result_entries: Vec<ResultEntry>,
    pub(crate) input_tokens: u32,
    pub(crate) output_tokens: u32,
    pub(crate) trajectory: Vec<TrajectoryMessage>,
}

#[tracing::instrument(name = "rlm.loop", skip(llm, agent, rt, pre_search_context, bench_config), fields(gw.max_iterations = max_iterations, gw.iterations_used, gw.status, gw.input_tokens, gw.output_tokens))]
fn run_rlm_loop(
    llm: &OllamaClient,
    model: &str,
    agent: &mut ReplAgent,
    query: &str,
    max_iterations: u32,
    rt: &tokio::runtime::Handle,
    pre_search_context: &str,
    bench_config: &BenchConfig,
) -> RlmLoopResult {
    let start_time = std::time::Instant::now();
    let system_prompt = bench_config.system_prompt();
    let mut messages: Vec<Message> = vec![Message {
        role: "system".into(),
        content: system_prompt.clone(),
    }];
    let mut trajectory: Vec<TrajectoryMessage> = vec![TrajectoryMessage {
        role: "system".into(),
        content: system_prompt,
        code_blocks: vec![],
        repl_output: None,
    }];

    let mut result_entries: Vec<ResultEntry> = Vec::new();
    let mut total_input = 0u32;
    let mut total_output = 0u32;
    let mut termination_reason = "max_turns".to_string();
    let mut iterations_used = 0u32;

    for iteration in 0..max_iterations {
        iterations_used = iteration + 1;
        let _iter_span = info_span!("rlm.iteration", n = iteration + 1).entered();
        // Check timeout
        if start_time.elapsed().as_secs() > bench_config.query_timeout_secs {
            warn!("Query timeout after {}s", start_time.elapsed().as_secs());
            termination_reason = "timeout".to_string();
            break;
        }
        // Build variables metadata preview (DSPy-style)
        let variables_info = build_variables_info(agent);

        // Build iteration prompt
        let user_prompt = if iteration == max_iterations - 1 {
            final_prompt(query, &variables_info)
        } else if iteration == 0 {
            let context_section = if !pre_search_context.is_empty() {
                format!(
                    "\n\nThe `context` variable has been pre-loaded with {} search results (list of {{docid, snippet}} dicts).\n\
                     Initial search results preview:\n{pre_search_context}",
                    pre_search_context.lines().count(),
                )
            } else {
                String::new()
            };
            format!(
                "[Iteration 1/{}] EXPLORE FIRST\n\n\
                 Query: \"{query}\"\
                 {context_section}\n\n\
                 Start by examining context. Identify which sub-facts from the query each document might help with. \
                 Use get_document() to read full texts and llm_query() to analyze them.",
                max_iterations,
            )
        } else {
            iteration_prompt(
                query,
                iteration as usize,
                max_iterations as usize,
                &variables_info,
            )
        };

        messages.push(Message {
            role: "user".into(),
            content: user_prompt.clone(),
        });
        trajectory.push(TrajectoryMessage {
            role: "user".into(),
            content: user_prompt,
            code_blocks: vec![],
            repl_output: None,
        });

        // Call root LLM
        let resp = rt.block_on(async { llm.chat(&messages, Some(model)).await });
        let resp = match resp {
            Ok(r) => r,
            Err(e) => {
                warn!(iteration, error = %e, "Root LLM call failed");
                break;
            }
        };

        total_input += resp.input_tokens.unwrap_or(0);
        total_output += resp.output_tokens.unwrap_or(0);

        // Strip <think>...</think> blocks (qwen3.5 thinking mode)
        let content = strip_think_tags(resp.content.trim());
        info!(
            iteration,
            len = content.len(),
            elapsed_s = start_time.elapsed().as_secs(),
            "LLM response"
        );

        // Add assistant response to conversation
        messages.push(Message {
            role: "assistant".into(),
            content: content.clone(),
        });
        // We'll add the full trajectory entry (with code_blocks + repl_output) after execution

        // Check for FINAL() in plain text (outside code blocks)
        if let Some(answer) = extract_final_answer(&content) {
            // Reject refusal answers — force the model to keep trying
            if is_refusal_answer(&answer) {
                info!(
                    iteration,
                    answer = answer.as_str(),
                    "Rejected refusal answer, continuing"
                );
                messages.push(Message {
                    role: "user".into(),
                    content: "Your answer was a refusal (\"unable to determine\" etc). This is NOT allowed. \
                              You MUST give a specific name, number, date, or phrase. \
                              Pick your BEST GUESS from the documents you've read. \
                              Write a ```repl``` block ending with FINAL(\"your concrete answer\"):".to_string(),
                });
                continue;
            }
            trajectory.push(TrajectoryMessage {
                role: "assistant".into(),
                content: content.clone(),
                code_blocks: vec![],
                repl_output: None,
            });
            if answer.starts_with("FINAL_VAR(") {
                // Extract variable name and read from session
                let var_name = answer
                    .strip_prefix("FINAL_VAR(")
                    .and_then(|s| s.strip_suffix(')'))
                    .unwrap_or("")
                    .trim()
                    .trim_matches('"')
                    .trim_matches('\'');
                if let Some(val) = agent.get_variable(var_name) {
                    let answer_str = match &val {
                        Object::String(s) => s.clone(),
                        other => format!("{other:?}"),
                    };
                    result_entries.push(ResultEntry {
                        entry_type: "output_text".into(),
                        tool_name: None,
                        arguments: None,
                        output: Some(serde_json::Value::String(format!(
                            "Exact Answer: {answer_str}"
                        ))),
                    });
                    let span = tracing::Span::current();
                    span.record("gw.status", "completed");
                    span.record("gw.iterations_used", iteration + 1);
                    span.record("gw.input_tokens", total_input as u64);
                    span.record("gw.output_tokens", total_output as u64);
                    return RlmLoopResult {
                        status: "completed".into(),
                        termination_reason: "final_called".into(),
                        iterations_used,
                        result_entries,
                        input_tokens: total_input,
                        output_tokens: total_output,
                        trajectory,
                    };
                }
            } else {
                result_entries.push(ResultEntry {
                    entry_type: "output_text".into(),
                    tool_name: None,
                    arguments: None,
                    output: Some(serde_json::Value::String(format!("Exact Answer: {answer}"))),
                });
                let span = tracing::Span::current();
                span.record("gw.status", "completed");
                span.record("gw.iterations_used", iteration + 1);
                span.record("gw.input_tokens", total_input as u64);
                span.record("gw.output_tokens", total_output as u64);
                return RlmLoopResult {
                    status: "completed".into(),
                    termination_reason: "final_called".into(),
                    iterations_used,
                    result_entries,
                    input_tokens: total_input,
                    output_tokens: total_output,
                    trajectory,
                };
            }
        }

        // Extract and execute code blocks
        let code_blocks = extract_code_blocks(&content);
        if code_blocks.is_empty() {
            // No code — if response contains something useful, record it
            if !content.is_empty() {
                result_entries.push(ResultEntry {
                    entry_type: "output_text".into(),
                    tool_name: None,
                    arguments: None,
                    output: Some(serde_json::Value::String(content.clone())),
                });
            }
            continue;
        }

        // Execute each code block
        let mut exec_output = String::new();
        for block in &code_blocks {
            match agent.execute(block) {
                Ok(result) => {
                    // Combine print output and return value
                    let mut block_output = String::new();
                    if !result.stdout.is_empty() {
                        block_output.push_str(&result.stdout);
                    }
                    // If no print output, show the return value (handles cases
                    // where model writes `search("query")` without print())
                    if block_output.is_empty() && !result.value.is_null() {
                        let val_str = match &result.value {
                            serde_json::Value::String(s) => s.clone(),
                            other => serde_json::to_string_pretty(other).unwrap_or_default(),
                        };
                        if !val_str.is_empty() && val_str != "null" {
                            block_output.push_str(&val_str);
                        }
                    }

                    // Truncate to prevent context explosion
                    let max_output = bench_config.repl_output_max_chars;
                    let truncated = if block_output.len() > max_output {
                        let trunc: String = block_output.chars().take(max_output).collect();
                        format!("{}...\n[truncated]", trunc)
                    } else {
                        block_output.clone()
                    };
                    exec_output.push_str(&truncated);

                    result_entries.push(ResultEntry {
                        entry_type: "tool_call".into(),
                        tool_name: Some("repl".into()),
                        arguments: Some(serde_json::Value::String(block.clone())),
                        output: Some(serde_json::Value::String(block_output)),
                    });

                    // Check if FINAL was called inside code
                    if result.is_final {
                        if let Some(val) = result.final_value {
                            let answer = match &val {
                                serde_json::Value::String(s) => s.clone(),
                                other => other.to_string(),
                            };
                            // Reject refusal answers from code too
                            if is_refusal_answer(&answer) {
                                info!(
                                    iteration,
                                    answer = answer.as_str(),
                                    "Rejected refusal FINAL from code"
                                );
                                exec_output.push_str("[Your answer was rejected because it was a refusal. Give a specific answer.]\n");
                            } else {
                                result_entries.push(ResultEntry {
                                    entry_type: "output_text".into(),
                                    tool_name: None,
                                    arguments: None,
                                    output: Some(serde_json::Value::String(format!(
                                        "Exact Answer: {answer}"
                                    ))),
                                });
                                return RlmLoopResult {
                                    status: "completed".into(),
                                    termination_reason: "final_called_code".into(),
                                    iterations_used,
                                    result_entries,
                                    input_tokens: total_input,
                                    output_tokens: total_output,
                                    trajectory,
                                };
                            }
                        }
                    }
                }
                Err(e) => {
                    let err_msg = format!("Error: {e}");
                    exec_output.push_str(&err_msg);
                    exec_output.push('\n');
                }
            }
        }

        // Record assistant + REPL trajectory
        trajectory.push(TrajectoryMessage {
            role: "assistant".into(),
            content: content.clone(),
            code_blocks: code_blocks.clone(),
            repl_output: if exec_output.is_empty() {
                None
            } else {
                Some(exec_output.clone())
            },
        });

        // Feed execution output back to the LLM
        if !exec_output.is_empty() {
            let repl_msg = format!("REPL output:\n```\n{exec_output}```");
            messages.push(Message {
                role: "user".into(),
                content: repl_msg.clone(),
            });
            trajectory.push(TrajectoryMessage {
                role: "repl_output".into(),
                content: repl_msg,
                code_blocks: vec![],
                repl_output: None,
            });
        }

        // Check if agent finished via FINAL in code
        if agent.is_finished() {
            if let Some(val) = agent.final_value() {
                let answer = match &val {
                    serde_json::Value::String(s) => s.clone(),
                    other => other.to_string(),
                };
                result_entries.push(ResultEntry {
                    entry_type: "output_text".into(),
                    tool_name: None,
                    arguments: None,
                    output: Some(serde_json::Value::String(format!("Exact Answer: {answer}"))),
                });
                let span = tracing::Span::current();
                span.record("gw.status", "completed");
                span.record("gw.iterations_used", iteration + 1);
                span.record("gw.input_tokens", total_input as u64);
                span.record("gw.output_tokens", total_output as u64);
                return RlmLoopResult {
                    status: "completed".into(),
                    termination_reason: "final_called".into(),
                    iterations_used,
                    result_entries,
                    input_tokens: total_input,
                    output_tokens: total_output,
                    trajectory,
                };
            }
        }
    }

    // Max iterations reached — first try facts.best_candidate(), then LLM fallback
    info!("Max iterations reached, attempting fallback extraction");

    // Try extracting the best candidate from the FactRegistry
    let facts_answer =
        match agent.execute("_bc = facts.best_candidate()\nif _bc:\n    print(_bc[0])") {
            Ok(result)
                if !result.stdout.trim().is_empty() && !is_refusal_answer(result.stdout.trim()) =>
            {
                let ans = result.stdout.trim().to_string();
                info!(answer = ans.as_str(), "FactRegistry fallback candidate");
                Some(ans)
            }
            _ => None,
        };

    let (fallback_answer, fb_input, fb_output) = if facts_answer.is_some() {
        (facts_answer, 0, 0)
    } else {
        fallback_extract(llm, model, query, &messages, rt)
    };
    total_input += fb_input;
    total_output += fb_output;

    let answer = fallback_answer.unwrap_or_else(|| "Unable to determine".to_string());
    result_entries.push(ResultEntry {
        entry_type: "output_text".into(),
        tool_name: None,
        arguments: None,
        output: Some(serde_json::Value::String(format!("Exact Answer: {answer}"))),
    });
    trajectory.push(TrajectoryMessage {
        role: "system".into(),
        content: format!("Fallback extraction: {answer}"),
        code_blocks: vec![],
        repl_output: None,
    });
    let span = tracing::Span::current();
    span.record("gw.status", "max_turns_fallback");
    span.record("gw.iterations_used", max_iterations as u64);
    span.record("gw.input_tokens", total_input as u64);
    span.record("gw.output_tokens", total_output as u64);
    RlmLoopResult {
        status: "max_turns_fallback".into(),
        termination_reason,
        iterations_used,
        result_entries,
        input_tokens: total_input,
        output_tokens: total_output,
        trajectory,
    }
}

// -------------------------------------------------------------------------- //
// Query loading
// -------------------------------------------------------------------------- //

fn load_queries_tsv(path: &Path) -> Vec<(String, String)> {
    let content = std::fs::read_to_string(path).expect("Failed to read TSV");
    content
        .lines()
        .filter_map(|line| {
            let parts: Vec<&str> = line.splitn(2, '\t').collect();
            if parts.len() == 2 {
                Some((parts[0].trim().to_string(), parts[1].trim().to_string()))
            } else {
                None
            }
        })
        .collect()
}

fn already_processed(output_dir: &Path) -> std::collections::HashSet<String> {
    let mut ids = std::collections::HashSet::new();
    if let Ok(entries) = std::fs::read_dir(output_dir) {
        for entry in entries.flatten() {
            if entry.path().extension().is_some_and(|e| e == "json") {
                if let Ok(data) = std::fs::read_to_string(entry.path()) {
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(&data) {
                        if let Some(qid) = v.get("query_id").and_then(|v| v.as_str()) {
                            ids.insert(qid.to_string());
                        }
                    }
                }
            }
        }
    }
    ids
}

// -------------------------------------------------------------------------- //
// CLI
// -------------------------------------------------------------------------- //

#[derive(Parser)]
#[command(
    name = "gw-bench",
    about = "Greatwheel BrowseComp-Plus benchmark (rLM + ouros REPL)"
)]
struct Cli {
    /// LLM backend: "ollama" (native API) or "sglang" (OpenAI-compatible)
    #[arg(long, default_value = "ollama", env = "GW_LLM_BACKEND")]
    llm_backend: String,

    /// LLM server URL (overrides --ollama-url when set)
    #[arg(long, env = "GW_LLM_URL")]
    llm_url: Option<String>,

    /// Ollama API URL (used when --llm-url is not set)
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_URL")]
    ollama_url: String,

    /// LLM model name
    #[arg(long, default_value = "qwen2.5:7b", env = "GW_MODEL")]
    model: String,

    /// Search backend: "http" (Python server) or "native" (in-process Rust)
    #[arg(long, default_value = "http")]
    search_backend: String,

    /// BrowseComp-Plus search server URL (for http backend)
    #[arg(long, default_value = "http://localhost:8000")]
    search_url: String,

    /// Search mode sent to HTTP backend (e.g. "bm25", "rerank", "hybrid")
    #[arg(long, default_value = "bm25")]
    search_mode: String,

    /// ColBERT rerank server URL (for native backend). BM25 retrieves top-50, reranker refines to top-k.
    #[arg(long)]
    rerank_url: Option<String>,

    /// Path to ColBERT LanceDB database (for native backend ColBERT search)
    #[arg(long)]
    colbert_lance: Option<String>,

    /// ColBERT LanceDB table name
    #[arg(long, default_value = "colbert_docs")]
    colbert_table: String,

    /// ColBERT encode server URL (for query-time token encoding)
    #[arg(long)]
    colbert_encode_url: Option<String>,

    /// Path to tantivy corpus index (for native backend)
    #[arg(long, default_value = "data/tantivy-corpus/")]
    tantivy_index: String,

    /// Path to LanceDB database (for native backend vector/hybrid search)
    #[arg(long)]
    lancedb_path: Option<String>,

    /// LanceDB table name (for native backend)
    #[arg(long, default_value = "browsecomp_docs")]
    lancedb_table: String,

    /// Build tantivy index from corpus JSONL, then exit
    #[arg(long)]
    build_index: bool,

    /// Build passage-level tantivy index from corpus JSONL, then exit
    #[arg(long)]
    build_passage_index: bool,

    /// Path to passage-level tantivy index (for native backend)
    #[arg(long)]
    passage_index: Option<String>,

    /// Passage chunk sizes in bytes, comma-separated (for --build-passage-index)
    #[arg(long, default_value = "512,1024,2048,4096")]
    passage_chunk_bytes: String,

    /// Passage overlap in bytes (for --build-passage-index)
    #[arg(long, default_value_t = 100)]
    passage_overlap_bytes: usize,

    /// Path to corpus JSONL file (for --build-index / --build-passage-index)
    #[arg(long)]
    corpus_jsonl: Option<String>,

    /// Benchmark config TOML file (overrides hardcoded defaults for all pipeline parameters)
    #[arg(long)]
    config: Option<String>,

    /// Number of search results per query
    #[arg(long, default_value_t = 10)]
    k: u32,

    /// Max rLM iterations per query
    #[arg(long, default_value_t = 12)]
    max_turns: u32,

    /// Best-of-N: run each query N times and pick majority answer
    #[arg(long, default_value_t = 1)]
    runs: u32,

    /// Query: a string or path to TSV file
    #[arg(long, default_value = "topics-qrels/queries.tsv")]
    query: String,

    /// Query ID (for single-query mode, sets query_id in output JSON)
    #[arg(long)]
    query_id: Option<String>,

    /// Output directory for result JSON files
    #[arg(long, default_value = "runs/rlm-agent")]
    output_dir: String,

    /// Use gw-loop ConversationLoop instead of the hand-rolled rLM loop.
    #[arg(long)]
    use_conv_loop: bool,
}

#[tracing::instrument(name = "rlm.question", skip(llm, cli, rt, native_searcher), fields(gw.model = %cli.model, gw.query_id = query_id, gw.status, gw.answer))]
fn run_single_query(
    llm: &OllamaClient,
    cli: &Cli,
    bench_config: &BenchConfig,
    query_text: &str,
    query_id: Option<&str>,
    rt: &tokio::runtime::Handle,
    native_searcher: Option<&Arc<CorpusSearcher>>,
) -> RunRecord {
    let backend = if let Some(searcher) = native_searcher {
        SearchBackend::Native {
            searcher: searcher.clone(),
        }
    } else {
        SearchBackend::Http {
            url: cli.search_url.clone(),
            client: reqwest::Client::new(),
        }
    };

    // Build a second backend ref for pre_search (avoid moving)
    let pre_search_backend = if let Some(searcher) = native_searcher {
        SearchBackend::Native {
            searcher: searcher.clone(),
        }
    } else {
        SearchBackend::Http {
            url: cli.search_url.clone(),
            client: reqwest::Client::new(),
        }
    };

    let query_start = std::time::Instant::now();

    let search_backend = backend;
    let docid_tracker: Arc<std::sync::Mutex<Vec<String>>> =
        Arc::new(std::sync::Mutex::new(Vec::new()));
    let mut bridge = BrowseCompBridge::new(
        search_backend.clone(),
        cli.k,
        cli.search_mode.clone(),
        cli.rerank_url.clone(),
        cli.colbert_encode_url.clone(),
        llm.clone(),
        cli.model.clone(),
        rt.clone(),
        docid_tracker.clone(),
    );
    // Apply config overrides
    bridge.max_llm_calls = bench_config.max_llm_calls;
    bridge.max_search_calls = bench_config.max_search_calls;
    bridge.timeout_secs = bench_config.bridge_timeout_secs;

    let external_fns = vec![
        "search".to_string(),
        "vector_search".to_string(),
        "get_document".to_string(),
        "llm_query".to_string(),
        "batch_llm_query".to_string(),
        "FINAL".to_string(),
    ];

    let mut agent = ReplAgent::new(external_fns.clone(), Box::new(bridge));

    // Inject question as a variable
    agent
        .set_variable("question", Object::String(query_text.to_string()))
        .ok();

    // Pre-search: decompose query into sub-facts and run diverse searches
    let pre_search_start = std::time::Instant::now();
    let (context_hits, context_text, pre_input, pre_output) =
        pre_search(llm, &cli.model, query_text, &pre_search_backend, cli.k, rt, bench_config);
    let pre_search_ms = pre_search_start.elapsed().as_millis() as u64;

    // Classify expected answer type (cheap — one short LLM call)
    let (answer_type, at_input, at_output) = classify_answer_type(llm, &cli.model, query_text, rt);
    let pre_input = pre_input + at_input;
    let pre_output = pre_output + at_output;

    // Inject context as ouros variable (list of {docid, snippet} dicts)
    let context_obj = json_to_object(serde_json::Value::Array(context_hits.clone()));
    agent.set_variable("context", context_obj).ok();
    agent
        .set_variable("answer_type", Object::String(answer_type.clone()))
        .ok();

    // Bootstrap FactRegistry and entity_search into the REPL namespace
    if let Err(e) = agent.execute(FACT_REGISTRY_BOOTSTRAP) {
        warn!(error = %e, "Failed to bootstrap FactRegistry");
    }
    if let Err(e) = agent.execute(ENTITY_SEARCH_BOOTSTRAP) {
        warn!(error = %e, "Failed to bootstrap entity_search");
    }

    let rlm_start = std::time::Instant::now();
    let rlm_result = if cli.use_conv_loop {
        // Use ConversationLoop-based runner.
        // We need to create a new agent since conv_loop_runner creates its own.
        let mut bridge2 = BrowseCompBridge::new(
            search_backend.clone(),
            cli.k,
            cli.search_mode.clone(),
            cli.rerank_url.clone(),
            cli.colbert_encode_url.clone(),
            llm.clone(),
            cli.model.clone(),
            rt.clone(),
            docid_tracker.clone(),
        );
        // Give the bridge its own runtime so its block_on calls don't
        // conflict with ConversationLoop's runtime.
        bridge2.use_dedicated_runtime();
        bridge2.max_llm_calls = bench_config.max_llm_calls;
        bridge2.max_search_calls = bench_config.max_search_calls;
        bridge2.timeout_secs = bench_config.bridge_timeout_secs;
        let bridge2_boxed: Box<dyn HostBridge> = Box::new(bridge2);
        // Apply config overrides to the new bridge.
        // (Can't access fields through Box<dyn HostBridge>, so we cast.)
        // Actually, let's just use the external_fns and let the runner handle it.
        let ext_fns = external_fns.clone();
        let mut agent2 = ReplAgent::new(ext_fns, bridge2_boxed);
        let context_obj2 = json_to_object(serde_json::Value::Array(context_hits));
        agent2.set_variable("context", context_obj2).ok();
        agent2
            .set_variable("question", Object::String(query_text.to_string()))
            .ok();
        agent2
            .set_variable("answer_type", Object::String(answer_type.clone()))
            .ok();
        if let Err(e) = agent2.execute(FACT_REGISTRY_BOOTSTRAP) {
            warn!(error = %e, "Failed to bootstrap FactRegistry (conv_loop)");
        }
        if let Err(e) = agent2.execute(ENTITY_SEARCH_BOOTSTRAP) {
            warn!(error = %e, "Failed to bootstrap entity_search (conv_loop)");
        }
        conv_loop_runner::run_rlm_loop_v2_with_agent(
            llm,
            &cli.model,
            agent2,
            query_text,
            cli.max_turns,
            rt,
            &context_text,
            bench_config,
        )
    } else {
        run_rlm_loop(
            llm,
            &cli.model,
            &mut agent,
            query_text,
            cli.max_turns,
            rt,
            &context_text,
            bench_config,
        )
    };
    let rlm_ms = rlm_start.elapsed().as_millis() as u64;
    let total_ms = query_start.elapsed().as_millis() as u64;
    let status = rlm_result.status;
    let termination_reason = rlm_result.termination_reason;
    let iterations_used = rlm_result.iterations_used;
    let result_entries = rlm_result.result_entries;
    let trajectory = rlm_result.trajectory;
    let input_tokens = rlm_result.input_tokens + pre_input;
    let output_tokens = rlm_result.output_tokens + pre_output;

    info!(
        total_ms,
        pre_search_ms,
        rlm_loop_ms = rlm_ms,
        "Query timing breakdown"
    );

    // Record status and answer on the rlm.question span
    tracing::Span::current().record("gw.status", status.as_str());

    // Extract tracking data from the bridge (it was moved into the agent)
    // We need to get it back — for now, count from result entries
    let mut tool_counts: HashMap<String, u32> = HashMap::new();
    for entry in &result_entries {
        if entry.entry_type == "tool_call" {
            if let Some(name) = &entry.tool_name {
                *tool_counts.entry(name.clone()).or_insert(0) += 1;
            }
        }
    }

    // Record answer on span for trace analysis
    let record = RunRecord {
        metadata: RunMetadata {
            model: cli.model.clone(),
            llm_backend: cli.llm_backend.clone(),
            llm_url: cli
                .llm_url
                .clone()
                .unwrap_or_else(|| cli.ollama_url.clone()),
            searcher: "rlm-ouros".into(),
            max_turns: cli.max_turns,
            k: cli.k,
        },
        query_id: query_id.map(|s| s.to_string()),
        tool_call_counts: tool_counts,
        usage: UsageInfo {
            input_tokens,
            output_tokens,
            total_tokens: input_tokens + output_tokens,
        },
        timing: Some(TimingInfo {
            total_ms,
            bm25_ms: 0,
            embed_ms: 0,
            vector_ms: 0,
            llm_query_ms: 0,
            get_doc_ms: 0,
            root_llm_ms: rlm_ms,
            other_ms: pre_search_ms,
        }),
        status,
        termination_reason,
        iterations_used,
        retrieved_docids: {
            let mut docids = docid_tracker
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .clone();
            docids.sort();
            docids.dedup();
            docids
        },
        result: result_entries,
        trajectory,
    };
    if let Some(answer) = extract_answer(&record) {
        tracing::Span::current().record("gw.answer", answer.as_str());
    }
    record
}

/// Extract the final answer string from a RunRecord.
fn extract_answer(record: &RunRecord) -> Option<String> {
    for entry in record.result.iter().rev() {
        if let Some(output) = &entry.output {
            if let Some(s) = output.as_str() {
                if let Some(answer) = s.strip_prefix("Exact Answer: ") {
                    let answer = answer.trim();
                    if !answer.is_empty() && answer != "Unable to determine" {
                        return Some(answer.to_string());
                    }
                }
            }
        }
    }
    None
}

/// Pick the most common answer from multiple runs (majority vote).
/// If there's a tie, pick the one that appeared first.
fn majority_vote(answers: &[String]) -> String {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut first_seen: HashMap<String, usize> = HashMap::new();
    for (i, a) in answers.iter().enumerate() {
        let normalized = a.trim().to_lowercase();
        *counts.entry(normalized.clone()).or_insert(0) += 1;
        first_seen.entry(normalized).or_insert(i);
    }
    // Sort by count descending, then by first_seen ascending
    let mut candidates: Vec<_> = counts.into_iter().collect();
    candidates.sort_by(|a, b| {
        b.1.cmp(&a.1).then_with(|| {
            first_seen
                .get(&a.0)
                .unwrap_or(&0)
                .cmp(first_seen.get(&b.0).unwrap_or(&0))
        })
    });
    // Return the original-cased version of the winning answer
    let winner = &candidates[0].0;
    for a in answers {
        if a.trim().to_lowercase() == *winner {
            return a.clone();
        }
    }
    answers[0].clone()
}

/// Run a query N times and return the best record (majority-voted answer).
fn run_with_voting(
    llm: &OllamaClient,
    cli: &Cli,
    bench_config: &BenchConfig,
    query_text: &str,
    query_id: Option<&str>,
    rt: &tokio::runtime::Handle,
    n_runs: u32,
    native_searcher: Option<&Arc<CorpusSearcher>>,
) -> RunRecord {
    if n_runs <= 1 {
        return run_single_query(
            llm,
            cli,
            bench_config,
            query_text,
            query_id,
            rt,
            native_searcher,
        );
    }

    let mut records: Vec<RunRecord> = Vec::with_capacity(n_runs as usize);
    let mut answers: Vec<String> = Vec::new();

    for run_idx in 0..n_runs {
        info!(run = run_idx + 1, total_runs = n_runs, "Best-of-N run");
        let record = run_single_query(
            llm,
            cli,
            bench_config,
            query_text,
            query_id,
            rt,
            native_searcher,
        );
        if let Some(answer) = extract_answer(&record) {
            answers.push(answer);
        }
        records.push(record);
    }

    if answers.is_empty() {
        // No valid answers from any run — return the last record
        return records.pop().unwrap();
    }

    let voted_answer = majority_vote(&answers);
    info!(
        n_answers = answers.len(),
        voted_answer = %voted_answer,
        all_answers = ?answers,
        "Majority vote result"
    );

    // Find the record whose answer matches the voted answer and return it,
    // but override the answer with the voted one
    let mut best_record = None;
    let mut total_input = 0u32;
    let mut total_output = 0u32;
    for record in &records {
        total_input += record.usage.input_tokens;
        total_output += record.usage.output_tokens;
        if best_record.is_none() {
            if let Some(a) = extract_answer(record) {
                if a == voted_answer {
                    best_record = Some(record);
                }
            }
        }
    }

    let mut result = best_record.unwrap_or(&records[0]).clone();
    // Update usage to reflect total across all runs
    result.usage = UsageInfo {
        input_tokens: total_input,
        output_tokens: total_output,
        total_tokens: total_input + total_output,
    };
    // Update the final answer to the voted answer
    if let Some(last_entry) = result.result.last_mut() {
        last_entry.output = Some(serde_json::Value::String(format!(
            "Exact Answer: {voted_answer}"
        )));
    }
    result.status = if result.status == "completed" {
        "completed_voted".into()
    } else {
        format!("{}_voted", result.status)
    };
    result
}

#[tokio::main]
async fn main() {
    let trace_config = gw_trace::TracingConfig {
        exporter: std::env::var("GW_TRACE_EXPORTER").unwrap_or_else(|_| "console".into()),
        otlp_endpoint: std::env::var("GW_TRACE_OTLP_ENDPOINT").ok(),
        postgres_export: std::env::var("GW_TRACE_POSTGRES").is_ok_and(|v| v == "true" || v == "1"),
        service_name: "gw-bench".into(),
    };
    gw_trace::init_tracing(&trace_config, None).expect("Failed to initialize tracing");

    let cli = Cli::parse();

    // Load bench config from TOML or use defaults
    let bench_config = if let Some(ref config_path) = cli.config {
        BenchConfig::load(config_path).expect("Failed to load config")
    } else {
        BenchConfig::default()
    };
    info!(config = ?cli.config, "Loaded bench config");
    let rt = tokio::runtime::Handle::current();

    // Handle --build-index: build tantivy corpus index and exit
    if cli.build_index {
        let jsonl_path = cli
            .corpus_jsonl
            .as_deref()
            .expect("--corpus-jsonl is required with --build-index");
        let tantivy_path = PathBuf::from(&cli.tantivy_index);
        info!(jsonl = jsonl_path, out = %tantivy_path.display(), "Building tantivy corpus index");
        let count = CorpusSearcher::build_index(Path::new(jsonl_path), &tantivy_path)
            .expect("Failed to build index");
        info!(count, "Index built successfully");
        return;
    }

    // Handle --build-passage-index: build passage-level tantivy index and exit
    if cli.build_passage_index {
        let jsonl_path = cli
            .corpus_jsonl
            .as_deref()
            .expect("--corpus-jsonl is required with --build-passage-index");
        let out_path = cli
            .passage_index
            .as_deref()
            .unwrap_or("data/tantivy-passages/");
        info!(
            jsonl = jsonl_path,
            out = out_path,
            "Building passage-level tantivy index"
        );
        let chunk_sizes: Vec<usize> = cli
            .passage_chunk_bytes
            .split(',')
            .map(|s| s.trim().parse::<usize>().expect("Invalid chunk size"))
            .collect();
        info!(chunk_sizes = ?chunk_sizes, overlap = cli.passage_overlap_bytes, "Building hierarchical passage index");
        let count = CorpusSearcher::build_passage_index(
            Path::new(jsonl_path),
            Path::new(out_path),
            &chunk_sizes,
            cli.passage_overlap_bytes,
        )
        .expect("Failed to build passage index");
        info!(count, "Passage index built successfully");
        return;
    }

    let backend: gw_llm::LlmBackend = cli.llm_backend.parse().unwrap_or_else(|e| {
        eprintln!("Error: {e}");
        std::process::exit(1);
    });
    // Resolve LLM URL: --llm-url > --ollama-url (with backend-specific default)
    let llm_chat_url = cli.llm_url.clone().unwrap_or_else(|| {
        if backend == gw_llm::LlmBackend::Sglang {
            "http://localhost:30000".to_string()
        } else {
            cli.ollama_url.clone()
        }
    });
    // Embeddings always go through Ollama
    let llm = OllamaClient::with_backend(
        llm_chat_url,
        cli.ollama_url.clone(),
        cli.model.clone(),
        "nomic-embed-text".into(),
        backend,
    );

    // Open native searcher if requested
    let native_searcher: Option<Arc<CorpusSearcher>> = if cli.search_backend == "native" {
        let tantivy_path = PathBuf::from(&cli.tantivy_index);
        let passage_path = cli.passage_index.as_ref().map(PathBuf::from);
        let searcher = CorpusSearcher::open_with_passages(
            &tantivy_path,
            passage_path.as_deref(),
            cli.lancedb_path.as_deref(),
            Some(&cli.lancedb_table),
            cli.colbert_lance.as_deref(),
            Some(&cli.colbert_table),
        )
        .await
        .expect("Failed to open CorpusSearcher");
        Some(Arc::new(searcher))
    } else {
        None
    };
    let native_ref = native_searcher.as_ref();

    let output_dir = PathBuf::from(&cli.output_dir);
    std::fs::create_dir_all(&output_dir).expect("Failed to create output dir");

    let query_path = PathBuf::from(&cli.query);
    if query_path.is_file() && cli.query.ends_with(".tsv") {
        let queries = load_queries_tsv(&query_path);
        let done = already_processed(&output_dir);
        let remaining: Vec<_> = queries
            .into_iter()
            .filter(|(qid, _)| !done.contains(qid))
            .collect();

        info!(
            total = remaining.len(),
            skipped = done.len(),
            backend = cli.search_backend,
            "Processing queries"
        );

        for (i, (qid, query_text)) in remaining.iter().enumerate() {
            info!(
                progress = format!("{}/{}", i + 1, remaining.len()),
                query_id = qid,
                "Running query"
            );

            let record = tokio::task::block_in_place(|| {
                run_with_voting(
                    &llm,
                    &cli,
                    &bench_config,
                    query_text,
                    Some(qid),
                    &rt,
                    cli.runs,
                    native_ref,
                )
            });

            let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
            let filename = output_dir.join(format!("run_{qid}_{ts}.json"));
            let json = serde_json::to_string_pretty(&record).unwrap();
            std::fs::write(&filename, json).expect("Failed to write result");

            info!(
                path = %filename.display(),
                status = record.status,
                "Saved"
            );
        }
    } else {
        let record = tokio::task::block_in_place(|| {
            run_with_voting(
                &llm,
                &cli,
                &bench_config,
                &cli.query,
                cli.query_id.as_deref(),
                &rt,
                cli.runs,
                native_ref,
            )
        });

        let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
        let filename = output_dir.join(format!("run_{ts}.json"));
        let json = serde_json::to_string_pretty(&record).unwrap();
        std::fs::write(&filename, json).expect("Failed to write result");

        info!(path = %filename.display(), "Saved");
    }
}
