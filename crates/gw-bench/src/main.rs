//! gw-bench: Greatwheel BrowseComp-Plus benchmark using rLM-style REPL agents.
//!
//! Architecture:
//! 1. Rust host drives the rLM loop (system prompt → LLM → extract code → execute → repeat)
//! 2. Code blocks execute inside ouros ReplSession with persistent state
//! 3. External functions (search, get_document, llm_query) pause ouros and get resolved by Rust
//! 4. FINAL(answer) signals completion

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use chrono::Utc;
use clap::Parser;
use ouros::Object;
use serde::{Deserialize, Serialize};
use tracing::{debug, info, warn};

use gw_llm::{Message, OllamaClient};
use gw_runtime::{extract_code_blocks, extract_final_answer, json_to_object, AgentError, HostBridge, ReplAgent};

// -------------------------------------------------------------------------- //
// Types matching BrowseComp-Plus output format
// -------------------------------------------------------------------------- //

#[derive(Debug, Clone, Serialize)]
struct RunRecord {
    metadata: RunMetadata,
    query_id: Option<String>,
    tool_call_counts: HashMap<String, u32>,
    usage: UsageInfo,
    status: String,
    retrieved_docids: Vec<String>,
    result: Vec<ResultEntry>,
    /// Full conversation trajectory (system + user + assistant messages)
    #[serde(skip_serializing_if = "Vec::is_empty")]
    trajectory: Vec<TrajectoryMessage>,
}

#[derive(Debug, Clone, Serialize)]
struct TrajectoryMessage {
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
    ollama_url: String,
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
struct ResultEntry {
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
// BrowseComp host bridge
// -------------------------------------------------------------------------- //

struct BrowseCompBridge {
    search_url: String,
    k: u32,
    llm: OllamaClient,
    model: String,
    client: reqwest::Client,
    rt: tokio::runtime::Handle,
    // Tracking
    tool_counts: HashMap<String, u32>,
    all_docids: Vec<String>,
    total_input_tokens: u32,
    total_output_tokens: u32,
    // Limits
    max_llm_calls: u32,
    max_search_calls: u32,
    start_time: std::time::Instant,
    timeout_secs: u64,
}

impl BrowseCompBridge {
    fn new(
        search_url: String,
        k: u32,
        llm: OllamaClient,
        model: String,
        rt: tokio::runtime::Handle,
    ) -> Self {
        Self {
            search_url,
            k,
            llm,
            model,
            client: reqwest::Client::new(),
            rt,
            tool_counts: HashMap::new(),
            all_docids: Vec::new(),
            total_input_tokens: 0,
            total_output_tokens: 0,
            max_llm_calls: 25,
            max_search_calls: 20,
            start_time: std::time::Instant::now(),
            timeout_secs: 150,
        }
    }

    fn is_timed_out(&self) -> bool {
        self.start_time.elapsed().as_secs() > self.timeout_secs
    }

    fn search(&mut self, query: &str) -> Result<Object, AgentError> {
        if self.is_timed_out() {
            return Ok(Object::List(vec![]));
        }
        let count = self.tool_counts.entry("search".into()).or_insert(0);
        *count += 1;
        if *count > self.max_search_calls {
            return Ok(Object::List(vec![]));
        }

        let url = format!("{}/call/search", self.search_url);
        let k = self.k;
        let client = self.client.clone();
        let query = query.to_string();

        let hits: Vec<SearchHit> = self.rt.block_on(async {
            let resp = client
                .post(&url)
                .json(&serde_json::json!({ "query": query, "k": k }))
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
        })?;

        for hit in &hits {
            self.all_docids.push(hit.docid.clone());
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

    fn get_document(&mut self, docid: &str) -> Result<Object, AgentError> {
        if self.is_timed_out() {
            return Ok(Object::String("[timeout]".into()));
        }
        *self.tool_counts.entry("get_document".into()).or_insert(0) += 1;

        let url = format!("{}/call/get_document", self.search_url);
        let client = self.client.clone();
        let docid = docid.to_string();

        let text: String = self.rt.block_on(async {
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
        })?;

        Ok(Object::String(text))
    }

    fn llm_query(&mut self, prompt: &str) -> Result<Object, AgentError> {
        if self.is_timed_out() {
            return Ok(Object::String("[timeout — provide your best answer now]".into()));
        }
        let count = self.tool_counts.entry("llm_query".into()).or_insert(0);
        *count += 1;
        if *count > self.max_llm_calls {
            return Ok(Object::String("[llm_query limit reached — provide your best answer now]".into()));
        }

        let messages = vec![Message {
            role: "user".into(),
            content: prompt.to_string(),
        }];
        let llm = self.llm.clone();
        let model = self.model.clone();

        let resp = self.rt.block_on(async {
            llm.chat(&messages, Some(&model)).await.map_err(|e| {
                AgentError::HostFunction {
                    function: "llm_query".into(),
                    message: format!("{e}"),
                }
            })
        })?;

        self.total_input_tokens += resp.input_tokens.unwrap_or(0);
        self.total_output_tokens += resp.output_tokens.unwrap_or(0);

        Ok(Object::String(resp.content))
    }

    /// Batch LLM query — runs multiple prompts in parallel and returns a list of responses.
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
                        input_tokens: None,
                        output_tokens: None,
                    }),
                    Err(e) => responses.push(gw_llm::CompletionResponse {
                        content: format!("[error: {e}]"),
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
// rLM System Prompt & Iteration Prompts
// -------------------------------------------------------------------------- //

const SYSTEM_PROMPT: &str = r#"You are tasked with answering a query by searching and analyzing a corpus of ~100K web documents. You work interactively in a REPL environment that persists state across code blocks.

TOOLS AVAILABLE:
- `context` — pre-loaded list of {docid, snippet} dicts from initial searches
- `search(query)` — BM25 keyword search, returns list of {docid, snippet} dicts
- `get_document(docid)` — retrieve full document text
- `llm_query(prompt)` — sub-LLM analysis (~10K char context). YOUR MOST POWERFUL TOOL.
- `batch_llm_query([p1, p2, ...])` — parallel LLM queries
- `print()` — view output to continue reasoning
- `FINAL("answer")` — submit your final answer

Write Python code in ```repl``` blocks. Variables persist between blocks.

DIRECTIVES:
1. EXPLORE FIRST — examine context, run diverse searches, load documents. Don't jump to conclusions.
2. DECOMPOSE THE QUERY — break complex questions into sub-parts. Search for each part separately:
   - If the query mentions a person + an event + a place, search each independently.
   - If early searches fail, try COMPLETELY DIFFERENT keywords — synonyms, related entities, broader/narrower terms.
   - Don't repeat similar searches. Each search should use substantially different keywords.
3. ITERATE — work in small code snippets. Store intermediate results in variables. Build evidence step by step.
4. USE llm_query() FOR SEMANTICS — keyword search finds documents, but llm_query() understands them. Feed document text to llm_query() to extract specific facts.
5. VERIFY — before submitting, sanity-check your answer. Search for the candidate answer by name to confirm.
6. SUBMIT — when confident, call FINAL("answer"). Answer must be precise: a name, number, date, or short phrase.

SEARCH STRATEGY (critical for success):
- BM25 search works on keyword overlap. Use distinctive nouns, names, and numbers.
- If a search returns irrelevant results, DON'T repeat it — try a completely different angle.
- Search for entities mentioned in the query: people, places, organizations, works, events.
- Once you identify a candidate answer, search for THAT NAME to find confirming documents.

EXAMPLE:
```repl
# Step 1: Examine initial search results
for h in context[:5]:
    print(h["docid"], h["snippet"][:200])
```
```repl
# Step 2: Decompose and search from multiple angles
hits_person = search("specific person name mentioned")
hits_event = search("specific event or work mentioned")
hits_place = search("specific place or organization")
# Combine all unique documents
all_docs = {h["docid"]: h for h in context + hits_person + hits_event + hits_place}
print(f"Found {len(all_docs)} unique documents")
```
```repl
# Step 3: Load and analyze with LLM
doc = get_document(list(all_docs.keys())[0])
evidence = llm_query(f"Extract the answer to: [question]\n\nDocument:\n{doc[:8000]}")
print(evidence)
```
```repl
# Step 4: Verify candidate answer
confirm = search("candidate answer name")
print(confirm[0]["snippet"][:500])
```
```repl
FINAL("verified answer")
```

RULES:
- NEVER answer "Unable to determine". Always give your BEST GUESS.
- NEVER repeat a search query you already tried. Each search must use different keywords.
- Store evidence in variables; don't repeat work.

/no_think"#;

fn iteration_prompt(query: &str, iteration: usize, max_iterations: usize, variables_info: &str) -> String {
    let counter = format!("[Iteration {}/{}]", iteration + 1, max_iterations);

    if iteration == 0 {
        format!(
            "{counter} EXPLORE FIRST — you have not seen your context yet.\n\n\
             Query: \"{query}\"\n\n\
             Start by examining the `context` variable (pre-loaded search results). \
             Load promising documents with get_document(). Do NOT submit a final answer yet.\n\n\
             Write a ```repl``` code block:"
        )
    } else {
        let vars_section = if variables_info.is_empty() {
            String::new()
        } else {
            format!("\n\nVariables in scope:\n{variables_info}\n")
        };
        let nudge = if iteration >= 4 {
            "\nHINT: If stuck, try a COMPLETELY DIFFERENT search angle. \
             Search for candidate answers BY NAME to verify them. \
             Use llm_query() to analyze documents you've already loaded."
        } else {
            ""
        };
        format!(
            "{counter} Continue investigating to answer: \"{query}\"\n\
             {vars_section}\n\
             Think: which sub-facts from the query have you established? Which are still unknown? \
             Use search() with DIFFERENT keywords targeting the unknown facts. \
             Use get_document() to read full texts and llm_query() to analyze content.{nudge}\n\n\
             Write a ```repl``` code block:"
        )
    }
}

fn final_prompt(query: &str, variables_info: &str) -> String {
    let vars_section = if variables_info.is_empty() {
        String::new()
    } else {
        format!("\n\nVariables in scope:\n{variables_info}\n")
    };
    format!(
        "[FINAL ITERATION] You MUST submit your answer NOW.\n\n\
         Query: \"{query}\"\n\
         {vars_section}\n\
         Based on ALL evidence gathered, provide your best answer. \
         NEVER say 'Unable to determine'. Give your BEST GUESS even if uncertain.\n\n\
         Write a ```repl``` block ending with FINAL(\"your answer\"):"
    )
}

/// Build a summary of REPL variables (DSPy-style metadata preview).
/// Shows type, length, and a short preview — NOT full content.
fn build_variables_info(agent: &ReplAgent) -> String {
    let var_names = ["context", "evidence", "answer", "doc", "doc1", "doc2", "doc3",
                     "docs", "results", "hits", "response", "info", "data", "text",
                     "combined", "analysis", "findings", "candidates", "best"];
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
            ("dict", m.len(), format!("{{{}}}", keys.iter().map(|k| k.as_str()).collect::<Vec<_>>().join(", ")))
        }
    }
}

/// Fallback extraction: if max_iterations reached without FINAL, ask LLM to extract answer
/// from the accumulated conversation history (DSPy-style extract_sig).
fn fallback_extract(
    llm: &OllamaClient,
    model: &str,
    query: &str,
    messages: &[Message],
    rt: &tokio::runtime::Handle,
) -> (Option<String>, u32, u32) {
    // Build a condensed history for the extraction prompt
    let mut history = String::new();
    for msg in messages.iter().skip(1) {  // skip system prompt
        let role = &msg.role;
        let content_preview: String = msg.content.chars().take(2000).collect();
        history.push_str(&format!("[{role}]: {content_preview}\n\n"));
    }
    // Cap total history
    let history: String = history.chars().take(15000).collect();

    let extract_prompt = format!(
        "You are extracting a final answer from a research session. The researcher was investigating this query but ran out of iterations.\n\n\
         Query: \"{query}\"\n\n\
         Research history:\n{history}\n\n\
         Based on ALL evidence found during the research, what is the BEST answer to the query?\n\
         Give ONLY the answer — a specific name, number, date, or short phrase. No explanation.\n\
         NEVER say 'Unable to determine'. Give your best guess.\n\n/no_think"
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
fn strip_think_tags(text: &str) -> String {
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

const QUERY_TIMEOUT_SECS: u64 = 180; // 3 minutes max per query (more iterations need more time)

/// Use the LLM to extract search queries from the question, then pre-search.
/// Returns (context_json_for_ouros, text_summary_for_prompt, input_tokens, output_tokens)
fn pre_search(
    llm: &OllamaClient,
    model: &str,
    query: &str,
    bridge_search_url: &str,
    k: u32,
    rt: &tokio::runtime::Handle,
) -> (Vec<serde_json::Value>, String, u32, u32) {
    // Ask LLM to extract diverse keyword queries — one per sub-fact in the question
    let extract_prompt = format!(
        "Given this question, output 5 SHORT BM25 keyword search queries (2-5 words each).\n\n\
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

    let resp = match rt.block_on(async { llm.chat_with_options(&messages, Some(model), Some(false)).await }) {
        Ok(r) => r,
        Err(e) => {
            warn!(error = %e, "Pre-search LLM call failed");
            return (vec![], String::new(), 0, 0);
        }
    };

    let input_tokens = resp.input_tokens.unwrap_or(0);
    let output_tokens = resp.output_tokens.unwrap_or(0);
    let content = strip_think_tags(&resp.content);
    let client = reqwest::Client::new();

    let mut all_hits: Vec<serde_json::Value> = Vec::new();
    let mut seen_docids = std::collections::HashSet::new();
    let mut context_text = String::new();

    let queries: Vec<&str> = content.lines()
        .map(|l| l.trim().trim_start_matches(|c: char| c.is_numeric() || c == '.' || c == '-' || c == ')'))
        .map(|l| l.trim().trim_matches('"'))
        .filter(|l| !l.is_empty() && l.len() < 100)
        .take(5)
        .collect();

    info!(queries = ?queries, "Pre-search queries");

    for search_query in &queries {
        let url = format!("{}/call/search", bridge_search_url);
        let resp = rt.block_on(async {
            client
                .post(&url)
                .json(&serde_json::json!({ "query": search_query, "k": std::cmp::min(k, 5) }))
                .send()
                .await
        });

        if let Ok(resp) = resp {
            if let Ok(hits) = rt.block_on(async { resp.json::<Vec<SearchHit>>().await }) {
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
        }
    }

    info!(n_hits = all_hits.len(), n_queries = queries.len(), "Pre-search complete");

    (all_hits, context_text, input_tokens, output_tokens)
}

fn run_rlm_loop(
    llm: &OllamaClient,
    model: &str,
    agent: &mut ReplAgent,
    query: &str,
    max_iterations: u32,
    rt: &tokio::runtime::Handle,
    pre_search_context: &str,
) -> (String, Vec<ResultEntry>, u32, u32, Vec<TrajectoryMessage>) {
    let start_time = std::time::Instant::now();
    let mut messages: Vec<Message> = vec![Message {
        role: "system".into(),
        content: SYSTEM_PROMPT.to_string(),
    }];
    let mut trajectory: Vec<TrajectoryMessage> = vec![TrajectoryMessage {
        role: "system".into(),
        content: SYSTEM_PROMPT.to_string(),
        code_blocks: vec![],
        repl_output: None,
    }];

    let mut result_entries: Vec<ResultEntry> = Vec::new();
    let mut total_input = 0u32;
    let mut total_output = 0u32;

    for iteration in 0..max_iterations {
        // Check timeout
        if start_time.elapsed().as_secs() > QUERY_TIMEOUT_SECS {
            warn!("Query timeout after {}s", start_time.elapsed().as_secs());
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
            iteration_prompt(query, iteration as usize, max_iterations as usize, &variables_info)
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
        info!(iteration, len = content.len(), elapsed_s = start_time.elapsed().as_secs(), "LLM response");

        // Add assistant response to conversation
        messages.push(Message {
            role: "assistant".into(),
            content: content.clone(),
        });
        // We'll add the full trajectory entry (with code_blocks + repl_output) after execution

        // Check for FINAL() in plain text (outside code blocks)
        if let Some(answer) = extract_final_answer(&content) {
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
                    return ("completed".into(), result_entries, total_input, total_output, trajectory);
                }
            } else {
                result_entries.push(ResultEntry {
                    entry_type: "output_text".into(),
                    tool_name: None,
                    arguments: None,
                    output: Some(serde_json::Value::String(format!("Exact Answer: {answer}"))),
                });
                return ("completed".into(), result_entries, total_input, total_output, trajectory);
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

                    // Truncate to prevent context explosion (DSPy uses 10K, we use 8K)
                    let truncated = if block_output.len() > 8000 {
                        let trunc: String = block_output.chars().take(8000).collect();
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
                            result_entries.push(ResultEntry {
                                entry_type: "output_text".into(),
                                tool_name: None,
                                arguments: None,
                                output: Some(serde_json::Value::String(format!(
                                    "Exact Answer: {answer}"
                                ))),
                            });
                            return (
                                "completed".into(),
                                result_entries,
                                total_input,
                                total_output,
                                trajectory,
                            );
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
            repl_output: if exec_output.is_empty() { None } else { Some(exec_output.clone()) },
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
                return ("completed".into(), result_entries, total_input, total_output, trajectory);
            }
        }
    }

    // Max iterations reached — try fallback extraction (DSPy-style extract_sig)
    info!("Max iterations reached, attempting fallback extraction");
    let (fallback_answer, fb_input, fb_output) =
        fallback_extract(llm, model, query, &messages, rt);
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
    (
        "max_turns_fallback".into(),
        result_entries,
        total_input,
        total_output,
        trajectory,
    )
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
    /// Ollama API URL
    #[arg(long, default_value = "http://localhost:11434", env = "OLLAMA_URL")]
    ollama_url: String,

    /// Ollama model name
    #[arg(long, default_value = "qwen2.5:7b", env = "GW_MODEL")]
    model: String,

    /// BrowseComp-Plus search server URL
    #[arg(long, default_value = "http://localhost:8000")]
    search_url: String,

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

    /// Output directory for result JSON files
    #[arg(long, default_value = "runs/rlm-agent")]
    output_dir: String,
}

fn run_single_query(
    llm: &OllamaClient,
    cli: &Cli,
    query_text: &str,
    query_id: Option<&str>,
    rt: &tokio::runtime::Handle,
) -> RunRecord {
    let bridge = BrowseCompBridge::new(
        cli.search_url.clone(),
        cli.k,
        llm.clone(),
        cli.model.clone(),
        rt.clone(),
    );

    let external_fns = vec![
        "search".to_string(),
        "get_document".to_string(),
        "llm_query".to_string(),
        "batch_llm_query".to_string(),
        "FINAL".to_string(),
    ];

    let mut agent = ReplAgent::new(external_fns, Box::new(bridge));

    // Inject question as a variable
    agent
        .set_variable("question", Object::String(query_text.to_string()))
        .ok();

    // Pre-search: decompose query into sub-facts and run diverse searches
    let (context_hits, context_text, pre_input, pre_output) =
        pre_search(llm, &cli.model, query_text, &cli.search_url, cli.k, rt);

    // Inject context as ouros variable (list of {docid, snippet} dicts)
    let context_obj = json_to_object(serde_json::Value::Array(context_hits));
    agent.set_variable("context", context_obj).ok();

    let (status, result_entries, input_tokens, output_tokens, trajectory) =
        run_rlm_loop(llm, &cli.model, &mut agent, query_text, cli.max_turns, rt, &context_text);
    let input_tokens = input_tokens + pre_input;
    let output_tokens = output_tokens + pre_output;

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

    RunRecord {
        metadata: RunMetadata {
            model: cli.model.clone(),
            ollama_url: cli.ollama_url.clone(),
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
        status,
        retrieved_docids: Vec::new(), // TODO: extract from bridge
        result: result_entries,
        trajectory,
    }
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
            first_seen.get(&a.0).unwrap_or(&0).cmp(first_seen.get(&b.0).unwrap_or(&0))
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
    query_text: &str,
    query_id: Option<&str>,
    rt: &tokio::runtime::Handle,
    n_runs: u32,
) -> RunRecord {
    if n_runs <= 1 {
        return run_single_query(llm, cli, query_text, query_id, rt);
    }

    let mut records: Vec<RunRecord> = Vec::with_capacity(n_runs as usize);
    let mut answers: Vec<String> = Vec::new();

    for run_idx in 0..n_runs {
        info!(run = run_idx + 1, total_runs = n_runs, "Best-of-N run");
        let record = run_single_query(llm, cli, query_text, query_id, rt);
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
        last_entry.output = Some(serde_json::Value::String(format!("Exact Answer: {voted_answer}")));
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
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "gw_bench=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let rt = tokio::runtime::Handle::current();

    let llm = OllamaClient::new(
        cli.ollama_url.clone(),
        cli.ollama_url.clone(),
        cli.model.clone(),
        "nomic-embed-text".into(),
    );

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
            "Processing queries"
        );

        for (i, (qid, query_text)) in remaining.iter().enumerate() {
            info!(
                progress = format!("{}/{}", i + 1, remaining.len()),
                query_id = qid,
                "Running query"
            );

            let record = tokio::task::block_in_place(|| {
                run_with_voting(&llm, &cli, query_text, Some(qid), &rt, cli.runs)
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
            run_with_voting(&llm, &cli, &cli.query, None, &rt, cli.runs)
        });

        let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
        let filename = output_dir.join(format!("run_{ts}.json"));
        let json = serde_json::to_string_pretty(&record).unwrap();
        std::fs::write(&filename, json).expect("Failed to write result");

        info!(path = %filename.display(), "Saved");
    }
}
