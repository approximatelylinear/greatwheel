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

#[derive(Debug, Serialize)]
struct RunRecord {
    metadata: RunMetadata,
    query_id: Option<String>,
    tool_call_counts: HashMap<String, u32>,
    usage: UsageInfo,
    status: String,
    retrieved_docids: Vec<String>,
    result: Vec<ResultEntry>,
}

#[derive(Debug, Serialize)]
struct RunMetadata {
    model: String,
    ollama_url: String,
    searcher: String,
    max_turns: u32,
    k: u32,
}

#[derive(Debug, Serialize)]
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
            max_llm_calls: 8,
            max_search_calls: 15,
            start_time: std::time::Instant::now(),
            timeout_secs: 90,
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

const SYSTEM_PROMPT: &str = r#"You are a deep research agent tasked with answering difficult questions by searching a corpus of ~100K web documents. You have access to a REPL environment with these functions:

- `search(query)` — BM25 keyword search over the corpus. Returns a list of {docid, snippet} dicts.
- `get_document(docid)` — Retrieve the full text of a document by its docid.
- `llm_query(prompt)` — Query a sub-LLM to analyze text or reason about evidence.
- `batch_llm_query([prompt1, prompt2, ...])` — Run multiple LLM queries IN PARALLEL. Returns a list of responses.
- `print()` — Print output to see results and continue reasoning.

When you want to execute code, wrap it in ```repl``` tags:
```repl
results = search("Tim Ellis Relativity Space")
for r in results:
    print(r["docid"], r["snippet"][:200])
```

SEARCH STRATEGY — this is critical for finding answers:
1. BM25 is keyword-based, NOT semantic. Use exact terms, names, and phrases from the question.
2. Run AT LEAST 3-5 searches per question with DIVERSE query formulations:
   - First: key entities and names from the question
   - Then: synonyms, alternative spellings, related terms
   - Then: broader or narrower slices (e.g., just a person's last name, or a specific date)
   - Try quoting distinctive multi-word phrases
3. When a search returns 0 useful results, DO NOT repeat similar keywords. Try completely different angles.
4. After finding a promising document, use get_document(docid) to read the FULL text — snippets often miss the answer.
5. Use batch_llm_query() to analyze multiple documents in parallel.

When you have found the answer, call FINAL(your_answer) — either in code or as plain text:
```repl
FINAL("Tim Ellis")
```
Or simply write: FINAL(Tim Ellis)

IMPORTANT RULES:
- The answer must be precise — a specific name, number, date, or short phrase.
- NEVER answer "Unable to determine", "None", "Not found", or similar. Always give your best guess based on ANY evidence.
- Include articles ("The"), full names, and exact formatting when relevant.
- Do not explain or hedge in your FINAL answer — just the answer itself.
- If your searches find nothing, reason from what you know and give your best guess anyway."#;

fn iteration_prompt(query: &str, iteration: usize) -> String {
    if iteration == 0 {
        format!(
            "Question: {query}\n\n\
             Start by identifying the key entities, names, and distinctive terms in this question. \
             Then run 2-3 searches with different keyword combinations. Show your reasoning."
        )
    } else if iteration == 1 {
        format!(
            "Question: {query}\n\n\
             Review what you found. If you haven't found the answer yet, try COMPLETELY DIFFERENT search queries — \
             different keywords, alternative phrasings, partial names, related terms. \
             If you found a promising document, use get_document() to read the full text."
        )
    } else {
        format!(
            "Question: {query}\n\n\
             Keep searching with new angles. If you have enough evidence, provide your answer with FINAL()."
        )
    }
}

fn final_prompt(query: &str) -> String {
    format!(
        "You MUST provide a final answer NOW. Based on everything you've found, give the most specific answer you can. \
         NEVER say 'Unable to determine' or 'None' — always give your best guess based on any evidence.\n\n\
         Question: {query}\n\n\
         Provide FINAL(your answer):"
    )
}

// -------------------------------------------------------------------------- //
// rLM Loop
// -------------------------------------------------------------------------- //

const QUERY_TIMEOUT_SECS: u64 = 120; // 2 minutes max per query

fn run_rlm_loop(
    llm: &OllamaClient,
    model: &str,
    agent: &mut ReplAgent,
    query: &str,
    max_iterations: u32,
    rt: &tokio::runtime::Handle,
) -> (String, Vec<ResultEntry>, u32, u32) {
    let start_time = std::time::Instant::now();
    let mut messages: Vec<Message> = vec![Message {
        role: "system".into(),
        content: SYSTEM_PROMPT.to_string(),
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
        // Build iteration prompt
        let user_prompt = if iteration == max_iterations - 1 {
            final_prompt(query)
        } else {
            iteration_prompt(query, iteration as usize)
        };

        messages.push(Message {
            role: "user".into(),
            content: user_prompt,
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

        let content = resp.content.trim().to_string();
        debug!(iteration, len = content.len(), "LLM response");

        // Add assistant response to conversation
        messages.push(Message {
            role: "assistant".into(),
            content: content.clone(),
        });

        // Check for FINAL() in plain text (outside code blocks)
        if let Some(answer) = extract_final_answer(&content) {
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
                    return ("completed".into(), result_entries, total_input, total_output);
                }
            } else {
                result_entries.push(ResultEntry {
                    entry_type: "output_text".into(),
                    tool_name: None,
                    arguments: None,
                    output: Some(serde_json::Value::String(format!("Exact Answer: {answer}"))),
                });
                return ("completed".into(), result_entries, total_input, total_output);
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
                    if !result.stdout.is_empty() {
                        // Truncate output to prevent context explosion
                        let truncated = if result.stdout.len() > 3000 {
                            format!("{}...\n[truncated]", &result.stdout[..3000])
                        } else {
                            result.stdout.clone()
                        };
                        exec_output.push_str(&truncated);
                    }

                    result_entries.push(ResultEntry {
                        entry_type: "tool_call".into(),
                        tool_name: Some("repl".into()),
                        arguments: Some(serde_json::Value::String(block.clone())),
                        output: Some(serde_json::Value::String(result.stdout)),
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

        // Feed execution output back to the LLM
        if !exec_output.is_empty() {
            messages.push(Message {
                role: "user".into(),
                content: format!("REPL output:\n```\n{exec_output}```"),
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
                return ("completed".into(), result_entries, total_input, total_output);
            }
        }
    }

    // Max iterations reached — take whatever we have
    result_entries.push(ResultEntry {
        entry_type: "output_text".into(),
        tool_name: None,
        arguments: None,
        output: Some(serde_json::Value::String(
            "Exact Answer: Unable to determine".into(),
        )),
    });
    (
        "max_turns".into(),
        result_entries,
        total_input,
        total_output,
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
    #[arg(long, default_value_t = 5)]
    k: u32,

    /// Max rLM iterations per query
    #[arg(long, default_value_t = 8)]
    max_turns: u32,

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

    let (status, result_entries, input_tokens, output_tokens) =
        run_rlm_loop(llm, &cli.model, &mut agent, query_text, cli.max_turns, rt);

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
    }
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
                run_single_query(&llm, &cli, query_text, Some(qid), &rt)
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
            run_single_query(&llm, &cli, &cli.query, None, &rt)
        });

        let ts = Utc::now().format("%Y%m%dT%H%M%SZ");
        let filename = output_dir.join(format!("run_{ts}.json"));
        let json = serde_json::to_string_pretty(&record).unwrap();
        std::fs::write(&filename, json).expect("Failed to write result");

        info!(path = %filename.display(), "Saved");
    }
}
