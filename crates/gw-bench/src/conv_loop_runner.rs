//! BrowseComp rLM loop powered by gw-loop ConversationLoop.
//!
//! Replaces the hand-rolled `run_rlm_loop()` with `ConversationLoop::handle_turn()`,
//! gaining structured session trees, tracing spans, and extensibility (branching,
//! compaction, etc.) while preserving the exact iteration prompt logic.

use gw_core::{EntryType, SessionId};
use gw_llm::{Message, OllamaClient};
use gw_loop::{ConversationLoop, IterationCallback, LoopConfig, OllamaLlmClient, SnapshotPolicy};
use gw_runtime::ReplAgent;
use tracing::info;
use uuid::Uuid;

use crate::{
    build_variables_info, fallback_extract, final_prompt, is_refusal_answer, iteration_prompt,
    BenchConfig, ResultEntry, RlmLoopResult, TrajectoryMessage,
};

/// Iteration callback that reproduces the exact gw-bench coaching prompts.
struct BenchIterationCallback {
    query: String,
    pre_search_context: String,
    llm: OllamaClient,
    model: String,
    rt: tokio::runtime::Handle,
    /// Accumulated messages for fallback extraction.
    messages: Vec<Message>,
}

impl IterationCallback for BenchIterationCallback {
    fn before_iteration(
        &mut self,
        iteration: usize,
        max_iterations: usize,
        repl: &ReplAgent,
    ) -> Option<String> {
        let variables_info = build_variables_info(repl);

        let prompt = if iteration == 1 && max_iterations > 1 {
            // First iteration: exploration prompt with context preview.
            let context_section = if !self.pre_search_context.is_empty() {
                format!(
                    "\n\nThe `context` variable has been pre-loaded with {} search results (list of {{docid, snippet}} dicts).\n\
                     Initial search results preview:\n{}",
                    self.pre_search_context.lines().count(),
                    self.pre_search_context,
                )
            } else {
                String::new()
            };
            format!(
                "[Iteration 1/{}] EXPLORE FIRST\n\n\
                 Query: \"{}\"\
                 {context_section}\n\n\
                 Start by examining context. Identify which sub-facts from the query each document might help with. \
                 Use get_document() to read full texts and llm_query() to analyze them.",
                max_iterations, self.query,
            )
        } else if iteration >= max_iterations {
            // Final iteration: force FINAL.
            final_prompt(&self.query, &variables_info)
        } else {
            // Standard iteration with nudges.
            iteration_prompt(&self.query, iteration - 1, max_iterations, &variables_info)
        };

        // Track for fallback extraction.
        self.messages.push(Message {
            role: "user".into(),
            content: prompt.clone(),
        });

        Some(prompt)
    }

    fn on_max_iterations(&mut self, query: &str) -> Option<String> {
        info!("Max iterations reached, attempting fallback extraction");
        let (answer, _, _) =
            fallback_extract(&self.llm, &self.model, query, &self.messages, &self.rt);
        answer
    }
}

/// Run the rLM loop using ConversationLoop infrastructure.
/// Takes a pre-built ReplAgent (with variables already seeded).
pub fn run_rlm_loop_v2_with_agent(
    llm: &OllamaClient,
    model: &str,
    agent: ReplAgent,
    query: &str,
    max_iterations: u32,
    rt: &tokio::runtime::Handle,
    pre_search_context: &str,
    bench_config: &BenchConfig,
) -> RlmLoopResult {
    let session_id = SessionId(Uuid::new_v4());
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

    // Create the LLM client for the conversation loop.
    let loop_llm: Box<dyn gw_loop::LlmClient> = Box::new(OllamaLlmClient::new(llm.clone()));

    let system_prompt = bench_config.system_prompt();

    let callback = BenchIterationCallback {
        query: query.to_string(),
        pre_search_context: pre_search_context.to_string(),
        llm: llm.clone(),
        model: model.to_string(),
        rt: rt.clone(),
        messages: vec![Message {
            role: "system".into(),
            content: system_prompt.clone(),
        }],
    };

    let config = LoopConfig {
        system_prompt,
        recency_window: 50, // Keep all iterations for a single query.
        max_iterations: max_iterations as usize,
        include_code_output: true,
        repl_output_max_chars: bench_config.repl_output_max_chars,
        strip_think_tags: true,
        answer_validator: Some(Box::new(|answer: &str| !is_refusal_answer(answer))),
        iteration_callback: Some(Box::new(callback)),
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 0,
            before_compaction: false,
        },
        compaction_keep_count: 0,
        auto_compact_after_turns: None,
    };

    let mut conv_loop = ConversationLoop::new(session_id, agent, loop_llm, config, event_tx);

    // Run the turn on a separate OS thread with its own tokio runtime.
    // This is necessary because:
    // - ConversationLoop::handle_turn() is async (needs a runtime)
    // - BrowseCompBridge internally calls rt.block_on() for its async operations
    // - tokio forbids nested block_on calls on the same thread
    // By spawning a fresh thread (via thread::scope for non-Send types),
    // we ensure no tokio runtime context is set.
    let query_str = format!("Answer this query: \"{query}\"");
    // Run on a fresh thread to avoid nested block_on.
    // We enter the runtime context (for reqwest/tokio to work) but
    // don't block_on — instead use futures::executor to drive the future.
    // The bridge's self.rt.block_on() calls are safe because they're
    // on the same runtime context that was entered (not nested block_on).
    let rt_handle = rt.clone();
    let turn_result = std::thread::scope(|s| {
        s.spawn(|| {
            // Enter the runtime so tokio-based code (reqwest) works,
            // but don't use block_on (which would prevent bridge from nesting).
            let _guard = rt_handle.enter();
            // Use futures::executor::block_on to drive the async future
            // without setting tokio's "in block_on" flag.
            futures::executor::block_on(conv_loop.handle_turn(&query_str))
        })
        .join()
        .expect("conv-loop thread panicked")
    });

    // Convert SessionTree entries to RlmLoopResult.
    match turn_result {
        Ok(result) => {
            let (result_entries, trajectory) = tree_to_records(&conv_loop);

            // Extract the answer from the turn result.
            if let Some(ref answer) = result.response {
                let mut entries = result_entries;
                entries.push(ResultEntry {
                    entry_type: "output_text".into(),
                    tool_name: None,
                    arguments: None,
                    output: Some(serde_json::Value::String(format!("Exact Answer: {answer}"))),
                });

                RlmLoopResult {
                    status: if result.is_final {
                        "completed"
                    } else {
                        "max_turns_fallback"
                    }
                    .into(),
                    termination_reason: if result.is_final {
                        "final_called"
                    } else {
                        "max_turns"
                    }
                    .into(),
                    iterations_used: result.iterations as u32,
                    result_entries: entries,
                    input_tokens: result.input_tokens,
                    output_tokens: result.output_tokens,
                    trajectory,
                }
            } else {
                RlmLoopResult {
                    status: "max_turns_fallback".into(),
                    termination_reason: "max_turns".into(),
                    iterations_used: result.iterations as u32,
                    result_entries,
                    input_tokens: result.input_tokens,
                    output_tokens: result.output_tokens,
                    trajectory,
                }
            }
        }
        Err(e) => {
            tracing::warn!(error = %e, "ConversationLoop error");
            RlmLoopResult {
                status: "error".into(),
                termination_reason: format!("error: {e}"),
                iterations_used: 0,
                result_entries: vec![],
                input_tokens: 0,
                output_tokens: 0,
                trajectory: vec![],
            }
        }
    }
}

/// Convert SessionTree entries to the existing ResultEntry + TrajectoryMessage format.
fn tree_to_records(conv_loop: &ConversationLoop) -> (Vec<ResultEntry>, Vec<TrajectoryMessage>) {
    let mut result_entries = Vec::new();
    let mut trajectory = Vec::new();

    for entry in conv_loop.tree.entries() {
        match &entry.entry_type {
            EntryType::UserMessage(content) => {
                trajectory.push(TrajectoryMessage {
                    role: "user".into(),
                    content: content.clone(),
                    code_blocks: vec![],
                    repl_output: None,
                });
            }
            EntryType::AssistantMessage { content, .. } => {
                let code_blocks = gw_runtime::extract_code_blocks(content);
                trajectory.push(TrajectoryMessage {
                    role: "assistant".into(),
                    content: content.clone(),
                    code_blocks,
                    repl_output: None,
                });
            }
            EntryType::CodeExecution {
                code,
                stdout,
                result,
            } => {
                let output_str = if !stdout.is_empty() {
                    stdout.clone()
                } else {
                    serde_json::to_string(result).unwrap_or_default()
                };
                result_entries.push(ResultEntry {
                    entry_type: "tool_call".into(),
                    tool_name: Some("repl".into()),
                    arguments: Some(serde_json::Value::String(code.clone())),
                    output: Some(serde_json::Value::String(output_str)),
                });
            }
            EntryType::HostCall { .. }
            | EntryType::ReplSnapshot(_)
            | EntryType::Compaction { .. }
            | EntryType::BranchSummary(_)
            | EntryType::System(_) => {}
        }
    }

    (result_entries, trajectory)
}
