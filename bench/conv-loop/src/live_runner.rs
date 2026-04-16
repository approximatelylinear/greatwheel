use std::time::Instant;

use gw_core::{EntryType, SessionId};
use gw_llm::OllamaClient;
use gw_loop::{ConversationLoop, LoopConfig, OllamaLlmClient};
use gw_runtime::{HostBridge, ReplAgent};
use ouros::Object;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use crate::assertions::{evaluate_assertion, AssertionResult};
use crate::scenario::{Scenario, ScenarioTurn};

/// Configuration for the live runner.
pub struct LiveConfig {
    pub ollama_url: String,
    pub model: String,
}

/// Result of a live scenario run, including timing.
pub struct LiveScenarioResult {
    pub assertions: Vec<AssertionResult>,
    pub total_turns: usize,
    pub total_input_tokens: u32,
    pub total_output_tokens: u32,
    pub elapsed_ms: u64,
}

/// Run a scenario in live mode using a real LLM.
pub async fn run_scenario_live(scenario: &Scenario, config: &LiveConfig) -> LiveScenarioResult {
    let start = Instant::now();
    let session_id = SessionId(Uuid::new_v4());

    // Create the LLM client.
    let ollama = OllamaClient::new(
        config.ollama_url.clone(),
        config.ollama_url.clone(),
        config.model.clone(),
        "nomic-embed-text".into(),
    );
    let llm: Box<dyn gw_loop::LlmClient> = Box::new(OllamaLlmClient::new(ollama));

    // Create the conversation loop.
    let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
    let repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));

    let system_prompt = if scenario.system_prompt.is_empty() {
        default_system_prompt()
    } else {
        scenario.system_prompt.clone()
    };

    let loop_config = LoopConfig {
        system_prompt,
        recency_window: scenario.recency_window,
        compaction_keep_count: scenario.compaction_keep_count,
        max_iterations: 5,
        ..Default::default()
    };

    let mut conv_loop = ConversationLoop::new(session_id, repl, llm, loop_config, event_tx);

    let mut results = Vec::new();
    let mut total_turns = 0;
    let mut total_input_tokens = 0u32;
    let mut total_output_tokens = 0u32;

    for turn in &scenario.turns {
        match turn {
            ScenarioTurn::UserMessage { content, .. } => {
                // In live mode, ignore mock_response — use the real LLM.
                total_turns += 1;
                println!("    turn {total_turns}: {}", truncate(content, 60));

                match conv_loop.handle_turn(content).await {
                    Ok(result) => {
                        total_input_tokens += result.input_tokens;
                        total_output_tokens += result.output_tokens;

                        let response_preview = result
                            .response
                            .as_deref()
                            .map(|r| truncate(r, 80))
                            .unwrap_or_else(|| "(no response)".into());
                        println!(
                            "      → {} iters, {} in/{} out tokens: {}",
                            result.iterations,
                            result.input_tokens,
                            result.output_tokens,
                            response_preview
                        );
                    }
                    Err(e) => {
                        println!("      → ERROR: {e}");
                    }
                }

                // Process any pending follow-ups.
                for follow_up in conv_loop.drain_follow_ups() {
                    total_turns += 1;
                    println!("    follow-up {total_turns}: {}", truncate(&follow_up, 60));
                    match conv_loop.handle_turn(&follow_up).await {
                        Ok(result) => {
                            total_input_tokens += result.input_tokens;
                            total_output_tokens += result.output_tokens;
                            println!(
                                "      → {} iters, {} in/{} out tokens",
                                result.iterations, result.input_tokens, result.output_tokens,
                            );
                        }
                        Err(e) => {
                            println!("      → follow-up ERROR: {e}");
                        }
                    }
                }
            }

            ScenarioTurn::Assert { checks } => {
                for check in checks {
                    results.push(evaluate_assertion(
                        check,
                        &conv_loop.repl,
                        &conv_loop.tree,
                        conv_loop
                            .tree
                            .path_to_leaf()
                            .first()
                            .map(|_| "")
                            .unwrap_or(""),
                        scenario.recency_window,
                    ));
                }
            }

            ScenarioTurn::Snapshot => {
                let _ = conv_loop.save_snapshot();
                println!("      (snapshot saved)");
            }

            ScenarioTurn::RestoreSnapshot => {
                // Restore REPL from the latest snapshot on the current path.
                if let Some(leaf) = conv_loop.tree.active_leaf() {
                    if let Some(snapshot) = conv_loop.tree.find_latest_snapshot(leaf) {
                        if let Some(raw_bytes) = &snapshot.raw_bytes {
                            match ReplAgent::restore_snapshot(raw_bytes, Box::new(NullBridge)) {
                                Ok(restored) => {
                                    conv_loop.repl = restored;
                                    println!("      (snapshot restored)");
                                }
                                Err(e) => {
                                    println!("      (snapshot restore failed: {e}, replaying)");
                                    conv_loop.repl = replay_repl(&conv_loop.tree);
                                }
                            }
                        } else {
                            println!("      (no raw bytes, replaying code)");
                            conv_loop.repl = replay_repl(&conv_loop.tree);
                        }
                    } else {
                        println!("      (no snapshot found, replaying code)");
                        conv_loop.repl = replay_repl(&conv_loop.tree);
                    }
                }
            }

            ScenarioTurn::SwitchBranch { target_turn } => {
                // Find the last entry of the Nth user turn and switch to it.
                let mut user_count = 0;
                let mut target_id = None;
                let mut in_target_turn = false;
                for entry in conv_loop.tree.entries() {
                    if matches!(entry.entry_type, EntryType::UserMessage(_)) {
                        user_count += 1;
                        if user_count == *target_turn {
                            in_target_turn = true;
                            target_id = Some(entry.id);
                        } else if user_count > *target_turn {
                            break;
                        }
                    } else if in_target_turn {
                        match &entry.entry_type {
                            EntryType::CodeExecution { .. }
                            | EntryType::AssistantMessage { .. }
                            | EntryType::ReplSnapshot(_) => {
                                target_id = Some(entry.id);
                            }
                            _ => {}
                        }
                    }
                }

                if let Some(target) = target_id {
                    conv_loop.tree.set_active_leaf(target);
                    // Restore REPL to the state at the branch point.
                    conv_loop.repl = restore_repl_to(&conv_loop.tree, target);
                    println!("      (switched to branch at turn {target_turn})");
                } else {
                    println!("      (switch_branch: turn {target_turn} not found)");
                }
            }

            ScenarioTurn::Compact => match conv_loop.compact().await {
                Ok(()) => println!("      (compacted)"),
                Err(e) => println!("      (compaction failed: {e})"),
            },

            ScenarioTurn::Steering { content } => {
                conv_loop.inject_steering(content);
                println!("      (steering injected)");
            }

            ScenarioTurn::FollowUp { content } => {
                conv_loop.queue_follow_up(content.clone());
                println!("      (follow-up queued)");
            }
        }
    }

    LiveScenarioResult {
        assertions: results,
        total_turns,
        total_input_tokens,
        total_output_tokens,
        elapsed_ms: start.elapsed().as_millis() as u64,
    }
}

fn default_system_prompt() -> String {
    r#"You are a helpful assistant with a Python REPL. When asked to perform tasks:

1. Write Python code in ```python fenced blocks to accomplish the task.
2. Store results in clearly named variables.
3. When you have the final answer, call FINAL(value) with the result.

Available variables persist between turns. You can reference variables set in previous turns.
Do NOT explain your code unless asked — just write it."#
        .into()
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        format!("{}...", &s[..max])
    }
}

/// Create a fresh REPL by replaying all CodeExecution entries on the current path.
fn replay_repl(tree: &gw_loop::SessionTree) -> ReplAgent {
    let mut repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    for entry in tree.path_to_leaf() {
        if let EntryType::CodeExecution { code, .. } = &entry.entry_type {
            let _ = repl.execute(code);
        }
    }
    repl
}

/// Restore REPL from the latest snapshot on the path to `target`, then replay
/// any code executions after the snapshot.
fn restore_repl_to(tree: &gw_loop::SessionTree, target: gw_core::EntryId) -> ReplAgent {
    let path = tree.path_to(target);

    // Find the most recent snapshot on the path.
    for (i, entry) in path.iter().enumerate().rev() {
        let raw_bytes = match &entry.entry_type {
            EntryType::ReplSnapshot(data) => data.raw_bytes.as_ref(),
            EntryType::Compaction { snapshot, .. } => snapshot.raw_bytes.as_ref(),
            _ => continue,
        };

        if let Some(bytes) = raw_bytes {
            if let Ok(mut repl) = ReplAgent::restore_snapshot(bytes, Box::new(NullBridge)) {
                // Replay code executions after the snapshot.
                for post_entry in &path[i + 1..] {
                    if let EntryType::CodeExecution { code, .. } = &post_entry.entry_type {
                        let _ = repl.execute(code);
                    }
                }
                return repl;
            }
        }
    }

    // No snapshot found — full replay from root.
    let mut repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    for entry in &path {
        if let EntryType::CodeExecution { code, .. } = &entry.entry_type {
            let _ = repl.execute(code);
        }
    }
    repl
}

struct NullBridge;

impl HostBridge for NullBridge {
    fn call(
        &mut self,
        function: &str,
        _args: Vec<Value>,
        _kwargs: HashMap<String, Value>,
    ) -> Result<Object, gw_runtime::AgentError> {
        Err(gw_runtime::AgentError::UnknownFunction(
            function.to_string(),
        ))
    }
}
