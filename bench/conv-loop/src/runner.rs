use gw_core::{EntryType, ReplSnapshotData, SessionId};
use gw_loop::SessionTree;
use gw_runtime::{extract_code_blocks, HostBridge, ReplAgent};
use ouros::Object;
use serde_json::Value;
use std::collections::HashMap;
use uuid::Uuid;

use crate::assertions::{evaluate_assertion, AssertionResult};
use crate::mock::MockBridge;
use crate::scenario::{Scenario, ScenarioTurn};

pub fn run_scenario(scenario: &Scenario) -> Vec<AssertionResult> {
    let session_id = SessionId(Uuid::new_v4());
    let mut tree = SessionTree::new(session_id);

    let bridge = Box::new(MockBridge);
    let mut repl = ReplAgent::new(vec!["FINAL".into()], bridge);

    let mut results = Vec::new();

    for turn in &scenario.turns {
        match turn {
            ScenarioTurn::UserMessage {
                content,
                mock_response,
            } => {
                // Append user message to tree.
                tree.append(EntryType::UserMessage(content.clone()));

                // In deterministic mode, use mock_response as the assistant reply.
                if let Some(response) = mock_response {
                    // Extract and execute code blocks.
                    let blocks = extract_code_blocks(response);
                    for block in &blocks {
                        match repl.execute(block) {
                            Ok(exec_result) => {
                                tree.append(EntryType::CodeExecution {
                                    code: block.clone(),
                                    stdout: exec_result.stdout.clone(),
                                    result: exec_result.value.clone(),
                                });
                            }
                            Err(e) => {
                                tracing::warn!(error = %e, "code execution failed");
                                tree.append(EntryType::CodeExecution {
                                    code: block.clone(),
                                    stdout: String::new(),
                                    result: serde_json::json!({"error": e.to_string()}),
                                });
                            }
                        }
                    }

                    // Append assistant message to tree.
                    tree.append(EntryType::AssistantMessage {
                        content: response.clone(),
                        model: None,
                    });
                }
            }

            ScenarioTurn::Assert { checks } => {
                for check in checks {
                    results.push(evaluate_assertion(
                        check,
                        &repl,
                        &tree,
                        &scenario.system_prompt,
                        scenario.recency_window,
                    ));
                }
            }

            ScenarioTurn::Snapshot => {
                // Save REPL snapshot to tree.
                let variables = serde_json::to_value(repl.get_all_variables())
                    .unwrap_or(serde_json::Value::Object(Default::default()));
                let definitions = repl.get_definitions();
                let raw_bytes = repl.save_snapshot().ok();

                tree.append(EntryType::ReplSnapshot(ReplSnapshotData {
                    variables,
                    definitions,
                    raw_bytes,
                }));
            }

            ScenarioTurn::RestoreSnapshot => {
                // Simulate session suspend/resume: restore REPL from latest snapshot.
                if let Some(leaf) = tree.active_leaf() {
                    if let Some(snapshot) = tree.find_latest_snapshot(leaf) {
                        if let Some(raw_bytes) = &snapshot.raw_bytes {
                            match ReplAgent::restore_snapshot(raw_bytes, Box::new(NullBridge)) {
                                Ok(restored) => {
                                    repl = restored;
                                }
                                Err(e) => {
                                    tracing::warn!(error = %e, "snapshot restore failed, replaying code");
                                    repl = replay_repl(&tree);
                                }
                            }
                        } else {
                            repl = replay_repl(&tree);
                        }
                    } else {
                        tracing::warn!("no snapshot found, replaying code");
                        repl = replay_repl(&tree);
                    }
                }
            }

            ScenarioTurn::Compact => {
                // In deterministic mode, compaction is a no-op since we don't have an LLM
                // to generate summaries. The tree state is preserved.
                tracing::info!("compact requested (no-op in deterministic mode)");
            }

            ScenarioTurn::Steering { .. } => {
                // No-op in deterministic mode — steering affects LLM context only.
                tracing::info!("steering injected (no-op in deterministic mode)");
            }

            ScenarioTurn::FollowUp { .. } => {
                // No-op in deterministic mode — follow-ups are processed by the loop.
                tracing::info!("follow-up queued (no-op in deterministic mode)");
            }

            ScenarioTurn::SwitchBranch { target_turn } => {
                // Find the last entry associated with the Nth user turn.
                // A turn consists of: UserMessage → CodeExecution(s) → AssistantMessage.
                // We want the AssistantMessage (or last CodeExecution) so the replay
                // includes code from that turn.
                let mut user_count = 0;
                let mut target_id = None;
                let mut in_target_turn = false;
                for entry in tree.entries() {
                    if matches!(entry.entry_type, EntryType::UserMessage(_)) {
                        user_count += 1;
                        if user_count == *target_turn {
                            in_target_turn = true;
                            target_id = Some(entry.id);
                        } else if user_count > *target_turn {
                            break;
                        }
                    } else if in_target_turn {
                        // Keep advancing to include CodeExecution and AssistantMessage
                        // entries that are part of this turn.
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
                    tree.set_active_leaf(target);

                    // Restore REPL from snapshot + replay code after snapshot.
                    repl = restore_repl_to(&tree, target);
                } else {
                    tracing::warn!(target_turn, "switch_branch: user turn not found");
                }
            }
        }
    }

    results
}

/// Create a fresh REPL and replay all CodeExecution entries on the current path.
fn replay_repl(tree: &SessionTree) -> ReplAgent {
    let mut repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    for entry in tree.path_to_leaf() {
        if let EntryType::CodeExecution { code, .. } = &entry.entry_type {
            let _ = repl.execute(code);
        }
    }
    repl
}

/// Create a fresh REPL and replay CodeExecution entries on the path to a specific entry.
fn replay_repl_to(tree: &SessionTree, target: gw_core::EntryId) -> ReplAgent {
    let mut repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
    for entry in tree.path_to(target) {
        if let EntryType::CodeExecution { code, .. } = &entry.entry_type {
            let _ = repl.execute(code);
        }
    }
    repl
}

/// Restore REPL from the latest snapshot on the path, then replay any
/// code executions between the snapshot and the target entry.
fn restore_repl_to(tree: &SessionTree, target: gw_core::EntryId) -> ReplAgent {
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
    replay_repl_to(tree, target)
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
