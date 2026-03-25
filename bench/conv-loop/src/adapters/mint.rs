use serde::Deserialize;
use std::path::Path;

use super::BenchmarkAdapter;
use crate::scenario::{Assertion, Scenario, ScenarioTurn};

/// MINT benchmark adapter.
///
/// Converts MINT tasks (code generation + multi-turn feedback) into
/// conv-loop scenarios. MINT tasks have a problem statement, optional
/// ground truth, and optional multi-turn feedback trajectories.
///
/// Input format: JSONL file where each line is a MintTask.
pub struct MintAdapter;

#[derive(Debug, Deserialize)]
struct MintTask {
    /// Unique task ID.
    id: String,
    /// Problem statement / user instruction.
    problem: String,
    /// Source dataset (e.g., "humaneval", "mbpp", "gsm8k").
    #[serde(default)]
    source: String,
    /// Expected answer or ground truth (for verification).
    #[serde(default)]
    expected: Option<serde_json::Value>,
    /// Variable name to check for the expected answer.
    #[serde(default)]
    check_variable: Option<String>,
    /// Optional feedback turns (simulating user corrections).
    #[serde(default)]
    feedback: Vec<String>,
    /// Optional additional checks (for multi-turn tasks).
    #[serde(default)]
    extra_checks: Vec<MintCheck>,
}

#[derive(Debug, Deserialize)]
struct MintCheck {
    variable: String,
    expected: serde_json::Value,
}

impl BenchmarkAdapter for MintAdapter {
    fn name(&self) -> &str {
        "mint"
    }

    fn load(&self, path: &Path) -> Result<Vec<Scenario>, String> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| format!("failed to read {path:?}: {e}"))?;

        let mut scenarios = Vec::new();

        for (i, line) in content.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            match serde_json::from_str::<MintTask>(line) {
                Ok(task) => {
                    scenarios.push(convert_mint_task(&task));
                }
                Err(e) => {
                    eprintln!("MINT: skipping line {}: {e}", i + 1);
                }
            }
        }

        Ok(scenarios)
    }
}

fn convert_mint_task(task: &MintTask) -> Scenario {
    let source_tag = if task.source.is_empty() {
        "mint".to_string()
    } else {
        format!("mint_{}", task.source)
    };

    let mut turns = Vec::new();

    // Turn 1: the problem statement.
    turns.push(ScenarioTurn::UserMessage {
        content: task.problem.clone(),
        mock_response: None, // Live mode — LLM generates the code.
    });

    // Optional feedback turns (simulating user corrections).
    for feedback in &task.feedback {
        turns.push(ScenarioTurn::UserMessage {
            content: feedback.clone(),
            mock_response: None,
        });
    }

    // Assertions.
    let mut checks = Vec::new();

    if let (Some(expected), Some(var_name)) = (&task.expected, &task.check_variable) {
        checks.push(Assertion::VariableEquals {
            name: var_name.clone(),
            expected: expected.clone(),
        });
    } else if let Some(var_name) = &task.check_variable {
        checks.push(Assertion::VariableExists {
            name: var_name.clone(),
        });
    }

    // Add extra checks for multi-turn verification.
    for extra in &task.extra_checks {
        checks.push(Assertion::VariableEquals {
            name: extra.variable.clone(),
            expected: extra.expected.clone(),
        });
    }

    if !checks.is_empty() {
        turns.push(ScenarioTurn::Assert { checks });
    }

    Scenario {
        name: format!("MINT/{}: {}", task.id, truncate(&task.problem, 50)),
        capability: source_tag,
        system_prompt: mint_system_prompt(),
        recency_window: 100,
        compaction_keep_count: 5,
        turns,
    }
}

fn mint_system_prompt() -> String {
    r#"You are a helpful assistant with a Python REPL. Solve the given problem by writing Python code.

1. Write Python code in ```python fenced blocks.
2. Store your final answer in a variable called `answer`.
3. Do NOT call FINAL() — just execute the code.
Keep responses brief — code only, no explanations."#
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
