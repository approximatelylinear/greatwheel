use serde::Deserialize;
use std::path::Path;

use super::BenchmarkAdapter;
use crate::scenario::{Assertion, Scenario, ScenarioTurn};

/// tau-bench adapter.
///
/// Converts tau-bench tasks (customer service multi-turn scenarios) into
/// conv-loop scenarios. Each task has a user scenario with instructions
/// and evaluation criteria.
///
/// Input format: JSON file with an array of TauBenchTask objects.
pub struct TauBenchAdapter;

#[derive(Debug, Deserialize)]
struct TauBenchTask {
    id: String,
    #[serde(default)]
    domain: String,
    /// The user's instructions / scenario description.
    user_instruction: String,
    /// Multi-turn dialogue steps.
    #[serde(default)]
    turns: Vec<TauTurn>,
    /// Expected outcomes to verify.
    #[serde(default)]
    expected_outcomes: Vec<TauOutcome>,
}

#[derive(Debug, Deserialize)]
struct TauTurn {
    /// "user" or "agent"
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct TauOutcome {
    /// Variable or state to check.
    check: String,
    /// Expected value.
    expected: serde_json::Value,
}

impl BenchmarkAdapter for TauBenchAdapter {
    fn name(&self) -> &str {
        "tau-bench"
    }

    fn load(&self, path: &Path) -> Result<Vec<Scenario>, String> {
        let content =
            std::fs::read_to_string(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;

        let tasks: Vec<TauBenchTask> = serde_json::from_str(&content)
            .map_err(|e| format!("failed to parse tau-bench JSON: {e}"))?;

        Ok(tasks.iter().map(convert_tau_task).collect())
    }
}

fn convert_tau_task(task: &TauBenchTask) -> Scenario {
    let domain_tag = if task.domain.is_empty() {
        "tau_bench".to_string()
    } else {
        format!("tau_{}", task.domain)
    };

    let mut turns = Vec::new();

    if task.turns.is_empty() {
        // Single-turn: just the user instruction.
        turns.push(ScenarioTurn::UserMessage {
            content: task.user_instruction.clone(),
            mock_response: None,
        });
    } else {
        // Multi-turn: convert each user turn.
        for tau_turn in &task.turns {
            if tau_turn.role == "user" {
                turns.push(ScenarioTurn::UserMessage {
                    content: tau_turn.content.clone(),
                    mock_response: None,
                });
            }
            // Agent turns are skipped in live mode — the LLM generates them.
        }
    }

    // Convert expected outcomes to assertions.
    let mut checks = Vec::new();
    for outcome in &task.expected_outcomes {
        checks.push(Assertion::VariableEquals {
            name: outcome.check.clone(),
            expected: outcome.expected.clone(),
        });
    }

    if !checks.is_empty() {
        turns.push(ScenarioTurn::Assert { checks });
    }

    Scenario {
        name: format!(
            "tau/{}/{}: {}",
            task.domain,
            task.id,
            truncate(&task.user_instruction, 40)
        ),
        capability: domain_tag,
        system_prompt: tau_system_prompt(&task.domain),
        recency_window: 100,
        compaction_keep_count: 5,
        turns,
    }
}

fn tau_system_prompt(domain: &str) -> String {
    format!(
        r#"You are a customer service assistant for {domain}. Help the user with their request.

1. Write Python code in ```python fenced blocks to process requests.
2. Store results in clearly named variables.
3. Do NOT call FINAL() — just execute the code.
4. Use available variables from previous turns.
Keep responses brief — code only, no explanations."#,
        domain = if domain.is_empty() {
            "a company"
        } else {
            domain
        }
    )
}

fn truncate(s: &str, max: usize) -> String {
    let s = s.replace('\n', " ");
    if s.len() <= max {
        s
    } else {
        format!("{}...", &s[..max])
    }
}
