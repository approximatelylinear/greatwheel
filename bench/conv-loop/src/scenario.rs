use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub capability: String,
    #[serde(default)]
    pub system_prompt: String,
    #[serde(default = "default_recency_window")]
    pub recency_window: usize,
    #[serde(default = "default_compaction_keep")]
    pub compaction_keep_count: usize,
    pub turns: Vec<ScenarioTurn>,
}

fn default_recency_window() -> usize {
    100
}

fn default_compaction_keep() -> usize {
    5
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "type")]
pub enum ScenarioTurn {
    #[serde(rename = "user_message")]
    UserMessage {
        content: String,
        /// Mock assistant response (used in deterministic mode).
        mock_response: Option<String>,
    },
    #[serde(rename = "assert")]
    Assert { checks: Vec<Assertion> },
    /// Save a REPL snapshot to the tree.
    #[serde(rename = "snapshot")]
    Snapshot,
    /// Restore the REPL from the latest snapshot on the current path.
    /// Simulates a session suspend/resume cycle.
    #[serde(rename = "restore_snapshot")]
    RestoreSnapshot,
    /// Switch the active branch to the entry at `target_turn` (1-indexed user turn number).
    /// Entries are matched by finding the Nth UserMessage entry in the tree.
    #[serde(rename = "switch_branch")]
    SwitchBranch { target_turn: usize },
    /// Trigger compaction of the session tree.
    #[serde(rename = "compact")]
    Compact,
    /// Inject a steering message before the next LLM call.
    #[serde(rename = "steering")]
    Steering { content: String },
    /// Queue a follow-up message to process after the current turn.
    #[serde(rename = "follow_up")]
    FollowUp { content: String },
}

#[derive(Debug, Clone, Deserialize)]
#[serde(tag = "kind")]
pub enum Assertion {
    /// Check that a REPL variable equals an expected JSON value.
    #[serde(rename = "variable_equals")]
    VariableEquals {
        name: String,
        expected: serde_json::Value,
    },
    /// Check that a REPL variable exists (any value).
    #[serde(rename = "variable_exists")]
    VariableExists { name: String },
    /// Check that a function is defined in the REPL.
    #[serde(rename = "function_defined")]
    FunctionDefined { name: String },
    /// Check that the turn context contains a substring.
    #[serde(rename = "context_contains")]
    ContextContains { substring: String },
    /// Check that the turn context does NOT contain a substring.
    #[serde(rename = "context_excludes")]
    ContextExcludes { substring: String },
    /// Check the number of entries in the session tree.
    #[serde(rename = "tree_size_gte")]
    TreeSizeGte { min: usize },
    /// Check the length of the root-to-leaf path.
    #[serde(rename = "path_length")]
    PathLength { expected: usize },
    /// Verify all parent_id references in the tree are valid.
    #[serde(rename = "all_parents_valid")]
    AllParentsValid,
    /// Check that a variable does NOT exist in the REPL.
    #[serde(rename = "variable_absent")]
    VariableAbsent { name: String },
}

pub fn load_scenarios(dir: &Path, filter: Option<&str>) -> Vec<Scenario> {
    let mut scenarios = Vec::new();

    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) => {
            eprintln!("Failed to read scenarios dir {dir:?}: {e}");
            return scenarios;
        }
    };

    for entry in entries.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "toml") {
            let filename = path.file_stem().unwrap_or_default().to_string_lossy();
            if let Some(prefix) = filter {
                if !filename.starts_with(prefix) {
                    continue;
                }
            }
            match std::fs::read_to_string(&path) {
                Ok(content) => match toml::from_str::<Scenario>(&content) {
                    Ok(s) => scenarios.push(s),
                    Err(e) => eprintln!("Failed to parse {path:?}: {e}"),
                },
                Err(e) => eprintln!("Failed to read {path:?}: {e}"),
            }
        }
    }

    scenarios.sort_by(|a, b| a.name.cmp(&b.name));
    scenarios
}
