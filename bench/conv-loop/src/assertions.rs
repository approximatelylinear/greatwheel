use gw_loop::{build_turn_context, SessionTree, TurnContext};
use gw_runtime::ReplAgent;

use crate::scenario::Assertion;

#[derive(Debug)]
pub struct AssertionResult {
    pub passed: bool,
    pub description: String,
    pub detail: Option<String>,
}

pub fn evaluate_assertion(
    assertion: &Assertion,
    repl: &ReplAgent,
    tree: &SessionTree,
    system_prompt: &str,
    recency_window: usize,
) -> AssertionResult {
    match assertion {
        Assertion::VariableEquals { name, expected } => {
            let actual = repl
                .get_variable(name)
                .map(|obj| gw_runtime::object_to_json(&obj));
            let passed = match (&actual, expected) {
                (Some(a), e) => values_approx_equal(a, e),
                _ => false,
            };
            AssertionResult {
                passed,
                description: format!("variable '{name}' == {expected}"),
                detail: if !passed {
                    Some(format!("got {:?}", actual))
                } else {
                    None
                },
            }
        }
        Assertion::VariableExists { name } => {
            let exists = repl.get_variable(name).is_some();
            AssertionResult {
                passed: exists,
                description: format!("variable '{name}' exists"),
                detail: if !exists {
                    Some("variable not found".into())
                } else {
                    None
                },
            }
        }
        Assertion::VariableAbsent { name } => {
            let exists = repl.get_variable(name).is_some();
            AssertionResult {
                passed: !exists,
                description: format!("variable '{name}' absent"),
                detail: if exists {
                    let val = repl
                        .get_variable(name)
                        .map(|o| gw_runtime::object_to_json(&o));
                    Some(format!("variable exists with value {:?}", val))
                } else {
                    None
                },
            }
        }
        Assertion::FunctionDefined { name } => {
            let defs = repl.get_definitions();
            let passed = defs.contains(name);
            AssertionResult {
                passed,
                description: format!("function '{name}' is defined"),
                detail: if !passed {
                    Some(format!("defined functions: {defs:?}"))
                } else {
                    None
                },
            }
        }
        Assertion::ContextContains { substring } => {
            let ctx = build_context(tree, system_prompt, repl, recency_window);
            let haystack = context_to_string(&ctx);
            let passed = haystack.contains(substring.as_str());
            AssertionResult {
                passed,
                description: format!("context contains '{substring}'"),
                detail: if !passed {
                    Some(format!("context length: {} chars", haystack.len()))
                } else {
                    None
                },
            }
        }
        Assertion::ContextExcludes { substring } => {
            let ctx = build_context(tree, system_prompt, repl, recency_window);
            let haystack = context_to_string(&ctx);
            let passed = !haystack.contains(substring.as_str());
            AssertionResult {
                passed,
                description: format!("context excludes '{substring}'"),
                detail: if !passed {
                    Some("substring found in context".into())
                } else {
                    None
                },
            }
        }
        Assertion::TreeSizeGte { min } => {
            let size = tree.entries().len();
            let passed = size >= *min;
            AssertionResult {
                passed,
                description: format!("tree has >= {min} entries"),
                detail: if !passed {
                    Some(format!("got {size}"))
                } else {
                    None
                },
            }
        }
        Assertion::PathLength { expected } => {
            let path_len = tree.path_to_leaf().len();
            let passed = path_len == *expected;
            AssertionResult {
                passed,
                description: format!("path length == {expected}"),
                detail: if !passed {
                    Some(format!("got {path_len}"))
                } else {
                    None
                },
            }
        }
        Assertion::AllParentsValid => {
            let entries = tree.entries();
            let ids: std::collections::HashSet<gw_core::EntryId> =
                entries.iter().map(|e| e.id).collect();

            let mut invalid = Vec::new();
            for entry in entries {
                if let Some(parent_id) = entry.parent_id {
                    if !ids.contains(&parent_id) {
                        invalid.push(format!(
                            "entry {:?} references missing parent {:?}",
                            entry.id, parent_id
                        ));
                    }
                }
            }

            let passed = invalid.is_empty();
            AssertionResult {
                passed,
                description: "all parent references valid".into(),
                detail: if !passed {
                    Some(invalid.join("; "))
                } else {
                    None
                },
            }
        }
    }
}

fn build_context(
    tree: &SessionTree,
    system_prompt: &str,
    repl: &ReplAgent,
    recency_window: usize,
) -> TurnContext {
    let path = tree.path_to_leaf();
    let state_summary = repl.state_summary();
    build_turn_context(&path, system_prompt, &state_summary, recency_window)
}

/// Compare two JSON values with approximate float equality (1e-9 tolerance).
fn values_approx_equal(a: &serde_json::Value, b: &serde_json::Value) -> bool {
    match (a, b) {
        (serde_json::Value::Number(an), serde_json::Value::Number(bn)) => {
            match (an.as_f64(), bn.as_f64()) {
                (Some(af), Some(bf)) => (af - bf).abs() < 1e-9,
                _ => a == b,
            }
        }
        (serde_json::Value::Array(aa), serde_json::Value::Array(ba)) => {
            aa.len() == ba.len()
                && aa
                    .iter()
                    .zip(ba.iter())
                    .all(|(x, y)| values_approx_equal(x, y))
        }
        (serde_json::Value::Object(ao), serde_json::Value::Object(bo)) => {
            ao.len() == bo.len()
                && ao
                    .iter()
                    .all(|(k, v)| bo.get(k).is_some_and(|bv| values_approx_equal(v, bv)))
        }
        _ => a == b,
    }
}

fn context_to_string(ctx: &TurnContext) -> String {
    ctx.messages
        .iter()
        .map(|m| format!("[{}] {}", m.role, m.content))
        .collect::<Vec<_>>()
        .join("\n")
}
