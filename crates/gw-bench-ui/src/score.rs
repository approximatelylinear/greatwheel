//! Answer scoring — port of `bench/browsecomp/quick_eval.py`.
//!
//! Two tiers in v1:
//!   - `exact`: case-insensitive substring match of the gold answer in
//!     the agent's answer, with a punctuation-stripped fallback.
//!   - `fuzzy`: any of (a) exact, (b) normalized edit distance / max
//!     length < 0.3, (c) gold-token-set overlap with agent ≥ 0.7,
//!     (d) agent (≥ 4 chars) is a substring of gold.
//!
//! LLM judge is not implemented; the dashboard sticks to the two
//! tiers the original Python evaluator runs without a network call.

use crate::types::ResultEntry;

/// Pull the agent's final answer out of a run's `result` array. Mirror
/// of `extract_agent_answer` in `quick_eval.py`: take the last
/// `output_text` entry, look for an `Exact Answer:` line, otherwise
/// return the whole text.
pub fn extract_agent_answer(result: &[ResultEntry]) -> String {
    let final_text = result
        .iter()
        .rev()
        .find(|e| e.entry_type == "output_text")
        .and_then(|e| e.output.as_ref())
        .and_then(|v| v.as_str())
        .unwrap_or("")
        .to_string();

    if final_text.is_empty() {
        return String::new();
    }

    const EXACT: &str = "exact answer:";
    for line in final_text.lines() {
        let trimmed = line.trim();
        let lower = trimmed.to_lowercase();
        // Prefix is ASCII, so byte and char offsets match — slicing
        // `trimmed[EXACT.len()..]` preserves the original casing.
        if lower.starts_with(EXACT) {
            return trimmed[EXACT.len()..].trim().to_string();
        }
        if lower.starts_with("**exact answer") {
            let cleaned: String = trimmed.chars().filter(|c| *c != '*').collect();
            if let Some((_, after)) = cleaned.split_once(':') {
                return after.trim().to_string();
            }
        }
    }

    final_text
}

pub fn score_exact(agent: &str, gold: &str) -> bool {
    if agent.is_empty() || gold.is_empty() {
        return false;
    }
    let a_lower = agent.to_lowercase();
    let g_lower = gold.to_lowercase();
    let a_lower = a_lower.trim();
    let g_lower = g_lower.trim();
    if a_lower.contains(g_lower) {
        return true;
    }
    let a_clean = strip_punctuation(a_lower);
    let g_clean = strip_punctuation(g_lower);
    if !g_clean.is_empty() && a_clean.contains(&g_clean) {
        return true;
    }
    false
}

pub fn score_fuzzy(agent: &str, gold: &str) -> bool {
    if agent.is_empty() || gold.is_empty() {
        return false;
    }
    if score_exact(agent, gold) {
        return true;
    }
    let a_norm = normalize_answer(agent);
    let g_norm = normalize_answer(gold);
    if a_norm.is_empty() || g_norm.is_empty() {
        return false;
    }
    let dist = edit_distance(&a_norm, &g_norm);
    let max_len = a_norm.chars().count().max(g_norm.chars().count());
    if max_len > 0 && (dist as f64) / (max_len as f64) < 0.3 {
        return true;
    }
    let g_tokens: Vec<&str> = g_norm.split_whitespace().collect();
    let a_tokens: std::collections::HashSet<&str> = a_norm.split_whitespace().collect();
    if !g_tokens.is_empty() {
        let overlap = g_tokens.iter().filter(|t| a_tokens.contains(*t)).count();
        if (overlap as f64) / (g_tokens.len() as f64) >= 0.7 {
            return true;
        }
    }
    if a_norm.chars().count() >= 4 && g_norm.contains(&a_norm) {
        return true;
    }
    false
}

fn normalize_answer(text: &str) -> String {
    let lower = text.to_lowercase();
    let mut s: &str = lower.trim();
    for prefix in ["the ", "a ", "an "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            s = rest;
            break;
        }
    }
    let no_punct = strip_punctuation(s);
    no_punct.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn strip_punctuation(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric() || c.is_whitespace())
        .collect()
}

/// Standard Wagner–Fischer over Unicode scalars. Bench answers are
/// short (single names, dates, places), so the quadratic cost is
/// inconsequential.
fn edit_distance(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    if a.len() < b.len() {
        return edit_distance_inner(&b, &a);
    }
    edit_distance_inner(&a, &b)
}

fn edit_distance_inner(a: &[char], b: &[char]) -> usize {
    if b.is_empty() {
        return a.len();
    }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut curr: Vec<usize> = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        curr[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            let cost = if ca == cb { 0 } else { 1 };
            curr[j + 1] = (curr[j] + 1).min(prev[j + 1] + 1).min(prev[j] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[b.len()]
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::ResultEntry;
    use serde_json::json;

    #[test]
    fn exact_substring_match() {
        assert!(score_exact("The answer is Tagalog.", "Tagalog"));
        assert!(score_exact("tagalog", "Tagalog"));
        assert!(!score_exact("Spanish", "Tagalog"));
    }

    #[test]
    fn exact_strips_trailing_punctuation() {
        // Substring path doesn't catch this (gold has a `!`); the
        // strip-punctuation fallback does.
        assert!(score_exact("the answer is tagalog", "tagalog!"));
        // Substring path catches the agent-side punctuation case.
        assert!(score_exact("Tagalog.", "Tagalog"));
    }

    #[test]
    fn fuzzy_token_overlap() {
        // 3-of-3 tokens overlap → ≥0.7.
        assert!(score_fuzzy("Richard Larson Jr", "Richard Larson"));
    }

    #[test]
    fn fuzzy_edit_distance() {
        // One-char typo across an 8-char string → 1/8 < 0.3.
        assert!(score_fuzzy("Tagaloog", "Tagalog"));
    }

    #[test]
    fn fuzzy_agent_substring_of_gold() {
        assert!(score_fuzzy(
            "Gingras Trading Post",
            "The Gingras Trading Post State Historic Site"
        ));
    }

    #[test]
    fn fuzzy_rejects_unrelated() {
        assert!(!score_fuzzy("Spanish", "Tagalog"));
    }

    #[test]
    fn extract_pulls_exact_answer_line() {
        let result = vec![
            ResultEntry {
                entry_type: "tool_call".into(),
                tool_name: Some("search".into()),
                arguments: None,
                output: None,
            },
            ResultEntry {
                entry_type: "output_text".into(),
                tool_name: None,
                arguments: None,
                output: Some(json!(
                    "Reasoning: ...\nExact Answer: Philippines\nConfidence: high"
                )),
            },
        ];
        assert_eq!(extract_agent_answer(&result), "Philippines");
    }

    #[test]
    fn extract_handles_bold_markdown() {
        let result = vec![ResultEntry {
            entry_type: "output_text".into(),
            tool_name: None,
            arguments: None,
            output: Some(json!("**Exact Answer:** Tagalog\n")),
        }];
        assert_eq!(extract_agent_answer(&result), "Tagalog");
    }

    #[test]
    fn extract_falls_back_to_full_text() {
        let result = vec![ResultEntry {
            entry_type: "output_text".into(),
            tool_name: None,
            arguments: None,
            output: Some(json!("Just some text.")),
        }];
        assert_eq!(extract_agent_answer(&result), "Just some text.");
    }
}
