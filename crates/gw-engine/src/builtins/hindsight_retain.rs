//! `hindsight-retain` plugin — enriches memories with Hindsight metadata.
//!
//! On `BeforeMemoryStore`, this plugin:
//! 1. Extracts entities from the memory value via regex NER
//! 2. Classifies the memory kind (fact/experience/opinion/observation)
//! 3. Sets the enriched metadata on the event payload
//!
//! Future work (requires async dispatch, see design-hindsight-memory.md §6.2 Q7):
//! - LLM-powered narrative fact extraction (2-5 facts per exchange)
//! - LLM-powered entity resolution (fuzzy match → canonical)
//! - Causal edge extraction
//! - Graph edge computation (entity, temporal, semantic edges)

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{
    EventData, EventPayload, EventResult, LifecycleEvent, Plugin, PluginContext, PluginError,
    PluginManifest,
};
use regex::Regex;
use serde_json::Value;
use tracing::debug;

/// Configuration for the hindsight-retain plugin.
struct RetainConfig {
    /// Maximum number of entities to extract per memory.
    max_entities: usize,
}

impl RetainConfig {
    fn from_plugin_config(config: &Value) -> Self {
        Self {
            max_entities: config
                .get("max_entities")
                .and_then(|v| v.as_u64())
                .unwrap_or(20) as usize,
        }
    }
}

/// Built-in plugin that enriches memories with Hindsight metadata.
pub struct HindsightRetainPlugin;

impl Plugin for HindsightRetainPlugin {
    fn name(&self) -> &str {
        "hindsight-retain"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["memory:retain".into(), "host_fn:memory.entities".into()],
            requires: vec![],
            priority: 50,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let config = RetainConfig::from_plugin_config(ctx.config);

        // Pre-compile regexes once at init time, shared across handlers
        let patterns = Arc::new(EntityPatterns::new());

        let max_entities = config.max_entities;

        // --- BeforeMemoryStore handler ---
        let patterns_for_store = Arc::clone(&patterns);
        ctx.on(
            LifecycleEvent::BeforeMemoryStore,
            Arc::new(move |payload: &mut EventPayload| {
                let EventData::Memory {
                    ref value,
                    ref mut meta,
                    ..
                } = payload.data
                else {
                    return EventResult::Continue;
                };

                let Some(value) = value else {
                    return EventResult::Continue;
                };

                // Extract text from the value
                let text = match value {
                    Value::String(s) => s.clone(),
                    other => other.to_string(),
                };

                if text.is_empty() {
                    return EventResult::Continue;
                }

                // Extract entities
                let entities = patterns_for_store.extract(&text, max_entities);

                // Classify memory kind
                let kind = classify_kind(&text);

                // Detect confidence for opinion-type memories
                let confidence = if kind == "opinion" {
                    Some(0.5_f64) // Default confidence for new opinions
                } else {
                    None
                };

                // Build or enrich the meta JSON
                let meta_obj = match meta.take() {
                    Some(Value::Object(mut m)) => {
                        // Enrich existing meta — don't overwrite fields the caller set
                        if m.get("kind").is_none() {
                            m.insert("kind".into(), Value::String(kind.into()));
                        }
                        if m.get("entities").is_none() && !entities.is_empty() {
                            m.insert(
                                "entities".into(),
                                Value::Array(entities.iter().map(|e| Value::String(e.clone())).collect()),
                            );
                        }
                        if m.get("confidence").is_none() {
                            if let Some(c) = confidence {
                                m.insert("confidence".into(), Value::Number(serde_json::Number::from_f64(c).unwrap_or(serde_json::Number::from(0))));
                            }
                        }
                        Value::Object(m)
                    }
                    _ => {
                        // Create new meta
                        let mut m = serde_json::Map::new();
                        m.insert("kind".into(), Value::String(kind.into()));
                        if !entities.is_empty() {
                            m.insert(
                                "entities".into(),
                                Value::Array(entities.iter().map(|e| Value::String(e.clone())).collect()),
                            );
                        }
                        if let Some(c) = confidence {
                            m.insert("confidence".into(), Value::Number(serde_json::Number::from_f64(c).unwrap_or(serde_json::Number::from(0))));
                        }
                        Value::Object(m)
                    }
                };

                debug!(
                    kind = kind,
                    entity_count = entities.len(),
                    "hindsight-retain enriched memory"
                );

                *meta = Some(meta_obj);
                EventResult::Modified
            }),
        );

        // --- Host function: memory.entities ---
        // Returns the entity list from a given text (for Python agents to use)
        let patterns_for_host = Arc::clone(&patterns);
        ctx.register_host_fn(
            "memory.extract_entities",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let text = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let entities = patterns_for_host.extract(text, max_entities);
                Ok(Value::Array(
                    entities.into_iter().map(Value::String).collect(),
                ))
            }),
        );

        Ok(())
    }
}

/// Pre-compiled regex patterns for entity extraction.
///
/// Mirrors the logic in `bench/browsecomp/fact_registry.py::_extract_entities`
/// but compiled once at plugin init time.
struct EntityPatterns {
    quoted: Regex,
    capitalized_phrases: Regex,
    proper_nouns: Regex,
    years: Regex,
    stopwords: Vec<&'static str>,
}

impl EntityPatterns {
    fn new() -> Self {
        Self {
            quoted: Regex::new(r#""([^"]{2,60})""#).unwrap(),
            capitalized_phrases: Regex::new(
                r"\b([A-Z][a-z]+(?:\s+(?:[A-Z][a-z]+|of|the|and|de|von|van|le|la|del|al))+)\b",
            )
            .unwrap(),
            // Rust regex doesn't support lookbehind. Match lowercase+space+Capitalized
            // and capture only the capitalized word.
            proper_nouns: Regex::new(r"[a-z,;:]\s([A-Z][a-z]{2,})\b").unwrap(),
            years: Regex::new(r"\b(1[5-9]\d{2}|20\d{2})\b").unwrap(),
            stopwords: vec![
                "The", "This", "That", "These", "Those", "However", "Also",
                "Additionally", "Furthermore", "Moreover", "Therefore", "Because",
                "Although",
            ],
        }
    }

    fn extract(&self, text: &str, max: usize) -> Vec<String> {
        let mut entities = Vec::new();
        let mut seen = std::collections::HashSet::new();

        let mut add = |s: &str| {
            let key = s.to_lowercase();
            if !seen.contains(&key) {
                seen.insert(key);
                entities.push(s.to_string());
            }
        };

        // Quoted strings
        for cap in self.quoted.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                add(m.as_str().trim());
            }
        }

        // Capitalized phrases
        for cap in self.capitalized_phrases.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let phrase = m.as_str().trim();
                if phrase.len() > 3
                    && !self
                        .stopwords
                        .iter()
                        .any(|&sw| sw == phrase)
                {
                    add(phrase);
                }
            }
        }

        // Proper nouns after lowercase context
        for cap in self.proper_nouns.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let word = m.as_str();
                if !self.stopwords.iter().any(|&sw| sw == word) {
                    add(word);
                }
            }
        }

        // Years
        for cap in self.years.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                add(m.as_str());
            }
        }

        entities.truncate(max);
        entities
    }
}

/// Classify a memory's kind based on content heuristics.
///
/// Returns one of: "fact", "experience", "opinion", "observation".
///
/// Matching is conservative to avoid false positives:
/// - Opinion/experience markers require first-person subject ("I"/"we") at
///   sentence boundaries, not inside quoted text.
/// - Observation markers require sentence-initial position or label syntax.
fn classify_kind(text: &str) -> &'static str {
    // Strip quoted passages to avoid matching first-person speech attributed
    // to others (e.g., 'Einstein said: "I found the result"').
    let unquoted = strip_quotes(text);
    let lower = unquoted.to_lowercase();

    // Opinion: first-person belief/preference markers
    if has_sentence_start(&lower, &[
        "i think", "i believe", "in my opinion", "i feel that",
        "i prefer", "i recommend", "i suggest",
    ]) {
        return "opinion";
    }
    // "seems like" / "it seems" are opinion even without "I"
    if has_sentence_start(&lower, &["it seems", "seems like"]) {
        return "opinion";
    }
    // "should" only counts as opinion when the subject is first-person
    if has_sentence_start(&lower, &["i should", "we should", "you should"]) {
        return "opinion";
    }

    // Experience: first-person past actions (at sentence start)
    if has_sentence_start(&lower, &[
        "i did", "i found", "i searched", "i discovered", "i tried",
        "i ran", "i called", "i asked", "i created", "i learned",
        "we found", "we discovered", "we tried", "we ran",
    ]) {
        return "experience";
    }

    // Observation: label-style prefixes (colon-terminated or sentence-initial)
    if has_label_prefix(&lower, &[
        "summary", "observation", "pattern", "note", "key finding", "takeaway",
    ]) {
        return "observation";
    }
    // "in summary" / "to summarize" at sentence start
    if has_sentence_start(&lower, &["in summary", "to summarize"]) {
        return "observation";
    }

    // Default: fact
    "fact"
}

/// Check if any marker appears at a sentence boundary in the text.
///
/// A sentence boundary is: start of string, or after `. ` / `! ` / `? ` / `\n`.
fn has_sentence_start(lower: &str, markers: &[&str]) -> bool {
    for marker in markers {
        // At the very start of text
        if lower.starts_with(marker) {
            return true;
        }
        // After sentence-ending punctuation + space
        for sep in [". ", "! ", "? ", "\n"] {
            if lower.contains(&format!("{sep}{marker}")) {
                return true;
            }
        }
    }
    false
}

/// Check if any label appears as "Label:" or "Label," at the start of text
/// or after a newline.
fn has_label_prefix(lower: &str, labels: &[&str]) -> bool {
    for label in labels {
        // "summary:" or "summary," at start
        if lower.starts_with(&format!("{label}:"))
            || lower.starts_with(&format!("{label},"))
        {
            return true;
        }
        // After newline
        if lower.contains(&format!("\n{label}:"))
            || lower.contains(&format!("\n{label},"))
        {
            return true;
        }
    }
    false
}

/// Remove double-quoted and single-quoted passages from text.
fn strip_quotes(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut in_double = false;
    let mut in_single = false;
    for ch in text.chars() {
        match ch {
            '"' if !in_single => in_double = !in_double,
            '\'' if !in_double => in_single = !in_single,
            _ if !in_double && !in_single => result.push(ch),
            _ => {} // inside quotes — skip
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_entities_basic() {
        let patterns = EntityPatterns::new();
        let text = "Marie Curie won the Nobel Prize in Physics in 1903";
        let entities = patterns.extract(text, 20);

        assert!(entities.iter().any(|e| e.contains("Marie Curie")));
        assert!(entities.iter().any(|e| e.contains("Nobel Prize")));
        assert!(entities.contains(&"1903".to_string()));
    }

    #[test]
    fn extract_entities_quoted() {
        let patterns = EntityPatterns::new();
        let text = r#"The document mentions "Project Apollo" and "lunar module""#;
        let entities = patterns.extract(text, 20);

        assert!(entities.contains(&"Project Apollo".to_string()));
        assert!(entities.contains(&"lunar module".to_string()));
    }

    #[test]
    fn extract_entities_deduplicates() {
        let patterns = EntityPatterns::new();
        let text = "Marie Curie and Marie Curie again, also Marie Curie";
        let entities = patterns.extract(text, 20);

        // "Marie Curie" appears 3 times but should be deduplicated to 1
        let curie_count = entities.iter().filter(|e| *e == "Marie Curie").count();
        assert_eq!(curie_count, 1);
        // Other sub-entities (e.g., "Curie" from proper noun pattern) are separate
    }

    #[test]
    fn extract_entities_respects_max() {
        let patterns = EntityPatterns::new();
        let text = "Alice met Bob at the Nobel Prize ceremony in 1903 in Stockholm Sweden";
        let entities = patterns.extract(text, 2);

        assert!(entities.len() <= 2);
    }

    #[test]
    fn classify_kind_opinion() {
        assert_eq!(classify_kind("I think this approach is better than the alternative"), "opinion");
        assert_eq!(classify_kind("I believe the data shows a trend"), "opinion");
        assert_eq!(classify_kind("It seems like the API changed"), "opinion");
        assert_eq!(classify_kind("First sentence. I recommend using Rust"), "opinion");
    }

    #[test]
    fn classify_kind_opinion_false_positives() {
        // "should" without first-person subject is a fact
        assert_eq!(classify_kind("The experiment should be run at 20°C"), "fact");
        // "probably" embedded in factual text is not enough
        assert_eq!(classify_kind("The server probably handles 1000 RPS"), "fact");
        // Quoted first-person is not the author's opinion
        assert_eq!(classify_kind(r#"Einstein said: "I believe in God""#), "fact");
    }

    #[test]
    fn classify_kind_experience() {
        assert_eq!(classify_kind("I found the document in the archive"), "experience");
        assert_eq!(classify_kind("We discovered a new pattern"), "experience");
        assert_eq!(classify_kind("Done. I searched for the term"), "experience");
    }

    #[test]
    fn classify_kind_experience_false_positives() {
        // Quoted first-person is attributed speech, not agent experience
        assert_eq!(classify_kind(r#"The explorer wrote "I found the ruins""#), "fact");
        // "found" without "I" subject
        assert_eq!(classify_kind("The team found a bug in production"), "fact");
    }

    #[test]
    fn classify_kind_observation() {
        assert_eq!(classify_kind("Summary: the results show improvement"), "observation");
        assert_eq!(classify_kind("Key finding: latency dropped 40%"), "observation");
        assert_eq!(classify_kind("In summary, the approach works"), "observation");
    }

    #[test]
    fn classify_kind_observation_false_positives() {
        // "overall" in a measurement is a fact
        assert_eq!(classify_kind("The overall length is 5 meters"), "fact");
        // "note" in normal prose
        assert_eq!(classify_kind("Please note the deadline"), "fact");
    }

    #[test]
    fn classify_kind_fact() {
        assert_eq!(classify_kind("The Earth orbits the Sun"), "fact");
        assert_eq!(classify_kind("Paris is the capital of France"), "fact");
    }
}
