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

        // Pre-compile regexes once at init time
        let patterns = EntityPatterns::new();

        let max_entities = config.max_entities;

        // --- BeforeMemoryStore handler ---
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
                let entities = patterns.extract(&text, max_entities);

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
                                m.insert("confidence".into(), Value::Number(serde_json::Number::from_f64(c).unwrap()));
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
                            m.insert("confidence".into(), Value::Number(serde_json::Number::from_f64(c).unwrap()));
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
        let patterns_for_host = EntityPatterns::new();
        ctx.register_host_fn(
            "memory.extract_entities",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let text = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");
                let entities = patterns_for_host.extract(text, 30);
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
fn classify_kind(text: &str) -> &'static str {
    let lower = text.to_lowercase();

    // Opinion indicators: subjective language, belief markers
    let opinion_markers = [
        "i think",
        "i believe",
        "in my opinion",
        "i feel",
        "seems like",
        "probably",
        "might be",
        "i prefer",
        "i recommend",
        "should",
        "better than",
        "worse than",
    ];
    if opinion_markers.iter().any(|m| lower.contains(m)) {
        return "opinion";
    }

    // Experience indicators: first-person past actions
    let experience_markers = [
        "i did",
        "i found",
        "i searched",
        "i discovered",
        "i tried",
        "i ran",
        "i called",
        "i asked",
        "i created",
        "i learned",
        "we found",
        "we discovered",
    ];
    if experience_markers.iter().any(|m| lower.contains(m)) {
        return "experience";
    }

    // Observation indicators: summaries, patterns
    let observation_markers = [
        "summary:",
        "in summary",
        "overall",
        "pattern:",
        "observation:",
        "note:",
        "key finding",
        "takeaway",
    ];
    if observation_markers.iter().any(|m| lower.contains(m)) {
        return "observation";
    }

    // Default: fact
    "fact"
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
    }

    #[test]
    fn classify_kind_experience() {
        assert_eq!(classify_kind("I found the document in the archive"), "experience");
        assert_eq!(classify_kind("We discovered a new pattern"), "experience");
    }

    #[test]
    fn classify_kind_observation() {
        assert_eq!(classify_kind("Summary: the results show improvement"), "observation");
        assert_eq!(classify_kind("Key finding: latency dropped 40%"), "observation");
    }

    #[test]
    fn classify_kind_fact() {
        assert_eq!(classify_kind("The Earth orbits the Sun"), "fact");
        assert_eq!(classify_kind("Paris is the capital of France"), "fact");
    }
}
