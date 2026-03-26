//! `hindsight-opinions` plugin — confidence-scored belief evolution.
//!
//! Implements Hindsight CARA opinion tracking (design-hindsight-memory.md §2.5).
//!
//! Provides host functions for Python agents to:
//! - Propose opinions with confidence scores
//! - Reinforce / weaken / contradict existing opinions
//! - Query opinions by entity
//!
//! The actual confidence updates go through `PgMemoryStore::update_confidence()`
//! via the host functions. The `BeforeMemoryStore` handler at priority 55 detects
//! opinion-type memories and ensures they have a default confidence.
//!
//! Full LLM-powered assessment (automatically detecting reinforce/weaken/contradict
//! when new facts arrive) is deferred to async dispatch resolution (§6.2 Q7).

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{
    EventData, EventPayload, EventResult, LifecycleEvent, Plugin, PluginContext, PluginError,
    PluginManifest,
};
use serde_json::Value;
use tracing::debug;

/// Configuration for the hindsight-opinions plugin.
struct OpinionsConfig {
    /// Confidence adjustment for reinforcement.
    reinforce_alpha: f64,
    /// Confidence adjustment for weakening.
    weaken_alpha: f64,
    /// Confidence adjustment for contradiction.
    contradict_alpha: f64,
    /// Default confidence for new opinions.
    default_confidence: f64,
}

impl OpinionsConfig {
    fn from_plugin_config(config: &Value) -> Self {
        Self {
            reinforce_alpha: config
                .get("reinforce_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.15),
            weaken_alpha: config
                .get("weaken_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.10),
            contradict_alpha: config
                .get("contradict_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25),
            default_confidence: config
                .get("default_confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5),
        }
    }
}

/// Built-in plugin for opinion confidence tracking.
pub struct HindsightOpinionsPlugin;

impl Plugin for HindsightOpinionsPlugin {
    fn name(&self) -> &str {
        "hindsight-opinions"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "memory:opinions".into(),
                "host_fn:memory.opinion_reinforce".into(),
                "host_fn:memory.opinion_contradict".into(),
            ],
            requires: vec!["memory:retain".into()],
            priority: 55, // After retain (50), before recall (60)
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let config = OpinionsConfig::from_plugin_config(ctx.config);
        let default_confidence = config.default_confidence;

        // --- BeforeMemoryStore handler ---
        // Ensures opinion-type memories have a default confidence if none was set.
        // Runs after hindsight-retain (priority 50) which classifies the kind.
        ctx.on(
            LifecycleEvent::BeforeMemoryStore,
            Arc::new(move |payload: &mut EventPayload| {
                let EventData::Memory { ref mut meta, .. } = payload.data else {
                    return EventResult::Continue;
                };

                let Some(Value::Object(ref mut m)) = meta else {
                    return EventResult::Continue;
                };

                // Only touch opinions
                let is_opinion = m
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .map(|k| k == "opinion")
                    .unwrap_or(false);

                if !is_opinion {
                    return EventResult::Continue;
                }

                // Set default confidence if not already present
                if m.get("confidence").is_none() {
                    if let Some(n) = serde_json::Number::from_f64(default_confidence) {
                        m.insert("confidence".into(), Value::Number(n));
                        debug!(confidence = default_confidence, "opinions plugin set default confidence");
                        return EventResult::Modified;
                    }
                }

                EventResult::Continue
            }),
        );

        // --- Host functions ---
        // These provide the confidence update API for Python agents.
        // The actual DB update is deferred — host functions currently return
        // the computed delta. When the host function router gains DB access
        // (via SharedState), these will call PgMemoryStore::update_confidence().

        let reinforce_alpha = config.reinforce_alpha;
        ctx.register_host_fn(
            "memory.opinion_reinforce",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let key = args.first().and_then(|v| v.as_str()).unwrap_or("");
                if key.is_empty() {
                    return Err(PluginError::HostFunction("key required".into()));
                }
                Ok(serde_json::json!({
                    "key": key,
                    "action": "reinforce",
                    "delta": reinforce_alpha,
                }))
            }),
        );

        let weaken_alpha = config.weaken_alpha;
        ctx.register_host_fn(
            "memory.opinion_weaken",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let key = args.first().and_then(|v| v.as_str()).unwrap_or("");
                if key.is_empty() {
                    return Err(PluginError::HostFunction("key required".into()));
                }
                Ok(serde_json::json!({
                    "key": key,
                    "action": "weaken",
                    "delta": -weaken_alpha,
                }))
            }),
        );

        let contradict_alpha = config.contradict_alpha;
        ctx.register_host_fn(
            "memory.opinion_contradict",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let key = args.first().and_then(|v| v.as_str()).unwrap_or("");
                if key.is_empty() {
                    return Err(PluginError::HostFunction("key required".into()));
                }
                Ok(serde_json::json!({
                    "key": key,
                    "action": "contradict",
                    "delta": -contradict_alpha,
                }))
            }),
        );

        Ok(())
    }
}

/// Compute confidence reinforcement rules (pure function, no DB).
///
/// Used by both the plugin and `PgMemoryStore::update_confidence()`.
pub fn confidence_delta(action: &str, config: &OpinionsDelta) -> f32 {
    match action {
        "reinforce" => config.reinforce_alpha,
        "weaken" => -config.weaken_alpha,
        "contradict" => -config.contradict_alpha,
        _ => 0.0,
    }
}

/// Confidence adjustment parameters, extractable from plugin config.
pub struct OpinionsDelta {
    pub reinforce_alpha: f32,
    pub weaken_alpha: f32,
    pub contradict_alpha: f32,
}

impl Default for OpinionsDelta {
    fn default() -> Self {
        Self {
            reinforce_alpha: 0.15,
            weaken_alpha: 0.10,
            contradict_alpha: 0.25,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn confidence_delta_reinforce() {
        let d = OpinionsDelta::default();
        assert!((confidence_delta("reinforce", &d) - 0.15).abs() < 0.001);
    }

    #[test]
    fn confidence_delta_weaken() {
        let d = OpinionsDelta::default();
        assert!((confidence_delta("weaken", &d) - (-0.10)).abs() < 0.001);
    }

    #[test]
    fn confidence_delta_contradict() {
        let d = OpinionsDelta::default();
        assert!((confidence_delta("contradict", &d) - (-0.25)).abs() < 0.001);
    }

    #[test]
    fn confidence_delta_neutral() {
        let d = OpinionsDelta::default();
        assert!((confidence_delta("neutral", &d)).abs() < 0.001);
    }
}
