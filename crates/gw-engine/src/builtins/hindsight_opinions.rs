//! `hindsight-opinions` plugin — confidence-scored belief evolution.
//!
//! Implements Hindsight CARA opinion tracking (design-hindsight-memory.md §2.5).
//!
//! Provides host functions for Python agents to:
//! - Reinforce / weaken / contradict existing opinions (persisted to Postgres)
//! - Query opinions by entity
//!
//! The `BeforeMemoryStore` handler at priority 55 detects opinion-type memories
//! and ensures they have a default confidence.
//!
//! **Requires:** PgPool provided via `engine.provide(pool)` in gw-server.
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
use sqlx::PgPool;
use tracing::{debug, warn};

/// Configuration for the hindsight-opinions plugin.
///
/// All values are f32 to match the Postgres FLOAT column and scoring math.
struct OpinionsConfig {
    reinforce_alpha: f32,
    weaken_alpha: f32,
    contradict_alpha: f32,
    default_confidence: f32,
}

impl OpinionsConfig {
    fn from_plugin_config(config: &Value) -> Self {
        Self {
            reinforce_alpha: config
                .get("reinforce_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.15) as f32,
            weaken_alpha: config
                .get("weaken_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.10) as f32,
            contradict_alpha: config
                .get("contradict_alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.25) as f32,
            default_confidence: config
                .get("default_confidence")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5) as f32,
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

        // Grab PgPool from SharedState (provided by gw-server via engine.provide())
        let pool: Option<PgPool> = ctx.shared.get::<PgPool>().cloned();
        if pool.is_none() {
            warn!("hindsight-opinions: PgPool not found in SharedState — host functions will be no-ops");
        }

        // --- BeforeMemoryStore handler ---
        ctx.on(
            LifecycleEvent::BeforeMemoryStore,
            Arc::new(move |payload: &mut EventPayload| {
                let EventData::Memory { ref mut meta, .. } = payload.data else {
                    return EventResult::Continue;
                };

                let Some(Value::Object(ref mut m)) = meta else {
                    return EventResult::Continue;
                };

                let is_opinion = m
                    .get("kind")
                    .and_then(|v| v.as_str())
                    .map(|k| k == "opinion")
                    .unwrap_or(false);

                if !is_opinion {
                    return EventResult::Continue;
                }

                if m.get("confidence").is_none() {
                    if let Some(n) = serde_json::Number::from_f64(f64::from(default_confidence)) {
                        m.insert("confidence".into(), Value::Number(n));
                        debug!(confidence = default_confidence, "opinions plugin set default confidence");
                        return EventResult::Modified;
                    }
                }

                EventResult::Continue
            }),
        );

        // --- Host functions ---
        // Each calls update_confidence() via block_in_place.
        // Signed deltas are computed by confidence_delta() — OpinionsDelta
        // stores magnitudes, the function applies the sign.
        // Args: [org_id (UUID string), key (string)]

        let deltas = OpinionsDelta {
            reinforce_alpha: config.reinforce_alpha,
            weaken_alpha: config.weaken_alpha,
            contradict_alpha: config.contradict_alpha,
        };

        let delta_r = confidence_delta("reinforce", &deltas);
        let pool_r = pool.clone();
        ctx.register_host_fn(
            "memory.opinion_reinforce",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                update_opinion(&pool_r, &args, delta_r, "reinforce")
            }),
        );

        let delta_w = confidence_delta("weaken", &deltas);
        let pool_w = pool.clone();
        ctx.register_host_fn(
            "memory.opinion_weaken",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                update_opinion(&pool_w, &args, delta_w, "weaken")
            }),
        );

        let delta_c = confidence_delta("contradict", &deltas);
        let pool_c = pool;
        ctx.register_host_fn(
            "memory.opinion_contradict",
            Arc::new(move |args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                update_opinion(&pool_c, &args, delta_c, "contradict")
            }),
        );

        Ok(())
    }
}

/// Execute an opinion confidence update against Postgres.
///
/// Args: [org_id (UUID string), key (string)]
fn update_opinion(
    pool: &Option<PgPool>,
    args: &[Value],
    delta: f32,
    action: &str,
) -> Result<Value, PluginError> {
    let Some(pool) = pool else {
        return Ok(serde_json::json!({
            "error": "PgPool not available — opinion updates require database access",
            "action": action,
            "delta": delta,
        }));
    };

    let org_id_str = args
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction("org_id (UUID string) required as first arg".into()))?;
    let org_id: uuid::Uuid = org_id_str
        .parse()
        .map_err(|e| PluginError::HostFunction(format!("invalid org_id: {e}")))?;

    let key = args
        .get(1)
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction("key required as second arg".into()))?;

    // Run the async DB update synchronously within the tokio runtime.
    let result = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            sqlx::query_as::<_, (f32,)>(
                r#"
                UPDATE memories
                SET confidence = GREATEST(0.0, LEAST(1.0, COALESCE(confidence, 0.5) + $3)),
                    updated_at = now()
                WHERE org_id = $1 AND key = $2 AND kind = 'opinion'
                RETURNING confidence
                "#,
            )
            .bind(org_id)
            .bind(key)
            .bind(delta)
            .fetch_optional(pool)
            .await
        })
    });

    match result {
        Ok(Some((new_confidence,))) => {
            debug!(key, action, new_confidence, "opinion confidence updated");
            Ok(serde_json::json!({
                "key": key,
                "action": action,
                "confidence": new_confidence,
            }))
        }
        Ok(None) => Ok(serde_json::json!({
            "key": key,
            "action": action,
            "error": "not found or not an opinion",
        })),
        Err(e) => {
            warn!(key, action, error = %e, "opinion update failed");
            Err(PluginError::HostFunction(format!("DB error: {e}")))
        }
    }
}

/// Compute a signed confidence delta from an action name and magnitude config.
///
/// Returns positive for reinforcement, negative for weaken/contradict.
/// This is the single place where the sign convention is applied —
/// `OpinionsDelta` stores magnitudes (always positive), and this function
/// adds the sign based on the action.
pub fn confidence_delta(action: &str, config: &OpinionsDelta) -> f32 {
    match action {
        "reinforce" => config.reinforce_alpha,
        "weaken" => -config.weaken_alpha,
        "contradict" => -config.contradict_alpha,
        _ => 0.0,
    }
}

/// Confidence adjustment magnitudes (always positive).
///
/// Use [`confidence_delta()`] to get signed values for a specific action.
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

    #[test]
    fn update_opinion_no_pool() {
        let result = update_opinion(&None, &[Value::String("org".into()), Value::String("key".into())], 0.15, "reinforce").unwrap();
        assert!(result.get("error").is_some());
    }
}
