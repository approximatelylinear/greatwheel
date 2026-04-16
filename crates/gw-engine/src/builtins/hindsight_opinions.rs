//! `hindsight-opinions` plugin — confidence-scored belief evolution.
//!
//! Provides host functions for Python agents to reinforce, weaken, or
//! contradict existing opinions (persisted to Postgres).
//!
//! **Requires:** PgPool provided via `engine.provide(pool)` in gw-server.

use std::sync::Arc;

use gw_core::{
    EventData, EventPayload, EventResult, LifecycleEvent, Plugin, PluginContext, PluginError,
    PluginManifest,
};
use serde_json::Value;
use sqlx::PgPool;
use tracing::{debug, warn};

pub struct HindsightOpinionsPlugin;

impl Plugin for HindsightOpinionsPlugin {
    fn name(&self) -> &str {
        "hindsight-opinions"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec!["memory:opinions".into()],
            requires: vec!["memory:retain".into()],
            priority: 55,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let default_confidence = ctx
            .config
            .get("default_confidence")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.5) as f32;

        let reinforce = ctx
            .config
            .get("reinforce_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.15) as f32;
        let weaken = -(ctx
            .config
            .get("weaken_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.10) as f32);
        let contradict = -(ctx
            .config
            .get("contradict_alpha")
            .and_then(|v| v.as_f64())
            .unwrap_or(0.25) as f32);

        let pool: Option<PgPool> = ctx.shared.get::<PgPool>().cloned();
        if pool.is_none() {
            warn!("hindsight-opinions: PgPool not in SharedState — host functions will be no-ops");
        }

        // Set default confidence on new opinions (after retain classifies kind)
        ctx.on(
            LifecycleEvent::BeforeMemoryStore,
            Arc::new(move |payload: &mut EventPayload| {
                let EventData::Memory { ref mut meta, .. } = payload.data else {
                    return EventResult::Continue;
                };
                let Some(Value::Object(ref mut m)) = meta else {
                    return EventResult::Continue;
                };
                let is_opinion = m.get("kind").and_then(|v| v.as_str()) == Some("opinion");
                if !is_opinion || m.contains_key("confidence") {
                    return EventResult::Continue;
                }
                if let Some(n) = serde_json::Number::from_f64(f64::from(default_confidence)) {
                    m.insert("confidence".into(), Value::Number(n));
                    return EventResult::Modified;
                }
                EventResult::Continue
            }),
        );

        // Host functions — each calls UPDATE with the pre-computed signed delta
        let pool_r = pool.clone();
        ctx.register_host_fn(
            "memory.opinion_reinforce",
            Arc::new(move |args: Vec<Value>, _| {
                update_opinion(&pool_r, &args, reinforce, "reinforce")
            }),
        );

        let pool_w = pool.clone();
        ctx.register_host_fn(
            "memory.opinion_weaken",
            Arc::new(move |args: Vec<Value>, _| update_opinion(&pool_w, &args, weaken, "weaken")),
        );

        let pool_c = pool;
        ctx.register_host_fn(
            "memory.opinion_contradict",
            Arc::new(move |args: Vec<Value>, _| {
                update_opinion(&pool_c, &args, contradict, "contradict")
            }),
        );

        Ok(())
    }
}

/// Execute an opinion confidence update against Postgres.
/// Args: [org_id (UUID string), key (string)]
fn update_opinion(
    pool: &Option<PgPool>,
    args: &[Value],
    delta: f32,
    action: &str,
) -> Result<Value, PluginError> {
    let Some(pool) = pool else {
        return Ok(serde_json::json!({
            "error": "PgPool not available",
            "action": action,
            "delta": delta,
        }));
    };

    let org_id: uuid::Uuid = args
        .first()
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction("org_id required as first arg".into()))?
        .parse()
        .map_err(|e| PluginError::HostFunction(format!("invalid org_id: {e}")))?;

    let key = args
        .get(1)
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction("key required as second arg".into()))?;

    let result = tokio::task::block_in_place(|| {
        tokio::runtime::Handle::current().block_on(async {
            sqlx::query_as::<_, (f32,)>(
                r#"UPDATE memories
                   SET confidence = GREATEST(0.0, LEAST(1.0, COALESCE(confidence, 0.5) + $3)),
                       updated_at = now()
                   WHERE org_id = $1 AND key = $2 AND kind = 'opinion'
                   RETURNING confidence"#,
            )
            .bind(org_id)
            .bind(key)
            .bind(delta)
            .fetch_optional(pool)
            .await
        })
    });

    match result {
        Ok(Some((confidence,))) => {
            debug!(key, action, confidence, "opinion confidence updated");
            Ok(serde_json::json!({ "key": key, "action": action, "confidence": confidence }))
        }
        Ok(None) => Ok(
            serde_json::json!({ "key": key, "action": action, "error": "not found or not an opinion" }),
        ),
        Err(e) => {
            warn!(key, action, error = %e, "opinion update failed");
            Err(PluginError::HostFunction(format!("DB error: {e}")))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn update_opinion_no_pool() {
        let result = update_opinion(
            &None,
            &[Value::String("org".into()), Value::String("key".into())],
            0.15,
            "reinforce",
        )
        .unwrap();
        assert!(result.get("error").is_some());
    }
}
