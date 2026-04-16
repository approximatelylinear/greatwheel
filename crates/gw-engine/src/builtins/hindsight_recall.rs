//! `hindsight-recall` plugin — exposes temporal parsing to Python agents.
//!
//! The heavy lifting (graph traversal, temporal filtering, four-channel RRF)
//! is in `gw-memory::graph` / `HybridStore::recall()` via `SearchMode::Full`.
//! This plugin provides the `memory.temporal_parse` host function.

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::temporal as tp;
use gw_core::{Plugin, PluginContext, PluginError, PluginManifest};
use serde_json::Value;

/// Built-in plugin that provides temporal and graph recall capabilities.
pub struct HindsightRecallPlugin;

impl Plugin for HindsightRecallPlugin {
    fn name(&self) -> &str {
        "hindsight-recall"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "memory:temporal_recall".into(),
                "host_fn:memory.temporal_parse".into(),
            ],
            requires: vec![],
            priority: 60,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        // --- Host function: memory.temporal_parse ---
        // Parses a temporal expression from a query string and returns the
        // resolved date range as JSON.  Uses the canonical parser from gw-core.
        // Returns null if no temporal expression found.
        ctx.register_host_fn(
            "memory.temporal_parse",
            Arc::new(|args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let query = args.first().and_then(|v| v.as_str()).unwrap_or("");

                let now = chrono::Utc::now();
                let range = tp::parse_temporal(query, now);

                match range {
                    Some(r) => Ok(serde_json::json!({
                        "start": r.start.to_rfc3339(),
                        "end": r.end.to_rfc3339(),
                    })),
                    None => Ok(Value::Null),
                }
            }),
        );

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use gw_core::temporal::parse_temporal;

    #[test]
    fn host_fn_uses_core_parser() {
        let now = chrono::Utc::now();
        // Verify the core parser handles the full feature set
        assert!(parse_temporal("What happened yesterday?", now).is_some());
        assert!(parse_temporal("Events last month", now).is_some());
        assert!(parse_temporal("In January 2026?", now).is_some());
        assert!(parse_temporal("Who is Marie Curie?", now).is_none());
    }
}
