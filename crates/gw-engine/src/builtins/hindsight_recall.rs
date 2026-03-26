//! `hindsight-recall` plugin — temporal parsing and graph traversal host functions.
//!
//! Provides host functions for Python agents to query the memory graph and
//! perform temporal queries directly.  The heavy lifting (graph traversal,
//! temporal filtering, four-channel RRF fusion) is implemented in
//! `gw-memory::graph` and wired into `HybridStore::recall()` via
//! `SearchMode::Full`.
//!
//! This plugin's role is to:
//! 1. Expose `memory.temporal_parse` host function (uses `gw_core::temporal`)
//! 2. Expose `memory.graph_neighbors` host function (placeholder until async dispatch)
//! 3. Provide configuration for the recall pipeline

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
                "memory:graph_recall".into(),
                "memory:temporal_recall".into(),
                "host_fn:memory.temporal_parse".into(),
            ],
            requires: vec![],
            priority: 60, // After retain plugin
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
                let query = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

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

        // --- Host function: memory.graph_neighbors ---
        // Placeholder — returns empty until async dispatch is resolved.
        // Graph traversal requires DB access which isn't available in sync
        // host function handlers. When async dispatch lands (§6.2 Q7), this
        // will query memory_edges via spreading activation.
        ctx.register_host_fn(
            "memory.graph_neighbors",
            Arc::new(|_args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                // TODO: implement once async host functions are available
                Ok(Value::Array(vec![]))
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
