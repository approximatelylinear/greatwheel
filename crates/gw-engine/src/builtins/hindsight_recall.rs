//! `hindsight-recall` plugin — temporal parsing and graph traversal host functions.
//!
//! Provides host functions for Python agents to query the memory graph and
//! perform temporal queries directly.  The heavy lifting (graph traversal,
//! temporal filtering, four-channel RRF fusion) is implemented in
//! `gw-memory::graph` and `gw-memory::temporal` and invoked by
//! `HybridStore::recall()` when `SearchMode::Full` is used.
//!
//! This plugin's role is to:
//! 1. Expose `memory.temporal_parse` host function
//! 2. Expose `memory.graph_neighbors` host function (placeholder until async dispatch)
//! 3. Provide configuration for the recall pipeline

use std::collections::HashMap;
use std::sync::Arc;

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
        // resolved date range as JSON. Returns null if no temporal expression found.
        ctx.register_host_fn(
            "memory.temporal_parse",
            Arc::new(|args: Vec<Value>, _kwargs: HashMap<String, Value>| {
                let query = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("");

                let now = chrono::Utc::now();

                // Use the gw-memory temporal parser (compiled into gw-engine
                // would require a dependency — so we inline a lightweight version
                // that covers the common cases for host function use).
                let range = parse_temporal_lightweight(query, now);

                match range {
                    Some((start, end)) => Ok(serde_json::json!({
                        "start": start.to_rfc3339(),
                        "end": end.to_rfc3339(),
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

/// Lightweight temporal parser for the host function.
///
/// Covers the most common patterns without depending on gw-memory.
/// The full parser in `gw_memory::temporal` handles more cases.
fn parse_temporal_lightweight(
    query: &str,
    now: chrono::DateTime<chrono::Utc>,
) -> Option<(chrono::DateTime<chrono::Utc>, chrono::DateTime<chrono::Utc>)> {
    let lower = query.to_lowercase();

    if lower.contains("yesterday") {
        let start = now - chrono::Duration::days(1);
        let start_day = start.date_naive().and_hms_opt(0, 0, 0)?.and_utc();
        let end_day = start_day + chrono::Duration::days(1);
        return Some((start_day, end_day));
    }

    if lower.contains("today") {
        let start_day = now.date_naive().and_hms_opt(0, 0, 0)?.and_utc();
        let end_day = start_day + chrono::Duration::days(1);
        return Some((start_day, end_day));
    }

    if lower.contains("last week") {
        return Some((now - chrono::Duration::weeks(2), now - chrono::Duration::weeks(1)));
    }

    // "last N days"
    for prefix in ["last ", "past "] {
        if let Some(pos) = lower.find(prefix) {
            let rest = &lower[pos + prefix.len()..];
            if let Some(space) = rest.find(' ') {
                if rest[space..].trim_start().starts_with("day") {
                    if let Ok(n) = rest[..space].parse::<i64>() {
                        return Some((now - chrono::Duration::days(n), now));
                    }
                }
            }
        }
    }

    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    #[test]
    fn lightweight_parse_yesterday() {
        let now = chrono::Utc.with_ymd_and_hms(2026, 3, 26, 12, 0, 0).unwrap();
        let (start, end) = parse_temporal_lightweight("What happened yesterday?", now).unwrap();
        assert_eq!(start.date_naive().to_string(), "2026-03-25");
        assert_eq!(end.date_naive().to_string(), "2026-03-26");
    }

    #[test]
    fn lightweight_parse_none() {
        let now = chrono::Utc::now();
        assert!(parse_temporal_lightweight("Who is Marie Curie?", now).is_none());
    }
}
