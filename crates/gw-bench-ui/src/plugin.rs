//! `gw-bench-ui` plugin — exposes BrowseComp dashboard host functions
//! to agents. Read-only in v1; no `bench:write` capability yet.
//!
//! Names are flat (no dots) per the same convention as `gw-ui`, so a
//! Python agent in ouros calls `list_experiments()` directly. The
//! dotted names appear in the manifest's `provides` for capability
//! scoping.

use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{Plugin, PluginContext, PluginError, PluginManifest};
use serde_json::Value;

use crate::store::{BenchError, BenchStore};

pub struct BenchPlugin {
    store: Arc<BenchStore>,
}

impl BenchPlugin {
    pub fn new(store: Arc<BenchStore>) -> Self {
        Self { store }
    }
}

impl Plugin for BenchPlugin {
    fn name(&self) -> &str {
        "gw-bench-ui"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "bench".into(),
                "host_fn:bench.list_experiments".into(),
                "host_fn:bench.load_run".into(),
                "host_fn:bench.load_query".into(),
                "host_fn:bench.load_config".into(),
                "host_fn:bench.compare".into(),
                "host_fn:bench.gold".into(),
                "host_fn:bench.read_annotations".into(),
                "host_fn:bench.write_annotation".into(),
                "host_fn:bench.difficulty_matrix".into(),
                "host_fn:bench.query_history".into(),
                "host_fn:bench.cost_summary".into(),
            ],
            requires: vec![],
            priority: 60,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        ctx.provide(self.store.clone());

        let s = self.store.clone();
        ctx.register_host_fn_async(
            "list_experiments",
            Some("bench:read"),
            move |_args, _kwargs| {
                let s = s.clone();
                async move {
                    let rows = s.list_experiments().map_err(map_err)?;
                    Ok(serialize(&rows))
                }
            },
        );

        let s = self.store.clone();
        ctx.register_host_fn_async("load_run", Some("bench:read"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let slug = pick_string(&args, &kwargs, "slug")?;
                let detail = s.load_run(&slug).map_err(map_err)?;
                Ok(serialize(&detail))
            }
        });

        let s = self.store.clone();
        ctx.register_host_fn_async("load_query", Some("bench:read"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let slug = pick_string(&args, &kwargs, "slug")?;
                let qid = pick_string(&args, &kwargs, "query_id")?;
                let detail = s.load_query(&slug, &qid).map_err(map_err)?;
                Ok(serialize(&detail))
            }
        });

        let s = self.store.clone();
        ctx.register_host_fn_async("load_config", Some("bench:read"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let slug = pick_string(&args, &kwargs, "slug")?;
                let info = s.load_config(&slug).map_err(map_err)?;
                Ok(serialize(&info))
            }
        });

        let s = self.store.clone();
        ctx.register_host_fn_async("compare", Some("bench:read"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let slug_a = pick_string(&args, &kwargs, "slug_a")?;
                let slug_b = pick_string(&args, &kwargs, "slug_b")?;
                let cmp = s.compare(&slug_a, &slug_b).map_err(map_err)?;
                Ok(serialize(&cmp))
            }
        });

        let s = self.store.clone();
        ctx.register_host_fn_async("gold", Some("bench:read"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let qid = pick_string(&args, &kwargs, "query_id")?;
                let entry = s.gold(&qid).map_err(map_err)?;
                Ok(serialize(&entry))
            }
        });

        let s = self.store.clone();
        ctx.register_host_fn_async(
            "read_annotations",
            Some("bench:read"),
            move |args, kwargs| {
                let s = s.clone();
                async move {
                    let slug = pick_string(&args, &kwargs, "slug")?;
                    let ann = s.read_annotations(&slug).map_err(map_err)?;
                    Ok(serialize(&ann))
                }
            },
        );

        let s = self.store.clone();
        ctx.register_host_fn_async(
            "write_annotation",
            Some("bench:write"),
            move |args, kwargs| {
                let s = s.clone();
                async move {
                    let slug = pick_string(&args, &kwargs, "slug")?;
                    let text = kwargs
                        .get("text")
                        .and_then(|v| v.as_str())
                        .unwrap_or("")
                        .to_string();
                    let tags = pick_string_list(&kwargs, "tags")?;
                    if text.is_empty() && tags.is_empty() {
                        return Err(PluginError::HostFunction(
                            "write_annotation requires `text` or `tags`".into(),
                        ));
                    }
                    let ann = s.write_annotation(&slug, &text, tags).map_err(map_err)?;
                    Ok(serialize(&ann))
                }
            },
        );

        let s = self.store.clone();
        ctx.register_host_fn_async(
            "difficulty_matrix",
            Some("bench:read"),
            move |_args, kwargs| {
                let s = s.clone();
                async move {
                    let slugs = pick_string_list(&kwargs, "slugs")?;
                    let m = s.difficulty_matrix(&slugs).map_err(map_err)?;
                    Ok(serialize(&m))
                }
            },
        );

        let s = self.store.clone();
        ctx.register_host_fn_async("query_history", Some("bench:read"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let qid = pick_string(&args, &kwargs, "query_id")?;
                let slugs = pick_string_list(&kwargs, "slugs")?;
                let h = s.query_history(&qid, &slugs).map_err(map_err)?;
                Ok(serialize(&h))
            }
        });

        let s = self.store.clone();
        ctx.register_host_fn_async("cost_summary", Some("bench:read"), move |_args, kwargs| {
            let s = s.clone();
            async move {
                let window_days = kwargs
                    .get("window_days")
                    .and_then(|v| v.as_u64())
                    .map(|n| n as u32);
                let rows = s.cost_summary(window_days).map_err(map_err)?;
                Ok(serialize(&rows))
            }
        });

        Ok(())
    }
}

fn pick_string_list(
    kwargs: &HashMap<String, Value>,
    key: &str,
) -> Result<Vec<String>, PluginError> {
    let Some(raw) = kwargs.get(key) else {
        return Ok(Vec::new());
    };
    if raw.is_null() {
        return Ok(Vec::new());
    }
    let arr = raw
        .as_array()
        .ok_or_else(|| PluginError::HostFunction(format!("{key} must be an array of strings")))?;
    let mut out = Vec::with_capacity(arr.len());
    for v in arr {
        let s = v.as_str().ok_or_else(|| {
            PluginError::HostFunction(format!("{key} must be an array of strings"))
        })?;
        out.push(s.to_string());
    }
    Ok(out)
}

/// Resolve a string argument from kwargs, falling back to the first
/// positional arg. Mirrors how `gw-ui` host fns are called from
/// Python — agents typically pass everything by keyword, but `args[0]`
/// is a useful shorthand for single-arg calls like `load_run("foo")`.
fn pick_string(
    args: &[Value],
    kwargs: &HashMap<String, Value>,
    key: &str,
) -> Result<String, PluginError> {
    if let Some(v) = kwargs.get(key) {
        return v
            .as_str()
            .map(String::from)
            .ok_or_else(|| PluginError::HostFunction(format!("{key} must be a string")));
    }
    if let Some(v) = args.first() {
        if let Some(s) = v.as_str() {
            return Ok(s.to_string());
        }
    }
    Err(PluginError::HostFunction(format!("{key} required")))
}

fn map_err(err: BenchError) -> PluginError {
    match err {
        BenchError::NotFound(_) => PluginError::HostFunction(err.to_string()),
        BenchError::InvalidPath(_) => PluginError::HostFunction(err.to_string()),
        other => PluginError::HostFunction(other.to_string()),
    }
}

/// Domain types use derive(Serialize) with infallible field types, so
/// `to_value` can't actually fail. We unwrap rather than complicate
/// every host fn with a JSON-serialization error path.
fn serialize<T: serde::Serialize>(value: &T) -> Value {
    serde_json::to_value(value).expect("BenchStore types are infallibly serializable")
}
