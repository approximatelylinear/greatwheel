//! `gw-ui` plugin — exposes UI host functions to agents.
//!
//! Host functions registered (all capability-gated under `ui:write`).
//! Names are flat (no dots) so Python agents in ouros can call them
//! directly by bare identifier:
//!
//!   - `emit_widget`        — emit an A2UI or MCP-UI widget
//!   - `supersede_widget`   — replace an active widget with a new one
//!   - `resolve_widget`     — agent-driven close with a terminal value
//!   - `pin_to_canvas`      — move a widget into the canvas slot
//!   - `emit_mcp_resource`  — convenience for MCP-UI resources
//!
//! The plugin owns an `Arc<UiSurfaceStore>` and publishes it via
//! `ctx.provide` so the AG-UI adapter (step 3) can pick it up from
//! `SharedState`.

use std::collections::HashMap;
use std::sync::Arc;

use chrono::Utc;
use gw_core::{
    EntryId, Plugin, PluginContext, PluginError, PluginManifest, SessionId, UiSurfaceId, Widget,
    WidgetId, WidgetKind, WidgetPayload, WidgetState,
};
use serde_json::{json, Value};
use uuid::Uuid;

use crate::surface::UiSurfaceStore;

pub struct UiPlugin;

impl Plugin for UiPlugin {
    fn name(&self) -> &str {
        "gw-ui"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "ui".into(),
                "host_fn:ui.emit_widget".into(),
                "host_fn:ui.supersede_widget".into(),
                "host_fn:ui.resolve_widget".into(),
                "host_fn:ui.pin_to_canvas".into(),
                "host_fn:ui.emit_mcp_resource".into(),
            ],
            requires: vec![],
            priority: 50,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let store = Arc::new(UiSurfaceStore::new());
        ctx.provide(store.clone());

        let s = store.clone();
        ctx.register_host_fn_async("emit_widget", Some("ui:write"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let widget = build_widget(&args, &kwargs)?;
                let id = widget.id;
                s.emit(widget)
                    .await
                    .map_err(|e| PluginError::HostFunction(e.to_string()))?;
                Ok(json!({ "widget_id": id.0.to_string() }))
            }
        });

        let s = store.clone();
        ctx.register_host_fn_async("supersede_widget", Some("ui:write"), move |args, kwargs| {
            let s = s.clone();
            async move {
                let old_widget_id = WidgetId(parse_uuid(&kwargs, "old_widget_id")?);
                let mut new_widget = build_widget(&args, &kwargs)?;
                new_widget.supersedes = Some(old_widget_id);
                let new_id = new_widget.id;
                s.supersede(old_widget_id, new_widget)
                    .await
                    .map_err(|e| PluginError::HostFunction(e.to_string()))?;
                Ok(json!({ "widget_id": new_id.0.to_string() }))
            }
        });

        let s = store.clone();
        ctx.register_host_fn_async("resolve_widget", Some("ui:write"), move |_args, kwargs| {
            let s = s.clone();
            async move {
                let widget_id = WidgetId(parse_uuid(&kwargs, "widget_id")?);
                let data = kwargs.get("data").cloned().unwrap_or(Value::Null);
                s.resolve(widget_id, data)
                    .await
                    .map_err(|e| PluginError::HostFunction(e.to_string()))?;
                Ok(Value::Null)
            }
        });

        let s = store.clone();
        ctx.register_host_fn_async("pin_to_canvas", Some("ui:write"), move |_args, kwargs| {
            let s = s.clone();
            async move {
                let widget_id = WidgetId(parse_uuid(&kwargs, "widget_id")?);
                s.pin_to_canvas(widget_id)
                    .await
                    .map_err(|e| PluginError::HostFunction(e.to_string()))?;
                Ok(Value::Null)
            }
        });

        let s = store;
        ctx.register_host_fn_async(
            "emit_mcp_resource",
            Some("ui:write"),
            move |_args, kwargs| {
                let s = s.clone();
                async move {
                    let widget = build_mcp_widget(&kwargs)?;
                    let id = widget.id;
                    s.emit(widget)
                        .await
                        .map_err(|e| PluginError::HostFunction(e.to_string()))?;
                    Ok(json!({ "widget_id": id.0.to_string() }))
                }
            },
        );

        Ok(())
    }
}

/// Build an MCP-UI widget from kwargs. Required: `session_id`, `uri`.
/// Optional: `surface_id`, `csp`, `origin_entry`.
fn build_mcp_widget(kwargs: &HashMap<String, Value>) -> Result<Widget, PluginError> {
    let session_id = SessionId(parse_uuid(kwargs, "session_id")?);
    let surface_id = optional_uuid(kwargs, "surface_id")?
        .map(UiSurfaceId)
        .unwrap_or_else(UiSurfaceId::new);
    let origin_entry = optional_uuid(kwargs, "origin_entry")?.map(EntryId);
    let uri = kwargs
        .get("uri")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction("uri required".into()))?
        .to_string();
    let csp = kwargs.get("csp").and_then(|v| v.as_str()).map(String::from);

    Ok(Widget {
        id: WidgetId::new(),
        surface_id,
        session_id,
        origin_entry,
        kind: WidgetKind::McpUi,
        state: WidgetState::Active,
        payload: WidgetPayload::Reference { uri, csp },
        supersedes: None,
        created_at: Utc::now(),
        resolved_at: None,
        resolution: None,
    })
}

/// Construct a `Widget` from host-function args. Kwargs keys:
///   - `session_id` (required, UUID string)
///   - `kind` (required: "a2ui" | "mcp-ui" | "custom:<name>")
///   - `payload` (required unless `payload_uri` is given): arbitrary JSON
///   - `payload_uri` (alternative): string; pair with optional `payload_csp`
///   - `surface_id` (optional UUID string; generated if absent)
///   - `origin_entry` (optional UUID string)
///   - `supersedes` (optional UUID string)
fn build_widget(_args: &[Value], kwargs: &HashMap<String, Value>) -> Result<Widget, PluginError> {
    let session_id = SessionId(parse_uuid(kwargs, "session_id")?);
    let surface_id = optional_uuid(kwargs, "surface_id")?
        .map(UiSurfaceId)
        .unwrap_or_else(UiSurfaceId::new);
    let kind = parse_kind(kwargs)?;
    let payload = parse_payload(kwargs)?;
    let origin_entry = optional_uuid(kwargs, "origin_entry")?.map(EntryId);
    let supersedes = optional_uuid(kwargs, "supersedes")?.map(WidgetId);

    Ok(Widget {
        id: WidgetId::new(),
        surface_id,
        session_id,
        origin_entry,
        kind,
        state: WidgetState::Active,
        payload,
        supersedes,
        created_at: Utc::now(),
        resolved_at: None,
        resolution: None,
    })
}

fn parse_uuid(kwargs: &HashMap<String, Value>, key: &str) -> Result<Uuid, PluginError> {
    kwargs
        .get(key)
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction(format!("{key} required (string UUID)")))?
        .parse()
        .map_err(|e| PluginError::HostFunction(format!("invalid {key}: {e}")))
}

fn optional_uuid(kwargs: &HashMap<String, Value>, key: &str) -> Result<Option<Uuid>, PluginError> {
    match kwargs.get(key) {
        None | Some(Value::Null) => Ok(None),
        Some(v) => {
            let s = v
                .as_str()
                .ok_or_else(|| PluginError::HostFunction(format!("{key} must be a string")))?;
            s.parse()
                .map(Some)
                .map_err(|e| PluginError::HostFunction(format!("invalid {key}: {e}")))
        }
    }
}

fn parse_kind(kwargs: &HashMap<String, Value>) -> Result<WidgetKind, PluginError> {
    let kind = kwargs
        .get("kind")
        .and_then(|v| v.as_str())
        .ok_or_else(|| PluginError::HostFunction("kind required".into()))?;
    Ok(match kind {
        "a2ui" => WidgetKind::A2ui,
        "mcp-ui" | "mcp_ui" => WidgetKind::McpUi,
        other if other.starts_with("custom:") => WidgetKind::Custom(other[7..].to_string()),
        other => {
            return Err(PluginError::HostFunction(format!(
                "unknown widget kind: {other}"
            )));
        }
    })
}

fn parse_payload(kwargs: &HashMap<String, Value>) -> Result<WidgetPayload, PluginError> {
    if let Some(uri) = kwargs.get("payload_uri").and_then(|v| v.as_str()) {
        let csp = kwargs
            .get("payload_csp")
            .and_then(|v| v.as_str())
            .map(String::from);
        return Ok(WidgetPayload::Reference {
            uri: uri.to_string(),
            csp,
        });
    }
    let payload = kwargs
        .get("payload")
        .cloned()
        .ok_or_else(|| PluginError::HostFunction("payload or payload_uri required".into()))?;
    Ok(WidgetPayload::Inline(payload))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kw(pairs: &[(&str, Value)]) -> HashMap<String, Value> {
        pairs
            .iter()
            .map(|(k, v)| ((*k).to_string(), v.clone()))
            .collect()
    }

    #[test]
    fn build_widget_minimal_a2ui() {
        let session = Uuid::new_v4();
        let kwargs = kw(&[
            ("session_id", Value::String(session.to_string())),
            ("kind", Value::String("a2ui".into())),
            ("payload", json!({"type": "Button"})),
        ]);
        let w = build_widget(&[], &kwargs).unwrap();
        assert_eq!(w.session_id.0, session);
        assert!(matches!(w.kind, WidgetKind::A2ui));
        assert!(matches!(w.payload, WidgetPayload::Inline(_)));
        assert_eq!(w.state, WidgetState::Active);
    }

    #[test]
    fn build_widget_mcp_ui_reference() {
        let kwargs = kw(&[
            ("session_id", Value::String(Uuid::new_v4().to_string())),
            ("kind", Value::String("mcp-ui".into())),
            ("payload_uri", Value::String("ui://stripe/inv".into())),
            ("payload_csp", Value::String("script-src 'self'".into())),
        ]);
        let w = build_widget(&[], &kwargs).unwrap();
        assert!(matches!(w.kind, WidgetKind::McpUi));
        match w.payload {
            WidgetPayload::Reference { uri, csp } => {
                assert_eq!(uri, "ui://stripe/inv");
                assert_eq!(csp.as_deref(), Some("script-src 'self'"));
            }
            other => panic!("expected Reference, got {:?}", other),
        }
    }

    #[test]
    fn build_widget_custom_kind() {
        let kwargs = kw(&[
            ("session_id", Value::String(Uuid::new_v4().to_string())),
            ("kind", Value::String("custom:invoice-card".into())),
            ("payload", json!({})),
        ]);
        let w = build_widget(&[], &kwargs).unwrap();
        match w.kind {
            WidgetKind::Custom(name) => assert_eq!(name, "invoice-card"),
            other => panic!("expected Custom, got {:?}", other),
        }
    }

    #[test]
    fn build_widget_missing_session_id() {
        let kwargs = kw(&[
            ("kind", Value::String("a2ui".into())),
            ("payload", json!({})),
        ]);
        let err = build_widget(&[], &kwargs).unwrap_err();
        assert!(matches!(err, PluginError::HostFunction(_)));
    }

    #[test]
    fn build_widget_unknown_kind() {
        let kwargs = kw(&[
            ("session_id", Value::String(Uuid::new_v4().to_string())),
            ("kind", Value::String("bogus".into())),
            ("payload", json!({})),
        ]);
        let err = build_widget(&[], &kwargs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("bogus"));
    }

    #[test]
    fn build_widget_missing_payload() {
        let kwargs = kw(&[
            ("session_id", Value::String(Uuid::new_v4().to_string())),
            ("kind", Value::String("a2ui".into())),
        ]);
        let err = build_widget(&[], &kwargs).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("payload"));
    }

    #[test]
    fn build_mcp_widget_minimal() {
        let session = Uuid::new_v4();
        let kwargs = kw(&[
            ("session_id", Value::String(session.to_string())),
            ("uri", Value::String("ui://acme/widget".into())),
        ]);
        let w = build_mcp_widget(&kwargs).unwrap();
        assert!(matches!(w.kind, WidgetKind::McpUi));
        match w.payload {
            WidgetPayload::Reference { uri, csp } => {
                assert_eq!(uri, "ui://acme/widget");
                assert!(csp.is_none());
            }
            other => panic!("expected Reference, got {:?}", other),
        }
    }

    #[test]
    fn build_mcp_widget_with_csp() {
        let kwargs = kw(&[
            ("session_id", Value::String(Uuid::new_v4().to_string())),
            ("uri", Value::String("ui://x".into())),
            ("csp", Value::String("script-src 'self'".into())),
        ]);
        let w = build_mcp_widget(&kwargs).unwrap();
        match w.payload {
            WidgetPayload::Reference { csp, .. } => {
                assert_eq!(csp.as_deref(), Some("script-src 'self'"));
            }
            other => panic!("expected Reference, got {:?}", other),
        }
    }

    #[test]
    fn build_mcp_widget_missing_uri() {
        let kwargs = kw(&[("session_id", Value::String(Uuid::new_v4().to_string()))]);
        let err = build_mcp_widget(&kwargs).unwrap_err();
        assert!(err.to_string().contains("uri"));
    }

    #[test]
    fn parse_uuid_requires_string() {
        let kwargs = kw(&[("widget_id", Value::Number(42.into()))]);
        let err = parse_uuid(&kwargs, "widget_id").unwrap_err();
        assert!(err.to_string().contains("widget_id"));
    }

    #[test]
    fn optional_uuid_accepts_null() {
        let kwargs = kw(&[("surface_id", Value::Null)]);
        let result = optional_uuid(&kwargs, "surface_id").unwrap();
        assert!(result.is_none());
    }
}
