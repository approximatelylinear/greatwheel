//! MCP-UI resource detection and widget wrapping.
//!
//! This module is a pure helper: it inspects a JSON value (typically an
//! MCP tool-call result) and, if the value matches the MCP-UI resource
//! shape, produces a `Widget` ready for `UiSurfaceStore::emit`.
//!
//! The expected shape (per MCP Apps / MCP-UI):
//!
//! ```json
//! {
//!   "resource": {
//!     "uri": "ui://...",
//!     "mimeType": "text/html" | "application/vnd.mcp-ui+json",
//!     "_meta": { "ui": { "csp": "..." } }
//!   }
//! }
//! ```
//!
//! The pipeline that invokes this (an MCP client integration in
//! `gw-engine` / `gw-runtime`) does not yet exist in greatwheel, so this
//! module is a ready-to-consume detector — when the MCP client lands,
//! it calls `detect()` on each tool-call result and, on `Some`, pushes
//! the widget through the normal emit path.

use chrono::Utc;
use gw_core::{SessionId, UiSurfaceId, Widget, WidgetId, WidgetKind, WidgetPayload, WidgetState};
use serde_json::Value;

/// Try to interpret `result` as an MCP-UI resource and wrap it as a
/// `Widget`. Returns `None` if the result is not an MCP-UI resource
/// — the caller should forward it normally in that case.
///
/// `session_id` and `surface_id` are supplied by the caller (the MCP
/// client integration knows which session the tool call came from).
pub fn detect(result: &Value, session_id: SessionId, surface_id: UiSurfaceId) -> Option<Widget> {
    let resource = result.get("resource")?;
    let mime = resource.get("mimeType").and_then(|v| v.as_str())?;
    if !is_mcp_ui_mime(mime) {
        return None;
    }

    let uri = resource.get("uri").and_then(|v| v.as_str())?.to_string();
    let csp = resource
        .pointer("/_meta/ui/csp")
        .and_then(|v| v.as_str())
        .map(String::from);

    Some(Widget {
        id: WidgetId::new(),
        surface_id,
        session_id,
        origin_entry: None,
        kind: WidgetKind::McpUi,
        state: WidgetState::Active,
        payload: WidgetPayload::Reference { uri, csp },
        supersedes: None,
        created_at: Utc::now(),
        resolved_at: None,
        resolution: None,
        multi_use: false,
            follow_up: false,
    })
}

fn is_mcp_ui_mime(mime: &str) -> bool {
    matches!(mime, "text/html" | "application/vnd.mcp-ui+json")
}

#[cfg(test)]
mod tests {
    use super::*;
    use uuid::Uuid;

    fn ids() -> (SessionId, UiSurfaceId) {
        (SessionId(Uuid::new_v4()), UiSurfaceId::new())
    }

    #[test]
    fn detects_text_html_resource_with_csp() {
        let (sid, surface) = ids();
        let v = serde_json::json!({
            "resource": {
                "uri": "ui://acme/invoice",
                "mimeType": "text/html",
                "_meta": { "ui": { "csp": "script-src 'self'" } }
            }
        });
        let w = detect(&v, sid, surface).unwrap();
        assert!(matches!(w.kind, WidgetKind::McpUi));
        match w.payload {
            WidgetPayload::Reference { uri, csp } => {
                assert_eq!(uri, "ui://acme/invoice");
                assert_eq!(csp.as_deref(), Some("script-src 'self'"));
            }
            other => panic!("expected Reference, got {:?}", other),
        }
    }

    #[test]
    fn detects_vnd_mcp_ui_json_resource() {
        let (sid, surface) = ids();
        let v = serde_json::json!({
            "resource": {
                "uri": "ui://x",
                "mimeType": "application/vnd.mcp-ui+json"
            }
        });
        assert!(detect(&v, sid, surface).is_some());
    }

    #[test]
    fn no_csp_when_metadata_absent() {
        let (sid, surface) = ids();
        let v = serde_json::json!({
            "resource": { "uri": "ui://x", "mimeType": "text/html" }
        });
        let w = detect(&v, sid, surface).unwrap();
        match w.payload {
            WidgetPayload::Reference { csp, .. } => assert!(csp.is_none()),
            other => panic!("expected Reference, got {:?}", other),
        }
    }

    #[test]
    fn ignores_non_ui_mime() {
        let (sid, surface) = ids();
        let v = serde_json::json!({
            "resource": { "uri": "ui://x", "mimeType": "text/plain" }
        });
        assert!(detect(&v, sid, surface).is_none());
    }

    #[test]
    fn ignores_missing_resource_key() {
        let (sid, surface) = ids();
        let v = serde_json::json!({ "text": "hello" });
        assert!(detect(&v, sid, surface).is_none());
    }

    #[test]
    fn ignores_missing_uri() {
        let (sid, surface) = ids();
        let v = serde_json::json!({
            "resource": { "mimeType": "text/html" }
        });
        assert!(detect(&v, sid, surface).is_none());
    }

    #[test]
    fn ignores_missing_mime() {
        let (sid, surface) = ids();
        let v = serde_json::json!({
            "resource": { "uri": "ui://x" }
        });
        assert!(detect(&v, sid, surface).is_none());
    }
}
