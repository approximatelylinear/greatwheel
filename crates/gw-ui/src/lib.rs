//! Generative UX for Greatwheel.
//!
//! See `docs/design-gw-ui.md` for the design. This crate owns the UI
//! surface store, the `UiPlugin` that exposes host functions to agents,
//! and (in later steps) the AG-UI channel adapter and MCP-UI relay.

pub mod ag_ui;
pub mod mcp_ui;
pub mod plugin;
pub mod surface;

pub use ag_ui::{AgUiAdapter, AgUiEvent};
pub use plugin::UiPlugin;
pub use surface::{UiError, UiNotification, UiSurface, UiSurfaceSnapshot, UiSurfaceStore};
