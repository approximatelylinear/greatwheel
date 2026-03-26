//! Greatwheel Engine — the framework's composable runtime.
//!
//! This crate provides the plugin registry, event dispatcher, host function
//! router, and the `GreatWheelEngine` builder that wires everything together.
//!
//! # Usage
//!
//! ```rust,ignore
//! use gw_engine::GreatWheelEngine;
//!
//! let engine = GreatWheelEngine::new()
//!     .add_plugin(MyPlugin::new());
//!
//! let initialized = engine.init(&plugin_configs)?;
//! initialized.before_startup();
//! // ... start server ...
//! initialized.after_startup();
//! ```

pub mod dispatcher;
pub mod engine;
pub mod host_fn_router;
pub mod registry;

pub use dispatcher::EventDispatcher;
pub use engine::{GreatWheelEngine, InitializedEngine};
pub use host_fn_router::HostFnRouter;
pub use registry::PluginRegistry;
