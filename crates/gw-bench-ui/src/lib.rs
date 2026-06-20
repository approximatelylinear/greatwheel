//! `gw-bench-ui` — read-only data layer + plugin for the BrowseComp
//! experiment dashboard.
//!
//! Walks `runs/` directories, loads per-query JSONs, joins against the
//! BrowseComp-Plus gold answers, and exposes the result through a
//! `Plugin` so a `BenchAgent` running in `gw-loop` can list, drill
//! into, and compare experiments.
//!
//! Design: `docs/design-experiment-dashboard.md`.

pub mod paths;
pub mod plugin;
pub mod score;
pub mod store;
pub mod types;

pub use paths::BenchPaths;
pub use plugin::BenchPlugin;
pub use store::{BenchError, BenchStore};
