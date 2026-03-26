//! Re-exports temporal parsing and scoring from `gw-core::temporal`.
//!
//! The pure functions live in `gw-core` so they can be shared by both
//! `gw-memory` (for `SearchMode::Full`) and `gw-engine` (for host functions)
//! without duplication.

pub use gw_core::temporal::*;
