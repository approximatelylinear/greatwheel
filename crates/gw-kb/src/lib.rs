//! gw-kb — Knowledge base ingestor and topic graph for greatwheel.
//!
//! See `docs/design-kb.md` for the full design.

pub mod chunk;
pub mod classify;
pub mod embed;
pub mod error;
pub mod extract;
pub mod feeds;
pub mod index;
pub mod ingest;
pub mod linking;
pub mod merge;
pub mod organize;
pub mod search;
pub mod source;
pub mod synthesize;
pub mod topics;

pub use error::KbError;
pub use extract::Extracted;
pub use source::Source;
