//! gw-kb — Knowledge base ingestor and topic graph for greatwheel.
//!
//! See `docs/design-kb.md` for the full design.

pub mod chunk;
pub mod classify;
pub mod clean;
pub mod digest;
pub mod embed;
pub mod entities;
pub mod error;
pub mod extract;
pub mod feeds;
pub mod index;
pub mod ingest;
pub mod linking;
pub mod llm_parse;
pub mod merge;
pub mod organize;
pub mod plugin;
pub mod search;
pub mod server;
pub mod source;
pub mod synthesize;
pub mod topics;

pub use error::KbError;
pub use extract::Extracted;
pub use source::Source;
