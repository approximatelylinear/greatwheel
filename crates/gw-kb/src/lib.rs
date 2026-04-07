//! gw-kb — Knowledge base ingestor and topic graph for greatwheel.
//!
//! See `docs/design-kb.md` for the full design.

pub mod chunk;
pub mod embed;
pub mod error;
pub mod extract;
pub mod index;
pub mod ingest;
pub mod linking;
pub mod organize;
pub mod search;
pub mod source;
pub mod topics;

pub use error::KbError;
pub use extract::Extracted;
pub use source::Source;
