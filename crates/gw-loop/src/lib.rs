pub mod bridge;
pub mod context;
pub mod conversation;
pub mod error;
pub mod llm;
pub mod pg_store;
pub mod session;
pub mod tree;

pub use context::{build_turn_context, build_turn_context_with_opts, ContextOptions, LlmMessage, TurnContext};
pub use conversation::{ConversationLoop, IterationCallback, LoopConfig, SnapshotPolicy, TurnResult};
pub use error::LoopError;
pub use llm::{LlmClient, LlmResponse, OllamaLlmClient};
pub use pg_store::PgSessionStore;
pub use session::{SessionManager, SessionStatus};
pub use tree::SessionTree;
