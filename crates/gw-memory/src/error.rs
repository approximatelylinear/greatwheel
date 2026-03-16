use thiserror::Error;

#[derive(Debug, Error)]
pub enum MemoryError {
    #[error("database error: {0}")]
    Database(#[from] sqlx::Error),

    #[error("lance error: {0}")]
    Lance(#[from] lancedb::Error),

    #[error("embedding error: {0}")]
    Embedding(String),

    #[error("serialization error: {0}")]
    Serialization(#[from] serde_json::Error),

    #[error("tantivy error: {0}")]
    Tantivy(String),

    #[error("not found: {0}")]
    NotFound(String),
}
