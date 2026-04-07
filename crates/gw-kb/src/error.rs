use thiserror::Error;

#[derive(Debug, Error)]
pub enum KbError {
    #[error("python error: {0}")]
    Python(String),

    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    #[error("sql error: {0}")]
    Sql(#[from] sqlx::Error),

    #[error("invalid url: {0}")]
    InvalidUrl(String),

    #[error("extraction failed: {0}")]
    Extraction(String),

    #[error("unsupported source format: {0}")]
    UnsupportedFormat(String),

    #[error("{0}")]
    Other(String),
}

impl From<pyo3::PyErr> for KbError {
    fn from(err: pyo3::PyErr) -> Self {
        KbError::Python(err.to_string())
    }
}

impl From<pyo3::DowncastError<'_, '_>> for KbError {
    fn from(err: pyo3::DowncastError<'_, '_>) -> Self {
        KbError::Python(format!("downcast: {}", err))
    }
}

impl From<pyo3::DowncastIntoError<'_>> for KbError {
    fn from(err: pyo3::DowncastIntoError<'_>) -> Self {
        KbError::Python(format!("downcast: {}", err))
    }
}
