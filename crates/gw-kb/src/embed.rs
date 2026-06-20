//! KB embedder with two backends:
//!
//! - `Local`: embedded Python sentence-transformers via pyo3 (in-process).
//!   Used for offline ingestion and dev. Required because Ollama's wrapper
//!   for nomic-embed-text produces collapsed vectors on short inputs.
//!
//! - `Remote`: delegates to a `gw_llm::OllamaClient` configured for an
//!   OpenAI-compatible `/v1/embeddings` endpoint (e.g. a Modal sglang or
//!   custom embed server). Used in prod images that have no Python or
//!   sentence-transformers installed.
//!
//! Public API (`embed_texts`, `embed_one`, `dim`) is identical for both
//! variants, so callers don't have to know which one they got.

use std::sync::Arc;

use gw_llm::OllamaClient;
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyListMethods};

use crate::error::KbError;
use crate::extract::init_python_path;

/// Default model identifier — passed to sentence-transformers (Local only).
pub const DEFAULT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";

#[derive(Clone)]
pub enum Embedder {
    Local(LocalEmbedder),
    Remote(RemoteEmbedder),
}

/// Embedded sentence-transformers (pyo3). The model is held resident inside
/// the embedded Python interpreter; first call pays the load cost.
#[derive(Debug, Clone)]
pub struct LocalEmbedder {
    pub model: String,
    pub batch_size: usize,
}

/// HTTP delegation to a shared `gw_llm::OllamaClient`. The client knows the
/// endpoint URL, backend shape (`/v1/embeddings` for sglang/openai), and
/// bearer token.
#[derive(Clone)]
pub struct RemoteEmbedder {
    pub client: Arc<OllamaClient>,
}

impl Embedder {
    /// Construct a Local (sentence-transformers) embedder. Cheap; does not
    /// load the model.
    pub fn new(model: impl Into<String>) -> Self {
        Self::Local(LocalEmbedder {
            model: model.into(),
            batch_size: 32,
        })
    }

    /// Construct a Remote embedder that delegates to the given LLM client.
    /// Must be called from within a multi-threaded Tokio runtime context;
    /// the embed call bridges async-to-sync via `block_in_place`.
    pub fn remote(client: Arc<OllamaClient>) -> Self {
        Self::Remote(RemoteEmbedder { client })
    }

    /// Encode a batch of texts. Returns L2-normalized vectors.
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KbError> {
        match self {
            Self::Local(l) => l.embed_texts(texts),
            Self::Remote(r) => r.embed_texts(texts),
        }
    }

    /// Convenience: embed a single string.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>, KbError> {
        let mut out = self.embed_texts(&[text.to_string()])?;
        out.pop()
            .ok_or_else(|| KbError::Other("embed returned no vectors".into()))
    }

    /// Query the model's embedding dimension.
    pub fn dim(&self) -> Result<usize, KbError> {
        match self {
            Self::Local(l) => l.dim(),
            Self::Remote(r) => r.dim(),
        }
    }
}

impl Default for Embedder {
    fn default() -> Self {
        Self::new(DEFAULT_MODEL)
    }
}

impl LocalEmbedder {
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KbError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        init_python_path()?;
        Python::with_gil(|py| {
            let module = py.import_bound("gw_kb_extract.embed")?;
            let py_texts = PyList::empty_bound(py);
            for t in texts {
                py_texts.append(t.as_str())?;
            }
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("model_name", &self.model)?;
            kwargs.set_item("batch_size", self.batch_size)?;
            let result = module.call_method("embed_texts", (py_texts,), Some(&kwargs))?;
            let outer: Bound<PyList> = result.downcast_into()?;
            let mut out: Vec<Vec<f32>> = Vec::with_capacity(outer.len());
            for row in outer.iter() {
                let inner: Bound<PyList> = row.downcast_into()?;
                let mut v: Vec<f32> = Vec::with_capacity(inner.len());
                for x in inner.iter() {
                    let f: f64 = x.extract()?;
                    v.push(f as f32);
                }
                out.push(v);
            }
            Ok(out)
        })
    }

    pub fn dim(&self) -> Result<usize, KbError> {
        init_python_path()?;
        Python::with_gil(|py| {
            let module = py.import_bound("gw_kb_extract.embed")?;
            let result = module.call_method1("embedding_dim", (&self.model,))?;
            let d: usize = result.extract()?;
            Ok(d)
        })
    }
}

impl RemoteEmbedder {
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KbError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let client = Arc::clone(&self.client);
        let texts = texts.to_vec();
        // Bridge async client.embed() to a sync API. Requires a multi-threaded
        // Tokio runtime; both gw-server and the gw_kb binary provide one.
        tokio::task::block_in_place(|| {
            tokio::runtime::Handle::current()
                .block_on(async move { client.embed(&texts).await })
                .map_err(|e| KbError::Other(format!("remote embed: {e}")))
        })
    }

    /// Probe the remote endpoint with a tiny input to learn the vector
    /// dimension. Called once at startup; not on the hot path.
    pub fn dim(&self) -> Result<usize, KbError> {
        let vec = self
            .embed_texts(&["a".to_string()])?
            .into_iter()
            .next()
            .ok_or_else(|| KbError::Other("remote embed returned no vectors".into()))?;
        Ok(vec.len())
    }
}
