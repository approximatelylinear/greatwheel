//! Embedding via the embedded Python `gw_kb_extract.embed` module.
//!
//! This bypasses gw-llm/Ollama for embeddings because Ollama's wrapper for
//! nomic-embed-text produces collapsed vectors on short inputs. The same
//! model loaded via sentence-transformers (canonical path) works correctly.
//!
//! The model is held resident inside the embedded Python interpreter,
//! across the GIL. First call pays the load cost; subsequent calls are fast.

use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyList, PyListMethods};

use crate::error::KbError;
use crate::extract::init_python_path;

/// Default model identifier — passed to sentence-transformers.
pub const DEFAULT_MODEL: &str = "nomic-ai/nomic-embed-text-v1.5";

/// Embedder bound to a specific sentence-transformers model.
///
/// Cheap to construct (does not load the model). The model is loaded inside
/// the Python interpreter on first call to `embed_texts`.
#[derive(Debug, Clone)]
pub struct Embedder {
    pub model: String,
    pub batch_size: usize,
}

impl Embedder {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            batch_size: 32,
        }
    }

    /// Encode a batch of texts. Returns L2-normalized vectors.
    pub fn embed_texts(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, KbError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        init_python_path()?;
        Python::with_gil(|py| {
            let module = py.import_bound("gw_kb_extract.embed")?;
            // Build a Python list of input strings
            let py_texts = PyList::empty_bound(py);
            for t in texts {
                py_texts.append(t.as_str())?;
            }
            // Call embed_texts(texts, model_name, batch_size)
            let kwargs = pyo3::types::PyDict::new_bound(py);
            kwargs.set_item("model_name", &self.model)?;
            kwargs.set_item("batch_size", self.batch_size)?;
            let result = module.call_method("embed_texts", (py_texts,), Some(&kwargs))?;
            // Result is list[list[float]]
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

    /// Convenience: embed a single string.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>, KbError> {
        let mut out = self.embed_texts(&[text.to_string()])?;
        out.pop()
            .ok_or_else(|| KbError::Other("embed returned no vectors".into()))
    }

    /// Query the model's embedding dimension. Loads the model if needed.
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

impl Default for Embedder {
    fn default() -> Self {
        Self::new(DEFAULT_MODEL)
    }
}
