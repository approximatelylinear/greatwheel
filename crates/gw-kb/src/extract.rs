//! Document extraction via embedded Python (trafilatura, pymupdf4llm).
//!
//! The Python helper package lives at `crates/gw-kb/python/gw_kb_extract/`.
//! Its directory must be on `sys.path` before any extract call. Set
//! `GW_KB_PYTHON_PATH` at runtime, or call [`init_python_path`] manually
//! with the path.

use std::path::Path;
use std::sync::Once;

use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use pyo3::types::{PyAnyMethods, PyDict, PyDictMethods, PyList, PyListMethods};
use serde::{Deserialize, Serialize};

use crate::error::KbError;

/// Result of extracting a single document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Extracted {
    pub markdown: String,
    pub title: Option<String>,
    pub author: Option<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub source_format: String,
    pub extractor: String,
}

static INIT: Once = Once::new();

/// Add the gw-kb Python helper package to `sys.path`.
///
/// Idempotent — safe to call multiple times. Reads `GW_KB_PYTHON_PATH`
/// from the environment if set; otherwise tries
/// `<CARGO_MANIFEST_DIR>/python` as a development fallback.
pub fn init_python_path() -> Result<(), KbError> {
    let mut result = Ok(());
    INIT.call_once(|| {
        // Set PYTHONHOME from compile-time baked value before any GIL
        // acquisition. uv-managed standalone Python builds have hardcoded
        // `/install` paths in their build config, so we must override at
        // runtime or `Py_Initialize` fails to find the stdlib.
        if std::env::var_os("PYTHONHOME").is_none() {
            if let Some(home) = option_env!("GW_KB_PYTHONHOME") {
                if !home.is_empty() {
                    std::env::set_var("PYTHONHOME", home);
                }
            }
        }

        let path = std::env::var("GW_KB_PYTHON_PATH").unwrap_or_else(|_| {
            // Dev fallback: assume we're running from the workspace root
            format!("{}/python", env!("CARGO_MANIFEST_DIR"))
        });

        result = Python::with_gil(|py| -> Result<(), KbError> {
            let sys = py.import_bound("sys")?;
            let sys_path: Bound<PyList> = sys.getattr("path")?.downcast_into()?;
            sys_path.insert(0, path)?;

            // Also add the venv site-packages so trafilatura/pymupdf4llm
            // are importable. The venv prefix is baked at build time.
            if let Some(prefix) = option_env!("GW_KB_PYTHON_PREFIX") {
                if !prefix.is_empty() {
                    let site = format!("{}/lib/python3.12/site-packages", prefix);
                    sys_path.insert(1, site)?;
                }
            }
            Ok(())
        });
    });
    result
}

/// Extract clean markdown from an HTML page.
///
/// If `html` is `None`, trafilatura will fetch the URL itself.
pub fn extract_html(url: &str, html: Option<&str>) -> Result<Extracted, KbError> {
    init_python_path()?;
    Python::with_gil(|py| {
        let module = py.import_bound("gw_kb_extract.html")?;
        let result = match html {
            Some(h) => module.call_method1("extract", (url, h))?,
            None => module.call_method1("extract", (url,))?,
        };
        let dict: Bound<PyDict> = result.downcast_into()?;
        Extracted::from_py_dict(&dict, "html")
    })
}

/// Extract markdown from a PDF file.
pub fn extract_pdf(path: &Path) -> Result<Extracted, KbError> {
    init_python_path()?;
    let path_str = path
        .to_str()
        .ok_or_else(|| KbError::Other(format!("non-utf8 path: {:?}", path)))?;
    Python::with_gil(|py| {
        let module = py.import_bound("gw_kb_extract.pdf")?;
        let result = module.call_method1("extract", (path_str,))?;
        let dict: Bound<PyDict> = result.downcast_into()?;
        Extracted::from_py_dict(&dict, "pdf")
    })
}

/// Read a markdown / plaintext file and parse YAML frontmatter if present.
pub fn extract_markdown(path: &Path) -> Result<Extracted, KbError> {
    init_python_path()?;
    let path_str = path
        .to_str()
        .ok_or_else(|| KbError::Other(format!("non-utf8 path: {:?}", path)))?;
    Python::with_gil(|py| {
        let module = py.import_bound("gw_kb_extract.markdown")?;
        let result = module.call_method1("extract", (path_str,))?;
        let dict: Bound<PyDict> = result.downcast_into()?;
        Extracted::from_py_dict(&dict, "markdown")
    })
}

impl Extracted {
    fn from_py_dict(dict: &Bound<PyDict>, default_format: &str) -> Result<Self, KbError> {
        let markdown: String = dict
            .get_item("markdown")?
            .ok_or_else(|| KbError::Extraction("missing markdown field".into()))?
            .extract()?;

        let title: Option<String> = dict
            .get_item("title")?
            .and_then(|v| v.extract().ok());
        let author: Option<String> = dict
            .get_item("author")?
            .and_then(|v| v.extract().ok());
        let published_raw: Option<String> = dict
            .get_item("published_at")?
            .and_then(|v| v.extract().ok());
        let source_format: String = dict
            .get_item("source_format")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| default_format.to_string());
        let extractor: String = dict
            .get_item("extractor")?
            .and_then(|v| v.extract().ok())
            .unwrap_or_else(|| "unknown".to_string());

        let published_at = published_raw.as_deref().and_then(parse_loose_datetime);

        Ok(Extracted {
            markdown,
            title,
            author,
            published_at,
            source_format,
            extractor,
        })
    }
}

/// Parse various datetime string shapes that extractors emit.
///
/// Trafilatura tends to emit `YYYY-MM-DD`. Pymupdf returns full ISO 8601.
/// Frontmatter is freeform.
fn parse_loose_datetime(s: &str) -> Option<DateTime<Utc>> {
    // Try full RFC3339 first
    if let Ok(dt) = DateTime::parse_from_rfc3339(s) {
        return Some(dt.with_timezone(&Utc));
    }
    // Try date-only
    if let Ok(date) = chrono::NaiveDate::parse_from_str(s, "%Y-%m-%d") {
        return date.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
    }
    None
}
