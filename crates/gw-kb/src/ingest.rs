//! Ingest pipeline: extract → upsert source → chunk → embed → store.
//!
//! Writes to Postgres (source + chunk metadata), LanceDB (chunk vectors),
//! and tantivy (chunk BM25). Postgres is the source of truth; the other
//! two are derived indexes.

use std::path::Path;
use std::sync::Arc;

use gw_llm::OllamaClient;
use sqlx::PgPool;
use tracing::{info, warn};

use crate::chunk::{chunk_markdown, insert_chunks, Chunk, ChunkOpts};
use crate::embed::Embedder;
use crate::error::KbError;
use crate::extract::{extract_html, extract_markdown, extract_pdf, Extracted};
use crate::index::{KbLanceStore, KbTantivyStore};
use crate::source::{delete_chunks_for_source, upsert_source, Source, UpsertOutcome, UpsertResult};

/// Bundle of stores the ingest pipeline writes to. Constructed once at
/// startup and reused across ingests.
#[derive(Clone)]
pub struct KbStores {
    pub pg: PgPool,
    pub lance: Arc<KbLanceStore>,
    pub tantivy: Arc<KbTantivyStore>,
    /// Sentence-transformers embedder for chunks + topic labels.
    /// Bypasses Ollama because Ollama's nomic-embed-text wrapper is broken
    /// on short label inputs.
    pub embedder: Arc<Embedder>,
    /// Ollama client for the tagger LLM (chat completions only — embeddings
    /// go through `embedder`).
    pub llm: Arc<OllamaClient>,
}

#[derive(Debug, Clone)]
pub struct IngestReport {
    pub source: Source,
    pub outcome: UpsertOutcome,
    pub chunks_written: usize,
}

/// Ingest a URL: fetch in Rust, dispatch based on Content-Type.
///
/// - `application/pdf` → save bytes to a temp file, run pymupdf4llm
/// - anything else      → assume HTML, hand the body to trafilatura
///
/// ### URL rewrites
/// arxiv `/abs/` URLs are rewritten to `/pdf/` so we ingest the full
/// paper, not just the abstract page. Any other host is used as-is.
pub async fn ingest_url(stores: &KbStores, url: &str) -> Result<IngestReport, KbError> {
    let url = rewrite_url_for_ingest(url);
    let url = url.as_str();
    info!(url, "ingesting url");
    let resp = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(60))
        .user_agent("gw-kb/0.1 (+https://github.com/anthropics)")
        .build()
        .map_err(|e| KbError::Other(format!("http client: {e}")))?
        .get(url)
        .send()
        .await
        .map_err(|e| KbError::Other(format!("fetch {url}: {e}")))?;

    let status = resp.status();
    if !status.is_success() {
        return Err(KbError::Other(format!("fetch {url}: HTTP {status}")));
    }
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|v| v.to_str().ok())
        .unwrap_or("")
        .to_lowercase();

    let extracted = if content_type.starts_with("application/pdf")
        || url.to_lowercase().ends_with(".pdf")
    {
        let bytes = resp
            .bytes()
            .await
            .map_err(|e| KbError::Other(format!("read body: {e}")))?;
        info!(url, bytes = bytes.len(), "fetched pdf, extracting");
        extract_pdf_bytes(&bytes)?
    } else {
        let body = resp
            .text()
            .await
            .map_err(|e| KbError::Other(format!("read body: {e}")))?;
        extract_html(url, Some(&body))?
    };

    finish_ingest(stores, Some(url), None, extracted).await
}

/// Public wrapper around `rewrite_url_for_ingest` for callers outside
/// this module (e.g. feed sync wants to pre-check the normalised URL
/// against kb_sources before paying for a full fetch).
pub fn rewrite_url_for_ingest_public(url: &str) -> String {
    rewrite_url_for_ingest(url)
}

/// Normalize a URL before ingest. Currently:
///   - arxiv.org/abs/ID → arxiv.org/pdf/ID (so we ingest the full paper,
///     not just the abstract page)
fn rewrite_url_for_ingest(url: &str) -> String {
    // Match both http and https variants of arxiv.org
    if let Some(rest) = url.strip_prefix("https://arxiv.org/abs/") {
        return format!("https://arxiv.org/pdf/{}", rest);
    }
    if let Some(rest) = url.strip_prefix("http://arxiv.org/abs/") {
        return format!("https://arxiv.org/pdf/{}", rest);
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn arxiv_abs_gets_rewritten_to_pdf() {
        assert_eq!(
            rewrite_url_for_ingest("https://arxiv.org/abs/2205.09707"),
            "https://arxiv.org/pdf/2205.09707"
        );
        assert_eq!(
            rewrite_url_for_ingest("http://arxiv.org/abs/2205.09707"),
            "https://arxiv.org/pdf/2205.09707"
        );
    }

    #[test]
    fn other_urls_are_passthrough() {
        assert_eq!(
            rewrite_url_for_ingest("https://en.wikipedia.org/wiki/Foo"),
            "https://en.wikipedia.org/wiki/Foo"
        );
        assert_eq!(
            rewrite_url_for_ingest("https://arxiv.org/pdf/2205.09707"),
            "https://arxiv.org/pdf/2205.09707"
        );
    }
}

/// Write PDF bytes to a temp file and run the file-based extractor.
/// The temp file is cleaned up before this function returns.
fn extract_pdf_bytes(bytes: &[u8]) -> Result<Extracted, KbError> {
    use std::io::Write;

    let mut tmp = std::env::temp_dir();
    let id = uuid::Uuid::new_v4();
    tmp.push(format!("gw-kb-{}.pdf", id));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(bytes)?;
        f.sync_all()?;
    }
    let result = extract_pdf(&tmp);
    let _ = std::fs::remove_file(&tmp);
    result
}

/// Ingest a local file. Format is detected from the extension.
pub async fn ingest_file(stores: &KbStores, path: &Path) -> Result<IngestReport, KbError> {
    info!(path = ?path, "ingesting file");
    let extracted = extract_file(path)?;
    let path_str = path
        .to_str()
        .ok_or_else(|| KbError::Other(format!("non-utf8 path: {:?}", path)))?;
    finish_ingest(stores, None, Some(path_str), extracted).await
}

/// Ingest a document whose text is already in hand — no extraction.
///
/// Used for corpora like BrowseComp-Plus where sources come pre-parsed
/// as `(docid, text, url)` tuples. Constructs an `Extracted` inline
/// and runs it through the same chunk → embed → persist path as
/// `ingest_url` and `ingest_file`.
///
/// `metadata` is persisted to `kb_sources.metadata` (jsonb). For
/// BrowseComp this is where we stash the `docid` so we can match
/// retrieved chunks against qrels at eval time without re-parsing
/// the original jsonl.
pub async fn ingest_inline(
    stores: &KbStores,
    url: &str,
    text: &str,
    metadata: Option<serde_json::Value>,
) -> Result<IngestReport, KbError> {
    let (title, published_at, body) = parse_frontmatter(text);
    let extracted = Extracted {
        markdown: body,
        title,
        author: None,
        published_at,
        source_format: "inline".into(),
        extractor: "inline".into(),
    };

    let report = finish_ingest(stores, Some(url), None, extracted).await?;

    // Stash metadata on the source row. We do this in a follow-up
    // UPDATE rather than threading metadata through upsert_source so
    // the source layer stays decoupled from ingest-specific concerns.
    if let Some(meta) = metadata {
        sqlx::query("UPDATE kb_sources SET metadata = $1 WHERE source_id = $2")
            .bind(meta)
            .bind(report.source.source_id)
            .execute(&stores.pg)
            .await?;
    }

    Ok(report)
}

/// Parse a YAML-ish frontmatter block from the top of a markdown
/// document. Returns `(title, published_at, body_without_frontmatter)`.
/// Handles the simple `---\ntitle: ...\ndate: ...\n---\n<body>` shape
/// that BrowseComp-Plus and most static-site generators produce. Any
/// fields beyond title and date are ignored; if the frontmatter is
/// missing or malformed, returns the text unchanged.
fn parse_frontmatter(text: &str) -> (Option<String>, Option<chrono::DateTime<chrono::Utc>>, String) {
    let trimmed = text.trim_start_matches('\u{feff}'); // strip BOM if present
    if !trimmed.starts_with("---\n") && !trimmed.starts_with("---\r\n") {
        return (None, None, text.to_string());
    }
    let rest = &trimmed[4..];
    // Find the closing `---` on its own line.
    let Some(end_idx) = rest.find("\n---\n").or_else(|| rest.find("\n---\r\n")) else {
        return (None, None, text.to_string());
    };
    let frontmatter = &rest[..end_idx];
    let body_start = end_idx + 5; // skip "\n---\n"
    let body = rest[body_start.min(rest.len())..].trim_start_matches('\n').to_string();

    let mut title: Option<String> = None;
    let mut date: Option<chrono::DateTime<chrono::Utc>> = None;
    for line in frontmatter.lines() {
        let line = line.trim();
        if let Some(rest) = line.strip_prefix("title:") {
            title = Some(rest.trim().trim_matches(|c| c == '"' || c == '\'').to_string());
        } else if let Some(rest) = line.strip_prefix("date:") {
            let raw = rest.trim().trim_matches(|c| c == '"' || c == '\'');
            // Try ISO date (YYYY-MM-DD) first, then full RFC3339
            if let Ok(d) = chrono::NaiveDate::parse_from_str(raw, "%Y-%m-%d") {
                date = d.and_hms_opt(0, 0, 0).map(|dt| dt.and_utc());
            } else if let Ok(dt) = chrono::DateTime::parse_from_rfc3339(raw) {
                date = Some(dt.with_timezone(&chrono::Utc));
            }
        }
    }

    (title, date, body)
}

fn extract_file(path: &Path) -> Result<Extracted, KbError> {
    let ext = path
        .extension()
        .and_then(|s| s.to_str())
        .map(|s| s.to_lowercase())
        .unwrap_or_default();
    match ext.as_str() {
        "pdf" => extract_pdf(path),
        "md" | "markdown" | "txt" => extract_markdown(path),
        other => Err(KbError::UnsupportedFormat(other.to_string())),
    }
}

async fn finish_ingest(
    stores: &KbStores,
    url: Option<&str>,
    file_path: Option<&str>,
    extracted: Extracted,
) -> Result<IngestReport, KbError> {
    let UpsertResult { source, outcome } =
        upsert_source(&stores.pg, url, file_path, &extracted).await?;

    let chunks_written = match outcome {
        UpsertOutcome::Unchanged => {
            info!(source_id = %source.source_id, "unchanged — skipping chunking");
            0
        }
        UpsertOutcome::Updated => {
            // Re-chunk: drop old chunks (and old index entries) first
            let removed = delete_chunks_for_source(&stores.pg, source.source_id).await?;
            stores.lance.delete_by_source(source.source_id).await?;
            stores.tantivy.delete_by_source(source.source_id)?;
            warn!(source_id = %source.source_id, removed, "content changed, re-chunking");
            write_chunks(stores, &source, &extracted).await?
        }
        UpsertOutcome::Inserted => write_chunks(stores, &source, &extracted).await?,
    };

    Ok(IngestReport {
        source,
        outcome,
        chunks_written,
    })
}

async fn write_chunks(
    stores: &KbStores,
    source: &Source,
    extracted: &Extracted,
) -> Result<usize, KbError> {
    // 1. Chunk
    let chunks: Vec<Chunk> = chunk_markdown(&extracted.markdown, ChunkOpts::default());
    verify_chunk_offsets(&extracted.markdown, &chunks)?;
    let count = chunks.len();
    if count == 0 {
        return Ok(0);
    }

    // 2. Persist to Postgres (source of truth) — gives us chunk_ids
    let chunk_ids = insert_chunks(&stores.pg, source.source_id, &chunks).await?;

    // 3. Embed via sentence-transformers (in-process Python)
    let texts: Vec<String> = chunks.iter().map(|c| c.content.clone()).collect();
    let vectors = stores.embedder.embed_texts(&texts)?;
    if vectors.len() != texts.len() {
        return Err(KbError::Other(format!(
            "embed returned {} vectors for {} texts",
            vectors.len(),
            texts.len()
        )));
    }

    // 4. Index in LanceDB
    stores
        .lance
        .insert_chunks(source.source_id, &chunk_ids, &texts, &vectors)
        .await?;

    // 5. Index in tantivy
    let heading_paths: Vec<Vec<String>> = chunks.iter().map(|c| c.heading_path.clone()).collect();
    stores
        .tantivy
        .insert_chunks(source.source_id, &source.title, &chunk_ids, &texts, &heading_paths)?;

    info!(source_id = %source.source_id, chunks = count, "chunks indexed");
    Ok(count)
}

#[cfg(test)]
mod frontmatter_tests {
    use super::parse_frontmatter;

    #[test]
    fn parses_title_and_date() {
        let text = "---\ntitle: Hello World\ndate: 2002-05-13\n---\nBody text here.";
        let (title, date, body) = parse_frontmatter(text);
        assert_eq!(title.as_deref(), Some("Hello World"));
        assert!(date.is_some());
        assert_eq!(body, "Body text here.");
    }

    #[test]
    fn missing_frontmatter_passes_through() {
        let text = "just some body\nwith multiple lines";
        let (title, date, body) = parse_frontmatter(text);
        assert!(title.is_none());
        assert!(date.is_none());
        assert_eq!(body, text);
    }

    #[test]
    fn malformed_frontmatter_passes_through() {
        // Opens with ---\n but never closes
        let text = "---\ntitle: Bad\nstill no closer";
        let (title, _, body) = parse_frontmatter(text);
        assert!(title.is_none());
        assert_eq!(body, text);
    }

    #[test]
    fn strips_quoted_title() {
        let text = "---\ntitle: \"Quoted Title\"\n---\nbody";
        let (title, _, _) = parse_frontmatter(text);
        assert_eq!(title.as_deref(), Some("Quoted Title"));
    }

    #[test]
    fn handles_browsecomp_sample() {
        let text = "---\ntitle: Arwa University holds annual cultural activities [Archives:2002/20/Local News]\ndate: 2002-05-13\n---\nArwa University holds annual cultural activities [Archives:2002/20/Local News]\n\nBY ABDUH AL-SABRI";
        let (title, date, body) = parse_frontmatter(text);
        assert!(title.as_deref().unwrap().contains("Arwa University"));
        assert!(date.is_some());
        assert!(body.starts_with("Arwa University"));
    }
}

/// Verify that every chunk's `(char_offset, char_length)` recovers its
/// content from the source markdown. Cheap correctness check on every ingest.
fn verify_chunk_offsets(markdown: &str, chunks: &[Chunk]) -> Result<(), KbError> {
    let doc_chars: Vec<char> = markdown.chars().collect();
    for c in chunks {
        let end = c.char_offset + c.char_length;
        if end > doc_chars.len() {
            return Err(KbError::Other(format!(
                "chunk {}: offset {} + length {} = {} exceeds doc len {}",
                c.ordinal,
                c.char_offset,
                c.char_length,
                end,
                doc_chars.len()
            )));
        }
        let slice: String = doc_chars[c.char_offset..end].iter().collect();
        if slice != c.content {
            return Err(KbError::Other(format!(
                "chunk {}: content does not match doc[{}..{}] (len {} vs {})",
                c.ordinal,
                c.char_offset,
                end,
                c.content.chars().count(),
                slice.chars().count()
            )));
        }
    }
    Ok(())
}
