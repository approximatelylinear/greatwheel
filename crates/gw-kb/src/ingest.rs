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
pub async fn ingest_url(stores: &KbStores, url: &str) -> Result<IngestReport, KbError> {
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
