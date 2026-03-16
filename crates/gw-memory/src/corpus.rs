//! Corpus search for gw-bench — tantivy BM25 + optional LanceDB vector search.
//!
//! Unlike `TantivyStore`/`HybridStore`, this has no scoping (org/user/agent/session),
//! stores full text in tantivy (STORED), and doesn't need Postgres.

use std::io::BufRead;
use std::path::Path;

use arrow_array::{Array, Float32Array, StringArray};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, TextFieldIndexing, TextOptions, Value, IndexRecordOption, STORED, STRING};
use tantivy::tokenizer::{Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer};
use tantivy::{doc, Index, IndexReader, ReloadPolicy};
use tracing;

use crate::error::MemoryError;
use crate::fusion::{reciprocal_rank_fusion, ScoredKey};

/// Tokenizer name for the text field — matches bm25s defaults:
/// lowercase + English stopword removal (no stemming).
const TOKENIZER_NAME: &str = "en_stopwords";

/// Build the text field options: TEXT + STORED, using our custom tokenizer.
fn text_field_options() -> TextOptions {
    TextOptions::default()
        .set_indexing_options(
            TextFieldIndexing::default()
                .set_tokenizer(TOKENIZER_NAME)
                .set_index_option(IndexRecordOption::WithFreqsAndPositions),
        )
        .set_stored()
}

/// Register our custom tokenizer on an index.
/// Pipeline: SimpleTokenizer → RemoveLongFilter(40) → LowerCaser → StopWordFilter(English)
fn register_tokenizer(index: &Index) {
    let analyzer = TextAnalyzer::builder(SimpleTokenizer::default())
        .filter(RemoveLongFilter::limit(40))
        .filter(LowerCaser)
        .filter(StopWordFilter::new(Language::English).unwrap())
        .build();
    index.tokenizers().register(TOKENIZER_NAME, analyzer);
}

/// A search hit from corpus search.
#[derive(Debug, Clone)]
pub struct CorpusHit {
    pub docid: String,
    pub text: String,
    pub score: f32,
}

/// In-process corpus searcher using tantivy BM25 and optional LanceDB vectors.
pub struct CorpusSearcher {
    // tantivy BM25
    index: Index,
    reader: IndexReader,
    f_docid: Field,
    f_text: Field,
    // optional LanceDB vector table
    lance_table: Option<lancedb::Table>,
}

impl CorpusSearcher {
    /// Open existing tantivy index (read-only) with optional LanceDB table.
    pub async fn open(
        tantivy_path: &Path,
        lance_db_path: Option<&str>,
        lance_table_name: Option<&str>,
    ) -> Result<Self, MemoryError> {
        // Open tantivy index
        let index = Index::open_in_dir(tantivy_path)
            .map_err(|e| MemoryError::Tantivy(format!("failed to open index: {e}")))?;

        // Register custom tokenizer so queries are analyzed consistently
        register_tokenizer(&index);

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e: tantivy::TantivyError| MemoryError::Tantivy(format!("{e}")))?;

        // Resolve field handles from the existing schema
        let schema = index.schema();
        let f_docid = schema
            .get_field("docid")
            .map_err(|e| MemoryError::Tantivy(format!("missing 'docid' field: {e}")))?;
        let f_text = schema
            .get_field("text")
            .map_err(|e| MemoryError::Tantivy(format!("missing 'text' field: {e}")))?;

        // Optionally open LanceDB table
        let lance_table = if let Some(db_path) = lance_db_path {
            let table_name = lance_table_name.unwrap_or("browsecomp_docs");
            let conn = lancedb::connect(db_path).execute().await?;
            match conn.open_table(table_name).execute().await {
                Ok(table) => {
                    tracing::info!(table = table_name, "Opened LanceDB corpus table");
                    Some(table)
                }
                Err(e) => {
                    tracing::warn!(error = %e, table = table_name, "Failed to open LanceDB table, vector search disabled");
                    None
                }
            }
        } else {
            None
        };

        tracing::info!(
            tantivy_path = %tantivy_path.display(),
            has_lance = lance_table.is_some(),
            "CorpusSearcher opened"
        );

        Ok(Self {
            index,
            reader,
            f_docid,
            f_text,
            lance_table,
        })
    }

    /// BM25 search. Returns docid + first `snippet_chars` of text as snippet.
    pub fn search_bm25(&self, query: &str, k: usize) -> Result<Vec<CorpusHit>, MemoryError> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_text]);

        // Sanitize query: strip characters that tantivy's query parser treats as syntax
        let sanitized: String = query
            .chars()
            .map(|c| match c {
                '"' | '\'' | '(' | ')' | '[' | ']' | '{' | '}' | '~' | '^' | '\\' => ' ',
                _ => c,
            })
            .collect();

        let parsed = query_parser.parse_query(&sanitized).map_err(|e| {
            MemoryError::Tantivy(format!("query parse error: {e}"))
        })?;

        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(k))
            .map_err(|e| MemoryError::Tantivy(format!("search failed: {e}")))?;

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(doc_address).map_err(|e| {
                MemoryError::Tantivy(format!("doc retrieval failed: {e}"))
            })?;

            let docid = doc
                .get_first(self.f_docid)
                .and_then(|v| Value::as_str(&v).map(|s| s.to_string()))
                .unwrap_or_default();

            let text = doc
                .get_first(self.f_text)
                .and_then(|v| Value::as_str(&v).map(|s| s.to_string()))
                .unwrap_or_default();

            // Return first 3000 chars as snippet
            let snippet: String = text.chars().take(3000).collect();

            hits.push(CorpusHit {
                docid,
                text: snippet,
                score,
            });
        }

        Ok(hits)
    }

    /// LanceDB vector search. Returns docid + text snippet.
    pub async fn search_vector(
        &self,
        query_vec: Vec<f32>,
        k: usize,
    ) -> Result<Vec<CorpusHit>, MemoryError> {
        let table = match &self.lance_table {
            Some(t) => t,
            None => {
                return Err(MemoryError::Embedding(
                    "LanceDB table not configured for vector search".into(),
                ))
            }
        };

        let mut stream = table
            .vector_search(query_vec)
            .map_err(|e| MemoryError::Embedding(format!("vector search error: {e}")))?
            .limit(k)
            .execute()
            .await?;

        let mut hits = Vec::new();
        while let Some(batch) = stream.try_next().await.map_err(|e| {
            MemoryError::Embedding(format!("error reading vector results: {e}"))
        })? {
            let docid_col = batch.column_by_name("docid");
            let text_col = batch.column_by_name("text");
            let dist_col = batch.column_by_name("_distance");

            if let (Some(docids), Some(texts), Some(dists)) = (docid_col, text_col, dist_col) {
                let docids = docids.as_any().downcast_ref::<StringArray>();
                let texts = texts.as_any().downcast_ref::<StringArray>();
                let dists = dists.as_any().downcast_ref::<Float32Array>();

                if let (Some(docids), Some(texts), Some(dists)) = (docids, texts, dists) {
                    for i in 0..docids.len() {
                        let docid = docids.value(i).to_string();
                        let text = texts.value(i);
                        let snippet: String = text.chars().take(3000).collect();
                        let distance = dists.value(i);
                        let score = 1.0 / (1.0 + distance);

                        hits.push(CorpusHit {
                            docid,
                            text: snippet,
                            score,
                        });
                    }
                }
            }
        }

        Ok(hits)
    }

    /// Hybrid search: BM25 + vector, merged with RRF.
    pub async fn search_hybrid(
        &self,
        query: &str,
        query_vec: Vec<f32>,
        k: usize,
    ) -> Result<Vec<CorpusHit>, MemoryError> {
        let bm25_hits = self.search_bm25(query, k)?;
        let vec_hits = self.search_vector(query_vec, k).await?;

        // Build ScoredKey lists for RRF
        let bm25_keys: Vec<ScoredKey> = bm25_hits
            .iter()
            .map(|h| ScoredKey {
                key: h.docid.clone(),
                score: h.score,
            })
            .collect();
        let vec_keys: Vec<ScoredKey> = vec_hits
            .iter()
            .map(|h| ScoredKey {
                key: h.docid.clone(),
                score: h.score,
            })
            .collect();

        let fused = reciprocal_rank_fusion(&[bm25_keys, vec_keys], 60);

        // Build a lookup of docid -> text from both result sets
        let mut text_map = std::collections::HashMap::new();
        for h in bm25_hits.iter().chain(vec_hits.iter()) {
            text_map.entry(h.docid.clone()).or_insert_with(|| h.text.clone());
        }

        let results: Vec<CorpusHit> = fused
            .into_iter()
            .take(k)
            .map(|(docid, score)| CorpusHit {
                text: text_map.remove(&docid).unwrap_or_default(),
                docid,
                score,
            })
            .collect();

        Ok(results)
    }

    /// Retrieve full document text by docid.
    pub fn get_document(&self, docid: &str) -> Result<Option<String>, MemoryError> {
        let searcher = self.reader.searcher();

        let term = tantivy::Term::from_field_text(self.f_docid, docid);
        let query = tantivy::query::TermQuery::new(
            term,
            tantivy::schema::IndexRecordOption::Basic,
        );

        let top_docs = searcher
            .search(&query, &TopDocs::with_limit(1))
            .map_err(|e| MemoryError::Tantivy(format!("get_document search failed: {e}")))?;

        if let Some((_score, doc_address)) = top_docs.first() {
            let doc: tantivy::TantivyDocument = searcher.doc(*doc_address).map_err(|e| {
                MemoryError::Tantivy(format!("doc retrieval failed: {e}"))
            })?;

            let text = doc
                .get_first(self.f_text)
                .and_then(|v| Value::as_str(&v).map(|s| s.to_string()));

            Ok(text)
        } else {
            Ok(None)
        }
    }

    /// Build a tantivy index from a corpus JSONL file.
    /// Each line must have `{"docid": "...", "text": "..."}`.
    /// Returns the number of documents indexed.
    pub fn build_index(jsonl_path: &Path, tantivy_out: &Path) -> Result<usize, MemoryError> {
        let mut schema_builder = Schema::builder();
        let f_docid = schema_builder.add_text_field("docid", STRING | STORED);
        let f_text = schema_builder.add_text_field("text", text_field_options());
        let schema = schema_builder.build();

        std::fs::create_dir_all(tantivy_out).map_err(|e| {
            MemoryError::Tantivy(format!("failed to create output dir: {e}"))
        })?;

        let index = Index::create_in_dir(tantivy_out, schema).map_err(|e| {
            MemoryError::Tantivy(format!("failed to create index: {e}"))
        })?;

        // Register custom tokenizer so indexing uses stopword removal
        register_tokenizer(&index);

        // 256 MB heap for bulk indexing
        let mut writer = index
            .writer(256_000_000)
            .map_err(|e| MemoryError::Tantivy(format!("{e}")))?;

        let file = std::fs::File::open(jsonl_path).map_err(|e| {
            MemoryError::Tantivy(format!("failed to open JSONL: {e}"))
        })?;
        let reader = std::io::BufReader::new(file);

        let mut count = 0usize;
        for (line_num, line) in reader.lines().enumerate() {
            let line = line.map_err(|e| {
                MemoryError::Tantivy(format!("read error at line {line_num}: {e}"))
            })?;
            if line.trim().is_empty() {
                continue;
            }

            let parsed: serde_json::Value = serde_json::from_str(&line).map_err(|e| {
                MemoryError::Tantivy(format!("JSON parse error at line {line_num}: {e}"))
            })?;

            let docid = parsed["docid"]
                .as_str()
                .ok_or_else(|| MemoryError::Tantivy(format!("missing 'docid' at line {line_num}")))?;
            let text = parsed["text"]
                .as_str()
                .ok_or_else(|| MemoryError::Tantivy(format!("missing 'text' at line {line_num}")))?;

            writer
                .add_document(doc!(
                    f_docid => docid,
                    f_text => text,
                ))
                .map_err(|e| MemoryError::Tantivy(format!("add_document failed: {e}")))?;

            count += 1;
            if count % 10_000 == 0 {
                tracing::info!(count, "Indexing progress");
            }
        }

        writer.commit().map_err(|e| {
            MemoryError::Tantivy(format!("commit failed: {e}"))
        })?;

        tracing::info!(count, "Corpus tantivy index built");
        Ok(count)
    }
}
