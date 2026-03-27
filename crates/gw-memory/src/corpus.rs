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
use tantivy::query::{BooleanQuery, BoostQuery, Occur, PhraseQuery, QueryParser};
use tantivy::schema::{Field, Schema, TextFieldIndexing, TextOptions, Value, IndexRecordOption, STORED, STRING};
use tantivy::tokenizer::{Language, LowerCaser, RemoveLongFilter, SimpleTokenizer, StopWordFilter, TextAnalyzer, TokenStream};
use tantivy::{doc, Index, IndexReader, ReloadPolicy, Term};
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
    // tantivy BM25 (document-level)
    index: Index,
    reader: IndexReader,
    f_docid: Field,
    f_text: Field,
    // optional passage-level tantivy index
    passage_index: Option<PassageIndex>,
    // optional LanceDB single-vector table (nomic-embed-text)
    lance_table: Option<lancedb::Table>,
    // optional LanceDB multi-vector table (ColBERT)
    colbert_table: Option<lancedb::Table>,
}

/// Passage-level tantivy index — documents split into ~512-byte chunks.
struct PassageIndex {
    index: Index,
    reader: IndexReader,
    f_docid: Field,
    f_text: Field,
}

impl CorpusSearcher {
    /// Open existing tantivy index (read-only) with optional LanceDB tables.
    pub async fn open(
        tantivy_path: &Path,
        lance_db_path: Option<&str>,
        lance_table_name: Option<&str>,
        colbert_lance_path: Option<&str>,
        colbert_table_name: Option<&str>,
    ) -> Result<Self, MemoryError> {
        Self::open_with_passages(tantivy_path, None, lance_db_path, lance_table_name, colbert_lance_path, colbert_table_name).await
    }

    /// Open existing tantivy index with optional passage index and LanceDB tables.
    pub async fn open_with_passages(
        tantivy_path: &Path,
        passage_path: Option<&Path>,
        lance_db_path: Option<&str>,
        lance_table_name: Option<&str>,
        colbert_lance_path: Option<&str>,
        colbert_table_name: Option<&str>,
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

        // Optionally open passage-level index
        let passage_index = if let Some(p_path) = passage_path {
            match PassageIndex::open(p_path) {
                Ok(pi) => {
                    tracing::info!(path = %p_path.display(), "Opened passage-level index");
                    Some(pi)
                }
                Err(e) => {
                    tracing::warn!(error = %e, "Passage index unavailable, passage search disabled");
                    None
                }
            }
        } else {
            None
        };

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

        // Optionally open ColBERT multi-vector LanceDB table
        let colbert_table = if let Some(db_path) = colbert_lance_path {
            let table_name = colbert_table_name.unwrap_or("colbert_docs");
            let conn = lancedb::connect(db_path).execute().await?;
            match conn.open_table(table_name).execute().await {
                Ok(table) => {
                    tracing::info!(table = table_name, "Opened ColBERT LanceDB table");
                    Some(table)
                }
                Err(e) => {
                    tracing::warn!(error = %e, table = table_name, "Failed to open ColBERT table, ColBERT search disabled");
                    None
                }
            }
        } else {
            None
        };

        tracing::info!(
            tantivy_path = %tantivy_path.display(),
            has_passages = passage_index.is_some(),
            has_lance = lance_table.is_some(),
            has_colbert = colbert_table.is_some(),
            "CorpusSearcher opened"
        );

        Ok(Self {
            index,
            reader,
            f_docid,
            f_text,
            passage_index,
            lance_table,
            colbert_table,
        })
    }

    /// BM25 search. Returns docid + first `snippet_chars` of text as snippet.
    pub fn search_bm25(&self, query: &str, k: usize) -> Result<Vec<CorpusHit>, MemoryError> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_text]);

        let sanitized = sanitize_query(query);
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

    /// Tokenize a query string using the same analyzer as the text field.
    fn tokenize_query(&self, query: &str) -> Vec<String> {
        let tokenizer_manager = self.index.tokenizers();
        let mut tokenizer = tokenizer_manager
            .get(TOKENIZER_NAME)
            .expect("en_stopwords tokenizer not registered");
        let mut stream = tokenizer.token_stream(query);
        let mut terms = Vec::new();
        while stream.advance() {
            terms.push(stream.token().text.clone());
        }
        terms
    }

    /// Extract document fields from a tantivy document address.
    fn extract_hit(&self, searcher: &tantivy::Searcher, score: f32, doc_address: tantivy::DocAddress) -> Result<CorpusHit, MemoryError> {
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
        let snippet: String = text.chars().take(3000).collect();
        Ok(CorpusHit { docid, text: snippet, score })
    }

    /// BM25 search with compound boosted query stacking multiple signals:
    /// 1. Exact phrase match (boost 4.0)
    /// 2. Phrase with slop (boost 2.0)
    /// 3. All terms AND'd (boost 1.5)
    /// 4. Individual terms OR'd (boost 1.0, fallback)
    pub fn search_bm25_boosted(&self, query: &str, k: usize) -> Result<Vec<CorpusHit>, MemoryError> {
        let terms = self.tokenize_query(query);
        if terms.is_empty() {
            return Ok(vec![]);
        }

        let tantivy_terms: Vec<Term> = terms.iter()
            .map(|t| Term::from_field_text(self.f_text, t))
            .collect();

        let mut clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = Vec::new();

        if tantivy_terms.len() > 1 {
            // Signal 1: Exact phrase match (boost 4.0)
            let phrase = PhraseQuery::new(tantivy_terms.clone());
            clauses.push((
                Occur::Should,
                Box::new(BoostQuery::new(Box::new(phrase), 4.0)),
            ));

            // Signal 2: Phrase with slop 2 (boost 2.0) — terms near each other
            let positioned: Vec<(usize, Term)> = tantivy_terms.iter()
                .enumerate()
                .map(|(i, t)| (i, t.clone()))
                .collect();
            let phrase_slop = PhraseQuery::new_with_offset_and_slop(positioned, 2);
            clauses.push((
                Occur::Should,
                Box::new(BoostQuery::new(Box::new(phrase_slop), 2.0)),
            ));

            // Signal 3: All terms AND'd together (boost 1.5)
            let and_clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = tantivy_terms.iter()
                .map(|t| {
                    let tq = tantivy::query::TermQuery::new(
                        t.clone(),
                        IndexRecordOption::WithFreqs,
                    );
                    (Occur::Must, Box::new(tq) as Box<dyn tantivy::query::Query>)
                })
                .collect();
            let and_query = BooleanQuery::new(and_clauses);
            clauses.push((
                Occur::Should,
                Box::new(BoostQuery::new(Box::new(and_query), 1.5)),
            ));
        }

        // Signal 4: Individual terms OR'd (boost 1.0) — broadest fallback
        for t in &tantivy_terms {
            let tq = tantivy::query::TermQuery::new(
                t.clone(),
                IndexRecordOption::WithFreqs,
            );
            clauses.push((Occur::Should, Box::new(tq)));
        }

        let compound = BooleanQuery::new(clauses);
        let searcher = self.reader.searcher();
        let top_docs = searcher
            .search(&compound, &TopDocs::with_limit(k))
            .map_err(|e| MemoryError::Tantivy(format!("boosted search failed: {e}")))?;

        let mut hits = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            hits.push(self.extract_hit(&searcher, score, doc_address)?);
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

    /// ColBERT multi-vector search using LanceDB native MaxSim.
    ///
    /// `query_token_vecs` is a list of 128-dim token embeddings from the ColBERT encoder.
    /// LanceDB computes MaxSim (sum of per-query-token max similarities) natively.
    pub async fn search_colbert(
        &self,
        query_token_vecs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<CorpusHit>, MemoryError> {
        let table = match &self.colbert_table {
            Some(t) => t,
            None => {
                return Err(MemoryError::Embedding(
                    "ColBERT LanceDB table not configured".into(),
                ))
            }
        };

        if query_token_vecs.is_empty() {
            return Ok(vec![]);
        }

        // For multi-vector search, pass each token vector as a flat Vec<f32>.
        // LanceDB detects the column is List(FixedSizeList) and routes to MaxSim.
        // First token goes to vector_search(), rest via add_query_vector().
        let mut vq = table
            .vector_search(query_token_vecs[0].clone())
            .map_err(|e| MemoryError::Embedding(format!("ColBERT search error: {e}")))?;

        for token_vec in query_token_vecs.iter().skip(1) {
            vq = vq.add_query_vector(token_vec.clone())
                .map_err(|e| MemoryError::Embedding(format!("ColBERT add_query_vector error: {e}")))?;
        }

        let mut stream = vq
            .column("vector")
            .limit(k)
            .execute()
            .await?;

        let mut hits = Vec::new();
        while let Some(batch) = stream.try_next().await.map_err(|e| {
            MemoryError::Embedding(format!("error reading ColBERT results: {e}"))
        })? {
            let docid_col = batch.column_by_name("docid");
            let dist_col = batch.column_by_name("_distance");

            if let (Some(docids), Some(dists)) = (docid_col, dist_col) {
                let docids = docids.as_any().downcast_ref::<StringArray>();
                let dists = dists.as_any().downcast_ref::<Float32Array>();

                if let (Some(docids), Some(dists)) = (docids, dists) {
                    for i in 0..docids.len() {
                        let docid = docids.value(i).to_string();
                        let distance = dists.value(i);
                        // MaxSim returns negative distance (higher = better)
                        let score = -distance;

                        // Look up text from tantivy (ColBERT table may not store text)
                        let text = self.get_document(&docid)?
                            .map(|t| t.chars().take(3000).collect::<String>())
                            .unwrap_or_default();

                        hits.push(CorpusHit {
                            docid,
                            text,
                            score,
                        });
                    }
                }
            }
        }

        Ok(hits)
    }

    /// Hybrid search: BM25-first with ColBERT augmentation.
    ///
    /// BM25 results maintain their order. ColBERT-unique documents (not in BM25
    /// top-k) are appended at the end. This preserves BM25 precision while
    /// adding retrieval diversity from ColBERT's reasoning-trained embeddings.
    pub async fn search_hybrid_colbert(
        &self,
        query: &str,
        query_token_vecs: &[Vec<f32>],
        k: usize,
    ) -> Result<Vec<CorpusHit>, MemoryError> {
        let bm25_hits = self.search_bm25_boosted(query, k)?;
        let colbert_hits = self.search_colbert(query_token_vecs, k).await?;

        // BM25 results first, in original order
        let mut seen: std::collections::HashSet<String> = std::collections::HashSet::new();
        let mut results: Vec<CorpusHit> = Vec::with_capacity(k * 2);

        for h in &bm25_hits {
            seen.insert(h.docid.clone());
            results.push(h.clone());
        }

        // Append ColBERT-unique docs (not already in BM25 results)
        for h in &colbert_hits {
            if seen.insert(h.docid.clone()) {
                results.push(h.clone());
            }
        }

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

    /// Build a passage-level tantivy index from the same JSONL corpus.
    ///
    /// Each document is split into ~`chunk_bytes`-byte passages with
    /// `overlap_bytes` overlap. Passages inherit the parent's docid.
    pub fn build_passage_index(
        jsonl_path: &Path,
        tantivy_out: &Path,
        chunk_bytes: usize,
        overlap_bytes: usize,
    ) -> Result<usize, MemoryError> {
        let mut schema_builder = Schema::builder();
        let f_passage_id = schema_builder.add_text_field("passage_id", STRING | STORED);
        let f_docid = schema_builder.add_text_field("docid", STRING | STORED);
        let f_text = schema_builder.add_text_field("text", text_field_options());
        let schema = schema_builder.build();

        std::fs::create_dir_all(tantivy_out).map_err(|e| {
            MemoryError::Tantivy(format!("failed to create passage index dir: {e}"))
        })?;

        let index = Index::create_in_dir(tantivy_out, schema).map_err(|e| {
            MemoryError::Tantivy(format!("failed to create passage index: {e}"))
        })?;
        register_tokenizer(&index);

        let mut writer = index
            .writer(256_000_000)
            .map_err(|e| MemoryError::Tantivy(format!("{e}")))?;

        let file = std::fs::File::open(jsonl_path).map_err(|e| {
            MemoryError::Tantivy(format!("failed to open JSONL: {e}"))
        })?;
        let reader = std::io::BufReader::new(file);

        let mut passage_count = 0usize;
        let mut doc_count = 0usize;
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

            // Split into passages
            let passages = split_into_passages(text, chunk_bytes, overlap_bytes);
            for (i, passage) in passages.iter().enumerate() {
                let pid = format!("{docid}#p{i}");
                writer
                    .add_document(doc!(
                        f_passage_id => pid.as_str(),
                        f_docid => docid,
                        f_text => passage.as_str(),
                    ))
                    .map_err(|e| MemoryError::Tantivy(format!("add_document failed: {e}")))?;
                passage_count += 1;
            }

            doc_count += 1;
            if doc_count % 10_000 == 0 {
                tracing::info!(doc_count, passage_count, "Passage indexing progress");
            }
        }

        writer.commit().map_err(|e| {
            MemoryError::Tantivy(format!("commit failed: {e}"))
        })?;

        tracing::info!(doc_count, passage_count, "Passage tantivy index built");
        Ok(passage_count)
    }

    /// Search passages, returning document-level hits (deduplicated by docid).
    ///
    /// Uses the passage index for retrieval, then maps passage hits back to
    /// their parent documents. Returns `CorpusHit`s with the passage text as
    /// the snippet (more focused than document-level snippets).
    /// Whether a passage-level index is available.
    pub fn has_passage_index(&self) -> bool {
        self.passage_index.is_some()
    }

    pub fn search_passages(&self, query: &str, k: usize) -> Result<Vec<CorpusHit>, MemoryError> {
        let pi = match &self.passage_index {
            Some(pi) => pi,
            None => return Ok(vec![]),
        };
        pi.search(query, k)
    }

    /// Multi-strategy search: doc-level BM25 + passage-level BM25, fused via RRF.
    ///
    /// Returns deduplicated results from both indexes, ranked by RRF score.
    /// Passage hits that surface documents not in the doc-level top-k are the
    /// key win — they catch answers buried in long documents.
    pub fn search_with_passages(&self, query: &str, k: usize) -> Result<Vec<CorpusHit>, MemoryError> {
        // Doc-level BM25
        let doc_hits = self.search_bm25_boosted(query, k)?;
        let doc_scored: Vec<ScoredKey> = doc_hits.iter()
            .map(|h| ScoredKey { key: h.docid.clone(), score: h.score })
            .collect();

        // Passage-level BM25 (if available)
        let passage_hits = self.search_passages(query, k)?;
        let passage_scored: Vec<ScoredKey> = passage_hits.iter()
            .map(|h| ScoredKey { key: h.docid.clone(), score: h.score })
            .collect();

        if passage_scored.is_empty() {
            return Ok(doc_hits);
        }

        // RRF fusion
        let fused = reciprocal_rank_fusion(&[doc_scored, passage_scored], 60);

        // Build result list — prefer passage snippets for passage-surfaced docs
        let doc_map: std::collections::HashMap<&str, &CorpusHit> =
            doc_hits.iter().map(|h| (h.docid.as_str(), h)).collect();
        let passage_map: std::collections::HashMap<&str, &CorpusHit> =
            passage_hits.iter().map(|h| (h.docid.as_str(), h)).collect();

        let results: Vec<CorpusHit> = fused
            .into_iter()
            .take(k)
            .map(|(docid, score)| {
                // Use passage snippet if available (more focused), else doc snippet
                let text = passage_map
                    .get(docid.as_str())
                    .or_else(|| doc_map.get(docid.as_str()))
                    .map(|h| h.text.clone())
                    .unwrap_or_default();
                CorpusHit { docid, text, score }
            })
            .collect();

        Ok(results)
    }
}

impl PassageIndex {
    fn open(path: &Path) -> Result<Self, MemoryError> {
        let index = Index::open_in_dir(path)
            .map_err(|e| MemoryError::Tantivy(format!("failed to open passage index: {e}")))?;
        register_tokenizer(&index);

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::Manual)
            .try_into()
            .map_err(|e: tantivy::TantivyError| MemoryError::Tantivy(format!("{e}")))?;

        let schema = index.schema();
        let f_docid = schema.get_field("docid")
            .map_err(|e| MemoryError::Tantivy(format!("missing 'docid' field: {e}")))?;
        let f_text = schema.get_field("text")
            .map_err(|e| MemoryError::Tantivy(format!("missing 'text' field: {e}")))?;

        Ok(Self { index, reader, f_docid, f_text })
    }

    fn search(&self, query: &str, k: usize) -> Result<Vec<CorpusHit>, MemoryError> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_text]);
        let sanitized = sanitize_query(query);
        let parsed = query_parser.parse_query(&sanitized)
            .map_err(|e| MemoryError::Tantivy(format!("passage query parse error: {e}")))?;

        // Retrieve more than k to account for dedup by docid
        let top_docs = searcher
            .search(&parsed, &TopDocs::with_limit(k * 3))
            .map_err(|e| MemoryError::Tantivy(format!("passage search failed: {e}")))?;

        let mut seen_docids = std::collections::HashSet::new();
        let mut hits = Vec::with_capacity(k);

        for (score, doc_address) in top_docs {
            let doc = searcher.doc::<tantivy::TantivyDocument>(doc_address)
                .map_err(|e| MemoryError::Tantivy(format!("doc retrieval failed: {e}")))?;

            let docid = doc.get_first(self.f_docid)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            if docid.is_empty() || !seen_docids.insert(docid.clone()) {
                continue; // Skip duplicates
            }

            let text = doc.get_first(self.f_text)
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_string();

            hits.push(CorpusHit { docid, text, score });
            if hits.len() >= k {
                break;
            }
        }

        Ok(hits)
    }
}

/// Split text into passages of approximately `chunk_bytes` bytes
/// with `overlap_bytes` overlap between consecutive passages.
///
/// All byte offsets are snapped to UTF-8 character boundaries via
/// [`snap_to_char_boundary`], so this never panics on multi-byte content.
/// Splits on sentence boundaries (`. `, `\n`) when possible. The `. `
/// heuristic can false-positive on abbreviations ("Dr. Smith", "U.S."),
/// but the overlap between passages ensures the full term appears intact
/// in at least one passage.
fn split_into_passages(text: &str, chunk_bytes: usize, overlap_bytes: usize) -> Vec<String> {
    if text.len() <= chunk_bytes {
        return vec![text.to_string()];
    }

    let mut passages = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let end = snap_to_char_boundary(text, (start + chunk_bytes).min(text.len()));

        // Try to break at a sentence boundary within the last 20% of the chunk
        let break_zone_start = snap_to_char_boundary(text, (start + (chunk_bytes * 4 / 5)).min(end));
        let actual_end = if end < text.len() {
            let zone = &text[break_zone_start..end];
            zone.rfind(". ")
                .or_else(|| zone.rfind('\n'))
                .map(|offset| snap_to_char_boundary(text, break_zone_start + offset + 2))
                .unwrap_or(end)
        } else {
            end
        };

        passages.push(text[start..actual_end].to_string());

        if actual_end >= text.len() {
            break;
        }

        // Advance with overlap, snapping to char boundary
        start = snap_to_char_boundary(text, actual_end.saturating_sub(overlap_bytes));
    }

    passages
}

/// Snap a byte offset to the nearest valid UTF-8 character boundary.
/// Moves forward (toward end of string) to find a valid boundary.
fn snap_to_char_boundary(text: &str, offset: usize) -> usize {
    if offset >= text.len() {
        return text.len();
    }
    // Find the next char boundary at or after offset
    let mut pos = offset;
    while pos < text.len() && !text.is_char_boundary(pos) {
        pos += 1;
    }
    pos
}

/// Sanitize a query string for tantivy's query parser.
fn sanitize_query(query: &str) -> String {
    query
        .chars()
        .map(|c| {
            if "+-&|!(){}[]^\"~*?:\\/".contains(c) {
                ' '
            } else {
                c
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_short_text() {
        let passages = split_into_passages("Hello world", 512, 100);
        assert_eq!(passages.len(), 1);
        assert_eq!(passages[0], "Hello world");
    }

    #[test]
    fn split_into_multiple_passages() {
        let text = "A".repeat(1200);
        let passages = split_into_passages(&text, 512, 100);
        assert!(passages.len() >= 2, "should produce multiple passages, got {}", passages.len());
        // Each passage should be roughly chunk_bytes
        for p in &passages {
            assert!(p.len() <= 600, "passage too long: {}", p.len());
        }
    }

    #[test]
    fn split_preserves_all_content() {
        let text = "The quick brown fox. Jumps over the lazy dog. More text here for testing passage splitting with overlap.";
        let passages = split_into_passages(text, 50, 10);
        // All content should appear in at least one passage
        assert!(passages.iter().any(|p| p.contains("quick")));
        assert!(passages.iter().any(|p| p.contains("lazy")));
        assert!(passages.iter().any(|p| p.contains("overlap")));
    }

    #[test]
    fn split_multibyte_utf8() {
        // Each char here is 2-4 bytes — slicing on arbitrary byte offsets would panic
        let text = "Ünïcödé tëxt wïth äccënts. Más información aquí. 日本語テスト。";
        let passages = split_into_passages(text, 30, 10);
        assert!(passages.len() >= 2, "should split, got {} passages", passages.len());
        // All passages should be valid UTF-8 (no panic)
        for p in &passages {
            assert!(p.len() <= 50, "passage too long: {} bytes", p.len());
        }
    }

    #[test]
    fn split_respects_overlap() {
        let text = "A".repeat(200);
        let passages = split_into_passages(&text, 100, 20);
        assert!(passages.len() >= 2);
        // With overlap, the second passage should start before the first ends
        // Total coverage should exceed text length
        let total_chars: usize = passages.iter().map(|p| p.len()).sum();
        assert!(total_chars > text.len(), "overlap should cause total > original");
    }
}
