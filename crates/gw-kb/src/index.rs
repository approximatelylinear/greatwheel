//! LanceDB (vector) and tantivy (BM25) indexes for knowledge base chunks.
//!
//! These are KB-specific indexes — independent from `gw-memory`'s agent
//! memory stores. The schema is simpler (no org/user/agent/session scoping)
//! and richer (heading path, source title for boosting).

use std::path::Path;
use std::sync::{Arc, Mutex};

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use gw_memory::fusion::ScoredKey;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection};
use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{
    Field as TField, IndexRecordOption, Schema as TSchema, Value, FAST, STORED, STRING, TEXT,
};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, TantivyDocument, Term};
use tracing::info;
use uuid::Uuid;

use crate::error::KbError;

const KB_TABLE: &str = "kb_chunks";

// ---------- LanceDB ----------

/// LanceDB vector store for KB chunks. Single table `kb_chunks`.
pub struct KbLanceStore {
    conn: Connection,
    dim: i32,
}

impl KbLanceStore {
    pub async fn open(path: &str, dim: i32) -> Result<Self, KbError> {
        let conn = connect(path).execute().await.map_err(lance_err)?;
        Ok(Self { conn, dim })
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("chunk_id", DataType::Utf8, false),
            Field::new("source_id", DataType::Utf8, false),
            Field::new("content", DataType::Utf8, false),
            Field::new(
                "vector",
                DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float32, true)),
                    self.dim,
                ),
                false,
            ),
        ]))
    }

    async fn open_or_create(&self) -> Result<lancedb::Table, KbError> {
        match self.conn.open_table(KB_TABLE).execute().await {
            Ok(table) => Ok(table),
            Err(_) => {
                info!(table = KB_TABLE, "creating new lance table");
                let schema = self.schema();
                let empty = RecordBatch::new_empty(schema.clone());
                let batches = RecordBatchIterator::new(vec![Ok(empty)], schema);
                self.conn
                    .create_table(KB_TABLE, Box::new(batches))
                    .execute()
                    .await
                    .map_err(lance_err)
            }
        }
    }

    /// Insert a batch of chunks. Caller has already deleted any prior rows
    /// for the source via [`delete_by_source`].
    pub async fn insert_chunks(
        &self,
        source_id: Uuid,
        chunk_ids: &[Uuid],
        contents: &[String],
        vectors: &[Vec<f32>],
    ) -> Result<(), KbError> {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        if chunk_ids.len() != contents.len() || chunk_ids.len() != vectors.len() {
            return Err(KbError::Other(
                "insert_chunks: chunk_ids/contents/vectors length mismatch".into(),
            ));
        }
        for v in vectors {
            if v.len() != self.dim as usize {
                return Err(KbError::Other(format!(
                    "embedding dim mismatch: expected {}, got {}",
                    self.dim,
                    v.len()
                )));
            }
        }

        let table = self.open_or_create().await?;

        let chunk_id_strs: Vec<String> = chunk_ids.iter().map(|u| u.to_string()).collect();
        let chunk_id_refs: Vec<&str> = chunk_id_strs.iter().map(|s| s.as_str()).collect();
        let source_str = source_id.to_string();
        let source_repeated: Vec<&str> = (0..chunk_ids.len()).map(|_| source_str.as_str()).collect();
        let content_refs: Vec<&str> = contents.iter().map(|s| s.as_str()).collect();

        let key_array = StringArray::from(chunk_id_refs);
        let source_array = StringArray::from(source_repeated);
        let content_array = StringArray::from(content_refs);

        let flat: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let values = Float32Array::from(flat);
        let item_field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = FixedSizeListArray::try_new(item_field, self.dim, Arc::new(values), None)
            .map_err(|e| KbError::Other(format!("arrow vector array: {e}")))?;

        let batch = RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(key_array),
                Arc::new(source_array),
                Arc::new(content_array),
                Arc::new(vector_array),
            ],
        )
        .map_err(|e| KbError::Other(format!("arrow record batch: {e}")))?;

        let schema = self.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(Box::new(batches)).execute().await.map_err(lance_err)?;
        Ok(())
    }

    /// Delete all rows for a source. Used before re-indexing changed content.
    pub async fn delete_by_source(&self, source_id: Uuid) -> Result<(), KbError> {
        let table = match self.conn.open_table(KB_TABLE).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };
        let filter = format!("source_id = '{}'", source_id);
        table.delete(&filter).await.map_err(lance_err)?;
        Ok(())
    }

    /// Vector similarity search. Returns chunk_ids as `ScoredKey.key`.
    pub async fn search(&self, query_vec: Vec<f32>, k: usize) -> Result<Vec<ScoredKey>, KbError> {
        let table = match self.conn.open_table(KB_TABLE).execute().await {
            Ok(t) => t,
            Err(_) => return Ok(vec![]),
        };

        let mut stream = table
            .vector_search(query_vec)
            .map_err(|e| KbError::Other(format!("vector search: {e}")))?
            .limit(k)
            .execute()
            .await
            .map_err(lance_err)?;

        let mut out = Vec::new();
        while let Some(batch) = stream
            .try_next()
            .await
            .map_err(|e| KbError::Other(format!("vector stream: {e}")))?
        {
            let key_col = batch.column_by_name("chunk_id");
            let dist_col = batch.column_by_name("_distance");
            if let (Some(k), Some(d)) = (key_col, dist_col) {
                let keys = k.as_any().downcast_ref::<StringArray>();
                let dists = d.as_any().downcast_ref::<Float32Array>();
                if let (Some(keys), Some(dists)) = (keys, dists) {
                    for i in 0..keys.len() {
                        out.push(ScoredKey {
                            key: keys.value(i).to_string(),
                            score: 1.0 / (1.0 + dists.value(i)),
                        });
                    }
                }
            }
        }
        Ok(out)
    }
}

fn lance_err(e: lancedb::Error) -> KbError {
    KbError::Other(format!("lancedb: {e}"))
}

// ---------- Tantivy ----------

/// Tantivy BM25 store for KB chunks. Single index covering all sources.
///
/// Schema:
///   - `chunk_id`   STRING | STORED  — UUID returned in results
///   - `source_id`  STRING | FAST   — for filtering / faceting
///   - `title`      TEXT  | STORED  — source title (boosted at query time)
///   - `headings`   TEXT            — joined heading path (boosted)
///   - `content`    TEXT            — chunk body
pub struct KbTantivyStore {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    f_chunk_id: TField,
    f_source_id: TField,
    f_title: TField,
    f_headings: TField,
    f_content: TField,
}

impl KbTantivyStore {
    pub fn open(index_path: &Path) -> Result<Self, KbError> {
        let mut sb = TSchema::builder();
        let f_chunk_id = sb.add_text_field("chunk_id", STRING | STORED);
        let f_source_id = sb.add_text_field("source_id", STRING | FAST | STORED);
        let f_title = sb.add_text_field("title", TEXT | STORED);
        let f_headings = sb.add_text_field("headings", TEXT);
        let f_content = sb.add_text_field("content", TEXT);
        let schema = sb.build();

        std::fs::create_dir_all(index_path)
            .map_err(|e| KbError::Other(format!("tantivy mkdir: {e}")))?;

        let index = if Index::open_in_dir(index_path).is_ok() {
            Index::open_in_dir(index_path).map_err(|e| KbError::Other(format!("tantivy open: {e}")))?
        } else {
            Index::create_in_dir(index_path, schema)
                .map_err(|e| KbError::Other(format!("tantivy create: {e}")))?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e: tantivy::TantivyError| KbError::Other(format!("tantivy reader: {e}")))?;
        let writer = index
            .writer(50_000_000)
            .map_err(|e| KbError::Other(format!("tantivy writer: {e}")))?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            f_chunk_id,
            f_source_id,
            f_title,
            f_headings,
            f_content,
        })
    }

    /// Index a batch of chunks. Caller has already deleted prior rows for
    /// the source via [`delete_by_source`].
    pub fn insert_chunks(
        &self,
        source_id: Uuid,
        title: &str,
        chunk_ids: &[Uuid],
        contents: &[String],
        heading_paths: &[Vec<String>],
    ) -> Result<(), KbError> {
        if chunk_ids.is_empty() {
            return Ok(());
        }
        if chunk_ids.len() != contents.len() || chunk_ids.len() != heading_paths.len() {
            return Err(KbError::Other(
                "tantivy insert_chunks: length mismatch".into(),
            ));
        }
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| KbError::Other(format!("tantivy writer poisoned: {e}")))?;
        let source_str = source_id.to_string();
        for ((cid, content), headings) in chunk_ids.iter().zip(contents).zip(heading_paths) {
            let headings_joined = headings.join(" ");
            let doc = doc!(
                self.f_chunk_id => cid.to_string(),
                self.f_source_id => source_str.as_str(),
                self.f_title => title,
                self.f_headings => headings_joined.as_str(),
                self.f_content => content.as_str(),
            );
            writer
                .add_document(doc)
                .map_err(|e| KbError::Other(format!("tantivy add_document: {e}")))?;
        }
        writer
            .commit()
            .map_err(|e| KbError::Other(format!("tantivy commit: {e}")))?;
        Ok(())
    }

    pub fn delete_by_source(&self, source_id: Uuid) -> Result<(), KbError> {
        let mut writer = self
            .writer
            .lock()
            .map_err(|e| KbError::Other(format!("tantivy writer poisoned: {e}")))?;
        writer.delete_term(Term::from_field_text(
            self.f_source_id,
            &source_id.to_string(),
        ));
        writer
            .commit()
            .map_err(|e| KbError::Other(format!("tantivy commit: {e}")))?;
        Ok(())
    }

    /// BM25 search across `content`, `title^2`, `headings^1.5`. Returns
    /// chunk_ids as `ScoredKey.key`.
    pub fn search(&self, query: &str, k: usize) -> Result<Vec<ScoredKey>, KbError> {
        let searcher = self.reader.searcher();
        let mut qp = QueryParser::for_index(
            &self.index,
            vec![self.f_content, self.f_title, self.f_headings],
        );
        qp.set_field_boost(self.f_title, 2.0);
        qp.set_field_boost(self.f_headings, 1.5);

        let parsed = qp
            .parse_query(query)
            .map_err(|e| KbError::Other(format!("tantivy parse: {e}")))?;
        let top = searcher
            .search(&parsed, &TopDocs::with_limit(k))
            .map_err(|e| KbError::Other(format!("tantivy search: {e}")))?;

        let mut out = Vec::with_capacity(top.len());
        for (score, addr) in top {
            let doc: TantivyDocument = searcher
                .doc(addr)
                .map_err(|e| KbError::Other(format!("tantivy doc: {e}")))?;
            if let Some(v) = doc.get_first(self.f_chunk_id) {
                if let Some(s) = Value::as_str(&v) {
                    out.push(ScoredKey {
                        key: s.to_string(),
                        score,
                    });
                }
            }
        }
        Ok(out)
    }
}

// Suppress unused-field warning for IndexRecordOption import
const _: IndexRecordOption = IndexRecordOption::Basic;
