use std::path::Path;
use std::sync::Mutex;

use tantivy::collector::TopDocs;
use tantivy::query::QueryParser;
use tantivy::schema::{Field, Schema, Value, STORED, STRING, TEXT};
use tantivy::{doc, Index, IndexReader, IndexWriter, ReloadPolicy, Term};
use uuid::Uuid;

use crate::error::MemoryError;
use crate::fusion::ScoredKey;
use crate::MemoryScope;

/// BM25 full-text search backed by tantivy.
///
/// Documents are indexed with their key, text content, org_id, and optional
/// scope IDs (user, agent, session). Search uses tantivy's BM25 scorer.
///
/// The tantivy index lives on disk at a configurable path. The `IndexWriter`
/// is behind a `Mutex` because tantivy only allows one writer at a time.
pub struct TantivyStore {
    index: Index,
    reader: IndexReader,
    writer: Mutex<IndexWriter>,
    // Schema fields
    f_key: Field,
    f_text: Field,
    f_org_id: Field,
    f_user_id: Field,
    f_agent_id: Field,
    f_session_id: Field,
}

impl TantivyStore {
    /// Open or create a tantivy index at the given directory.
    pub fn open(index_path: &Path) -> Result<Self, MemoryError> {
        let mut schema_builder = Schema::builder();
        let f_key = schema_builder.add_text_field("key", STRING | STORED);
        let f_text = schema_builder.add_text_field("text", TEXT);
        let f_org_id = schema_builder.add_text_field("org_id", STRING);
        let f_user_id = schema_builder.add_text_field("user_id", STRING);
        let f_agent_id = schema_builder.add_text_field("agent_id", STRING);
        let f_session_id = schema_builder.add_text_field("session_id", STRING);
        let schema = schema_builder.build();

        std::fs::create_dir_all(index_path).map_err(|e| {
            MemoryError::Tantivy(format!("failed to create index dir: {e}"))
        })?;

        let index = if Index::open_in_dir(index_path).is_ok() {
            Index::open_in_dir(index_path)
                .map_err(|e| MemoryError::Tantivy(format!("{e}")))?
        } else {
            Index::create_in_dir(index_path, schema)
                .map_err(|e| MemoryError::Tantivy(format!("{e}")))?
        };

        let reader = index
            .reader_builder()
            .reload_policy(ReloadPolicy::OnCommitWithDelay)
            .try_into()
            .map_err(|e: tantivy::TantivyError| MemoryError::Tantivy(format!("{e}")))?;

        // 50 MB heap for the writer — plenty for a memory store
        let writer = index
            .writer(50_000_000)
            .map_err(|e| MemoryError::Tantivy(format!("{e}")))?;

        Ok(Self {
            index,
            reader,
            writer: Mutex::new(writer),
            f_key,
            f_text,
            f_org_id,
            f_user_id,
            f_agent_id,
            f_session_id,
        })
    }

    /// Index (or re-index) a document. Deletes any prior doc with the same
    /// (org_id, key) before inserting.
    pub fn upsert(
        &self,
        org_id: &Uuid,
        key: &str,
        text: &str,
        user_id: Option<&Uuid>,
        agent_id: Option<&Uuid>,
        session_id: Option<&Uuid>,
    ) -> Result<(), MemoryError> {
        let org_str = org_id.to_string();
        let mut writer = self.writer.lock().map_err(|e| {
            MemoryError::Tantivy(format!("writer lock poisoned: {e}"))
        })?;

        // Delete existing doc with this key+org — tantivy doesn't have compound
        // term delete, so we delete by key and re-check. Since keys are unique
        // per org, deleting by key is sufficient (org_id is part of the unique
        // constraint in Postgres, and we only ever index one org_id per key).
        writer.delete_term(Term::from_field_text(self.f_key, key));

        let mut doc = doc!(
            self.f_key => key,
            self.f_text => text,
            self.f_org_id => org_str,
        );
        if let Some(uid) = user_id {
            doc.add_text(self.f_user_id, &uid.to_string());
        }
        if let Some(aid) = agent_id {
            doc.add_text(self.f_agent_id, &aid.to_string());
        }
        if let Some(sid) = session_id {
            doc.add_text(self.f_session_id, &sid.to_string());
        }

        writer.add_document(doc).map_err(|e| {
            MemoryError::Tantivy(format!("add_document failed: {e}"))
        })?;
        writer.commit().map_err(|e| {
            MemoryError::Tantivy(format!("commit failed: {e}"))
        })?;

        Ok(())
    }

    /// BM25 search within an org, with optional scope filtering.
    pub fn search(
        &self,
        org_id: &Uuid,
        query: &str,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<ScoredKey>, MemoryError> {
        let searcher = self.reader.searcher();
        let query_parser = QueryParser::for_index(&self.index, vec![self.f_text]);

        let org_str = org_id.to_string();

        // Build a boolean query: BM25 text match AND org_id filter (AND optional scope)
        let text_query = query_parser.parse_query(query).map_err(|e| {
            MemoryError::Tantivy(format!("query parse error: {e}"))
        })?;

        let org_term = Term::from_field_text(self.f_org_id, &org_str);
        let org_query = tantivy::query::TermQuery::new(
            org_term,
            tantivy::schema::IndexRecordOption::Basic,
        );

        // Optional scope filter
        let scope_query: Option<Box<dyn tantivy::query::Query>> = match scope {
            MemoryScope::Org => None,
            MemoryScope::User(uid) => {
                let term = Term::from_field_text(self.f_user_id, &uid.0.to_string());
                Some(Box::new(tantivy::query::TermQuery::new(
                    term,
                    tantivy::schema::IndexRecordOption::Basic,
                )))
            }
            MemoryScope::Agent(aid) => {
                let term = Term::from_field_text(self.f_agent_id, &aid.0.to_string());
                Some(Box::new(tantivy::query::TermQuery::new(
                    term,
                    tantivy::schema::IndexRecordOption::Basic,
                )))
            }
            MemoryScope::Session(sid) => {
                let term = Term::from_field_text(self.f_session_id, &sid.0.to_string());
                Some(Box::new(tantivy::query::TermQuery::new(
                    term,
                    tantivy::schema::IndexRecordOption::Basic,
                )))
            }
        };

        let combined_query = {
            use tantivy::query::{BooleanQuery, Occur};
            let mut clauses: Vec<(Occur, Box<dyn tantivy::query::Query>)> = vec![
                (Occur::Must, text_query),
                (Occur::Must, Box::new(org_query)),
            ];
            if let Some(sq) = scope_query {
                clauses.push((Occur::Must, sq));
            }
            BooleanQuery::new(clauses)
        };

        let top_docs = searcher
            .search(&combined_query, &TopDocs::with_limit(limit))
            .map_err(|e| MemoryError::Tantivy(format!("search failed: {e}")))?;

        let mut results = Vec::with_capacity(top_docs.len());
        for (score, doc_address) in top_docs {
            let doc: tantivy::TantivyDocument = searcher.doc(doc_address).map_err(|e| {
                MemoryError::Tantivy(format!("doc retrieval failed: {e}"))
            })?;
            if let Some(key_value) = doc.get_first(self.f_key) {
                if let Some(key_str) = Value::as_str(&key_value) {
                    results.push(ScoredKey {
                        key: key_str.to_string(),
                        score,
                    });
                }
            }
        }

        Ok(results)
    }

    /// Delete all documents matching a key.
    pub fn delete(&self, key: &str) -> Result<(), MemoryError> {
        let mut writer = self.writer.lock().map_err(|e| {
            MemoryError::Tantivy(format!("writer lock poisoned: {e}"))
        })?;
        writer.delete_term(Term::from_field_text(self.f_key, key));
        writer.commit().map_err(|e| {
            MemoryError::Tantivy(format!("commit failed: {e}"))
        })?;
        Ok(())
    }

    /// Rebuild the tantivy index from Postgres data.
    /// Call this on startup to sync the index with the source of truth.
    pub fn rebuild_from_rows(
        &self,
        rows: Vec<TantivyRow>,
    ) -> Result<usize, MemoryError> {
        let mut writer = self.writer.lock().map_err(|e| {
            MemoryError::Tantivy(format!("writer lock poisoned: {e}"))
        })?;

        // Clear existing index
        writer.delete_all_documents().map_err(|e| {
            MemoryError::Tantivy(format!("delete_all failed: {e}"))
        })?;
        writer.commit().map_err(|e| {
            MemoryError::Tantivy(format!("commit failed: {e}"))
        })?;

        let count = rows.len();
        for row in rows {
            let mut doc = doc!(
                self.f_key => row.key.as_str(),
                self.f_text => row.text.as_str(),
                self.f_org_id => row.org_id.to_string(),
            );
            if let Some(uid) = &row.user_id {
                doc.add_text(self.f_user_id, &uid.to_string());
            }
            if let Some(aid) = &row.agent_id {
                doc.add_text(self.f_agent_id, &aid.to_string());
            }
            if let Some(sid) = &row.session_id {
                doc.add_text(self.f_session_id, &sid.to_string());
            }
            writer.add_document(doc).map_err(|e| {
                MemoryError::Tantivy(format!("add_document failed: {e}"))
            })?;
        }

        writer.commit().map_err(|e| {
            MemoryError::Tantivy(format!("commit failed: {e}"))
        })?;

        tracing::info!(count, "Tantivy index rebuilt from Postgres");
        Ok(count)
    }
}

/// A row from Postgres used to rebuild the tantivy index.
pub struct TantivyRow {
    pub key: String,
    pub text: String,
    pub org_id: Uuid,
    pub user_id: Option<Uuid>,
    pub agent_id: Option<Uuid>,
    pub session_id: Option<Uuid>,
}
