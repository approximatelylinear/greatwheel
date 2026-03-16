use std::sync::Arc;

use arrow_array::{
    Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::query::{ExecutableQuery, QueryBase};
use lancedb::{connect, Connection};
use tracing;

use crate::error::MemoryError;
use crate::fusion::ScoredKey;

/// LanceDB vector store — one table per org.
pub struct LanceStore {
    conn: Connection,
    dim: i32,
}

impl LanceStore {
    pub async fn new(path: &str, dim: i32) -> Result<Self, MemoryError> {
        let conn = connect(path).execute().await?;
        Ok(Self { conn, dim })
    }

    fn table_name(org_id: &uuid::Uuid) -> String {
        format!("memory_{}", org_id.simple())
    }

    fn schema(&self) -> Arc<Schema> {
        Arc::new(Schema::new(vec![
            Field::new("key", DataType::Utf8, false),
            Field::new("text", DataType::Utf8, false),
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

    fn build_batch(
        &self,
        keys: &[&str],
        texts: &[&str],
        vectors: &[Vec<f32>],
    ) -> Result<RecordBatch, MemoryError> {
        let key_array = StringArray::from(keys.to_vec());
        let text_array = StringArray::from(texts.to_vec());

        let flat_values: Vec<f32> = vectors.iter().flat_map(|v| v.iter().copied()).collect();
        let values_array = Float32Array::from(flat_values);
        let field = Arc::new(Field::new("item", DataType::Float32, true));
        let vector_array = FixedSizeListArray::try_new(
            field,
            self.dim,
            Arc::new(values_array),
            None,
        )
        .map_err(|e| MemoryError::Embedding(format!("Arrow FixedSizeList error: {e}")))?;

        RecordBatch::try_new(
            self.schema(),
            vec![
                Arc::new(key_array),
                Arc::new(text_array),
                Arc::new(vector_array),
            ],
        )
        .map_err(|e| MemoryError::Embedding(format!("Arrow RecordBatch error: {e}")))
    }

    /// Open or create the per-org table.
    async fn open_or_create_table(
        &self,
        org_id: &uuid::Uuid,
    ) -> Result<lancedb::Table, MemoryError> {
        let name = Self::table_name(org_id);
        match self.conn.open_table(&name).execute().await {
            Ok(table) => Ok(table),
            Err(_) => {
                tracing::info!(table = %name, "Creating new LanceDB table");
                let schema = self.schema();
                let empty_batch = RecordBatch::new_empty(schema.clone());
                let batches =
                    RecordBatchIterator::new(vec![Ok(empty_batch)], schema);
                let table = self.conn.create_table(&name, Box::new(batches)).execute().await?;
                Ok(table)
            }
        }
    }

    /// Upsert a memory vector (delete-then-insert since LanceDB has no native upsert).
    pub async fn upsert(
        &self,
        org_id: &uuid::Uuid,
        key: &str,
        text: &str,
        vector: Vec<f32>,
    ) -> Result<(), MemoryError> {
        let table = self.open_or_create_table(org_id).await?;

        // Delete existing entry with this key
        let filter = format!("key = '{}'", key.replace('\'', "''"));
        let _ = table.delete(&filter).await;

        // Insert new
        let batch = self.build_batch(&[key], &[text], &[vector])?;
        let schema = self.schema();
        let batches = RecordBatchIterator::new(vec![Ok(batch)], schema);
        table.add(Box::new(batches)).execute().await?;

        Ok(())
    }

    /// Vector similarity search, returns scored keys.
    pub async fn search(
        &self,
        org_id: &uuid::Uuid,
        query_vector: Vec<f32>,
        k: usize,
    ) -> Result<Vec<ScoredKey>, MemoryError> {
        let table = match self
            .conn
            .open_table(&Self::table_name(org_id))
            .execute()
            .await
        {
            Ok(t) => t,
            Err(_) => return Ok(vec![]),
        };

        let mut stream = table
            .vector_search(query_vector)
            .map_err(|e| MemoryError::Embedding(format!("Vector search error: {e}")))?
            .limit(k)
            .execute()
            .await?;

        let mut scored = Vec::new();
        while let Some(batch) = stream.try_next().await.map_err(|e| {
            MemoryError::Embedding(format!("Error reading vector search results: {e}"))
        })? {
            if let (Some(key_col), Some(dist_col)) = (
                batch.column_by_name("key"),
                batch.column_by_name("_distance"),
            ) {
                let keys = key_col.as_any().downcast_ref::<StringArray>();
                let dists = dist_col.as_any().downcast_ref::<Float32Array>();

                if let (Some(keys), Some(dists)) = (keys, dists) {
                    for i in 0..keys.len() {
                        let key = keys.value(i).to_string();
                        let distance = dists.value(i);
                        let score = 1.0 / (1.0 + distance);
                        scored.push(ScoredKey { key, score });
                    }
                }
            }
        }

        Ok(scored)
    }

    /// Delete a memory vector by key.
    pub async fn delete(&self, org_id: &uuid::Uuid, key: &str) -> Result<(), MemoryError> {
        let table = match self
            .conn
            .open_table(&Self::table_name(org_id))
            .execute()
            .await
        {
            Ok(t) => t,
            Err(_) => return Ok(()),
        };

        let filter = format!("key = '{}'", key.replace('\'', "''"));
        table.delete(&filter).await?;

        Ok(())
    }
}
