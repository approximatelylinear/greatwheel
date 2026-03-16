use std::collections::HashMap;
use std::sync::Arc;

use gw_core::CallContext;

use crate::error::MemoryError;
use crate::fusion;
use crate::lance::LanceStore;
use crate::postgres::PgMemoryStore;
use crate::tantivy_store::TantivyStore;
use crate::{MemoryRecord, RecallOpts, SearchMode};
use gw_llm::OllamaClient;

/// Hybrid memory store backed by LanceDB (vector) + tantivy (BM25) + Postgres (persistence).
pub struct HybridStore {
    pg: PgMemoryStore,
    lance: LanceStore,
    tantivy: TantivyStore,
    llm: Arc<OllamaClient>,
}

impl HybridStore {
    pub fn new(
        pg: PgMemoryStore,
        lance: LanceStore,
        tantivy: TantivyStore,
        llm: Arc<OllamaClient>,
    ) -> Self {
        Self {
            pg,
            lance,
            tantivy,
            llm,
        }
    }

    /// Store a memory: embed, then upsert to Postgres + LanceDB + tantivy.
    #[tracing::instrument(name = "memory.store", skip(self, ctx, value), fields(gw.memory.key = key))]
    pub async fn store(
        &self,
        ctx: &CallContext,
        key: &str,
        value: serde_json::Value,
    ) -> Result<(), MemoryError> {
        let text = match &value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        };

        // Generate embedding
        let embeddings = self
            .llm
            .embed(&[text.clone()])
            .await
            .map_err(|e| MemoryError::Embedding(e.to_string()))?;

        let vector = embeddings
            .into_iter()
            .next()
            .ok_or_else(|| MemoryError::Embedding("No embedding returned".into()))?;

        let org_id = ctx.org_id.0;

        // Upsert to Postgres and LanceDB concurrently
        let pg_fut = self.pg.upsert(
            &org_id,
            Some(&ctx.user_id.0),
            Some(&ctx.agent_id.0),
            Some(&ctx.session_id.0),
            key,
            &value,
        );
        let lance_fut = self.lance.upsert(&org_id, key, &text, vector);

        tokio::try_join!(pg_fut, lance_fut)?;

        // Index into tantivy (sync, but fast)
        self.tantivy.upsert(
            &org_id,
            key,
            &text,
            Some(&ctx.user_id.0),
            Some(&ctx.agent_id.0),
            Some(&ctx.session_id.0),
        )?;

        tracing::debug!(key, "Memory stored in Postgres, LanceDB, and tantivy");
        Ok(())
    }

    /// Recall memories using the specified search mode.
    #[tracing::instrument(
        name = "memory.recall",
        skip(self, ctx),
        fields(
            gw.memory.search_mode = tracing::field::Empty,
            gw.memory.top_k = tracing::field::Empty,
            gw.memory.results_count = tracing::field::Empty,
        )
    )]
    pub async fn recall(
        &self,
        ctx: &CallContext,
        query: &str,
        opts: RecallOpts,
    ) -> Result<Vec<MemoryRecord>, MemoryError> {
        let org_id = ctx.org_id.0;
        let top_k = opts.top_k;

        let span = tracing::Span::current();
        span.record("gw.memory.search_mode", tracing::field::debug(&opts.mode));
        span.record("gw.memory.top_k", top_k);

        match opts.mode {
            SearchMode::Vector => {
                let query_vec = self
                    .llm
                    .embed(&[query.to_string()])
                    .await
                    .map_err(|e| MemoryError::Embedding(e.to_string()))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| MemoryError::Embedding("No embedding returned".into()))?;

                let scored = self.lance.search(&org_id, query_vec, top_k).await?;
                let keys: Vec<String> = scored.iter().map(|s| s.key.clone()).collect();
                let values = self.pg.get_by_keys(&org_id, &keys).await?;

                let value_map: HashMap<String, serde_json::Value> = values.into_iter().collect();
                let score_map: HashMap<&str, f32> =
                    scored.iter().map(|s| (s.key.as_str(), s.score)).collect();

                Ok(keys
                    .iter()
                    .filter_map(|k| {
                        value_map.get(k).map(|v| MemoryRecord {
                            key: k.clone(),
                            value: v.clone(),
                            score: *score_map.get(k.as_str()).unwrap_or(&0.0),
                        })
                    })
                    .collect())
            }

            SearchMode::FullText => {
                // BM25 search via tantivy
                let scored = self.tantivy.search(&org_id, query, &opts.scope, top_k)?;
                let keys: Vec<String> = scored.iter().map(|s| s.key.clone()).collect();
                let values = self.pg.get_by_keys(&org_id, &keys).await?;

                let value_map: HashMap<String, serde_json::Value> = values.into_iter().collect();
                let score_map: HashMap<&str, f32> =
                    scored.iter().map(|s| (s.key.as_str(), s.score)).collect();

                Ok(keys
                    .iter()
                    .filter_map(|k| {
                        value_map.get(k).map(|v| MemoryRecord {
                            key: k.clone(),
                            value: v.clone(),
                            score: *score_map.get(k.as_str()).unwrap_or(&0.0),
                        })
                    })
                    .collect())
            }

            SearchMode::Hybrid { alpha: _ } => {
                // Embed the query first
                let query_text = query.to_string();
                let query_vec = self
                    .llm
                    .embed(&[query_text])
                    .await
                    .map_err(|e| MemoryError::Embedding(e.to_string()))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| MemoryError::Embedding("No embedding returned".into()))?;

                // Vector search (async) + BM25 search (sync, run on blocking thread)
                let vector_fut = self.lance.search(&org_id, query_vec, top_k);

                let bm25_query = query.to_string();
                let bm25_scope = opts.scope.clone();
                let tantivy = &self.tantivy;
                let bm25_results =
                    tantivy.search(&org_id, &bm25_query, &bm25_scope, top_k)?;

                let vector_results = vector_fut.await?;

                // RRF fusion
                let fused = fusion::reciprocal_rank_fusion(
                    &[vector_results, bm25_results],
                    60,
                );

                let top_keys: Vec<String> = fused
                    .iter()
                    .take(top_k)
                    .map(|(k, _)| k.clone())
                    .collect();

                let values = self.pg.get_by_keys(&org_id, &top_keys).await?;
                let value_map: HashMap<String, serde_json::Value> = values.into_iter().collect();

                Ok(fused
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(k, score)| {
                        value_map.get(&k).map(|v| MemoryRecord {
                            key: k,
                            value: v.clone(),
                            score,
                        })
                    })
                    .collect())
            }
        }
    }

    /// Forget a memory: delete from all three stores.
    #[tracing::instrument(name = "memory.forget", skip(self, ctx), fields(gw.memory.key = key))]
    pub async fn forget(&self, ctx: &CallContext, key: &str) -> Result<(), MemoryError> {
        let org_id = ctx.org_id.0;

        let pg_fut = self.pg.delete(&org_id, key);
        let lance_fut = self.lance.delete(&org_id, key);

        tokio::try_join!(pg_fut, lance_fut)?;

        // Tantivy delete (sync)
        self.tantivy.delete(key)?;

        tracing::debug!(key, "Memory forgotten from all stores");
        Ok(())
    }
}
