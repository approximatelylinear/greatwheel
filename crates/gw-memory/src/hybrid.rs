use std::collections::HashMap;
use std::sync::Arc;

use gw_core::{CallContext, EventData, EventPayload, EventResult, LifecycleEvent};

use crate::error::MemoryError;
use crate::fusion;
use crate::lance::LanceStore;
use crate::postgres::PgMemoryStore;
use crate::tantivy_store::TantivyStore;
use crate::{MemoryMeta, MemoryRecord, RecallOpts, SearchMode};
use gw_llm::OllamaClient;

/// Function type for dispatching lifecycle events.
///
/// Accepts a mutable event payload and returns an `EventResult`.
/// Provided by `gw-engine`'s `EventDispatcher`; `gw-memory` does not
/// depend on `gw-engine` directly to avoid circular dependencies.
///
/// TODO(phase-3): This is sync but called inside async store()/recall().
/// Phase 3 plugins (hindsight-retain) need async LLM calls for entity
/// extraction. Options: (a) make this an async trait, (b) have handlers
/// spawn async work via tokio::spawn and a runtime handle in SharedState,
/// or (c) add a parallel AsyncDispatchFn. See design-hindsight-memory.md §6.2 Q7.
pub type DispatchFn = dyn Fn(&mut EventPayload) -> EventResult + Send + Sync;

/// Hybrid memory store backed by LanceDB (vector) + tantivy (BM25) + Postgres (persistence).
pub struct HybridStore {
    pg: PgMemoryStore,
    lance: LanceStore,
    tantivy: TantivyStore,
    llm: Arc<OllamaClient>,
    /// Optional event dispatcher for plugin lifecycle hooks.
    /// When set, `BeforeMemoryStore` and `AfterMemoryRecall` events are
    /// dispatched, allowing plugins to enrich/augment memory operations.
    dispatcher: Option<Arc<DispatchFn>>,
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
            dispatcher: None,
        }
    }

    /// Attach an event dispatcher for plugin lifecycle hooks.
    pub fn with_dispatcher(mut self, dispatch: Arc<DispatchFn>) -> Self {
        self.dispatcher = Some(dispatch);
        self
    }

    /// Store a memory: embed, then upsert to Postgres + LanceDB + tantivy.
    #[tracing::instrument(name = "memory.store", skip(self, ctx, value, meta), fields(gw.memory.key = key))]
    pub async fn store(
        &self,
        ctx: &CallContext,
        key: &str,
        value: serde_json::Value,
        meta: Option<MemoryMeta>,
    ) -> Result<(), MemoryError> {
        // Dispatch BeforeMemoryStore event — plugins can enrich the value/meta.
        let (value, meta) = if let Some(ref dispatch) = self.dispatcher {
            let meta_json = meta.as_ref().and_then(|m| serde_json::to_value(m).ok());
            let mut payload = EventPayload {
                event: LifecycleEvent::BeforeMemoryStore,
                data: EventData::Memory {
                    key: key.to_string(),
                    value: Some(value.clone()),
                    meta: meta_json,
                },
            };
            dispatch(&mut payload);
            // Extract potentially modified value and meta from the payload
            match payload.data {
                EventData::Memory { value: Some(v), meta: Some(m), .. } => {
                    let enriched_meta = serde_json::from_value::<MemoryMeta>(m).ok();
                    (v, enriched_meta.or(meta))
                }
                EventData::Memory { value: Some(v), meta: None, .. } => (v, meta),
                _ => (value, meta),
            }
        } else {
            (value, meta)
        };

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
            meta.as_ref(),
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

        let results = match opts.mode {
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

                keys.iter()
                    .filter_map(|k| {
                        value_map.get(k).map(|v| MemoryRecord {
                            key: k.clone(),
                            value: v.clone(),
                            score: *score_map.get(k.as_str()).unwrap_or(&0.0),
                            ..Default::default()
                        })
                    })
                    .collect()
            }

            SearchMode::FullText => {
                // BM25 search via tantivy
                let scored = self.tantivy.search(&org_id, query, &opts.scope, top_k)?;
                let keys: Vec<String> = scored.iter().map(|s| s.key.clone()).collect();
                let values = self.pg.get_by_keys(&org_id, &keys).await?;

                let value_map: HashMap<String, serde_json::Value> = values.into_iter().collect();
                let score_map: HashMap<&str, f32> =
                    scored.iter().map(|s| (s.key.as_str(), s.score)).collect();

                keys.iter()
                    .filter_map(|k| {
                        value_map.get(k).map(|v| MemoryRecord {
                            key: k.clone(),
                            value: v.clone(),
                            score: *score_map.get(k.as_str()).unwrap_or(&0.0),
                            ..Default::default()
                        })
                    })
                    .collect()
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

                fused
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(k, score)| {
                        value_map.get(&k).map(|v| MemoryRecord {
                            key: k,
                            value: v.clone(),
                            score,
                            ..Default::default()
                        })
                    })
                    .collect()
            }

            SearchMode::Full { graph_hops, graph_decay, recency_sigma_days } => {
                // Four-channel retrieval: vector + BM25 + graph + temporal

                // Channel 1 & 2: vector + BM25 (same as Hybrid)
                let query_text = query.to_string();
                let query_vec = self
                    .llm
                    .embed(&[query_text])
                    .await
                    .map_err(|e| MemoryError::Embedding(e.to_string()))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| MemoryError::Embedding("No embedding returned".into()))?;

                let vector_fut = self.lance.search(&org_id, query_vec, top_k);
                let bm25_query = query.to_string();
                let bm25_scope = opts.scope.clone();
                let bm25_results = self.tantivy.search(&org_id, &bm25_query, &bm25_scope, top_k)?;
                let vector_results = vector_fut.await?;

                // Channel 3: graph traversal (spreading activation from vector+BM25 seeds)
                let seed_fused = fusion::reciprocal_rank_fusion(
                    &[vector_results.clone(), bm25_results.clone()],
                    60,
                );
                let seeds: Vec<fusion::ScoredKey> = seed_fused
                    .iter()
                    .take(top_k)
                    .map(|(k, s)| fusion::ScoredKey { key: k.clone(), score: *s })
                    .collect();

                // Parse temporal range for temporal-constrained graph traversal
                let now = chrono::Utc::now();
                let temporal_range = crate::temporal::parse_temporal(query, now);

                let temporal_filter = if let Some(ref range) = temporal_range {
                    match crate::graph::fetch_temporal_set(
                        self.pg.pool(), &org_id, range.start, range.end,
                    ).await {
                        Ok(set) => Some(set),
                        Err(e) => {
                            tracing::warn!(error = %e, "fetch_temporal_set failed, falling back to unfiltered graph traversal");
                            None
                        }
                    }
                } else {
                    None
                };

                let graph_results = match crate::graph::spreading_activation(
                    self.pg.pool(),
                    &org_id,
                    &seeds,
                    graph_hops,
                    graph_decay,
                    temporal_filter.as_ref(),
                )
                .await {
                    Ok(r) => r,
                    Err(e) => {
                        tracing::warn!(error = %e, "graph spreading_activation failed, skipping graph channel");
                        Vec::new()
                    }
                };

                // Channel 4: temporal scoring
                let temporal_results = if let Some(ref range) = temporal_range {
                    match crate::graph::temporal_score_memories(
                        self.pg.pool(), &org_id, range.start, range.end, top_k,
                    ).await {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!(error = %e, "temporal_score_memories failed, skipping temporal channel");
                            Vec::new()
                        }
                    }
                } else {
                    // Recency decay fallback
                    let rows: Vec<(String, chrono::DateTime<chrono::Utc>)> = match sqlx::query_as(
                        "SELECT key, occurred_at FROM memories WHERE org_id = $1 AND occurred_at IS NOT NULL ORDER BY occurred_at DESC LIMIT $2",
                    )
                    .bind(&org_id)
                    .bind(top_k as i64)
                    .fetch_all(self.pg.pool())
                    .await {
                        Ok(r) => r,
                        Err(e) => {
                            tracing::warn!(error = %e, "recency fallback query failed, skipping temporal channel");
                            Vec::new()
                        }
                    };

                    rows.into_iter()
                        .map(|(key, occ)| fusion::ScoredKey {
                            key,
                            score: crate::temporal::recency_score(occ, now, recency_sigma_days),
                        })
                        .collect()
                };

                // Fuse all four channels via RRF
                let fused = fusion::reciprocal_rank_fusion(
                    &[vector_results, bm25_results, graph_results, temporal_results],
                    60,
                );

                let top_keys: Vec<String> = fused
                    .iter()
                    .take(top_k)
                    .map(|(k, _)| k.clone())
                    .collect();

                let values = self.pg.get_by_keys(&org_id, &top_keys).await?;
                let value_map: HashMap<String, serde_json::Value> = values.into_iter().collect();

                fused
                    .into_iter()
                    .take(top_k)
                    .filter_map(|(k, score)| {
                        value_map.get(&k).map(|v| MemoryRecord {
                            key: k,
                            value: v.clone(),
                            score,
                            ..Default::default()
                        })
                    })
                    .collect()
            }
        };

        // Dispatch AfterMemoryRecall — plugins can augment/re-score/filter results.
        // Currently a notification; plugins that need to modify results will use
        // the Custom EventData variant with the result set once Phase 3 plugins land.
        if let Some(ref dispatch) = self.dispatcher {
            let mut payload = EventPayload {
                event: LifecycleEvent::AfterMemoryRecall,
                data: EventData::Memory {
                    key: query.to_string(),
                    value: None,
                    meta: None,
                },
            };
            dispatch(&mut payload);
        }

        Ok(results)
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
