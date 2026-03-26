use chrono::{DateTime, Utc};
use gw_core::MemoryKind;
use sqlx::PgPool;
use uuid::Uuid;

use crate::error::MemoryError;
use crate::fusion::ScoredKey;
use crate::{MemoryMeta, MemoryScope};

/// A fully hydrated memory row from Postgres, including Hindsight metadata.
#[derive(Debug, Clone)]
pub struct HydratedMemory {
    pub key: String,
    pub value: serde_json::Value,
    pub kind: MemoryKind,
    pub confidence: Option<f32>,
    pub occurred_at: Option<DateTime<Utc>>,
    pub occurred_end: Option<DateTime<Utc>>,
    pub entities: Vec<String>,
}

/// Postgres-backed full-text search store.
pub struct PgMemoryStore {
    pool: PgPool,
}

impl PgMemoryStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Access the underlying connection pool (needed by graph/temporal queries).
    pub fn pool(&self) -> &PgPool {
        &self.pool
    }

    /// Flatten a JSON value to a text string for FTS indexing.
    fn flatten_value(value: &serde_json::Value) -> String {
        match value {
            serde_json::Value::String(s) => s.clone(),
            other => other.to_string(),
        }
    }

    /// Upsert a memory (INSERT ON CONFLICT UPDATE).
    pub async fn upsert(
        &self,
        org_id: &Uuid,
        user_id: Option<&Uuid>,
        agent_id: Option<&Uuid>,
        session_id: Option<&Uuid>,
        key: &str,
        value: &serde_json::Value,
        meta: Option<&MemoryMeta>,
    ) -> Result<(), MemoryError> {
        let text_content = Self::flatten_value(value);

        // Extract metadata fields (default to Fact with no extras)
        let kind = meta.map(|m| m.kind).unwrap_or_default();
        let confidence: Option<f32> = meta.and_then(|m| m.confidence);
        let occurred_at: Option<DateTime<Utc>> = meta.and_then(|m| m.occurred_at);
        let occurred_end: Option<DateTime<Utc>> = meta.and_then(|m| m.occurred_end);
        let entities: Option<&[String]> = meta
            .map(|m| m.entities.as_slice())
            .filter(|e| !e.is_empty());

        sqlx::query(
            r#"
            INSERT INTO memories (org_id, user_id, agent_id, session_id, key, value, text_content,
                                  kind, confidence, occurred_at, occurred_end, entities)
            VALUES ($1, $2, $3, $4, $5, $6, $7,
                    $8, $9, $10, $11, $12)
            ON CONFLICT (org_id, key) DO UPDATE SET
                value = EXCLUDED.value,
                text_content = EXCLUDED.text_content,
                user_id = EXCLUDED.user_id,
                agent_id = EXCLUDED.agent_id,
                session_id = EXCLUDED.session_id,
                kind = EXCLUDED.kind,
                confidence = EXCLUDED.confidence,
                occurred_at = EXCLUDED.occurred_at,
                occurred_end = EXCLUDED.occurred_end,
                entities = EXCLUDED.entities,
                updated_at = now()
            "#,
        )
        .bind(org_id)
        .bind(user_id)
        .bind(agent_id)
        .bind(session_id)
        .bind(key)
        .bind(value)
        .bind(&text_content)
        .bind(kind)
        .bind(confidence)
        .bind(occurred_at)
        .bind(occurred_end)
        .bind(entities)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Full-text search with scope filtering.
    pub async fn fts_search(
        &self,
        org_id: &Uuid,
        query: &str,
        scope: &MemoryScope,
        limit: usize,
    ) -> Result<Vec<ScoredKey>, MemoryError> {
        // Build scope filter clause
        let (scope_clause, scope_id): (String, Option<Uuid>) = match scope {
            MemoryScope::Org => (String::new(), None),
            MemoryScope::User(uid) => (
                " AND (user_id = $3 OR user_id IS NULL)".into(),
                Some(uid.0),
            ),
            MemoryScope::Agent(aid) => (
                " AND (agent_id = $3 OR agent_id IS NULL)".into(),
                Some(aid.0),
            ),
            MemoryScope::Session(sid) => (
                " AND (session_id = $3 OR session_id IS NULL)".into(),
                Some(sid.0),
            ),
        };

        let sql = format!(
            r#"
            SELECT key, ts_rank(tsv, plainto_tsquery('english', $1)) AS rank
            FROM memories
            WHERE org_id = $2
              AND tsv @@ plainto_tsquery('english', $1)
              {scope_clause}
            ORDER BY rank DESC
            LIMIT {limit}
            "#
        );

        let rows: Vec<(String, f32)> = if let Some(sid) = scope_id {
            sqlx::query_as(&sql)
                .bind(query)
                .bind(org_id)
                .bind(sid)
                .fetch_all(&self.pool)
                .await?
        } else {
            sqlx::query_as(&sql)
                .bind(query)
                .bind(org_id)
                .fetch_all(&self.pool)
                .await?
        };

        Ok(rows
            .into_iter()
            .map(|(key, score)| ScoredKey { key, score })
            .collect())
    }

    /// Look up memory values by keys (lightweight — value only).
    pub async fn get_by_keys(
        &self,
        org_id: &Uuid,
        keys: &[String],
    ) -> Result<Vec<(String, serde_json::Value)>, MemoryError> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        let rows: Vec<(String, serde_json::Value)> = sqlx::query_as(
            "SELECT key, value FROM memories WHERE org_id = $1 AND key = ANY($2)",
        )
        .bind(org_id)
        .bind(keys)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows)
    }

    /// Look up memories by keys with full Hindsight metadata.
    ///
    /// Returns all columns needed to populate `MemoryRecord` including kind,
    /// confidence, temporal fields, and entities.
    pub async fn get_by_keys_hydrated(
        &self,
        org_id: &Uuid,
        keys: &[String],
    ) -> Result<Vec<HydratedMemory>, MemoryError> {
        if keys.is_empty() {
            return Ok(vec![]);
        }

        let rows: Vec<(
            String,                   // key
            serde_json::Value,        // value
            String,                   // kind (as text)
            Option<f32>,              // confidence
            Option<DateTime<Utc>>,    // occurred_at
            Option<DateTime<Utc>>,    // occurred_end
            Option<Vec<String>>,      // entities
        )> = sqlx::query_as(
            r#"SELECT key, value, kind::text, confidence, occurred_at, occurred_end, entities
               FROM memories WHERE org_id = $1 AND key = ANY($2)"#,
        )
        .bind(org_id)
        .bind(keys)
        .fetch_all(&self.pool)
        .await?;

        Ok(rows
            .into_iter()
            .map(|(key, value, kind_str, confidence, occurred_at, occurred_end, entities)| {
                let kind = match kind_str.as_str() {
                    "experience" => MemoryKind::Experience,
                    "opinion" => MemoryKind::Opinion,
                    "observation" => MemoryKind::Observation,
                    _ => MemoryKind::Fact,
                };
                HydratedMemory {
                    key,
                    value,
                    kind,
                    confidence,
                    occurred_at,
                    occurred_end,
                    entities: entities.unwrap_or_default(),
                }
            })
            .collect())
    }

    /// Delete a memory by key.
    pub async fn delete(&self, org_id: &Uuid, key: &str) -> Result<(), MemoryError> {
        sqlx::query("DELETE FROM memories WHERE org_id = $1 AND key = $2")
            .bind(org_id)
            .bind(key)
            .execute(&self.pool)
            .await?;

        Ok(())
    }

}
