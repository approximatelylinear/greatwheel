use chrono::{DateTime, Utc};
use gw_core::{EntryId, EntryType, SessionEntry, SessionId};
use sqlx::PgPool;
use tracing::warn;
use uuid::Uuid;

/// Row type returned when loading session entries from Postgres.
type EntryRow = (
    Uuid,
    Uuid,
    Option<Uuid>,
    String,
    serde_json::Value,
    DateTime<Utc>,
);

/// Postgres persistence layer for session tree entries.
///
/// Uses runtime queries (not compile-time checked) following the
/// project convention from gw-memory.
#[derive(Clone)]
pub struct PgSessionStore {
    pool: PgPool,
}

impl PgSessionStore {
    pub fn new(pool: PgPool) -> Self {
        Self { pool }
    }

    /// Insert a single session entry.
    pub async fn insert_entry(&self, entry: &SessionEntry) -> Result<(), sqlx::Error> {
        let entry_type_tag = entry_type_tag(&entry.entry_type);
        let content = serde_json::to_value(&entry.entry_type).unwrap_or_default();

        sqlx::query(
            r#"
            INSERT INTO session_entries (id, session_id, parent_id, entry_type, content, created_at)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO NOTHING
            "#,
        )
        .bind(entry.id.0)
        .bind(entry.session_id.0)
        .bind(entry.parent_id.map(|e| e.0))
        .bind(entry_type_tag)
        .bind(content)
        .bind(entry.created_at)
        .execute(&self.pool)
        .await?;

        Ok(())
    }

    /// Insert multiple entries in a single transaction.
    pub async fn insert_entries(&self, entries: &[SessionEntry]) -> Result<(), sqlx::Error> {
        if entries.is_empty() {
            return Ok(());
        }

        let mut tx = self.pool.begin().await?;

        for entry in entries {
            let entry_type_tag = entry_type_tag(&entry.entry_type);
            let content = serde_json::to_value(&entry.entry_type).unwrap_or_default();

            sqlx::query(
                r#"
                INSERT INTO session_entries (id, session_id, parent_id, entry_type, content, created_at)
                VALUES ($1, $2, $3, $4, $5, $6)
                ON CONFLICT (id) DO NOTHING
                "#,
            )
            .bind(entry.id.0)
            .bind(entry.session_id.0)
            .bind(entry.parent_id.map(|e| e.0))
            .bind(entry_type_tag)
            .bind(content)
            .bind(entry.created_at)
            .execute(&mut *tx)
            .await?;
        }

        tx.commit().await?;
        Ok(())
    }

    /// Update the active leaf for a session.
    pub async fn update_active_leaf(
        &self,
        session_id: SessionId,
        leaf_id: Option<EntryId>,
    ) -> Result<(), sqlx::Error> {
        sqlx::query("UPDATE sessions SET active_leaf_id = $1 WHERE id = $2")
            .bind(leaf_id.map(|e| e.0))
            .bind(session_id.0)
            .execute(&self.pool)
            .await?;
        Ok(())
    }

    /// Load all entries for a session, ordered by created_at.
    pub async fn load_entries(
        &self,
        session_id: SessionId,
    ) -> Result<Vec<SessionEntry>, sqlx::Error> {
        let rows: Vec<EntryRow> = sqlx::query_as(
            r#"
                SELECT id, session_id, parent_id, entry_type, content, created_at
                FROM session_entries
                WHERE session_id = $1
                ORDER BY created_at ASC
                "#,
        )
        .bind(session_id.0)
        .fetch_all(&self.pool)
        .await?;

        let mut entries = Vec::with_capacity(rows.len());
        for (id, sid, parent_id, _entry_type_tag, content, created_at) in rows {
            // Deserialize entry_type from the JSONB content column.
            match serde_json::from_value::<EntryType>(content) {
                Ok(entry_type) => {
                    entries.push(SessionEntry {
                        id: EntryId(id),
                        session_id: SessionId(sid),
                        parent_id: parent_id.map(EntryId),
                        entry_type,
                        created_at,
                    });
                }
                Err(e) => {
                    warn!(entry_id = %id, error = %e, "failed to deserialize session entry, skipping");
                }
            }
        }

        Ok(entries)
    }

    /// Load the active leaf ID for a session.
    pub async fn load_active_leaf(
        &self,
        session_id: SessionId,
    ) -> Result<Option<EntryId>, sqlx::Error> {
        let row: Option<(Option<Uuid>,)> =
            sqlx::query_as("SELECT active_leaf_id FROM sessions WHERE id = $1")
                .bind(session_id.0)
                .fetch_optional(&self.pool)
                .await?;

        Ok(row.and_then(|(leaf,)| leaf.map(EntryId)))
    }

    /// Delete all entries for a session (used in tests or session cleanup).
    pub async fn delete_session_entries(&self, session_id: SessionId) -> Result<u64, sqlx::Error> {
        let result = sqlx::query("DELETE FROM session_entries WHERE session_id = $1")
            .bind(session_id.0)
            .execute(&self.pool)
            .await?;
        Ok(result.rows_affected())
    }
}

/// Map an EntryType variant to its tag string for the entry_type column.
fn entry_type_tag(entry_type: &EntryType) -> &'static str {
    match entry_type {
        EntryType::UserMessage(_) => "user_message",
        EntryType::AssistantMessage { .. } => "assistant_message",
        EntryType::CodeExecution { .. } => "code_execution",
        EntryType::HostCall { .. } => "host_call",
        EntryType::ReplSnapshot(_) => "repl_snapshot",
        EntryType::Compaction { .. } => "compaction",
        EntryType::BranchSummary(_) => "branch_summary",
        EntryType::System(_) => "system",
    }
}
