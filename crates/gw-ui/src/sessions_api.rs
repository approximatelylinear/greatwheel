//! HTTP API for the sessions sidebar — Phase 1 of the multi-user /
//! multi-session plan in `docs/design-conversation-loop.md` (and the
//! brainstorm in our session log).
//!
//! Four endpoints, all JSON:
//!
//!   POST   /users
//!     Idempotent upsert of a user row. Body `{user_id?, name?}`.
//!     Lets the frontend stamp a localStorage-generated UUID into PG
//!     on first load. Returns the stored row.
//!
//!   GET    /users/{uid}/sessions?include_archived=0
//!     Sessions for this user, newest `last_active_at` first. Default
//!     hides archived; `?include_archived=1` adds them to the listing
//!     (active sessions still appear too). Each row carries enough for
//!     the sidebar to render without a follow-up fetch (title,
//!     timestamps, message_count).
//!
//!   POST   /users/{uid}/sessions
//!     Creates a fresh session row owned by `uid`. Body `{title?}`.
//!     Returns `{session_id}`.
//!
//!   PATCH  /sessions/{sid}
//!     Body `{title?, archived?}`. Used by the auto-titler and the
//!     archive button.
//!
//! The router is mounted by example binaries that own the org/agent_def
//! defaults (literature_assistant seeds those via fixed UUIDs). Pass
//! the same defaults in via `SessionsApiConfig` so users/sessions
//! created here pin to the same FK chain that `flush_to_pg` expects.
//!
//! No auth. The `?user=<uuid>` model is trust-the-client; layer real
//! auth on top later without changing this surface.

use axum::extract::{Path, Query, State};
use axum::http::StatusCode;
use axum::routing::{get, patch, post};
use axum::{Json, Router};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use sqlx::PgPool;
use uuid::Uuid;

/// Defaults the binary mounting this router owns: which org / agent_def
/// new users and sessions are tied to. Required because the schema
/// FKs `users.org_id` and `sessions.{org_id, agent_id}` to NOT NULL
/// columns; the API itself doesn't pick.
#[derive(Clone, Debug)]
pub struct SessionsApiConfig {
    pub default_org_id: Uuid,
    pub default_agent_id: Uuid,
}

#[derive(Clone)]
struct ApiState {
    pool: PgPool,
    config: SessionsApiConfig,
}

/// Build the sessions API router. Mount with `.merge(...)` alongside
/// the rest of the binary's routes; CORS / tracing layers are the
/// caller's responsibility (consistent with the existing spine routes
/// in `literature_assistant`).
pub fn router(pool: PgPool, config: SessionsApiConfig) -> Router {
    let state = ApiState { pool, config };
    Router::new()
        .route("/users", post(handle_upsert_user))
        .route("/users/{uid}/sessions", get(handle_list_sessions))
        .route("/users/{uid}/sessions", post(handle_create_session))
        .route("/sessions/{sid}", patch(handle_patch_session))
        .with_state(state)
}

// ─── shapes ───────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct UpsertUserBody {
    /// Client-supplied UUID — typically the value the frontend
    /// generated and stored in localStorage. When absent, the server
    /// mints one.
    pub user_id: Option<Uuid>,
    pub name: Option<String>,
    pub email: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct UserRow {
    pub user_id: Uuid,
    pub name: String,
    pub email: Option<String>,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Deserialize)]
pub struct ListSessionsQuery {
    /// `?include_archived=1` adds archived sessions to the listing
    /// (active ones still appear). Default omits archived.
    #[serde(default)]
    pub include_archived: Option<u8>,
}

#[derive(Debug, Serialize)]
pub struct SessionListItem {
    pub session_id: Uuid,
    pub title: Option<String>,
    pub summary: Option<String>,
    pub created_at: DateTime<Utc>,
    pub last_active_at: DateTime<Utc>,
    pub archived_at: Option<DateTime<Utc>>,
    pub message_count: i64,
}

#[derive(Debug, Deserialize)]
pub struct CreateSessionBody {
    pub title: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct CreateSessionResp {
    pub session_id: Uuid,
}

#[derive(Debug, Deserialize)]
pub struct PatchSessionBody {
    pub title: Option<String>,
    pub archived: Option<bool>,
}

// ─── handlers ─────────────────────────────────────────────────────────

async fn handle_upsert_user(
    State(state): State<ApiState>,
    Json(body): Json<UpsertUserBody>,
) -> Result<Json<UserRow>, (StatusCode, String)> {
    let user_id = body.user_id.unwrap_or_else(Uuid::new_v4);
    // ON CONFLICT (id) DO UPDATE so a returning client can also rename
    // themselves later (POST /users with same id + new name).
    type Row = (Uuid, String, Option<String>, DateTime<Utc>);
    let row: Row = sqlx::query_as(
        r#"
        INSERT INTO users (id, org_id, name, email)
        VALUES ($1, $2, $3, $4)
        ON CONFLICT (id) DO UPDATE
            SET name  = COALESCE(EXCLUDED.name, users.name),
                email = COALESCE(EXCLUDED.email, users.email)
        RETURNING id, name, email, created_at
        "#,
    )
    .bind(user_id)
    .bind(state.config.default_org_id)
    .bind(body.name.unwrap_or_else(|| short_uid(user_id)))
    .bind(body.email)
    .fetch_one(&state.pool)
    .await
    .map_err(internal)?;

    Ok(Json(UserRow {
        user_id: row.0,
        name: row.1,
        email: row.2,
        created_at: row.3,
    }))
}

async fn handle_list_sessions(
    Path(uid): Path<Uuid>,
    Query(q): Query<ListSessionsQuery>,
    State(state): State<ApiState>,
) -> Result<Json<Vec<SessionListItem>>, (StatusCode, String)> {
    // Two SQL paths because the archived filter changes the planner's
    // index choice; clearer than a CASE in WHERE.
    let include_archived = matches!(q.include_archived, Some(1));
    type Row = (
        Uuid,
        Option<String>,
        Option<String>,
        DateTime<Utc>,
        DateTime<Utc>,
        Option<DateTime<Utc>>,
        i64,
    );
    let rows: Vec<Row> = if include_archived {
        sqlx::query_as(
            r#"
            SELECT s.id, s.title, s.summary,
                   s.created_at, s.last_active_at, s.archived_at,
                   COALESCE(e.entry_count, 0) AS message_count
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS entry_count
                FROM session_entries
                GROUP BY session_id
            ) e ON e.session_id = s.id
            WHERE s.user_id = $1
            ORDER BY s.last_active_at DESC
            LIMIT 200
            "#,
        )
    } else {
        sqlx::query_as(
            r#"
            SELECT s.id, s.title, s.summary,
                   s.created_at, s.last_active_at, s.archived_at,
                   COALESCE(e.entry_count, 0) AS message_count
            FROM sessions s
            LEFT JOIN (
                SELECT session_id, COUNT(*) AS entry_count
                FROM session_entries
                GROUP BY session_id
            ) e ON e.session_id = s.id
            WHERE s.user_id = $1
              AND s.archived_at IS NULL
            ORDER BY s.last_active_at DESC
            LIMIT 200
            "#,
        )
    }
    .bind(uid)
    .fetch_all(&state.pool)
    .await
    .map_err(internal)?;

    Ok(Json(
        rows.into_iter()
            .map(|(id, title, summary, created, active, archived, count)| SessionListItem {
                session_id: id,
                title,
                summary,
                created_at: created,
                last_active_at: active,
                archived_at: archived,
                message_count: count,
            })
            .collect(),
    ))
}

async fn handle_create_session(
    Path(uid): Path<Uuid>,
    State(state): State<ApiState>,
    Json(body): Json<CreateSessionBody>,
) -> Result<Json<CreateSessionResp>, (StatusCode, String)> {
    // Verify the user exists. POST /users is idempotent so the
    // frontend usually calls it first, but a missing user yielding a
    // FK error is unhelpful — surface a 404 instead.
    let exists: Option<(Uuid,)> = sqlx::query_as("SELECT id FROM users WHERE id = $1")
        .bind(uid)
        .fetch_optional(&state.pool)
        .await
        .map_err(internal)?;
    if exists.is_none() {
        return Err((
            StatusCode::NOT_FOUND,
            format!("user {uid} not found — POST /users first"),
        ));
    }

    let session_id = Uuid::new_v4();
    sqlx::query(
        r#"
        INSERT INTO sessions
            (id, org_id, user_id, agent_id, session_key, title)
        VALUES ($1, $2, $3, $4, $5, $6)
        "#,
    )
    .bind(session_id)
    .bind(state.config.default_org_id)
    .bind(uid)
    .bind(state.config.default_agent_id)
    .bind(session_id.to_string())
    .bind(body.title)
    .execute(&state.pool)
    .await
    .map_err(internal)?;

    Ok(Json(CreateSessionResp { session_id }))
}

async fn handle_patch_session(
    Path(sid): Path<Uuid>,
    State(state): State<ApiState>,
    Json(body): Json<PatchSessionBody>,
) -> Result<Json<SessionListItem>, (StatusCode, String)> {
    // Two updates rather than one COALESCE-heavy statement so an
    // empty PATCH body is a no-op (returns the row unchanged) and so
    // the archive bool maps cleanly to a NULL/now() flip.
    if let Some(title) = body.title {
        sqlx::query("UPDATE sessions SET title = $1 WHERE id = $2")
            .bind(title)
            .bind(sid)
            .execute(&state.pool)
            .await
            .map_err(internal)?;
    }
    if let Some(archived) = body.archived {
        let new_value = if archived { Some(Utc::now()) } else { None };
        sqlx::query("UPDATE sessions SET archived_at = $1 WHERE id = $2")
            .bind(new_value)
            .bind(sid)
            .execute(&state.pool)
            .await
            .map_err(internal)?;
    }

    type Row = (
        Uuid,
        Option<String>,
        Option<String>,
        DateTime<Utc>,
        DateTime<Utc>,
        Option<DateTime<Utc>>,
        i64,
    );
    let row: Option<Row> = sqlx::query_as::<_, Row>(
        r#"
        SELECT s.id, s.title, s.summary,
               s.created_at, s.last_active_at, s.archived_at,
               COALESCE(e.entry_count, 0) AS message_count
        FROM sessions s
        LEFT JOIN (
            SELECT session_id, COUNT(*) AS entry_count
            FROM session_entries
            WHERE session_id = $1
            GROUP BY session_id
        ) e ON e.session_id = s.id
        WHERE s.id = $1
        "#,
    )
    .bind(sid)
    .fetch_optional(&state.pool)
    .await
    .map_err(internal)?;

    let row = row.ok_or((
        StatusCode::NOT_FOUND,
        format!("session {sid} not found"),
    ))?;
    Ok(Json(SessionListItem {
        session_id: row.0,
        title: row.1,
        summary: row.2,
        created_at: row.3,
        last_active_at: row.4,
        archived_at: row.5,
        message_count: row.6,
    }))
}

// ─── helpers ──────────────────────────────────────────────────────────

fn internal(e: sqlx::Error) -> (StatusCode, String) {
    (StatusCode::INTERNAL_SERVER_ERROR, e.to_string())
}

/// Default display name for a freshly-minted user — "user-abc12345"
/// using the first 8 hex chars of the UUID. Lets the sidebar render
/// something readable before the user picks a real name.
fn short_uid(uid: Uuid) -> String {
    let s = uid.simple().to_string();
    format!("user-{}", &s[..8])
}
