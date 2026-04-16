use axum::{extract::State, Json};
use gw_core::SessionId;
use reqwest::StatusCode;
use serde::Deserialize;

use crate::AppState;

#[derive(Deserialize)]
pub struct SessionMessageRequest {
    pub session_id: String,
    pub message: String,
}

#[derive(Deserialize)]
pub struct SessionIdRequest {
    pub session_id: String,
}

#[derive(Deserialize)]
pub struct BranchRequest {
    pub session_id: String,
    pub target: String,
    #[serde(default = "default_true")]
    pub summarize: bool,
}

fn default_true() -> bool {
    true
}

fn parse_sid(s: &str) -> Result<SessionId, (StatusCode, String)> {
    let uuid: uuid::Uuid = s
        .parse()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid session ID: {e}")))?;
    Ok(SessionId(uuid))
}

pub async fn create_session(State(app): State<AppState>) -> Json<serde_json::Value> {
    let session_id = app.session_mgr.create_session().await;
    Json(serde_json::json!({ "session_id": session_id.0 }))
}

pub async fn send_message(
    State(app): State<AppState>,
    Json(req): Json<SessionMessageRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    let mgr = app.session_mgr.clone();
    let message = req.message.clone();

    // send_message may trigger auto-compaction which holds non-Send ouros types.
    let result = tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(mgr.send_message(sid, &message))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("task error: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "response": result.response,
        "is_final": result.is_final,
        "iterations": result.iterations,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
    })))
}

pub async fn get_tree(
    State(app): State<AppState>,
    Json(req): Json<SessionIdRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    let entries = app
        .session_mgr
        .get_tree(sid)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(serde_json::json!({ "entries": entries })))
}

pub async fn get_repl_state(
    State(app): State<AppState>,
    Json(req): Json<SessionIdRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    let state = app
        .session_mgr
        .get_repl_state(sid)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(serde_json::json!({ "repl_state": state })))
}

pub async fn compact(
    State(app): State<AppState>,
    Json(req): Json<SessionIdRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    // compact() holds non-Send ouros types across LLM await.
    // Run on a dedicated thread with its own tokio runtime.
    let mgr = app.session_mgr.clone();
    tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(mgr.compact(sid))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("task error: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "status": "compacted" })))
}

pub async fn switch_branch(
    State(app): State<AppState>,
    Json(req): Json<BranchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    let target_uuid: uuid::Uuid = req
        .target
        .parse()
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("invalid target ID: {e}")))?;
    let target = gw_core::EntryId(target_uuid);
    let summarize = req.summarize;
    let mgr = app.session_mgr.clone();

    tokio::task::spawn_blocking(move || {
        tokio::runtime::Handle::current().block_on(mgr.switch_branch(sid, target, summarize))
    })
    .await
    .map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("task error: {e}"),
        )
    })?
    .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(serde_json::json!({ "status": "switched" })))
}

pub async fn get_pending_ask(
    State(app): State<AppState>,
    Json(req): Json<SessionIdRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    let prompt = app
        .session_mgr
        .get_pending_ask(sid)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "pending": prompt.is_some(),
        "prompt": prompt,
    })))
}

#[derive(Deserialize)]
pub struct ReplyRequest {
    pub session_id: String,
    pub reply: String,
}

pub async fn reply_to_ask(
    State(app): State<AppState>,
    Json(req): Json<ReplyRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    let delivered = app
        .session_mgr
        .reply_to_ask(sid, req.reply)
        .await
        .map_err(|e| (StatusCode::NOT_FOUND, e.to_string()))?;

    Ok(Json(serde_json::json!({
        "delivered": delivered,
    })))
}

pub async fn resume_session(
    State(app): State<AppState>,
    Json(req): Json<SessionIdRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let sid = parse_sid(&req.session_id)?;
    app.session_mgr
        .resume_session(sid)
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;

    Ok(Json(
        serde_json::json!({ "status": "resumed", "session_id": req.session_id }),
    ))
}

pub async fn list_sessions(State(app): State<AppState>) -> Json<serde_json::Value> {
    let sessions = app.session_mgr.list_sessions().await;
    let list: Vec<serde_json::Value> = sessions
        .iter()
        .map(|(id, status)| {
            serde_json::json!({
                "session_id": id.0,
                "status": format!("{status:?}"),
            })
        })
        .collect();
    Json(serde_json::json!({ "sessions": list }))
}

pub async fn end_session(
    State(app): State<AppState>,
    Json(req): Json<SessionIdRequest>,
) -> Json<serde_json::Value> {
    let sid = parse_sid(&req.session_id).unwrap_or(SessionId(uuid::Uuid::nil()));
    let removed = app.session_mgr.end_session(sid).await;
    Json(serde_json::json!({ "ended": removed }))
}
