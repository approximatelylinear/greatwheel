use axum::{
    extract::{
        ws::{Message, WebSocket},
        State, WebSocketUpgrade,
    },
    response::IntoResponse,
};
use serde::{Deserialize, Serialize};
use tokio::sync::mpsc;

use crate::AppState;

/// Inbound WebSocket message from the client.
#[derive(Debug, Deserialize)]
#[serde(tag = "type")]
enum WsInbound {
    /// Send a user message to the session.
    #[serde(rename = "message")]
    UserMessage { content: String },
    /// Reply to a pending ask_user() call.
    #[serde(rename = "reply")]
    Reply { content: String },
    /// Request current REPL state.
    #[serde(rename = "get_state")]
    GetState,
}

/// Outbound WebSocket message to the client.
#[derive(Debug, Serialize)]
#[serde(tag = "type")]
enum WsOutbound {
    /// Session created successfully.
    #[serde(rename = "session_created")]
    SessionCreated { session_id: String },
    /// Response from the rLM.
    #[serde(rename = "response")]
    Response {
        content: String,
        is_final: bool,
        iterations: usize,
        input_tokens: u32,
        output_tokens: u32,
    },
    /// The rLM is asking the user for input.
    #[serde(rename = "input_request")]
    InputRequest { prompt: String },
    /// A turn has completed.
    #[serde(rename = "turn_complete")]
    TurnComplete,
    /// Current REPL state.
    #[serde(rename = "state")]
    State { repl_state: String },
    /// Error message.
    #[serde(rename = "error")]
    Error { message: String },
}

/// WebSocket upgrade handler. Creates a session on connect.
pub async fn ws_session(ws: WebSocketUpgrade, State(app): State<AppState>) -> impl IntoResponse {
    ws.on_upgrade(move |socket| handle_ws_session(socket, app))
}

/// Main WebSocket session handler.
async fn handle_ws_session(socket: WebSocket, app: AppState) {
    let (mut ws_tx, mut ws_rx) = socket.split();

    // Create a session for this WebSocket connection.
    let session_id = app.session_mgr.create_session().await;
    let sid_str = session_id.0.to_string();

    // Send session_created message.
    let created_msg = WsOutbound::SessionCreated {
        session_id: sid_str.clone(),
    };
    if send_json(&mut ws_tx, &created_msg).await.is_err() {
        return;
    }

    // Channel for outbound messages from background tasks.
    let (out_tx, mut out_rx) = mpsc::unbounded_channel::<WsOutbound>();

    // Spawn a task to forward outbound messages to the WebSocket.
    let forward_handle = tokio::spawn(async move {
        use futures::SinkExt;
        while let Some(msg) = out_rx.recv().await {
            if let Ok(json) = serde_json::to_string(&msg) {
                if ws_tx.send(Message::Text(json.into())).await.is_err() {
                    break;
                }
            }
        }
    });

    // Process inbound messages.
    use futures::StreamExt;
    while let Some(Ok(msg)) = ws_rx.next().await {
        match msg {
            Message::Text(text) => {
                let text_str: &str = &text;
                match serde_json::from_str::<WsInbound>(text_str) {
                    Ok(inbound) => {
                        handle_inbound(&app, session_id, inbound, out_tx.clone()).await;
                    }
                    Err(e) => {
                        let _ = out_tx.send(WsOutbound::Error {
                            message: format!("invalid message: {e}"),
                        });
                    }
                }
            }
            Message::Close(_) => break,
            _ => {} // Ignore ping/pong/binary
        }
    }

    // Clean up: end the session.
    app.session_mgr.end_session(session_id).await;
    forward_handle.abort();
}

/// Handle a single inbound WebSocket message.
async fn handle_inbound(
    app: &AppState,
    session_id: gw_core::SessionId,
    inbound: WsInbound,
    out_tx: mpsc::UnboundedSender<WsOutbound>,
) {
    match inbound {
        WsInbound::UserMessage { content } => {
            let mgr = app.session_mgr.clone();
            let out = out_tx.clone();
            let sid = session_id;

            // Run the turn in a blocking task (ouros types are not Send).
            tokio::task::spawn_blocking(move || {
                let result =
                    tokio::runtime::Handle::current().block_on(mgr.send_message(sid, &content));

                match result {
                    Ok(turn_result) => {
                        let _ = out.send(WsOutbound::Response {
                            content: turn_result.response.unwrap_or_default(),
                            is_final: turn_result.is_final,
                            iterations: turn_result.iterations,
                            input_tokens: turn_result.input_tokens,
                            output_tokens: turn_result.output_tokens,
                        });
                        let _ = out.send(WsOutbound::TurnComplete);
                    }
                    Err(e) => {
                        let _ = out.send(WsOutbound::Error {
                            message: e.to_string(),
                        });
                    }
                }
            });

            // While the turn is running, poll for ask_user() requests
            // and forward them to the client.
            let mgr = app.session_mgr.clone();
            let out_ask = out_tx.clone();
            tokio::spawn(async move {
                loop {
                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    match mgr.get_pending_ask(sid).await {
                        Ok(Some(prompt)) => {
                            let _ = out_ask.send(WsOutbound::InputRequest { prompt });
                            // Stop polling — the client will send a Reply message.
                            break;
                        }
                        Ok(None) => {
                            // Check if the turn completed (no more ask pending).
                            // We detect this by checking if the session is still
                            // processing — simplest: just keep polling briefly.
                        }
                        Err(_) => break,
                    }
                }
            });
        }

        WsInbound::Reply { content } => {
            match app.session_mgr.reply_to_ask(session_id, content).await {
                Ok(true) => {} // Reply delivered; the turn will continue.
                Ok(false) => {
                    let _ = out_tx.send(WsOutbound::Error {
                        message: "no pending ask to reply to".into(),
                    });
                }
                Err(e) => {
                    let _ = out_tx.send(WsOutbound::Error {
                        message: e.to_string(),
                    });
                }
            }
        }

        WsInbound::GetState => match app.session_mgr.get_repl_state(session_id).await {
            Ok(state) => {
                let _ = out_tx.send(WsOutbound::State { repl_state: state });
            }
            Err(e) => {
                let _ = out_tx.send(WsOutbound::Error {
                    message: e.to_string(),
                });
            }
        },
    }
}

/// Send a JSON message over the WebSocket.
async fn send_json(
    ws_tx: &mut futures::stream::SplitSink<WebSocket, Message>,
    msg: &WsOutbound,
) -> Result<(), ()> {
    use futures::SinkExt;
    let json = serde_json::to_string(msg).map_err(|_| ())?;
    ws_tx.send(Message::Text(json.into())).await.map_err(|_| ())
}
