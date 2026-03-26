mod session_api;
mod ws_handler;

use axum::{
    extract::State,
    response::{
        sse::{Event, Sse},
        Html, IntoResponse, Json,
    },
    routing::{get, post},
    Router,
};
use clap::Parser;
use gw_engine::GreatWheelEngine;
use gw_llm::{Message, OllamaClient};
use gw_loop::{LoopConfig, OllamaLlmClient, SessionManager};
use reqwest::StatusCode;
use serde::Deserialize;
use std::collections::HashMap;
use std::{convert::Infallible, sync::Arc, time::Duration};
use tokio_stream::wrappers::ReceiverStream;

#[derive(Parser)]
#[command(name = "greatwheel", about = "Greatwheel agentic runtime")]
struct Cli {
    #[arg(short, long, default_value = "config/greatwheel.toml")]
    config: String,
}

#[derive(Debug, Deserialize)]
struct Config {
    server: ServerConfig,
    database: DatabaseConfig,
    llm: LlmConfig,
    memory: MemoryConfig,
    #[serde(default)]
    tracing: gw_trace::TracingConfig,
    /// Per-plugin config sections: [plugins.name] → value.
    #[serde(default)]
    plugins: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Deserialize)]
struct ServerConfig {
    host: String,
    port: u16,
}

#[derive(Debug, Deserialize)]
struct DatabaseConfig {
    url: String,
}

#[derive(Debug, Clone, Deserialize)]
struct MemoryConfig {
    lance_db_path: String,
    embedding_dim: i32,
    #[serde(default = "default_tantivy_path")]
    tantivy_index_path: String,
}

fn default_tantivy_path() -> String {
    "data/tantivy".into()
}

#[derive(Debug, Clone, Deserialize)]
struct LlmConfig {
    proxy_url: String,
    ollama_url: String,
    default_model: String,
    embedding_model: String,
}

#[derive(Clone)]
struct AppState {
    llm: Arc<OllamaClient>,
    config: Arc<LlmConfig>,
    session_mgr: Arc<SessionManager>,
}

#[derive(Deserialize)]
struct ChatRequest {
    messages: Vec<Message>,
    model: Option<String>,
}

async fn health() -> Json<serde_json::Value> {
    Json(serde_json::json!({ "status": "ok" }))
}

async fn index() -> Html<&'static str> {
    Html(include_str!("chat.html"))
}

async fn get_config(State(app): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "default_model": app.config.default_model,
        "proxy_url": app.config.proxy_url,
        "ollama_url": app.config.ollama_url,
    }))
}

async fn list_models(State(app): State<AppState>) -> impl IntoResponse {
    let client = reqwest::Client::new();
    for url in [
        format!("{}/api/tags", app.config.proxy_url),
        format!("{}/api/tags", app.config.ollama_url),
    ] {
        if let Ok(resp) = client.get(&url).send().await {
            if let Ok(json) = resp.json::<serde_json::Value>().await {
                return Json(json);
            }
        }
    }
    Json(serde_json::json!({ "models": [] }))
}

async fn chat(
    State(app): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Sse<ReceiverStream<Result<Event, Infallible>>>, (StatusCode, String)> {
    let model = req.model.as_deref();

    let mut resp = app
        .llm
        .chat_stream(&req.messages, model)
        .await
        .map_err(|e| (StatusCode::BAD_GATEWAY, format!("LLM error: {e}")))?;

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);

    tokio::spawn(async move {
        let mut buffer = String::new();

        while let Ok(Some(chunk)) = resp.chunk().await {
            buffer.push_str(&String::from_utf8_lossy(&chunk));

            while let Some(pos) = buffer.find('\n') {
                let line: String = buffer.drain(..=pos).collect();
                let line = line.trim();
                if line.is_empty() {
                    continue;
                }

                if let Ok(json) = serde_json::from_str::<serde_json::Value>(line) {
                    if let Some(content) = json["message"]["content"].as_str() {
                        if !content.is_empty() {
                            let _ = tx
                                .send(Ok(Event::default().event("token").data(content)))
                                .await;
                        }
                    }

                    if json["done"].as_bool() == Some(true) {
                        let stats = serde_json::json!({
                            "model": json["model"],
                            "input_tokens": json["prompt_eval_count"],
                            "output_tokens": json["eval_count"],
                            "total_duration_ms": json["total_duration"]
                                .as_u64()
                                .map(|ns| ns / 1_000_000),
                        });
                        let _ = tx
                            .send(Ok(Event::default().event("done").data(stats.to_string())))
                            .await;
                    }
                }
            }
        }
    });

    Ok(Sse::new(ReceiverStream::new(rx)))
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    let config_str = tokio::fs::read_to_string(&cli.config).await?;
    let config: Config = toml::from_str(&config_str)?;

    // Connect Postgres first — needed for trace layer
    let pg_pool = match sqlx::postgres::PgPoolOptions::new()
        .max_connections(20)
        .connect(&config.database.url)
        .await
    {
        Ok(pool) => Some(pool),
        Err(e) => {
            eprintln!("Database unavailable (chat still works): {e}");
            None
        }
    };

    // Initialize tracing (console + optional OTLP + optional Postgres)
    let trace_pool = if config.tracing.postgres_export {
        pg_pool.clone()
    } else {
        None
    };
    gw_trace::init_tracing(&config.tracing, trace_pool)
        .expect("Failed to initialize tracing");

    if pg_pool.is_some() {
        tracing::info!("Database connected");
    }

    let llm = OllamaClient::new(
        config.llm.proxy_url.clone(),
        config.llm.ollama_url.clone(),
        config.llm.default_model.clone(),
        config.llm.embedding_model.clone(),
    );

    let llm = Arc::new(llm);

    // Clone pool for session manager before memory store consumes it.
    let session_pool = pg_pool.clone();

    // Initialize memory store if database is available
    if let Some(pool) = pg_pool {
        match gw_memory::lance::LanceStore::new(
            &config.memory.lance_db_path,
            config.memory.embedding_dim,
        )
        .await
        {
            Ok(lance) => {
                match gw_memory::tantivy_store::TantivyStore::open(
                    std::path::Path::new(&config.memory.tantivy_index_path),
                ) {
                    Ok(tantivy) => {
                        let pg_store = gw_memory::postgres::PgMemoryStore::new(pool);
                        let _memory_store =
                            gw_memory::HybridStore::new(pg_store, lance, tantivy, llm.clone());
                        tracing::info!("Memory store initialized (LanceDB + tantivy + Postgres)");
                    }
                    Err(e) => tracing::warn!("Tantivy unavailable: {e}"),
                }
            }
            Err(e) => tracing::warn!("LanceDB unavailable: {e}"),
        }
    }

    // Initialize plugin engine.
    let engine = GreatWheelEngine::new()
        .add_plugin(gw_engine::builtins::hindsight_retain::HindsightRetainPlugin)
        .add_plugin(gw_engine::builtins::hindsight_recall::HindsightRecallPlugin);
    let engine = engine.init(&config.plugins).map_err(|e| {
        format!("Plugin engine init failed: {e}")
    })?;
    engine.before_startup();

    // Create session manager with LLM factory.
    let llm_for_factory = llm.clone();
    let llm_factory: Arc<dyn Fn() -> Box<dyn gw_loop::LlmClient> + Send + Sync> =
        Arc::new(move || {
            Box::new(OllamaLlmClient::new((*llm_for_factory).clone()))
        });

    let session_mgr = Arc::new(match session_pool {
        Some(pool) => SessionManager::with_pg(
            llm_factory,
            LoopConfig::default(),
            Duration::from_secs(30 * 60),
            pool,
        ),
        None => SessionManager::new(
            llm_factory,
            LoopConfig::default(),
            Duration::from_secs(30 * 60),
        ),
    });

    let state = AppState {
        llm,
        config: Arc::new(config.llm),
        session_mgr,
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/api/config", get(get_config))
        .route("/api/models", get(list_models))
        .route("/api/chat", post(chat))
        // Session-based conversation endpoints
        .route("/api/sessions", get(session_api::list_sessions))
        .route("/api/sessions/create", post(session_api::create_session))
        .route("/api/sessions/message", post(session_api::send_message))
        .route("/api/sessions/tree", post(session_api::get_tree))
        .route("/api/sessions/state", post(session_api::get_repl_state))
        .route("/api/sessions/compact", post(session_api::compact))
        .route("/api/sessions/branch", post(session_api::switch_branch))
        .route("/api/sessions/ask", post(session_api::get_pending_ask))
        .route("/api/sessions/reply", post(session_api::reply_to_ask))
        .route("/api/sessions/resume", post(session_api::resume_session))
        .route("/api/sessions/end", post(session_api::end_session))
        // WebSocket session endpoint
        .route("/api/ws", get(ws_handler::ws_session))
        .with_state(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Greatwheel running at http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;

    engine.after_startup();
    axum::serve(listener, app).await?;

    // Shutdown plugins and tracing.
    let shutdown_errors = engine.shutdown();
    for (name, err) in &shutdown_errors {
        tracing::warn!(plugin = name, error = %err, "plugin shutdown error");
    }
    gw_trace::shutdown_tracing();

    Ok(())
}
