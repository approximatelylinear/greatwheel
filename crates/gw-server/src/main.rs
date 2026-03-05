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
use gw_llm::{Message, OllamaClient};
use reqwest::StatusCode;
use serde::Deserialize;
use std::{convert::Infallible, sync::Arc};
use tokio_stream::wrappers::ReceiverStream;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

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
    tracing_subscriber::registry()
        .with(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .with(tracing_subscriber::fmt::layer())
        .init();

    let cli = Cli::parse();

    let config_str = tokio::fs::read_to_string(&cli.config).await?;
    let config: Config = toml::from_str(&config_str)?;

    // Postgres is optional for dev — chat works without it
    match sqlx::postgres::PgPoolOptions::new()
        .max_connections(20)
        .connect(&config.database.url)
        .await
    {
        Ok(_pool) => tracing::info!("Database connected"),
        Err(e) => tracing::warn!("Database unavailable (chat still works): {e}"),
    }

    let llm = OllamaClient::new(
        config.llm.proxy_url.clone(),
        config.llm.ollama_url.clone(),
        config.llm.default_model.clone(),
        config.llm.embedding_model.clone(),
    );

    let state = AppState {
        llm: Arc::new(llm),
        config: Arc::new(config.llm),
    };

    let app = Router::new()
        .route("/", get(index))
        .route("/health", get(health))
        .route("/api/config", get(get_config))
        .route("/api/models", get(list_models))
        .route("/api/chat", post(chat))
        .with_state(state);

    let addr = format!("{}:{}", config.server.host, config.server.port);
    tracing::info!("Greatwheel running at http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
