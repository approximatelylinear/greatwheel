use chrono::{DateTime, Utc};
use gw_core::{AgentId, OrgId, SessionId};
use opentelemetry::trace::TracerProvider as _;
use opentelemetry_otlp::WithExportConfig as _;
use serde::{Deserialize, Serialize};
use std::sync::OnceLock;

mod postgres_layer;

pub use postgres_layer::PostgresTraceLayer;

// -------------------------------------------------------------------------- //
// Config
// -------------------------------------------------------------------------- //

/// Tracing configuration — maps to `[tracing]` in greatwheel.toml.
#[derive(Debug, Clone, Deserialize)]
pub struct TracingConfig {
    /// "console", "otlp", or "both"
    #[serde(default = "default_exporter")]
    pub exporter: String,
    /// OTLP gRPC endpoint (e.g. "http://localhost:4317")
    #[serde(default)]
    pub otlp_endpoint: Option<String>,
    /// Whether to persist trace spans to the Postgres `traces` table
    #[serde(default)]
    pub postgres_export: bool,
    /// OTel service name
    #[serde(default = "default_service_name")]
    pub service_name: String,
}

fn default_exporter() -> String {
    "console".into()
}
fn default_service_name() -> String {
    "greatwheel".into()
}

impl Default for TracingConfig {
    fn default() -> Self {
        Self {
            exporter: default_exporter(),
            otlp_endpoint: None,
            postgres_export: false,
            service_name: default_service_name(),
        }
    }
}

// -------------------------------------------------------------------------- //
// Error
// -------------------------------------------------------------------------- //

#[derive(Debug, thiserror::Error)]
pub enum TraceError {
    #[error("OpenTelemetry trace error: {0}")]
    OtelTrace(#[from] opentelemetry::trace::TraceError),
    #[error("Tracing subscriber init error: {0}")]
    Init(String),
}

// -------------------------------------------------------------------------- //
// TraceRecord — maps to the `traces` Postgres table
// -------------------------------------------------------------------------- //

/// A recorded trace span for OTel GenAI instrumentation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TraceRecord {
    pub trace_id: String,
    pub span_id: String,
    pub parent_span_id: Option<String>,
    pub operation_name: String,
    pub org_id: OrgId,
    pub agent_id: Option<AgentId>,
    pub session_id: Option<SessionId>,
    pub model: Option<String>,
    pub provider: Option<String>,
    pub input_tokens: Option<i32>,
    pub output_tokens: Option<i32>,
    pub duration_ms: i64,
    pub status: String,
    pub attributes: Option<serde_json::Value>,
    pub created_at: DateTime<Utc>,
}

// -------------------------------------------------------------------------- //
// Init / Shutdown
// -------------------------------------------------------------------------- //

/// Global handle to the OTel tracer provider so we can flush on shutdown.
static TRACER_PROVIDER: OnceLock<opentelemetry_sdk::trace::TracerProvider> = OnceLock::new();

/// Initialize the tracing subscriber stack.
///
/// Layers:
/// 1. `fmt::Layer` + `EnvFilter` — always (console output)
/// 2. `tracing_opentelemetry::OpenTelemetryLayer` — when exporter is "otlp" or "both"
/// 3. `PostgresTraceLayer` — when `postgres_export` is true and a pool is provided
pub fn init_tracing(
    config: &TracingConfig,
    pg_pool: Option<sqlx::PgPool>,
) -> Result<(), TraceError> {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

    let env_filter = EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info"));
    let fmt_layer = tracing_subscriber::fmt::layer();

    let registry = tracing_subscriber::registry()
        .with(env_filter)
        .with(fmt_layer);

    let want_otlp = matches!(config.exporter.as_str(), "otlp" | "both");

    // Build optional OTLP layer
    let otel_layer = if want_otlp {
        let endpoint = config
            .otlp_endpoint
            .as_deref()
            .unwrap_or("http://localhost:4317");

        let exporter = opentelemetry_otlp::SpanExporter::builder()
            .with_tonic()
            .with_endpoint(endpoint)
            .build()
            .map_err(TraceError::OtelTrace)?;

        let provider = opentelemetry_sdk::trace::TracerProvider::builder()
            .with_batch_exporter(exporter, opentelemetry_sdk::runtime::Tokio)
            .with_resource(opentelemetry_sdk::Resource::new(vec![
                opentelemetry::KeyValue::new("service.name", config.service_name.clone()),
            ]))
            .build();

        let tracer = provider.tracer("greatwheel");
        let _ = TRACER_PROVIDER.set(provider);

        Some(tracing_opentelemetry::layer().with_tracer(tracer))
    } else {
        None
    };

    // Build optional Postgres layer
    let pg_layer = if config.postgres_export {
        pg_pool.map(PostgresTraceLayer::new)
    } else {
        None
    };

    registry.with(otel_layer).with(pg_layer).init();

    tracing::info!(
        exporter = %config.exporter,
        postgres_export = config.postgres_export,
        service_name = %config.service_name,
        "Tracing initialized"
    );

    Ok(())
}

/// Flush pending spans and shut down the OTel tracer provider.
pub fn shutdown_tracing() {
    if let Some(provider) = TRACER_PROVIDER.get() {
        let _ = provider.shutdown();
    }
}
