use std::collections::HashMap;
use std::sync::Arc;

use sqlx::PgPool;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id};
use tracing::Subscriber;
use tracing_subscriber::layer::{Context, Layer};
use tracing_subscriber::registry::LookupSpan;

/// Span names we capture to Postgres. Everything else is ignored.
const CAPTURED_PREFIXES: &[&str] = &[
    "gen_ai.",
    "memory.",
    "host_function",
    "invoke_agent",
    "repl.",
    "rlm.",
];

fn should_capture(name: &str) -> bool {
    CAPTURED_PREFIXES.iter().any(|p| name.starts_with(p))
}

/// Per-span storage attached via `Extensions`.
#[derive(Debug, Default)]
struct SpanFields {
    fields: HashMap<String, serde_json::Value>,
    start: Option<std::time::Instant>,
}

impl Visit for SpanFields {
    fn record_debug(&mut self, field: &Field, value: &dyn std::fmt::Debug) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::String(format!("{:?}", value)),
        );
    }

    fn record_str(&mut self, field: &Field, value: &str) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::Value::String(value.to_string()),
        );
    }

    fn record_i64(&mut self, field: &Field, value: i64) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::json!(value),
        );
    }

    fn record_u64(&mut self, field: &Field, value: u64) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::json!(value),
        );
    }

    fn record_bool(&mut self, field: &Field, value: bool) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::json!(value),
        );
    }

    fn record_f64(&mut self, field: &Field, value: f64) {
        self.fields.insert(
            field.name().to_string(),
            serde_json::json!(value),
        );
    }
}

/// A `tracing_subscriber::Layer` that persists completed spans to the Postgres `traces` table.
///
/// Only captures spans whose names match our conventions (gen_ai.*, memory.*, etc.).
/// Inserts are fire-and-forget via `tokio::spawn`.
pub struct PostgresTraceLayer {
    pool: Arc<PgPool>,
}

impl PostgresTraceLayer {
    pub fn new(pool: PgPool) -> Self {
        Self {
            pool: Arc::new(pool),
        }
    }
}

impl<S> Layer<S> for PostgresTraceLayer
where
    S: Subscriber + for<'a> LookupSpan<'a>,
{
    fn on_new_span(&self, attrs: &Attributes<'_>, id: &Id, ctx: Context<'_, S>) {
        let name = attrs.metadata().name();
        if !should_capture(name) {
            return;
        }

        let mut fields = SpanFields::default();
        fields.start = Some(std::time::Instant::now());
        attrs.record(&mut fields);

        if let Some(span) = ctx.span(id) {
            span.extensions_mut().insert(fields);
        }
    }

    fn on_record(&self, id: &Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        if let Some(span) = ctx.span(id) {
            let mut ext = span.extensions_mut();
            if let Some(fields) = ext.get_mut::<SpanFields>() {
                values.record(fields);
            }
        }
    }

    fn on_close(&self, id: Id, ctx: Context<'_, S>) {
        let span = match ctx.span(&id) {
            Some(s) => s,
            None => return,
        };

        let name = span.name().to_string();
        if !should_capture(&name) {
            return;
        }

        let ext = span.extensions();
        let fields = match ext.get::<SpanFields>() {
            Some(f) => f,
            None => return,
        };

        let duration_ms = fields
            .start
            .map(|s| s.elapsed().as_millis() as i64)
            .unwrap_or(0);

        let model = fields
            .fields
            .get("gen_ai.request.model")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let provider = fields
            .fields
            .get("gen_ai.provider.name")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        let input_tokens = fields
            .fields
            .get("gen_ai.usage.input_tokens")
            .and_then(|v| v.as_i64())
            .map(|n| n as i32);
        let output_tokens = fields
            .fields
            .get("gen_ai.usage.output_tokens")
            .and_then(|v| v.as_i64())
            .map(|n| n as i32);

        // Collect remaining fields as attributes JSON
        let skip_keys = [
            "gen_ai.request.model",
            "gen_ai.provider.name",
            "gen_ai.usage.input_tokens",
            "gen_ai.usage.output_tokens",
        ];
        let attributes: HashMap<&str, &serde_json::Value> = fields
            .fields
            .iter()
            .filter(|(k, _)| !skip_keys.contains(&k.as_str()))
            .map(|(k, v)| (k.as_str(), v))
            .collect();
        let attributes_json = if attributes.is_empty() {
            None
        } else {
            Some(serde_json::to_value(&attributes).unwrap_or_default())
        };

        let pool = self.pool.clone();

        // Fire-and-forget insert
        tokio::spawn(async move {
            let result = sqlx::query(
                "INSERT INTO traces (trace_id, span_id, parent_span_id, operation_name, \
                 org_id, model, provider, input_tokens, output_tokens, duration_ms, \
                 status, attributes) \
                 VALUES ('none', gen_random_uuid()::text, NULL, $1, \
                 '00000000-0000-0000-0000-000000000000'::uuid, $2, $3, $4, $5, $6, \
                 'ok', $7)",
            )
            .bind(&name)
            .bind(&model)
            .bind(&provider)
            .bind(input_tokens)
            .bind(output_tokens)
            .bind(duration_ms)
            .bind(&attributes_json)
            .execute(pool.as_ref())
            .await;

            if let Err(e) = result {
                // Don't use tracing here to avoid recursion
                eprintln!("gw-trace: failed to insert trace span: {e}");
            }
        });
    }
}
