use serde::{Deserialize, Serialize};

pub use gw_core::{LlmMessage, LlmResponse};

pub type Message = LlmMessage;
pub type CompletionResponse = LlmResponse;

/// Parse an embedding vector from a JSON array of numbers.
fn parse_embedding(value: &serde_json::Value) -> Vec<f32> {
    value
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect()
        })
        .unwrap_or_default()
}

/// LLM inference backend.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum LlmBackend {
    /// Ollama native API (/api/chat, /api/embed).
    Ollama,
    /// OpenAI-compatible API (/v1/chat/completions) — SGLang, vLLM, etc.
    Sglang,
}

impl std::fmt::Display for LlmBackend {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LlmBackend::Ollama => write!(f, "ollama"),
            LlmBackend::Sglang => write!(f, "sglang"),
        }
    }
}

impl std::str::FromStr for LlmBackend {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "ollama" => Ok(LlmBackend::Ollama),
            "sglang" | "vllm" | "openai" => Ok(LlmBackend::Sglang),
            _ => Err(format!(
                "unknown LLM backend: {s} (expected: ollama, sglang)"
            )),
        }
    }
}

/// LLM client that supports Ollama and OpenAI-compatible (SGLang/vLLM) backends.
///
/// - Chat requests route through `proxy_url` (Ollama: /api/chat, SGLang: /v1/chat/completions).
/// - Embedding requests always use Ollama's /api/embed via `direct_url`.
#[derive(Clone)]
pub struct OllamaClient {
    pub proxy_url: String,
    pub direct_url: String,
    pub default_model: String,
    pub embedding_model: String,
    pub backend: LlmBackend,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(
        proxy_url: String,
        direct_url: String,
        default_model: String,
        embedding_model: String,
    ) -> Self {
        Self::with_backend(
            proxy_url,
            direct_url,
            default_model,
            embedding_model,
            LlmBackend::Ollama,
        )
    }

    pub fn with_backend(
        proxy_url: String,
        direct_url: String,
        default_model: String,
        embedding_model: String,
        backend: LlmBackend,
    ) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            proxy_url,
            direct_url,
            default_model,
            embedding_model,
            backend,
            client,
        }
    }

    /// Returns the chat endpoint URL based on the active backend.
    fn chat_url(&self) -> String {
        match self.backend {
            LlmBackend::Ollama => format!("{}/api/chat", self.proxy_url),
            LlmBackend::Sglang => format!("{}/v1/chat/completions", self.proxy_url),
        }
    }

    /// Non-streaming chat completion.
    /// For SGLang backend, defaults to think=false (Qwen3.5 thinking mode off).
    pub async fn chat(
        &self,
        messages: &[Message],
        model: Option<&str>,
    ) -> Result<CompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let think = match self.backend {
            LlmBackend::Sglang => Some(false), // SGLang + Qwen3.5: disable thinking by default
            _ => None,
        };
        self.chat_with_options(messages, model, think).await
    }

    /// Non-streaming chat completion with optional think parameter.
    /// think: None = default, Some(true) = enable thinking, Some(false) = disable thinking.
    #[tracing::instrument(
        name = "gen_ai.chat",
        skip(self, messages),
        fields(
            gen_ai.operation.name = "chat",
            gen_ai.request.model = tracing::field::Empty,
            gen_ai.provider.name = "ollama",
            gen_ai.usage.input_tokens = tracing::field::Empty,
            gen_ai.usage.output_tokens = tracing::field::Empty,
        )
    )]
    pub async fn chat_with_options(
        &self,
        messages: &[Message],
        model: Option<&str>,
        think: Option<bool>,
    ) -> Result<CompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
        let model = model.unwrap_or(&self.default_model);
        tracing::Span::current().record("gen_ai.request.model", model);

        let mut body = serde_json::json!({
            "model": model,
            "messages": messages,
            "stream": false,
        });
        if let Some(think_val) = think {
            match self.backend {
                LlmBackend::Ollama => {
                    body["think"] = serde_json::Value::Bool(think_val);
                }
                LlmBackend::Sglang => {
                    // SGLang uses chat_template_kwargs for Qwen3.5 thinking mode
                    body["chat_template_kwargs"] = serde_json::json!({
                        "enable_thinking": think_val,
                    });
                }
            }
        }

        let resp = self.client.post(self.chat_url()).json(&body).send().await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let backend = self.backend;
            return Err(format!("{backend} returned {status}: {body}").into());
        }

        let json: serde_json::Value = resp.json().await?;

        let (content, input_tokens, output_tokens) = match self.backend {
            LlmBackend::Ollama => {
                let content = json["message"]["content"]
                    .as_str()
                    .unwrap_or("")
                    .to_string();
                let input = json["prompt_eval_count"].as_u64().map(|n| n as u32);
                let output = json["eval_count"].as_u64().map(|n| n as u32);
                (content, input, output)
            }
            LlmBackend::Sglang => {
                let content = json["choices"]
                    .as_array()
                    .and_then(|c| c.first())
                    .and_then(|c| c["message"]["content"].as_str())
                    .unwrap_or("")
                    .to_string();
                let input = json["usage"]["prompt_tokens"].as_u64().map(|n| n as u32);
                let output = json["usage"]["completion_tokens"]
                    .as_u64()
                    .map(|n| n as u32);
                (content, input, output)
            }
        };

        let span = tracing::Span::current();
        if let Some(t) = input_tokens {
            span.record("gen_ai.usage.input_tokens", t);
        }
        if let Some(t) = output_tokens {
            span.record("gen_ai.usage.output_tokens", t);
        }

        Ok(CompletionResponse {
            content,
            model: Some(model.to_string()),
            input_tokens,
            output_tokens,
        })
    }

    /// Generate embeddings for a batch of texts via Ollama's /api/embed endpoint.
    /// Batches in groups of 32, truncates texts to 8192 chars.
    /// On batch failure, retries individually (truncated to 4096), zero-vector fallback.
    #[tracing::instrument(
        name = "gen_ai.embeddings",
        skip(self, texts),
        fields(
            gen_ai.operation.name = "embeddings",
            gen_ai.request.model = %self.embedding_model,
            gen_ai.provider.name = "ollama",
            gw.batch_size = texts.len(),
        )
    )]
    pub async fn embed(
        &self,
        texts: &[String],
    ) -> Result<Vec<Vec<f32>>, Box<dyn std::error::Error + Send + Sync>> {
        const BATCH_SIZE: usize = 32;
        const MAX_CHARS: usize = 8192;
        const RETRY_MAX_CHARS: usize = 4096;

        let url = format!("{}/api/embed", self.direct_url);
        let mut all_embeddings: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
        let mut embedding_dim: Option<usize> = None;

        for batch in texts.chunks(BATCH_SIZE) {
            let truncated: Vec<String> = batch
                .iter()
                .map(|t| {
                    if t.len() > MAX_CHARS {
                        t[..MAX_CHARS].to_string()
                    } else {
                        t.clone()
                    }
                })
                .collect();

            let body = serde_json::json!({
                "model": self.embedding_model,
                "input": truncated,
            });

            let resp = self.client.post(&url).json(&body).send().await?;
            let status = resp.status();

            if status.is_success() {
                let json: serde_json::Value = resp.json().await?;
                if let Some(embeddings) = json["embeddings"].as_array() {
                    for emb in embeddings {
                        let vec = parse_embedding(emb);
                        if embedding_dim.is_none() && !vec.is_empty() {
                            embedding_dim = Some(vec.len());
                        }
                        all_embeddings.push(vec);
                    }
                }
            } else {
                // Batch failed — retry individually.
                tracing::warn!(status = %status, "Batch embed failed, retrying individually");
                for text in batch {
                    let t = if text.len() > RETRY_MAX_CHARS {
                        &text[..RETRY_MAX_CHARS]
                    } else {
                        text.as_str()
                    };
                    let body = serde_json::json!({
                        "model": self.embedding_model,
                        "input": [t],
                    });
                    let vec = match self.client.post(&url).json(&body).send().await {
                        Ok(r) if r.status().is_success() => {
                            let json: serde_json::Value = r.json().await?;
                            json["embeddings"]
                                .as_array()
                                .and_then(|a| a.first())
                                .map(parse_embedding)
                                .unwrap_or_default()
                        }
                        _ => {
                            tracing::warn!("Single embed failed, using zero-vector fallback");
                            Vec::new()
                        }
                    };
                    if embedding_dim.is_none() && !vec.is_empty() {
                        embedding_dim = Some(vec.len());
                    }
                    if vec.is_empty() {
                        all_embeddings.push(vec![0.0; embedding_dim.unwrap_or(768)]);
                    } else {
                        all_embeddings.push(vec);
                    }
                }
            }
        }

        Ok(all_embeddings)
    }

    /// Streaming chat — returns the raw reqwest Response for line-by-line processing.
    ///
    /// Stream format varies by backend:
    /// - Ollama: NDJSON lines with `message.content` (token) and `done` flag.
    /// - SGLang: SSE lines (`data: {...}`) with `choices[0].delta.content`.
    #[tracing::instrument(
        name = "gen_ai.chat",
        skip(self, messages),
        fields(
            gen_ai.operation.name = "chat",
            gen_ai.request.model = tracing::field::Empty,
            gen_ai.provider.name = "ollama",
            gw.streaming = true,
        )
    )]
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        model: Option<&str>,
    ) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
        let model = model.unwrap_or(&self.default_model);
        tracing::Span::current().record("gen_ai.request.model", model);

        let resp = self
            .client
            .post(self.chat_url())
            .json(&serde_json::json!({
                "model": model,
                "messages": messages,
                "stream": true,
            }))
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            let backend = self.backend;
            return Err(format!("{backend} returned {status}: {body}").into());
        }

        Ok(resp)
    }
}
