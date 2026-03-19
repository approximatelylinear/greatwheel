use serde::{Deserialize, Serialize};
use tracing;

/// A chat message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: String,
}

/// LLM completion response.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompletionResponse {
    pub content: String,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

/// Ollama client that routes chat through rl-play and embeddings direct.
#[derive(Clone)]
pub struct OllamaClient {
    pub proxy_url: String,
    pub direct_url: String,
    pub default_model: String,
    pub embedding_model: String,
    client: reqwest::Client,
}

impl OllamaClient {
    pub fn new(
        proxy_url: String,
        direct_url: String,
        default_model: String,
        embedding_model: String,
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
            client,
        }
    }

    /// Returns the effective base URL for chat. Tries proxy first, falls back to direct.
    fn chat_url(&self) -> String {
        format!("{}/api/chat", self.proxy_url)
    }

    /// Non-streaming chat completion.
    pub async fn chat(
        &self,
        messages: &[Message],
        model: Option<&str>,
    ) -> Result<CompletionResponse, Box<dyn std::error::Error + Send + Sync>> {
        self.chat_with_options(messages, model, None).await
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
            body["think"] = serde_json::Value::Bool(think_val);
        }

        let resp = self
            .client
            .post(self.chat_url())
            .json(&body)
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Ollama returned {status}: {body}").into());
        }

        let json: serde_json::Value = resp.json().await?;
        let input_tokens = json["prompt_eval_count"].as_u64().map(|n| n as u32);
        let output_tokens = json["eval_count"].as_u64().map(|n| n as u32);

        let span = tracing::Span::current();
        if let Some(t) = input_tokens {
            span.record("gen_ai.usage.input_tokens", t);
        }
        if let Some(t) = output_tokens {
            span.record("gen_ai.usage.output_tokens", t);
        }

        Ok(CompletionResponse {
            content: json["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string(),
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
                        let vec: Vec<f32> = emb
                            .as_array()
                            .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                            .unwrap_or_default();
                        if embedding_dim.is_none() && !vec.is_empty() {
                            embedding_dim = Some(vec.len());
                        }
                        all_embeddings.push(vec);
                    }
                }
            } else {
                // Batch failed — retry individually
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
                    match self.client.post(&url).json(&body).send().await {
                        Ok(r) if r.status().is_success() => {
                            let json: serde_json::Value = r.json().await?;
                            if let Some(emb) = json["embeddings"].as_array().and_then(|a| a.first()) {
                                let vec: Vec<f32> = emb
                                    .as_array()
                                    .map(|a| a.iter().filter_map(|v| v.as_f64().map(|f| f as f32)).collect())
                                    .unwrap_or_default();
                                if embedding_dim.is_none() && !vec.is_empty() {
                                    embedding_dim = Some(vec.len());
                                }
                                all_embeddings.push(vec);
                            } else {
                                // Zero-vector fallback
                                let dim = embedding_dim.unwrap_or(768);
                                tracing::warn!("Using zero-vector fallback for text");
                                all_embeddings.push(vec![0.0; dim]);
                            }
                        }
                        _ => {
                            let dim = embedding_dim.unwrap_or(768);
                            tracing::warn!("Single embed failed, using zero-vector fallback");
                            all_embeddings.push(vec![0.0; dim]);
                        }
                    }
                }
            }
        }

        Ok(all_embeddings)
    }


    /// Streaming chat — returns the raw reqwest Response for NDJSON processing.
    /// Each line is a JSON object with `message.content` (token) and `done` flag.
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
            return Err(format!("Ollama returned {status}: {body}").into());
        }

        Ok(resp)
    }
}
