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
        Self {
            proxy_url,
            direct_url,
            default_model,
            embedding_model,
            client: reqwest::Client::new(),
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
        let model = model.unwrap_or(&self.default_model);
        tracing::debug!(model, "Sending chat completion");

        let resp = self
            .client
            .post(self.chat_url())
            .json(&serde_json::json!({
                "model": model,
                "messages": messages,
                "stream": false,
            }))
            .send()
            .await?;

        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Ollama returned {status}: {body}").into());
        }

        let json: serde_json::Value = resp.json().await?;
        Ok(CompletionResponse {
            content: json["message"]["content"]
                .as_str()
                .unwrap_or("")
                .to_string(),
            input_tokens: json["prompt_eval_count"].as_u64().map(|n| n as u32),
            output_tokens: json["eval_count"].as_u64().map(|n| n as u32),
        })
    }

    /// Streaming chat — returns the raw reqwest Response for NDJSON processing.
    /// Each line is a JSON object with `message.content` (token) and `done` flag.
    pub async fn chat_stream(
        &self,
        messages: &[Message],
        model: Option<&str>,
    ) -> Result<reqwest::Response, Box<dyn std::error::Error + Send + Sync>> {
        let model = model.unwrap_or(&self.default_model);
        tracing::debug!(model, "Starting streaming chat");

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
