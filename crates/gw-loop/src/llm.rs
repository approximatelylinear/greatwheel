use std::future::Future;
use std::pin::Pin;

use gw_core::{LlmMessage, LlmResponse};

use crate::error::LoopError;

/// Trait for LLM backends used by the conversation loop.
///
/// Abstracted so tests can provide a mock implementation.
pub trait LlmClient: Send + Sync {
    fn chat<'a>(
        &'a self,
        messages: &'a [LlmMessage],
        model: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LoopError>> + Send + 'a>>;
}

/// Adapter that wraps gw_llm::OllamaClient to implement LlmClient.
pub struct OllamaLlmClient {
    inner: gw_llm::OllamaClient,
    /// If `Some`, explicitly passes this `think` flag on every chat
    /// call. qwen3.5 and other thinking-mode models produce
    /// `<think>...</think>` content by default; setting `Some(false)`
    /// disables that and gets you visible output without the
    /// pre-reasoning. `None` (the default) uses whatever the backend
    /// decides.
    think: Option<bool>,
}

impl OllamaLlmClient {
    pub fn new(client: gw_llm::OllamaClient) -> Self {
        Self {
            inner: client,
            think: None,
        }
    }

    /// Force the `think` flag on every chat call. For qwen3.5-family
    /// models on Ollama, `Some(false)` is the usual choice when you
    /// need responses rather than reasoning traces.
    pub fn with_think(mut self, think: Option<bool>) -> Self {
        self.think = think;
        self
    }
}

impl LlmClient for OllamaLlmClient {
    fn chat<'a>(
        &'a self,
        messages: &'a [LlmMessage],
        model: Option<&'a str>,
    ) -> Pin<Box<dyn Future<Output = Result<LlmResponse, LoopError>> + Send + 'a>> {
        Box::pin(async move {
            self.inner
                .chat_with_options(messages, model, self.think)
                .await
                .map_err(|e| LoopError::Llm(e.to_string()))
        })
    }
}
