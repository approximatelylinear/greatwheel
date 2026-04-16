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
}

impl OllamaLlmClient {
    pub fn new(client: gw_llm::OllamaClient) -> Self {
        Self { inner: client }
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
                .chat(messages, model)
                .await
                .map_err(|e| LoopError::Llm(e.to_string()))
        })
    }
}
