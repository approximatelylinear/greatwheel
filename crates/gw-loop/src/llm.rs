use std::future::Future;
use std::pin::Pin;

use crate::context::LlmMessage;
use crate::error::LoopError;

/// Response from an LLM completion call.
#[derive(Debug, Clone)]
pub struct LlmResponse {
    pub content: String,
    pub model: Option<String>,
    pub input_tokens: Option<u32>,
    pub output_tokens: Option<u32>,
}

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
            // Convert LlmMessage to gw_llm::Message
            let llm_messages: Vec<gw_llm::Message> = messages
                .iter()
                .map(|m| gw_llm::Message {
                    role: m.role.clone(),
                    content: m.content.clone(),
                })
                .collect();

            let resp = self
                .inner
                .chat(&llm_messages, model)
                .await
                .map_err(|e| LoopError::Llm(e.to_string()))?;

            Ok(LlmResponse {
                content: resp.content,
                model: model.map(|s| s.to_string()),
                input_tokens: resp.input_tokens,
                output_tokens: resp.output_tokens,
            })
        })
    }
}
