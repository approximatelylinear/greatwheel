use gw_engine::HostFnRouter;
use gw_runtime::{json_to_object, AgentError, HostBridge};
use ouros::Object;
use serde_json::Value;
use std::collections::HashMap;
use std::sync::{mpsc as std_mpsc, Arc, Mutex as StdMutex};
use tokio::sync::mpsc;

use gw_core::LoopEvent;

/// Shared slot for pending `channel.ask()` calls.
/// The bridge writes the sender here; the SessionManager reads it to deliver replies.
/// Uses std::sync channels (not tokio) because the bridge runs synchronously.
#[derive(Default)]
pub struct AskSlot {
    /// The std sender for the pending reply. None if no ask is pending.
    sender: Option<std_mpsc::Sender<String>>,
    /// The prompt text for the pending ask.
    prompt: Option<String>,
}

/// Thread-safe handle to the ask slot, shared between bridge and session manager.
pub type AskHandle = Arc<StdMutex<AskSlot>>;

/// Create a new ask handle.
pub fn new_ask_handle() -> AskHandle {
    Arc::new(StdMutex::new(AskSlot::default()))
}

/// HostBridge that supports conversation-aware functions including
/// `channel.ask()` with blocking reply.
///
/// Dispatch order in [`HostBridge::call`]:
///
///   1. Hardcoded conversation primitives (`send_message`, `ask_user`,
///      `compact_session`).
///   2. Plugin host function router, if one was provided. Covers any
///      function registered by a plugin via `ctx.register_host_fn*`.
///      Async handlers are resolved via `block_in_place` +
///      `Handle::current().block_on(...)` inside the router.
///   3. Optional inner `HostBridge` fallback (e.g. legacy hand-wired
///      dispatchers). Returns `UnknownFunction` if still no match.
pub struct ConversationBridge {
    event_tx: mpsc::UnboundedSender<LoopEvent>,
    ask_handle: AskHandle,
    inner: Option<Box<dyn HostBridge>>,
    plugin_router: Option<Arc<HostFnRouter>>,
}

impl ConversationBridge {
    pub fn new(
        event_tx: mpsc::UnboundedSender<LoopEvent>,
        ask_handle: AskHandle,
        inner: Option<Box<dyn HostBridge>>,
    ) -> Self {
        Self {
            event_tx,
            ask_handle,
            inner,
            plugin_router: None,
        }
    }

    /// Construct a bridge with a plugin router in the dispatch chain.
    pub fn with_plugin_router(
        event_tx: mpsc::UnboundedSender<LoopEvent>,
        ask_handle: AskHandle,
        inner: Option<Box<dyn HostBridge>>,
        plugin_router: Option<Arc<HostFnRouter>>,
    ) -> Self {
        Self {
            event_tx,
            ask_handle,
            inner,
            plugin_router,
        }
    }
}

impl HostBridge for ConversationBridge {
    fn call(
        &mut self,
        function: &str,
        args: Vec<Value>,
        kwargs: HashMap<String, Value>,
    ) -> Result<Object, AgentError> {
        match function {
            "send_message" | "channel.send" => {
                let message = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();
                let _ = self.event_tx.send(LoopEvent::Response {
                    content: message,
                    model: None,
                });
                Ok(Object::None)
            }

            "ask_user" | "channel.ask" => {
                let prompt = args
                    .first()
                    .and_then(|v| v.as_str())
                    .unwrap_or("")
                    .to_string();

                // Create a std channel for the reply (not tokio — we block synchronously).
                let (tx, rx) = std_mpsc::channel();

                // Store the sender and prompt in the shared ask slot.
                {
                    let mut slot = self.ask_handle.lock().unwrap();
                    slot.sender = Some(tx);
                    slot.prompt = Some(prompt.clone());
                }

                // Notify listeners that we're waiting for input.
                let _ = self.event_tx.send(LoopEvent::InputRequest(prompt));

                // Block until the reply arrives. This is a std::sync::mpsc::recv()
                // which blocks the OS thread — safe because the REPL execution is
                // inherently synchronous.
                match rx.recv() {
                    Ok(reply) => {
                        let mut slot = self.ask_handle.lock().unwrap();
                        slot.sender = None;
                        slot.prompt = None;
                        Ok(Object::String(reply))
                    }
                    Err(_) => {
                        let mut slot = self.ask_handle.lock().unwrap();
                        slot.sender = None;
                        slot.prompt = None;
                        Err(AgentError::HostFunction {
                            function: "ask_user".into(),
                            message: "reply channel closed".into(),
                        })
                    }
                }
            }

            "compact_session" | "session.compact" => {
                let _ = self.event_tx.send(LoopEvent::Compact);
                Ok(Object::None)
            }

            _ => {
                // 1. Try the plugin router. Sync handlers run inline;
                //    async handlers are resolved via block_in_place +
                //    Handle::current().block_on inside HostFnRouter::dispatch.
                if let Some(router) = &self.plugin_router {
                    if let Some(result) = router.dispatch(function, args.clone(), kwargs.clone()) {
                        return match result {
                            Ok(value) => Ok(json_to_object(value)),
                            Err(e) => Err(AgentError::HostFunction {
                                function: function.to_string(),
                                message: e.to_string(),
                            }),
                        };
                    }
                }
                // 2. Fall through to an inner hand-wired bridge if one
                //    was provided (legacy path).
                if let Some(inner) = &mut self.inner {
                    inner.call(function, args, kwargs)
                } else {
                    Err(AgentError::UnknownFunction(function.to_string()))
                }
            }
        }
    }
}

/// Check if there's a pending ask and return the prompt.
pub fn get_pending_ask(handle: &AskHandle) -> Option<String> {
    let slot = handle.lock().unwrap();
    slot.prompt.clone()
}

/// Send a reply to a pending ask. Returns true if delivered, false if no ask pending.
pub fn reply_to_ask(handle: &AskHandle, reply: String) -> bool {
    let slot = handle.lock().unwrap();
    if let Some(sender) = &slot.sender {
        sender.send(reply).is_ok()
    } else {
        false
    }
}
