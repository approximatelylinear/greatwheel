//! Event dispatcher — runs lifecycle event handlers.

use gw_core::{EventHandler, EventPayload, EventResult, LifecycleEvent};
use std::collections::HashMap;
use tracing::warn;

/// Dispatches lifecycle events to registered handlers.
///
/// Handlers run in registration order (which follows plugin priority).
/// If any handler returns `ShortCircuit`, remaining handlers are skipped.
pub struct EventDispatcher {
    handlers: HashMap<LifecycleEvent, Vec<EventHandler>>,
}

impl EventDispatcher {
    pub fn new(handlers: HashMap<LifecycleEvent, Vec<EventHandler>>) -> Self {
        Self { handlers }
    }

    /// Dispatch an event to all registered handlers.
    ///
    /// Returns the final `EventResult`. If a handler short-circuits,
    /// that result is returned immediately.
    pub fn dispatch(&self, payload: &mut EventPayload) -> EventResult {
        let handlers = match self.handlers.get(&payload.event) {
            Some(h) => h,
            None => return EventResult::Continue,
        };

        let mut result = EventResult::Continue;

        for handler in handlers {
            let handler_result =
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| handler(payload)));

            match handler_result {
                Ok(EventResult::Continue) => {}
                Ok(EventResult::Modified) => {
                    result = EventResult::Modified;
                }
                Ok(short @ EventResult::ShortCircuit(_)) => {
                    return short;
                }
                Err(_) => {
                    warn!(
                        event = ?payload.event,
                        "event handler panicked — skipping"
                    );
                }
            }
        }

        result
    }

    /// Check if any handlers are registered for an event.
    pub fn has_handlers(&self, event: LifecycleEvent) -> bool {
        self.handlers
            .get(&event)
            .map(|h| !h.is_empty())
            .unwrap_or(false)
    }
}
