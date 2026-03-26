//! The Greatwheel engine — the framework's public API.
//!
//! Replaces the monolithic wiring in `gw-server/main.rs` with a composable
//! engine that loads plugins, wires components, and manages lifecycle.

use gw_core::{
    EventData, EventPayload, EventResult, LifecycleEvent, Plugin, PluginError, SharedState,
};
use serde_json::Value;
use std::collections::HashMap;
use tracing::info;

use crate::dispatcher::EventDispatcher;
use crate::host_fn_router::HostFnRouter;
use crate::registry::PluginRegistry;

/// Builder for the Greatwheel engine.
///
/// ```rust,ignore
/// let engine = GreatWheelEngine::new()
///     .add_plugin(MyPlugin::new())
///     .add_plugin(AnotherPlugin::new());
///
/// let running = engine.init(plugin_configs)?;
/// ```
pub struct GreatWheelEngine {
    plugins: Vec<Box<dyn Plugin>>,
    shared: SharedState,
}

impl GreatWheelEngine {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            shared: SharedState::default(),
        }
    }

    /// Add a plugin to the engine.
    pub fn add_plugin(mut self, plugin: impl Plugin) -> Self {
        self.plugins.push(Box::new(plugin));
        self
    }

    /// Add a boxed plugin to the engine.
    pub fn add_boxed_plugin(mut self, plugin: Box<dyn Plugin>) -> Self {
        self.plugins.push(plugin);
        self
    }

    /// Pre-seed a typed value into SharedState before plugin initialization.
    ///
    /// Use this to provide infrastructure (e.g., PgPool, runtime handles)
    /// that plugins need during init or in their host function handlers.
    pub fn provide<T: Send + Sync + 'static>(mut self, value: T) -> Self {
        self.shared.insert(value);
        self
    }

    /// Initialize all plugins and return the initialized engine.
    ///
    /// `plugin_configs` maps plugin name → its TOML config section.
    /// Plugins are initialized in priority order; dependencies are checked.
    pub fn init(
        self,
        plugin_configs: &HashMap<String, Value>,
    ) -> Result<InitializedEngine, PluginError> {
        let mut registry = PluginRegistry::new_with_shared(self.shared);
        registry.init_plugins(self.plugins, plugin_configs)?;

        // Build event dispatcher from registry's collected handlers.
        let mut handler_map = HashMap::new();
        for event in all_lifecycle_events() {
            let handlers = registry.handlers_for(event);
            if !handlers.is_empty() {
                handler_map.insert(event, handlers.to_vec());
            }
        }
        let dispatcher = EventDispatcher::new(handler_map);

        let host_fn_router = HostFnRouter::new(registry.host_functions().clone());

        info!(
            plugins = ?registry.plugin_names(),
            capabilities = ?registry.capabilities(),
            host_fns = ?host_fn_router.function_names(),
            "engine initialized"
        );

        Ok(InitializedEngine {
            registry,
            dispatcher,
            host_fn_router,
        })
    }
}

impl Default for GreatWheelEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// An engine with all plugins initialized and ready.
pub struct InitializedEngine {
    pub registry: PluginRegistry,
    pub dispatcher: EventDispatcher,
    pub host_fn_router: HostFnRouter,
}

impl InitializedEngine {
    /// Dispatch a lifecycle event.
    pub fn dispatch(&self, event: LifecycleEvent, data: EventData) -> EventResult {
        let mut payload = EventPayload { event, data };
        self.dispatcher.dispatch(&mut payload)
    }

    /// Dispatch BeforeStartup event.
    pub fn before_startup(&self) -> EventResult {
        self.dispatch(LifecycleEvent::BeforeStartup, EventData::Server)
    }

    /// Dispatch AfterStartup event.
    pub fn after_startup(&self) -> EventResult {
        self.dispatch(LifecycleEvent::AfterStartup, EventData::Server)
    }

    /// Dispatch BeforeShutdown event, then shutdown all plugins.
    pub fn shutdown(self) -> Vec<(String, PluginError)> {
        self.dispatch(LifecycleEvent::BeforeShutdown, EventData::Server);
        self.registry.shutdown()
    }

    /// Get the host function router.
    pub fn host_fn_router(&self) -> &HostFnRouter {
        &self.host_fn_router
    }

    /// Get the event dispatcher.
    pub fn dispatcher(&self) -> &EventDispatcher {
        &self.dispatcher
    }

    /// Get the plugin registry.
    pub fn registry(&self) -> &PluginRegistry {
        &self.registry
    }
}

/// All lifecycle events — used to iterate during init.
fn all_lifecycle_events() -> Vec<LifecycleEvent> {
    vec![
        LifecycleEvent::BeforeStartup,
        LifecycleEvent::AfterStartup,
        LifecycleEvent::BeforeShutdown,
        LifecycleEvent::SessionCreated,
        LifecycleEvent::SessionResumed,
        LifecycleEvent::SessionEvicted,
        LifecycleEvent::SessionEnded,
        LifecycleEvent::BeforeTurn,
        LifecycleEvent::AfterContextBuild,
        LifecycleEvent::BeforeLlmCall,
        LifecycleEvent::AfterLlmCall,
        LifecycleEvent::BeforeCodeExec,
        LifecycleEvent::AfterCodeExec,
        LifecycleEvent::BeforeHostCall,
        LifecycleEvent::AfterHostCall,
        LifecycleEvent::AfterTurn,
        LifecycleEvent::TurnError,
        LifecycleEvent::BeforeMemoryStore,
        LifecycleEvent::AfterMemoryRecall,
        LifecycleEvent::AgentSpawned,
        LifecycleEvent::AgentCompleted,
    ]
}
