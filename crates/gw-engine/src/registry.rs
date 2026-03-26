//! Plugin registry — collects registrations from all plugins.

use gw_core::{
    EventHandler, HostFnHandler, LifecycleEvent, Plugin, PluginContext, PluginError,
    PluginRegistrations, SharedState,
};
use serde_json::Value;
use std::collections::HashMap;
use tracing::{info, warn};

/// Holds all merged registrations from every plugin.
pub struct PluginRegistry {
    /// All loaded plugins (retained for shutdown).
    plugins: Vec<Box<dyn Plugin>>,

    /// Merged event handlers, keyed by event, ordered by plugin priority.
    event_handlers: HashMap<LifecycleEvent, Vec<EventHandler>>,

    /// Merged host function handlers.
    host_functions: HashMap<String, HostFnHandler>,

    /// Shared state across plugins.
    shared: SharedState,

    /// Capabilities provided by loaded plugins.
    capabilities: Vec<String>,
}

impl PluginRegistry {
    pub fn new() -> Self {
        Self {
            plugins: Vec::new(),
            event_handlers: HashMap::new(),
            host_functions: HashMap::new(),
            shared: SharedState::default(),
            capabilities: Vec::new(),
        }
    }

    /// Initialize all plugins in priority order.
    ///
    /// 1. Sort by manifest priority (lower first).
    /// 2. Check dependency graph.
    /// 3. Call init() on each, merging registrations.
    pub fn init_plugins(
        &mut self,
        mut plugins: Vec<Box<dyn Plugin>>,
        plugin_configs: &HashMap<String, Value>,
    ) -> Result<(), PluginError> {
        // Sort by priority (lower = earlier).
        plugins.sort_by_key(|p| p.manifest().priority);

        let empty_config = Value::Null;

        for plugin in &plugins {
            let manifest = plugin.manifest();
            let name = plugin.name();

            // Check requirements.
            for req in &manifest.requires {
                let base_req = req.split(':').next().unwrap_or(req);
                let satisfied = self.capabilities.iter().any(|cap| {
                    let base_cap = cap.split(':').next().unwrap_or(cap);
                    base_cap == base_req
                });
                if !satisfied {
                    return Err(PluginError::MissingCapability(format!(
                        "plugin '{}' requires '{}' but no loaded plugin provides it",
                        name, req
                    )));
                }
            }

            // Init the plugin.
            let config = plugin_configs.get(name).unwrap_or(&empty_config);
            let mut registrations = PluginRegistrations::default();

            let mut ctx = PluginContext::new(
                config,
                &mut self.shared,
                &mut registrations,
            );

            let init_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                plugin.init(&mut ctx)
            }));

            match init_result {
                Ok(Ok(())) => {
                    info!(plugin = name, version = plugin.version(), "plugin initialized");
                }
                Ok(Err(e)) => {
                    return Err(PluginError::Init(format!(
                        "plugin '{}' init failed: {}",
                        name, e
                    )));
                }
                Err(panic_info) => {
                    let msg = if let Some(s) = panic_info.downcast_ref::<String>() {
                        s.clone()
                    } else if let Some(s) = panic_info.downcast_ref::<&str>() {
                        s.to_string()
                    } else {
                        "unknown panic".to_string()
                    };
                    warn!(plugin = name, error = %msg, "plugin panicked during init — skipping");
                    continue;
                }
            }

            // Merge registrations.
            for (event, handlers) in registrations.event_handlers {
                self.event_handlers
                    .entry(event)
                    .or_default()
                    .extend(handlers);
            }
            for (name, handler) in registrations.host_functions {
                if self.host_functions.contains_key(&name) {
                    warn!(
                        host_fn = %name,
                        "host function already registered — overwriting"
                    );
                }
                self.host_functions.insert(name, handler);
            }

            // Track capabilities.
            self.capabilities.extend(manifest.provides.clone());
        }

        self.plugins = plugins;
        Ok(())
    }

    /// Shutdown all plugins in reverse init order.
    pub fn shutdown(&self) -> Vec<(String, PluginError)> {
        let mut errors = Vec::new();
        for plugin in self.plugins.iter().rev() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                plugin.shutdown()
            }));
            match result {
                Ok(Ok(())) => {
                    info!(plugin = plugin.name(), "plugin shut down");
                }
                Ok(Err(e)) => {
                    warn!(plugin = plugin.name(), error = %e, "plugin shutdown error");
                    errors.push((plugin.name().to_string(), e));
                }
                Err(_) => {
                    let e = PluginError::Panic("panic during shutdown".into());
                    warn!(plugin = plugin.name(), "plugin panicked during shutdown");
                    errors.push((plugin.name().to_string(), e));
                }
            }
        }
        errors
    }

    /// Get event handlers for a given event.
    pub fn handlers_for(&self, event: LifecycleEvent) -> &[EventHandler] {
        self.event_handlers
            .get(&event)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Get all registered host functions.
    pub fn host_functions(&self) -> &HashMap<String, HostFnHandler> {
        &self.host_functions
    }

    /// Get the shared state.
    pub fn shared(&self) -> &SharedState {
        &self.shared
    }

    /// Get list of loaded plugin names.
    pub fn plugin_names(&self) -> Vec<&str> {
        self.plugins.iter().map(|p| p.name()).collect()
    }

    /// Get list of provided capabilities.
    pub fn capabilities(&self) -> &[String] {
        &self.capabilities
    }
}
