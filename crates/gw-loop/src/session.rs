use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};
use std::time::{Duration, Instant};

use gw_core::SessionId;
use gw_runtime::{HostBridge, ReplAgent};
use sqlx::PgPool;
use tokio::sync::Mutex;
use tracing::info;
use uuid::Uuid;

use crate::bridge::{self, AskHandle, ConversationBridge};
use crate::conversation::{ConversationLoop, LoopConfig, SnapshotPolicy, TurnResult};
use crate::error::LoopError;
use crate::llm::LlmClient;
use crate::pg_store::PgSessionStore;
use crate::tree::SessionTree;

/// Per-session state. The loop is wrapped in Option to allow temporary
/// extraction for async operations without holding the lock.
struct ManagedSession {
    /// The conversation loop. Wrapped in StdMutex<Option<>> so we can
    /// take it out for async work without holding the lock across awaits.
    inner: StdMutex<Option<ConversationLoop>>,
    last_activity: StdMutex<Instant>,
    status: StdMutex<SessionStatus>,
    /// Shared handle for pending channel.ask() calls.
    ask_handle: AskHandle,
}

/// Session status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SessionStatus {
    Active,
    Suspended,
}

/// Clonable subset of LoopConfig for the session manager's defaults.
/// Non-cloneable fields (answer_validator, iteration_callback) are
/// provided per-session, not stored globally.
struct DefaultLoopConfig {
    system_prompt: String,
    recency_window: usize,
    max_iterations: usize,
    snapshot_policy: SnapshotPolicy,
    compaction_keep_count: usize,
    auto_compact_after_turns: Option<u32>,
    include_code_output: bool,
    repl_output_max_chars: usize,
    strip_think_tags: bool,
}

impl From<LoopConfig> for DefaultLoopConfig {
    fn from(c: LoopConfig) -> Self {
        Self {
            system_prompt: c.system_prompt,
            recency_window: c.recency_window,
            max_iterations: c.max_iterations,
            snapshot_policy: c.snapshot_policy,
            compaction_keep_count: c.compaction_keep_count,
            auto_compact_after_turns: c.auto_compact_after_turns,
            include_code_output: c.include_code_output,
            repl_output_max_chars: c.repl_output_max_chars,
            strip_think_tags: c.strip_think_tags,
        }
    }
}

impl DefaultLoopConfig {
    fn to_loop_config(&self) -> LoopConfig {
        LoopConfig {
            system_prompt: self.system_prompt.clone(),
            recency_window: self.recency_window,
            max_iterations: self.max_iterations,
            snapshot_policy: self.snapshot_policy.clone(),
            compaction_keep_count: self.compaction_keep_count,
            auto_compact_after_turns: self.auto_compact_after_turns,
            include_code_output: self.include_code_output,
            repl_output_max_chars: self.repl_output_max_chars,
            strip_think_tags: self.strip_think_tags,
            answer_validator: None,
            iteration_callback: None,
        }
    }
}

/// Manages the lifecycle of conversation sessions.
pub struct SessionManager {
    sessions: Mutex<HashMap<SessionId, Arc<ManagedSession>>>,
    llm_factory: Arc<dyn Fn() -> Box<dyn LlmClient> + Send + Sync>,
    default_config: DefaultLoopConfig,
    idle_timeout: Duration,
    /// Optional Postgres pool for session persistence.
    pg_pool: Option<PgPool>,
    /// Optional plugin host function router. When set, every
    /// `ConversationBridge` spawned by this manager gets a clone of it
    /// and will dispatch unknown host function calls through the
    /// registered plugins (async-capable via block_in_place + block_on).
    plugin_router: Option<Arc<gw_engine::HostFnRouter>>,
}

/// RAII guard that ensures the ConversationLoop is returned to the session
/// even if the async operation panics.
struct LoopGuard {
    session: Arc<ManagedSession>,
    loop_: Option<ConversationLoop>,
}

impl LoopGuard {
    fn take(session: &Arc<ManagedSession>) -> Result<Self, LoopError> {
        let loop_ = session
            .inner
            .lock()
            .unwrap()
            .take()
            .ok_or(LoopError::SessionEnded)?;
        Ok(Self {
            session: Arc::clone(session),
            loop_: Some(loop_),
        })
    }

    fn get_mut(&mut self) -> &mut ConversationLoop {
        self.loop_.as_mut().unwrap()
    }
}

impl Drop for LoopGuard {
    fn drop(&mut self) {
        if let Some(loop_) = self.loop_.take() {
            *self.session.inner.lock().unwrap() = Some(loop_);
        }
    }
}

impl SessionManager {
    pub fn new(
        llm_factory: Arc<dyn Fn() -> Box<dyn LlmClient> + Send + Sync>,
        default_config: LoopConfig,
        idle_timeout: Duration,
    ) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            llm_factory,
            default_config: default_config.into(),
            idle_timeout,
            pg_pool: None,
            plugin_router: None,
        }
    }

    /// Create a session manager with Postgres persistence.
    pub fn with_pg(
        llm_factory: Arc<dyn Fn() -> Box<dyn LlmClient> + Send + Sync>,
        default_config: LoopConfig,
        idle_timeout: Duration,
        pg_pool: PgPool,
    ) -> Self {
        Self {
            sessions: Mutex::new(HashMap::new()),
            llm_factory,
            default_config: default_config.into(),
            idle_timeout,
            pg_pool: Some(pg_pool),
            plugin_router: None,
        }
    }

    /// Attach a plugin host function router. Every `ConversationBridge`
    /// created from this point on will dispatch through the router
    /// before falling through to its legacy inner bridge.
    ///
    /// Builder-style so server startup can chain:
    /// `SessionManager::with_pg(...).with_plugin_router(router)`.
    pub fn with_plugin_router(
        mut self,
        router: Arc<gw_engine::HostFnRouter>,
    ) -> Self {
        self.plugin_router = Some(router);
        self
    }

    /// Get a session reference, or return SessionEnded.
    async fn get_session(&self, session_id: SessionId) -> Result<Arc<ManagedSession>, LoopError> {
        self.sessions
            .lock()
            .await
            .get(&session_id)
            .cloned()
            .ok_or(LoopError::SessionEnded)
    }

    /// Create a new session and return its ID.
    pub async fn create_session(&self) -> SessionId {
        let session_id = SessionId(Uuid::new_v4());
        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();

        let ask_handle = bridge::new_ask_handle();

        // Create a bridge that supports channel.ask() with blocking reply
        // and dispatches through the plugin router (if configured).
        let conv_bridge = ConversationBridge::with_plugin_router(
            event_tx.clone(),
            ask_handle.clone(),
            None,
            self.plugin_router.clone(),
        );

        // Register external functions: FINAL for turn completion,
        // ask_user and send_message for interactive I/O.
        let external_functions = vec![
            "FINAL".into(),
            "ask_user".into(),
            "send_message".into(),
            "compact_session".into(),
        ];

        let repl = ReplAgent::new(external_functions, Box::new(conv_bridge));
        let llm = (self.llm_factory)();

        let config = self.default_config.to_loop_config();

        let tree = match &self.pg_pool {
            Some(pool) => SessionTree::with_pg(session_id, PgSessionStore::new(pool.clone())),
            None => SessionTree::new(session_id),
        };

        let loop_ = ConversationLoop::with_tree(tree, repl, llm, config, event_tx);

        let managed = Arc::new(ManagedSession {
            inner: StdMutex::new(Some(loop_)),
            last_activity: StdMutex::new(Instant::now()),
            status: StdMutex::new(SessionStatus::Active),
            ask_handle,
        });

        self.sessions.lock().await.insert(session_id, managed);
        info!(?session_id, "session created");
        session_id
    }

    /// Resume a session from Postgres. Loads the tree and restores
    /// REPL state from the most recent snapshot.
    ///
    /// Returns SessionEnded if the session is not found in Postgres,
    /// or if no Postgres pool is configured.
    pub async fn resume_session(&self, session_id: SessionId) -> Result<(), LoopError> {
        let pg_pool = self.pg_pool.as_ref().ok_or(LoopError::SessionEnded)?;
        let pg_store = PgSessionStore::new(pg_pool.clone());

        // Load tree from Postgres.
        let tree = SessionTree::load_from_pg(session_id, pg_store)
            .await
            .map_err(|e| LoopError::Llm(format!("failed to load session from Postgres: {e}")))?;

        if tree.entries().is_empty() {
            return Err(LoopError::SessionEnded);
        }

        // Restore REPL from the latest snapshot on the active path.
        let repl = if let Some(leaf) = tree.active_leaf() {
            if let Some(snapshot) = tree.find_latest_snapshot(leaf) {
                if let Some(raw_bytes) = &snapshot.raw_bytes {
                    ReplAgent::restore_snapshot(raw_bytes, Box::new(NullBridge))?
                } else {
                    // Replay code executions.
                    let mut repl =
                        ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
                    let path = tree.path_to(leaf);
                    for entry in &path {
                        if let gw_core::EntryType::CodeExecution { code, .. } =
                            &entry.entry_type
                        {
                            let _ = repl.execute(code);
                        }
                    }
                    repl
                }
            } else {
                // No snapshot — replay all code.
                let mut repl = ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge));
                let path = tree.path_to(leaf);
                for entry in &path {
                    if let gw_core::EntryType::CodeExecution { code, .. } = &entry.entry_type
                    {
                        let _ = repl.execute(code);
                    }
                }
                repl
            }
        } else {
            ReplAgent::new(vec!["FINAL".into()], Box::new(NullBridge))
        };

        let (event_tx, _event_rx) = tokio::sync::mpsc::unbounded_channel();
        let llm = (self.llm_factory)();

        let config = self.default_config.to_loop_config();

        let loop_ = ConversationLoop::with_tree(tree, repl, llm, config, event_tx);

        let managed = Arc::new(ManagedSession {
            inner: StdMutex::new(Some(loop_)),
            last_activity: StdMutex::new(Instant::now()),
            status: StdMutex::new(SessionStatus::Active),
            ask_handle: bridge::new_ask_handle(),
        });

        self.sessions.lock().await.insert(session_id, managed);
        info!(?session_id, "session resumed from Postgres");
        Ok(())
    }

    /// Send a user message to a session and get the turn result.
    pub async fn send_message(
        &self,
        session_id: SessionId,
        message: &str,
    ) -> Result<TurnResult, LoopError> {
        let session = self.get_session(session_id).await?;
        *session.last_activity.lock().unwrap() = Instant::now();
        let mut guard = LoopGuard::take(&session)?;
        let loop_ = guard.get_mut();
        let result = loop_.handle_turn(message).await?;
        loop_.check_auto_compact().await;
        Ok(result)
    }

    /// Get the session tree entries.
    pub async fn get_tree(
        &self,
        session_id: SessionId,
    ) -> Result<Vec<gw_core::SessionEntry>, LoopError> {
        let session = self.get_session(session_id).await?;
        let inner = session.inner.lock().unwrap();
        let loop_ = inner.as_ref().ok_or(LoopError::SessionEnded)?;
        Ok(loop_.tree.entries().to_vec())
    }

    /// Get the REPL state summary for a session.
    pub async fn get_repl_state(
        &self,
        session_id: SessionId,
    ) -> Result<String, LoopError> {
        let session = self.get_session(session_id).await?;
        let inner = session.inner.lock().unwrap();
        let loop_ = inner.as_ref().ok_or(LoopError::SessionEnded)?;
        Ok(loop_.repl.state_summary())
    }

    /// Trigger compaction for a session.
    pub async fn compact(&self, session_id: SessionId) -> Result<(), LoopError> {
        let session = self.get_session(session_id).await?;
        let mut guard = LoopGuard::take(&session)?;
        guard.get_mut().compact().await
    }

    /// Switch branch in a session.
    pub async fn switch_branch(
        &self,
        session_id: SessionId,
        target: gw_core::EntryId,
        summarize: bool,
    ) -> Result<(), LoopError> {
        let session = self.get_session(session_id).await?;
        let mut guard = LoopGuard::take(&session)?;
        guard.get_mut().switch_branch(target, summarize).await
    }

    /// Suspend a session — snapshot REPL and mark as suspended.
    pub async fn suspend(&self, session_id: SessionId) -> Result<(), LoopError> {
        let session = self.get_session(session_id).await?;
        {
            let status = session.status.lock().unwrap();
            if *status == SessionStatus::Suspended {
                return Ok(());
            }
        }
        let mut inner = session.inner.lock().unwrap();
        if let Some(loop_) = inner.as_mut() {
            loop_.save_snapshot()?;
        }
        *session.status.lock().unwrap() = SessionStatus::Suspended;
        info!(?session_id, "session suspended");
        Ok(())
    }

    /// List active session IDs.
    pub async fn list_sessions(&self) -> Vec<(SessionId, SessionStatus)> {
        let sessions = self.sessions.lock().await;
        sessions
            .iter()
            .map(|(id, managed)| {
                let status = *managed.status.lock().unwrap();
                (*id, status)
            })
            .collect()
    }

    /// Check if a session has a pending `channel.ask()` and return the prompt.
    pub async fn get_pending_ask(
        &self,
        session_id: SessionId,
    ) -> Result<Option<String>, LoopError> {
        let session = self.get_session(session_id).await?;
        Ok(bridge::get_pending_ask(&session.ask_handle))
    }

    /// Reply to a pending `channel.ask()`. Returns true if delivered.
    pub async fn reply_to_ask(
        &self,
        session_id: SessionId,
        reply: String,
    ) -> Result<bool, LoopError> {
        let session = self.get_session(session_id).await?;
        Ok(bridge::reply_to_ask(&session.ask_handle, reply))
    }

    /// End and remove a session.
    pub async fn end_session(&self, session_id: SessionId) -> bool {
        let removed = self.sessions.lock().await.remove(&session_id).is_some();
        if removed {
            info!(?session_id, "session ended");
        }
        removed
    }

    /// Evict sessions that have been idle longer than the timeout.
    pub async fn evict_idle(&self) -> usize {
        let sessions = self.sessions.lock().await;
        let now = Instant::now();
        let timeout = self.idle_timeout;
        let mut count = 0;

        for (id, managed) in sessions.iter() {
            let status = *managed.status.lock().unwrap();
            let last = *managed.last_activity.lock().unwrap();
            if status == SessionStatus::Active && now - last > timeout {
                let mut inner = managed.inner.lock().unwrap();
                if let Some(loop_) = inner.as_mut() {
                    let _ = loop_.save_snapshot();
                }
                *managed.status.lock().unwrap() = SessionStatus::Suspended;
                info!(?id, "session evicted due to idle timeout");
                count += 1;
            }
        }
        count
    }
}

/// Null bridge for sessions that don't need host functions.
struct NullBridge;

impl HostBridge for NullBridge {
    fn call(
        &mut self,
        function: &str,
        _args: Vec<serde_json::Value>,
        _kwargs: HashMap<String, serde_json::Value>,
    ) -> Result<ouros::Object, gw_runtime::AgentError> {
        Err(gw_runtime::AgentError::UnknownFunction(
            function.to_string(),
        ))
    }
}
