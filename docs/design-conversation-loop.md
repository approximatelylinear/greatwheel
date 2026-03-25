# Design: Multi-Turn Conversation Loop

**Status:** Implemented (Phases 1-7)
**Date:** 2026-03-20 (design), 2026-03-23 (implementation complete)
**Inspiration:** [pi-mono agent-loop](https://github.com/badlogic/pi-mono) — session trees, steering messages, provider-agnostic turn engine

---

## 0. Implementation Status

All seven phases from the original plan are implemented. The system is
validated by 40 deterministic assertions (9 scenarios, 8 capabilities)
and 47 live assertions (17 scenarios, 3 capabilities) running against
qwen3.5:9b via Ollama.

### Crate layout

| Crate | Files | What's implemented |
|-------|-------|--------------------|
| `gw-core` | `session_tree.rs`, `loop_event.rs` | `EntryId`, `SessionEntry`, `EntryType` (8 variants), `ReplSnapshotData`, `SessionState`, `LoopEvent` (9 variants) |
| `gw-runtime` | `lib.rs` | `ReplAgent` extensions: `state_summary()` (shows values for scalars), `get_all_variables()`, `save_snapshot()`, `restore_snapshot()`, `get_definitions()`. `HostBridge: Send` |
| `gw-loop` | `conversation.rs` | `ConversationLoop` with `handle_turn()`, rLM inner loop, steering injection, follow-up queue, auto-snapshot, auto-compaction, OTel tracing spans |
| `gw-loop` | `tree.rs` | `SessionTree` — in-memory with optional Postgres write-through (`flush_to_pg()`), `load_from_pg()` for session resumption |
| `gw-loop` | `context.rs` | `build_turn_context()` — compaction-aware, recency window, REPL state summary injection |
| `gw-loop` | `bridge.rs` | `ConversationBridge` — `ask_user()` with blocking reply via `std::sync::mpsc`, `send_message()`, `compact_session()` |
| `gw-loop` | `session.rs` | `SessionManager` — create, message, compact, branch, suspend, resume (from Postgres), evict idle, ask/reply. `LoopGuard` RAII pattern for non-Send ouros types |
| `gw-loop` | `pg_store.rs` | `PgSessionStore` — insert/load entries, update active leaf, transactional batch insert |
| `gw-loop` | `llm.rs` | `LlmClient` trait, `OllamaLlmClient` adapter |
| `gw-loop` | `error.rs` | `LoopError` — 7 variants including `EntryNotFound`, `NoSnapshot`, `NothingToCompact` |
| `gw-channels` | `lib.rs` | Event-driven `ChannelAdapter` trait + legacy `TaskChannelAdapter` |
| `gw-server` | `session_api.rs` | 11 HTTP endpoints: create, message, tree, state, compact, branch, ask, reply, resume, list, end |
| `gw-server` | `ws_handler.rs` | WebSocket adapter — JSON protocol, auto-session, ask/reply, streaming events |

### Server endpoints

| Method | Path | Purpose |
|--------|------|---------|
| POST | `/api/sessions/create` | Create a new session |
| POST | `/api/sessions/message` | Send user message (runs rLM turn) |
| POST | `/api/sessions/tree` | Get session tree entries |
| POST | `/api/sessions/state` | Get REPL state summary |
| POST | `/api/sessions/compact` | Trigger compaction |
| POST | `/api/sessions/branch` | Switch branch |
| POST | `/api/sessions/ask` | Poll for pending `ask_user()` |
| POST | `/api/sessions/reply` | Reply to `ask_user()` |
| POST | `/api/sessions/resume` | Resume session from Postgres |
| GET | `/api/sessions` | List active sessions |
| POST | `/api/sessions/end` | End a session |
| GET | `/api/ws` | WebSocket session (auto-create, streaming) |

### Host functions available to the rLM

| Python name | Purpose | Blocking? |
|-------------|---------|-----------|
| `FINAL(value)` | End the turn with a response | No |
| `ask_user(prompt)` | Ask the user for input, block until reply | Yes |
| `send_message(text)` | Send a message to the channel | No |
| `compact_session()` | Trigger context compaction | No |

### Open questions resolved

| Question | Resolution |
|----------|-----------|
| REPL snapshot fidelity | Approach (a): serialize what we can via `get_all_variables()` + raw ouros `save()` bytes |
| Recency window size | Default 5, configurable per-scenario. Validated with window=3 in L07 (long-range reference) |
| Steering message format | `[Steering]` prefix — works well with qwen3.5 |
| Compaction trigger | Both manual (via host function or API) and auto (`auto_compact_after_turns` config) |

### Remaining work

- **Slack channel adapter** — implement `ChannelAdapter` for Slack API
- **Multi-channel multiplexing** — multiple adapters per session, channel-aware routing
- **gw-scheduler integration** — rate limiting across sessions
- **Session Postgres row creation** — `create_session()` should INSERT into `sessions` table for FK integrity

---

## 1. Context

Greatwheel's rLM executes single turns: user message → LLM emits Python →
ouros executes with host function pauses → `FINAL()` terminates. Session
state lives in Python variables across code blocks within one invocation,
but there is no persistent multi-turn conversation structure, no branching,
no mid-turn steering, and no channel-agnostic event loop.

The pi-mono codebase (agent-loop.ts / AgentSession) provides a mature
reference for multi-turn orchestration:

- **Session tree** — entries with `id`/`parentId`, active leaf determines
  context. `buildSessionContext()` walks root→leaf to reconstruct messages.
- **Turn engine** — provider-agnostic loop with `transformContext` and
  `convertToLlm` hooks, steering/follow-up message injection, tool
  execution, and incremental context mutation.
- **Branch navigation** — change the active leaf, optionally summarize the
  abandoned branch, rebuild context from the new path.
- **Compaction** — replace old history with a summary + retained suffix.

This document describes how to bring those capabilities into greatwheel,
adapted to our rLM foundation where **context lives in Python variables,
not the LLM context window**.

### 1.1 The rLM Advantage

In pi-mono, compaction and context management are complex because the full
conversation history *is* the agent's working memory. Summarizing it loses
information. The context window is the bottleneck.

In greatwheel, the rLM stores working state in ouros REPL variables. The
LLM context window only needs:

- System prompt (agent definition + available SDK functions)
- Truncated session state summary (variable names + types)
- Recent user messages (for conversational coherence)
- The current user request

This means:
- **Compaction is cheap** — snapshot REPL state, keep a short summary.
- **Branching is cheap** — restore a REPL snapshot, not replay messages.
- **Context windows scale** — adding turns doesn't bloat the LLM prompt.
- **The session tree serves conversation structure**, not LLM context
  management.

### 1.2 What We Need

| Capability | pi-mono | Greatwheel (current) | Greatwheel (proposed) |
|---|---|---|---|
| Multi-turn conversations | Session tree with branching | Single invocation | Session tree + conversation loop |
| Persistent history | Session file (JSON) | None (in-memory only) | Postgres session entries |
| Context reconstruction | `buildSessionContext()` | N/A | `build_turn_context()` |
| Mid-turn steering | Steering messages | None | Event-driven steering |
| Compaction | Summary + retained suffix | Variables persist (no need) | REPL snapshot + short summary |
| Branching | Tree navigation + branch summary | None | Tree navigation + REPL restore |
| Channel multiplexing | N/A (single UI) | HTTP only | Multi-channel via events |
| Provider abstraction | `convertToLlm()` + provider modules | Ollama only | Provider trait (future) |

---

## 2. Design

### 2.1 Session Tree

The session tree is the persistent record of a conversation. Each entry
has a parent, forming a tree rooted at the session start. The "active
leaf" determines which path the agent sees.

#### 2.1.1 Entry Types

```rust
// gw-core/src/session_tree.rs

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use serde_json::Value;
use uuid::Uuid;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EntryId(pub Uuid);

/// A single node in the session tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEntry {
    pub id: EntryId,
    pub session_id: SessionId,
    pub parent_id: Option<EntryId>,
    pub entry_type: EntryType,
    pub created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EntryType {
    /// User message from a channel.
    UserMessage {
        channel: String,
        content: String,
    },

    /// LLM response — the raw text including code blocks.
    AssistantMessage {
        content: String,
        model: String,
    },

    /// Code block extracted from an assistant message and executed.
    CodeExecution {
        code: String,
        stdout: String,
        return_value: Option<Value>,
        is_final: bool,
    },

    /// Host function call and its result.
    HostCall {
        function: String,
        args: Value,
        result: Value,
        duration_ms: u64,
    },

    /// REPL state snapshot — serialized variable state.
    /// Enables cheap branching and compaction.
    ReplSnapshot {
        /// Variable names → serialized values (JSON where possible,
        /// type+repr fallback for non-serializable objects).
        variables: HashMap<String, Value>,
        /// Python-side function definitions (source code).
        definitions: Vec<String>,
    },

    /// Compaction node — replaces all ancestors with a summary.
    Compaction {
        summary: String,
        /// First entry after this compaction that is kept verbatim.
        first_kept_id: EntryId,
        /// Snapshot of REPL state at compaction time.
        snapshot: Option<Box<ReplSnapshot>>,
    },

    /// Summary of an abandoned branch (created on branch switch).
    BranchSummary {
        /// The leaf of the abandoned branch.
        from_leaf: EntryId,
        summary: String,
    },

    /// System event — not sent to LLM, but recorded for audit.
    System {
        event: String,
        data: Option<Value>,
    },
}

/// Tracks which entry is the current active position.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionState {
    pub session_id: SessionId,
    pub active_leaf: Option<EntryId>,
    pub model: String,
}
```

#### 2.1.2 Why a Tree?

A flat transcript cannot represent:
- **Branching** — user wants to try a different approach from turn 3.
- **Compaction** — summarize turns 1-10, keep 11+, without losing the
  ability to inspect the originals.
- **Multi-channel interleaving** — messages from Slack and HTTP arriving
  at different points in the same session.

The tree makes these operations structural rather than heuristic.

#### 2.1.3 Postgres Schema

```sql
-- New migration: session_entries

CREATE TABLE session_entries (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id  UUID NOT NULL REFERENCES sessions(id),
    parent_id   UUID REFERENCES session_entries(id),
    entry_type  TEXT NOT NULL,    -- 'user_message', 'assistant_message', etc.
    content     JSONB NOT NULL,   -- type-specific payload
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_session_entries_session ON session_entries(session_id, created_at);
CREATE INDEX idx_session_entries_parent  ON session_entries(parent_id);

-- Track active leaf per session
ALTER TABLE sessions ADD COLUMN active_leaf_id UUID REFERENCES session_entries(id);
```

### 2.2 Context Builder

`build_turn_context()` reconstructs what the LLM should see for a given
turn. This is the equivalent of pi-mono's `buildSessionContext()`, but
adapted for the rLM pattern.

```rust
// gw-runtime/src/context.rs

/// What the LLM needs to produce the next code block.
pub struct TurnContext {
    /// Messages for the LLM prompt.
    pub messages: Vec<LlmMessage>,
    /// Variables to inject into the REPL before execution.
    pub variables: HashMap<String, Value>,
    /// Function definitions to re-evaluate in the REPL.
    pub definitions: Vec<String>,
    /// The model to use for this turn.
    pub model: String,
}

/// Reconstruct context from the session tree.
pub fn build_turn_context(
    entries: &[SessionEntry],  // root → leaf path
    system_prompt: &str,
    repl_state_summary: &str,  // current variable names + types
) -> TurnContext {
    // 1. Walk entries from root to leaf.
    // 2. If a Compaction entry is found:
    //    - Use its snapshot to populate variables/definitions.
    //    - Use its summary as the first message.
    //    - Skip all entries before first_kept_id.
    // 3. If a ReplSnapshot is found (most recent wins):
    //    - Use it for variables/definitions instead of replaying code.
    // 4. Collect recent UserMessage and AssistantMessage entries
    //    as LLM messages (for conversational coherence).
    // 5. BranchSummary entries become assistant context messages.
    // 6. CodeExecution and HostCall entries are NOT sent to the LLM
    //    (the REPL variables already reflect their results).
    // 7. Apply a recency window: keep last N user/assistant pairs.
    //    Default N=5. Older turns are represented only by the
    //    REPL state they produced.
}
```

**Key difference from pi-mono:** In pi-mono, `buildSessionContext()`
reconstructs the full message chain because messages *are* the context.
Here, we reconstruct a **thin message chain** (recent turns only) plus
**REPL state** (variables + definitions). The rLM's working memory is
in the REPL, not the message history.

#### 2.2.1 What the LLM Sees

For a 20-turn conversation, the LLM prompt looks like:

```
System: {system_prompt}
         Available variables: {repl_state_summary}

[If compacted] Assistant: "Summary of earlier work: ..."

User (turn 18): "Can you also check the auth module?"
Assistant (turn 18): "```python\nauth_issues = analyze('auth')...\n```"

User (turn 19): "What about the rate limiter?"
Assistant (turn 19): "```python\nrl_issues = analyze('rate_limiter')...\n```"

User (turn 20): "Compare auth and rate limiter findings."
```

Turns 1-17 are not in the prompt. Their results live in REPL variables
(`auth_issues`, `rl_issues`, etc.) which the rLM can reference in code.

### 2.3 Conversation Loop

The conversation loop is the event-driven core that replaces the current
single-turn execution model.

#### 2.3.1 Events

```rust
// gw-core/src/loop_event.rs

/// Events that drive the conversation loop.
#[derive(Debug, Clone)]
pub enum LoopEvent {
    // === Inbound (from channels or internal) ===

    /// New user message from a channel.
    UserMessage {
        channel: String,
        content: String,
        /// If set, this message steers the current turn rather than
        /// starting a new one.
        steering: bool,
    },

    /// A follow-up message injected after the current turn would
    /// otherwise end.
    FollowUp {
        content: String,
    },

    // === Outbound (emitted by rLM via host functions) ===

    /// Send a response back to a channel.
    Response {
        channel: String,
        content: String,
    },

    /// The rLM is asking the user for input (blocks until reply).
    InputRequest {
        channel: String,
        prompt: String,
    },

    /// A host function was called (for event listeners).
    HostCallCompleted {
        function: String,
        duration_ms: u64,
    },

    // === Control ===

    /// Switch the active branch to a different entry.
    SwitchBranch {
        target: EntryId,
        summarize_abandoned: bool,
    },

    /// Compact the session — snapshot + summarize.
    Compact,

    /// The rLM called FINAL() — this turn is complete.
    TurnComplete {
        response: Option<String>,
    },

    /// Session-level termination.
    SessionEnd,
}
```

#### 2.3.2 The Loop

```rust
// gw-loop/src/lib.rs (new crate)

pub struct ConversationLoop {
    session_id: SessionId,
    tree: SessionTree,        // Postgres-backed
    repl: ReplAgent,          // ouros session
    bridge: Box<dyn HostBridge>,
    event_tx: mpsc::Sender<LoopEvent>,
    event_rx: mpsc::Receiver<LoopEvent>,
}

impl ConversationLoop {
    /// Main loop — runs until SessionEnd.
    pub async fn run(&mut self) -> Result<(), LoopError> {
        loop {
            let event = self.event_rx.recv().await?;

            match event {
                LoopEvent::UserMessage { channel, content, steering } => {
                    if steering {
                        self.inject_steering(&content).await?;
                    } else {
                        self.handle_turn(&channel, &content).await?;
                    }
                }

                LoopEvent::SwitchBranch { target, summarize_abandoned } => {
                    self.switch_branch(target, summarize_abandoned).await?;
                }

                LoopEvent::Compact => {
                    self.compact().await?;
                }

                LoopEvent::SessionEnd => break,

                // Outbound events are handled by channel adapters
                // listening on event_tx.subscribe()
                _ => {}
            }
        }
        Ok(())
    }

    /// Execute a single user turn.
    async fn handle_turn(&mut self, channel: &str, content: &str) -> Result<(), LoopError> {
        // 1. Append UserMessage entry to session tree.
        let entry_id = self.tree.append(EntryType::UserMessage {
            channel: channel.to_string(),
            content: content.to_string(),
        }).await?;

        // 2. Build context from tree + REPL state.
        let path = self.tree.path_to_leaf().await?;
        let repl_summary = self.repl.state_summary().await?;
        let context = build_turn_context(&path, &self.system_prompt, &repl_summary);

        // 3. Inject any pending steering messages.
        let mut messages = context.messages;
        messages.extend(self.drain_steering());

        // 4. Call LLM.
        let response = self.llm.chat_with_options(
            messages,
            Some(&context.model),
            None,
        ).await?;

        // 5. Append AssistantMessage entry.
        self.tree.append(EntryType::AssistantMessage {
            content: response.content.clone(),
            model: context.model.clone(),
        }).await?;

        // 6. Extract and execute code blocks.
        let code_blocks = extract_code_blocks(&response.content);
        for code in &code_blocks {
            let result = self.repl.execute(code).await?;

            // 6a. Append CodeExecution entry.
            self.tree.append(EntryType::CodeExecution {
                code: code.clone(),
                stdout: result.stdout.clone(),
                return_value: result.value.clone(),
                is_final: result.is_final,
            }).await?;

            // 6b. Host function calls are recorded individually
            //     (the bridge emits HostCall entries via tree handle).

            // 6c. Check for FINAL().
            if result.is_final {
                if let Some(answer) = result.final_value {
                    self.event_tx.send(LoopEvent::Response {
                        channel: channel.to_string(),
                        content: answer,
                    }).await?;
                }
                self.event_tx.send(LoopEvent::TurnComplete {
                    response: result.final_value,
                }).await?;

                // Check for follow-up messages before truly ending.
                if let Some(follow_up) = self.drain_follow_ups().await {
                    return self.handle_turn(channel, &follow_up.content).await;
                }
                return Ok(());
            }
        }

        // 7. If no FINAL(), continue the rLM loop — the LLM needs
        //    to see execution results and produce more code.
        //    This is the inner rLM loop (same as today's single-turn).
        self.continue_rlm_loop(channel, &response.content).await
    }
}
```

#### 2.3.3 Steering and Follow-Up Messages

Adapted from pi-mono's `getSteeringMessages()` / `getFollowUpMessages()`:

- **Steering messages** arrive while the rLM is mid-turn. They are
  injected into the LLM context before the next code generation step.
  Use case: user sends "actually, focus on the auth module" while the
  agent is iterating.

- **Follow-up messages** are queued and only processed when the current
  turn would otherwise end (after `FINAL()`). Use case: user types
  several messages in quick succession; the first triggers a turn, the
  rest queue as follow-ups.

```rust
impl ConversationLoop {
    fn inject_steering(&mut self, content: &str) -> Result<(), LoopError> {
        self.pending_steering.push(LlmMessage {
            role: "user".into(),
            content: format!("[Steering] {content}"),
        });
        Ok(())
    }

    fn drain_steering(&mut self) -> Vec<LlmMessage> {
        std::mem::take(&mut self.pending_steering)
    }

    async fn drain_follow_ups(&mut self) -> Option<FollowUp> {
        // Non-blocking check for queued follow-up events.
        match self.event_rx.try_recv() {
            Ok(LoopEvent::UserMessage { content, .. }) => {
                Some(FollowUp { content })
            }
            Ok(LoopEvent::FollowUp { content }) => {
                Some(FollowUp { content })
            }
            _ => None,
        }
    }
}
```

### 2.4 Host Functions as the Event Bridge

The existing `HostBridge` trait is the rLM's only interface to the outside
world. The conversation loop extends it with new functions that emit events.

```rust
// Extended host functions available to the rLM:

// === Existing ===
llm.complete(system, messages, model?)     // LLM call
memory.store(key, value)                   // persist to gw-memory
memory.recall(query, top_k?, mode?)        // query gw-memory
agent.call(agent_id, task, context?)       // synchronous sub-agent
agent.notify(agent_id, message)            // fire-and-forget

// === New: conversation-aware ===
channel.send(message)                      // → emits Response event
channel.send(message, channel="slack")     // → emits Response to specific channel
channel.ask(prompt)                        // → emits InputRequest, pauses until reply
session.snapshot()                         // → saves ReplSnapshot entry
session.compact(instructions?)             // → triggers Compaction
session.branch(label?)                     // → creates new branch from current point

// === New: turn control ===
turn.thinking(status)                      // → emits streaming status to channel
turn.yield_()                              // → pause, let steering messages arrive
```

The bridge implementation translates these into `LoopEvent`s:

```rust
impl HostBridge for ConversationBridge {
    fn call(&mut self, function: &str, args: Vec<Value>, kwargs: HashMap<String, Value>)
        -> Result<Object, AgentError>
    {
        match function {
            "channel.send" => {
                let message = args[0].as_str().unwrap();
                let channel = kwargs.get("channel")
                    .and_then(|v| v.as_str())
                    .unwrap_or(&self.default_channel);
                self.event_tx.blocking_send(LoopEvent::Response {
                    channel: channel.to_string(),
                    content: message.to_string(),
                })?;
                Ok(Object::None)
            }

            "channel.ask" => {
                let prompt = args[0].as_str().unwrap();
                self.event_tx.blocking_send(LoopEvent::InputRequest {
                    channel: self.default_channel.clone(),
                    prompt: prompt.to_string(),
                })?;
                // Block until the channel adapter sends a reply.
                let reply = self.reply_rx.blocking_recv()?;
                Ok(Object::String(reply))
            }

            "session.snapshot" => {
                let vars = self.repl.get_all_variables()?;
                let defs = self.repl.get_definitions()?;
                self.tree.append(EntryType::ReplSnapshot {
                    variables: vars,
                    definitions: defs,
                })?;
                Ok(Object::None)
            }

            // ... existing functions (llm.complete, memory.recall, etc.)
            _ => self.inner_bridge.call(function, args, kwargs),
        }
    }
}
```

### 2.5 Channel Multiplexing

Each channel adapter converts its protocol into `LoopEvent`s and listens
for outbound events.

```rust
// gw-channels/src/lib.rs

#[async_trait]
pub trait ChannelAdapter: Send + Sync {
    /// Unique channel identifier (e.g., "http", "ws", "slack-C04N8BKRM").
    fn channel_id(&self) -> &str;

    /// Start listening. Send inbound events to event_tx.
    async fn start(
        &self,
        session_id: SessionId,
        event_tx: mpsc::Sender<LoopEvent>,
    ) -> Result<(), ChannelError>;

    /// Handle an outbound event (Response, InputRequest, etc.).
    async fn handle_outbound(&self, event: &LoopEvent) -> Result<(), ChannelError>;
}
```

A single `ConversationLoop` can have multiple channel adapters connected.
The session tree records which channel each message came from, so
responses can be routed back correctly.

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  HTTP / SSE  │     │  WebSocket   │     │    Slack      │
│   Adapter    │     │   Adapter    │     │   Adapter     │
└──────┬───────┘     └──────┬───────┘     └──────┬───────┘
       │                    │                    │
       │    LoopEvent       │    LoopEvent       │    LoopEvent
       ▼                    ▼                    ▼
┌─────────────────────────────────────────────────────────┐
│                    ConversationLoop                      │
│                                                         │
│  event_rx ──→ match event ──→ handle_turn()             │
│                                   │                     │
│                              build_turn_context()       │
│                                   │                     │
│                              LLM call                   │
│                                   │                     │
│                              REPL execute               │
│                                   │                     │
│                              emit Response/             │
│                              InputRequest               │
│                                   │                     │
│  event_tx ◄──────────────────────┘                     │
│       │                                                 │
└───────┼─────────────────────────────────────────────────┘
        │
        ▼ (broadcast to all adapters)
   handle_outbound()
```

### 2.6 Compaction

Compaction replaces old history with a summary. In pi-mono this is
critical because old messages consume context window. In greatwheel it's
less urgent — variables carry state — but still valuable for:

- Reducing Postgres storage for long-lived sessions.
- Providing a readable summary when switching branches.
- Keeping the session tree navigable.

```rust
impl ConversationLoop {
    async fn compact(&mut self) -> Result<(), LoopError> {
        // 1. Snapshot current REPL state.
        let snapshot = ReplSnapshot {
            variables: self.repl.get_all_variables().await?,
            definitions: self.repl.get_definitions().await?,
        };

        // 2. Determine what to compact (entries before a cutoff).
        let path = self.tree.path_to_leaf().await?;
        let keep_count = 5; // keep last 5 user/assistant pairs
        let (compact_entries, kept_entries) = split_at_recency(&path, keep_count);

        if compact_entries.is_empty() {
            return Ok(());
        }

        // 3. Generate summary of compacted entries.
        let summary = self.summarize_entries(&compact_entries).await?;

        // 4. Append Compaction entry.
        self.tree.append(EntryType::Compaction {
            summary,
            first_kept_id: kept_entries[0].id,
            snapshot: Some(Box::new(snapshot)),
        }).await?;

        Ok(())
    }
}
```

### 2.7 Branch Navigation

When the user wants to try a different approach, they switch branches.
This is equivalent to pi-mono's `/tree` command.

```rust
impl ConversationLoop {
    async fn switch_branch(
        &mut self,
        target: EntryId,
        summarize_abandoned: bool,
    ) -> Result<(), LoopError> {
        let old_leaf = self.tree.active_leaf();

        // 1. Optionally summarize the abandoned branch.
        if summarize_abandoned {
            let abandoned_path = self.tree.path_from_common_ancestor(
                old_leaf, target
            ).await?;
            let summary = self.summarize_entries(&abandoned_path).await?;
            self.tree.append_at(target, EntryType::BranchSummary {
                from_leaf: old_leaf,
                summary,
            }).await?;
        }

        // 2. Set new active leaf.
        self.tree.set_active_leaf(target).await?;

        // 3. Rebuild REPL state from the new branch.
        let path = self.tree.path_to_leaf().await?;

        // Find the most recent ReplSnapshot on this path.
        if let Some(snapshot) = find_latest_snapshot(&path) {
            self.repl.restore_snapshot(snapshot).await?;
        } else {
            // No snapshot — must replay code executions from root.
            // This is expensive; auto-snapshot should prevent it.
            self.repl.reset().await?;
            for entry in &path {
                if let EntryType::CodeExecution { code, .. } = &entry.entry_type {
                    self.repl.execute(code).await?;
                }
            }
        }

        Ok(())
    }
}
```

### 2.8 Auto-Snapshot Policy

To make branching cheap, the loop should periodically snapshot REPL state.
This avoids expensive code replay when switching branches.

```rust
/// Snapshot policy — when to auto-save REPL state.
pub struct SnapshotPolicy {
    /// Snapshot every N user turns.
    pub every_n_turns: u32,         // default: 3
    /// Snapshot after N host function calls in a turn.
    pub after_n_host_calls: u32,    // default: 10
    /// Always snapshot before compaction.
    pub before_compaction: bool,    // default: true
}
```

The loop checks the policy after each turn completes and appends a
`ReplSnapshot` entry if warranted. Users can also trigger snapshots
explicitly via `session.snapshot()`.

---

## 3. Interaction with Existing Architecture

### 3.1 Where the New Crate Fits

```
┌─────────────────────────────────────────────────────────┐
│                   Channel Layer                          │
│   HTTP · WebSocket · CLI · Slack                        │
│   (gw-channels — ChannelAdapter impls)                  │
└──────────────────────┬──────────────────────────────────┘
                       │ LoopEvent
┌──────────────────────▼──────────────────────────────────┐
│               Conversation Loop (NEW)                    │
│               gw-loop crate                              │
│                                                          │
│  ConversationLoop                                        │
│  ├── SessionTree (Postgres-backed)                       │
│  ├── build_turn_context()                                │
│  ├── Steering / follow-up queues                         │
│  ├── Compaction + branch navigation                      │
│  └── Event dispatch                                      │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Agent Runtime                            │
│  gw-runtime (existing)                                   │
│                                                          │
│  ReplAgent (ouros session)                               │
│  └── HostBridge → ConversationBridge (extended)          │
└──────────────────────┬──────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────┐
│                  Core Services (existing)                 │
│  gw-llm · gw-memory · gw-bus · gw-trace                │
└─────────────────────────────────────────────────────────┘
```

### 3.2 Dependency Graph

```
gw-loop
├── gw-core         (types: SessionEntry, EntryType, LoopEvent, etc.)
├── gw-runtime      (ReplAgent, HostBridge)
├── gw-llm          (LLM calls for turn execution + compaction summaries)
├── gw-memory       (optional — for episode integration)
├── gw-trace        (span creation for turn-level tracing)
├── sqlx            (Postgres session tree persistence)
└── tokio           (mpsc channels, async runtime)
```

### 3.3 Changes to Existing Crates

| Crate | Change | Scope |
|---|---|---|
| `gw-core` | Add `SessionEntry`, `EntryType`, `LoopEvent`, `SnapshotPolicy` types | Additive |
| `gw-runtime` | Add `state_summary()`, `get_all_variables()`, `get_definitions()`, `restore_snapshot()` to `ReplAgent` | Additive |
| `gw-runtime` | `ConversationBridge` wrapping existing `HostBridge` with event emission | Additive |
| `gw-channels` | `ChannelAdapter` trait gains `handle_outbound()` method | Breaking (trait change) |
| `gw-server` | Wire up `ConversationLoop` instead of direct `chat_stream()` | Rewrite of chat endpoint |

### 3.4 Interaction with Episodes

The conversation loop and the episode system (see `design-episodes.md`)
are complementary:

- **Episodes** summarize *sub-agent* work — what a child did.
- **Compaction** summarizes *conversation history* — what happened in
  this session.
- **Branch summaries** summarize *abandoned paths* — what was tried and
  discarded.

When a sub-agent completes via `agent.call()`, its episode is both
stored in `gw-memory` (for future recall) and appended to the session
tree as part of the `HostCall` entry's result.

---

## 4. Session Lifecycle

### 4.1 Session Creation

```
Channel receives first message for (org, user, agent)
  → Create Session row in Postgres
  → Create ouros ReplAgent
  → Create ConversationLoop
  → Register channel adapter(s)
  → Start loop
```

### 4.2 Session Resumption

```
Channel receives message for existing session
  → Load Session from Postgres
  → Load session_entries (root → active_leaf path)
  → Restore ReplAgent from most recent ReplSnapshot
    (or replay code from root if no snapshot)
  → Create ConversationLoop with restored state
  → Register channel adapter(s)
  → Send LoopEvent::UserMessage
```

### 4.3 Session Idle Eviction

```
Session idle > org.session_idle_timeout
  → Auto-snapshot REPL state (ReplSnapshot entry)
  → Drop ouros session from memory
  → Set session.status = 'suspended'
  → Next message triggers Session Resumption
```

### 4.4 Session Tree Exploration

Expose the tree structure via API for UIs that want branch navigation:

```
GET /api/sessions/{id}/tree
  → Returns tree structure with entry types, timestamps, previews

POST /api/sessions/{id}/branch
  { "target": entry_id, "summarize": true }
  → Emits LoopEvent::SwitchBranch
```

---

## 5. Tracing Integration

New span hierarchy for conversation-level tracing:

```
conversation.turn{session_id, turn_number, channel}
├── conversation.build_context{entries_count, has_compaction}
├── gen_ai.chat{model, input_tokens, output_tokens}  (LLM call)
├── repl.execute{code_length}                         (code execution)
│   ├── host_function{function="search"}
│   ├── host_function{function="llm_query"}
│   │   └── gen_ai.chat{...}
│   └── host_function{function="channel.send"}
├── repl.execute{...}                                 (more code blocks)
└── conversation.turn_complete{is_final, has_follow_up}

conversation.compact{session_id, compacted_count, summary_tokens}
├── gen_ai.chat{...}  (summarization call)
└── repl.snapshot{variable_count}

conversation.branch{session_id, from_leaf, to_leaf}
├── gen_ai.chat{...}  (branch summarization, optional)
└── repl.restore{snapshot_id}
```

---

## 6. Implementation Plan

| Phase | Change | Crate(s) | Status |
|---|---|---|---|
| 1 | `SessionEntry`, `EntryType`, `EntryId` types | `gw-core` | **Done** |
| 1 | `LoopEvent` enum | `gw-core` | **Done** |
| 1 | `SnapshotPolicy` type | `gw-loop` | **Done** |
| 1 | Postgres migration for `session_entries` table | `migrations/` | **Done** |
| 2 | `SessionTree` — append, path_to_leaf, set_active_leaf | `gw-loop` | **Done** |
| 2 | `build_turn_context()` | `gw-loop` | **Done** |
| 2 | `ReplAgent` extensions — `state_summary()`, snapshot/restore | `gw-runtime` | **Done** |
| 3 | `ConversationLoop::handle_turn()` — basic turn execution | `gw-loop` | **Done** |
| 3 | `ConversationBridge` — event-emitting `HostBridge` wrapper | `gw-loop` | **Done** |
| 4 | Steering and follow-up message injection | `gw-loop` | **Done** |
| 4 | Compaction — snapshot + summarize + prune | `gw-loop` | **Done** |
| 4 | Branch navigation — switch + summarize + restore | `gw-loop` | **Done** |
| 5 | `ChannelAdapter` trait update + WebSocket adapter | `gw-channels`, `gw-server` | **Done** |
| 5 | `gw-server` session endpoints | `gw-server` | **Done** (11 REST + 1 WebSocket) |
| 6 | Auto-snapshot policy | `gw-loop` | **Done** |
| 6 | Auto-compaction policy | `gw-loop` | **Done** (`auto_compact_after_turns`) |
| 6 | Session lifecycle — create/resume/evict | `gw-loop` | **Done** (with Postgres persistence) |
| 6 | Tracing spans for conversation-level events | `gw-loop` | **Done** |
| 6 | `channel.ask()` blocking reply | `gw-loop`, `gw-server` | **Done** (`ask_user()` + `std::sync::mpsc`) |
| 7 | Session tree API (`/tree`, `/branch`) | `gw-server` | **Done** |
| 7 | Postgres write-through for session entries | `gw-loop` | **Done** (`PgSessionStore`, `flush_to_pg()`) |
| 7 | Slack channel adapter | `gw-channels` | Not started |

Phases 1-2 are type definitions and tree operations — safe, testable in
isolation. Phase 3 is the core loop. Phases 4-6 add the features that
make multi-turn conversations robust. Phase 7 is channel expansion.

---

## 7. Open Questions

**REPL snapshot fidelity.** Ouros variables may include non-serializable
Python objects (file handles, generators, custom classes). Options:
(a) serialize what we can, repr the rest; (b) require agents to use
JSON-serializable state; (c) only snapshot on explicit `session.snapshot()`
calls where the agent ensures serializability. Start with (a) and see
what breaks.

**Recency window size.** How many recent user/assistant pairs should
`build_turn_context()` include? Too few and the rLM loses conversational
thread; too many wastes context window. Default 5, make configurable.
May need to adapt based on model context length.

**Steering message format.** Should steering messages be visually
distinguished in the LLM prompt (e.g., `[Steering] ...`)? Or should
they look like regular user messages? The `[Steering]` prefix helps the
rLM understand the message arrived mid-turn, but it's a prompt
engineering choice that may vary by model.

**Channel routing for responses.** If a user sends a message via Slack
but the agent's response goes to HTTP, that's confusing. Default: respond
on the same channel the most recent user message came from. But the rLM
should be able to override via `channel.send(msg, channel="slack")`.

**Compaction trigger.** Should compaction be automatic (after N turns)?
Manual only? Or based on context window usage? Start manual + auto after
20 turns, make configurable.

**Relationship to gw-scheduler.** The conversation loop handles
per-session orchestration. The scheduler handles cross-session
prioritization and rate limiting. They interact at session
creation/resumption: the scheduler decides *whether* to start a turn,
the loop handles *how* to execute it.

---

## 8. Comparison with pi-mono

| Aspect | pi-mono | Greatwheel |
|---|---|---|
| Context lives in | LLM message history | Python REPL variables |
| Compaction cost | High (summarize messages, lose detail) | Low (snapshot REPL + short summary) |
| Branching cost | Replay full message chain | Restore REPL snapshot + recent messages |
| Provider abstraction | `convertToLlm()` + provider modules | Less critical (Ollama proxy normalizes) |
| `transformContext` | Core hook for pruning/rewriting | Mostly unnecessary — REPL carries state |
| Session persistence | JSON file | Postgres (multi-tenant, queryable) |
| Tool execution | LLM emits tool_use → host executes | LLM emits Python → ouros executes → host functions |
| Multi-channel | N/A (single UI) | First-class via ChannelAdapter |
| Tree visualization | `/tree` command in UI | `/tree` API endpoint |
| Fine-tuning | None | rl-play captures all LLM calls |

The fundamental difference: pi-mono fights the context window. Greatwheel
sidesteps it. The session tree serves conversation structure and auditability,
not LLM context management.

---

## 9. References

- [pi-mono agent-loop.ts](https://github.com/badlogic/pi-mono/blob/main/packages/agent/src/agent-loop.ts) — Turn engine
- [pi-mono agent-session.ts](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/src/core/agent-session.ts) — Session management
- [pi-mono session.md](https://github.com/badlogic/pi-mono/blob/main/packages/coding-agent/docs/session.md) — Session tree documentation
- [Recursive Language Models](https://alexzhang13.github.io/blog/2025/rlm/) — Alex Zhang
- [Slate: moving beyond ReAct and RLM](https://randomlabs.ai/blog/slate) — Random Labs
- [Greatwheel ARCHITECTURE.md](../ARCHITECTURE.md) — Current architecture
- [Greatwheel design-episodes.md](design-episodes.md) — Episode system design
