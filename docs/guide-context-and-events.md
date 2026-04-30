# Guide: Conversation history, LLM context, and the event system

**Status:** Reference, 2026-04-29.
**Audience:** anyone touching `gw-loop`, the AG-UI adapter, or any
example that emits user-facing prose. Companion to
[`design-conversation-loop.md`](design-conversation-loop.md), which
covers the broader rLM design; this doc is narrowly focused on
**how history is stored, how the LLM message chain is built each turn,
and how the event system relates to both**.

## 1. The two flows

Every running `ConversationLoop` owns two `mpsc` channels and one
in-memory tree. They form two independent flows that intersect only
at the loop's `handle_turn` method:

```
                    ┌──────────────────────┐
HTTP POST /messages ─┤  inbound event_rx   ├──→ ConversationLoop::run()
HTTP POST /widget-events ↑                 │      │
                    │                      │      ├─ tree.append(...)
                    │                      │      ├─ flush_tree → PG
                    │                      │      └─ event_tx.send(...)
                    └──────────────────────┘                ↓
                                                   ┌────────────────────┐
                                                   │ outbound event_tx  │
                                                   └─────────┬──────────┘
                                                             ↓
                                                   AG-UI codec projection
                                                             ↓
                                                       SSE to client

build_turn_context_with_opts(path = tree.path_to_leaf(), …)
                ↑
         reads ONLY the tree, never the event stream
```

Both channels carry the same `LoopEvent` enum (`crates/gw-core/src/loop_event.rs`),
but each direction uses a different subset of variants:

| Direction | Variants used                                                                                             | Driven by                |
|-----------|-----------------------------------------------------------------------------------------------------------|--------------------------|
| Inbound   | `UserMessage`, `WidgetInteraction`, `FollowUp`, `Compact`, `SwitchBranch`, `SessionEnd`                   | HTTP entry points        |
| Outbound  | `TextMessageStart/Delta/End`, `TurnStarted/Complete/Error`, `CodeExecuted`, `HostCallStarted/Args/Completed`, `WidgetEmitted/Superseded`, `InputRequest`, `UserMessageAnchor`, `SpineEntryExtracted`, `SpineSegmentsUpdated` | The loop, fired during turns |

The split is convention — they're typed the same so the loop's `run()`
can match against the inbound subset without ceremony, and the codec
can match against the outbound subset.

## 2. The session tree (the truth)

The source of truth for "what's happened in this session" is the
`SessionTree` (`crates/gw-loop/src/tree.rs`), a tree of `SessionEntry`
rows. Every turn appends one or more entries. Entries persist to
Postgres on `flush_tree` and are reloaded on session resume.

`EntryType` (`crates/gw-core/src/session_tree.rs`) enumerates what can
be in history:

| Variant | Carries | Created by |
|---|---|---|
| `UserMessage(String)` | the user's text | typed message or synthetic widget-event prompt |
| `AssistantMessage { content, model }` | the LLM's **raw** response (often prose + Python code fences) | every iteration of the rLM loop |
| `CodeExecution { code, stdout, result }` | what the REPL did with one extracted code block | each successful execution |
| `AssistantNarration { content }` | the resolved `FINAL("...")` prose the user actually saw | end-of-turn, when there's a non-empty response |
| `Compaction { summary, first_kept_id, snapshot }` | a checkpoint marker | `compact()` |
| `BranchSummary(String)` | narration injected when switching branches | `switch_branch()` |
| `HostCall { function, args, result }` | a host-fn invocation record | optional; not used by all bridges |
| `ReplSnapshot(ReplSnapshotData)` | REPL state checkpoint | snapshot policy |
| `System(String)` | internal-only marker | rare, internal use |

Persistence (`crates/gw-loop/src/pg_store.rs`) maps each variant to a
short `entry_type` string in `session_entries.entry_type` (e.g.
`"user_message"`, `"assistant_message"`, `"assistant_narration"`),
with the variant's payload as JSONB in `content`.

## 3. Building the LLM message chain

`build_turn_context_with_opts` (`crates/gw-loop/src/context.rs:43`)
projects the tree's path-to-leaf into a flat `Vec<LlmMessage>`. The
result has roles `system | user | assistant` only — no `tool` role.

### 3.1 The projection

```
for entry in path:
  match entry.entry_type:
    UserMessage(s)                          → {role: "user",      content: s}
    AssistantMessage{content, model}        → {role: "assistant", content}
                                              (also captures model)
    CodeExecution{stdout, result}           → {role: "user",      content: "REPL output:\n```\n…\n```"}
                                              (only when opts.include_code_output)
    BranchSummary(s)                        → {role: "system",    content: "[Branch summary] " + s}
    everything else                         → skipped
```

**Tool I/O round-trips as `assistant` (the code) + `user` (the output).**
Agents call Python in fenced code blocks; the runtime executes those
blocks and feeds stdout/result back as a user message framed as REPL
output. There is no `tool` role on the wire to the model.

Two reasons baked into this design:

1. **The tool *is* code.** The action is a Python block, full stop.
   Modeling it as text-in-fence keeps the model + runtime + serialization
   uniform and works with any chat-completion endpoint.
2. **Many providers we care about don't really do `tool` well** —
   local models via Ollama, older `chat` endpoints, etc.

The trade-off is that the model has to recognize "this user message
starting with `REPL output:` is something I just produced, not
something the human said." The system prompt establishes that
contract.

### 3.2 `AssistantNarration` is intentionally skipped

Narration entries are persisted (so the spine extractor and the
frontend's `data-entry-id` anchoring can use them) but they're
**omitted from LLM context**. The raw `AssistantMessage` already
contains the `FINAL("...")` call (literal or f-string template) that
produced the narration; including the narration too would double-count
what the model said. See `context.rs:120-125` for the explicit skip.

### 3.3 Compaction

Before the projection, `build_turn_context_with_opts` walks `path` in
reverse looking for the most recent `Compaction` entry
(`context.rs:55-66`). If it finds one, the entry's `summary` is
remembered and `first_kept_id` becomes a boundary — entries before
that boundary are silently dropped from the second pass. Their
content is represented by the summary instead, appended to the system
message later.

### 3.4 Recency window

If `recency_window > 0`, the projected messages get trimmed from the
front to keep only the last N user/assistant pairs (`context.rs:130`).
Older content that survived compaction can still be cut here.

### 3.5 System-message assembly

After projecting and trimming, the final system message is built
(`context.rs:149`):

```
[system_prompt configured on the loop]
\n\n[Conversation summary]\n[summary]   (if a Compaction was found)
\n\n[REPL state]\n[repl_state_summary]  (if non-empty)
```

So the `system` role isn't static — it grows with the long-term state
the model needs to remember, and contracts when older context gets
compacted into a summary line.

### 3.6 Steering messages

`pending_steering: Vec<LlmMessage>` lives on the loop. Bench harnesses
push into it; `handle_turn` drains it and **appends to the *built*
context** (`conversation.rs`, after the `build_turn_context_with_opts`
call), then clears it. Steering messages don't become entries — they
don't persist, don't reappear next turn, don't affect the spine.
Pure one-shot prompt nudges.

### 3.7 Final shape

`[system, user, assistant, user, assistant, …]`, with no role
interleaving beyond U/A/S. The `system` carries any compaction
summary + REPL state inline. Tool I/O is encoded as fenced code in
assistant turns and "REPL output:" framing in subsequent user turns.

## 4. The inbound event side

`ConversationLoop::run(event_rx)` (`conversation.rs:264`) is a single
match-and-dispatch loop. HTTP entry points (the AG-UI adapter's
`post_message` and `post_widget_event` handlers) construct an inbound
`LoopEvent` and send it on `event_rx`:

| Inbound `LoopEvent`         | Effect |
|------------------------------|--------|
| `UserMessage(content)`       | `run_input_turn(content)` → `handle_turn` → appends `EntryType::UserMessage` |
| `WidgetInteraction(event)`   | Translated to a user prompt (spine action menu items get a templated synthetic prompt via `translate_spine_action`; everything else uses `event.to_user_message()`), then `run_input_turn` |
| `FollowUp(content)`          | Queued via `queue_follow_up` to run after the current turn |
| `Compact`                    | `compact()` — appends `EntryType::Compaction` |
| `SwitchBranch(id)`           | `switch_branch()` — appends `EntryType::BranchSummary` and resets the active leaf |
| `SessionEnd`                 | Exits `run()` |
| Everything else              | Explicit no-op match arm — those variants exist for the *outbound* direction; if one shows up on the inbound channel it's just dropped. |

Widget events deserve special note. When the user clicks a button, the
adapter forwards a `LoopEvent::WidgetInteraction(event)`. The loop
either:

- builds a templated prompt from the segment's content (spine
  Revisit / Go deeper / Compare actions — see
  `translate_spine_action`), or
- defaults to `event.to_user_message()`, which serialises to
  `"[widget-event] widget=… action=… data=…"`.

Either way the result runs through `handle_turn` **exactly like a
typed user message**: it appends an `EntryType::UserMessage`, builds
context, calls the LLM. From the LLM's perspective there's no
distinction between "user typed this" and "user clicked a button" —
just user prose.

This is also why the spine's segmenter has the
`is_widget_event_user_message` filter (matches the `[widget-event]`
prefix): those synthetic user messages shouldn't open a new turn
block on the rail, even though they're "user" entries in the tree.

## 5. The outbound event side

Inside `handle_turn` (and helpers), the loop fires `event_tx.send(...)`
at every notable transition. The list is in §1 above. None of these
are read back into context — they're write-only telemetry.

The codec (`crates/gw-ui/src/ag_ui/codec.rs::loop_event_to_ag_ui`)
projects each outbound `LoopEvent` to an `AgUiEvent` for the wire.
Some return `None` and never leave the loop's process (inbound-only
variants, or events handled via the surface store's notification path
rather than as direct events).

### 5.1 The end-of-turn fan-out

A typical successful turn ends like this (from `handle_turn` plus the
`run_input_turn` wrapper):

```rust
// Inside handle_turn:
let assistant_entry_id = self.tree.append(EntryType::AssistantMessage { ... });
// later, after FINAL is resolved:
let narration_id = self.tree.append(EntryType::AssistantNarration { content: text });
result.assistant_entry_id = Some(narration_id);
self.flush_tree().await; // → spine extraction tasks spawn here

// In run_input_turn after handle_turn returns:
let _ = self.event_tx.send(LoopEvent::TurnComplete);
emit_text_message(&self.event_tx, response, result.assistant_entry_id);
//                                                ^ narration_id
```

The tree gets the durable record. The event stream gets `TurnComplete`,
then `TextMessageStart { entry_id: narration_id }` / `Content` / `End`.
The frontend stamps the chat row's `data-entry-id` from
`TextMessageStart.entry_id`. The next turn's `build_turn_context`
reads the tree, sees the raw assistant message (with its `FINAL("…")`
call), skips the narration, and the LLM gets the right history.

### 5.2 Spine fan-out from `flush_tree`

`flush_tree` spawns per-entry extraction tasks, each of which fires
`event_tx.send(LoopEvent::SpineEntryExtracted)` after persisting
typed entities. After all per-entry tasks complete, a single
coordinator task runs `resegment` once and fires
`SpineSegmentsUpdated`. Both events ride out via the codec to clients
that subscribe to the SSE stream — but neither feeds back into the
loop.

### 5.3 `InputRequest` is the one outbound event that pauses execution

When the agent calls `channel.ask`, the bridge fires
`LoopEvent::InputRequest(prompt)` outbound *and blocks the REPL
thread* on a `std::sync::mpsc::recv` waiting for the user's reply. The
reply arrives via a separate HTTP path (the adapter's ask-reply
endpoint), which signals the bridge's blocking channel. The reply
string eventually flows back as an `EntryType::UserMessage` in the
next turn. So `InputRequest` is the one outbound event that genuinely
gates execution on inbound action.

## 6. Why the split matters

Keeping the event stream out of `build_turn_context`'s input gives:

- **Trivial replay.** Loading a session from PG and rebuilding context
  just walks `session_entries`. No event-log replay; no chance of
  "missed an event so the LLM sees stale history."
- **Independent telemetry granularity.** We can emit `CodeExecuted` for
  every block (the DebugPane wants this) without that becoming a
  separate message in the chain.
- **Frontend agnostic to `EntryType`.** Clients bind to AG-UI events
  and JSON-Patch state deltas; entry types stay an internal Rust
  concept.

The cost: when adding a new piece of context (e.g. `AssistantNarration`),
you have to update **two places** — the projection in `context.rs`
(decide whether the LLM sees it) and the codec (decide whether clients
see it). Independent decisions, but the matrix has to be kept in your
head.

## 7. Trace of a typical turn

```
client POST /messages "What's new in MI?"
   └─ adapter sends LoopEvent::UserMessage to event_rx
       └─ run() dequeues → run_input_turn → handle_turn
           ├─ tree.append(UserMessage)                       ← tree grows
           ├─ event_tx.send(UserMessageAnchor)               ← outbound
           ├─ event_tx.send(TurnStarted)                     ← outbound
           ├─ build_turn_context(path)                       ← reads tree
           ├─ llm.chat(messages)                             ← uses what context built
           ├─ extract_code_blocks → repl.execute             
           │    ├─ event_tx.send(HostCallStarted/Args/Completed)
           │    │   (per host-fn call inside the block)
           │    └─ event_tx.send(CodeExecuted)
           ├─ tree.append(CodeExecution)                     ← tree grows
           ├─ tree.append(AssistantMessage)                  ← tree grows (raw response)
           ├─ tree.append(AssistantNarration)                ← tree grows (FINAL prose)
           ├─ flush_tree → PG persist + spawn extraction tasks
           ├─ event_tx.send(TurnComplete)                    ← outbound
           ├─ emit_text_message(response, narration_id)      ← outbound:
           │     event_tx.send(TextMessageStart{entry_id})        TextMessageStart
           │     event_tx.send(TextMessageDelta{model})           TextMessageDelta
           │     event_tx.send(TextMessageEnd)                    TextMessageEnd
           └─ later, from spawned tasks:
               ├─ event_tx.send(SpineEntryExtracted)         ← outbound (per entry)
               └─ event_tx.send(SpineSegmentsUpdated)        ← outbound (once, coordinator)
```

The events fan out to whoever's subscribed (SSE clients, the
DebugPane, the standalone Observer). The next turn's context is built
fresh from the tree the moment it runs, ignoring the event stream
entirely.

## 8. Where to look

Code pointers if you need to confirm something:

- `crates/gw-core/src/loop_event.rs` — the `LoopEvent` enum.
- `crates/gw-core/src/session_tree.rs` — `EntryType`, `SessionEntry`.
- `crates/gw-loop/src/context.rs` — the projection (`build_turn_context_with_opts`).
- `crates/gw-loop/src/conversation.rs` — `ConversationLoop::run()`,
  `handle_turn`, `run_input_turn`, `flush_tree`, `translate_spine_action`.
- `crates/gw-loop/src/tree.rs` — `SessionTree`, `flush_to_pg`, `path_to_leaf`.
- `crates/gw-loop/src/pg_store.rs` — PG persistence shape, `entry_type_tag`.
- `crates/gw-ui/src/ag_ui/codec.rs` — outbound `LoopEvent` →
  `AgUiEvent` projection.
- `crates/gw-ui/src/ag_ui/adapter.rs` — HTTP entry points, the
  inbound construction site for `LoopEvent::UserMessage` and
  `LoopEvent::WidgetInteraction`.
- `crates/gw-loop/tests/conversation_test.rs` — integration tests
  for the projection (`test_compaction`, `test_multi_iteration_turn`).

## 9. Future variants

If a future agent path needs **native tool calls** (e.g. the
Anthropic Messages API's `tool_use` / `tool_result` shape) instead
of fenced code blocks, the cleanest extension is a parallel
projection function — `build_turn_context_anthropic_tools` —
that emits the same chain in the new shape against the same
`EntryType` source. The tree wouldn't change. Two consequences:

- Each example would pick its projection at loop construction time,
  so a single binary can host examples on both shapes.
- The decision about whether `AssistantNarration` is sent to the LLM
  remains the same (no — it's still a duplicate of what the raw
  `AssistantMessage` already contains).

**No fundamental restructure is required** because the tree was
designed to be the truth and the message chain a derived view.
