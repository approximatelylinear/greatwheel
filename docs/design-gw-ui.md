# Design: Generative UX (`gw-ui`)

**Status:** All six steps landed
**Date:** 2026-04-22

---

## 0. Implementation Status

Steps 1 and 2 are landed. Subsequent steps follow the order in §11.

| Component | Status | Location |
|-----------|--------|----------|
| `Widget` / `WidgetEvent` / `UiSurfaceId` / `WidgetId` types | Done (step 1) | `gw-core/src/ui.rs` |
| `LoopEvent` widget variants | Done (step 1) | `gw-core/src/loop_event.rs` |
| `UiSurfaceStore` (in-memory) | Done (step 2) | `gw-ui/src/surface.rs` |
| `UiPlugin` + `ui.emit_widget` | Done (step 2) | `gw-ui/src/plugin.rs` |
| `AgUiAdapter` (SSE outbound, POST inbound) | Done (step 3) | `gw-ui/src/ag_ui/` |
| `WidgetInteraction` → turn delivery | Done (step 4) | `gw-loop/src/conversation.rs`, `gw-ui/src/ag_ui/adapter.rs` |
| `supersede` / `resolve` / `pin_to_canvas` / `emit_mcp_resource` host fns | Done (step 5) | `gw-ui/src/plugin.rs` |
| MCP-UI resource detector | Done (step 5, not yet wired) | `gw-ui/src/mcp_ui.rs` |
| React frontend (Vite) | Done (step 6) | `frontend/` |
| `echo_server` example for frontend smoke-testing | Done (step 6) | `crates/gw-ui/examples/echo_server.rs` |

---

## 1. Motivation

Agents inside Greatwheel need to present interactive UX — buttons,
selectors, forms, rich embedded apps — as part of a conversation.
Historically the naive approach ties the UX to the chat message that
produced it, which breaks along three axes:

- **Scrolling.** Complex widgets in the message stream compete with
  narrative text for vertical space.
- **History replay.** Showing live buttons for a past turn is wrong;
  hiding them loses the record of what was offered.
- **Interaction flow.** When a user clicks a button the agent often
  responds with another set of buttons. A naive implementation
  produces an awkward cascade of ephemeral bubbles.

The shared root cause is conflating *widget lifecycle* with *message
lifecycle*. The ecosystem's 2025–2026 answer is to keep widgets in a
separate store referenced by ID from messages, with explicit widget
states (active / resolved / expired / superseded), and to treat user
interaction as a first-class event feeding the agent loop — not a
hidden side effect of a button click.

---

## 2. Design Principles

**Widget lifecycle ≠ message lifecycle.** The session tree records what
was *said*. A separate per-session surface store records what was
*shown* and its current state. Messages reference widgets by ID, never
contain them.

**Interaction is a user-side event.** A user clicking a button produces
a `WidgetInteraction` `LoopEvent` that looks, to the conversation loop,
very much like a `UserMessage`. No host function blocks on UI state.

**Protocol first, frontend second.** The wire contract is a public
protocol (AG-UI for transport, A2UI for widget description, MCP-UI for
rich embedded apps). The first React frontend is treated as a
throwaway reference client; nothing in the backend design depends on
it.

**Relay MCP-UI, don't become MCP.** gw is conversation-shaped; MCP is
tool-call-shaped. When an external MCP server returns an MCP-UI
resource, we wrap it as a `Widget` and forward it over our own
AG-UI channel. gw does not itself expose an MCP server to the
frontend.

---

## 3. Protocol Choice

| Layer | Choice | Why |
|-------|--------|-----|
| Transport (fe ↔ gw) | **AG-UI** over SSE (+ HTTP POST for inbound) | Designed for exactly this channel; bidirectional event stream; React/Angular first-party clients; Rust community client available. |
| Widget description | **A2UI v0.9** (Google) | Declarative JSON component tree, flat with ID refs (streaming-friendly), framework-agnostic; `json-render.dev` provides a React renderer to start from. |
| Rich embedded apps | **MCP-UI** resources, relayed through AG-UI | Sandboxed iframe model when we need third-party apps; frontend uses `@mcp-ui/client`'s Web Component so any framework can mount it. |

Non-choices: **OpenAI Apps SDK** (walled garden), **Vercel AI SDK**
(JS-only backend assumption), **wrap gw as an MCP server** (wrong
protocol shape for a conversation channel).

---

## 4. Core Types (`gw-core::ui`)

```rust
pub struct UiSurfaceId(pub Uuid);
pub struct WidgetId(pub Uuid);

pub struct Widget {
    pub id: WidgetId,
    pub surface_id: UiSurfaceId,
    pub session_id: SessionId,
    /// Turn this widget was emitted alongside. Lets the frontend
    /// place the widget with its originating message during history
    /// replay, without embedding the widget into the message.
    pub origin_entry: Option<EntryId>,
    pub kind: WidgetKind,
    pub state: WidgetState,
    pub payload: WidgetPayload,
    pub supersedes: Option<WidgetId>,
    pub created_at: i64,
    pub resolved_at: Option<i64>,
    pub resolution: Option<serde_json::Value>,
}

pub enum WidgetKind {
    A2ui,           // declarative component tree
    McpUi,          // sandboxed iframe resource
    Custom(String), // escape hatch
}

pub enum WidgetState { Active, Resolved, Expired, Superseded }

pub enum WidgetPayload {
    Inline(serde_json::Value),
    Reference { uri: String, csp: Option<String> },
}

pub struct WidgetEvent {
    pub widget_id: WidgetId,
    pub surface_id: UiSurfaceId,
    pub action: String,         // "submit" | "select" | custom
    pub data: serde_json::Value,
}
```

**Invariant.** `Widget` is never stored inside a `SessionEntry`.
`origin_entry` is a back-reference, never a containment relationship.

---

## 5. `LoopEvent` Additions

Three new variants on `gw_core::LoopEvent`:

```rust
WidgetEmitted(Widget),
WidgetSuperseded { old: WidgetId, new: Widget },
WidgetInteraction(WidgetEvent),
```

Handling:

- `WidgetEmitted` and `WidgetSuperseded` are **outbound**. The
  conversation loop forwards them to channel adapters; no state
  machine transition happens in `gw-loop`.
- `WidgetInteraction` will become **turn-starting** in step 4 — it
  will feed into `handle_turn` the same way `UserMessage` does, with
  the widget's action and data rendered into the agent's context.
  For step 1 it is a no-op in the loop, consistent with other
  outbound variants, and existing match sites simply add it to the
  no-op arm.

The workspace lint forbids catch-alls, so every existing exhaustive
match on `LoopEvent` must add the three variants explicitly. The
only such site today is `gw-loop/src/conversation.rs::run`.

---

## 6. `UiSurfaceStore`

Session-scoped, owns widget state.

```rust
pub struct UiSurface {
    pub id: UiSurfaceId,
    pub session_id: SessionId,
    pub widget_order: Vec<WidgetId>,
    pub canvas_slot: Option<WidgetId>,
}

pub struct UiSurfaceStore { /* session_id -> UiSurface, widget_id -> Widget */ }

impl UiSurfaceStore {
    pub async fn emit(&self, w: Widget) -> Result<(), UiError>;
    pub async fn supersede(&self, old: WidgetId, new: Widget) -> Result<(), UiError>;
    pub async fn resolve(&self, id: WidgetId, data: Value) -> Result<(), UiError>;
    pub async fn expire(&self, id: WidgetId) -> Result<(), UiError>;
    pub async fn snapshot(&self, session: SessionId) -> UiSurfaceSnapshot;
}
```

MVP is in-memory. Persistence mirrors `gw-loop::pg_store`; a single
`ui_widgets` table is enough (columns for each field above) and gets
added when we need restart-durability.

**One surface per session** in v1. Multi-surface (side modals,
pop-overs) is a future extension keyed off `UiSurfaceId` being already
a distinct type.

---

## 7. Host Functions

Registered by `UiPlugin::init()` via the existing
`ctx.register_host_fn_async`. All are capability-gated under
`ui:write` / `ui:read`.

| Name | Python signature | Effect |
|------|------------------|--------|
| `ui.emit_widget` | `(kind, payload, supersedes=None) -> widget_id` | Insert into store; emit `LoopEvent::WidgetEmitted`. |
| `ui.supersede_widget` | `(old_id, new_payload) -> widget_id` | Mark old `Superseded`; emit `WidgetSuperseded`. |
| `ui.emit_mcp_resource` | `(uri, csp=None) -> widget_id` | Convenience: emit an `McpUi` widget with `WidgetPayload::Reference`. |
| `ui.resolve_widget` | `(widget_id, data) -> None` | Force-resolve, for agent-driven close. |
| `ui.pin_to_canvas` | `(widget_id) -> None` | Move widget into `UiSurface::canvas_slot`. |

**No `await_interaction`.** The agent does not block on UI state. When
a user interacts, the resulting `WidgetInteraction` loops back
through `gw-loop` and becomes input on the agent's next turn.

---

## 8. AG-UI Channel Adapter

Lives in `gw-ui/src/ag_ui/`. **Does not** currently implement
`gw_core::ChannelAdapter` — that trait's `handle_outbound(&self,
&LoopEvent)` signature is session-agnostic, which clashes with AG-UI's
one-adapter-many-sessions model. The adapter exposes richer methods
instead: `register_session(sid, inbound_tx)`, `dispatch(sid, event)`,
`subscribe_session(sid)`, and `router() -> axum::Router`. Once the
trait grows a session dimension (or gets replaced), the impl slots in
trivially.

Endpoints (axum 0.8):

- `POST /sessions/:id/messages` → `LoopEvent::UserMessage`
- `POST /sessions/:id/widget-events` → `LoopEvent::WidgetInteraction`
- `GET  /sessions/:id/stream` (SSE) → outbound event stream
- `GET  /sessions/:id/surface` → `UiSurfaceSnapshot` (reconnect bootstrap)

`handle_outbound` maps each `LoopEvent` into the AG-UI event envelope
and pushes to the SSE broadcast:

| `LoopEvent` | AG-UI event |
|---|---|
| `UserMessage` (inbound) | `TEXT_MESSAGE` |
| `WidgetInteraction` (inbound) | `TOOL_CALL_RESULT` with synthetic tool `ui.widget_event` |
| `Response { content, .. }` | `TEXT_MESSAGE_CONTENT` (streaming delta) |
| `WidgetEmitted(w)` | `UI_EVENT { kind, payload }` |
| `WidgetSuperseded { .. }` | `STATE_DELTA` on the surface |
| `TurnComplete` | `RUN_FINISHED` |
| `InputRequest` | `TEXT_MESSAGE` + `UI_EVENT` carrying a prompt widget |

Auth on the inbound endpoints uses a signed widget token derived from
`SessionKey` so the frontend cannot forge interactions for widgets it
does not own.

---

### 8a. Widget event resolution

When a `POST /sessions/:id/widget-events` arrives, the adapter calls
`store.resolve(widget_id, data)` *before* forwarding
`LoopEvent::WidgetInteraction` to the conversation loop. This keeps
widget state ownership in the adapter layer — the loop never touches
the store — and ensures the SSE stream sees the `Resolved` state
delta immediately rather than after the agent's turn completes. If
the widget is no longer `Active` (superseded, expired, or already
resolved), the resolve error is logged and the interaction still
forwards to the agent; the agent may choose how to react to
late/duplicate clicks.

## 9. MCP-UI Relay

`gw-ui/src/mcp_ui.rs` exposes `detect(result, session_id, surface_id)`
— a pure function that inspects a JSON value and, if it matches the
MCP-UI resource shape (`resource.mimeType` ∈ {`text/html`,
`application/vnd.mcp-ui+json`}), returns a `Widget` ready for
`UiSurfaceStore::emit`. Metadata at `resource._meta.ui.csp` is copied
into `WidgetPayload::Reference.csp`.

**Not yet wired.** Greatwheel does not currently have an MCP client
integration, so the detector has no caller. Once `gw-engine` or
`gw-runtime` gains MCP tool-call support, the integration code calls
`detect()` on each tool-call result and, on `Some`, pushes the widget
through the normal emit path — no agent code changes needed.

Agents that call MCP servers through some other path (or receive
MCP-UI URIs via any mechanism) can emit a widget immediately via the
`ui.emit_mcp_resource` host function.

Widget-to-server postMessage traffic from the iframe is tunneled back
by the frontend via the AG-UI `widget-events` endpoint; gw forwards to
the originating MCP server.

---

## 10. Frontend (`frontend/`, Vite + React)

```
frontend/
├── src/
│   ├── main.tsx
│   ├── App.tsx                  # ChatPane | CanvasPane layout
│   ├── agui/
│   │   ├── client.ts            # @ag-ui/client wrapper
│   │   └── store.ts             # zustand: messages[], surface.widgets{}
│   ├── widgets/
│   │   ├── WidgetRenderer.tsx   # switch on Widget.kind
│   │   ├── A2uiWidget.tsx       # json-render-based, overridable
│   │   └── McpUiWidget.tsx      # UIResourceRenderer from @mcp-ui/client
│   └── render/a2uiCatalog.tsx   # our React component mapping
└── vite.config.ts
```

Messages and widgets live in two separate stores keyed by
`surfaceId`. Historical widgets render inline next to their
`origin_entry` in their *terminal* state; active widgets render
either inline on the most recent turn or in a canvas pane if
pinned. State transitions re-render without touching the message
stream.

**Why Vite, not Next.js.** Greatwheel orchestrates in Rust. Next.js's
value prop — server-side LLM orchestration in JS — is not useful
here. Vite is minimal setup and the protocol work transfers
unchanged when we later pick a real frontend.

**Throwaway choices made in step 6.** To keep the reference client
small and dependency-light:

- No `@ag-ui/client` npm package — just `EventSource` + `fetch`. We
  still speak the AG-UI wire format.
- No `json-render` — a hand-rolled A2UI subset (Column / Row / Text /
  Button). Swap in the real renderer when we need the full v0.9
  catalog.
- No `@mcp-ui/client` — a single sandboxed `<iframe>` stands in for
  `UIResourceRenderer`. Production needs the two-level iframe + CSP
  + postMessage bridge that library provides.
- One fixed session per run via `echo_server` (`cargo run -p gw-ui
  --example echo_server` prints a UUID; point the frontend at it).
  Proper session bootstrap lives with `gw-server` integration, not
  here.

---

## 11. Implementation Order

Each step ends at a runnable system.

1. **Types.** `gw-core::ui` module + `LoopEvent` variants. No
   behavior. `cargo check --workspace` green.
2. **Store + plugin.** `gw-ui` crate with `UiSurfaceStore` +
   `UiPlugin` registering `ui.emit_widget`. Host function writes
   store; emits `WidgetEmitted`. No transport yet.
3. **AG-UI adapter.** SSE outbound + `/messages` inbound. Smoke
   test chat end-to-end over AG-UI.
4. **Interaction.** `/widget-events` inbound →
   `WidgetInteraction` → `handle_turn`.
5. **Rest of host functions.** `supersede`, `resolve`,
   `pin_to_canvas`, `emit_mcp_resource`; MCP-UI relay in tool-call
   result pipeline.
6. **Frontend.** Vite + React with A2UI renderer; MCP-UI renderer
   plugs in when we need it.

---

## 12. Open Questions (for later revisit)

- **Multi-surface support.** V1 is one surface per session. If we
  want side modals or pop-overs, what's the right abstraction for
  "which surface does this widget go to"?
- **Streaming partial A2UI trees.** A2UI's flat-list-with-IDs
  design supports incremental emit. Plumbing this needs a
  `LoopEvent::WidgetPatch` variant and an AG-UI patch event. Defer
  to v2.
- **Widget expiration policy.** Server-side timeout vs. client-side
  timeout vs. agent-driven resolve-only. Probably all three, with
  sensible defaults.
- **A2UI component catalog.** Start with `json-render`'s v0.9
  coverage and override the subset we need. Track divergence as we
  customize.
- **Permission model.** `ui:write` / `ui:read` capabilities are
  declared but not yet enforced (capability enforcement is a
  cross-cutting TODO in `HostFnRouter`).

## 13. Follow-ups from Step 6 Frontend

The throwaway frontend works end-to-end (chat + widget emit +
interaction + resolve) but has rough edges worth cleaning up before
anyone builds seriously on it:

- **Widget placement by origin turn.** Widgets currently render at
  the bottom of the chat scroll, not next to the turn that emitted
  them. Needs `Widget.origin_entry` to be stamped on emit (agent or
  loop side) and the frontend to group widgets by `origin_entry` /
  turn boundary. Until then, historical widgets float out of place.
- **Terminal-state button styling.** `.a2ui-button:disabled` looks
  identical to an active button. Bump opacity / cursor / border
  treatment under `.a2ui-widget.terminal` so post-click widgets
  clearly read as "done".
- **Hydrate from `/surface` on mount.** The frontend currently only
  applies SSE events, so refreshing the browser drops all prior
  widgets from the UI even though they're still in the server's
  `UiSurfaceStore`. Call `fetchSurface(sessionId)` before opening the
  stream and seed the widget store from the snapshot. This is the
  feature that actually demonstrates "widget lifecycle ≠ message
  lifecycle" — durable widget state survives a frontend reload even
  when message history doesn't.
