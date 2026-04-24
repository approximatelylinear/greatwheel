# Design: agent-writable session state (`set_state`)

**Status:** Drafted 2026-04-24. Follow-up to
`docs/design-json-render-migration.md` §3 (canonical state shape).

## 1. Goal

Let agents write to arbitrary JSON-Pointer paths inside the session's
canonical state. Today the agent can only influence client-visible
state by emitting widgets — the widget payload carries the
"information" and the agent has to `supersede_widget` to mutate. This
conflates two concerns: **domain state** (what the agent has
figured out) and **presentation** (how it's shown).

`set_state` separates them. Agents write facts to `/agent/<path>`;
widgets declare themselves as views via `{$state: "/agent/<path>"}`
bindings. Updating domain state no longer requires re-emitting
widgets — the view re-renders itself.

## 2. Transparency to AG-UI

**Zero wire change.** `set_state` writes translate into ordinary
`STATE_DELTA` JSON-Patch events against the canonical state — the
same shape the adapter already emits for widget lifecycle and
focused-scope updates. A vanilla AG-UI client that's never heard of
`set_state` sees a patch arrive and applies it; components bound to
the touched path re-render. No event-type negotiation, no extension
fields, no client plugin.

From the client's perspective, adding `set_state` is indistinguishable
from the server learning to emit patches at new paths.

## 3. Host-function surface

Minimum useful set, all registered by a new tiny `AgentStatePlugin`
(or folded into `UiPlugin`):

```python
set_state(path: str, value: Any) -> None
# Replace value at a JSON-Pointer path inside /agent/**.
#   set_state("/agent/query", "select * from users")
#   set_state("/agent/results/rows", [[1, "alice"], [2, "bob"]])

append_state(path: str, value: Any) -> None
# Append to the array at a JSON-Pointer path. Equivalent to
# JSON-Patch `/.../- `.
#   append_state("/agent/results/rows", [3, "carol"])

delete_state(path: str) -> None
# Remove the key at a JSON-Pointer path. Used for cleanup when a
# widget of a particular scope is no longer relevant.
```

The three together cover everything the agent can express with JSON-
Patch. No `test` / `move` / `copy` — not needed, and they're rarely
useful from a sandboxed-Python perspective.

### Namespace

Agent-writable paths are **constrained to `/agent/**`.** System-managed
paths (`/widgets`, `/widgetOrder`, `/canvasSlot`, `/canvasAuxSlot`,
`/pinnedIds`, `/pressed`, `/focusedScope`) reject `set_state` writes
with a `PluginError::HostFunction("path reserved by UI adapter")`. This
is enforced at the plugin layer, not with complex permissions — one
`path.starts_with("/agent/")` check.

Rationale: the UI paths are a contract between the adapter and
clients (pressed flips on click, focusedScope flips on scope-bearing
click, etc.). If the agent could write them directly, the adapter's
invariants become impossible to reason about. The fix is
cheap — give the agent its own namespace.

## 4. State storage

Per-session `serde_json::Value` on the adapter's `AgUiState`, mirrored
into `STATE_SNAPSHOT` on subscribe. Sketch:

```rust
pub struct AgUiState {
    // ...existing fields...
    /// Per-session `/agent/*` state bucket. Writes emit STATE_DELTA
    /// patches against the corresponding path in the canonical state.
    agent_state: Mutex<HashMap<SessionId, serde_json::Value>>,
}
```

The plugin receives a handle to this map (or a shim that knows how to
fire `STATE_DELTA` events) via the session context. On each
`set_state` / `append_state` / `delete_state`:

1. Apply the mutation to the per-session `Value`.
2. Construct a JSON-Patch op:
   - `set_state`  → `{op: "replace" or "add", path, value}`
   - `append_state` → `{op: "add", path: "<path>/-", value}`
   - `delete_state` → `{op: "remove", path}`
3. Fan out as `AgUiEvent::StateDelta` on the session's SSE broadcast.

`canonical_state(snapshot, focused_scope, agent_state)` includes the
bucket under the `/agent` key so `STATE_SNAPSHOT` carries it on
reconnect.

## 5. Session context threading

The plugin needs to know *which* session is calling. Today the
Frankenstein demo injects `gw_session_id` as a Python variable via
`ReplAgent::set_variable` and the prompt begs the agent to pass it
as a kwarg — accidental but working. `set_state` inherits the same
convention for the first cut:

```python
set_state(session_id=gw_session_id, path="/agent/foo", value=42)
```

The proper fix — plumb `CallContext` through `HostFnRouter::dispatch`
so host fns can read `session_id` from context — is already on the
§14 follow-up list in `design-gw-ui.md`. Worth doing before we ship
too many more session-aware host fns.

## 6. Wire format (reminder)

Every `set_state` / `append_state` / `delete_state` produces one
`STATE_DELTA` event on `/stream`:

```json
{
  "type": "STATE_DELTA",
  "surface_id": "<uuid>",
  "patches": [
    {"op": "replace", "path": "/agent/query", "value": "select ..."}
  ]
}
```

Plus a `TOOL_CALL_*` trilogy around the dispatch (phase 4 gives us
this for free).

## 7. Existing demos: no impact

Running the current Frankenstein demo after shipping `set_state`
produces zero new wire events unless the agent prompt is updated to
use the new host fns. The change is strictly additive — existing
`emit_widget` / `supersede_widget` / `pin_to_canvas` continue to
work unchanged.

## 8. Companion primitive (future): `watch_state`

The natural partner is:

```python
watch_state(path: str, handler: callable) -> None
```

Lets the agent react to state changes — e.g., "when `/agent/filters`
updates, re-run the query and write results to `/agent/results`."
Turns the agent from a turn-based responder into a reactive function
over state.

Deliberately **out of scope** for the first cut because:
- It needs a new loop primitive (wake the agent on state change, not
  just on user input).
- `set_state` on its own already unlocks the biggest wins (agents
  can maintain domain state instead of cramming it into widget
  payloads).
- Once we have real demos using `set_state`, we'll know what shape
  `watch_state` actually needs (path filters? debouncing? batching?).

Revisit after at least one `set_state`-native demo ships.

## 9. Open questions

- **Path-level observability.** Should `set_state` show in the
  `TOOL_CALL_*` stream with the value, or just the path? Including
  the value makes the debug pane much richer but could get noisy for
  large writes. **Proposal:** include value; truncate for display in
  the client's debug pane.
- **Rejection vs silent drop for reserved paths.** Today's
  `highlight_button` gracefully skips when no pin exists; should
  `set_state("/widgets/...", ...)` error or be a no-op? **Proposal:**
  error — reserved-path writes are programming bugs, not graceful
  fallbacks.
- **Initial state.** Does the agent get to seed `/agent/*` at session
  creation, or only on first turn? **Proposal:** only on first turn.
  Keeps the "session = source of truth on the server" invariant
  clean; all state changes are triggered by user input or a prior
  agent action.
- **Size limits.** Should large values (multi-MB state blobs) be
  rejected? **Proposal:** cap per-patch at 1MB with a clear error;
  agents that need bulk data should page via `append_state`.

## 10. Acceptance

1. Agent calls `set_state(session_id=..., path="/agent/counter", value=1)`;
   client's StateStore reflects it; a `{$state: "/agent/counter"}`
   binding renders `1`.
2. A `STATE_DELTA` with `{op: "replace", path: "/agent/counter",
   value: 1}` appears on `/stream` immediately after the call.
3. Reconnect mid-session; `STATE_SNAPSHOT` includes
   `state.agent.counter === 1`.
4. `TOOL_CALL_*` trilogy fires for each `set_state` call, visible in
   the debug pane.
5. `set_state(path="/widgets/foo", ...)` returns a `HostFunction`
   error without mutating anything.
6. Frankenstein demo runs unchanged — zero regression.

## 11. Implementation order

1. Add the three host fns to `UiPlugin` (or new `AgentStatePlugin`) —
   wired to the adapter's `agent_state` map.
2. Extend `canonical_state` to include `/agent` and the adapter's
   `STATE_SNAPSHOT` handler to pass the bucket.
3. Add a reserved-path guard. Test: writes to `/widgets/**` and
   other system paths return `Err`.
4. Integration test: an example with one session, one `set_state`
   call, assert STATE_DELTA lands with the right shape.
5. Update `docs/design-json-render-migration.md` §3 to note
   `/agent` is now part of the canonical state shape.
6. Ship. No frontend changes required.
