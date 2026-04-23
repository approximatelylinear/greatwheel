# Design: full migration to the AG-UI + json-render protocol

**Status:** Drafted 2026-04-23 after spike + client migration
(commit `51bca65`). Follow-up to `docs/design-gw-ui.md`; extends its
§8 (AG-UI channel adapter) and §14 (Frankenstein demo follow-ups).

## 1. Why this doc

The frontend now renders through `@json-render/react`, but our
server's AG-UI adapter is still a **dialect**, not vanilla AG-UI.
Specifically:

- We emit `TEXT_MESSAGE_CONTENT` in one chunk per turn, no
  `TEXT_MESSAGE_START` / `_END`, no streaming.
- We emit `RUN_FINISHED` but not `RUN_STARTED` or `RUN_ERROR`.
- Host-function calls surface only via our own `DEBUG_CODE_EXEC`
  extension; there's no `TOOL_CALL_START` / `_ARGS` / `_END`.
- `STATE_DELTA` patches use our domain vocabulary (`supersede`,
  `resolve`, `expire`, `pin`, `pin_aux`, `highlight`) rather than
  JSON-Pointer writes against a generic state object.

A vanilla AG-UI client (including an unmodified
`@json-render/react` app someone else writes) cannot consume our
stream today. The client migration hides that because we wrote the
one client that consumes it.

This doc plans the server-side work to make us vanilla-AG-UI.

## 2. What the spike and client migration proved

- Our 5-tag widget vocabulary (Column/Row/Text/Button/Card) maps
  cleanly to a json-render catalog with Zod props schemas.
- The nested → flat Spec translation is a ~100-line pure function
  (`frontend/src/jr/translate.ts`).
- json-render's event resolution (component `emit('press')` →
  element `on.press` ActionBinding → catalog `interact` action →
  `JSONUIProvider` handler) works end-to-end for our click flow.
- Widget lifecycle (Active/Superseded/Resolved/Expired, pin to
  canvas, follow-up anchoring) lives cleanly *outside* json-render
  in our session reducer. The lifecycle shell wraps json-render's
  `<Renderer>`; the inline tree is json-render's job.

**Remaining work is entirely on the server adapter** — the client is
ready to consume whatever we send.

## 3. Open decision: STATE_DELTA translation strategy

This is the one load-bearing decision. Everything else is mechanical.

### Option A — generic state writes

Translate our patches into JSON-Pointer writes against a conventional
`{widgets: {<id>: Widget}, canvasSlot: id, pressed: {<id>: bid}}`
state shape, emitted via `STATE_DELTA` events shaped like JSON
Patch (`{op: "replace", path: "/widgets/abc/state", value: "Resolved"}`).

- **Pro:** vanilla AG-UI. Any client that speaks AG-UI (including an
  unmodified json-render app) can consume our stream and bind props
  with `{$state: "/widgets/<id>/state"}`.
- **Pro:** the client migration's current session reducer becomes
  redundant — json-render's `StateStore` is the source of truth.
- **Con:** the receiving client must implement our lifecycle rules
  (what does `state == "Superseded"` imply about rendering?) by
  consuming raw state, without our reducer's help. That's
  intrinsically our protocol complexity anyway; only the *location*
  moves.
- **Con:** write amplification. `highlight` becomes
  `{op: "replace", path: "/pressed/<widget_id>", value: "<button_id>"}`
  which is fine; `supersede` becomes two writes (mark old Superseded,
  add new Active) which is also fine. No real concern.

### Option B — keep our patches, emit as `CUSTOM` events

Wrap each domain patch in an AG-UI `CUSTOM` event and require the
client to install a plugin (a json-render `$computed` function map
or similar) that understands our vocabulary.

- **Pro:** no state-shape decision forced on the server.
- **Con:** vanilla AG-UI clients can't consume it without the
  plugin. Loses the "swap in any AG-UI client" portability we gain
  from A.
- **Con:** our current frontend's session reducer stays load-bearing
  forever.

### Recommendation: **A**

The vocabulary complexity of our patches isn't eliminated by B —
it's just pushed to a required plugin. A pays a one-time translator
cost on the server for permanent client portability, and makes our
state shape inspectable in any AG-UI debugger.

State shape I'd commit to:

```
{
  widgets: { [id]: Widget },        // full widget records
  widgetOrder: string[],             // scroll-tail ordering
  canvasSlot: string | null,
  canvasAuxSlot: string | null,
  pinnedIds: { [id]: true },         // ever-pinned set
  pressed: { [widgetId]: buttonId }, // agent + user both write here
  focusedScope: { [kind]: key }      // see §3.1 "Widget scope"
}
```

`messageFollowUps` and `pendingFollowUps` can stay client-side — they
exist only because our widget→message anchoring is a UI convention,
not protocol state. (Candidate for later: first-class `anchor`
field on Widget; see §14's "First-class widget scoping / linkage".)

### 3.1 Widget scope → visibility

The current demo has an implicit parent-child relationship: the
characters widget for chapter N belongs to chapter N. Today we infer
this on the frontend by inspecting Card `data.section` values, and
the agent is asked (via prompt discipline) to re-emit on navigation.
§14 of `design-gw-ui.md` flagged the fragility.

json-render has no named "widget scope" concept, but its built-in
`visible` condition on every UIElement is a natural fit:

```json
{"type": "Column", "props": {...}, "visible": {"$state": "/focusedScope/section", "eq": 4}}
```

Resolution:

- **New field `Widget.scope: Option<WidgetScope>`** where
  `WidgetScope = { kind: String, key: serde_json::Value }`. The agent
  declares a widget's scope once at emission time; stays loosely
  typed like the rest of our Widget JSON (tighter enums can come
  later if we need them across demos).
- **Translator emits a `visible` condition** on the widget's root
  UIElement: `{$state: "/focusedScope/<kind>", eq: <key>}`. Widgets
  scoped to a kind+key other than the current focus simply don't
  render — json-render's visibility resolver handles the rest.
- **Server infers `focusedScope` from button clicks.** When a
  `WidgetEvent` arrives whose `data` contains `{scope: {kind, key}}`
  (or, for back-compat during migration, `{section: N}`), the
  adapter writes `state.focusedScope[kind] = key` as part of the
  same STATE_DELTA burst. Keeps the agent out of state plumbing;
  moves the existing frontend heuristic to the server where it
  belongs.
- **Deprecates** the current `canvas_aux_slot` auto-swap, the
  frontend `sectionScopedWidgets` map, and the prompt rule "re-emit
  characters on navigation." All three become dead code after
  phase 3.

Host-function impact: `emit_widget` grows a `scope=` kwarg. No new
host functions; no `focus_scope` or `set_state`. The agent never
touches state directly — the boundary we wanted to keep.

## 4. Proposed ordering

Each phase independently leaves the demo green.

### Phase 1 — `RUN_STARTED` / `RUN_ERROR`

~30 lines in `gw-ui/src/ag_ui/codec.rs`. Emit `RUN_STARTED` on
first event of a turn, `RUN_ERROR` when the loop returns `Err`.
Existing clients ignore unknown events, so no client change required.

**Acceptance:** SSE stream for a turn begins with `RUN_STARTED` and
ends with `RUN_FINISHED` or `RUN_ERROR`. Existing demo still works.

### Phase 2 — STATE_DELTA translator (Option A)

In `gw-ui/src/ag_ui/codec.rs`, convert each internal
`UiNotification` into JSON-Patch writes against the canonical state
shape. Emit both the JSON-Patch `STATE_DELTA` **and** (temporarily)
the legacy domain patch, so the current frontend keeps working
during migration. Add a `STATE_SNAPSHOT` event on SSE subscribe that
matches `/surface` output in the new shape.

**Acceptance:** a fresh `@json-render/react` app wired only to
`/stream` (no `/surface` fetch, no custom reducer) can render the
demo with just `{$state: "/widgets/<id>/state"}`-style bindings.
Our current client still renders identically.

### Phase 3 — port frontend to state-driven bindings

With phase 2 shipping both patch shapes, update
`frontend/src/jr/translate.ts` and the session reducer to consume
the new shape. Delete the legacy patches from the adapter.

**Acceptance:** session reducer's widget/pressed state is empty —
all props resolve via `$state`. Follow-up anchoring + section
scoping logic keep their own local state as noted above.

### Phase 4 — `TOOL_CALL_*` events

Plumb host-function calls through `UiPlugin` as
`TOOL_CALL_START` / `_ARGS` / `_END` events. Keep `DEBUG_CODE_EXEC`
as a superset for `?debug=1`.

**Acceptance:** a json-render `useChatUI`-style client (or AG-UI
inspector) shows `emit_widget` / `pin_to_canvas` /
`highlight_button` as first-class tool calls with resolved args.

### Phase 5 — streaming text

Plumb LLM token streams from `gw-llm` through the rLM loop so
`TEXT_MESSAGE_CONTENT` arrives incrementally, bracketed by
`TEXT_MESSAGE_START` / `_END`. Biggest change — touches `gw-llm`,
`gw-loop`, and the Ollama/OpenAI clients. Out of scope for the
first sprint; split into its own doc when ready.

**Acceptance:** assistant prose appears token-by-token in the demo,
and our session reducer's `assistant-chunk` path exercises
incremental append (it already can).

## 5. Open questions

- **Lifecycle in state: pruning.** With Option A, `state.widgets`
  grows unboundedly across a session. Need an explicit eviction
  policy (e.g. prune `Expired` after N minutes). Deferrable.
- **Widget scope field.** §14 flags this as the right fix for the
  current section-scoping heuristic. Orthogonal to AG-UI protocol
  work, but likely to affect Widget JSON shape — decide before
  phase 2 freezes the state shape.
- **Does `gw_core::ChannelAdapter` come back?** Its current
  signature is session-unaware. If we want the same AG-UI adapter
  to also serve a future MCP channel, rework the trait to take
  `&CallContext`. Low priority; flag when relevant.

## 6. What is NOT in scope

- Migrating `@json-render/react` to a newer major. 0.18.0 was
  released 2026-04-17 (one week before this doc); API may churn.
- Implementing `@ag-ui/client` on the frontend. Our hand-rolled
  `fetch` + `EventSource` works, and switching would be tactical
  at best.
- Forms (`Action.Submit`) or navigate actions. Neither demo uses
  them; defer until a use case surfaces.
