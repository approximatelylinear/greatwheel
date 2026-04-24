# greatwheel frontend

A React + Vite client for the generative-UI layer of [greatwheel](../).
The server runs a Python-sandbox agent that emits structured UI
widgets; this frontend renders them and routes user clicks back.
Written for a frontend engineer who's never touched the backend.

## What makes this interesting

- **The agent authors UI.** On each turn, Python code running inside
  an `ouros` sandbox can call `emit_widget(payload=...)` and the
  result materialises in the browser a few hundred milliseconds later.
  No pre-built components, no form schemas — the agent composes from
  a small typed catalog.
- **State lives server-side; bindings drive the DOM.** Clicks don't
  mutate local React state directly. They post to the server, the
  server re-emits canonical state as JSON-Patch deltas over SSE, and
  the frontend's [json-render](https://json-render.dev) `StateStore`
  re-renders anything bound to the touched paths. Makes refreshing
  the page free — all durable state is the server's.
- **Vanilla [AG-UI](https://docs.ag-ui.com) on the wire.** `/stream`
  emits the standard event vocabulary (`STATE_SNAPSHOT`,
  `STATE_DELTA` with RFC 6902 JSON-Patch, `RUN_STARTED` etc.). A
  different AG-UI client could consume the same stream and render
  the demo without our code.

## 30-second mental model

```
┌─ browser ──────────────────────────────────────────────────────┐
│                                                                │
│   App.tsx ─ openStream() ─┐                                    │
│      │                    │                                    │
│      │           ┌────────▼─ stateBridge.ts ────────────┐      │
│      │           │  STATE_SNAPSHOT → store.set(…)       │      │
│      │           │  STATE_DELTA    → JSON-Patch ops     │      │
│      │           └────────┬───────────────────────────── ┘     │
│      │                    │                                    │
│      │                    ▼                                    │
│      │           json-render StateStore                        │
│      │           { widgets, widgetOrder, canvasSlot,           │
│      │             pressed, focusedScope, … }                  │
│      │                    │                                    │
│      ▼                    ▼                                    │
│   ChatPane / CanvasPane ◀── useStateValue("/widgets/…")        │
│      │                                                         │
│      ▼                                                         │
│   WidgetRenderer ─▶ toJrSpec(widget)  ── <Renderer spec=…/>    │
│                          │                  │                  │
│                          ▼                  ▼                  │
│                    json-render catalog  emit('press')          │
│                    (Column/Row/Text/    ──────▶ interact       │
│                     Button/Card)                handler        │
│                                                    │           │
└────────────────────────────────────────────────────┼───────────┘
                                                     │
                                          POST /widget-events
                                                     │
                                                     ▼
                                            gw-ui AG-UI adapter
                                                     │
                                                     ▼
                                             conversation loop
                                                     │
                                     ┌───────────────┴────────────┐
                                     │ ouros Python sandbox       │
                                     │   emit_widget(…)           │
                                     │   pin_to_canvas(…)         │
                                     │   highlight_button(…)      │
                                     └────────────────────────────┘
```

## Running it

Two processes: a Rust backend and this Vite frontend. You need a
working LLM — either `OPENAI_API_KEY` set for GPT, or Ollama running
locally with `qwen3.5:9b`.

```sh
# 1. Backend — the Frankenstein demo (serves /stream on :8787)
cargo run -p gw-ui --example frankenstein_server

# 2. Frontend — Vite dev server on :5173 (or :5174 if occupied)
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173/?session=<uuid>` — any fresh UUID works,
the server creates sessions lazily. Send a message, or click a
chapter card in the generated picker.

### Environment variables

| Var | Default | Notes |
|---|---|---|
| `VITE_API_BASE` | `http://127.0.0.1:8787` | Backend origin |
| `VITE_SESSION_ID` | _(none)_ | Session UUID; overridden by `?session=` |

Append `?debug=1` to the URL to reveal an agent code-trace panel at
the bottom — every Python block the sandbox executes, with stdout
and `is_final`.

## The widget model in detail

Three pieces collaborate to turn a JSON blob from the server into
interactive DOM:

### 1. The catalog (`src/jr/catalog.ts`)

A typed vocabulary of components the server may emit, each with a
Zod schema for its props. The server can only pick from this set —
rendering is closed-world, hallucinations fail validation rather
than reaching the DOM.

```ts
export const catalog = defineCatalog(schema, {
  components: {
    Column: { props: z.object({}).passthrough(), slots: ['default'], ... },
    Card:   { props: z.object({ title: z.string(), ... }), ... },
    …
  },
  actions: {
    interact: { params: z.object({ widgetId, buttonId, action, data }) },
  },
});
```

### 2. The registry (`src/jr/registry.tsx`)

One render function per catalog entry. These are plain React
components — no lifecycle surprises.

```tsx
Card: ({ props, emit }) => (
  <button
    className={`a2ui-card${props.pressed ? ' pressed' : ''}`}
    onClick={() => emit('press')}
  >
    <div className="a2ui-card-title">{props.title}</div>
  </button>
)
```

Two things to notice:

- `props.pressed` is a plain boolean from the component's perspective.
  The translator (next) feeds it through a `$state` binding so the
  value tracks the server's canonical state without any local wiring.
- `emit('press')` resolves against the spec's `on.press` action
  binding and fires the `interact` handler registered in `App.tsx`,
  which posts to `/widget-events`.

### 3. The translator (`src/jr/translate.ts`)

Converts the server's nested payload shape into a json-render flat
`Spec`, baking state bindings in along the way:

```ts
// server emits: {type: "Card", id: "ch-5", title: "5", data: {section: 5}}
// translator produces an element like:
{
  type: 'Card',
  props: {
    title: '5',
    pressed: { $state: '/pressed/<widget_id>/ch-5' },  // ←  lookup, not literal
  },
  on: {
    press: {
      action: 'interact',
      params: { widgetId, surfaceId, buttonId: 'ch-5', ... },
    },
  },
}
```

When the widget declares a `scope`, the translator also adds a
`visible` condition on the root element:

```ts
// Widget.scope = { kind: "section", key: 5 }  →
elements[root].visible = {
  $state: '/focusedScope/section',
  eq: 5,
};
```

The widget now auto-hides whenever the user navigates to a different
chapter. No local state, no heuristic, no re-emit.

## Canonical state shape

The StateStore mirrors this (populated by `STATE_SNAPSHOT` and
JSON-Patched by `STATE_DELTA`):

```
{
  widgets:       { <uuid>: Widget },           // every live or terminal widget
  widgetOrder:   [ <uuid>, … ],                // scroll-tail ordering
  canvasSlot:    <uuid> | null,                // primary pin
  canvasAuxSlot: <uuid> | null,                // secondary pin
  pinnedIds:     { <uuid>: true },             // ever-pinned set
  pressed:       { <widget>: { <button>: true } },   // highlight map
  focusedScope:  { <kind>: <key> },            // e.g. { section: 5 }
}
```

Anything the UI needs to respond to lives in one of these paths; you
bind to them with `useStateValue('/widgets/…')` or via `$state`
expressions in specs.

## File tour

```
src/
  App.tsx                 # one StateStore + JSONUIProvider + SSE plumbing
  main.tsx                # entry point
  types.ts                # TS mirrors of the server's Widget / AgUiEvent shapes

  api/
    client.ts             # postMessage / postWidgetEvent / streamUrl
    sse.ts                # thin EventSource wrapper

  jr/                     # json-render integration
    catalog.ts            # closed-world component vocabulary (Zod)
    registry.tsx          # render function per catalog entry
    translate.ts          # server payload → json-render Spec
    stateBridge.ts        # AG-UI events → StateStore writes (+ widget-added cb)

  store/
    session.ts            # useReducer for chat-side state (messages, running,
                          #   codeTraces, messageFollowUps) — no widget state

  components/
    ChatPane.tsx          # message tail + inline widgets
    CanvasPane.tsx        # canvas + aux slots
    WidgetRenderer.tsx    # lifecycle shell around <Renderer>
    MessageInput.tsx      # the text box
    DebugPane.tsx         # ?debug=1 code traces

  widgets/
    McpUiWidget.tsx       # sandboxed iframe for MCP-UI widget kind
```

## How to extend it

### Add a new widget component

1. Define the shape in `src/jr/catalog.ts` with a Zod prop schema.
2. Write the React render function in `src/jr/registry.tsx`.
3. Teach `src/jr/translate.ts` how to map the server's payload shape
   to your new element type (if it's not already a passthrough).
4. Add a new branch to the agent prompt in
   `crates/gw-ui/examples/frankenstein_server.rs` so the LLM knows
   it can emit the new type.

Zod rejects unknown props on the wire, so you'll get a clear error
if the server emits something the catalog doesn't accept.

### Add a new scope kind

A scope is a key the agent uses to bucket widgets so they can be
hidden when not in focus (e.g. `{ kind: "section", key: 5 }`).

1. Agent declares `scope={"kind": "<kind>", "key": <value>}` on
   `emit_widget`. No code change needed; the backend accepts any
   `{kind, key}` shape.
2. Click payloads that should update the focus should include either
   `data.scope = {kind, key}` (preferred) or, for back-compat,
   `data.<kind> = <value>`. The adapter's `extract_scope_update`
   in `crates/gw-ui/src/ag_ui/adapter.rs` handles both.
3. The translator's `widget.scope →  visible` logic in
   `src/jr/translate.ts` is already kind-agnostic; no frontend
   change required.

### Plug in a different backend

The frontend is pure AG-UI. If you swap `frankenstein_server` for
any other adapter that speaks the same event vocabulary, this app
should render it. The two custom bits are:

- `DEBUG_CODE_EXEC` — greatwheel-only extension; unknown events are
  silently ignored, so non-greatwheel servers just won't fill the
  `?debug=1` panel.
- `Widget.scope` — a field we added to the standard `UIElement`
  shape. Servers that don't emit it get no visibility-driven
  auto-hide, but everything else works.

## Wire protocol, cheat sheet

Received over SSE at `GET /sessions/<id>/stream`:

| Event | Body |
|---|---|
| `STATE_SNAPSHOT` | Full canonical state. Emitted once on subscribe. |
| `STATE_DELTA` | `{patches: JsonPatch[]}` — RFC 6902 ops against state. |
| `TEXT_MESSAGE_CONTENT` | `{message_id, delta}` — assistant text (bulk, not token-streamed yet). |
| `RUN_STARTED` / `RUN_FINISHED` / `RUN_ERROR` | Turn lifecycle. |
| `INPUT_REQUEST` | Agent asking for user input mid-turn. |
| `DEBUG_CODE_EXEC` | `{code, stdout, is_final, error?}` — greatwheel-only. |

Sent over HTTP:

- `POST /sessions/<id>/messages` — `{content: "..."}` — user types.
- `POST /sessions/<id>/widget-events` — `{widget_id, surface_id, action, data}`
  — user clicked a Button/Card.

## Further reading

- [`docs/design-gw-ui.md`](../docs/design-gw-ui.md) — the backend
  generative-UI design doc (types, plugin, adapter).
- [`docs/design-json-render-migration.md`](../docs/design-json-render-migration.md) —
  why we migrated off the hand-rolled A2UI renderer onto
  `@json-render/react`, and the remaining phases (tool-call
  events, streaming text).
- [json-render.dev](https://json-render.dev) — the underlying
  renderer. The catalog / registry / state-binding concepts come
  from here.
- [ag-ui.com](https://docs.ag-ui.com) — the wire protocol.

## Scripts

```sh
npm run dev         # vite on :5173
npm run build       # tsc && vite build → dist/
npm run preview     # serve dist/
npm run typecheck   # tsc --noEmit
```
