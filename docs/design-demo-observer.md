# Design: greatwheel observer

**Status:** Drafted 2026-04-24.

## 1. Goal

A separate frontend page that watches an existing greatwheel session
and visualises what's happening inside it. Shows: turn timeline,
tool-call graph, widget lineage, live STATE_DELTA feed, token + cost
rollups. Purely diagnostic — never emits to the watched session.

Ships first because it **requires zero backend changes**. It's a
client-side consumer of the AG-UI protocol we already speak.
Visualising the agent's behaviour on the same page as the agent's
output also makes the demo self-explanatory: you see both what the
agent is doing and what the agent built.

## 2. What it proves

- **AG-UI wire format is sufficient to reconstruct system behaviour.**
  A client with no privileged access to greatwheel internals — just
  `/stream` — can render a deeply informative diagnostic view. Proof
  that we are actually shipping a protocol, not a private wire
  format.
- **Two clients, one session.** The demo runs the Frankenstein UI
  *and* the observer against the same session UUID and demonstrates
  they stay in lockstep. Models what multi-client observability
  looks like.
- **Tool-call graph + state changes are navigable.** Scope-driven
  visibility works for ops-style views, not just chat.

## 3. Architecture

```
                  ┌─ Frankenstein demo tab ─┐
                  │  /?session=<uuid>       │
                  └───────────┬─────────────┘
                              │ /stream
                              ▼
                   ┌────────────────────┐
                   │ gw-ui AG-UI adapter│ ←— session <uuid>
                   └────────┬───────────┘
                            │ /stream (same events)
                            ▼
                  ┌─ Observer tab ──────────┐
                  │  /?observe=<uuid>       │
                  │  (read-only consumer)   │
                  └─────────────────────────┘
```

The observer is a new route: `/?observe=<uuid>` (or `/observer.html`).
It:

1. Opens an EventSource against `/sessions/<uuid>/stream`.
2. Maintains its **own** local state — a log of every event in order.
3. Renders that log as a dashboard.

No writes to the watched session. No interaction. It's a reader.

## 4. State shape (observer-local)

Not part of the AG-UI protocol — this is the observer's internal
React state, built from ingested events:

```ts
interface ObserverState {
  turns: {
    id: string;           // generated per RUN_STARTED
    startedAt: number;
    completedAt?: number;
    status: 'running' | 'done' | 'error';
    error?: string;
    toolCallIds: string[];
    textMessages: string[]; // message_ids
  }[];
  toolCalls: Map<string, {
    id: string;
    name: string;
    args?: unknown;
    result?: unknown;
    error?: string;
    startedAt: number;
    completedAt?: number;
    turnId: string | null;  // which turn it belongs to
  }>;
  messages: Map<string, { id: string; started: number; text: string; ended?: number }>;
  widgets: Map<string, Widget>;        // mirrored from STATE_DELTA
  widgetOrder: string[];
  canvasSlot: string | null;
  canvasAuxSlot: string | null;
  focusedScope: Record<string, unknown>;
  patchLog: { at: number; patches: JsonPatchOp[] }[]; // raw patch history
  totalTokens: { input: number; output: number };    // phase-later
}
```

Turn/toolCall/message are reconstructed from the existing events:
`RUN_STARTED/FINISHED/ERROR` bracket a turn; `TOOL_CALL_START/END`
bracket a tool call; `TEXT_MESSAGE_START/END` bracket a message.
Everything keyed by the correlation ids we're already emitting.

## 5. Visualisations

Five panels, each one rendered with existing widget primitives where
possible; where we need custom rendering, add it in
`src/components/observer/`.

### 5.1 Turn timeline
Horizontal stack of bars, one per turn, left-to-right. Each bar
shows elapsed time, error state, and number of tool calls inside.
Click a turn → scope = turn, other panels filter.

### 5.2 Tool-call graph
Per selected turn: nodes for each tool call in chronological order,
grouped by host-fn name (emit_widget / pin_to_canvas / etc.).
Edges show causal proximity (call N+1 was dispatched in the same
code block as N). For the first cut, a layered vertical list is
fine; upgrade to force-directed later.

### 5.3 Widget lineage
Tree view of widgets: supersede edges shown as parent→child. Each
node shows widget id (short), kind, state, scope. Clicking a widget
highlights the tool calls that produced/mutated it.

### 5.4 State-delta log
Reverse-chronological feed of every `STATE_DELTA` event with the
paths and (truncated) values. Searchable by path prefix. Essentially
the debug-pane-for-wire, not for agent code.

### 5.5 Session summary
Aggregate stats card: turn count, total tool calls, widget count,
focused scope, session age. Pinned to canvas primary.

## 6. Routing

Minimal addition to `main.tsx`:

```tsx
const url = new URL(window.location.href);
const observe = url.searchParams.get('observe');
createRoot(root).render(
  <StrictMode>
    {observe ? <Observer sessionId={observe} /> : <App />}
  </StrictMode>
);
```

`Observer` component lives in `src/observer/`. Same CSS design system
as the main app.

## 7. File layout

```
src/observer/
  Observer.tsx            # entry — opens SSE, ingest loop
  ingest.ts               # event → ObserverState reducer
  panes/
    TurnTimeline.tsx
    ToolCallGraph.tsx
    WidgetLineage.tsx
    StateDeltaLog.tsx
    SessionSummary.tsx
```

No json-render registry changes — the observer UI is hand-rolled
React since it's not agent-authored.

## 8. User flow

1. Run the Frankenstein demo in one tab with `?session=X`.
2. Open another tab with `?observe=X`.
3. Interact with Frankenstein; watch the observer update in real time.
   - Each chapter click spawns a turn bar.
   - Tool-call graph fills with `get_section`, `emit_widget`,
     `pin_to_canvas`, `highlight_button`.
   - Widget lineage grows; superseded widgets fade.
   - State-delta log ticks every write.

## 9. What's deliberately out of scope

- **Persistence.** Observer starts empty each time; no replay from
  backend history. Backend doesn't keep the event log anyway.
  Adding that is a separate "session replay" feature.
- **Filtering the event stream.** Observer receives everything for
  that session. Filter on the client.
- **Multiple sessions.** One observer = one session.
- **Performance at scale.** Fine for hundreds of events; past
  thousands, the patch log needs virtualisation.

## 10. Open questions

- **Observer-generated widgets.** Could the observer itself be a
  greatwheel session, running an agent that watches another? That's
  the "meta" version and needs a `watch_session` host fn. Interesting
  but much more machinery. **Proposal:** start with the pure-client
  version; revisit if the extra ceremony is worth it.
- **Shared types.** The observer's `ObserverState` and the main
  app's session reducer cover overlapping ground (widgets,
  messages). Worth extracting? **Proposal:** copy-paste for v1;
  extract to `src/shared/` only if the divergence forces it.
- **Time travel.** "Show me the state at t=…". Requires keeping
  every intermediate state snapshot. Nice but defer.

## 11. Acceptance

1. Observer tab opens, connects to a live session, and within one
   interaction shows a populated turn timeline.
2. Every tool call the agent makes appears in the tool-call graph
   with args + result.
3. Widget lineage correctly shows supersede edges.
4. STATE_DELTA log entries match 1:1 with events seen in DevTools
   Network tab.
5. Observer survives SSE reconnect (drop network, restore, continue
   receiving events) — because STATE_SNAPSHOT re-hydrates on
   subscribe.
6. Zero writes to the target session — verifiable by watching the
   main app and confirming nothing new arrives there from the
   observer.

## 12. Scope estimate

1–2 days for a complete first cut. No backend work. ~800 lines of
new TS including five visualisation panels and the ingest reducer.
CSS additions modest — palette and layout primitives already exist.
