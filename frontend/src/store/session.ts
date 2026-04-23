import { useReducer } from 'react';
import type { AgUiEvent, CodeTrace, UiSurfaceSnapshot, Widget } from '../types';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

export interface SessionState {
  messages: Message[];
  widgets: Record<string, Widget>;
  widgetOrder: string[];
  canvasSlot: string | null;
  canvasAuxSlot: string | null;
  running: boolean;
  codeTraces: CodeTrace[];
  /** Per-widget "currently selected" button id. Set on user click OR
   *  via the agent's `highlight_button` host call (so asking a
   *  question about Chapter 5 highlights the Chapter 5 button in the
   *  picker). */
  pressedButtonIds: Record<string, string>;
  /** messageId → widgetIds anchored to that message. Follow-up-style
   *  widgets (agent-emitted with `follow_up: true`) render under
   *  their matching message instead of in the scroll tail. */
  messageFollowUps: Record<string, string[]>;
  /** Follow-up widgets received before their anchor message existed.
   *  Drained onto the next assistant message. */
  pendingFollowUps: string[];
  /** Widget ids that have ever been pinned to a canvas slot (primary
   *  or auxiliary). Once pinned, a widget never renders in the chat
   *  scroll tail even after it's been superseded out of the slot —
   *  it either lives in canvas or is gone. */
  pinnedIds: Record<string, true>;
  /** Section-scoped widget memoisation. A widget whose Cards/Buttons
   *  all carry `data.section = N` is considered "scoped" to section
   *  N; we index `N → widget_id`. When a button bearing
   *  `data.section = N` is pressed, we auto-swap the indexed widget
   *  into `canvasAuxSlot` — so navigating back to a previously-viewed
   *  section restores its characters widget without the agent having
   *  to re-emit and re-pin it. Tactical workaround for the missing
   *  "widgets are children of a section" data model; see
   *  docs/design-gw-ui.md §14 follow-up. */
  sectionScopedWidgets: Record<number, string>;
}

type Action =
  | { type: 'hydrate'; snapshot: UiSurfaceSnapshot }
  | { type: 'append-user'; content: string }
  | { type: 'mark-running' }
  | { type: 'assistant-chunk'; message_id: string; delta: string }
  | { type: 'run-finished' }
  | { type: 'widget-emitted'; widget: Widget }
  | { type: 'widget-superseded'; old: string; new_widget: Widget }
  | { type: 'widget-resolved'; widget_id: string; data: unknown }
  | { type: 'widget-expired'; widget_id: string }
  | { type: 'widget-pinned'; widget_id: string }
  | { type: 'widget-aux-pinned'; widget_id: string }
  | { type: 'button-pressed'; widget_id: string; button_id: string }
  | { type: 'code-trace'; trace: CodeTrace };

const MAX_TRACES = 50;

const initial: SessionState = {
  messages: [],
  widgets: {},
  widgetOrder: [],
  canvasSlot: null,
  canvasAuxSlot: null,
  running: false,
  codeTraces: [],
  pressedButtonIds: {},
  messageFollowUps: {},
  pendingFollowUps: [],
  pinnedIds: {},
  sectionScopedWidgets: {},
};

/** Collect all distinct `data.section` values from Cards/Buttons in
 *  a widget payload tree. */
function collectSections(node: unknown, out: Set<number> = new Set()): Set<number> {
  if (!node || typeof node !== 'object') return out;
  const n = node as Record<string, unknown>;
  if (n.type === 'Button' || n.type === 'Card') {
    const data = n.data as { section?: unknown } | undefined;
    if (data && typeof data.section === 'number') out.add(data.section);
  }
  const children = n.children;
  if (Array.isArray(children)) {
    for (const c of children) collectSections(c, out);
  }
  return out;
}

/** Return the section index a widget is "scoped" to, or null. A
 *  widget qualifies when all its Card/Button `data.section` values
 *  point to the SAME section (excludes pickers which reference
 *  many sections). */
function sectionScopeOf(widget: Widget): number | null {
  if (!('Inline' in widget.payload)) return null;
  const sections = collectSections(widget.payload.Inline);
  if (sections.size !== 1) return null;
  return [...sections][0]!;
}

/** Walk a widget payload to find the Card/Button with a given id
 *  and return its `data.section`, if any. Lets us learn "what
 *  section does this pressed button refer to?" without committing
 *  to a particular id convention (e.g. `sec-{N}`). */
function findButtonSection(widget: Widget, buttonId: string): number | null {
  if (!('Inline' in widget.payload)) return null;
  let found: number | null = null;
  (function walk(node: unknown) {
    if (found != null || !node || typeof node !== 'object') return;
    const n = node as Record<string, unknown>;
    if ((n.type === 'Button' || n.type === 'Card') && n.id === buttonId) {
      const data = n.data as { section?: unknown } | undefined;
      if (data && typeof data.section === 'number') found = data.section;
      return;
    }
    const children = n.children;
    if (Array.isArray(children)) for (const c of children) walk(c);
  })(widget.payload.Inline);
  return found;
}

function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case 'hydrate': {
      const widgets: Record<string, Widget> = {};
      for (const w of action.snapshot.widgets) widgets[w.id] = w;
      const pinnedIds: Record<string, true> = {};
      if (action.snapshot.surface.canvas_slot) {
        pinnedIds[action.snapshot.surface.canvas_slot] = true;
      }
      if (action.snapshot.surface.canvas_aux_slot) {
        pinnedIds[action.snapshot.surface.canvas_aux_slot] = true;
      }
      const sectionScopedWidgets: Record<number, string> = {};
      for (const w of action.snapshot.widgets) {
        const n = sectionScopeOf(w);
        if (n != null) sectionScopedWidgets[n] = w.id;
      }
      return {
        ...state,
        widgets,
        widgetOrder: [...action.snapshot.surface.widget_order],
        canvasSlot: action.snapshot.surface.canvas_slot,
        canvasAuxSlot: action.snapshot.surface.canvas_aux_slot ?? null,
        pinnedIds,
        sectionScopedWidgets,
      };
    }
    case 'append-user':
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: crypto.randomUUID(), role: 'user', content: action.content },
        ],
        running: true,
      };
    case 'mark-running':
      return { ...state, running: true };
    case 'assistant-chunk': {
      // If the last message is from the assistant with this id, append.
      // Otherwise, start a new assistant message. Either way the agent
      // has committed to a response, so the "thinking" indicator can
      // come down — independent of whether RUN_FINISHED has arrived.
      const last = state.messages[state.messages.length - 1];
      if (last && last.role === 'assistant' && last.id === action.message_id) {
        const updated = { ...last, content: last.content + action.delta };
        return {
          ...state,
          messages: [...state.messages.slice(0, -1), updated],
          running: false,
        };
      }
      // New assistant message — drain any pending follow-up widgets
      // onto it so buttons emitted before FINAL anchor correctly.
      let messageFollowUps = state.messageFollowUps;
      let pendingFollowUps = state.pendingFollowUps;
      if (pendingFollowUps.length > 0) {
        messageFollowUps = {
          ...messageFollowUps,
          [action.message_id]: pendingFollowUps,
        };
        pendingFollowUps = [];
      }
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: action.message_id, role: 'assistant', content: action.delta },
        ],
        running: false,
        messageFollowUps,
        pendingFollowUps,
      };
    }
    case 'run-finished':
      return { ...state, running: false };
    case 'widget-emitted': {
      const w = action.widget;
      if (state.widgets[w.id]) return state;
      const scope = sectionScopeOf(w);
      const sectionScopedWidgets =
        scope != null
          ? { ...state.sectionScopedWidgets, [scope]: w.id }
          : state.sectionScopedWidgets;
      const baseState = {
        ...state,
        widgets: { ...state.widgets, [w.id]: w },
        widgetOrder: [...state.widgetOrder, w.id],
        sectionScopedWidgets,
      };
      if (!w.follow_up) return baseState;
      // Anchor follow-up widgets to a message.
      //
      //   - If we're mid-turn (`running` is true), the agent's own
      //     reply has not arrived yet — buffer to pendingFollowUps
      //     and drain on the next new assistant message. Attaching
      //     to the prior assistant message here would misplace the
      //     widget under the previous turn's answer.
      //   - If we're idle, there's no forthcoming reply to anchor
      //     to; attach to the latest assistant message in place.
      if (state.running) {
        return {
          ...baseState,
          pendingFollowUps: [...state.pendingFollowUps, w.id],
        };
      }
      const lastAssistant = [...state.messages]
        .reverse()
        .find((m) => m.role === 'assistant');
      if (lastAssistant) {
        const prev = state.messageFollowUps[lastAssistant.id] ?? [];
        return {
          ...baseState,
          messageFollowUps: {
            ...state.messageFollowUps,
            [lastAssistant.id]: [...prev, w.id],
          },
        };
      }
      return {
        ...baseState,
        pendingFollowUps: [...state.pendingFollowUps, w.id],
      };
    }
    case 'widget-superseded': {
      const oldW = state.widgets[action.old];
      const newW = action.new_widget;
      const widgets = { ...state.widgets };
      if (oldW) widgets[action.old] = { ...oldW, state: 'Superseded' };
      widgets[newW.id] = newW;
      const widgetOrder = state.widgetOrder.includes(newW.id)
        ? state.widgetOrder
        : [...state.widgetOrder, newW.id];
      return { ...state, widgets, widgetOrder };
    }
    case 'widget-resolved': {
      const w = state.widgets[action.widget_id];
      if (!w) return state;
      return {
        ...state,
        widgets: {
          ...state.widgets,
          [w.id]: { ...w, state: 'Resolved', resolution: action.data },
        },
      };
    }
    case 'widget-expired': {
      const w = state.widgets[action.widget_id];
      if (!w) return state;
      return {
        ...state,
        widgets: { ...state.widgets, [w.id]: { ...w, state: 'Expired' } },
      };
    }
    case 'widget-pinned':
      return {
        ...state,
        canvasSlot: action.widget_id,
        pinnedIds: { ...state.pinnedIds, [action.widget_id]: true },
      };
    case 'widget-aux-pinned':
      return {
        ...state,
        canvasAuxSlot: action.widget_id,
        pinnedIds: { ...state.pinnedIds, [action.widget_id]: true },
      };
    case 'button-pressed': {
      const pressedButtonIds = {
        ...state.pressedButtonIds,
        [action.widget_id]: action.button_id,
      };
      // If the pressed button is section-scoped (its data.section is
      // set) and we've previously indexed a section-scoped widget for
      // that section, auto-swap the aux canvas slot to it. Lets you
      // re-navigate to a past chapter and have its characters widget
      // reappear without the agent re-emitting.
      const widget = state.widgets[action.widget_id];
      if (widget) {
        const sectionIdx = findButtonSection(widget, action.button_id);
        if (sectionIdx != null) {
          const scopedId = state.sectionScopedWidgets[sectionIdx];
          if (scopedId && scopedId !== state.canvasAuxSlot) {
            return {
              ...state,
              pressedButtonIds,
              canvasAuxSlot: scopedId,
              pinnedIds: { ...state.pinnedIds, [scopedId]: true },
            };
          }
        }
      }
      return { ...state, pressedButtonIds };
    }
    case 'code-trace': {
      const nextTraces = [...state.codeTraces, action.trace];
      if (nextTraces.length > MAX_TRACES) {
        nextTraces.splice(0, nextTraces.length - MAX_TRACES);
      }
      return { ...state, codeTraces: nextTraces };
    }
  }
}

export function useSessionStore() {
  const [state, dispatch] = useReducer(reducer, initial);
  return {
    state,
    hydrate: (snapshot: UiSurfaceSnapshot) => dispatch({ type: 'hydrate', snapshot }),
    appendUser: (content: string) => dispatch({ type: 'append-user', content }),
    markRunning: () => dispatch({ type: 'mark-running' }),
    pressButton: (widgetId: string, buttonId: string) =>
      dispatch({ type: 'button-pressed', widget_id: widgetId, button_id: buttonId }),
    ingest: (ev: AgUiEvent) => {
      const action = agUiToAction(ev);
      if (action) dispatch(action);
    },
  };
}

function agUiToAction(ev: AgUiEvent): Action | null {
  switch (ev.type) {
    case 'TEXT_MESSAGE_CONTENT':
      return { type: 'assistant-chunk', message_id: ev.message_id, delta: ev.delta };
    case 'RUN_STARTED':
      return { type: 'mark-running' };
    case 'RUN_FINISHED':
      return { type: 'run-finished' };
    case 'RUN_ERROR':
      // Reuse assistant-chunk so the error appears in the scroll tail as a
      // system-style note. A later phase should give errors their own
      // surface; for now this matches how INPUT_REQUEST is rendered.
      return {
        type: 'assistant-chunk',
        message_id: crypto.randomUUID(),
        delta: `⚠ ${ev.message}`,
      };
    case 'INPUT_REQUEST':
      return { type: 'assistant-chunk', message_id: crypto.randomUUID(), delta: ev.prompt };
    case 'UI_EVENT':
      return { type: 'widget-emitted', widget: ev.widget };
    case 'STATE_SNAPSHOT':
    case 'STATE_DELTA':
      // Phase 2: ignore. We still hydrate from /surface and consume
      // UI_PATCH for live updates. Phase 3 will flip both on.
      return null;
    case 'UI_PATCH':
      switch (ev.patch.kind) {
        case 'supersede':
          return { type: 'widget-superseded', old: ev.patch.old, new_widget: ev.patch.new };
        case 'resolve':
          return { type: 'widget-resolved', widget_id: ev.patch.widget_id, data: ev.patch.data };
        case 'expire':
          return { type: 'widget-expired', widget_id: ev.patch.widget_id };
        case 'pin':
          return { type: 'widget-pinned', widget_id: ev.patch.widget_id };
        case 'pin_aux':
          return { type: 'widget-aux-pinned', widget_id: ev.patch.widget_id };
        case 'highlight':
          return {
            type: 'button-pressed',
            widget_id: ev.patch.widget_id,
            button_id: ev.patch.button_id,
          };
      }
    case 'DEBUG_CODE_EXEC':
      return {
        type: 'code-trace',
        trace: {
          id: crypto.randomUUID(),
          code: ev.code,
          stdout: ev.stdout,
          is_final: ev.is_final,
          error: ev.error,
          at: Date.now(),
        },
      };
  }
}
