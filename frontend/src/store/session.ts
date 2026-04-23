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
  | { type: 'button-pressed'; widget_id: string; button_id: string }
  | { type: 'code-trace'; trace: CodeTrace };

const MAX_TRACES = 50;

const initial: SessionState = {
  messages: [],
  widgets: {},
  widgetOrder: [],
  canvasSlot: null,
  running: false,
  codeTraces: [],
  pressedButtonIds: {},
  messageFollowUps: {},
  pendingFollowUps: [],
};

function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case 'hydrate': {
      const widgets: Record<string, Widget> = {};
      for (const w of action.snapshot.widgets) widgets[w.id] = w;
      return {
        ...state,
        widgets,
        widgetOrder: [...action.snapshot.surface.widget_order],
        canvasSlot: action.snapshot.surface.canvas_slot,
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
      const baseState = {
        ...state,
        widgets: { ...state.widgets, [w.id]: w },
        widgetOrder: [...state.widgetOrder, w.id],
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
      return { ...state, canvasSlot: action.widget_id };
    case 'button-pressed':
      return {
        ...state,
        pressedButtonIds: {
          ...state.pressedButtonIds,
          [action.widget_id]: action.button_id,
        },
      };
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
    ingest: (ev: AgUiEvent) => dispatch(agUiToAction(ev)),
  };
}

function agUiToAction(ev: AgUiEvent): Action {
  switch (ev.type) {
    case 'TEXT_MESSAGE_CONTENT':
      return { type: 'assistant-chunk', message_id: ev.message_id, delta: ev.delta };
    case 'RUN_FINISHED':
      return { type: 'run-finished' };
    case 'INPUT_REQUEST':
      return { type: 'assistant-chunk', message_id: crypto.randomUUID(), delta: ev.prompt };
    case 'UI_EVENT':
      return { type: 'widget-emitted', widget: ev.widget };
    case 'STATE_DELTA':
      switch (ev.patch.kind) {
        case 'supersede':
          return { type: 'widget-superseded', old: ev.patch.old, new_widget: ev.patch.new };
        case 'resolve':
          return { type: 'widget-resolved', widget_id: ev.patch.widget_id, data: ev.patch.data };
        case 'expire':
          return { type: 'widget-expired', widget_id: ev.patch.widget_id };
        case 'pin':
          return { type: 'widget-pinned', widget_id: ev.patch.widget_id };
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
