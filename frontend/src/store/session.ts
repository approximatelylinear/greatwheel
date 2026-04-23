import { useReducer } from 'react';
import type { AgUiEvent, CodeTrace, Widget } from '../types';

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
}

type Action =
  | { type: 'append-user'; content: string }
  | { type: 'assistant-chunk'; message_id: string; delta: string }
  | { type: 'run-finished' }
  | { type: 'widget-emitted'; widget: Widget }
  | { type: 'widget-superseded'; old: string; new_widget: Widget }
  | { type: 'widget-resolved'; widget_id: string; data: unknown }
  | { type: 'widget-expired'; widget_id: string }
  | { type: 'widget-pinned'; widget_id: string }
  | { type: 'code-trace'; trace: CodeTrace };

const MAX_TRACES = 50;

const initial: SessionState = {
  messages: [],
  widgets: {},
  widgetOrder: [],
  canvasSlot: null,
  running: false,
  codeTraces: [],
};

function reducer(state: SessionState, action: Action): SessionState {
  switch (action.type) {
    case 'append-user':
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: crypto.randomUUID(), role: 'user', content: action.content },
        ],
        running: true,
      };
    case 'assistant-chunk': {
      // If the last message is from the assistant with this id, append.
      // Otherwise, start a new assistant message.
      const last = state.messages[state.messages.length - 1];
      if (last && last.role === 'assistant' && last.id === action.message_id) {
        const updated = { ...last, content: last.content + action.delta };
        return { ...state, messages: [...state.messages.slice(0, -1), updated] };
      }
      return {
        ...state,
        messages: [
          ...state.messages,
          { id: action.message_id, role: 'assistant', content: action.delta },
        ],
      };
    }
    case 'run-finished':
      return { ...state, running: false };
    case 'widget-emitted': {
      const w = action.widget;
      if (state.widgets[w.id]) return state;
      return {
        ...state,
        widgets: { ...state.widgets, [w.id]: w },
        widgetOrder: [...state.widgetOrder, w.id],
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
    case 'code-trace': {
      const next = [...state.codeTraces, action.trace];
      if (next.length > MAX_TRACES) next.splice(0, next.length - MAX_TRACES);
      return { ...state, codeTraces: next };
    }
  }
}

export function useSessionStore() {
  const [state, dispatch] = useReducer(reducer, initial);
  return {
    state,
    appendUser: (content: string) => dispatch({ type: 'append-user', content }),
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
