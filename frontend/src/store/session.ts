import { useReducer } from 'react';
import type { AgUiEvent, CodeTrace, Widget } from '../types';

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}

/**
 * Phase 3: the reducer holds only chat-side state. Widget records,
 * canvas slots, pressed highlights, and focused scope live in the
 * json-render StateStore (populated by STATE_SNAPSHOT / STATE_DELTA
 * via `src/jr/stateBridge.ts`). Components that need widget data
 * read it through json-render's state hooks.
 *
 * What stays here:
 *   - messages / running / codeTraces      — pure chat surface
 *   - messageFollowUps / pendingFollowUps  — UI convention tying
 *     follow-up widgets to a specific assistant message; the server
 *     doesn't model message ids so this has nowhere else to live
 */
export interface SessionState {
  messages: Message[];
  running: boolean;
  codeTraces: CodeTrace[];
  /** messageId → widgetIds anchored to that message. Follow-up-style
   *  widgets (agent-emitted with `follow_up: true`) render under
   *  their matching message instead of in the scroll tail. */
  messageFollowUps: Record<string, string[]>;
  /** Follow-up widgets received before their anchor message existed.
   *  Drained onto the next assistant message. */
  pendingFollowUps: string[];
}

type Action =
  | { type: 'append-user'; content: string }
  | { type: 'mark-running' }
  | { type: 'assistant-chunk'; message_id: string; delta: string }
  | { type: 'run-finished' }
  | { type: 'widget-emitted'; widget: Widget }
  | { type: 'code-trace'; trace: CodeTrace };

const MAX_TRACES = 50;

const initial: SessionState = {
  messages: [],
  running: false,
  codeTraces: [],
  messageFollowUps: {},
  pendingFollowUps: [],
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
      // The widget itself lives in the json-render StateStore now.
      // We only track follow-up anchoring here, since that's a
      // chat-side concern (widget → message linkage).
      const w = action.widget;
      if (!w.follow_up) return state;
      if (state.running) {
        // Buffer — no assistant message has arrived for this turn yet,
        // attaching to the prior turn's message would misplace it.
        return {
          ...state,
          pendingFollowUps: [...state.pendingFollowUps, w.id],
        };
      }
      const lastAssistant = [...state.messages]
        .reverse()
        .find((m) => m.role === 'assistant');
      if (lastAssistant) {
        const prev = state.messageFollowUps[lastAssistant.id] ?? [];
        return {
          ...state,
          messageFollowUps: {
            ...state.messageFollowUps,
            [lastAssistant.id]: [...prev, w.id],
          },
        };
      }
      return {
        ...state,
        pendingFollowUps: [...state.pendingFollowUps, w.id],
      };
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
    appendUser: (content: string) => dispatch({ type: 'append-user', content }),
    markRunning: () => dispatch({ type: 'mark-running' }),
    /** Called by the state bridge when a new widget lands in
     *  `/widgets/<id>`. Drives the follow-up anchoring heuristic. */
    widgetAdded: (widget: Widget) =>
      dispatch({ type: 'widget-emitted', widget }),
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
      return {
        type: 'assistant-chunk',
        message_id: crypto.randomUUID(),
        delta: `⚠ ${ev.message}`,
      };
    case 'INPUT_REQUEST':
      return { type: 'assistant-chunk', message_id: crypto.randomUUID(), delta: ev.prompt };
    case 'STATE_SNAPSHOT':
    case 'STATE_DELTA':
      // Consumed by the state bridge (widget/canvas/pressed/focus
      // live in the json-render StateStore). Follow-up anchoring is
      // triggered from the bridge's onWidgetAdded callback via
      // `widgetAdded` below, not through this reducer path.
      return null;
    case 'TOOL_CALL_START':
    case 'TOOL_CALL_ARGS':
    case 'TOOL_CALL_END':
      // Host-function tool-call events. Currently unrendered; a
      // future commit surfaces them in DebugPane alongside the
      // existing code-trace stream.
      return null;
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
