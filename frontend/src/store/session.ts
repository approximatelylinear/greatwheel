import { useReducer } from 'react';
import type {
  AgUiEvent,
  CodeTrace,
  DebugSpineSegment,
  ToolCall,
  Widget,
} from '../types';

/**
 * One spine diagnostic event for the DebugPane's "spine" tab.
 * Either an entry-level extraction landed (`entry-extracted`) or a
 * re-segment pass committed (`segments-updated`). The pane renders
 * a chronological list so the user can see the spine pipeline in
 * motion turn-by-turn.
 */
export type SpineDebugEvent =
  | {
      kind: 'entry-extracted';
      id: string;
      at: number;
      entry_id: string;
      entity_count: number;
      relation_count: number;
    }
  | {
      kind: 'segments-updated';
      id: string;
      at: number;
      session_id: string;
      segments: DebugSpineSegment[];
    };

/** One inbound AG-UI event as captured for the DebugPane's "sse" tab.
 *  Held as a thin summary (event type + one-line description) so the
 *  pane stays readable across a few hundred events. Full payloads
 *  live in the StateStore / reducer; this is just an ordering aid. */
export interface SseDebugEvent {
  id: string;
  at: number;
  type: string;
  summary: string;
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  /** Persistent `session_entries.id` of the entry that produced this
   *  message. Sent for assistant messages via `TEXT_MESSAGE_START.entry_id`;
   *  user messages don't carry it today (the entry is appended server-
   *  side after the client locally optimistic-renders the user's
   *  message). The spine rail uses this to anchor segments to chat
   *  rows via `data-entry-id`. */
  entryId?: string;
  /** RFC 3339 timestamp of the underlying `session_entries` row.
   *  Populated by `hydrate-transcript`; absent for messages appended
   *  locally during the live session (those are positioned by
   *  array-append order, not by timestamp). Used to rebuild
   *  `messageFollowUps` after refresh, since the reducer-only
   *  anchoring map is otherwise lost on reload. */
  createdAt?: string;
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
  /** Insertion-ordered tool calls (each is a host-function invocation
   *  the agent dispatched). START/ARGS/END events accumulate into the
   *  same record keyed by `tool_call_id`. Surfaced in the debug pane. */
  toolCalls: ToolCall[];
  /** messageId → widgetIds anchored to that message. Follow-up-style
   *  widgets (agent-emitted with `follow_up: true`) render under
   *  their matching message instead of in the scroll tail. */
  messageFollowUps: Record<string, string[]>;
  /** Follow-up widgets received before their anchor message existed.
   *  Drained onto the next assistant message. */
  pendingFollowUps: string[];
  /** Insertion-ordered spine pipeline events. Surfaced in the debug
   *  pane's "spine" tab so the user can see extraction + re-segment
   *  passes as they fire. Capped to keep memory bounded. */
  spineEvents: SpineDebugEvent[];
  /** Insertion-ordered AG-UI SSE events with one-line summaries.
   *  Surfaced in the debug pane's "sse" tab — meant for tracing
   *  ordering bugs (e.g. unexpected RUN_STARTED after RUN_FINISHED).
   *  Capped at MAX_SSE_EVENTS. */
  sseEvents: SseDebugEvent[];
  /** Buffer mapping `message_id → entry_id` populated by
   *  `TEXT_MESSAGE_START`. Drained on the matching `assistant-chunk`
   *  to stamp `Message.entryId` so the spine can anchor segments to
   *  chat rows. Server emits START before any CONTENT, so the lookup
   *  is always populated by the time the message is created. */
  pendingAssistantEntryIds: Record<string, string>;
}

type Action =
  | { type: 'append-user'; content: string }
  | { type: 'hydrate-transcript'; messages: Message[] }
  | { type: 'rehydrate-follow-ups'; map: Record<string, string[]> }
  | { type: 'mark-running' }
  | { type: 'text-message-start'; message_id: string; entry_id?: string }
  | { type: 'user-message-anchor'; entry_id: string }
  | { type: 'assistant-chunk'; message_id: string; delta: string }
  | { type: 'run-finished' }
  | { type: 'widget-emitted'; widget: Widget }
  | { type: 'code-trace'; trace: CodeTrace }
  | { type: 'tool-call-start'; id: string; name: string; at: number }
  | { type: 'tool-call-args'; id: string; args: unknown }
  | {
      type: 'tool-call-end';
      id: string;
      result?: unknown;
      error?: string;
      at: number;
    }
  | { type: 'spine-event'; event: SpineDebugEvent }
  | { type: 'sse-event'; event: SseDebugEvent };

const MAX_TRACES = 50;
const MAX_TOOL_CALLS = 50;
const MAX_SPINE_EVENTS = 100;
const MAX_SSE_EVENTS = 200;

const initial: SessionState = {
  messages: [],
  running: false,
  codeTraces: [],
  toolCalls: [],
  messageFollowUps: {},
  pendingFollowUps: [],
  spineEvents: [],
  sseEvents: [],
  pendingAssistantEntryIds: {},
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
    case 'hydrate-transcript': {
      // Replace the message list wholesale with the server-shipped
      // transcript. Called once on session (re)load before any new
      // events stream in. If the user has already typed a message
      // locally (unlikely but possible during the brief async
      // window) it would get clobbered — we keep the rule simple
      // and let the server-side persisted log win.
      return { ...state, messages: action.messages };
    }
    case 'rehydrate-follow-ups': {
      // After a refresh, the live-built `messageFollowUps` map is
      // empty (it's reducer-only state, never persisted). Without
      // this rebuild, every follow-up widget falls through to inline
      // rendering at the bottom of the chat, and the pin-ack
      // messages they should sit beneath get demoted to `.message-log`
      // styling (since `followUps.length === 0` is part of the
      // log-line test). The caller computes the map from the
      // hydrated messages + the widgets in the StateStore using
      // timestamps; we just install it. Safe to call repeatedly —
      // it's a wholesale replace, idempotent for stable inputs.
      return { ...state, messageFollowUps: action.map };
    }
    case 'mark-running':
      return { ...state, running: true };
    case 'user-message-anchor': {
      // FIFO-match the next unstamped user-message row. The local
      // `appendUser` runs optimistically before any server round-trip,
      // so the order of locally-appended user messages matches the
      // order of server-side persistence. Server only emits this for
      // typed user messages (widget-event synthetic messages skip the
      // emit), so the FIFO across both sides stays aligned.
      const idx = state.messages.findIndex(
        (m) => m.role === 'user' && !m.entryId,
      );
      if (idx < 0) return state;
      const next = state.messages.slice();
      next[idx] = { ...next[idx]!, entryId: action.entry_id };
      return { ...state, messages: next };
    }
    case 'text-message-start': {
      // Buffer the entry_id keyed by message_id; the matching
      // 'assistant-chunk' creates the Message and consumes it. If the
      // server omitted entry_id (synthetic emit, no tree entry), we
      // skip the buffer entry — the message just won't be anchorable.
      if (!action.entry_id) return state;
      return {
        ...state,
        pendingAssistantEntryIds: {
          ...state.pendingAssistantEntryIds,
          [action.message_id]: action.entry_id,
        },
      };
    }
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
      // Drain entry_id buffered by the prior TEXT_MESSAGE_START so
      // the new message carries the data-entry-id anchor the spine
      // uses for timeline alignment.
      const entryId = state.pendingAssistantEntryIds[action.message_id];
      let pendingAssistantEntryIds = state.pendingAssistantEntryIds;
      if (entryId !== undefined) {
        pendingAssistantEntryIds = { ...pendingAssistantEntryIds };
        delete pendingAssistantEntryIds[action.message_id];
      }
      return {
        ...state,
        messages: [
          ...state.messages,
          {
            id: action.message_id,
            role: 'assistant',
            content: action.delta,
            entryId,
          },
        ],
        running: false,
        messageFollowUps,
        pendingFollowUps,
        pendingAssistantEntryIds,
      };
    }
    case 'run-finished':
      return { ...state, running: false };
    case 'widget-emitted': {
      // The widget itself lives in the json-render StateStore now.
      // We only track follow-up anchoring here, since that's a
      // chat-side concern (widget → message linkage).
      const w = action.widget;
      // Terminal/drawer widgets (KbDocWiki, KbClusterWiki) signal
      // "user, look at this dedicated surface." Once one of those
      // lands, the visible turn is over from the user's perspective,
      // even if gw-loop hasn't sent RUN_FINISHED yet (the agent may
      // still be in a follow-up iteration or a stuck loop). Hide the
      // typing indicator and re-enable input. RUN_FINISHED, when it
      // arrives, is a no-op since `running` is already false.
      const terminalKinds = new Set(['KbDocWiki', 'KbClusterWiki']);
      const isTerminal =
        w.kind === 'A2ui'
        && 'Inline' in w.payload
        && terminalKinds.has(
          ((w.payload as { Inline: { type?: string } }).Inline?.type ?? ''),
        );
      const baseState = isTerminal ? { ...state, running: false } : state;
      if (!w.follow_up) return baseState;
      if (state.running) {
        // Buffer — no assistant message has arrived for this turn yet,
        // attaching to the prior turn's message would misplace it.
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
    case 'code-trace': {
      const nextTraces = [...state.codeTraces, action.trace];
      if (nextTraces.length > MAX_TRACES) {
        nextTraces.splice(0, nextTraces.length - MAX_TRACES);
      }
      return { ...state, codeTraces: nextTraces };
    }
    case 'tool-call-start': {
      const next: ToolCall[] = [
        ...state.toolCalls,
        {
          id: action.id,
          name: action.name,
          args: undefined,
          status: 'running',
          startedAt: action.at,
        },
      ];
      if (next.length > MAX_TOOL_CALLS) {
        next.splice(0, next.length - MAX_TOOL_CALLS);
      }
      return { ...state, toolCalls: next };
    }
    case 'tool-call-args': {
      const idx = state.toolCalls.findIndex((c) => c.id === action.id);
      if (idx < 0) return state;
      const next = state.toolCalls.slice();
      next[idx] = { ...next[idx]!, args: action.args };
      return { ...state, toolCalls: next };
    }
    case 'tool-call-end': {
      const idx = state.toolCalls.findIndex((c) => c.id === action.id);
      if (idx < 0) return state;
      const next = state.toolCalls.slice();
      next[idx] = {
        ...next[idx]!,
        result: action.result,
        error: action.error,
        status: action.error ? 'error' : 'done',
        completedAt: action.at,
      };
      return { ...state, toolCalls: next };
    }
    case 'spine-event': {
      const next = [...state.spineEvents, action.event];
      if (next.length > MAX_SPINE_EVENTS) {
        next.splice(0, next.length - MAX_SPINE_EVENTS);
      }
      return { ...state, spineEvents: next };
    }
    case 'sse-event': {
      const next = [...state.sseEvents, action.event];
      if (next.length > MAX_SSE_EVENTS) {
        next.splice(0, next.length - MAX_SSE_EVENTS);
      }
      return { ...state, sseEvents: next };
    }
  }
}

/** Dev-only wrapper that logs every action that changes the
 *  `running` flag. Helps trace stuck-typing-indicator bugs: each line
 *  is `[running] false → true (action: mark-running)`, which surfaces
 *  the exact event that flipped state. No-op in production builds. */
function reducerWithRunningTrace(
  state: SessionState,
  action: Action,
): SessionState {
  const next = reducer(state, action);
  if (import.meta.env.DEV && next.running !== state.running) {
    // eslint-disable-next-line no-console
    console.log(
      `[running] ${state.running} → ${next.running} (action: ${action.type})`,
    );
  }
  return next;
}

export function useSessionStore() {
  const [state, dispatch] = useReducer(reducerWithRunningTrace, initial);
  return {
    state,
    appendUser: (content: string) => dispatch({ type: 'append-user', content }),
    /** Replace the message list with a server-shipped transcript.
     *  App.tsx fires this once on session load via `fetchTranscript`. */
    hydrateTranscript: (messages: Message[]) =>
      dispatch({ type: 'hydrate-transcript', messages }),
    /** Replace the live-built follow-up anchoring map. App.tsx fires
     *  this after both transcript and STATE_SNAPSHOT have landed, so
     *  follow_up widgets re-anchor to their assistant message instead
     *  of piling up in the inline tail. */
    rehydrateFollowUps: (map: Record<string, string[]>) =>
      dispatch({ type: 'rehydrate-follow-ups', map }),
    markRunning: () => dispatch({ type: 'mark-running' }),
    /** Called by the state bridge when a new widget lands in
     *  `/widgets/<id>`. Drives the follow-up anchoring heuristic. */
    widgetAdded: (widget: Widget) =>
      dispatch({ type: 'widget-emitted', widget }),
    ingest: (ev: AgUiEvent) => {
      // Log every inbound SSE event into the debug feed, regardless
      // of whether it produces a reducer action. STATE_SNAPSHOT /
      // STATE_DELTA are consumed by the StateStore bridge (not the
      // reducer) but still belong in the ordering trace.
      dispatch({
        type: 'sse-event',
        event: {
          id: crypto.randomUUID(),
          at: Date.now(),
          type: ev.type,
          summary: summariseSseEvent(ev),
        },
      });
      const action = agUiToAction(ev);
      if (action) dispatch(action);
    },
  };
}

/** Build a one-line summary for the SSE debug feed. The goal is to
 *  surface the bits most useful for ordering bugs (which message_id,
 *  which JSON-Patch path was touched, what error fired) without
 *  flooding the pane with full payloads. */
function summariseSseEvent(ev: AgUiEvent): string {
  switch (ev.type) {
    case 'RUN_STARTED':
      return 'turn started';
    case 'RUN_FINISHED':
      return 'turn complete';
    case 'RUN_ERROR':
      return `error: ${ev.message}`;
    case 'TEXT_MESSAGE_START':
      return `message=${ev.message_id.slice(0, 8)} entry=${
        ev.entry_id?.slice(0, 8) ?? '∅'
      }`;
    case 'TEXT_MESSAGE_CONTENT':
      return `message=${ev.message_id.slice(0, 8)} +${ev.delta.length} chars`;
    case 'TEXT_MESSAGE_END':
      return `message=${ev.message_id.slice(0, 8)}`;
    case 'USER_MESSAGE_ANCHOR':
      return `entry=${ev.entry_id.slice(0, 8)}`;
    case 'STATE_SNAPSHOT':
      return 'full state replace';
    case 'STATE_DELTA': {
      const paths = ev.patches
        .slice(0, 4)
        .map((p) => {
          const patch = p as { op?: string; path?: string };
          return `${patch.op ?? '?'} ${patch.path ?? '?'}`;
        })
        .join(', ');
      const more =
        ev.patches.length > 4 ? ` (+${ev.patches.length - 4} more)` : '';
      return `${ev.patches.length} ops: ${paths}${more}`;
    }
    case 'TOOL_CALL_START':
      return `${ev.tool_name} (${ev.tool_call_id.slice(0, 8)})`;
    case 'TOOL_CALL_ARGS':
      return `args ${ev.tool_call_id.slice(0, 8)}`;
    case 'TOOL_CALL_END':
      return `end ${ev.tool_call_id.slice(0, 8)}${ev.error ? ` (error)` : ''}`;
    case 'INPUT_REQUEST':
      return ev.prompt.slice(0, 80);
    case 'DEBUG_CODE_EXEC':
      return `${ev.is_final ? 'FINAL ' : ''}${ev.code.length} chars${
        ev.error ? ' (error)' : ''
      }`;
    case 'DEBUG_SPINE_ENTRY_EXTRACTED':
      return `entry=${ev.entry_id.slice(0, 8)} ${ev.entity_count}e ${ev.relation_count}r`;
    case 'DEBUG_SPINE_SEGMENTS_UPDATED':
      return `${ev.segments.length} segment(s)`;
  }
}

function agUiToAction(ev: AgUiEvent): Action | null {
  switch (ev.type) {
    case 'TEXT_MESSAGE_START':
      // Carries entry_id so the spine rail can anchor segments to
      // chat rows. The reducer buffers it keyed by message_id; the
      // matching CONTENT event drains it onto Message.entryId.
      return {
        type: 'text-message-start',
        message_id: ev.message_id,
        entry_id: ev.entry_id,
      };
    case 'TEXT_MESSAGE_END':
      // Bracket marker, no state change needed today.
      return null;
    case 'USER_MESSAGE_ANCHOR':
      return { type: 'user-message-anchor', entry_id: ev.entry_id };
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
      return {
        type: 'tool-call-start',
        id: ev.tool_call_id,
        name: ev.tool_name,
        at: Date.now(),
      };
    case 'TOOL_CALL_ARGS':
      return { type: 'tool-call-args', id: ev.tool_call_id, args: ev.delta };
    case 'TOOL_CALL_END':
      return {
        type: 'tool-call-end',
        id: ev.tool_call_id,
        result: ev.result,
        error: ev.error,
        at: Date.now(),
      };
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
    case 'DEBUG_SPINE_ENTRY_EXTRACTED':
      return {
        type: 'spine-event',
        event: {
          kind: 'entry-extracted',
          id: crypto.randomUUID(),
          at: Date.now(),
          entry_id: ev.entry_id,
          entity_count: ev.entity_count,
          relation_count: ev.relation_count,
        },
      };
    case 'DEBUG_SPINE_SEGMENTS_UPDATED':
      return {
        type: 'spine-event',
        event: {
          kind: 'segments-updated',
          id: crypto.randomUUID(),
          at: Date.now(),
          session_id: ev.session_id,
          segments: ev.segments,
        },
      };
  }
}
