import type { AgUiEvent, Widget } from '../types';

/**
 * State for the observer UI — reconstructed entirely from the AG-UI
 * event stream of the target session. See
 * `docs/design-demo-observer.md`.
 */
export interface ObserverState {
  turns: TurnRecord[];
  /** tool_call_id → record */
  toolCalls: Record<string, ToolCallRecord>;
  /** message_id → record */
  messages: Record<string, MessageRecord>;
  /** Mirror of /widgets from the target session's STATE_DELTAs. */
  widgets: Record<string, Widget>;
  widgetOrder: string[];
  canvasSlot: string | null;
  canvasAuxSlot: string | null;
  focusedScope: Record<string, unknown>;
  /** Raw patch log, most-recent-first-truncated. */
  patchLog: PatchEntry[];
  /** Total connected time in ms. */
  connectedAt: number | null;
  lastEventAt: number | null;
  eventCount: number;
}

export interface TurnRecord {
  /** Synthesised id — we don't have `run_id` on the wire yet. */
  id: string;
  status: 'running' | 'done' | 'error';
  startedAt: number;
  completedAt?: number;
  error?: string;
  toolCallIds: string[];
  messageIds: string[];
}

export interface ToolCallRecord {
  id: string;
  name: string;
  args?: unknown;
  result?: unknown;
  error?: string;
  status: 'running' | 'done' | 'error';
  startedAt: number;
  completedAt?: number;
  turnId: string | null;
}

export interface MessageRecord {
  id: string;
  text: string;
  startedAt: number;
  endedAt?: number;
  turnId: string | null;
}

export interface PatchEntry {
  at: number;
  op: string;
  path: string;
  value?: unknown;
}

export const MAX_PATCH_LOG = 500;

export const initialObserverState: ObserverState = {
  turns: [],
  toolCalls: {},
  messages: {},
  widgets: {},
  widgetOrder: [],
  canvasSlot: null,
  canvasAuxSlot: null,
  focusedScope: {},
  patchLog: [],
  connectedAt: null,
  lastEventAt: null,
  eventCount: 0,
};

/**
 * Reduce a single AG-UI event into the observer state. Pure; the
 * observer component wraps this in `useReducer`.
 */
export function observerReducer(
  state: ObserverState,
  ev: AgUiEvent,
): ObserverState {
  const now = Date.now();
  const base: ObserverState = {
    ...state,
    lastEventAt: now,
    eventCount: state.eventCount + 1,
    connectedAt: state.connectedAt ?? now,
  };

  switch (ev.type) {
    case 'RUN_STARTED': {
      // We don't get a run_id on the wire today (it's always None);
      // synthesise one so turns stay keyed. See design-json-render-
      // migration.md — if run_ids land later, we just use them.
      const id = ev.run_id ?? synthTurnId(base);
      const turn: TurnRecord = {
        id,
        status: 'running',
        startedAt: now,
        toolCallIds: [],
        messageIds: [],
      };
      return { ...base, turns: [...base.turns, turn] };
    }
    case 'RUN_FINISHED':
      return updateLatestTurn(base, (t) => ({
        ...t,
        status: 'done',
        completedAt: now,
      }));
    case 'RUN_ERROR':
      return updateLatestTurn(base, (t) => ({
        ...t,
        status: 'error',
        completedAt: now,
        error: ev.message,
      }));
    case 'TOOL_CALL_START': {
      const turnId = latestRunningTurnId(base);
      const rec: ToolCallRecord = {
        id: ev.tool_call_id,
        name: ev.tool_name,
        status: 'running',
        startedAt: now,
        turnId,
      };
      return {
        ...base,
        toolCalls: { ...base.toolCalls, [rec.id]: rec },
        turns: turnId
          ? base.turns.map((t) =>
              t.id === turnId
                ? { ...t, toolCallIds: [...t.toolCallIds, rec.id] }
                : t,
            )
          : base.turns,
      };
    }
    case 'TOOL_CALL_ARGS': {
      const existing = base.toolCalls[ev.tool_call_id];
      if (!existing) return base;
      return {
        ...base,
        toolCalls: {
          ...base.toolCalls,
          [ev.tool_call_id]: { ...existing, args: ev.delta },
        },
      };
    }
    case 'TOOL_CALL_END': {
      const existing = base.toolCalls[ev.tool_call_id];
      if (!existing) return base;
      return {
        ...base,
        toolCalls: {
          ...base.toolCalls,
          [ev.tool_call_id]: {
            ...existing,
            result: ev.result,
            error: ev.error,
            status: ev.error ? 'error' : 'done',
            completedAt: now,
          },
        },
      };
    }
    case 'TEXT_MESSAGE_START': {
      const turnId = latestRunningTurnId(base);
      const rec: MessageRecord = {
        id: ev.message_id,
        text: '',
        startedAt: now,
        turnId,
      };
      return {
        ...base,
        messages: { ...base.messages, [rec.id]: rec },
        turns: turnId
          ? base.turns.map((t) =>
              t.id === turnId ? { ...t, messageIds: [...t.messageIds, rec.id] } : t,
            )
          : base.turns,
      };
    }
    case 'TEXT_MESSAGE_CONTENT': {
      const existing = base.messages[ev.message_id];
      if (existing) {
        return {
          ...base,
          messages: {
            ...base.messages,
            [ev.message_id]: { ...existing, text: existing.text + ev.delta },
          },
        };
      }
      // No START seen — start implicit record.
      return {
        ...base,
        messages: {
          ...base.messages,
          [ev.message_id]: {
            id: ev.message_id,
            text: ev.delta,
            startedAt: now,
            turnId: latestRunningTurnId(base),
          },
        },
      };
    }
    case 'TEXT_MESSAGE_END': {
      const existing = base.messages[ev.message_id];
      if (!existing) return base;
      return {
        ...base,
        messages: {
          ...base.messages,
          [ev.message_id]: { ...existing, endedAt: now },
        },
      };
    }
    case 'STATE_SNAPSHOT':
      return applySnapshot(base, ev.state as Record<string, unknown>);
    case 'STATE_DELTA':
      return applyPatches(base, ev.patches as JsonPatchOp[], now);
    case 'INPUT_REQUEST':
    case 'DEBUG_CODE_EXEC':
      // Not rendered directly by the observer in v1; could add a
      // dedicated panel later.
      return base;
  }
}

interface JsonPatchOp {
  op: string;
  path: string;
  value?: unknown;
}

function applySnapshot(
  state: ObserverState,
  snap: Record<string, unknown>,
): ObserverState {
  const widgets = (snap.widgets ?? {}) as Record<string, Widget>;
  const widgetOrder = (snap.widgetOrder ?? []) as string[];
  return {
    ...state,
    widgets,
    widgetOrder,
    canvasSlot: (snap.canvasSlot as string | null | undefined) ?? null,
    canvasAuxSlot: (snap.canvasAuxSlot as string | null | undefined) ?? null,
    focusedScope: (snap.focusedScope as Record<string, unknown>) ?? {},
  };
}

function applyPatches(
  state: ObserverState,
  patches: JsonPatchOp[],
  at: number,
): ObserverState {
  let widgets = state.widgets;
  let widgetOrder = state.widgetOrder;
  let canvasSlot = state.canvasSlot;
  let canvasAuxSlot = state.canvasAuxSlot;
  let focusedScope = state.focusedScope;
  const logEntries: PatchEntry[] = [];

  for (const op of patches) {
    logEntries.push({ at, op: op.op, path: op.path, value: op.value });
    // Widget record add/replace: /widgets/<id>
    const widgetMatch = op.path.match(/^\/widgets\/([^/]+)$/);
    if (widgetMatch && (op.op === 'add' || op.op === 'replace')) {
      widgets = { ...widgets, [widgetMatch[1]!]: op.value as Widget };
      continue;
    }
    // Widget field update: /widgets/<id>/<field>
    const widgetFieldMatch = op.path.match(/^\/widgets\/([^/]+)\/(.+)$/);
    if (widgetFieldMatch && (op.op === 'add' || op.op === 'replace')) {
      const [, wid, field] = widgetFieldMatch;
      const prev = widgets[wid!];
      if (prev) {
        widgets = {
          ...widgets,
          [wid!]: { ...prev, [field!]: op.value } as Widget,
        };
      }
      continue;
    }
    if (op.path === '/widgetOrder/-' && op.op === 'add') {
      widgetOrder = [...widgetOrder, op.value as string];
      continue;
    }
    if (op.path === '/widgetOrder' && (op.op === 'add' || op.op === 'replace')) {
      widgetOrder = op.value as string[];
      continue;
    }
    if (op.path === '/canvasSlot' && (op.op === 'add' || op.op === 'replace')) {
      canvasSlot = op.value as string | null;
      continue;
    }
    if (op.path === '/canvasAuxSlot' && (op.op === 'add' || op.op === 'replace')) {
      canvasAuxSlot = op.value as string | null;
      continue;
    }
    const scopeMatch = op.path.match(/^\/focusedScope\/(.+)$/);
    if (scopeMatch && (op.op === 'add' || op.op === 'replace')) {
      focusedScope = { ...focusedScope, [scopeMatch[1]!]: op.value };
      continue;
    }
  }

  const nextLog = [...state.patchLog, ...logEntries];
  if (nextLog.length > MAX_PATCH_LOG) {
    nextLog.splice(0, nextLog.length - MAX_PATCH_LOG);
  }

  return {
    ...state,
    widgets,
    widgetOrder,
    canvasSlot,
    canvasAuxSlot,
    focusedScope,
    patchLog: nextLog,
  };
}

function updateLatestTurn(
  state: ObserverState,
  fn: (t: TurnRecord) => TurnRecord,
): ObserverState {
  if (state.turns.length === 0) return state;
  const idx = state.turns.length - 1;
  const updated = fn(state.turns[idx]!);
  const turns = state.turns.slice();
  turns[idx] = updated;
  return { ...state, turns };
}

function latestRunningTurnId(state: ObserverState): string | null {
  for (let i = state.turns.length - 1; i >= 0; i--) {
    if (state.turns[i]!.status === 'running') return state.turns[i]!.id;
  }
  return null;
}

let turnCounter = 0;
function synthTurnId(_state: ObserverState): string {
  turnCounter += 1;
  return `turn-${turnCounter}`;
}
