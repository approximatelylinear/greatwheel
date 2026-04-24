import type { StateStore } from '@json-render/core';
import type { AgUiEvent, Widget } from '../types';

/**
 * Canonical state shape the server ships in STATE_SNAPSHOT (see
 * docs/design-json-render-migration.md §3). Kept as the initialiser
 * for a local json-render StateStore so components can bind to
 * `{$state: "/widgets/<id>"}`, `{$state: "/focusedScope/section"}`,
 * etc. before the first snapshot arrives.
 */
export const INITIAL_CANONICAL_STATE: Record<string, unknown> = {
  widgets: {},
  widgetOrder: [],
  canvasSlot: null,
  canvasAuxSlot: null,
  pinnedIds: {},
  pressed: {},
  focusedScope: {},
};

export interface StateBridgeCallbacks {
  /**
   * Fires when a JSON-Patch op adds a new Widget to `/widgets/<id>`.
   * The session reducer uses this to drive follow-up anchoring
   * (attaching `follow_up: true` widgets to the nearest assistant
   * message) — a chat-surface concern the server doesn't model.
   */
  onWidgetAdded?: (widget: Widget) => void;
}

/**
 * Fan out an AG-UI event into a json-render StateStore. Recognises
 * STATE_SNAPSHOT (full replace) and STATE_DELTA (RFC 6902 JSON-Patch
 * array). Silently ignores every other event type — those are
 * consumed by the session reducer.
 */
export function applyAgUiEventToStore(
  store: StateStore,
  ev: AgUiEvent,
  callbacks: StateBridgeCallbacks = {},
): void {
  if (ev.type === 'STATE_SNAPSHOT') {
    const next = (ev.state ?? {}) as Record<string, unknown>;
    for (const [k, v] of Object.entries(INITIAL_CANONICAL_STATE)) {
      store.set(`/${k}`, next[k] ?? v);
    }
    for (const [k, v] of Object.entries(next)) {
      if (!(k in INITIAL_CANONICAL_STATE)) store.set(`/${k}`, v);
    }
    // Snapshot can also carry pre-existing widgets; treat each as an
    // "add" for follow-up anchoring purposes so a client reconnecting
    // mid-session still wires them up.
    if (callbacks.onWidgetAdded) {
      const widgets = (next.widgets ?? {}) as Record<string, Widget>;
      for (const w of Object.values(widgets)) {
        callbacks.onWidgetAdded(w);
      }
    }
    return;
  }
  if (ev.type === 'STATE_DELTA') {
    for (const raw of ev.patches ?? []) {
      const op = raw as JsonPatchOp;
      applyPatchOp(store, op);
      // Detect widget additions so the session reducer can anchor
      // follow-up widgets to chat messages. Only triggers on a
      // top-level `/widgets/<uuid>` add — not on children or
      // replacements of existing widgets' fields.
      if (
        callbacks.onWidgetAdded &&
        op.op === 'add' &&
        /^\/widgets\/[^/]+$/.test(op.path) &&
        op.value &&
        typeof op.value === 'object'
      ) {
        callbacks.onWidgetAdded(op.value as Widget);
      }
    }
  }
}

interface JsonPatchOp {
  op: 'add' | 'replace' | 'remove';
  path: string;
  value?: unknown;
}

function applyPatchOp(store: StateStore, op: JsonPatchOp): void {
  switch (op.op) {
    case 'add':
    case 'replace': {
      if (op.path.endsWith('/-')) {
        const parentPath = op.path.slice(0, -2);
        const parent = store.get(parentPath);
        const arr = Array.isArray(parent) ? parent : [];
        store.set(parentPath, [...arr, op.value]);
      } else {
        store.set(op.path, op.value);
      }
      return;
    }
    case 'remove': {
      store.set(op.path, undefined);
      return;
    }
  }
}
