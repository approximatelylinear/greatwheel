import type { StateStore } from '@json-render/core';
import type { AgUiEvent } from '../types';

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

/**
 * Fan out an AG-UI event into a json-render StateStore. Recognises
 * STATE_SNAPSHOT (full replace) and STATE_DELTA (RFC 6902 JSON-Patch
 * array). Silently ignores every other event type — those are
 * consumed by the session reducer.
 */
export function applyAgUiEventToStore(
  store: StateStore,
  ev: AgUiEvent,
): void {
  if (ev.type === 'STATE_SNAPSHOT') {
    const next = (ev.state ?? {}) as Record<string, unknown>;
    // Seed any canonical keys the server omitted so path binds don't
    // resolve to undefined mid-render.
    for (const [k, v] of Object.entries(INITIAL_CANONICAL_STATE)) {
      store.set(`/${k}`, next[k] ?? v);
    }
    // Also carry through any extras (forward-compat for keys we add
    // server-side without touching this file).
    for (const [k, v] of Object.entries(next)) {
      if (!(k in INITIAL_CANONICAL_STATE)) store.set(`/${k}`, v);
    }
    return;
  }
  if (ev.type === 'STATE_DELTA') {
    for (const raw of ev.patches ?? []) {
      applyPatchOp(store, raw as JsonPatchOp);
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
        // JSON-Patch array-append shorthand. StateStore has no
        // append primitive — resolve parent, push, write back.
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
      // Best-effort: StateStore doesn't have a delete primitive, so
      // we blank out the key. No consumers in phase 3 rely on
      // distinguishing missing vs null, so this is fine.
      store.set(op.path, undefined);
      return;
    }
  }
}
