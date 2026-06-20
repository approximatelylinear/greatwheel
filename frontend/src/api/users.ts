import { useCallback, useEffect, useState } from 'react';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8787';

const STORAGE_KEY = 'gw_user_id';

// ─── shapes (mirror gw_ui::sessions_api) ─────────────────────────

export interface UserRow {
  user_id: string;
  name: string;
  email: string | null;
  created_at: string;
}

export interface SessionListItem {
  session_id: string;
  title: string | null;
  summary: string | null;
  created_at: string;
  last_active_at: string;
  archived_at: string | null;
  message_count: number;
}

export interface UpsertUserBody {
  user_id?: string;
  name?: string;
  email?: string;
}

// ─── api wrappers ────────────────────────────────────────────────

export async function upsertUser(
  body: UpsertUserBody,
  signal?: AbortSignal,
): Promise<UserRow> {
  const r = await fetch(`${BASE}/users`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
    signal,
  });
  if (!r.ok) throw new Error(`upsertUser ${r.status}: ${await r.text()}`);
  return (await r.json()) as UserRow;
}

export async function fetchSessionsForUser(
  userId: string,
  includeArchived = false,
  signal?: AbortSignal,
): Promise<SessionListItem[]> {
  const qs = includeArchived ? '?include_archived=1' : '';
  const r = await fetch(`${BASE}/users/${userId}/sessions${qs}`, { signal });
  if (!r.ok) {
    throw new Error(`fetchSessionsForUser ${r.status}: ${await r.text()}`);
  }
  return (await r.json()) as SessionListItem[];
}

export async function createSession(
  userId: string,
  title?: string,
): Promise<{ session_id: string }> {
  const r = await fetch(`${BASE}/users/${userId}/sessions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ title: title ?? null }),
  });
  if (!r.ok) throw new Error(`createSession ${r.status}: ${await r.text()}`);
  return (await r.json()) as { session_id: string };
}

export async function patchSession(
  sessionId: string,
  body: { title?: string; archived?: boolean },
): Promise<SessionListItem> {
  const r = await fetch(`${BASE}/sessions/${sessionId}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`patchSession ${r.status}: ${await r.text()}`);
  return (await r.json()) as SessionListItem;
}

// ─── identity hook ───────────────────────────────────────────────

export interface UseUserIdResult {
  userId: string;
  displayName: string | null;
  /** True once `POST /users` has completed and the user row is known
   *  to exist in PG. Gates downstream calls (e.g. `createSession`)
   *  that 404 if the user FK isn't yet present. */
  ready: boolean;
  setDisplayName(name: string): Promise<void>;
}

/**
 * Resolves the current user id in this order:
 *   1. `?user=<uuid>` URL param (so a profile URL can be pasted).
 *   2. `localStorage.gw_user_id`.
 *   3. Generate fresh UUID, persist to localStorage.
 *
 * Then POSTs `/users` once on mount — the endpoint is an idempotent
 * upsert, so this both creates the row on first load and fetches the
 * stored display name (server default: `user-<8hex>`) on returning
 * loads.
 */
export function useUserId(): UseUserIdResult {
  const [userId] = useState(resolveUserId);
  const [displayName, setDisplayNameState] = useState<string | null>(null);
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    upsertUser({ user_id: userId }, ac.signal)
      .then((row) => {
        if (ac.signal.aborted) return;
        setDisplayNameState(row.name);
        setReady(true);
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === 'AbortError') return;
        console.error('upsertUser failed', e);
      });
    return () => ac.abort();
  }, [userId]);

  const setDisplayName = useCallback(
    async (name: string) => {
      const row = await upsertUser({ user_id: userId, name });
      setDisplayNameState(row.name);
    },
    [userId],
  );

  return { userId, displayName, ready, setDisplayName };
}

function resolveUserId(): string {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get('user');
  if (fromUrl) {
    window.localStorage.setItem(STORAGE_KEY, fromUrl);
    return fromUrl;
  }
  const fromStorage = window.localStorage.getItem(STORAGE_KEY);
  if (fromStorage) return fromStorage;
  const fresh = crypto.randomUUID();
  window.localStorage.setItem(STORAGE_KEY, fresh);
  return fresh;
}

// ─── session routing hook ────────────────────────────────────────

export interface UseSessionRoutingResult {
  userId: string;
  /** Null during bootstrap — no `?session=` in URL and the
   *  most-recent / auto-create lookup is in flight. */
  sessionId: string | null;
  /** Server-stored display name (default `user-<8hex>`); null until
   *  `POST /users` returns. */
  displayName: string | null;
  /** True once the user row is confirmed in PG (gates session
   *  creation). Re-exposed from `useUserId` so callers can render
   *  a loading state without composing the two hooks themselves. */
  ready: boolean;
  /** Re-POST `/users` with a new name (idempotent upsert). */
  setDisplayName(name: string): Promise<void>;
  /** Navigate to a different session. Updates `?user=&session=` via
   *  pushState; the SSE / transcript effects in `App` re-fire on the
   *  new `sessionId`. */
  switchSession(id: string): void;
}

/**
 * Resolves `(userId, sessionId)` for the running app and exposes a
 * `switchSession` to navigate between sessions without a full page
 * reload.
 *
 * Bootstrap order:
 *   1. `useUserId()` resolves `userId` (URL → localStorage → mint)
 *      and POSTs `/users` to ensure the FK row exists.
 *   2. Initial `sessionId` is read sync from `?session=<uuid>`.
 *   3. If still null after the user row is ready, `GET /users/{uid}/
 *      sessions` is fetched. The most recently active session wins;
 *      otherwise a fresh one is `POST`-ed and we land on it.
 *   4. The URL is `replaceState`-stamped with the resolved
 *      `?user=&session=` so a refresh comes back to the same place.
 *
 * `switchSession` uses `pushState` so back/forward navigates between
 * sessions; a `popstate` listener syncs `sessionId` when the user
 * does so.
 */
export function useSessionRouting(): UseSessionRoutingResult {
  const { userId, displayName, ready, setDisplayName } = useUserId();
  const [sessionId, setSessionId] = useState<string | null>(
    resolveInitialSessionId,
  );

  useEffect(() => {
    if (sessionId) return;
    if (!ready) return;
    const ac = new AbortController();
    (async () => {
      try {
        const sessions = await fetchSessionsForUser(userId, false, ac.signal);
        if (ac.signal.aborted) return;
        if (sessions.length > 0) {
          const target = sessions[0]!.session_id;
          setSessionId(target);
          writeUrl(userId, target, 'replace');
          return;
        }
        const { session_id } = await createSession(userId);
        if (ac.signal.aborted) return;
        setSessionId(session_id);
        writeUrl(userId, session_id, 'replace');
      } catch (e: unknown) {
        if (ac.signal.aborted) return;
        if (e instanceof DOMException && e.name === 'AbortError') return;
        console.error('session bootstrap failed', e);
      }
    })();
    return () => ac.abort();
  }, [userId, sessionId, ready]);

  // Stamp `?user=` into the URL once we know it, even when the
  // initial URL only carried `?session=`. Idempotent; skips when the
  // params already match.
  useEffect(() => {
    if (!sessionId) return;
    const cur = new URL(window.location.href);
    if (
      cur.searchParams.get('user') === userId &&
      cur.searchParams.get('session') === sessionId
    ) {
      return;
    }
    writeUrl(userId, sessionId, 'replace');
  }, [userId, sessionId]);

  const switchSession = useCallback(
    (id: string) => {
      if (id === sessionId) return;
      setSessionId(id);
      writeUrl(userId, id, 'push');
    },
    [userId, sessionId],
  );

  useEffect(() => {
    const onPop = () => {
      const url = new URL(window.location.href);
      const next = url.searchParams.get('session');
      if (next && next !== sessionId) setSessionId(next);
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [sessionId]);

  return {
    userId,
    sessionId,
    displayName,
    ready,
    setDisplayName,
    switchSession,
  };
}

function resolveInitialSessionId(): string | null {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get('session');
  if (fromUrl) return fromUrl;
  const fromEnv = (import.meta.env.VITE_SESSION_ID as string | undefined) ?? '';
  return fromEnv || null;
}

function writeUrl(
  userId: string,
  sessionId: string,
  mode: 'push' | 'replace',
): void {
  const url = new URL(window.location.href);
  url.searchParams.set('user', userId);
  url.searchParams.set('session', sessionId);
  if (mode === 'push') {
    window.history.pushState({}, '', url);
  } else {
    window.history.replaceState({}, '', url);
  }
}
