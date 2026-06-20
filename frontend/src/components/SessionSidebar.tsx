import { useCallback, useEffect, useState } from 'react';
import {
  createSession,
  fetchSessionsForUser,
  patchSession,
  type SessionListItem,
} from '../api/users';

/**
 * Left-rail session list for the literature_assistant demo. Header
 * with the user's display name (click to rename), a primary "+ New
 * session" CTA, and a scrollable list of the user's sessions ordered
 * by `last_active_at` desc. Hover-reveal archive icon per row;
 * "Show archived" checkbox toggles whether archived sessions appear.
 *
 * State management is local — the sidebar owns its `items` cache and
 * refetches when:
 *   - the component mounts,
 *   - the active session changes (so a freshly-bumped row re-sorts),
 *   - a new session is created,
 *   - a row is archived / unarchived,
 *   - the "Show archived" toggle flips.
 */
interface Props {
  userId: string;
  currentSessionId: string;
  displayName: string | null;
  onSetDisplayName(name: string): Promise<void>;
  onSwitchSession(id: string): void;
}

export function SessionSidebar({
  userId,
  currentSessionId,
  displayName,
  onSetDisplayName,
  onSwitchSession,
}: Props) {
  const [items, setItems] = useState<SessionListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [showArchived, setShowArchived] = useState(false);
  const [reloadKey, setReloadKey] = useState(0);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchSessionsForUser(userId, showArchived, ac.signal)
      .then((rows) => {
        if (ac.signal.aborted) return;
        setItems(rows);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === 'AbortError') return;
        setError(String(e));
        setLoading(false);
      });
    return () => ac.abort();
  }, [userId, showArchived, reloadKey, currentSessionId]);

  const bumpReload = useCallback(() => setReloadKey((k) => k + 1), []);

  const handleNewSession = useCallback(async () => {
    try {
      const { session_id } = await createSession(userId);
      onSwitchSession(session_id);
      bumpReload();
    } catch (e: unknown) {
      setError(String(e));
    }
  }, [userId, onSwitchSession, bumpReload]);

  const handleSelectSession = useCallback(
    (id: string) => {
      if (id === currentSessionId) return;
      onSwitchSession(id);
    },
    [currentSessionId, onSwitchSession],
  );

  const handleArchiveToggle = useCallback(
    async (sid: string, archive: boolean) => {
      try {
        await patchSession(sid, { archived: archive });
        bumpReload();
      } catch (e: unknown) {
        setError(String(e));
      }
    },
    [bumpReload],
  );

  const handleRenameUser = useCallback(async () => {
    const next = window.prompt('Display name', displayName ?? '');
    if (next == null) return;
    const trimmed = next.trim();
    if (!trimmed || trimmed === displayName) return;
    try {
      await onSetDisplayName(trimmed);
    } catch (e: unknown) {
      setError(String(e));
    }
  }, [displayName, onSetDisplayName]);

  return (
    <aside className="session-sidebar" aria-label="Sessions">
      <header className="session-sidebar-header">
        <button
          type="button"
          className="session-sidebar-user"
          onClick={handleRenameUser}
          title={`User ${userId} — click to rename`}
        >
          {displayName ?? '…'}
        </button>
      </header>
      <button
        type="button"
        className="session-sidebar-new"
        onClick={handleNewSession}
      >
        + New session
      </button>
      <div className="session-sidebar-body">
        {loading && !items && (
          <div className="session-sidebar-empty">Loading…</div>
        )}
        {error && (
          <div className="session-sidebar-empty session-sidebar-error">
            {error}
          </div>
        )}
        {!loading && !error && items && items.length === 0 && (
          <div className="session-sidebar-empty">
            No sessions yet — start one.
          </div>
        )}
        {items && items.length > 0 && (
          <ul className="session-list">
            {items.map((s) => {
              const archived = s.archived_at != null;
              const active = s.session_id === currentSessionId;
              const title = s.title?.trim() || 'Untitled session';
              return (
                <li
                  key={s.session_id}
                  className={`session-row${active ? ' active' : ''}${
                    archived ? ' archived' : ''
                  }`}
                >
                  <button
                    type="button"
                    className="session-row-main"
                    onClick={() => handleSelectSession(s.session_id)}
                    title={s.session_id}
                  >
                    <span className="session-row-title">{title}</span>
                    <span className="session-row-meta">
                      <span className="session-row-time">
                        {relativeTime(s.last_active_at)}
                      </span>
                      {archived && (
                        <span className="session-row-tag">archived</span>
                      )}
                    </span>
                  </button>
                  <button
                    type="button"
                    className="session-row-archive"
                    onClick={(e) => {
                      e.stopPropagation();
                      void handleArchiveToggle(s.session_id, !archived);
                    }}
                    title={archived ? 'Unarchive' : 'Archive'}
                    aria-label={archived ? 'Unarchive session' : 'Archive session'}
                  >
                    {archived ? '↺' : '×'}
                  </button>
                </li>
              );
            })}
          </ul>
        )}
      </div>
      <footer className="session-sidebar-footer">
        <label className="session-sidebar-toggle">
          <input
            type="checkbox"
            checked={showArchived}
            onChange={(e) => setShowArchived(e.target.checked)}
          />
          Show archived
        </label>
      </footer>
    </aside>
  );
}

function relativeTime(iso: string): string {
  const ms = Date.now() - new Date(iso).getTime();
  if (Number.isNaN(ms)) return '';
  if (ms < 60_000) return 'just now';
  if (ms < 3_600_000) return `${Math.round(ms / 60_000)}m ago`;
  if (ms < 86_400_000) return `${Math.round(ms / 3_600_000)}h ago`;
  if (ms < 7 * 86_400_000) return `${Math.round(ms / 86_400_000)}d ago`;
  return new Date(iso).toLocaleDateString();
}
