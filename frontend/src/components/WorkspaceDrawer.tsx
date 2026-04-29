import { useEffect, useState } from 'react';
import { fetchWorkspace, type WorkspaceItem } from '../api/client';

/**
 * Right-side drawer that lists every segment the user has committed
 * to the workspace (Issue #5). Opens from the app-header "Workspace"
 * button; refetches on each open, and again whenever the SpineSidebar
 * fires its `onWorkspaceInvalidate` callback (a commit toggled).
 *
 * Each item is a card with the segment's label, kind, top entities,
 * and two affordances:
 *   - "Open" — focuses the segment so the SpineSidebar takes over
 *     (restores the same context the user committed from).
 *   - "Jump" — scrolls the chat to the segment's entry_first row
 *     without changing focus.
 *
 * Invalidated-but-committed rows render with a "(superseded)" tag —
 * resegmentation may have replaced them but the workspace keeps
 * them visible so the user's curated reading list survives churn.
 */
interface Props {
  sessionId: string;
  open: boolean;
  /** Bumped by the parent every time a commit toggle happens; the
   *  drawer refetches when this changes (including on open). */
  reloadKey: number;
  onClose: () => void;
  /** Click "Open" on a card → focus this segment id. The parent
   *  fires the same focus widget event as a rail-marker click. */
  onOpen: (segmentId: string) => void;
  /** Click "Jump" on a card → scroll the chat to entry_first. */
  onJump: (entryFirst: string, entryLast: string) => void;
}

const KIND_BADGE: Record<string, string> = {
  comparison: '⇄',
  decision: '✓',
  deep_dive: '↳',
  construction: '∎',
  other: '·',
};

export function WorkspaceDrawer({
  sessionId,
  open,
  reloadKey,
  onClose,
  onOpen,
  onJump,
}: Props) {
  const [items, setItems] = useState<WorkspaceItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchWorkspace(sessionId, ac.signal)
      .then((rows) => {
        setItems(rows);
        setLoading(false);
      })
      .catch((e: unknown) => {
        if (e instanceof DOMException && e.name === 'AbortError') return;
        setError(String(e));
        setLoading(false);
      });
    return () => ac.abort();
  }, [sessionId, open, reloadKey]);

  if (!open) return null;
  return (
    <div className="workspace-drawer-backdrop" onClick={onClose}>
      <aside
        className="workspace-drawer"
        onClick={(e) => e.stopPropagation()}
        aria-label="Workspace"
      >
        <header className="workspace-drawer-header">
          <span className="workspace-drawer-title">Workspace</span>
          <span className="workspace-drawer-count">
            {items ? `${items.length} saved` : ''}
          </span>
          <button
            type="button"
            className="workspace-drawer-close"
            onClick={onClose}
            title="Close workspace"
            aria-label="Close workspace"
          >
            ×
          </button>
        </header>
        <div className="workspace-drawer-body">
          {loading && (
            <div className="workspace-empty">Loading workspace…</div>
          )}
          {error && (
            <div className="workspace-empty workspace-error">
              Couldn't load workspace: {error}
            </div>
          )}
          {!loading && !error && items && items.length === 0 && (
            <div className="workspace-empty">
              Nothing saved yet. Open a segment in the spine and click{' '}
              <span className="workspace-empty-glyph">★ Save</span> to keep
              it here for later.
            </div>
          )}
          {!loading && !error && items && items.length > 0 && (
            <ul className="workspace-list">
              {items.map((it) => (
                <li
                  key={it.segment_id}
                  className={`workspace-card${
                    it.invalidated ? ' superseded' : ''
                  }`}
                >
                  <div className="workspace-card-header">
                    <span
                      className={`workspace-card-kind workspace-card-kind-${it.kind}`}
                      title={it.kind.replace('_', ' ')}
                    >
                      {KIND_BADGE[it.kind] ?? '·'}
                    </span>
                    <span className="workspace-card-label">{it.label}</span>
                    {it.invalidated && (
                      <span
                        className="workspace-card-stale"
                        title="The active spine has moved past this segment"
                      >
                        superseded
                      </span>
                    )}
                  </div>
                  {it.summary && (
                    <p className="workspace-card-summary">{it.summary}</p>
                  )}
                  {it.top_entities.length > 0 && (
                    <div className="workspace-card-entities">
                      {it.top_entities.map((label) => (
                        <span key={label} className="workspace-card-entity">
                          {label}
                        </span>
                      ))}
                      {it.entity_count > it.top_entities.length && (
                        <span className="workspace-card-entity-more">
                          +{it.entity_count - it.top_entities.length}
                        </span>
                      )}
                    </div>
                  )}
                  <div className="workspace-card-actions">
                    <button
                      type="button"
                      className="workspace-card-btn"
                      onClick={() => onOpen(it.segment_id)}
                    >
                      Open
                    </button>
                    <button
                      type="button"
                      className="workspace-card-btn"
                      onClick={() => onJump(it.entry_first, it.entry_last)}
                    >
                      Jump
                    </button>
                  </div>
                </li>
              ))}
            </ul>
          )}
        </div>
      </aside>
    </div>
  );
}
