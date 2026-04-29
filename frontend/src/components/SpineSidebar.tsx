import { useEffect, useState } from 'react';
import {
  fetchSegmentDetail,
  setSegmentCommitted,
  type EntityCard,
  type RelationRow,
  type SegmentDetail,
} from '../api/client';

/**
 * Sidebar that hangs under the spine rail when a marker is focused.
 * Fetches `/sessions/:session_id/segments/:segment_id` from the
 * literature_assistant's segment-detail endpoint and renders three
 * tabs: Entities, Relations, Notes.
 *
 * Refetches on focus change. AbortController cancels stale requests
 * if the user clicks through markers quickly.
 *
 * Phase B scope: read-only. Phase C wires the action menu (revisit /
 * expand / compare). Phase D moves the sidebar from inline-under-rail
 * to canvas-pinned per the design doc.
 */
interface Props {
  sessionId: string;
  segmentId: string;
  /** Fires a spine action-menu click. The adapter forwards it to
   *  ConversationLoop, which translates `revisit`/`expand`/`compare`
   *  into a server-side templated prompt and runs the next turn. */
  onAction: (action: 'revisit' | 'expand' | 'compare') => void;
  /** Phase D footer: scroll the chat to the segment's first chat
   *  row anchor. Receives the segment id so the parent can resolve
   *  entry_first/entry_last from its own segment list (the sidebar
   *  doesn't carry those — they live alongside the rail). */
  onJump?: () => void;
  /** Phase D footer: clear `/focusedScope/segment` so the sidebar
   *  dismisses. Wired by the parent to fire a focus widget event
   *  with `key: null`. */
  onClose?: () => void;
  /** Issue #5: invalidate the workspace listing cache after a
   *  commit toggle so a subsequent workspace-drawer open shows the
   *  current state. Optional — when absent the workspace just
   *  refetches on its own next open. */
  onWorkspaceInvalidate?: () => void;
  /** Issue #6: forwarded to App.tsx so ChatPane can highlight the
   *  entities of the focused segment in their corresponding chat
   *  rows. Fires on every successful detail load (segment change /
   *  refetch) and with `null` when the sidebar unmounts. */
  onDetailLoaded?: (detail: SegmentDetail | null) => void;
  /** Issue #6 follow-up: an entity card was clicked. The parent
   *  switches the chat-pane highlighter to this single entity (label
   *  + aliases) across the whole conversation and scrolls to the
   *  first match. Receives `null` to clear the selection back to
   *  "all segment entities within segment range". */
  onSelectEntity?: (entity: EntityCard | null) => void;
  /** Currently-selected entity id (driven by the parent so it
   *  survives sidebar refetches). */
  selectedEntityId?: string | null;
}

type SpineActionKind = 'revisit' | 'expand' | 'compare';

const ACTIONS: Array<{
  key: SpineActionKind;
  label: string;
  glyph: string;
  hint: string;
}> = [
  {
    key: 'revisit',
    label: 'Revisit',
    glyph: '↻',
    hint: 'Summarise what we concluded and what is still open',
  },
  {
    key: 'expand',
    label: 'Go deeper',
    glyph: '↳',
    hint: "Explore what we haven't covered",
  },
  {
    key: 'compare',
    label: 'Compare',
    glyph: '⇄',
    hint: 'Compare with current direction',
  },
];

type Tab = 'entities' | 'relations' | 'notes';

const KIND_COLORS: Record<string, string> = {
  // entity-kind palette — matches the EntityCloud's kind colours so
  // an author dot in the cloud reads the same in the sidebar.
  author: '#b48cde',
  concept: '#7bd38f',
  method: '#7ba5ff',
  dataset: '#6ec3d2',
  venue: '#d6a56e',
};

function kindColor(kind: string): string {
  return KIND_COLORS[kind.toLowerCase()] ?? '#a8b1bf';
}

export function SpineSidebar({
  sessionId,
  segmentId,
  onAction,
  onJump,
  onClose,
  onWorkspaceInvalidate,
  onDetailLoaded,
  onSelectEntity,
  selectedEntityId,
}: Props) {
  const [tab, setTab] = useState<Tab>('entities');
  const [detail, setDetail] = useState<SegmentDetail | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  // In-flight commit toggle so the user can't double-click the
  // button and fire two requests for the same row.
  const [commitBusy, setCommitBusy] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    fetchSegmentDetail(sessionId, segmentId, ac.signal)
      .then((d) => {
        setDetail(d);
        setLoading(false);
        onDetailLoaded?.(d);
      })
      .catch((e: unknown) => {
        // Aborted requests show up as DOMException AbortError; ignore.
        if (e instanceof DOMException && e.name === 'AbortError') return;
        setError(String(e));
        setLoading(false);
      });
    return () => {
      ac.abort();
      // Sidebar unmounting (segment change, focus dismiss) — clear
      // the upstream cache so the chat-pane highlighter doesn't
      // keep marking stale entities.
      onDetailLoaded?.(null);
    };
    // onDetailLoaded purposely excluded from deps — its identity may
    // change every render in the parent (useCallback deps), and we
    // don't want to refetch just because the callback was rebuilt.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, segmentId]);

  const onToggleCommit = async () => {
    if (!detail || commitBusy) return;
    const isCommitted = detail.segment.committed_at != null;
    setCommitBusy(true);
    try {
      const r = await setSegmentCommitted(
        sessionId,
        segmentId,
        !isCommitted,
      );
      // Splice the new committed_at onto local state without refetch
      // so the toggle flips immediately.
      setDetail({
        ...detail,
        segment: { ...detail.segment, committed_at: r.committed_at },
      });
      onWorkspaceInvalidate?.();
    } catch (e) {
      setError(String(e));
    } finally {
      setCommitBusy(false);
    }
  };

  if (error) {
    return (
      <div className="spine-sidebar spine-sidebar-error">
        <header className="spine-sidebar-header">
          <span className="spine-sidebar-title">Couldn't load segment</span>
          {onClose && (
            <button
              type="button"
              className="spine-sidebar-close"
              onClick={onClose}
              title="Dismiss"
              aria-label="Dismiss segment focus"
            >
              ×
            </button>
          )}
        </header>
        <div>{error}</div>
      </div>
    );
  }
  if (loading || !detail) {
    return (
      <div className="spine-sidebar spine-sidebar-loading">
        <div className="spine-sidebar-header">
          <span className="spine-sidebar-title">Loading…</span>
          {onClose && (
            <button
              type="button"
              className="spine-sidebar-close"
              onClick={onClose}
              title="Dismiss"
              aria-label="Dismiss segment focus"
            >
              ×
            </button>
          )}
        </div>
      </div>
    );
  }

  const { segment, entities, relations } = detail;
  return (
    <div className="spine-sidebar" aria-live="polite">
      <header className="spine-sidebar-header">
        <span className="spine-sidebar-title">{segment.label}</span>
        <span className={`spine-kind spine-kind-${segment.kind}`}>
          {segment.kind.replace('_', ' ')}
        </span>
        {segment.invalidated && (
          <span
            className="spine-sidebar-stale"
            title="A later resegment pass superseded this segment, but it stays here because you committed it"
          >
            superseded
          </span>
        )}
        <button
          type="button"
          className={`spine-sidebar-save${
            segment.committed_at ? ' active' : ''
          }`}
          onClick={onToggleCommit}
          disabled={commitBusy}
          title={
            segment.committed_at
              ? 'Remove from workspace'
              : 'Save to workspace'
          }
          aria-pressed={segment.committed_at != null}
        >
          <span aria-hidden>
            {segment.committed_at ? '★' : '☆'}
          </span>
          <span>{segment.committed_at ? 'Saved' : 'Save'}</span>
        </button>
        {onClose && (
          <button
            type="button"
            className="spine-sidebar-close"
            onClick={onClose}
            title="Dismiss"
            aria-label="Dismiss segment focus"
          >
            ×
          </button>
        )}
      </header>
      {segment.summary && (
        <p className="spine-sidebar-summary">{segment.summary}</p>
      )}
      <div className="spine-action-row">
        {ACTIONS.map((a) => (
          <button
            key={a.key}
            type="button"
            className="spine-action"
            title={a.hint}
            onClick={() => onAction(a.key)}
          >
            <span className="spine-action-glyph" aria-hidden>
              {a.glyph}
            </span>
            <span className="spine-action-label">{a.label}</span>
          </button>
        ))}
      </div>
      <nav className="spine-sidebar-tabs" role="tablist">
        <TabButton
          tab="entities"
          active={tab}
          onSelect={setTab}
          label={`Entities · ${entities.length}`}
        />
        <TabButton
          tab="relations"
          active={tab}
          onSelect={setTab}
          label={`Relations · ${relations.length}`}
        />
        <TabButton tab="notes" active={tab} onSelect={setTab} label="Notes" />
      </nav>
      <div className="spine-sidebar-body">
        {tab === 'entities' && (
          <EntitiesTab
            entities={entities}
            selectedEntityId={selectedEntityId ?? null}
            onSelectEntity={(e) => onSelectEntity?.(e)}
          />
        )}
        {tab === 'relations' && <RelationsTab relations={relations} />}
        {tab === 'notes' && <NotesTab />}
      </div>
      {onJump && (
        <footer className="spine-sidebar-footer">
          <button
            type="button"
            className="spine-sidebar-footer-btn"
            onClick={onJump}
            title="Scroll the chat to where this segment starts"
          >
            <span aria-hidden>↑</span>
            <span>Jump to message</span>
          </button>
        </footer>
      )}
    </div>
  );
}

function TabButton({
  tab,
  active,
  onSelect,
  label,
}: {
  tab: Tab;
  active: Tab;
  onSelect: (t: Tab) => void;
  label: string;
}) {
  return (
    <button
      type="button"
      role="tab"
      aria-selected={active === tab}
      className={`spine-tab${active === tab ? ' active' : ''}`}
      onClick={() => onSelect(tab)}
    >
      {label}
    </button>
  );
}

function EntitiesTab({
  entities,
  selectedEntityId,
  onSelectEntity,
}: {
  entities: EntityCard[];
  selectedEntityId: string | null;
  onSelectEntity: (e: EntityCard | null) => void;
}) {
  if (entities.length === 0) {
    return (
      <div className="spine-tab-empty">
        No named entities extracted from this segment yet.
      </div>
    );
  }
  return (
    <ul className="spine-entity-list">
      {entities.map((e) => {
        const isSelected = e.entity_id === selectedEntityId;
        return (
          <li key={e.entity_id} className="spine-entity-card">
            <button
              type="button"
              className={`spine-entity-card-btn${
                isSelected ? ' selected' : ''
              }`}
              onClick={() =>
                onSelectEntity(isSelected ? null : e)
              }
              aria-pressed={isSelected}
              title={
                isSelected
                  ? 'Click to clear highlight'
                  : 'Click to highlight every mention in the chat and jump to the first one'
              }
            >
              <div className="spine-entity-row">
                <span
                  className="spine-entity-dot"
                  style={{ backgroundColor: kindColor(e.kind) }}
                  aria-hidden
                />
                <span className="spine-entity-label" title={e.slug}>
                  {e.label}
                </span>
                <span className="spine-entity-kind">{e.kind}</span>
              </div>
              <div className="spine-entity-meta">
                <span title="mentions inside this segment">
                  {e.mentions_in_segment}× here
                </span>
                <span className="spine-entity-meta-sep">·</span>
                <span title="mentions across the whole KB corpus">
                  {e.global_mentions}× total
                </span>
                {e.aliases.length > 0 && (
                  <>
                    <span className="spine-entity-meta-sep">·</span>
                    <span
                      className="spine-entity-aliases"
                      title={e.aliases.join(', ')}
                    >
                      {e.aliases.length} alias
                      {e.aliases.length === 1 ? '' : 'es'}
                    </span>
                  </>
                )}
              </div>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

function RelationsTab({ relations }: { relations: RelationRow[] }) {
  if (relations.length === 0) {
    return (
      <div className="spine-tab-empty">
        No typed relations asserted in this segment.
      </div>
    );
  }
  return (
    <ul className="spine-relation-list">
      {relations.map((r) => (
        <li key={r.id} className="spine-relation-row" title={r.surface}>
          <span className="spine-relation-subject">{r.subject_label}</span>
          <span
            className={`spine-relation-arrow${r.directed ? ' directed' : ''}`}
            aria-label={r.predicate}
          >
            {r.directed ? '→' : '↔'}
          </span>
          <span className="spine-relation-predicate">
            {r.predicate.replace('_', ' ')}
          </span>
          <span
            className={`spine-relation-arrow${r.directed ? ' directed' : ''}`}
            aria-hidden
          >
            {r.directed ? '→' : '↔'}
          </span>
          <span className="spine-relation-object">{r.object_label}</span>
        </li>
      ))}
    </ul>
  );
}

function NotesTab() {
  return (
    <div className="spine-tab-empty">
      Notes are coming next — pin observations to a segment for later
      reference.
    </div>
  );
}
