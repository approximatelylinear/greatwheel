import { useMemo } from 'react';
import { useStateValue } from '@json-render/react';
import { postWidgetEvent } from '../api/client';
import type { Widget } from '../types';

/**
 * Right-side drawer that lists every widget the current session has
 * emitted (Issue: widget history). Lets the user re-pin a previously
 * superseded widget into its appropriate slot without involving the
 * agent — pure UI op, see `docs/design-widget-history.md`.
 *
 * Reads `/widgets`, `/canvasSlot`, `/canvasAuxSlot`, `/wikiSlot` from
 * the canonical state store; no fetch. Restoring fires a
 * `restore_widget` widget-event that the ag-ui adapter short-circuits
 * to a `pin_*` call on the surface store, which mirrors back as a
 * STATE_DELTA replacing the slot pointer.
 */
interface Props {
  sessionId: string;
  open: boolean;
  onClose: () => void;
}

type SlotKey = 'canvas' | 'aux' | 'wiki';

const SLOT_TITLES: Record<SlotKey, string> = {
  canvas: 'Canvas',
  aux: 'Aux',
  wiki: 'Wiki',
};

const SLOT_ORDER: SlotKey[] = ['canvas', 'wiki', 'aux'];

/** Mirror of `pick_slot_for_widget` in `crates/gw-ui/src/surface.rs`.
 *  Wiki-style widgets → wiki, EntityCloud → canvas, else → aux. */
function pickSlot(w: Widget): SlotKey {
  if (!('Inline' in w.payload)) return 'aux';
  const inner = (w.payload as { Inline: unknown }).Inline as
    | { type?: unknown }
    | null;
  const ty = inner && typeof inner.type === 'string' ? inner.type : null;
  if (ty === 'KbDocWiki' || ty === 'KbClusterWiki') return 'wiki';
  if (ty === 'EntityCloud') return 'canvas';
  return 'aux';
}

/** Best-effort human label from the widget's payload. Falls back to
 *  `<kind> · <id-prefix>` for unknown shapes — keeps the drawer
 *  populated for widget kinds the frontend doesn't recognise yet. */
function deriveLabel(w: Widget): string {
  if ('Inline' in w.payload) {
    const inner = (w.payload as { Inline: unknown }).Inline as
      | Record<string, unknown>
      | null;
    if (inner && typeof inner === 'object') {
      const ty = typeof inner.type === 'string' ? (inner.type as string) : null;
      if (ty === 'KbDocWiki') {
        const doc = inner.doc as { source?: { title?: string } } | undefined;
        const title = doc?.source?.title;
        if (typeof title === 'string' && title.length > 0) return title;
      }
      if (ty === 'KbClusterWiki') {
        const cluster = inner.cluster as { title?: string } | undefined;
        if (typeof cluster?.title === 'string' && cluster.title.length > 0) {
          return cluster.title;
        }
      }
      if (ty === 'EntityCloud') {
        const points = Array.isArray(inner.points) ? inner.points : [];
        return `${points.length} ${points.length === 1 ? 'paper' : 'papers'}`;
      }
      if (ty && typeof ty === 'string') {
        return `${ty} · ${w.id.slice(0, 6)}`;
      }
    }
  }
  return `${describeKind(w)} · ${w.id.slice(0, 6)}`;
}

function describeKind(w: Widget): string {
  if (w.kind === 'A2ui') return 'A2ui';
  if (w.kind === 'McpUi') return 'McpUi';
  return `Custom(${w.kind.Custom})`;
}

function relativeTime(iso: string, now: number): string {
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return '';
  const dt = Math.max(0, Math.round((now - t) / 1000));
  if (dt < 60) return `${dt}s ago`;
  if (dt < 3600) return `${Math.round(dt / 60)}m ago`;
  if (dt < 86400) return `${Math.round(dt / 3600)}h ago`;
  return `${Math.round(dt / 86400)}d ago`;
}

export function WidgetHistoryDrawer({ sessionId, open, onClose }: Props) {
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const canvasSlot = useStateValue<string | null>('/canvasSlot') ?? null;
  const canvasAuxSlot = useStateValue<string | null>('/canvasAuxSlot') ?? null;
  const wikiSlot = useStateValue<string | null>('/wikiSlot') ?? null;

  const groups = useMemo(() => {
    const byGroup: Record<SlotKey, Widget[]> = { canvas: [], aux: [], wiki: [] };
    for (const w of Object.values(widgets)) {
      if (w.state === 'Expired') continue;
      byGroup[pickSlot(w)].push(w);
    }
    for (const key of SLOT_ORDER) {
      byGroup[key].sort((a, b) => (a.created_at < b.created_at ? 1 : -1));
    }
    return byGroup;
  }, [widgets]);

  const totalCount = groups.canvas.length + groups.aux.length + groups.wiki.length;

  if (!open) return null;

  const activeForSlot: Record<SlotKey, string | null> = {
    canvas: canvasSlot,
    aux: canvasAuxSlot,
    wiki: wikiSlot,
  };

  const now = Date.now();
  return (
    <div className="widget-history-backdrop" onClick={onClose}>
      <aside
        className="widget-history-drawer"
        onClick={(e) => e.stopPropagation()}
        aria-label="Widget history"
      >
        <header className="widget-history-header">
          <span className="widget-history-title">History</span>
          <span className="widget-history-count">{totalCount} widget{totalCount === 1 ? '' : 's'}</span>
          <button
            type="button"
            className="widget-history-close"
            onClick={onClose}
            title="Close history"
            aria-label="Close history"
          >
            ×
          </button>
        </header>
        <div className="widget-history-body">
          {totalCount === 0 && (
            <div className="widget-history-empty">
              No widgets emitted yet. Run a search or open a paper to populate history.
            </div>
          )}
          {SLOT_ORDER.map((slot) => {
            const items = groups[slot];
            if (items.length === 0) return null;
            return (
              <section key={slot} className={`widget-history-group widget-history-group-${slot}`}>
                <h3 className="widget-history-group-title">{SLOT_TITLES[slot]}</h3>
                <ul className="widget-history-list">
                  {items.map((w) => {
                    const isActive = activeForSlot[slot] === w.id;
                    return (
                      <li
                        key={w.id}
                        className={`widget-history-row${isActive ? ' active' : ''}`}
                      >
                        <button
                          type="button"
                          className="widget-history-row-btn"
                          onClick={() => {
                            if (isActive) return;
                            void postWidgetEvent(sessionId, {
                              widget_id: w.id,
                              surface_id: w.surface_id,
                              action: 'restore_widget',
                              data: {},
                            }).catch(() => {
                              /* surfaced via stream-error path on next event */
                            });
                          }}
                          disabled={isActive}
                          title={isActive ? 'Currently in this slot' : 'Restore to slot'}
                        >
                          <span className="widget-history-row-label">{deriveLabel(w)}</span>
                          <span className="widget-history-row-meta">
                            <span className="widget-history-row-time">{relativeTime(w.created_at, now)}</span>
                            {isActive && <span className="widget-history-row-active">active</span>}
                          </span>
                        </button>
                      </li>
                    );
                  })}
                </ul>
              </section>
            );
          })}
        </div>
      </aside>
    </div>
  );
}
