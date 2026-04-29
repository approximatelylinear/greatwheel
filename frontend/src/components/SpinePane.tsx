import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { SpineSidebar } from './SpineSidebar';

/**
 * "Semantic spine" rail that runs alongside the chat as a true
 * timeline. Each marker corresponds to a contiguous segment of session
 * entries the backend grouped together via shared entities, and is
 * positioned at the y-offset of its `entry_first` chat row inside the
 * chat scroll content.
 *
 * Layout:
 *   .spine-pane
 *     .spine-rail-scroll  ← own overflow, scroll-synced with .chat-pane
 *       .spine-rail        ← height = chat.scrollHeight; markers absolute
 *     .spine-sidebar       ← bounded, sits below the rail
 *
 * The chat pane is the source of truth for content positions: rows
 * carry `data-entry-id` attributes (stamped by ChatPane from
 * `Message.entryId`, which flows from `TEXT_MESSAGE_START.entry_id`).
 * On every chat scroll/mutation/resize we re-measure each segment's
 * anchor and absolute-position the matching marker.
 *
 * User messages don't carry entry_id today, so segments whose
 * `entry_first` is a user message fall back to `entry_last`, then to
 * even-spacing so the marker still renders.
 *
 * Phase A interactivity (Issue #4): clicks fire a `focus` widget
 * event whose adapter short-circuits the agent loop and only updates
 * `/focusedScope/segment`. The matching marker lights up; a focus
 * card appears under the rail with the segment's detail.
 */
export interface SpineSegment {
  id: string;
  label: string;
  kind: string;
  entry_first: string;
  entry_last: string;
  entity_count: number;
  entity_ids: string[];
  summary?: string | null;
}

interface Props {
  segments: SpineSegment[];
  /** From `/focusedScope/segment` — the segment the user clicked. */
  focusedSegmentId: string | null;
  /** Fires a focus event the adapter routes to scope state without
   *  running the agent. Receives the clicked segment's id. */
  onSegmentFocus: (segmentId: string) => void;
  /** Session id, used by the sidebar to fetch
   *  /sessions/:id/segments/:segment_id. Optional so the registry
   *  fallback path (no session context) doesn't break. */
  sessionId?: string;
  /** Spine action-menu click — forwards to App.tsx which fires the
   *  matching widget event. Optional for the registry fallback. */
  onSegmentAction?: (
    segmentId: string,
    action: 'revisit' | 'expand' | 'compare',
  ) => void;
}

const KIND_COLORS: Record<string, string> = {
  comparison: '#7ba5ff', // blue — weighing options
  decision: '#d6a56e', // amber — committing to one
  deep_dive: '#b48cde', // purple — exploring depth
  construction: '#7bd38f', // green — building
  other: '#a8b1bf', // grey — fallback
};

function kindColor(kind: string): string {
  return KIND_COLORS[kind] ?? KIND_COLORS.other!;
}

interface RailLayout {
  /** Per-segment top offset (px) inside the chat's scrollable content. */
  tops: Record<string, number>;
  /** Total scroll-content height to mirror on the rail. */
  contentHeight: number;
}

const FALLBACK_MARKER_HEIGHT = 56;

export function SpinePane({
  segments,
  focusedSegmentId,
  onSegmentFocus,
  sessionId,
  onSegmentAction,
}: Props) {
  const railScrollRef = useRef<HTMLDivElement>(null);
  const [layout, setLayout] = useState<RailLayout | null>(null);

  // Measurement is cheap (O(segments + DOM anchors)) but we still want
  // to debounce relayout calls into a single rAF tick so a burst of
  // mutations during streaming doesn't thrash setState.
  const rafRef = useRef<number | null>(null);
  const segmentsRef = useRef(segments);
  segmentsRef.current = segments;

  const measure = useCallback(() => {
    rafRef.current = null;
    const chat = document.querySelector<HTMLElement>('.chat-pane');
    if (!chat) return;
    const segs = segmentsRef.current;
    if (segs.length === 0) {
      setLayout({ tops: {}, contentHeight: 0 });
      return;
    }
    const chatRect = chat.getBoundingClientRect();
    const anchors = chat.querySelectorAll<HTMLElement>('[data-entry-id]');
    const anchorById = new Map<string, HTMLElement>();
    anchors.forEach((el) => {
      const id = el.dataset.entryId;
      if (id) anchorById.set(id, el);
    });
    const contentHeight = Math.max(chat.scrollHeight, chatRect.height);
    const anchorTop = (id: string): number | null => {
      const el = anchorById.get(id);
      if (!el) return null;
      return el.getBoundingClientRect().top - chatRect.top + chat.scrollTop;
    };
    const fallbackStep = Math.max(
      contentHeight / Math.max(segs.length, 1),
      FALLBACK_MARKER_HEIGHT,
    );
    const tops: Record<string, number> = {};
    segs.forEach((seg, idx) => {
      const top =
        anchorTop(seg.entry_first) ??
        anchorTop(seg.entry_last) ??
        idx * fallbackStep;
      tops[seg.id] = top;
    });
    setLayout({ tops, contentHeight });
  }, []);

  const scheduleMeasure = useCallback(() => {
    if (rafRef.current != null) return;
    rafRef.current = window.requestAnimationFrame(measure);
  }, [measure]);

  // Wire chat scroll/mutation/resize → relayout, and sync our rail
  // scroll position with chat scroll so markers stay anchored.
  useLayoutEffect(() => {
    const chat = document.querySelector<HTMLElement>('.chat-pane');
    if (!chat) return;

    measure();

    const mo = new MutationObserver(scheduleMeasure);
    mo.observe(chat, { childList: true, subtree: true, characterData: true });
    const ro = new ResizeObserver(scheduleMeasure);
    ro.observe(chat);

    const onChatScroll = () => {
      const r = railScrollRef.current;
      if (r) r.scrollTop = chat.scrollTop;
    };
    chat.addEventListener('scroll', onChatScroll, { passive: true });
    onChatScroll();

    return () => {
      mo.disconnect();
      ro.disconnect();
      chat.removeEventListener('scroll', onChatScroll);
      if (rafRef.current != null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
      }
    };
  }, [measure, scheduleMeasure]);

  // Re-measure when the segment list itself changes shape — a label
  // rename without new chat content wouldn't trip the chat MO.
  useEffect(() => {
    scheduleMeasure();
  }, [segments, scheduleMeasure]);

  if (segments.length === 0) {
    return (
      <aside className="spine-pane spine-empty">
        <div className="spine-empty-text">
          Spine appears once the conversation has a few topics.
        </div>
      </aside>
    );
  }

  const contentHeight = layout?.contentHeight ?? 0;

  return (
    <aside className="spine-pane">
      <div className="spine-rail-scroll" ref={railScrollRef}>
        <div
          className="spine-rail"
          style={{
            height: contentHeight ? `${contentHeight}px` : undefined,
          }}
        >
          {segments.map((seg) => {
            const isActive = seg.id === focusedSegmentId;
            const top = layout?.tops[seg.id];
            return (
              <button
                key={seg.id}
                type="button"
                className={`spine-marker${isActive ? ' active' : ''}`}
                aria-pressed={isActive}
                onClick={() => onSegmentFocus(seg.id)}
                title={`Focus segment "${seg.label}"`}
                style={top != null ? { top: `${top}px` } : undefined}
              >
                <span
                  className="spine-dot"
                  style={{ backgroundColor: kindColor(seg.kind) }}
                  aria-hidden
                />
                <div className="spine-label-card">
                  <div className="spine-label">{seg.label}</div>
                  <div className="spine-meta">
                    <span className={`spine-kind spine-kind-${seg.kind}`}>
                      {seg.kind.replace('_', ' ')}
                    </span>
                    <span className="spine-count">
                      {seg.entity_count}{' '}
                      {seg.entity_count === 1 ? 'entity' : 'entities'}
                    </span>
                  </div>
                </div>
              </button>
            );
          })}
        </div>
      </div>
      {focusedSegmentId && sessionId && (
        <SpineSidebar
          sessionId={sessionId}
          segmentId={focusedSegmentId}
          onAction={(action) =>
            onSegmentAction?.(focusedSegmentId, action)
          }
        />
      )}
    </aside>
  );
}
