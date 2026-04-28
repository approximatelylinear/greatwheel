import { SpineSidebar } from './SpineSidebar';

/**
 * Read-only "semantic spine" rail that runs alongside the chat.
 * Each marker corresponds to a contiguous segment of session entries
 * the backend grouped together via shared entities.
 *
 * Phase A interactivity (Issue #4): clicks fire a `focus` widget
 * event whose adapter short-circuits the agent loop and only updates
 * `/focusedScope/segment`. The rail re-reads that scope and the
 * matching marker lights up; a focus card appears below the rail
 * with a stub of the segment's detail (entity ids, count). The
 * full sidebar with detail-endpoint-fed entity cards lives in the
 * follow-up phase.
 *
 * Active-segment scroll sync (the auto-highlight as the user scrolls
 * chat) needs the chat pane to stamp each message with its
 * `session_entries.id`. Today `Message.id` is the AG-UI
 * `message_id` — a different UUID. Plumbing entry IDs through
 * SSE → store → ChatPane is its own work; the rail keeps user-
 * driven focus until that lands.
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

export function SpinePane({
  segments,
  focusedSegmentId,
  onSegmentFocus,
  sessionId,
}: Props) {
  if (segments.length === 0) {
    return (
      <aside className="spine-pane spine-empty">
        <div className="spine-empty-text">
          Spine appears once the conversation has a few topics.
        </div>
      </aside>
    );
  }

  return (
    <aside className="spine-pane">
      <div className="spine-rail">
        {segments.map((seg) => {
          const isActive = seg.id === focusedSegmentId;
          return (
            <button
              key={seg.id}
              type="button"
              className={`spine-marker${isActive ? ' active' : ''}`}
              aria-pressed={isActive}
              onClick={() => onSegmentFocus(seg.id)}
              title={`Focus segment "${seg.label}"`}
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
      {focusedSegmentId && sessionId && (
        <SpineSidebar
          sessionId={sessionId}
          segmentId={focusedSegmentId}
        />
      )}
    </aside>
  );
}
