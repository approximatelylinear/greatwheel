/**
 * Read-only "semantic spine" rail that runs alongside the chat.
 * Each marker corresponds to a contiguous segment of session entries
 * the backend grouped together via shared entities. The widget
 * itself only re-renders when the backend supersedes it (new
 * `SpineSegmentsUpdated` arrives from the LoopEvent stream).
 *
 * Active-segment scroll sync (the "you are here" highlight as the
 * user scrolls chat) needs the chat pane to stamp each message
 * with its `session_entries.id`. Today the chat pane sees
 * `Message.id`, which is the AG-UI `message_id` — a different
 * UUID. Plumbing entry IDs through SSE → store → ChatPane is its
 * own work; scroll sync re-lights once that lands. For v1 the
 * spine renders the segments cleanly with no active highlight,
 * which is enough to validate the data is flowing.
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

export function SpinePane({ segments }: Props) {
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
        {segments.map((seg) => (
          <div key={seg.id} className="spine-marker">
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
          </div>
        ))}
      </div>
    </aside>
  );
}
