import type { PatchEntry } from '../ingest';

interface Props {
  entries: PatchEntry[];
}

/**
 * Reverse-chronological feed of every JSON-Patch op seen on
 * STATE_DELTA events. Values truncated for display.
 */
export function StateDeltaLog({ entries }: Props) {
  return (
    <div className="obs-pane">
      <header className="obs-pane-head">
        State patches <span className="obs-head-meta">{entries.length}</span>
      </header>
      <div className="obs-patch-list">
        {entries.length === 0 && (
          <div className="obs-empty">No patches yet.</div>
        )}
        {entries
          .slice()
          .reverse()
          .map((e, i) => (
            <div key={`${e.at}-${i}`} className="obs-patch-row">
              <span className="obs-patch-time">{formatTime(e.at)}</span>
              <span className={`obs-patch-op obs-patch-op-${e.op}`}>{e.op}</span>
              <span className="obs-patch-path">{e.path}</span>
              {e.value !== undefined && (
                <span className="obs-patch-value">{truncate(e.value)}</span>
              )}
            </div>
          ))}
      </div>
    </div>
  );
}

function formatTime(at: number): string {
  const d = new Date(at);
  return `${d.getMinutes().toString().padStart(2, '0')}:${d
    .getSeconds()
    .toString()
    .padStart(2, '0')}.${d.getMilliseconds().toString().padStart(3, '0')}`;
}

function truncate(v: unknown, max = 80): string {
  const s = typeof v === 'string' ? v : JSON.stringify(v);
  if (!s) return '';
  return s.length > max ? `${s.slice(0, max)}…` : s;
}
