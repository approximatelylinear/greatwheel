import type { TurnRecord } from '../ingest';

interface Props {
  turns: TurnRecord[];
  now: number;
}

/**
 * Horizontal bar per turn, scaled to session duration. Colour-coded
 * by status; width proportional to turn wall time.
 */
export function TurnTimeline({ turns, now }: Props) {
  if (turns.length === 0) {
    return (
      <div className="obs-pane">
        <header className="obs-pane-head">Turns</header>
        <div className="obs-empty">No turns yet.</div>
      </div>
    );
  }
  const sessionStart = turns[0]!.startedAt;
  const sessionEnd = turns[turns.length - 1]!.completedAt ?? now;
  const total = Math.max(1, sessionEnd - sessionStart);

  return (
    <div className="obs-pane">
      <header className="obs-pane-head">
        Turns <span className="obs-head-meta">{turns.length}</span>
      </header>
      <div className="obs-timeline">
        {turns.map((t) => {
          const start = t.startedAt - sessionStart;
          const end = (t.completedAt ?? now) - sessionStart;
          const leftPct = (start / total) * 100;
          const widthPct = Math.max(0.5, ((end - start) / total) * 100);
          const duration = (t.completedAt ?? now) - t.startedAt;
          return (
            <div key={t.id} className="obs-turn-row">
              <div className="obs-turn-meta">
                <span className="obs-turn-id">{t.id}</span>
                <span className={`obs-turn-status obs-turn-${t.status}`}>
                  {t.status}
                </span>
                <span className="obs-turn-dur">{duration}ms</span>
                <span className="obs-turn-count">
                  {t.toolCallIds.length} calls · {t.messageIds.length} msgs
                </span>
              </div>
              <div className="obs-turn-bar-track">
                <div
                  className={`obs-turn-bar obs-turn-bar-${t.status}`}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  title={`${duration}ms`}
                />
              </div>
              {t.error && <div className="obs-turn-error">⚠ {t.error}</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
