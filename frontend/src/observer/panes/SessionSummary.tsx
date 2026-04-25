import type { ObserverState } from '../ingest';

interface Props {
  sessionId: string;
  state: ObserverState;
  connected: boolean;
  error: string | null;
  now: number;
}

export function SessionSummary({
  sessionId,
  state,
  connected,
  error,
  now,
}: Props) {
  const age =
    state.connectedAt != null ? Math.floor((now - state.connectedAt) / 1000) : 0;
  const lastAge =
    state.lastEventAt != null
      ? Math.floor((now - state.lastEventAt) / 1000)
      : null;
  const toolCallCount = Object.keys(state.toolCalls).length;
  const widgetCount = Object.keys(state.widgets).length;

  return (
    <div className="obs-pane obs-summary">
      <header className="obs-pane-head">
        Session <span className="obs-head-meta">{sessionId.slice(0, 8)}…</span>
      </header>
      <div className="obs-summary-grid">
        <Stat
          label="status"
          value={connected ? 'connected' : error ? 'error' : 'connecting…'}
          kind={connected ? 'ok' : error ? 'err' : 'dim'}
        />
        <Stat label="age" value={`${age}s`} />
        <Stat
          label="last event"
          value={lastAge == null ? '—' : `${lastAge}s ago`}
        />
        <Stat label="events" value={`${state.eventCount}`} />
        <Stat label="turns" value={`${state.turns.length}`} />
        <Stat label="tool calls" value={`${toolCallCount}`} />
        <Stat label="widgets" value={`${widgetCount}`} />
        <Stat
          label="focused"
          value={
            Object.entries(state.focusedScope)
              .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
              .join(', ') || '—'
          }
        />
      </div>
      {error && <div className="obs-error-banner">{error}</div>}
    </div>
  );
}

function Stat({
  label,
  value,
  kind,
}: {
  label: string;
  value: string;
  kind?: 'ok' | 'err' | 'dim';
}) {
  return (
    <div className={`obs-stat obs-stat-${kind ?? ''}`}>
      <div className="obs-stat-label">{label}</div>
      <div className="obs-stat-value">{value}</div>
    </div>
  );
}
