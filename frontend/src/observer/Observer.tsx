import { useEffect, useReducer, useState } from 'react';
import { openStream } from '../api/sse';
import {
  initialObserverState,
  observerReducer,
  type ObserverState,
} from './ingest';
import type { AgUiEvent } from '../types';
import { TurnTimeline } from './panes/TurnTimeline';
import { StateDeltaLog } from './panes/StateDeltaLog';
import { SessionSummary } from './panes/SessionSummary';

interface Props {
  sessionId: string;
}

/**
 * Read-only observer: subscribes to a target session's /stream and
 * reconstructs a diagnostic dashboard from the AG-UI events alone.
 * Never writes to the session. See docs/design-demo-observer.md.
 */
export function Observer({ sessionId }: Props) {
  const [state, dispatch] = useReducer<ObserverState, [AgUiEvent]>(
    observerReducer,
    initialObserverState,
  );
  const [connected, setConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Recomputed every second so elapsed-time displays don't freeze
  // when no events arrive.
  const [now, setNow] = useState(Date.now());

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    if (!sessionId) return;
    setConnected(false);
    setError(null);
    const close = openStream(
      sessionId,
      (ev) => {
        setConnected(true);
        dispatch(ev);
      },
      (e) => {
        setError(String(e));
        setConnected(false);
      },
    );
    return close;
  }, [sessionId]);

  return (
    <div className="obs-app">
      <header className="obs-header">
        <div className="obs-brand">
          <span className="obs-title">greatwheel observer</span>
          <span className="obs-subtitle">watching session {sessionId}</span>
        </div>
        <span
          className={`obs-pulse ${connected ? 'on' : error ? 'err' : ''}`}
          title={
            connected ? 'connected' : error ? `error: ${error}` : 'connecting…'
          }
        />
      </header>
      <main className="obs-main">
        <SessionSummary
          sessionId={sessionId}
          state={state}
          connected={connected}
          error={error}
          now={now}
        />
        <TurnTimeline turns={state.turns} now={now} />
        <StateDeltaLog entries={state.patchLog} />
      </main>
    </div>
  );
}
