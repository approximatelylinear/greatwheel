import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createStateStore, type StateStore } from '@json-render/core';
import { JSONUIProvider } from '@json-render/react';
import { postMessage, postWidgetEvent } from './api/client';
import { openStream } from './api/sse';
import { useSessionStore } from './store/session';
import { ChatPane } from './components/ChatPane';
import { CanvasPane } from './components/CanvasPane';
import { DebugPane } from './components/DebugPane';
import { MessageInput } from './components/MessageInput';
import { registry } from './jr/registry';
import {
  INITIAL_CANONICAL_STATE,
  applyAgUiEventToStore,
} from './jr/stateBridge';

/**
 * Session ID source: ?session=<uuid> in the URL, else VITE_SESSION_ID,
 * else "". If empty, we render a placeholder asking the user to set
 * one. The echo_server example prints its session UUID on startup.
 */
function resolveSessionId(): string {
  const url = new URL(window.location.href);
  const fromUrl = url.searchParams.get('session');
  if (fromUrl) return fromUrl;
  const fromEnv = (import.meta.env.VITE_SESSION_ID as string | undefined) ?? '';
  return fromEnv;
}

function debugEnabled(): boolean {
  const url = new URL(window.location.href);
  return url.searchParams.get('debug') === '1';
}

export function App() {
  const [sessionId] = useState(resolveSessionId);
  const [debug] = useState(debugEnabled);
  const [streamError, setStreamError] = useState<string | null>(null);
  const { state, appendUser, markRunning, widgetAdded, ingest: sessionIngest } =
    useSessionStore();

  // Single json-render StateStore for the whole app. Populated by
  // STATE_SNAPSHOT + STATE_DELTA events via the stateBridge helper;
  // all widget/canvas/pressed/focusedScope consumers bind to it
  // through useStateValue rather than reducer props.
  const storeRef = useRef<StateStore | null>(null);
  if (!storeRef.current) {
    storeRef.current = createStateStore(INITIAL_CANONICAL_STATE);
  }
  const store = storeRef.current;

  const ingest = useCallback(
    (ev: Parameters<typeof sessionIngest>[0]) => {
      sessionIngest(ev);
      applyAgUiEventToStore(store, ev, { onWidgetAdded: widgetAdded });
    },
    [sessionIngest, store, widgetAdded],
  );

  useEffect(() => {
    if (!sessionId) return;
    // STATE_SNAPSHOT on subscribe hydrates the store; no separate
    // /surface fetch needed.
    const close = openStream(
      sessionId,
      (ev) => ingest(ev),
      (e) => setStreamError(String(e)),
    );
    return close;
    // sessionIngest is stable (useReducer dispatch).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  if (!sessionId) {
    return (
      <div className="no-session">
        <h1>greatwheel</h1>
        <p>
          No session ID configured. Start the echo server
          (<code>cargo run -p gw-ui --example echo_server</code>) and append
          <code>?session=&lt;uuid&gt;</code> to this URL, or set
          <code>VITE_SESSION_ID</code> and restart Vite.
        </p>
      </div>
    );
  }

  const onSend = async (content: string) => {
    appendUser(content);
    try {
      await postMessage(sessionId, content);
    } catch (e) {
      setStreamError(String(e));
    }
  };

  // Single interact handler for every widget. The catalog translator
  // bakes widget_id / surface_id into each `on.press` ActionBinding,
  // so this handler routes to the right backend session. Local
  // pressed-state is no longer mirrored here — the server's
  // STATE_DELTA writes `/pressed/<widget_id>` and the store re-emits
  // to all $state bindings automatically.
  const handlers = useMemo(
    () => ({
      interact: async (params: Record<string, unknown>) => {
        markRunning();
        try {
          await postWidgetEvent(sessionId, {
            widget_id: String(params.widgetId),
            surface_id: String(params.surfaceId),
            action: String(params.action),
            data: params.data,
          });
        } catch (e) {
          setStreamError(String(e));
        }
      },
    }),
    [sessionId, markRunning],
  );

  return (
    <div className="app">
      <header className="app-header">
        <div className="app-brand">
          <span className="app-title">Frankenstein</span>
          <span className="app-subtitle">a conversational reading companion</span>
        </div>
        {streamError && <span className="app-error">{streamError}</span>}
        <span className="app-mark" title={`session ${sessionId}`}>greatwheel</span>
      </header>
      <JSONUIProvider registry={registry} store={store} handlers={handlers}>
        <main className="app-main">
          <ChatPane
            messages={state.messages}
            running={state.running}
            messageFollowUps={state.messageFollowUps}
            onSuggest={onSend}
          />
          <CanvasPane />
        </main>
      </JSONUIProvider>
      {debug && <DebugPane traces={state.codeTraces} />}
      <footer className="app-footer">
        <MessageInput onSend={onSend} disabled={state.running} />
      </footer>
    </div>
  );
}
