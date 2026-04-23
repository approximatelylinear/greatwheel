import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createStateStore, type StateStore } from '@json-render/core';
import { JSONUIProvider } from '@json-render/react';
import { fetchSurface, postMessage, postWidgetEvent } from './api/client';
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
  const { state, hydrate, appendUser, markRunning, pressButton, ingest: sessionIngest } =
    useSessionStore();

  // json-render StateStore seeded with the canonical shape. Populated
  // by STATE_SNAPSHOT + STATE_DELTA events via stateBridge. Phase 3a:
  // coexists with the session reducer — reducer still drives
  // components; this store is the migration target for phase 3b.
  const storeRef = useRef<StateStore | null>(null);
  if (!storeRef.current) {
    storeRef.current = createStateStore(INITIAL_CANONICAL_STATE);
  }
  const store = storeRef.current;

  const ingest = useCallback(
    (ev: Parameters<typeof sessionIngest>[0]) => {
      sessionIngest(ev);
      applyAgUiEventToStore(store, ev);
    },
    [sessionIngest, store],
  );

  useEffect(() => {
    if (!sessionId) return;
    let closed = false;
    let close: (() => void) | null = null;
    // Hydrate durable widget state from the server before subscribing
    // to the live event stream. Survives browser refresh — the session
    // messages are ephemeral but widgets live in UiSurfaceStore.
    fetchSurface(sessionId)
      .then((snapshot) => {
        if (closed) return;
        hydrate(snapshot);
      })
      .catch((e) => setStreamError(String(e)))
      .finally(() => {
        if (closed) return;
        close = openStream(
          sessionId,
          (ev) => ingest(ev),
          (e) => setStreamError(String(e)),
        );
      });
    return () => {
      closed = true;
      close?.();
    };
    // hydrate/sessionIngest are stable across renders (useReducer dispatch).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const canvasWidget = useMemo(() => {
    if (!state.canvasSlot) return null;
    return state.widgets[state.canvasSlot] ?? null;
  }, [state.canvasSlot, state.widgets]);

  const canvasAuxWidget = useMemo(() => {
    if (!state.canvasAuxSlot) return null;
    return state.widgets[state.canvasAuxSlot] ?? null;
  }, [state.canvasAuxSlot, state.widgets]);

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

  // Single interact handler for every widget in the app. json-render
  // resolves `on.press` bindings against this handler; the catalog
  // translator bakes widget_id/surface_id into each binding's params
  // so we can route each click to the right backend session.
  const handlers = useMemo(
    () => ({
      interact: async (params: Record<string, unknown>) => {
        const widgetId = String(params.widgetId);
        const buttonId = String(params.buttonId);
        pressButton(widgetId, buttonId);
        markRunning();
        try {
          await postWidgetEvent(sessionId, {
            widget_id: widgetId,
            surface_id: String(params.surfaceId),
            action: String(params.action),
            data: params.data,
          });
        } catch (e) {
          setStreamError(String(e));
        }
      },
    }),
    [sessionId, pressButton, markRunning],
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
            widgets={state.widgets}
            widgetOrder={state.widgetOrder}
            pinnedIds={state.pinnedIds}
            running={state.running}
            pressedButtonIds={state.pressedButtonIds}
            messageFollowUps={state.messageFollowUps}
            onSuggest={onSend}
          />
          <CanvasPane
            widget={canvasWidget}
            auxWidget={canvasAuxWidget}
            pressedId={
              canvasWidget ? state.pressedButtonIds[canvasWidget.id] ?? null : null
            }
            auxPressedId={
              canvasAuxWidget
                ? state.pressedButtonIds[canvasAuxWidget.id] ?? null
                : null
            }
          />
        </main>
      </JSONUIProvider>
      {debug && <DebugPane traces={state.codeTraces} />}
      <footer className="app-footer">
        <MessageInput onSend={onSend} disabled={state.running} />
      </footer>
    </div>
  );
}
