import { useEffect, useMemo, useState } from 'react';
import type { WidgetEvent } from './types';
import { fetchSurface, postMessage, postWidgetEvent } from './api/client';
import { openStream } from './api/sse';
import { useSessionStore } from './store/session';
import { ChatPane } from './components/ChatPane';
import { CanvasPane } from './components/CanvasPane';
import { DebugPane } from './components/DebugPane';
import { MessageInput } from './components/MessageInput';

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
  const { state, hydrate, appendUser, markRunning, pressButton, ingest } = useSessionStore();

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
    // hydrate/ingest are stable across renders (from useReducer dispatch).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  const canvasWidget = useMemo(() => {
    if (!state.canvasSlot) return null;
    return state.widgets[state.canvasSlot] ?? null;
  }, [state.canvasSlot, state.widgets]);

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

  const onInteract = async (ev: WidgetEvent, buttonId: string) => {
    pressButton(ev.widget_id, buttonId);
    markRunning();
    try {
      await postWidgetEvent(sessionId, ev);
    } catch (e) {
      setStreamError(String(e));
    }
  };

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
      <main className="app-main">
        <ChatPane
          messages={state.messages}
          widgets={state.widgets}
          widgetOrder={state.widgetOrder}
          canvasSlot={state.canvasSlot}
          running={state.running}
          pressedButtonIds={state.pressedButtonIds}
          messageFollowUps={state.messageFollowUps}
          onInteract={onInteract}
        />
        <CanvasPane
          widget={canvasWidget}
          pressedId={
            canvasWidget ? state.pressedButtonIds[canvasWidget.id] ?? null : null
          }
          onInteract={onInteract}
        />
      </main>
      {debug && <DebugPane traces={state.codeTraces} />}
      <footer className="app-footer">
        <MessageInput onSend={onSend} disabled={state.running} />
      </footer>
    </div>
  );
}
