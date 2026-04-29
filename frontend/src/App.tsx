import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createStateStore, type StateStore } from '@json-render/core';
import { JSONUIProvider, useStateValue } from '@json-render/react';
import { postMessage, postWidgetEvent } from './api/client';
import { openStream } from './api/sse';
import { useSessionStore } from './store/session';
import { ChatPane } from './components/ChatPane';
import { CanvasPane } from './components/CanvasPane';
import { DebugPane } from './components/DebugPane';
import { MessageInput } from './components/MessageInput';
import { DragSplitter } from './components/DragSplitter';
import { SpinePane, type SpineSegment } from './components/SpinePane';
import { registry } from './jr/registry';
import type { Widget } from './types';
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
    <JSONUIProvider registry={registry} store={store} handlers={handlers}>
      <AppShell
        sessionId={sessionId}
        debug={debug}
        streamError={streamError}
        state={state}
        onSend={onSend}
      />
    </JSONUIProvider>
  );
}

interface AppShellProps {
  sessionId: string;
  debug: boolean;
  streamError: string | null;
  state: ReturnType<typeof useSessionStore>['state'];
  onSend: (content: string) => void;
}

/**
 * Inner shell that lives inside JSONUIProvider so it can read the
 * branding's layout hint to swap the top-level grid (chat-primary
 * vs canvas-primary). Default chat-primary keeps Frankenstein's
 * narrow-right-rail layout; canvas-primary widens the canvas for
 * data demos.
 */
function AppShell({ sessionId, debug, streamError, state, onSend }: AppShellProps) {
  const branding = useStateValue<{ layout?: string | null }>('/branding');
  const layout = branding?.layout ?? 'chat-primary';
  // Widget map + canvas slot — read here so log-line "re-pin" clicks
  // can find the EntityCloud widget and replay the same widget event
  // a point click would have fired.
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const canvasSlot = useStateValue<string | null>('/canvasSlot') ?? null;

  // Find the SemanticSpine widget the AG-UI adapter emits/supersedes
  // on each SpineSegmentsUpdated. Only one is live per session at a
  // time (the supersede chain keeps the chain), so first match wins.
  // We render it in a dedicated third pane rather than inline,
  // because conceptually the spine annotates chat (not a turn output).
  // Find the spine widget once and remember widget_id + surface_id
  // alongside the segment list. The SpinePane needs both to fire
  // postWidgetEvent on marker click; a click with action="focus"
  // doesn't run the agent (adapter short-circuits), it just updates
  // /focusedScope/segment which the rail listens to.
  const spineState = useMemo(() => {
    for (const w of Object.values(widgets)) {
      if (w.kind !== 'A2ui') continue;
      // Skip superseded / resolved widgets — when the spine fires a
      // new SpineSegmentsUpdated, the adapter supersedes the old
      // widget but it stays in /widgets with state="Superseded".
      // Without this filter the walk can return the stale widget,
      // leaving the rail showing entity_count=0 even though a newer
      // widget with the real count is also present.
      if (w.state !== 'Active') continue;
      if (!('Inline' in w.payload)) continue;
      const inline = (w.payload as { Inline: unknown }).Inline as
        | { type?: unknown; segments?: unknown }
        | null;
      if (!inline || (inline as { type?: unknown }).type !== 'SemanticSpine') {
        continue;
      }
      const raw = Array.isArray(inline.segments)
        ? (inline.segments as Array<Record<string, unknown>>)
        : [];
      const segments: SpineSegment[] = raw.map((s) => ({
        id: String(s.id ?? ''),
        label: String(s.label ?? ''),
        kind: String(s.kind ?? 'other'),
        entry_first: String(s.entry_first ?? ''),
        entry_last: String(s.entry_last ?? ''),
        entity_count:
          typeof s.entity_count === 'number' ? s.entity_count : 0,
        entity_ids: Array.isArray(s.entity_ids)
          ? (s.entity_ids as unknown[]).map((x) => String(x))
          : [],
        summary: s.summary != null ? String(s.summary) : null,
      }));
      return {
        segments,
        widgetId: w.id,
        surfaceId: w.surface_id,
      };
    }
    return null;
  }, [widgets]);
  const spineSegments = spineState?.segments ?? null;
  const focusedSegmentId =
    useStateValue<string | null>('/focusedScope/segment') ?? null;

  const onSegmentFocus = useCallback(
    (segmentId: string) => {
      if (!spineState) return;
      void postWidgetEvent(sessionId, {
        widget_id: spineState.widgetId,
        surface_id: spineState.surfaceId,
        action: 'focus',
        data: {
          segment_id: segmentId,
          scope: { kind: 'segment', key: segmentId },
        },
      }).catch(() => {
        /* surfaced via stream-error path on next event */
      });
    },
    [sessionId, spineState],
  );

  // Spine action menu: revisit / expand / compare. Each fires a
  // WidgetInteraction with the action verb; the adapter forwards
  // it to ConversationLoop, which translates the action + segment
  // detail into a server-side templated prompt and runs the next
  // turn. The user sees a real agent response with the synthetic
  // prompt in the conversation log.
  const onSegmentAction = useCallback(
    (segmentId: string, action: 'revisit' | 'expand' | 'compare') => {
      if (!spineState) return;
      void postWidgetEvent(sessionId, {
        widget_id: spineState.widgetId,
        surface_id: spineState.surfaceId,
        action,
        data: { segment_id: segmentId },
      }).catch(() => {
        /* surfaced via stream-error path on next event */
      });
    },
    [sessionId, spineState],
  );

  const onRepin = useCallback(
    (arxivId: string) => {
      const cloud = findEntityCloud(widgets, canvasSlot);
      if (!cloud) return;
      void postWidgetEvent(sessionId, {
        widget_id: cloud.id,
        surface_id: cloud.surface_id,
        action: 'select',
        data: {
          pointId: arxivId,
          scope: { kind: 'paper', key: arxivId },
        },
      }).catch(() => {
        /* surfaced via stream-error path on next event */
      });
    },
    [sessionId, widgets, canvasSlot],
  );

  return (
    <div className={`app app-${layout}`}>
      <header className="app-header">
        <BrandedTitle />
        {streamError && <span className="app-error">{streamError}</span>}
        <span className="app-mark" title={`session ${sessionId}`}>greatwheel</span>
      </header>
      <main
        className={`app-main app-main-${layout}${spineSegments ? ' has-spine' : ''}`}
      >
        <ChatPane
          messages={state.messages}
          running={state.running}
          messageFollowUps={state.messageFollowUps}
          onSuggest={onSend}
          onRepin={onRepin}
        />
        {spineSegments && (
          <SpinePane
            segments={spineSegments}
            focusedSegmentId={focusedSegmentId}
            onSegmentFocus={onSegmentFocus}
            sessionId={sessionId}
            onSegmentAction={onSegmentAction}
          />
        )}
        <DragSplitter storageKey={`app.chatW.${layout}`} />
        <CanvasPane />
      </main>
      {debug && (
        <DebugPane
          traces={state.codeTraces}
          toolCalls={state.toolCalls}
          spineEvents={state.spineEvents}
        />
      )}
      <footer className="app-footer">
        <MessageInput onSend={onSend} disabled={state.running} />
      </footer>
    </div>
  );
}

/**
 * Find the EntityCloud widget in the canonical state so we can replay
 * its point-click events from elsewhere in the UI (e.g. a "re-pin"
 * log-line click in the chat rail). Prefers the widget pinned to the
 * primary canvas slot — the literature_assistant pins the cloud there
 * with `multi_use=True` precisely so its action surface stays
 * available across drill-downs.
 */
function findEntityCloud(
  widgets: Record<string, Widget>,
  canvasSlot: string | null,
): Widget | null {
  const isCloud = (w?: Widget): boolean => {
    if (!w || w.kind !== 'A2ui') return false;
    if (!('Inline' in w.payload)) return false;
    const inner = (w.payload as { Inline: unknown }).Inline as
      | { type?: unknown }
      | null;
    return !!inner && (inner as { type?: unknown }).type === 'EntityCloud';
  };
  if (canvasSlot && isCloud(widgets[canvasSlot])) return widgets[canvasSlot]!;
  for (const w of Object.values(widgets)) {
    if (isCloud(w)) return w;
  }
  return null;
}

/**
 * Header brand block sourced from `/branding` in the canonical state
 * (set per-demo via `AgUiAdapter::set_branding` on the server). Falls
 * back to "greatwheel" until the first STATE_SNAPSHOT lands.
 */
function BrandedTitle() {
  const branding = useStateValue<{ title: string; subtitle: string }>(
    '/branding',
  );
  return (
    <div className="app-brand">
      <span className="app-title">{branding?.title ?? 'greatwheel'}</span>
      {branding?.subtitle && (
        <span className="app-subtitle">{branding.subtitle}</span>
      )}
    </div>
  );
}
