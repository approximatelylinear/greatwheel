import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createStateStore, type StateStore } from '@json-render/core';
import { JSONUIProvider, useStateValue } from '@json-render/react';
import { fetchTranscript, postMessage, postWidgetEvent } from './api/client';
import { useSessionRouting } from './api/users';
import { openStream } from './api/sse';
import { useSessionStore, type Message } from './store/session';
import { ChatPane } from './components/ChatPane';
import { CanvasPane } from './components/CanvasPane';
import { DebugPane } from './components/DebugPane';
import { MessageInput } from './components/MessageInput';
import { DragSplitter } from './components/DragSplitter';
import { SpinePane, type SpineSegment } from './components/SpinePane';
import { SpineSidebar } from './components/SpineSidebar';
import { WorkspaceDrawer } from './components/WorkspaceDrawer';
import { WidgetHistoryDrawer } from './components/WidgetHistoryDrawer';
import { SessionSidebar } from './components/SessionSidebar';
import { WikiPane } from './components/WikiPane';
import type { EntityCard, SegmentDetail } from './api/client';
import { registry } from './jr/registry';
import type { Widget } from './types';
import {
  INITIAL_CANONICAL_STATE,
  applyAgUiEventToStore,
} from './jr/stateBridge';

function debugEnabled(): boolean {
  const url = new URL(window.location.href);
  return url.searchParams.get('debug') === '1';
}

export function App() {
  const {
    userId,
    sessionId,
    displayName,
    ready: userReady,
    setDisplayName,
    switchSession,
  } = useSessionRouting();
  const [debug] = useState(debugEnabled);
  const [streamError, setStreamError] = useState<string | null>(null);
  const {
    state,
    appendUser,
    hydrateTranscript,
    rehydrateFollowUps,
    markRunning,
    widgetAdded,
    ingest: sessionIngest,
  } = useSessionStore();

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
      (err) => setStreamError(err),
    );
    return close;
    // sessionIngest is stable (useReducer dispatch).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Backfill chat transcript on session (re)load. The SSE stream above
  // hydrates widgets via STATE_SNAPSHOT but doesn't replay prior chat
  // messages — those live in `session_entries` (PG) and need an
  // explicit fetch. Runs in parallel with the stream open; on a fresh
  // session the endpoint returns [] and this is a cheap no-op.
  useEffect(() => {
    if (!sessionId) return;
    const ac = new AbortController();
    fetchTranscript(sessionId, ac.signal)
      .then((rows) => {
        if (ac.signal.aborted) return;
        hydrateTranscript(
          rows.map((r) => ({
            id: r.entry_id,
            role: r.role,
            content: r.content,
            entryId: r.entry_id,
            createdAt: r.created_at,
          })),
        );
      })
      .catch((e: unknown) => {
        // 404 on backends without the endpoint mounted is fine — log
        // for debug-level diagnosis rather than surfacing as a stream
        // error (which would block the chat UI).
        if (ac.signal.aborted) return;
        // eslint-disable-next-line no-console
        console.debug('transcript backfill skipped:', e);
      });
    return () => ac.abort();
    // hydrateTranscript is stable (useReducer dispatch).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId]);

  // Rebuild `messageFollowUps` after refresh. The live-built map is
  // reducer-only state — on session reload it's empty, so follow_up
  // widgets fall through to inline rendering at the bottom of the
  // chat, and their pin-ack messages (which used to render as full
  // bubbles with the widget anchored beneath) get demoted to the
  // `.message-log` thin-line styling. Effect: chat looks fine live,
  // looks broken on refresh.
  //
  // We rebuild from timestamps. Both the transcript (assistant
  // messages, with `createdAt`) and the StateStore (widgets, with
  // `created_at`) carry them, so we can recover the live anchoring:
  // each follow_up widget anchors to the next assistant message that
  // arrived *after* it on the wire. Subscribe to the StateStore so
  // newly-arriving widgets re-run the rebuild — the live
  // `widget-emitted` reducer path also adds to the map, but the two
  // converge to the same result so the redundancy is a no-op.
  useEffect(() => {
    if (!sessionId) return;
    const compute = () => {
      const widgetsObj = (store.get('/widgets') ?? {}) as Record<
        string,
        Widget
      >;
      const widgetOrder = (store.get('/widgetOrder') ?? []) as string[];
      const map = rebuildMessageFollowUps(
        state.messages,
        widgetsObj,
        widgetOrder,
      );
      rehydrateFollowUps(map);
    };
    compute();
    return store.subscribe(compute);
    // rehydrateFollowUps is stable; store identity is stable across renders.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [sessionId, state.messages]);

  if (!sessionId) {
    return (
      <div className="no-session">
        <h1>greatwheel</h1>
        <p>{userReady ? 'Opening your session…' : 'Signing you in…'}</p>
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
        userId={userId}
        sessionId={sessionId}
        displayName={displayName}
        setDisplayName={setDisplayName}
        switchSession={switchSession}
        debug={debug}
        streamError={streamError}
        state={state}
        onSend={onSend}
      />
    </JSONUIProvider>
  );
}

interface AppShellProps {
  userId: string;
  sessionId: string;
  displayName: string | null;
  setDisplayName: (name: string) => Promise<void>;
  switchSession: (id: string) => void;
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
function AppShell({
  userId,
  sessionId,
  displayName,
  setDisplayName,
  switchSession,
  debug,
  streamError,
  state,
  onSend,
}: AppShellProps) {
  const branding = useStateValue<{ layout?: string | null }>('/branding');
  const layout = branding?.layout ?? 'chat-primary';
  // Workspace drawer state (Issue #5). `reloadKey` bumps every time
  // a SpineSidebar commit toggle fires so the drawer's open-on-mount
  // refetch sees the new state without us having to keep its list
  // in sync from outside.
  const [workspaceOpen, setWorkspaceOpen] = useState(false);
  const [workspaceReloadKey, setWorkspaceReloadKey] = useState(0);
  const [historyOpen, setHistoryOpen] = useState(false);
  // Issue #6: focused segment's loaded detail, captured from the
  // SpineSidebar so ChatPane can highlight the segment's entities
  // in the corresponding chat rows. Cleared on focus dismiss.
  const [focusedDetail, setFocusedDetail] = useState<SegmentDetail | null>(
    null,
  );
  // Issue #6 follow-up: an entity within the focused segment was
  // clicked — when set, the highlighter switches from "all segment
  // entities, segment range only" to "this one entity, whole chat"
  // and scrolls to the first occurrence.
  const [selectedEntity, setSelectedEntity] = useState<EntityCard | null>(
    null,
  );

  const onSegmentDetailLoaded = useCallback((d: SegmentDetail | null) => {
    setFocusedDetail(d);
    // Drop entity selection when the segment changes / sidebar
    // unmounts; the old entity may not even exist in the new segment.
    setSelectedEntity(null);
  }, []);

  const onSelectEntity = useCallback((entity: EntityCard | null) => {
    setSelectedEntity(entity);
  }, []);

  // Build the highlight terms list. Two modes:
  //   - selectedEntity → just that entity's label + aliases (single
  //     entity highlighted across the whole chat)
  //   - else focusedDetail → every entity in the segment (label +
  //     aliases each) constrained to the segment's row range
  const highlightTerms = useMemo(() => {
    if (selectedEntity) {
      const out: string[] = [];
      if (selectedEntity.label) out.push(selectedEntity.label);
      for (const a of selectedEntity.aliases) {
        if (a) out.push(a);
      }
      return out;
    }
    if (!focusedDetail) return undefined;
    const out: string[] = [];
    for (const e of focusedDetail.entities) {
      if (e.label) out.push(e.label);
      for (const a of e.aliases) {
        if (a) out.push(a);
      }
    }
    return out;
  }, [selectedEntity, focusedDetail]);

  const highlightRange = useMemo(() => {
    // When an entity is selected, drop the row range — we want every
    // occurrence across the chat, not just inside the segment.
    if (selectedEntity) return undefined;
    if (!focusedDetail) return undefined;
    return {
      first: focusedDetail.segment.entry_first,
      last: focusedDetail.segment.entry_last,
    };
  }, [selectedEntity, focusedDetail]);

  // When selected entity changes, scroll the chat to the first
  // matching `<mark>` so the user can find what they just clicked.
  useEffect(() => {
    if (!selectedEntity) return;
    // Defer to next frame so ChatPane has rendered the new marks.
    const handle = window.requestAnimationFrame(() => {
      const mark = document.querySelector<HTMLElement>(
        '.chat-pane .entity-mark',
      );
      if (mark) {
        mark.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    });
    return () => window.cancelAnimationFrame(handle);
  }, [selectedEntity]);
  // Widget map + canvas slot — read here so log-line "re-pin" clicks
  // can find the EntityCloud widget and replay the same widget event
  // a point click would have fired.
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const canvasSlot = useStateValue<string | null>('/canvasSlot') ?? null;
  const wikiSlot = useStateValue<string | null>('/wikiSlot') ?? null;

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

  // Phase D footer: dismiss the focused segment by clearing
  // /focusedScope/segment. The adapter's extract_scope_update
  // accepts `key: null` and emits a STATE_DELTA setting the path
  // to null; the SpinePane stops highlighting and the sidebar
  // unmounts.
  const onSegmentClose = useCallback(() => {
    if (!spineState) return;
    void postWidgetEvent(sessionId, {
      widget_id: spineState.widgetId,
      surface_id: spineState.surfaceId,
      action: 'focus',
      data: {
        segment_id: null,
        scope: { kind: 'segment', key: null },
      },
    }).catch(() => {
      /* surfaced via stream-error path on next event */
    });
  }, [sessionId, spineState]);

  // Phase D footer: scroll the chat to the focused segment's first
  // anchored row (entry_first); fall back to entry_last if the
  // first isn't a chat-rendered entry (e.g. a user message before
  // we plumbed user-message anchoring, or a non-anchored kind).
  const focusedSegment = useMemo(() => {
    if (!focusedSegmentId || !spineSegments) return null;
    return spineSegments.find((s) => s.id === focusedSegmentId) ?? null;
  }, [focusedSegmentId, spineSegments]);

  const onSegmentJump = useCallback(() => {
    if (!focusedSegment) return;
    const ids = [focusedSegment.entry_first, focusedSegment.entry_last];
    for (const id of ids) {
      if (!id) continue;
      const el = document.querySelector<HTMLElement>(
        `[data-entry-id="${id}"]`,
      );
      if (el) {
        el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        return;
      }
    }
  }, [focusedSegment]);

  // Issue #6: keyboard nav for the spine. Esc dismisses the focused
  // sidebar; ↑/↓ moves focus between segments in chat order. Bound
  // globally on window so the user doesn't have to focus the rail
  // first, but skipped when typing into an input / textarea / any
  // contenteditable so the MessageInput stays usable.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      if (t) {
        const tag = t.tagName;
        if (
          tag === 'INPUT' ||
          tag === 'TEXTAREA' ||
          tag === 'SELECT' ||
          (t.isContentEditable ?? false)
        ) {
          return;
        }
      }
      if (e.key === 'Escape' && focusedSegmentId) {
        e.preventDefault();
        // Same path the sidebar's × button uses.
        onSegmentClose();
        return;
      }
      if (
        (e.key === 'ArrowDown' || e.key === 'ArrowUp') &&
        spineSegments &&
        spineSegments.length > 0
      ) {
        e.preventDefault();
        const cur = focusedSegmentId
          ? spineSegments.findIndex((s) => s.id === focusedSegmentId)
          : -1;
        const next =
          e.key === 'ArrowDown'
            ? Math.min(spineSegments.length - 1, cur < 0 ? 0 : cur + 1)
            : Math.max(0, cur < 0 ? spineSegments.length - 1 : cur - 1);
        const target = spineSegments[next];
        if (!target) return;
        onSegmentFocus(target.id);
        // Auto-scroll the chat so the new focus lands in view —
        // otherwise keyboard nav feels blind.
        for (const id of [target.entry_first, target.entry_last]) {
          if (!id) continue;
          const el = document.querySelector<HTMLElement>(
            `[data-entry-id="${id}"]`,
          );
          if (el) {
            el.scrollIntoView({ behavior: 'smooth', block: 'start' });
            return;
          }
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [focusedSegmentId, spineSegments, onSegmentClose, onSegmentFocus]);

  // Workspace card actions (Issue #5).
  const onWorkspaceInvalidate = useCallback(() => {
    // Bump the reload key so the drawer refetches on its next open.
    setWorkspaceReloadKey((k) => k + 1);
  }, []);

  const onWorkspaceOpenSegment = useCallback(
    (segmentId: string) => {
      setWorkspaceOpen(false);
      // Re-uses the spine focus path so the SpineSidebar takes over
      // with the same context the user committed from.
      onSegmentFocus(segmentId);
    },
    [onSegmentFocus],
  );

  const onWikiClose = useCallback(() => {
    if (!wikiSlot) return;
    const w = widgets[wikiSlot];
    if (!w) return;
    void postWidgetEvent(sessionId, {
      widget_id: w.id,
      surface_id: w.surface_id,
      action: 'close_wiki',
      data: {},
    }).catch(() => {
      /* surfaced via stream-error path on next event */
    });
  }, [sessionId, widgets, wikiSlot]);

  const onWorkspaceJump = useCallback(
    (entryFirst: string, entryLast: string) => {
      setWorkspaceOpen(false);
      for (const id of [entryFirst, entryLast]) {
        if (!id) continue;
        const el = document.querySelector<HTMLElement>(
          `[data-entry-id="${id}"]`,
        );
        if (el) {
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
          return;
        }
      }
    },
    [],
  );

  return (
    <div className={`app app-${layout}`}>
      <SessionSidebar
        userId={userId}
        currentSessionId={sessionId}
        displayName={displayName}
        onSetDisplayName={setDisplayName}
        onSwitchSession={switchSession}
      />
      <header className="app-header">
        <BrandedTitle />
        {streamError && <span className="app-error">{streamError}</span>}
        <button
          type="button"
          className="app-workspace-btn"
          onClick={() => setWorkspaceOpen(true)}
          title="Open workspace — segments you've saved"
        >
          <span aria-hidden>★</span>
          <span>Workspace</span>
        </button>
        <button
          type="button"
          className="app-history-btn"
          onClick={() => setHistoryOpen(true)}
          title="Widget history — restore a previously emitted widget"
        >
          <span aria-hidden>↺</span>
          <span>History</span>
        </button>
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
          highlightTerms={highlightTerms}
          highlightRange={highlightRange}
        />
        {spineSegments && (
          <SpinePane
            segments={spineSegments}
            focusedSegmentId={focusedSegmentId}
            onSegmentFocus={onSegmentFocus}
          />
        )}
        <DragSplitter storageKey={`app.chatW.${layout}`} />
        <div className="canvas-col">
          {focusedSegmentId && (
            <SpineSidebar
              sessionId={sessionId}
              segmentId={focusedSegmentId}
              onAction={(action) =>
                onSegmentAction(focusedSegmentId, action)
              }
              onJump={onSegmentJump}
              onClose={onSegmentClose}
              onWorkspaceInvalidate={onWorkspaceInvalidate}
              onDetailLoaded={onSegmentDetailLoaded}
              onSelectEntity={onSelectEntity}
              selectedEntityId={selectedEntity?.entity_id ?? null}
            />
          )}
          <CanvasPane sessionId={sessionId} />
        </div>
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
      <WorkspaceDrawer
        sessionId={sessionId}
        open={workspaceOpen}
        reloadKey={workspaceReloadKey}
        onClose={() => setWorkspaceOpen(false)}
        onOpen={onWorkspaceOpenSegment}
        onJump={onWorkspaceJump}
      />
      <WidgetHistoryDrawer
        sessionId={sessionId}
        open={historyOpen}
        onClose={() => setHistoryOpen(false)}
      />
      <WikiPane sessionId={sessionId} onClose={onWikiClose} />
    </div>
  );
}

/**
 * Recover the live-built `messageFollowUps` anchoring map from
 * timestamps. Called on session refresh once the transcript +
 * STATE_SNAPSHOT have landed, and again on each StateStore change
 * during the live session (the live `widget-emitted` reducer path
 * also maintains the map; rebuilding converges to the same result,
 * so the duplication is a no-op).
 *
 * Anchoring rule: each follow_up widget anchors to the *next*
 * assistant message that arrived after the widget on the wire. This
 * matches the live path's `pendingFollowUps → drain on next
 * assistant-chunk` sequence: gw-loop emits the widget during the
 * Python iteration (early in the turn), then the FINAL prose lands
 * as an `AssistantNarration` row after the iteration returns, so the
 * narration's timestamp is always greater than the widget's.
 *
 * Skips widgets without `created_at` and messages without
 * `createdAt` — both fields are absent on locally-appended
 * placeholders that pre-date a server roundtrip, and there's nothing
 * meaningful to anchor in those cases.
 */
function rebuildMessageFollowUps(
  messages: Message[],
  widgets: Record<string, Widget>,
  widgetOrder: string[],
): Record<string, string[]> {
  const result: Record<string, string[]> = {};
  const assistantAt = messages
    .filter((m) => m.role === 'assistant' && m.createdAt)
    .map((m) => ({ id: m.id, t: Date.parse(m.createdAt!) }))
    .filter((m) => !Number.isNaN(m.t))
    .sort((a, b) => a.t - b.t);
  if (assistantAt.length === 0) return result;
  for (const widgetId of widgetOrder) {
    const w = widgets[widgetId];
    if (!w?.follow_up) continue;
    if (!w.created_at) continue;
    const wt = Date.parse(w.created_at);
    if (Number.isNaN(wt)) continue;
    const anchor = assistantAt.find((m) => m.t >= wt);
    if (!anchor) continue;
    (result[anchor.id] ??= []).push(widgetId);
  }
  return result;
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
