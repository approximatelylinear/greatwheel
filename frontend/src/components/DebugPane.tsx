import { useState } from 'react';
import type { CodeTrace, ToolCall } from '../types';
import type { SpineDebugEvent } from '../store/session';

interface Props {
  traces: CodeTrace[];
  toolCalls: ToolCall[];
  spineEvents: SpineDebugEvent[];
}

type Tab = 'tools' | 'code' | 'spine';

/**
 * Collapsible strip that surfaces every agent-side observation:
 * host function (tool) calls, every Python code block the sandbox
 * executed, and the spine pipeline's per-turn extractions and
 * re-segments. Enable via `?debug=1`.
 */
export function DebugPane({ traces, toolCalls, spineEvents }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [tab, setTab] = useState<Tab>('tools');
  return (
    <aside className={`debug-pane ${collapsed ? 'collapsed' : ''}`}>
      <div className="debug-toolbar">
        <button
          type="button"
          className="debug-toggle"
          onClick={() => setCollapsed((c) => !c)}
        >
          {collapsed ? '▲' : '▼'} agent traces
        </button>
        {!collapsed && (
          <div className="debug-tabs">
            <button
              type="button"
              className={`debug-tab ${tab === 'tools' ? 'active' : ''}`}
              onClick={() => setTab('tools')}
            >
              tool calls ({toolCalls.length})
            </button>
            <button
              type="button"
              className={`debug-tab ${tab === 'code' ? 'active' : ''}`}
              onClick={() => setTab('code')}
            >
              code ({traces.length})
            </button>
            <button
              type="button"
              className={`debug-tab ${tab === 'spine' ? 'active' : ''}`}
              onClick={() => setTab('spine')}
            >
              spine ({spineEvents.length})
            </button>
          </div>
        )}
      </div>
      {!collapsed && tab === 'tools' && <ToolCallList calls={toolCalls} />}
      {!collapsed && tab === 'code' && <CodeTraceList traces={traces} />}
      {!collapsed && tab === 'spine' && <SpineEventList events={spineEvents} />}
    </aside>
  );
}

function ToolCallList({ calls }: { calls: ToolCall[] }) {
  if (calls.length === 0) {
    return (
      <div className="debug-list">
        <div className="debug-empty">
          Nothing yet. Host function calls the agent dispatches (emit_widget,
          pin_to_canvas, highlight_button, …) will appear here.
        </div>
      </div>
    );
  }
  return (
    <div className="debug-list">
      {calls
        .slice()
        .reverse()
        .map((c) => {
          const elapsed =
            c.completedAt != null ? c.completedAt - c.startedAt : null;
          return (
            <article
              key={c.id}
              className={`debug-trace debug-toolcall ${c.status}`}
            >
              <header className="debug-trace-head">
                <span className="debug-trace-time">
                  {new Date(c.startedAt).toLocaleTimeString()}
                </span>
                <span className="debug-toolcall-name">{c.name}</span>
                {c.status === 'running' && (
                  <span className="debug-trace-badge">running…</span>
                )}
                {c.status === 'done' && elapsed != null && (
                  <span className="debug-trace-badge">{elapsed}ms</span>
                )}
                {c.status === 'error' && (
                  <span className="debug-trace-badge err">error</span>
                )}
              </header>
              {c.args !== undefined && (
                <pre className="debug-code">{stringify(c.args)}</pre>
              )}
              {c.status === 'done' && c.result !== undefined && (
                <pre className="debug-stdout">→ {stringify(c.result)}</pre>
              )}
              {c.error && <pre className="debug-error">{c.error}</pre>}
            </article>
          );
        })}
    </div>
  );
}

function CodeTraceList({ traces }: { traces: CodeTrace[] }) {
  if (traces.length === 0) {
    return (
      <div className="debug-list">
        <div className="debug-empty">
          Nothing yet. The agent's Python code + stdout will appear here.
        </div>
      </div>
    );
  }
  return (
    <div className="debug-list">
      {traces
        .slice()
        .reverse()
        .map((t) => (
          <article key={t.id} className={`debug-trace ${t.error ? 'error' : ''}`}>
            <header className="debug-trace-head">
              <span className="debug-trace-time">
                {new Date(t.at).toLocaleTimeString()}
              </span>
              {t.is_final && <span className="debug-trace-badge">FINAL</span>}
              {t.error && <span className="debug-trace-badge err">error</span>}
            </header>
            <pre className="debug-code">{t.code}</pre>
            {t.stdout && <pre className="debug-stdout">{t.stdout}</pre>}
            {t.error && <pre className="debug-error">{t.error}</pre>}
          </article>
        ))}
    </div>
  );
}

function SpineEventList({ events }: { events: SpineDebugEvent[] }) {
  if (events.length === 0) {
    return (
      <div className="debug-list">
        <div className="debug-empty">
          Nothing yet. Spine extractions and re-segments will appear here as
          you talk to the agent — entry-extracted events show how many
          entities + relations the LLM pulled out of each message;
          segments-updated events show the current rail snapshot.
        </div>
      </div>
    );
  }
  return (
    <div className="debug-list">
      {events
        .slice()
        .reverse()
        .map((e) =>
          e.kind === 'entry-extracted' ? (
            <article key={e.id} className="debug-trace">
              <header className="debug-trace-head">
                <span className="debug-trace-time">
                  {new Date(e.at).toLocaleTimeString()}
                </span>
                <span className="debug-toolcall-name">entry-extracted</span>
                <span className="debug-trace-badge">
                  {e.entity_count} ent · {e.relation_count} rel
                </span>
              </header>
              <pre className="debug-stdout">entry_id: {e.entry_id}</pre>
            </article>
          ) : (
            <article key={e.id} className="debug-trace">
              <header className="debug-trace-head">
                <span className="debug-trace-time">
                  {new Date(e.at).toLocaleTimeString()}
                </span>
                <span className="debug-toolcall-name">segments-updated</span>
                <span className="debug-trace-badge">
                  {e.segments.length} segment{e.segments.length === 1 ? '' : 's'}
                </span>
              </header>
              {e.segments.length > 0 ? (
                <pre className="debug-stdout">
                  {e.segments
                    .map(
                      (s) =>
                        `· ${s.label} (${s.kind}, ${s.entity_count} ${
                          s.entity_count === 1 ? 'entity' : 'entities'
                        })`,
                    )
                    .join('\n')}
                </pre>
              ) : (
                <pre className="debug-stdout">(empty)</pre>
              )}
            </article>
          ),
        )}
    </div>
  );
}

function stringify(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}
