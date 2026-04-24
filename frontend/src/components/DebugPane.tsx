import { useState } from 'react';
import type { CodeTrace, ToolCall } from '../types';

interface Props {
  traces: CodeTrace[];
  toolCalls: ToolCall[];
}

/**
 * Collapsible strip that surfaces every agent-side observation:
 * host function (tool) calls and every Python code block the
 * sandbox executed. Enable via `?debug=1`.
 */
export function DebugPane({ traces, toolCalls }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  const [tab, setTab] = useState<'tools' | 'code'>('tools');
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
          </div>
        )}
      </div>
      {!collapsed && tab === 'tools' && <ToolCallList calls={toolCalls} />}
      {!collapsed && tab === 'code' && <CodeTraceList traces={traces} />}
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

function stringify(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}
