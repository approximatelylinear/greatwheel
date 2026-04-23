import { useState } from 'react';
import type { CodeTrace } from '../types';

interface Props {
  traces: CodeTrace[];
}

/**
 * Collapsible strip that surfaces every code block the agent ran,
 * along with stdout and any host-call error. Enable via `?debug=1`.
 */
export function DebugPane({ traces }: Props) {
  const [collapsed, setCollapsed] = useState(false);
  return (
    <aside className={`debug-pane ${collapsed ? 'collapsed' : ''}`}>
      <button
        type="button"
        className="debug-toggle"
        onClick={() => setCollapsed((c) => !c)}
      >
        {collapsed ? '▲' : '▼'} agent traces ({traces.length})
      </button>
      {!collapsed && (
        <div className="debug-list">
          {traces.length === 0 && (
            <div className="debug-empty">
              Nothing yet. The agent's Python code + stdout will appear here.
            </div>
          )}
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
      )}
    </aside>
  );
}
