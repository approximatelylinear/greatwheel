import { defineRegistry } from '@json-render/react';
import { spikeCatalog } from './catalog';
import { EntityCloudWidget } from '../widgets/EntityCloudWidget';

function formatCell(v: unknown): string {
  if (v == null) return '—';
  if (typeof v === 'number') {
    return Number.isInteger(v) ? String(v) : v.toLocaleString();
  }
  if (typeof v === 'string') return v;
  return JSON.stringify(v);
}

// Reuse the same CSS classes as A2uiWidget so the spike visually
// matches the existing demo. The structural wiring (emit, children)
// is what we're testing here, not the look.
const built = defineRegistry(spikeCatalog, {
  components: {
    Column: ({ children }) => <div className="a2ui-column">{children}</div>,
    Row: ({ children }) => <div className="a2ui-row">{children}</div>,
    Text: ({ props }) => <span className="a2ui-text">{props.text}</span>,
    Heading: ({ props }) => {
      const lvl = props.level ?? 1;
      const cls = `a2ui-heading a2ui-heading-l${lvl}`;
      if (lvl === 2) return <h2 className={cls}>{props.text}</h2>;
      if (lvl === 3) return <h3 className={cls}>{props.text}</h3>;
      return <h1 className={cls}>{props.text}</h1>;
    },
    Link: ({ props }) => (
      <a
        className="a2ui-link"
        href={props.url}
        target="_blank"
        rel="noopener noreferrer"
      >
        {props.label ?? props.url}
      </a>
    ),
    Button: ({ props, emit }) => (
      <button
        type="button"
        className={`a2ui-button${props.pressed ? ' pressed' : ''}`}
        disabled={props.disabled}
        onClick={() => emit('press')}
      >
        {props.label}
      </button>
    ),
    Card: ({ props, emit }) => (
      <button
        type="button"
        className={`a2ui-card${props.pressed ? ' pressed' : ''}`}
        disabled={props.disabled}
        onClick={() => emit('press')}
      >
        <div className="a2ui-card-title">{props.title}</div>
        {props.subtitle && (
          <div className="a2ui-card-sub">{props.subtitle}</div>
        )}
      </button>
    ),
    DataTable: ({ props, emit }) => (
      <div className="a2ui-table-wrap">
        <table className="a2ui-table">
          <thead>
            <tr>
              {props.columns.map((c, i) => (
                <th key={`${i}-${c}`}>{c}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {props.rows.map((row, i) => (
              <tr
                key={i}
                className="a2ui-table-row"
                onClick={() => {
                  // The translator wires `onSelect` via `on.select`
                  // ActionBinding rather than the default `press`
                  // event. See translate.ts.
                  emit(`row:${i}`);
                }}
              >
                {row.map((cell, j) => (
                  <td key={j}>{formatCell(cell)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
        {props.truncated && (
          <div className="a2ui-table-truncated">… more rows truncated</div>
        )}
      </div>
    ),
    QueryCard: ({ props }) => (
      <div className={`a2ui-query-card${props.error ? ' error' : ''}`}>
        {props.summary && (
          <div className="a2ui-query-summary">{props.summary}</div>
        )}
        <pre className="a2ui-query-sql">{props.sql}</pre>
        {props.error && (
          <pre className="a2ui-query-error">{props.error}</pre>
        )}
      </div>
    ),
    EntityCloud: ({ props, emit }) => (
      <EntityCloudWidget
        points={props.points}
        clusters={props.clusters ?? null}
        highlight={props.highlight ?? null}
        onPointClick={(id) => {
          // Per-point ActionBinding registered by the translator;
          // names match `point:<id>` so each click routes through
          // the catalog's `interact` action with the paper id.
          emit(`point:${id}`);
        }}
      />
    ),
  },
  actions: {
    interact: async (params) => {
      // eslint-disable-next-line no-console
      console.log('[spike-jr] interact', params);
    },
  },
});

export const registry = built.registry;

// Flat handler map that JSONUIProvider understands directly. The
// spike doesn't use json-render state, so we hand null state to
// executeAction — params are passed through to our handler above.
export const handlers: Record<
  string,
  (params: Record<string, unknown>) => Promise<unknown> | unknown
> = {
  interact: (params) =>
    built.executeAction('interact', params, () => ({}), {}),
};
