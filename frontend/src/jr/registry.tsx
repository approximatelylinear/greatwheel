import { defineRegistry } from '@json-render/react';
import { spikeCatalog } from './catalog';

// Reuse the same CSS classes as A2uiWidget so the spike visually
// matches the existing demo. The structural wiring (emit, children)
// is what we're testing here, not the look.
const built = defineRegistry(spikeCatalog, {
  components: {
    Column: ({ children }) => <div className="a2ui-column">{children}</div>,
    Row: ({ children }) => <div className="a2ui-row">{children}</div>,
    Text: ({ props }) => <span className="a2ui-text">{props.text}</span>,
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
