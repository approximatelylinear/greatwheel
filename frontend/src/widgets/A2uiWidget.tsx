import type { Widget, WidgetEvent } from '../types';

interface Props {
  widget: Widget;
  onInteract: (ev: WidgetEvent) => void;
}

/**
 * Minimal A2UI-compatible subset: Column | Row | Text | Button.
 * Richer components (Slider, TextInput, Image, etc.) are left to a
 * proper json-render integration. This renderer is intentionally a
 * throwaway.
 */
export function A2uiWidget({ widget, onInteract }: Props) {
  if (!('Inline' in widget.payload)) {
    return <div className="widget-error">A2UI widget missing inline payload</div>;
  }
  const terminal = widget.state !== 'Active';
  return (
    <div className={`a2ui-widget ${terminal ? 'terminal' : ''}`} data-state={widget.state}>
      {terminal && (
        <div className="widget-terminal-banner">
          {widget.state}
          {widget.resolution != null && <>: <code>{JSON.stringify(widget.resolution)}</code></>}
        </div>
      )}
      <Component
        node={widget.payload.Inline}
        widget={widget}
        disabled={terminal}
        onInteract={onInteract}
      />
    </div>
  );
}

interface ComponentProps {
  node: unknown;
  widget: Widget;
  disabled: boolean;
  onInteract: (ev: WidgetEvent) => void;
}

function Component({ node, widget, disabled, onInteract }: ComponentProps) {
  if (!node || typeof node !== 'object') return <span>{String(node)}</span>;
  const n = node as Record<string, unknown>;
  switch (n.type) {
    case 'Column':
    case 'Row': {
      const children = Array.isArray(n.children) ? (n.children as unknown[]) : [];
      return (
        <div className={`a2ui-${String(n.type).toLowerCase()}`}>
          {children.map((c, i) => (
            <Component
              key={i}
              node={c}
              widget={widget}
              disabled={disabled}
              onInteract={onInteract}
            />
          ))}
        </div>
      );
    }
    case 'Text':
      return <span className="a2ui-text">{String(n.text ?? '')}</span>;
    case 'Button': {
      const id = String(n.id ?? 'button');
      const label = String(n.label ?? id);
      const action = String(n.action ?? 'click');
      const data = (n.data as unknown) ?? { id };
      return (
        <button
          className="a2ui-button"
          disabled={disabled}
          onClick={() =>
            onInteract({
              widget_id: widget.id,
              surface_id: widget.surface_id,
              action,
              data,
            })
          }
        >
          {label}
        </button>
      );
    }
    default:
      return (
        <pre className="a2ui-unknown">{JSON.stringify(n, null, 2)}</pre>
      );
  }
}
