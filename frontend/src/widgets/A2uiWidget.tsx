import type { Widget, WidgetEvent } from '../types';

interface Props {
  widget: Widget;
  pressedId: string | null;
  onInteract: (ev: WidgetEvent, buttonId: string) => void;
}

/**
 * Minimal A2UI-compatible subset: Column | Row | Text | Button.
 * `pressedId` is sourced from the session store so both user clicks
 * and agent-driven inferences (e.g. the agent calling `get_section`
 * in response to a free-text question) can highlight the same button.
 */
export function A2uiWidget({ widget, pressedId, onInteract }: Props) {
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
        pressedId={pressedId}
        onInteract={onInteract}
      />
    </div>
  );
}

interface ComponentProps {
  node: unknown;
  widget: Widget;
  disabled: boolean;
  pressedId: string | null;
  onInteract: (ev: WidgetEvent, buttonId: string) => void;
}

function Component({ node, widget, disabled, pressedId, onInteract }: ComponentProps) {
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
              pressedId={pressedId}
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
      const isPressed = pressedId === id;
      return (
        <button
          className={`a2ui-button${isPressed ? ' pressed' : ''}`}
          disabled={disabled}
          onClick={() =>
            onInteract(
              {
                widget_id: widget.id,
                surface_id: widget.surface_id,
                action,
                data,
              },
              id,
            )
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
