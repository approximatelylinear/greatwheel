import { useMemo } from 'react';
import { Renderer } from '@json-render/react';
import type { Widget } from '../types';
import { registry } from '../jr/registry';
import { toJrSpec } from '../jr/translate';
import { McpUiWidget } from '../widgets/McpUiWidget';

interface Props {
  widget: Widget;
  pressedId: string | null;
}

/**
 * Thin per-widget wrapper: handles the lifecycle shell (terminal
 * banner, data-state attribute) and delegates the inline payload
 * tree to json-render's Renderer. Clicks flow up to the single
 * JSONUIProvider in App via the catalog's `interact` action.
 */
export function WidgetRenderer({ widget, pressedId }: Props) {
  const spec = useMemo(() => {
    if (widget.kind !== 'A2ui') return null;
    if (!('Inline' in widget.payload)) return null;
    return toJrSpec(widget.payload.Inline, widget.id, widget.surface_id, pressedId);
  }, [widget, pressedId]);

  if (widget.kind === 'McpUi') return <McpUiWidget widget={widget} />;
  if (widget.kind !== 'A2ui') {
    return (
      <div className="widget-error">
        Unsupported widget kind: <code>{JSON.stringify(widget.kind)}</code>
      </div>
    );
  }
  if (!spec) {
    return <div className="widget-error">A2UI widget missing inline payload</div>;
  }

  const terminal = widget.state !== 'Active';
  return (
    <div
      className={`a2ui-widget ${terminal ? 'terminal' : ''}`}
      data-state={widget.state}
    >
      {terminal && (
        <div className="widget-terminal-banner">
          <span className="checkmark" aria-hidden="true">✓</span>
          <span>{terminalLabel(widget.state)}</span>
        </div>
      )}
      <Renderer spec={spec} registry={registry} />
    </div>
  );
}

function terminalLabel(state: Widget['state']): string {
  switch (state) {
    case 'Resolved':
      return 'Answered';
    case 'Superseded':
      return 'Replaced';
    case 'Expired':
      return 'Expired';
    case 'Active':
      return 'Active';
  }
}
