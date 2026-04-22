import type { Widget, WidgetEvent } from '../types';
import { A2uiWidget } from '../widgets/A2uiWidget';
import { McpUiWidget } from '../widgets/McpUiWidget';

interface Props {
  widget: Widget;
  onInteract: (ev: WidgetEvent) => void;
}

export function WidgetRenderer({ widget, onInteract }: Props) {
  if (widget.kind === 'A2ui') return <A2uiWidget widget={widget} onInteract={onInteract} />;
  if (widget.kind === 'McpUi') return <McpUiWidget widget={widget} />;
  // Custom kinds fall through — host app would plug in its own renderer.
  return (
    <div className="widget-error">
      Unsupported widget kind: <code>{JSON.stringify(widget.kind)}</code>
    </div>
  );
}
