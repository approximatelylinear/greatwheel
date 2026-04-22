import type { Widget, WidgetEvent } from '../types';
import { WidgetRenderer } from './WidgetRenderer';

interface Props {
  widget: Widget | null;
  onInteract: (ev: WidgetEvent) => void;
}

export function CanvasPane({ widget, onInteract }: Props) {
  return (
    <aside className="canvas-pane">
      <div className="canvas-header">Canvas</div>
      {widget ? (
        <WidgetRenderer widget={widget} onInteract={onInteract} />
      ) : (
        <div className="canvas-empty">No widget pinned.</div>
      )}
    </aside>
  );
}
