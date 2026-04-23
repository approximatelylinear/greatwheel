import type { Widget, WidgetEvent } from '../types';
import { WidgetRenderer } from './WidgetRenderer';

interface Props {
  widget: Widget | null;
  pressedId: string | null;
  onInteract: (ev: WidgetEvent, buttonId: string) => void;
}

export function CanvasPane({ widget, pressedId, onInteract }: Props) {
  return (
    <aside className="canvas-pane">
      <div className="canvas-header">Canvas</div>
      {widget ? (
        <WidgetRenderer widget={widget} pressedId={pressedId} onInteract={onInteract} />
      ) : (
        <div className="canvas-empty">No widget pinned.</div>
      )}
    </aside>
  );
}
