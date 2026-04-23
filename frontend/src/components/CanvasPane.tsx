import type { Widget } from '../types';
import { WidgetRenderer } from './WidgetRenderer';

interface Props {
  widget: Widget | null;
  auxWidget: Widget | null;
  pressedId: string | null;
  auxPressedId: string | null;
}

export function CanvasPane({
  widget,
  auxWidget,
  pressedId,
  auxPressedId,
}: Props) {
  return (
    <aside className="canvas-pane">
      <div className="canvas-header">Canvas</div>
      {widget ? (
        <WidgetRenderer widget={widget} pressedId={pressedId} />
      ) : (
        <div className="canvas-empty">No widget pinned.</div>
      )}
      {auxWidget && (
        <div className="canvas-aux">
          <WidgetRenderer widget={auxWidget} pressedId={auxPressedId} />
        </div>
      )}
    </aside>
  );
}
