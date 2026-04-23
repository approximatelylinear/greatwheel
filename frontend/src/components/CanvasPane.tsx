import { useStateValue } from '@json-render/react';
import type { Widget } from '../types';
import { WidgetRenderer } from './WidgetRenderer';

/**
 * Canvas reads directly from json-render state — `canvasSlot` and
 * `canvasAuxSlot` are JSON-Pointer bindings populated by server
 * STATE_DELTA patches (pin / pin_aux). No props needed.
 */
export function CanvasPane() {
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const primaryId = useStateValue<string | null>('/canvasSlot') ?? null;
  const auxId = useStateValue<string | null>('/canvasAuxSlot') ?? null;
  const primary = primaryId ? widgets[primaryId] ?? null : null;
  const aux = auxId ? widgets[auxId] ?? null : null;
  return (
    <aside className="canvas-pane">
      <div className="canvas-header">Canvas</div>
      {primary ? (
        <WidgetRenderer widget={primary} />
      ) : (
        <div className="canvas-empty">No widget pinned.</div>
      )}
      {aux && (
        <div className="canvas-aux">
          <WidgetRenderer widget={aux} />
        </div>
      )}
    </aside>
  );
}
