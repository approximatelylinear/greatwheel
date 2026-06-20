import { useStateValue } from '@json-render/react';
import type { Widget } from '../types';
import { WidgetRenderer } from './WidgetRenderer';
import { SlotNav } from './SlotNav';

/**
 * Canvas reads directly from json-render state — `canvasSlot` and
 * `canvasAuxSlot` are JSON-Pointer bindings populated by server
 * STATE_DELTA patches (pin / pin_aux). `sessionId` is the only
 * required prop, needed by the SlotNav chevrons to post nav events.
 */
interface Props {
  sessionId: string;
}

export function CanvasPane({ sessionId }: Props) {
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const primaryId = useStateValue<string | null>('/canvasSlot') ?? null;
  const auxId = useStateValue<string | null>('/canvasAuxSlot') ?? null;
  const primary = primaryId ? widgets[primaryId] ?? null : null;
  const aux = auxId ? widgets[auxId] ?? null : null;
  return (
    <aside className="canvas-pane">
      <div className="canvas-header">Canvas</div>
      <SlotNav
        slot="canvas"
        sessionId={sessionId}
        surfaceId={primary?.surface_id ?? null}
      />
      {primary ? (
        <WidgetRenderer widget={primary} />
      ) : (
        <div className="canvas-empty">No widget pinned.</div>
      )}
      {aux && (
        <div className="canvas-aux">
          <SlotNav
            slot="aux"
            sessionId={sessionId}
            surfaceId={aux.surface_id ?? null}
          />
          <WidgetRenderer widget={aux} />
        </div>
      )}
    </aside>
  );
}
