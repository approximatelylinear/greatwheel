import { useStateValue } from '@json-render/react';
import { postWidgetEvent } from '../api/client';

/**
 * Browser-style back/forward chevrons for one widget slot.
 * Renders ‹ › buttons that fire `action: "nav_slot"` widget events;
 * the AG-UI adapter short-circuits to the corresponding `nav_slot`
 * call on the surface store, which moves the slot's cursor without
 * touching pin history. See `docs/design-slot-nav.md`.
 *
 * Hidden entirely when neither direction is enabled — so a freshly
 * pinned slot (one entry) shows no chevrons, and consumers don't
 * have to wrap the mount in a conditional.
 */
interface Props {
  slot: 'canvas' | 'aux' | 'wiki';
  sessionId: string;
  surfaceId: string | null;
  /** Extra class on the wrapper, used by hosts to position the
   *  chevrons (top-right overlay, header-bar inline, etc.). */
  className?: string;
}

const NAV_PATH: Record<Props['slot'], string> = {
  canvas: '/canvasNav',
  aux: '/canvasAuxNav',
  wiki: '/wikiNav',
};

const NIL_UUID = '00000000-0000-0000-0000-000000000000';

export function SlotNav({ slot, sessionId, surfaceId, className }: Props) {
  const nav = useStateValue<{ can_back: boolean; can_forward: boolean }>(
    NAV_PATH[slot],
  );
  const canBack = nav?.can_back ?? false;
  const canForward = nav?.can_forward ?? false;
  if (!canBack && !canForward) return null;

  const fire = (direction: 'back' | 'forward') => {
    if (!surfaceId) return;
    void postWidgetEvent(sessionId, {
      widget_id: NIL_UUID,
      surface_id: surfaceId,
      action: 'nav_slot',
      data: { slot, direction },
    }).catch(() => {
      /* surfaced via stream-error path on next event */
    });
  };

  return (
    <div className={`slot-nav slot-nav-${slot}${className ? ` ${className}` : ''}`}>
      <button
        type="button"
        className="slot-nav-btn"
        disabled={!canBack || !surfaceId}
        onClick={() => fire('back')}
        title="Previous"
        aria-label="Back"
      >
        ‹
      </button>
      <button
        type="button"
        className="slot-nav-btn"
        disabled={!canForward || !surfaceId}
        onClick={() => fire('forward')}
        title="Next"
        aria-label="Forward"
      >
        ›
      </button>
    </div>
  );
}
