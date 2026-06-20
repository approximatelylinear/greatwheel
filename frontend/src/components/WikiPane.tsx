import { useEffect } from 'react';
import { useStateValue } from '@json-render/react';
import type { Widget } from '../types';
import { WidgetRenderer } from './WidgetRenderer';
import { SlotNav } from './SlotNav';

interface Props {
  sessionId: string;
  onClose: () => void;
}

/**
 * Big right-side drawer that hosts the widget pinned to `wikiSlot` —
 * either a single-source `KbDocWiki` or a multi-source `KbClusterWiki`.
 * Reads from json-render state, no props.
 *
 * Closing the drawer fires `onClose` so App.tsx can post a
 * `close_wiki` widget event back to the server, which clears
 * `wiki_slot` on the surface.
 */
export function WikiPane({ sessionId, onClose }: Props) {
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const wikiId = useStateValue<string | null>('/wikiSlot') ?? null;
  const widget = wikiId ? widgets[wikiId] ?? null : null;

  // Esc closes the drawer.
  useEffect(() => {
    if (!widget) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [widget, onClose]);

  if (!widget) return null;
  return (
    <>
      <div className="kb-wiki-backdrop" onClick={onClose} />
      <div className="kb-wiki-drawer" role="dialog" aria-label="Wiki">
        <SlotNav
          slot="wiki"
          sessionId={sessionId}
          surfaceId={widget.surface_id ?? null}
          className="slot-nav-wiki-overlay"
        />
        <WidgetRenderer widget={widget} />
      </div>
    </>
  );
}
