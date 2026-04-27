import { useEffect, useRef, useState } from 'react';

interface Props {
  /** localStorage key for persisting the chat-pane width. Pass a
   *  layout-scoped key (e.g. `app.chatW.chat-primary`) so chat- and
   *  canvas-primary demos remember their own preferred widths. */
  storageKey: string;
  /** Min width in px for either pane. */
  min?: number;
}

/**
 * Vertical drag splitter that sits in the middle column of `.app-main`.
 * Drag updates `--chat-w` (a px value) on the parent `.app-main`
 * element; CSS uses that variable to set `grid-template-columns`.
 * The pane's preferred width persists to localStorage and is
 * reapplied on mount.
 *
 * The splitter does NOT manage a CSS-var fallback — when the user
 * has never dragged, the layout default kicks in via the
 * `.app-main-{layout}` rule. Once they drag, we override.
 */
export function DragSplitter({ storageKey, min = 240 }: Props) {
  const ref = useRef<HTMLDivElement | null>(null);
  const [dragging, setDragging] = useState(false);

  // Restore the persisted width on mount. Done as an effect because
  // the parent `.app-main` may not exist yet during the initial
  // render pass when this component is first instantiated.
  useEffect(() => {
    const saved = localStorage.getItem(storageKey);
    if (!saved) return;
    const main = ref.current?.parentElement;
    if (main) main.style.setProperty('--chat-w', saved);
  }, [storageKey]);

  const onPointerDown = (e: React.PointerEvent) => {
    e.preventDefault();
    const main = ref.current?.parentElement;
    if (!main) return;
    const mainRect = main.getBoundingClientRect();
    // Use the splitter's current x as the drag origin so we don't
    // jump when the user grabs anywhere within the hit zone. The
    // splitter's pixel-x = chat width.
    const splitterX = ref.current!.getBoundingClientRect().left;
    const offset = e.clientX - splitterX;
    setDragging(true);
    document.body.classList.add('app-resizing');

    const onMove = (ev: PointerEvent) => {
      const wantedX = ev.clientX - offset;
      const relX = wantedX - mainRect.left;
      const max = mainRect.width - min - 6; // 6px = splitter width
      const clamped = Math.max(min, Math.min(max, relX));
      main.style.setProperty('--chat-w', `${clamped}px`);
    };
    const onUp = () => {
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      window.removeEventListener('pointercancel', onUp);
      document.body.classList.remove('app-resizing');
      setDragging(false);
      // Persist the inline value so reload restores the same layout.
      const cur = main.style.getPropertyValue('--chat-w');
      if (cur) localStorage.setItem(storageKey, cur);
    };
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    window.addEventListener('pointercancel', onUp);
  };

  const onDoubleClick = () => {
    // Double-click resets to the layout default.
    const main = ref.current?.parentElement;
    if (!main) return;
    main.style.removeProperty('--chat-w');
    localStorage.removeItem(storageKey);
  };

  return (
    <div
      ref={ref}
      className={`app-splitter${dragging ? ' dragging' : ''}`}
      role="separator"
      aria-orientation="vertical"
      onPointerDown={onPointerDown}
      onDoubleClick={onDoubleClick}
      title="Drag to resize · double-click to reset"
    />
  );
}
