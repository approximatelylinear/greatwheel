import type { UiSurfaceSnapshot, WidgetEvent } from '../types';

const BASE = (import.meta.env.VITE_API_BASE as string | undefined) ?? 'http://127.0.0.1:8787';

export async function postMessage(sessionId: string, content: string): Promise<void> {
  const r = await fetch(`${BASE}/sessions/${sessionId}/messages`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content }),
  });
  if (!r.ok) throw new Error(`postMessage failed: ${r.status} ${await r.text()}`);
}

export async function postWidgetEvent(sessionId: string, event: WidgetEvent): Promise<void> {
  const r = await fetch(`${BASE}/sessions/${sessionId}/widget-events`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(event),
  });
  if (!r.ok) throw new Error(`postWidgetEvent failed: ${r.status} ${await r.text()}`);
}

export async function fetchSurface(sessionId: string): Promise<UiSurfaceSnapshot> {
  const r = await fetch(`${BASE}/sessions/${sessionId}/surface`);
  if (!r.ok) throw new Error(`fetchSurface failed: ${r.status}`);
  return r.json();
}

export function streamUrl(sessionId: string): string {
  return `${BASE}/sessions/${sessionId}/stream`;
}
