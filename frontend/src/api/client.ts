import type { WidgetEvent } from '../types';

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

export function streamUrl(sessionId: string): string {
  return `${BASE}/sessions/${sessionId}/stream`;
}

/**
 * Fetch the spine sidebar's segment detail. The literature_assistant
 * binary mounts this endpoint when the KB is wired up; if a session
 * runs without KB the request 404s and the sidebar shows an empty
 * state. Throws on network errors and non-2xx HTTP responses.
 */
export async function fetchSegmentDetail(
  sessionId: string,
  segmentId: string,
  signal?: AbortSignal,
): Promise<SegmentDetail> {
  const r = await fetch(
    `${BASE}/sessions/${sessionId}/segments/${segmentId}`,
    { signal },
  );
  if (!r.ok) {
    throw new Error(`fetchSegmentDetail ${r.status}: ${await r.text()}`);
  }
  return (await r.json()) as SegmentDetail;
}

// ─── Spine sidebar shapes (mirror gw_loop::spine::query) ─────────

export interface SegmentDetail {
  segment: SegmentSummary;
  entries: EntrySummary[];
  entities: EntityCard[];
  relations: RelationRow[];
}

export interface SegmentSummary {
  id: string;
  session_id: string;
  label: string;
  kind: string;
  entry_first: string;
  entry_last: string;
  entity_count: number;
  summary: string | null;
  created_at: string;
}

export interface EntrySummary {
  id: string;
  role: string;
  created_at: string;
  preview: string;
  preview_truncated: boolean;
}

export interface EntityCard {
  entity_id: string;
  label: string;
  slug: string;
  kind: string;
  mentions_in_segment: number;
  global_mentions: number;
  aliases: string[];
  summary: string | null;
}

export interface RelationRow {
  id: string;
  entry_id: string;
  subject_id: string;
  subject_label: string;
  object_id: string;
  object_label: string;
  predicate: string;
  directed: boolean;
  surface: string;
  confidence: number;
}
