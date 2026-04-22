import type { AgUiEvent } from '../types';
import { streamUrl } from './client';

/**
 * Subscribe to a session's AG-UI event stream. Returns a cleanup
 * function that closes the connection.
 */
export function openStream(
  sessionId: string,
  onEvent: (ev: AgUiEvent) => void,
  onError?: (e: unknown) => void,
): () => void {
  const es = new EventSource(streamUrl(sessionId));
  es.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as AgUiEvent);
    } catch (e) {
      onError?.(e);
    }
  };
  es.onerror = (e) => onError?.(e);
  return () => es.close();
}
