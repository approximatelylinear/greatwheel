import type { AgUiEvent } from '../types';
import { streamUrl } from './client';

/**
 * Subscribe to a session's AG-UI event stream. Returns a cleanup
 * function that closes the connection.
 *
 * `onStatus` reports a human-readable error string, or `null` to clear
 * it. EventSource auto-reconnects, so a dropped connection first lands
 * in CONNECTING (transient — stay quiet) and only reaches CLOSED when
 * the browser has given up; we surface the banner only in that case.
 * A successful (re)connect fires `onopen`, which clears the banner.
 */
export function openStream(
  sessionId: string,
  onEvent: (ev: AgUiEvent) => void,
  onStatus?: (error: string | null) => void,
): () => void {
  const es = new EventSource(streamUrl(sessionId));
  es.onopen = () => onStatus?.(null);
  es.onmessage = (msg) => {
    try {
      onEvent(JSON.parse(msg.data) as AgUiEvent);
    } catch (e) {
      onStatus?.(
        `malformed event: ${e instanceof Error ? e.message : String(e)}`,
      );
    }
  };
  es.onerror = () => {
    // CONNECTING (0) means the browser is retrying — don't alarm the
    // user over a normal blip. CLOSED (2) is a real, terminal failure.
    if (es.readyState === EventSource.CLOSED) {
      onStatus?.('Event stream disconnected — is the server running?');
    }
  };
  return () => es.close();
}
