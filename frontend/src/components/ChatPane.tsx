import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import type { Widget, WidgetEvent } from '../types';
import type { Message } from '../store/session';
import { WidgetRenderer } from './WidgetRenderer';

interface Props {
  messages: Message[];
  widgets: Record<string, Widget>;
  widgetOrder: string[];
  canvasSlot: string | null;
  running: boolean;
  pressedButtonIds: Record<string, string>;
  messageFollowUps: Record<string, string[]>;
  onInteract: (ev: WidgetEvent, buttonId: string) => void;
}

export function ChatPane({
  messages,
  widgets,
  widgetOrder,
  canvasSlot,
  running,
  pressedButtonIds,
  messageFollowUps,
  onInteract,
}: Props) {
  // Widgets anchored to a message should NOT also appear in the
  // scroll tail; collect their ids and exclude.
  const anchored = new Set<string>(
    Object.values(messageFollowUps).flat(),
  );
  const inlineIds = widgetOrder.filter(
    (id) => id !== canvasSlot && !anchored.has(id),
  );
  const showTyping = running;
  return (
    <div className="chat-pane">
      <div className="messages">
        {messages.map((m) => (
          <div key={m.id}>
            <div className={`message message-${m.role}`}>
              <div className="message-role">{m.role}</div>
              <div className="message-content">
                {m.role === 'assistant' ? (
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    rehypePlugins={[rehypeRaw]}
                  >
                    {pullQuote(normaliseNewlines(m.content))}
                  </ReactMarkdown>
                ) : (
                  m.content
                )}
              </div>
            </div>
            {m.role === 'assistant' && (messageFollowUps[m.id]?.length ?? 0) > 0 && (
              <div className="message-followups">
                {messageFollowUps[m.id]!.map((wid) => {
                  const w = widgets[wid];
                  if (!w) return null;
                  return (
                    <WidgetRenderer
                      key={wid}
                      widget={w}
                      pressedId={pressedButtonIds[w.id] ?? null}
                      onInteract={onInteract}
                    />
                  );
                })}
              </div>
            )}
          </div>
        ))}
        {showTyping && <TypingBubble />}
        {inlineIds.map((id) => {
          const w = widgets[id];
          if (!w) return null;
          return (
            <div key={id} className="widget-inline">
              <WidgetRenderer
                widget={w}
                pressedId={pressedButtonIds[w.id] ?? null}
                onInteract={onInteract}
              />
            </div>
          );
        })}
      </div>
    </div>
  );
}

function normaliseNewlines(text: string): string {
  return text.replace(/\\n/g, '\n').replace(/\\t/g, '\t');
}

/**
 * Wrap 'single-quoted' spans in a <q class="pull-quote"> tag so the
 * markdown renderer (with rehype-raw) can style them as literary pull-
 * quotes. Handles both ASCII straight quotes ('…') and Unicode curly
 * quotes (‘…’). We require whitespace / punctuation / start
 * before the opening quote and whitespace / punctuation / end after
 * the closing quote, so possessives and contractions (Victor's,
 * don't) don't accidentally match.
 */
function pullQuote(text: string): string {
  // Word boundaries: require whitespace / opening punct / em-dash /
  // start before the opening quote and whitespace / closing punct /
  // end after the closing one, so possessives and contractions don't
  // trigger matches. ≥3 chars between quotes also skips things like
  // "s" or "ll".
  const pre = `(^|[\\s(\\[—–-])`;
  const post = `(?=[\\s.,!?;:)\\]—–-]|$)`;
  const patterns: RegExp[] = [
    // ASCII single quotes 'like this'
    new RegExp(`${pre}'([^'\\n]{3,}?)'${post}`, 'g'),
    // Unicode curly single quotes ‘like this’
    new RegExp(`${pre}‘([^’\\n]{3,}?)’${post}`, 'g'),
    // ASCII double quotes "like this"
    new RegExp(`${pre}"([^"\\n]{3,}?)"${post}`, 'g'),
    // Unicode curly double quotes “like this”
    new RegExp(`${pre}“([^”\\n]{3,}?)”${post}`, 'g'),
  ];
  let out = text;
  for (const re of patterns) {
    out = out.replace(re, '$1<q class="pull-quote">$2</q>');
  }
  return out;
}

function TypingBubble() {
  return (
    <div className="message message-assistant message-typing">
      <div className="message-role">assistant</div>
      <div className="typing-dots" aria-label="thinking">
        <span />
        <span />
        <span />
      </div>
    </div>
  );
}
