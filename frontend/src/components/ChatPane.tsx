import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { useStateValue } from '@json-render/react';
import type { Widget } from '../types';
import type { Message } from '../store/session';
import { WidgetRenderer } from './WidgetRenderer';

interface Props {
  messages: Message[];
  running: boolean;
  messageFollowUps: Record<string, string[]>;
  onSuggest: (content: string) => void;
  /** Re-pin a paper by replaying the EntityCloud's point-click event.
   *  Called from a log-line click when the message matches the
   *  `Pinned · arxiv:<id> · <title>` shape. */
  onRepin: (arxivId: string) => void;
}

interface Welcome {
  heading: string;
  body: string;
  suggestions: string[];
}

const FALLBACK_WELCOME: Welcome = {
  heading: 'An agent reading <em>Frankenstein</em> with you.',
  body:
    "Ask about the novel in plain language. Pick a chapter from the list on the right and you'll get a summary grounded in the actual text — with follow-up questions and the characters who appear in that section ready to click.",
  suggestions: [
    'Summarize Chapter 5',
    'Who is Robert Walton?',
    'What themes run through the novel?',
  ],
};

export function ChatPane({
  messages,
  running,
  messageFollowUps,
  onSuggest,
  onRepin,
}: Props) {
  // Widget records, order, and pinned set now live in the json-render
  // StateStore (populated by STATE_SNAPSHOT + STATE_DELTA). Components
  // subscribe via useStateValue; the reducer only tracks chat state.
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const widgetOrder = useStateValue<string[]>('/widgetOrder') ?? [];
  const pinnedIds = useStateValue<Record<string, true>>('/pinnedIds') ?? {};

  // Widgets anchored to a message should NOT also appear in the
  // scroll tail; collect their ids and exclude.
  const anchored = new Set<string>(
    Object.values(messageFollowUps).flat(),
  );
  const inlineIds = widgetOrder.filter(
    (id) => !pinnedIds[id] && !anchored.has(id),
  );
  const showTyping = running;
  // Empty state: pre-interaction landing. Hides as soon as anything
  // (message, typing indicator, inline widget) appears.
  const isEmpty =
    messages.length === 0 && !running && inlineIds.length === 0;
  const welcome =
    useStateValue<Welcome | null>('/branding/welcome') ?? FALLBACK_WELCOME;
  return (
    <div className="chat-pane">
      {isEmpty && <EmptyState welcome={welcome} onSuggest={onSuggest} />}
      <div className="messages">
        {messages.map((m) => {
          const followUps = messageFollowUps[m.id] ?? [];
          // A "log line" is a short assistant turn with no markdown
          // structure (no headers / lists / code) and no anchored
          // follow-up widgets. Pin acknowledgements like
          // "Pinned X · 5 neighbors listed below." land here and get
          // rendered as a quiet timeline entry instead of a full
          // bubble — so a long click-drilldown session stays readable.
          const isAssistantLogLine =
            m.role === 'assistant' &&
            followUps.length === 0 &&
            isShortPlainText(m.content);
          const repin = isAssistantLogLine
            ? parseRepinLogLine(m.content)
            : null;
          return (
            <div key={m.id}>
              {isAssistantLogLine && repin ? (
                <button
                  type="button"
                  className="message-log message-log-repin"
                  onClick={() => onRepin(repin.arxivId)}
                  title={`Re-pin ${repin.title}`}
                >
                  <span className="message-log-prefix">Pinned</span>
                  <span className="message-log-title">{repin.title}</span>
                </button>
              ) : isAssistantLogLine ? (
                <div className="message-log">{m.content.trim()}</div>
              ) : (
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
              )}
              {m.role === 'assistant' && followUps.length > 0 && (
                <div className="message-followups">
                  {followUps.map((wid) => {
                    const w = widgets[wid];
                    if (!w) return null;
                    return <WidgetRenderer key={wid} widget={w} />;
                  })}
                </div>
              )}
            </div>
          );
        })}
        {showTyping && <TypingBubble />}
        {inlineIds.map((id) => {
          const w = widgets[id];
          if (!w) return null;
          return (
            <div key={id} className="widget-inline">
              <WidgetRenderer widget={w} />
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

/** Parse the literature_assistant pin acknowledgement convention,
 *  `Pinned · arxiv:<id> · <title>`. Returns null for log lines that
 *  don't fit (so non-pin one-liners stay non-clickable). */
function parseRepinLogLine(
  text: string,
): { arxivId: string; title: string } | null {
  const m = text.trim().match(/^Pinned\s*·\s*arxiv:(\S+)\s*·\s*(.+)$/i);
  if (!m) return null;
  return { arxivId: m[1]!, title: m[2]!.trim() };
}

/** True when the message is short and contains no markdown structure
 *  (heading, list, code fence, blockquote). Used to demote pin
 *  acknowledgements into a compact log-line style. */
function isShortPlainText(text: string): boolean {
  const t = text.trim();
  if (t.length === 0 || t.length > 200) return false;
  if (t.includes('\n')) return false;
  if (/^(#|>|\*\s|-\s|\d+\.\s|```)/.test(t)) return false;
  if (t.includes('```')) return false;
  return true;
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
  const pre = `(^|[\\s(\\[—–-])`;
  const post = `(?=[\\s.,!?;:)\\]—–-]|$)`;
  const patterns: RegExp[] = [
    new RegExp(`${pre}'([^'\\n]{3,}?)'${post}`, 'g'),
    new RegExp(`${pre}‘([^’\\n]{3,}?)’${post}`, 'g'),
    new RegExp(`${pre}"([^"\\n]{3,}?)"${post}`, 'g'),
    new RegExp(`${pre}“([^”\\n]{3,}?)”${post}`, 'g'),
  ];
  let out = text;
  for (const re of patterns) {
    out = out.replace(re, '$1<q class="pull-quote">$2</q>');
  }
  return out;
}

function EmptyState({
  welcome,
  onSuggest,
}: {
  welcome: Welcome;
  onSuggest: (content: string) => void;
}) {
  return (
    <div className="empty-state">
      {/* heading + body are server-controlled and trusted (set per
          example via AgUiAdapter::set_welcome). They support light HTML
          so demos can italicise titles or bold a phrase. */}
      <h1 dangerouslySetInnerHTML={{ __html: welcome.heading }} />
      <p dangerouslySetInnerHTML={{ __html: welcome.body }} />
      {welcome.suggestions.length > 0 && (
        <div className="empty-suggestions">
          <div className="empty-suggestions-label">Try asking</div>
          <div className="empty-suggestions-list">
            {welcome.suggestions.map((s) => (
              <button
                key={s}
                type="button"
                className="empty-suggestion"
                onClick={() => onSuggest(s)}
              >
                {s}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
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
