import { useMemo } from 'react';
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
  /** Issue #6: surface forms / labels / aliases of the focused
   *  segment's entities. Each occurrence inside a row that falls
   *  within `highlightRange` gets wrapped in `<mark>` so the user
   *  can spot them while reading. Empty / undefined → no highlight. */
  highlightTerms?: string[];
  /** Issue #6: only highlight rows whose `entryId` falls in this
   *  range (inclusive on both ends), measured by index in the
   *  messages array. Mirrors the segment's [entry_first, entry_last]. */
  highlightRange?: { first: string; last: string };
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
  highlightTerms,
  highlightRange,
}: Props) {
  // Widget records, order, and pinned set now live in the json-render
  // StateStore (populated by STATE_SNAPSHOT + STATE_DELTA). Components
  // subscribe via useStateValue; the reducer only tracks chat state.
  const widgets = useStateValue<Record<string, Widget>>('/widgets') ?? {};
  const widgetOrder = useStateValue<string[]>('/widgetOrder') ?? [];
  const pinnedIds = useStateValue<Record<string, true>>('/pinnedIds') ?? {};

  // Widgets anchored to a message should NOT also appear in the
  // scroll tail; collect their ids and exclude. SemanticSpine
  // widgets also short-circuit — they render in their own pane via
  // App.tsx and shouldn't double up in the chat scroll.
  const anchored = new Set<string>(
    Object.values(messageFollowUps).flat(),
  );
  const isSpineWidget = (id: string): boolean => {
    const w = widgets[id];
    if (!w || w.kind !== 'A2ui' || !('Inline' in w.payload)) return false;
    const inline = (w.payload as { Inline: unknown }).Inline as
      | { type?: unknown }
      | null;
    return !!inline && (inline as { type?: unknown }).type === 'SemanticSpine';
  };
  const inlineIds = widgetOrder.filter(
    (id) => !pinnedIds[id] && !anchored.has(id) && !isSpineWidget(id),
  );
  const showTyping = running;
  // Empty state: pre-interaction landing. Hides as soon as anything
  // (message, typing indicator, inline widget) appears.
  const isEmpty =
    messages.length === 0 && !running && inlineIds.length === 0;
  const welcome =
    useStateValue<Welcome | null>('/branding/welcome') ?? FALLBACK_WELCOME;

  // Issue #6: build the highlight regex once per terms-list change.
  // Sort by length descending so "FAIR-RAG" matches before "RAG"
  // (regex alternation picks the first match at a given position).
  const highlightRegex = useMemo(
    () => buildHighlightRegex(highlightTerms),
    [highlightTerms],
  );

  // Compute the [firstIdx, lastIdx] inclusive range of messages
  // whose entryId falls between the segment's entry_first and
  // entry_last. Lookup is done by entryId match — messages whose
  // entryId hasn't landed yet (race between SSE and append) just
  // miss the highlight on this render and get it next time.
  const highlightIndices = useMemo(() => {
    if (!highlightRange || !highlightRegex) return null;
    const firstIdx = messages.findIndex(
      (m) => m.entryId === highlightRange.first,
    );
    const lastIdx = messages.findIndex(
      (m) => m.entryId === highlightRange.last,
    );
    if (firstIdx < 0 && lastIdx < 0) return null;
    // If only one bound is found, clamp the other to the array edge
    // so we still highlight the partial range.
    const lo = firstIdx < 0 ? 0 : firstIdx;
    const hi = lastIdx < 0 ? messages.length - 1 : lastIdx;
    return { lo, hi };
  }, [messages, highlightRange, highlightRegex]);

  // Memoise the rehype plugin so ReactMarkdown doesn't see a new
  // function identity on every render (would force a full re-parse).
  const rehypePluginsActive = useMemo(
    () =>
      highlightRegex
        ? [rehypeRaw, rehypeHighlightTerms(highlightRegex)]
        : [rehypeRaw],
    [highlightRegex],
  );
  const rehypePluginsInert = useMemo(() => [rehypeRaw], []);
  return (
    <div className="chat-pane">
      {isEmpty && <EmptyState welcome={welcome} onSuggest={onSuggest} />}
      <div className="messages" data-chat-messages>
        {messages.map((m, idx) => {
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
          // Issue #6: should this row participate in the highlighter?
          //   - regex set + range set → row idx must be in [lo, hi]
          //   - regex set + range absent (single-entity mode) → every
          //     row is eligible so all occurrences across the chat
          //     pick up marks
          const inHighlightRange =
            highlightRegex != null &&
            (highlightIndices == null ||
              (idx >= highlightIndices.lo && idx <= highlightIndices.hi));
          // Only the segment-range mode draws the left-border tick
          // — in single-entity mode tick'ing every row would be loud.
          const drawSegmentTick =
            inHighlightRange && highlightIndices != null;
          return (
            <div
              key={m.id}
              className={`message-row${
                drawSegmentTick ? ' message-row-highlighted' : ''
              }`}
              data-entry-id={m.entryId ?? undefined}
              data-message-id={m.id}
              data-role={m.role}
            >
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
                <div className="message-log">
                  {inHighlightRange && highlightRegex ? (
                    <HighlightedText
                      text={m.content.trim()}
                      regex={highlightRegex}
                    />
                  ) : (
                    m.content.trim()
                  )}
                </div>
              ) : (
                <div className={`message message-${m.role}`}>
                  <div className="message-role">{m.role}</div>
                  <div className="message-content">
                    {m.role === 'assistant' ? (
                      <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        rehypePlugins={
                          inHighlightRange
                            ? rehypePluginsActive
                            : rehypePluginsInert
                        }
                      >
                        {pullQuote(normaliseNewlines(m.content))}
                      </ReactMarkdown>
                    ) : inHighlightRange && highlightRegex ? (
                      <HighlightedText
                        text={m.content}
                        regex={highlightRegex}
                      />
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

/* ─── Issue #6: in-message entity highlighting ──────────────────── */

/** Build a single case-insensitive regex from the highlight terms.
 *  Returns null when there's nothing useful to match — empty list,
 *  or every term shorter than 2 chars (which would cause noise on
 *  any prose). Sorts terms by length descending so longer matches
 *  win the alternation (e.g. "FAIR-RAG" beats "RAG"). */
function buildHighlightRegex(terms: string[] | undefined): RegExp | null {
  if (!terms || terms.length === 0) return null;
  const usable = Array.from(
    new Set(
      terms
        .map((t) => t.trim())
        .filter((t) => t.length >= 2),
    ),
  );
  if (usable.length === 0) return null;
  usable.sort((a, b) => b.length - a.length);
  const escaped = usable.map((t) =>
    t.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'),
  );
  // \b is ASCII-only in JS regex but works for all our entity surface
  // forms in practice (acronyms + name-ish tokens). Hyphenated names
  // like "FAIR-RAG" still match because the inner `-` is literal —
  // \b at the boundaries (before F, after G) hits non-word chars.
  return new RegExp(`\\b(?:${escaped.join('|')})\\b`, 'gi');
}

/** Render plain text with regex matches wrapped in `<mark>`. Used
 *  for log lines and user-message rows where ReactMarkdown isn't
 *  in play. */
function HighlightedText({ text, regex }: { text: string; regex: RegExp }) {
  const parts = useMemo(() => splitOnRegex(text, regex), [text, regex]);
  return (
    <>
      {parts.map((p, i) =>
        p.match ? (
          <mark key={i} className="entity-mark">
            {p.text}
          </mark>
        ) : (
          <span key={i}>{p.text}</span>
        ),
      )}
    </>
  );
}

interface RegexPart {
  text: string;
  match: boolean;
}

function splitOnRegex(text: string, regex: RegExp): RegexPart[] {
  const parts: RegexPart[] = [];
  let last = 0;
  // Defensive: clone the regex with a fresh lastIndex so reuse
  // across calls doesn't drop matches mid-string.
  const re = new RegExp(regex.source, regex.flags);
  let m: RegExpExecArray | null;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) {
      parts.push({ text: text.slice(last, m.index), match: false });
    }
    parts.push({ text: m[0], match: true });
    last = m.index + m[0].length;
    // Guard against zero-width matches (shouldn't happen with our
    // pattern but defensive for future regex changes).
    if (m[0].length === 0) re.lastIndex++;
  }
  if (last < text.length) {
    parts.push({ text: text.slice(last), match: false });
  }
  return parts.length > 0 ? parts : [{ text, match: false }];
}

/** Rehype plugin (no external deps) that walks the HAST tree from
 *  ReactMarkdown and wraps regex matches inside text nodes with
 *  `<mark class="entity-mark">`. Skips `<code>` / `<pre>` so we
 *  don't garble code blocks, and skips text already inside `<mark>`
 *  (defensive — we only mount this plugin on focused-segment rows
 *  but a future caller might double-wrap).  */
type HastNode = {
  type: string;
  tagName?: string;
  value?: string;
  children?: HastNode[];
  properties?: Record<string, unknown>;
};

function rehypeHighlightTerms(regex: RegExp) {
  const SKIP_TAGS = new Set(['code', 'pre', 'mark', 'script', 'style']);
  return () => (tree: HastNode) => {
    walk(tree, false);
    function walk(node: HastNode, insideSkip: boolean) {
      if (!node.children) return;
      const out: HastNode[] = [];
      for (const child of node.children) {
        if (
          child.type === 'element' &&
          child.tagName &&
          SKIP_TAGS.has(child.tagName)
        ) {
          // Pass through unchanged but recurse so nested marks still
          // get their inert text preserved.
          walk(child, true);
          out.push(child);
          continue;
        }
        if (child.type === 'text' && !insideSkip && child.value) {
          const parts = splitOnRegex(child.value, regex);
          if (parts.length === 1 && !parts[0]!.match) {
            out.push(child);
          } else {
            for (const p of parts) {
              if (!p.text) continue;
              if (p.match) {
                out.push({
                  type: 'element',
                  tagName: 'mark',
                  properties: { className: ['entity-mark'] },
                  children: [{ type: 'text', value: p.text }],
                });
              } else {
                out.push({ type: 'text', value: p.text });
              }
            }
          }
          continue;
        }
        if (child.children) {
          walk(child, insideSkip);
        }
        out.push(child);
      }
      node.children = out;
    }
  };
}
