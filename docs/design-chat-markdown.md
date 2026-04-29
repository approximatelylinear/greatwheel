# Design: Markdown formatting in chat messages

**Status:** Draft, 2026-04-29.
**Touches:** `frontend/src/components/ChatPane.tsx`, `frontend/src/styles.css`,
the system prompt of each example that emits user-facing prose
(`crates/gw-ui/examples/literature_assistant.rs`,
`crates/gw-ui/examples/echo_server.rs`, etc.).

## 1. Motivation

Assistant messages in the literature-assistant currently flow as
single paragraphs of prose. A typical research response — comparing
three papers, summarizing methodology, listing experimental results —
is hard to scan because everything has the same visual weight.

Concrete examples from the live session log:

> From the paper text, EVOR's method is an iterative retrieval
> framework for code generation: it keeps updating the retrieved
> evidence as generation progresses, so retrieval adapts to the
> coding task and the model's partial solution rather than staying
> fixed from the start.

This is one paragraph that wants to be:

> ### EVOR: iterative retrieval for code generation
>
> - **Frame:** retrieval adapts as generation proceeds
> - **Contrast with baseline:** evidence isn't fixed from the start
> - **Mechanism:** retrieved set updates from the model's partial
>   solution

Or compare-three-papers responses that want a table. None of this
landing well today is a *prompt + styling* problem — the rendering
infrastructure already exists.

## 2. What's already there

`frontend/src/components/ChatPane.tsx` runs assistant messages
through `ReactMarkdown` with `remark-gfm` and `rehype-raw`:

```tsx
<ReactMarkdown
  remarkPlugins={[remarkGfm]}
  rehypePlugins={inHighlightRange ? rehypePluginsActive : rehypePluginsInert}
>
  {pullQuote(normaliseNewlines(m.content))}
</ReactMarkdown>
```

This means GFM is parsed today: tables, task lists, autolinks,
strikethrough, fenced code blocks. The rehype layer handles raw HTML
(used by `pullQuote()` for literary quote tags) and now also wraps
entity surface forms in `<mark>` (Issue #6).

CSS in `styles.css` has rules for:

- `.message-assistant .message-content h1 / h2 / h3` (heading sizes,
  margins).
- `.message-assistant .message-content > :first-child` /
  `:last-child` margin reset.
- `.entity-mark` (entity highlights from Issue #6).
- `.pull-quote` (literary single-quoted spans).

Notably **not** styled or only minimally so:

- Bulleted / numbered lists.
- Tables (GFM).
- Fenced code blocks (`<pre><code>`) — get browser default monospace
  but no syntax-aware treatment, language tag, or scroll-on-overflow.
- Inline code (`<code>`) — no background pill / monospace contrast.
- Blockquotes (other than the `pullQuote` tag injected by us).
- Horizontal rules.
- Definition lists, footnotes — out of scope (not GFM).

So the rendering path is real but the CSS is half-finished and the
model isn't using the elements that would benefit most.

## 3. Why the model doesn't use markdown today

In tool-using agents like the literature_assistant, user-visible
prose flows through `FINAL("...")`. The system prompt asks for
prose, not markdown. Examples:

> FINAL: a one-paragraph narration — "I've laid out 124 entities
> across 50 papers..."
>
> FINAL(f"Pinned · arxiv:{arxiv_id} · {paper['title']}")

Two specific frictions for the agent:

- **Python f-strings escape `{` / `}`** — markdown's table syntax
  (`| col | col |`) and code-block fences (` ``` `) play fine, but
  curly braces in tables / inline code (rare) need `{{` / `}}`. The
  agent has to be aware of this when composing.
- **Multi-line FINAL strings** require triple-quoted Python literals.
  The agent can do this; it just rarely does because nothing in the
  prompt tells it to.

## 4. Proposal

### 4.1 System-prompt guidance per example

Each example that emits user-facing FINAL prose gains a "formatting"
section in its system prompt. Concretely, the literature_assistant's
prompt today tells the agent:

> FINAL("text") — terminates the turn with a chat narration.

It would extend with a Markdown style guide tuned to the demo:

> **FINAL formatting (markdown).** Your `FINAL` text is rendered as
> markdown. Use structure when it helps the reader scan:
>
> - **Headings** (`### Foo`) for multi-part responses comparing
>   options or describing a method's pieces.
> - **Bulleted lists** for parallel items (papers, methods,
>   findings); aim for ≤6 items per list.
> - **Tables** when comparing 3+ things on shared axes:
>   `| Method | Key idea | Where it shines |`
> - **Inline code** (\`backticks\`) for arxiv ids, function names,
>   variable names — never wrap full sentences.
> - **Fenced code blocks** for queries or snippets the user might
>   re-run.
>
> **Don't** wrap a 2-sentence answer in headings or bullets — short
> answers stay as plain prose. Markdown is a tool for scanability,
> not an aesthetic.

The "don't overuse" line is load-bearing — without it small models
turn every response into a wall of headings.

Per-demo flavor:

| Example                | Default style                                         |
|------------------------|-------------------------------------------------------|
| literature_assistant   | Tables + bullets + code; rich markdown encouraged.    |
| Frankenstein           | Mostly prose; blockquotes for novel excerpts; minimal headings. |
| echo_server            | Whatever the user types — passthrough.                |
| sql_explorer           | Fenced SQL blocks always; tables for result previews. |

### 4.2 CSS coverage for unstyled elements

Add styles to `frontend/src/styles.css` under
`.message-assistant .message-content` for:

- `ul, ol` — tight `gap` / `margin`, accent-tinted markers, indent
  matching the rest of the bubble.
- `code` (inline) — soft background pill, monospace, slightly larger
  than browser default to read clean against `--font-serif`.
- `pre > code` (fenced) — dark surface, padded, scrolls horizontally
  on overflow, optional language-name chip in the top-right when
  present (`pre.language-python::before { content: "python" }`).
- `table` — full-width, zebra rows, sticky header if `> 6` rows
  (CSS-only via `position: sticky`), `overflow-x: auto` wrapper for
  narrow columns.
- `blockquote` — left-rule + italic + reduced opacity (matches the
  pullQuote treatment for tonal consistency).
- `hr` — muted 1px divider with extra vertical margin.

### 4.3 Edge cases worth handling explicitly

- **Curly-brace escaping in f-strings.** Document in the system
  prompt that `|` in tables works directly; only `{` / `}` need
  doubling. Add a small example.
- **Code block contents passing through entity highlighter.** The
  rehype plugin from Issue #6 already skips `<code>` / `<pre>` to
  avoid garbling code with `<mark>` injections — keep that.
- **Markdown-looking text that isn't markdown.** "issue #1234" would
  not trigger header parsing (no leading `#` at line start), but a
  one-line response of "# 1234" would. Prompt guidance: only use
  `#` for actual headings.
- **`pullQuote` interaction.** `pullQuote()` runs before
  ReactMarkdown and wraps single-quoted spans in `<q class="pull-quote">`.
  If the agent emits `'BM25'` inside a code block today, pullQuote
  ignores code-block syntax and could rewrite inside backticks. Fix:
  make `pullQuote` skip text inside ` ``` ` fences and inline backticks.
  Small, contained.

### 4.4 Streaming considerations (deferred)

The current emit pattern is one full `TEXT_MESSAGE_DELTA` per
message — no token streaming yet. When streaming lands, partial
markdown will break rendering mid-token (a half-typed `### F` is a
heading; the next chunk that completes "Foo" needs to not double-
render the heading). Mitigation:

- Buffer deltas for a few hundred ms before flushing to the
  ReactMarkdown re-parse, so most "incomplete syntax" windows
  collapse.
- Or: render the in-flight message as monospace plain text until
  `TEXT_MESSAGE_END`, swap to markdown rendering on bracket close.
  Simpler; acceptable polish gap during streaming.

Pick when streaming actually exists. Don't pre-engineer.

### 4.5 Out of scope

- **KaTeX / math rendering.** Worth its own doc if it matters; not
  blocking general formatting.
- **Mermaid diagrams.** Heavy, niche, separate doc.
- **Syntax highlighting for code blocks** (e.g. via
  `rehype-highlight` or `shiki`). Worth doing later — bundle size
  matters; pick a lightweight option (`shiki` is too big for our
  bundle; `rehype-highlight` + a small subset of languages is
  reasonable). v1 just gets monospace + dark background.
- **User-typed markdown.** The MessageInput is a single-line text
  field — users are sending chat queries, not authoring documents.
  No need to render their input as markdown.
- **Sanitization beyond rehype-raw default.** XSS risk acceptable
  because the server is trusted (it's the agent we own); revisit if
  a remote agent ever lands.

## 5. Acceptance

1. Literature-assistant output for "compare these three papers"
   produces a markdown table that renders with zebra rows and
   horizontal scroll if narrow.
2. A response with `### Headings`, `- bullets`, fenced code blocks,
   inline code, and a blockquote all render with consistent
   typography matching the assistant bubble's body font.
3. The agent doesn't gratuitously add headings to one-sentence
   answers — system prompt's "use markdown sparingly" line is
   exercised.
4. Pin-acknowledgement log lines (the short "Pinned · arxiv:…"
   form, kept as plain text by the existing `isShortPlainText`
   heuristic) still render as the compact log style, not as a full
   markdown bubble.
5. Entity highlighting (Issue #6) still works inside markdown
   content — `<mark>` lands in paragraphs and list items but never
   inside `<code>`.

## 6. Scope

~1 day:

- 1–2 hours: extend the literature_assistant system prompt with the
  formatting section + 2–3 examples of well-formatted FINALs.
- 2 hours: CSS for `ul / ol / code / pre / table / blockquote / hr`
  inside `.message-assistant .message-content`.
- 1 hour: fix `pullQuote()` to skip code blocks / inline backticks.
- 1 hour: smoke-test with a few representative literature_assistant
  prompts; tune the system-prompt examples based on what the agent
  actually produces.
- The Frankenstein / sql_explorer per-demo flavors land as smaller
  follow-ups when those demos see active use.

## 7. Risks

- **Model overcorrects.** Small models (qwen3.5:9b) tend to apply
  any new prompt instruction maximally — every response becomes a
  wall of headings. Mitigation: 2–3 explicit "DON'T" examples in
  the prompt showing prose-only answers for short questions.
- **Tables overflow chat column.** The literature_assistant's
  chat column is narrow (chat-primary layout). CSS `overflow-x: auto`
  on a `<div class="md-table-wrap">` wrapper handles it; the wrapper
  needs to come from a small rehype plugin since GFM tables don't
  emit a wrapper by default. Acceptable scope.
- **Existing literary `pullQuote` collides with markdown `>`
  blockquotes.** Today `>` at line start triggers blockquote and
  `'foo'` triggers pullQuote — they can co-exist. Just make sure
  the test suite covers a row with both.

## 8. Cross-references

- `design-demo-literature-assistant.md` §5 (agent prompt sketch) —
  the FINAL formatting addition goes alongside its existing
  `FINAL("text")` line.
- `design-gw-ui.md` (rendering pipeline) — no protocol changes; the
  catalog already knows how to render text.
- `design-semantic-spine.md` §14 (entity highlighting) — confirm the
  rehype plugin's `code` / `pre` skip-list stays in place.
