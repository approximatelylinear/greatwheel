import { Fragment, isValidElement, cloneElement, useMemo, type ReactNode } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface WikiSource {
  source_id: string;
  title: string;
  author?: string | null;
  url?: string | null;
  file_path?: string | null;
  source_format: string;
  published_at?: string | null;
  ingested_at: string;
  metadata?: unknown;
}

interface WikiTocEntry {
  anchor: string;
  label: string;
  depth: number;
}

interface WikiMention {
  entity_id: string;
  label: string;
  slug: string;
  kind: string;
  // Backend ships UTF-8 byte offsets into the section markdown. Held
  // on the wire for M2 (PDF/DOM rendering); M1 highlighting drives off
  // `surface` matching since markdown rendering reshapes the source.
  norm_start: number;
  norm_end: number;
  surface: string;
}

interface WikiSection {
  anchor: string;
  heading_path: string[];
  markdown: string;
  mentions?: WikiMention[];
}

interface WikiEntityRef {
  entity_id: string;
  label: string;
  slug: string;
  kind: string;
  mentions_in_doc: number;
}

interface WikiTopicRef {
  topic_id: string;
  label: string;
  slug: string;
  chunks_in_doc: number;
}

export interface WikiDoc {
  source: WikiSource;
  toc: WikiTocEntry[];
  sections: WikiSection[];
  entities: WikiEntityRef[];
  topics: WikiTopicRef[];
}

interface Props {
  doc: WikiDoc;
  onEntityClick: (entityId: string) => void;
  onTopicClick: (topicId: string) => void;
  onClose: () => void;
}

function fmtDate(iso?: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return iso;
  return d.toLocaleDateString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  });
}

function headingTag(depth: number, text: string, id: string) {
  // depth 0 → h2 (the title is h1), depth 1 → h3, etc. Cap at h4.
  const level = Math.min(2 + depth, 4);
  if (level === 2) return <h2 id={id}>{text}</h2>;
  if (level === 3) return <h3 id={id}>{text}</h3>;
  return <h4 id={id}>{text}</h4>;
}

const REGEX_META = /[.*+?^${}()|[\]\\]/g;
function escapeRegex(s: string): string {
  return s.replace(REGEX_META, '\\$&');
}

/// Compile a single regex that matches any entity surface in the
/// section. Longest-first ordering ensures "ColBERT-v2" wins over
/// "ColBERT" when both are present. `\b` boundaries on alphanumeric
/// surfaces stop "BERT" from highlighting inside "ColBERT".
function compileSurfaceRegex(mentions: WikiMention[]): {
  pattern: RegExp;
  bySurface: Map<string, WikiMention>;
} | null {
  if (mentions.length === 0) return null;
  const bySurface = new Map<string, WikiMention>();
  for (const m of mentions) {
    // First wins — collapses duplicate surfaces from re-mentions.
    if (!bySurface.has(m.surface)) bySurface.set(m.surface, m);
  }
  const surfaces = Array.from(bySurface.keys()).sort(
    (a, b) => b.length - a.length,
  );
  // Per-surface boundary: word-char surfaces get \b on both ends;
  // surfaces starting/ending with punctuation get raw matching so we
  // don't lose e.g. "Müller, P.".
  const alternation = surfaces
    .map((s) => {
      const left = /^\w/.test(s) ? '\\b' : '';
      const right = /\w$/.test(s) ? '\\b' : '';
      return `${left}${escapeRegex(s)}${right}`;
    })
    .join('|');
  return { pattern: new RegExp(alternation, 'g'), bySurface };
}

/// Split a plain string into a mix of plain text and `<mark>` nodes,
/// one per surface match. Recursive callers handle nested React nodes.
function splitTextWithMentions(
  text: string,
  pattern: RegExp,
  bySurface: Map<string, WikiMention>,
  onClick: (entityId: string) => void,
): ReactNode[] {
  pattern.lastIndex = 0;
  const out: ReactNode[] = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = pattern.exec(text)) !== null) {
    if (match.index > cursor) {
      out.push(text.slice(cursor, match.index));
    }
    const surface = match[0];
    const mention = bySurface.get(surface);
    if (!mention) {
      // Pattern matched but we lost the mapping — emit as plain text.
      out.push(surface);
    } else {
      out.push(
        <mark
          key={`${match.index}-${mention.entity_id}`}
          className={`kb-wiki-mention kind-${mention.kind}`}
          data-entity-id={mention.entity_id}
          data-slug={mention.slug}
          onClick={(e) => {
            e.stopPropagation();
            onClick(mention.entity_id);
          }}
        >
          {surface}
        </mark>,
      );
    }
    cursor = match.index + surface.length;
    // Defensive: a zero-width match would infinite-loop.
    if (match.index === pattern.lastIndex) pattern.lastIndex += 1;
  }
  if (cursor < text.length) out.push(text.slice(cursor));
  return out;
}

/// Recursively walk React children, replacing string fragments with
/// highlight-injected runs. Skips inline `<code>` and `<a>` content so
/// link text and code spans stay untouched — false positives there are
/// disruptive (and the user can click the sidebar chip to navigate to
/// the entity).
function walkChildrenForMentions(
  node: ReactNode,
  pattern: RegExp,
  bySurface: Map<string, WikiMention>,
  onClick: (entityId: string) => void,
): ReactNode {
  if (typeof node === 'string') {
    const pieces = splitTextWithMentions(node, pattern, bySurface, onClick);
    return pieces.length === 1 && typeof pieces[0] === 'string' ? node : pieces;
  }
  if (Array.isArray(node)) {
    return node.map((child, i) => (
      <Fragment key={i}>
        {walkChildrenForMentions(child, pattern, bySurface, onClick)}
      </Fragment>
    ));
  }
  if (isValidElement(node)) {
    const t = node.type;
    if (typeof t === 'string' && (t === 'code' || t === 'a' || t === 'pre')) {
      return node;
    }
    const props = node.props as { children?: ReactNode };
    if (props.children === undefined) return node;
    return cloneElement(
      node,
      undefined,
      walkChildrenForMentions(props.children, pattern, bySurface, onClick),
    );
  }
  return node;
}

function makeHighlightedComponents(
  mentions: WikiMention[] | undefined,
  onClick: (entityId: string) => void,
) {
  const compiled = mentions ? compileSurfaceRegex(mentions) : null;
  if (!compiled) return undefined;
  const { pattern, bySurface } = compiled;
  // Wrap a small set of prose containers. ReactMarkdown calls these
  // with `children` set to the rendered React nodes for the block.
  // We mutate children in-place; markdown semantics (bold/italic/list
  // structure) survive because we only touch strings.
  const wrap =
    (Tag: 'p' | 'li' | 'td' | 'th' | 'blockquote') =>
    (props: { children?: ReactNode }) => (
      <Tag>
        {walkChildrenForMentions(props.children, pattern, bySurface, onClick)}
      </Tag>
    );
  return {
    p: wrap('p'),
    li: wrap('li'),
    td: wrap('td'),
    th: wrap('th'),
    blockquote: wrap('blockquote'),
  };
}

export function KbDocWikiWidget({
  doc,
  onEntityClick,
  onTopicClick,
  onClose,
}: Props) {
  const { source, toc, sections, entities, topics } = doc;

  // Group entities by kind for the sidebar — a flat list gets noisy
  // once you cross ~30 mentions. Order kinds by total mentions so the
  // most-talked-about kind shows first.
  const entityGroups = useMemo(() => {
    const byKind = new Map<string, WikiEntityRef[]>();
    for (const e of entities) {
      const list = byKind.get(e.kind) ?? [];
      list.push(e);
      byKind.set(e.kind, list);
    }
    return Array.from(byKind.entries())
      .map(([kind, items]) => ({
        kind,
        items,
        total: items.reduce((s, x) => s + x.mentions_in_doc, 0),
      }))
      .sort((a, b) => b.total - a.total);
  }, [entities]);

  const publishedAt = fmtDate(source.published_at);
  const ingestedAt = fmtDate(source.ingested_at);

  // Memoise per-section component overrides so we don't recompile the
  // surface regex on every render. Each section has its own regex —
  // different sections mention different entity subsets, and a
  // section's regex skips surfaces that aren't in that section.
  const sectionComponents = useMemo(() => {
    const out = new Map<string, ReturnType<typeof makeHighlightedComponents>>();
    for (const sec of sections) {
      out.set(sec.anchor, makeHighlightedComponents(sec.mentions, onEntityClick));
    }
    return out;
  }, [sections, onEntityClick]);

  return (
    <div className="kb-wiki">
      <header className="kb-wiki-header">
        <div className="kb-wiki-header-text">
          <div className="kb-wiki-eyebrow">{source.source_format}</div>
          <h1 className="kb-wiki-title">{source.title}</h1>
          {source.author && (
            <div className="kb-wiki-byline">{source.author}</div>
          )}
        </div>
        <button
          type="button"
          className="kb-wiki-close"
          onClick={onClose}
          aria-label="Close wiki"
        >
          ×
        </button>
      </header>

      <div className="kb-wiki-body">
        <nav className="kb-wiki-toc" aria-label="Contents">
          <div className="kb-wiki-rail-label">Contents</div>
          {toc.length === 0 ? (
            <div className="kb-wiki-empty">(no headings)</div>
          ) : (
            <ol>
              {toc.map((entry) => (
                <li
                  key={entry.anchor}
                  className={`kb-wiki-toc-item depth-${entry.depth}`}
                >
                  <a href={`#${entry.anchor}`}>{entry.label}</a>
                </li>
              ))}
            </ol>
          )}
        </nav>

        <main className="kb-wiki-content">
          {sections.length === 0 ? (
            <p className="kb-wiki-empty">(no content)</p>
          ) : (
            sections.map((sec) => {
              const label =
                sec.heading_path[sec.heading_path.length - 1] ?? '';
              const depth = Math.max(0, sec.heading_path.length - 1);
              const components = sectionComponents.get(sec.anchor);
              return (
                <section key={sec.anchor} className="kb-wiki-section">
                  {label && headingTag(depth, label, sec.anchor)}
                  <ReactMarkdown
                    remarkPlugins={[remarkGfm]}
                    components={components}
                  >
                    {sec.markdown}
                  </ReactMarkdown>
                </section>
              );
            })
          )}
        </main>

        <aside className="kb-wiki-aside">
          <section className="kb-wiki-infobox">
            <div className="kb-wiki-rail-label">Source</div>
            <dl>
              {source.url && (
                <>
                  <dt>URL</dt>
                  <dd>
                    <a
                      href={source.url}
                      target="_blank"
                      rel="noopener noreferrer"
                    >
                      {source.url}
                    </a>
                  </dd>
                </>
              )}
              {source.file_path && (
                <>
                  <dt>Path</dt>
                  <dd className="mono">{source.file_path}</dd>
                </>
              )}
              {publishedAt && (
                <>
                  <dt>Published</dt>
                  <dd>{publishedAt}</dd>
                </>
              )}
              {ingestedAt && (
                <>
                  <dt>Ingested</dt>
                  <dd>{ingestedAt}</dd>
                </>
              )}
              <dt>ID</dt>
              <dd className="mono">{source.source_id.slice(0, 8)}</dd>
            </dl>
          </section>

          {entityGroups.length > 0 && (
            <section className="kb-wiki-entities">
              <div className="kb-wiki-rail-label">Mentioned entities</div>
              {entityGroups.map((group) => (
                <div key={group.kind} className="kb-wiki-entity-group">
                  <div className="kb-wiki-entity-kind">{group.kind}</div>
                  <ul>
                    {group.items.map((e) => (
                      <li key={e.entity_id}>
                        <button
                          type="button"
                          className="kb-wiki-chip"
                          onClick={() => onEntityClick(e.entity_id)}
                        >
                          {e.label}
                          <span className="kb-wiki-chip-count">
                            {e.mentions_in_doc}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </section>
          )}

          {topics.length > 0 && (
            <section className="kb-wiki-topics">
              <div className="kb-wiki-rail-label">Topics</div>
              <ul>
                {topics.map((t) => (
                  <li key={t.topic_id}>
                    <button
                      type="button"
                      className="kb-wiki-chip"
                      onClick={() => onTopicClick(t.topic_id)}
                    >
                      {t.label}
                      <span className="kb-wiki-chip-count">
                        {t.chunks_in_doc}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}
