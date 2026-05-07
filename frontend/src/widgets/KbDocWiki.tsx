import { useMemo } from 'react';
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

interface WikiSection {
  anchor: string;
  heading_path: string[];
  markdown: string;
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
              return (
                <section key={sec.anchor} className="kb-wiki-section">
                  {label && headingTag(depth, label, sec.anchor)}
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
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
