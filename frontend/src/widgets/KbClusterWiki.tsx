import { useMemo } from 'react';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';

interface WikiClusterSource {
  source_id: string;
  title: string;
  author?: string | null;
  url?: string | null;
  arxiv_id?: string | null;
  source_format: string;
  published_at?: string | null;
  ingested_at: string;
  intro?: string | null;
  top_entities: WikiEntityRef[];
  top_topics: WikiTopicRef[];
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

interface WikiClusterEntity {
  entity_id: string;
  label: string;
  slug: string;
  kind: string;
  total_mentions: number;
  source_count: number;
}

interface WikiClusterTopic {
  topic_id: string;
  label: string;
  slug: string;
  total_chunks: number;
  source_count: number;
}

export interface WikiCluster {
  title: string;
  summary: string | null;
  sources: WikiClusterSource[];
  shared_entities: WikiClusterEntity[];
  shared_topics: WikiClusterTopic[];
}

interface Props {
  cluster: WikiCluster;
  onEntityClick: (entityId: string) => void;
  onTopicClick: (topicId: string) => void;
  onSourceOpen: (sourceRef: string) => void;
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

export function KbClusterWikiWidget({
  cluster,
  onEntityClick,
  onTopicClick,
  onSourceOpen,
  onClose,
}: Props) {
  const { title, summary, sources, shared_entities, shared_topics } = cluster;

  const sharedEntityGroups = useMemo(() => {
    const byKind = new Map<string, WikiClusterEntity[]>();
    for (const e of shared_entities) {
      const list = byKind.get(e.kind) ?? [];
      list.push(e);
      byKind.set(e.kind, list);
    }
    return Array.from(byKind.entries())
      .map(([kind, items]) => ({
        kind,
        items,
        total: items.reduce((s, x) => s + x.total_mentions, 0),
      }))
      .sort((a, b) => b.total - a.total);
  }, [shared_entities]);

  return (
    <div className="kb-wiki kb-cluster">
      <header className="kb-wiki-header">
        <div className="kb-wiki-header-text">
          <div className="kb-wiki-eyebrow">
            cluster · {sources.length} source{sources.length === 1 ? '' : 's'}
          </div>
          <h1 className="kb-wiki-title">{title}</h1>
          {summary && <p className="kb-cluster-summary">{summary}</p>}
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

      <div className="kb-wiki-body kb-cluster-body">
        <nav className="kb-wiki-toc" aria-label="Sources">
          <div className="kb-wiki-rail-label">Sources</div>
          {sources.length === 0 ? (
            <div className="kb-wiki-empty">(empty)</div>
          ) : (
            <ol>
              {sources.map((s, i) => (
                <li
                  key={s.source_id}
                  className="kb-wiki-toc-item depth-0"
                >
                  <a href={`#cluster-src-${i}`}>{s.title}</a>
                </li>
              ))}
            </ol>
          )}
        </nav>

        <main className="kb-wiki-content kb-cluster-content">
          {sources.length === 0 ? (
            <p className="kb-wiki-empty">(no sources)</p>
          ) : (
            sources.map((src, i) => {
              const published = fmtDate(src.published_at);
              const sourceRef = src.arxiv_id ?? src.source_id;
              return (
                <article
                  key={src.source_id}
                  id={`cluster-src-${i}`}
                  className="kb-cluster-card"
                >
                  <header className="kb-cluster-card-header">
                    <h2>{src.title}</h2>
                    <button
                      type="button"
                      className="kb-cluster-open"
                      onClick={() => onSourceOpen(sourceRef)}
                      title="Open this source as a full wiki page"
                    >
                      Open as wiki →
                    </button>
                  </header>
                  <div className="kb-cluster-meta">
                    {src.author && <span>{src.author}</span>}
                    {published && <span>{published}</span>}
                    {src.arxiv_id && (
                      <span className="mono">arXiv:{src.arxiv_id}</span>
                    )}
                    {src.url && (
                      <a
                        href={src.url}
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        link ↗
                      </a>
                    )}
                  </div>
                  {src.intro && (
                    <div className="kb-cluster-intro">
                      <ReactMarkdown remarkPlugins={[remarkGfm]}>
                        {src.intro}
                      </ReactMarkdown>
                    </div>
                  )}
                  {(src.top_entities.length > 0 ||
                    src.top_topics.length > 0) && (
                    <div className="kb-cluster-chips">
                      {src.top_entities.map((e) => (
                        <button
                          key={`e-${e.entity_id}`}
                          type="button"
                          className="kb-wiki-chip"
                          onClick={() => onEntityClick(e.entity_id)}
                        >
                          {e.label}
                          <span className="kb-wiki-chip-count">
                            {e.mentions_in_doc}
                          </span>
                        </button>
                      ))}
                      {src.top_topics.map((t) => (
                        <button
                          key={`t-${t.topic_id}`}
                          type="button"
                          className="kb-wiki-chip kb-wiki-chip-topic"
                          onClick={() => onTopicClick(t.topic_id)}
                        >
                          #{t.label}
                          <span className="kb-wiki-chip-count">
                            {t.chunks_in_doc}
                          </span>
                        </button>
                      ))}
                    </div>
                  )}
                </article>
              );
            })
          )}
        </main>

        <aside className="kb-wiki-aside">
          {sharedEntityGroups.length > 0 && (
            <section className="kb-wiki-entities">
              <div className="kb-wiki-rail-label">Shared entities</div>
              {sharedEntityGroups.map((group) => (
                <div key={group.kind} className="kb-wiki-entity-group">
                  <div className="kb-wiki-entity-kind">{group.kind}</div>
                  <ul>
                    {group.items.map((e) => (
                      <li key={e.entity_id}>
                        <button
                          type="button"
                          className="kb-wiki-chip"
                          onClick={() => onEntityClick(e.entity_id)}
                          title={`in ${e.source_count} sources · ${e.total_mentions} mentions`}
                        >
                          {e.label}
                          <span className="kb-wiki-chip-count">
                            {e.source_count}/{sources.length}
                          </span>
                        </button>
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </section>
          )}
          {shared_topics.length > 0 && (
            <section className="kb-wiki-topics">
              <div className="kb-wiki-rail-label">Shared topics</div>
              <ul>
                {shared_topics.map((t) => (
                  <li key={t.topic_id}>
                    <button
                      type="button"
                      className="kb-wiki-chip"
                      onClick={() => onTopicClick(t.topic_id)}
                      title={`in ${t.source_count} sources · ${t.total_chunks} chunks`}
                    >
                      {t.label}
                      <span className="kb-wiki-chip-count">
                        {t.source_count}/{sources.length}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </section>
          )}
          {sharedEntityGroups.length === 0 && shared_topics.length === 0 && (
            <section>
              <div className="kb-wiki-rail-label">Overlap</div>
              <div className="kb-wiki-empty">
                No entities or topics span multiple sources yet — entity
                extraction may still be running.
              </div>
            </section>
          )}
        </aside>
      </div>
    </div>
  );
}
