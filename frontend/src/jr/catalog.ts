import { defineCatalog } from '@json-render/core';
import { schema } from '@json-render/react/schema';
import { z } from 'zod';

export const spikeCatalog = defineCatalog(schema, {
  components: {
    Column: {
      props: z.object({}).passthrough(),
      slots: ['default'],
      description: 'Vertical stack container',
    },
    Row: {
      props: z.object({}).passthrough(),
      slots: ['default'],
      description: 'Horizontal stack container',
    },
    Text: {
      props: z.object({ text: z.string() }),
      description: 'Plain text',
    },
    Heading: {
      props: z.object({
        text: z.string(),
        level: z.number().int().min(1).max(3).optional(),
      }),
      description:
        'Serif title heading. Use as the FIRST child of a detail Column when the content is a real "title-of-thing" (paper title, person name, dataset name) — not a small-caps section label. Level 1 is largest (default).',
    },
    Link: {
      props: z.object({
        url: z.string(),
        label: z.string().nullable().optional(),
      }),
      description:
        'External hyperlink rendered as a real <a target="_blank">. Use for URLs (paper PDFs, project pages) instead of putting them in a Text node.',
    },
    Button: {
      props: z.object({
        label: z.string(),
        pressed: z.boolean().optional(),
        disabled: z.boolean().optional(),
      }),
      description: 'Clickable button that fires the press event',
    },
    Card: {
      props: z.object({
        title: z.string(),
        subtitle: z.string().nullable().optional(),
        pressed: z.boolean().optional(),
        disabled: z.boolean().optional(),
      }),
      description: 'Clickable card with title and optional subtitle',
    },
    DataTable: {
      props: z.object({
        columns: z.array(z.string()),
        rows: z.array(z.array(z.unknown())),
        rowKey: z.string().nullable().optional(),
        truncated: z.boolean().optional(),
      }),
      description:
        'Interactive table; clicking a row fires a select event with the row payload',
    },
    QueryCard: {
      props: z.object({
        sql: z.string(),
        summary: z.string().nullable().optional(),
        error: z.string().nullable().optional(),
      }),
      description:
        'Read-only display of the SQL the agent ran (transparency), with optional one-line summary or error',
    },
    SemanticSpine: {
      props: z.object({
        segments: z.array(
          z.object({
            id: z.string(),
            label: z.string(),
            kind: z.string(),
            entry_first: z.string(),
            entry_last: z.string(),
            entity_count: z.number().int(),
            entity_ids: z.array(z.string()),
            summary: z.string().nullable().optional(),
          }),
        ),
      }),
      description:
        'Vertical rail showing the conversation\'s segments — contiguous runs of session entries that share entities. Each segment carries a label, kind (comparison / decision / deep_dive / construction / other), and the entry-id range it spans so the rail can sync with chat scroll. Emitted by the backend when SpineSegmentsUpdated fires.',
    },
    Code: {
      props: z.object({
        language: z.string().nullable().optional(),
        content: z.string(),
        diff: z.boolean().optional(),
        title: z.string().nullable().optional(),
      }),
      description:
        'Read-only monospace block for source / config / diff content. Set `diff: true` to color lines beginning with `+` / `-` / `@@` as a unified diff. `language` is a hint label rendered in the corner; no actual syntax highlighting is performed.',
    },
    Markdown: {
      props: z.object({
        content: z.string(),
      }),
      description:
        'Renders GitHub-flavored markdown. Use for `EXPERIMENTS.md` excerpts, hypothesis-ledger narrative, or any longer-form prose the agent emits as a widget rather than in chat.',
    },
    DifficultyMatrix: {
      props: z.object({
        queries: z.array(z.string()),
        runs: z.array(z.string()),
        cells: z.array(
          z.array(z.enum(['exact', 'fuzzy', 'wrong', 'error', 'missing'])),
        ),
      }),
      description:
        'Cross-run query difficulty grid. Rows are queries (in order of `queries`), columns are runs (in order of `runs`). `cells[q][r]` classifies the outcome on correctness only: exact / fuzzy / wrong / error / missing. Click a cell to drill into that (run, query).',
    },
    CostTrend: {
      props: z.object({
        rows: z.array(
          z.object({
            slug: z.string(),
            model: z.string(),
            input_tokens: z.number().int().nonnegative(),
            output_tokens: z.number().int().nonnegative(),
            est_usd: z.number().nullable().optional(),
            mtime: z.string(),
          }),
        ),
        metric: z.enum(['tokens', 'usd']).optional(),
      }),
      description:
        'Inline sparkline of token usage (default) or estimated USD per run, sorted by `mtime`. `est_usd` is null for local-Ollama rows; if `metric: "usd"` and any rows are null, those points are omitted from the line.',
    },
    KbDocWiki: {
      props: z.object({
        doc: z.object({
          source: z.object({
            source_id: z.string(),
            title: z.string(),
            author: z.string().nullable().optional(),
            url: z.string().nullable().optional(),
            file_path: z.string().nullable().optional(),
            source_format: z.string(),
            published_at: z.string().nullable().optional(),
            ingested_at: z.string(),
            metadata: z.unknown(),
          }),
          toc: z.array(
            z.object({
              anchor: z.string(),
              label: z.string(),
              depth: z.number().int(),
            }),
          ),
          sections: z.array(
            z.object({
              anchor: z.string(),
              heading_path: z.array(z.string()),
              markdown: z.string(),
            }),
          ),
          entities: z.array(
            z.object({
              entity_id: z.string(),
              label: z.string(),
              slug: z.string(),
              kind: z.string(),
              mentions_in_doc: z.number().int(),
            }),
          ),
          topics: z.array(
            z.object({
              topic_id: z.string(),
              label: z.string(),
              slug: z.string(),
              chunks_in_doc: z.number().int(),
            }),
          ),
        }),
      }),
      description:
        'Wikipedia-style view of one KB source — infobox, TOC, sections (chunks grouped by heading_path), mentioned entities, referenced topics. Built by the kb_get_wiki host fn. Renders in the wiki_slot drawer; entity/topic clicks fire interact with action="open_kb_entity" / "open_kb_topic".',
    },
    KbClusterWiki: {
      props: z.object({
        cluster: z.object({
          title: z.string(),
          summary: z.string().nullable().optional(),
          sources: z.array(
            z.object({
              source_id: z.string(),
              title: z.string(),
              author: z.string().nullable().optional(),
              url: z.string().nullable().optional(),
              arxiv_id: z.string().nullable().optional(),
              source_format: z.string(),
              published_at: z.string().nullable().optional(),
              ingested_at: z.string(),
              intro: z.string().nullable().optional(),
              top_entities: z.array(
                z.object({
                  entity_id: z.string(),
                  label: z.string(),
                  slug: z.string(),
                  kind: z.string(),
                  mentions_in_doc: z.number().int(),
                }),
              ),
              top_topics: z.array(
                z.object({
                  topic_id: z.string(),
                  label: z.string(),
                  slug: z.string(),
                  chunks_in_doc: z.number().int(),
                }),
              ),
            }),
          ),
          shared_entities: z.array(
            z.object({
              entity_id: z.string(),
              label: z.string(),
              slug: z.string(),
              kind: z.string(),
              total_mentions: z.number().int(),
              source_count: z.number().int(),
            }),
          ),
          shared_topics: z.array(
            z.object({
              topic_id: z.string(),
              label: z.string(),
              slug: z.string(),
              total_chunks: z.number().int(),
              source_count: z.number().int(),
            }),
          ),
        }),
      }),
      description:
        'Cluster-style wiki digest of N KB sources — per-source card (title, author, intro, top entities/topics) plus shared-entities and shared-topics rails. Built by the kb_get_cluster_wiki host fn. Renders in the wiki_slot drawer alongside KbDocWiki; entity/topic chip clicks fire interact with action="open_kb_entity" / "open_kb_topic"; per-source "Open as wiki" buttons fire action="open_source_wiki" with data={source_ref}.',
    },
    EntityCloud: {
      props: z.object({
        points: z.array(
          z
            .object({
              id: z.string(),
              label: z.string(),
              x: z.number(),
              y: z.number(),
              kind: z.string().optional(),
              cluster: z.number().int().optional(),
              year: z.string().optional(),
              category: z.string().optional(),
              meta: z.record(z.string(), z.unknown()).optional(),
            })
            // Allow arbitrary extra fields on each point (the agent
            // stashes paper metadata so click handlers don't need to
            // re-fetch).
            .loose(),
        ),
        clusters: z
          .array(
            z.object({
              id: z.number().int(),
              label: z.string(),
              x: z.number(),
              y: z.number(),
            }),
          )
          .nullable()
          .optional(),
        highlight: z.record(z.string(), z.boolean()).nullable().optional(),
      }),
      description:
        '2D scatter plot of typed entities/papers; click a point to drill in. Each point may carry a `cluster` id (color), and the optional top-level `clusters` array provides faint always-on centroid labels.',
    },
  },
  actions: {
    interact: {
      params: z.object({
        widgetId: z.string(),
        surfaceId: z.string(),
        buttonId: z.string(),
        action: z.string(),
        data: z.unknown().optional(),
      }),
      description:
        'Forward a widget button/card press back to the greatwheel server',
    },
  },
});
