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
              meta: z.record(z.string(), z.unknown()).optional(),
            })
            // Allow arbitrary extra fields on each point (the agent
            // stashes paper metadata so click handlers don't need to
            // re-fetch).
            .loose(),
        ),
        highlight: z.record(z.string(), z.boolean()).nullable().optional(),
      }),
      description:
        '2D scatter plot of typed entities/papers; click a point to drill in. Each point may carry a `meta` object that round-trips back in the click event.',
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
