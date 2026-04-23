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
