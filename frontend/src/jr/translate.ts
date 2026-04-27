import type { Spec, UIElement } from '@json-render/core';
import type { Widget } from '../types';

/**
 * Convert one of our nested A2UI widget payloads (as emitted by the
 * Frankenstein server) into a json-render flat Spec.
 *
 *   our shape:  {type: "Column", children: [{type: "Text", text: "..."}, ...]}
 *   json-render: {root: "n0", elements: {n0: {type, props, children: ["n1"]}, n1: {...}}}
 *
 * Button/Card clicks are translated to an `on.press` ActionBinding
 * targeting the catalog's `interact` action. The server routes that
 * back to the agent via /widget-events.
 *
 * Pressed highlight is a `{$state: "/pressed/<widget_id>/<button_id>"}`
 * binding — truthy iff the server-side focus map names that button.
 * Scope (when declared on the widget) becomes a `visible` condition
 * on the root UIElement so json-render auto-hides widgets whose scope
 * is not currently focused. See
 * `docs/design-json-render-migration.md` §3.1.
 */
export function toJrSpec(widget: Widget): Spec | null {
  if (widget.kind !== 'A2ui') return null;
  if (!('Inline' in widget.payload)) return null;
  const widgetId = widget.id;
  const surfaceId = widget.surface_id;
  const elements: Record<string, UIElement> = {};
  let counter = 0;
  const nextKey = () => `n${counter++}`;

  const visit = (n: unknown): string => {
    const key = nextKey();
    if (!n || typeof n !== 'object') {
      elements[key] = { type: 'Text', props: { text: String(n ?? '') } };
      return key;
    }
    const node = n as Record<string, unknown>;
    const type = String(node.type ?? '');

    switch (type) {
      case 'Column':
      case 'Row': {
        const children = Array.isArray(node.children)
          ? (node.children as unknown[]).map(visit)
          : [];
        elements[key] = { type, props: {}, children };
        return key;
      }
      case 'Text': {
        elements[key] = {
          type: 'Text',
          props: { text: String(node.text ?? '') },
        };
        return key;
      }
      case 'Button': {
        const id = String(node.id ?? key);
        elements[key] = {
          type: 'Button',
          props: {
            label: String(node.label ?? id),
            pressed: { $state: `/pressed/${widgetId}/${id}` },
          },
          on: {
            press: {
              action: 'interact',
              params: {
                widgetId,
                surfaceId,
                buttonId: id,
                action: String(node.action ?? 'click'),
                data: (node.data as unknown) ?? { id },
              },
            },
          },
        };
        return key;
      }
      case 'Card': {
        const id = String(node.id ?? key);
        elements[key] = {
          type: 'Card',
          props: {
            title: String(node.title ?? id),
            subtitle: node.subtitle != null ? String(node.subtitle) : null,
            pressed: { $state: `/pressed/${widgetId}/${id}` },
          },
          on: {
            press: {
              action: 'interact',
              params: {
                widgetId,
                surfaceId,
                buttonId: id,
                action: String(node.action ?? 'select'),
                data: (node.data as unknown) ?? { id },
              },
            },
          },
        };
        return key;
      }
      case 'DataTable': {
        const columns = Array.isArray(node.columns)
          ? (node.columns as unknown[]).map(String)
          : [];
        const rawRows: unknown[][] = Array.isArray(node.rows)
          ? (node.rows as unknown[][]).map((r) =>
              Array.isArray(r) ? r : [],
            )
          : [];
        const rowKey =
          node.rowKey != null && node.rowKey !== '' ? String(node.rowKey) : null;
        // Per-row ActionBindings: registry's `emit('row:N')` routes
        // through the matching `on` entry to the catalog's `interact`
        // action with the row's data baked in.
        const on: Record<string, unknown> = {};
        rawRows.forEach((row, i) => {
          const rowObj: Record<string, unknown> = {};
          columns.forEach((c, j) => {
            rowObj[c] = row[j];
          });
          const rowId =
            rowKey && Object.prototype.hasOwnProperty.call(rowObj, rowKey)
              ? rowObj[rowKey]
              : i;
          on[`row:${i}`] = {
            action: 'interact',
            params: {
              widgetId,
              surfaceId,
              buttonId: `row:${i}`,
              action: 'select',
              data: { rowId, row: rowObj },
            },
          };
        });
        elements[key] = {
          type: 'DataTable',
          props: {
            columns,
            rows: rawRows,
            rowKey,
            truncated: Boolean(node.truncated),
          },
          on,
        } as UIElement;
        return key;
      }
      case 'QueryCard': {
        elements[key] = {
          type: 'QueryCard',
          props: {
            sql: String(node.sql ?? ''),
            summary: node.summary != null ? String(node.summary) : null,
            error: node.error != null ? String(node.error) : null,
          },
        };
        return key;
      }
      case 'EntityCloud': {
        const rawPoints = Array.isArray(node.points)
          ? (node.points as Array<Record<string, unknown>>)
          : [];
        // Render-side props: only what the SVG widget actually
        // displays. Extra fields (like `meta`) ride along the wire
        // for click bindings but the renderer doesn't need them.
        const points = rawPoints.map((p) => ({
          id: String(p.id ?? ''),
          label: String(p.label ?? ''),
          x: typeof p.x === 'number' ? p.x : 0,
          y: typeof p.y === 'number' ? p.y : 0,
          kind: p.kind != null ? String(p.kind) : undefined,
        }));
        const highlight =
          node.highlight && typeof node.highlight === 'object'
            ? (node.highlight as Record<string, boolean>)
            : null;
        // Per-point ActionBindings — the registry's
        // `emit('point:<id>')` resolves through these to fire
        // `interact` with the point baked in. Click data also
        // includes `scope: {kind: "paper", key: <id>}` so the
        // server's extract_scope_update flips
        // `/focusedScope/paper` to this id; any drill-down widget
        // emitted with the matching scope then becomes visible
        // (and prior detail widgets auto-hide).
        const on: Record<string, unknown> = {};
        for (const raw of rawPoints) {
          const id = String(raw.id ?? '');
          const kind = raw.kind != null ? String(raw.kind) : 'paper';
          on[`point:${id}`] = {
            action: 'interact',
            params: {
              widgetId,
              surfaceId,
              buttonId: `point:${id}`,
              action: 'select',
              data: {
                pointId: id,
                point: raw,
                scope: { kind, key: id },
              },
            },
          };
        }
        elements[key] = {
          type: 'EntityCloud',
          props: { points, highlight },
          on,
        } as UIElement;
        return key;
      }
      default: {
        // Unknown — fall back to a Text node showing the raw JSON so
        // we can see what we missed instead of blowing up silently.
        elements[key] = {
          type: 'Text',
          props: { text: JSON.stringify(node) },
        };
        return key;
      }
    }
  };

  const root = visit(widget.payload.Inline);

  // Scope → visibility. json-render resolves this every render against
  // the StateStore; when `focusedScope[<kind>] !== key`, the root
  // element (and therefore the whole widget) hides silently. The
  // server populates focusedScope from `post_widget_event` click data.
  if (widget.scope) {
    elements[root] = {
      ...elements[root],
      visible: {
        $state: `/focusedScope/${widget.scope.kind}`,
        eq: widget.scope.key,
      },
    };
  }

  return { root, elements };
}
