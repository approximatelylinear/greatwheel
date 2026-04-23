import type { Spec, UIElement } from '@json-render/core';

/**
 * Convert one of our nested A2UI widget payloads (as emitted by the
 * Frankenstein server) into a json-render flat Spec.
 *
 *   our shape:  {type: "Column", children: [{type: "Text", text: "..."}, ...]}
 *   json-render: {root: "n0", elements: {n0: {type, props, children: ["n1"]}, n1: {...}}}
 *
 * Button/Card clicks are translated to an `on.press` ActionBinding
 * targeting the catalog's `interact` action. The server routes that
 * back to the agent via the existing /widget-events endpoint.
 */
/**
 * If `pressedId` is provided, the Card/Button `pressed` prop is
 * baked as a concrete boolean at translation time. If omitted, the
 * spec uses a `{$state: "/pressed/<id>"}` binding instead, so a
 * json-render StateStore drives highlight updates directly.
 */
export function toJrSpec(
  node: unknown,
  widgetId: string,
  surfaceId: string,
  pressedId: string | null = null,
): Spec {
  const pressedProp = (id: string) =>
    pressedId === null ? { $state: `/pressed/${id}` } : pressedId === id;

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
            pressed: pressedProp(id),
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
            pressed: pressedProp(id),
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

  const root = visit(node);
  return { root, elements };
}
