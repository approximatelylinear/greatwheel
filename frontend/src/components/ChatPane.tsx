import type { Widget, WidgetEvent } from '../types';
import type { Message } from '../store/session';
import { WidgetRenderer } from './WidgetRenderer';

interface Props {
  messages: Message[];
  widgets: Record<string, Widget>;
  widgetOrder: string[];
  canvasSlot: string | null;
  onInteract: (ev: WidgetEvent) => void;
}

/**
 * Chat pane: messages in order, followed by any widgets that haven't
 * been pinned to the canvas. Terminal widgets stay in place with a
 * state banner; active ones are interactive.
 */
export function ChatPane({ messages, widgets, widgetOrder, canvasSlot, onInteract }: Props) {
  const inlineIds = widgetOrder.filter((id) => id !== canvasSlot);
  return (
    <div className="chat-pane">
      <div className="messages">
        {messages.map((m) => (
          <div key={m.id} className={`message message-${m.role}`}>
            <div className="message-role">{m.role}</div>
            <div className="message-content">{m.content}</div>
          </div>
        ))}
        {inlineIds.map((id) => {
          const w = widgets[id];
          if (!w) return null;
          return (
            <div key={id} className="widget-inline">
              <WidgetRenderer widget={w} onInteract={onInteract} />
            </div>
          );
        })}
      </div>
    </div>
  );
}
