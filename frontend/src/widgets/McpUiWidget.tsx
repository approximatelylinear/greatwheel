import type { Widget } from '../types';

interface Props {
  widget: Widget;
}

/**
 * Throwaway MCP-UI renderer: loads the resource URI in a sandboxed
 * iframe. A production integration should use `@mcp-ui/client`'s
 * `UIResourceRenderer` which implements the two-level iframe sandbox,
 * CSP enforcement, and the postMessage JSON-RPC bridge back to the
 * originating MCP server. None of that is here.
 */
export function McpUiWidget({ widget }: Props) {
  if (!('Reference' in widget.payload)) {
    return <div className="widget-error">MCP-UI widget missing Reference payload</div>;
  }
  const { uri } = widget.payload.Reference;
  return (
    <iframe
      className="mcp-ui-iframe"
      src={uri}
      sandbox="allow-scripts allow-forms"
      title={`mcp-ui ${widget.id}`}
    />
  );
}
