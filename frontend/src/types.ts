// TypeScript mirrors of gw-core serde shapes.
//
// Rust enums use the default serde representation:
//   - unit variants become bare strings: "A2ui", "McpUi"
//   - tuple/struct variants become { VariantName: payload }
// Newtype structs (like WidgetId(Uuid)) serialize as the inner value.

export type WidgetKind = 'A2ui' | 'McpUi' | { Custom: string };
export type WidgetState = 'Active' | 'Resolved' | 'Expired' | 'Superseded';

export type WidgetPayload =
  | { Inline: unknown }
  | { Reference: { uri: string; csp?: string | null } };

export interface Widget {
  id: string;
  surface_id: string;
  session_id: string;
  origin_entry?: string | null;
  kind: WidgetKind;
  state: WidgetState;
  payload: WidgetPayload;
  supersedes?: string | null;
  created_at: string;
  resolved_at?: string | null;
  resolution?: unknown;
  /** When true, the adapter never auto-resolves on click. */
  multi_use?: boolean;
  /** When true, the frontend anchors this widget to the nearest
   *  assistant chat message instead of rendering it in the general
   *  scroll tail — for follow-up question buttons, etc. */
  follow_up?: boolean;
  /** Agent-declared scope. When present, the json-render translator
   *  bakes `visible: {$state: "/focusedScope/<kind>", eq: key}` onto
   *  the widget's root element so it auto-hides when the user
   *  navigates away. */
  scope?: { kind: string; key: unknown } | null;
}

export interface WidgetEvent {
  widget_id: string;
  surface_id: string;
  action: string;
  data: unknown;
}

// AG-UI outbound event shapes — the codec on the server emits these.
// All widget-state updates flow through STATE_SNAPSHOT + STATE_DELTA
// (JSON-Patch ops against the canonical state shape). See
// docs/design-json-render-migration.md §3.
export type AgUiEvent =
  /** Start / end bracket every assistant-authored text message.
   *  CONTENT deltas between them carry the text. Today there's one
   *  CONTENT per message with the full text; wire is forward-
   *  compatible with token-level streaming. */
  | { type: 'TEXT_MESSAGE_START'; message_id: string }
  | { type: 'TEXT_MESSAGE_CONTENT'; message_id: string; delta: string }
  | { type: 'TEXT_MESSAGE_END'; message_id: string }
  | { type: 'RUN_STARTED'; run_id?: string }
  | { type: 'RUN_FINISHED'; run_id?: string }
  | { type: 'RUN_ERROR'; message: string; run_id?: string }
  | { type: 'INPUT_REQUEST'; prompt: string }
  /** Full canonical state snapshot, emitted once on SSE subscribe. */
  | { type: 'STATE_SNAPSHOT'; surface_id: string; state: unknown }
  /** Vanilla AG-UI JSON-Patch delta (RFC 6902). */
  | { type: 'STATE_DELTA'; surface_id: string; patches: unknown[] }
  /** One host function is about to run. `tool_call_id` correlates
   *  with matching _ARGS and _END events. */
  | { type: 'TOOL_CALL_START'; tool_call_id: string; tool_name: string }
  /** Arguments for a dispatched tool call. `delta` carries the full
   *  `{args, kwargs}` payload (not yet streamed). */
  | { type: 'TOOL_CALL_ARGS'; tool_call_id: string; delta: unknown }
  /** Tool call completed. `result` / `error` are an AG-UI extension —
   *  spec-strict clients ignore them and treat this as a plain END. */
  | {
      type: 'TOOL_CALL_END';
      tool_call_id: string;
      result?: unknown;
      error?: string;
    }
  | {
      type: 'DEBUG_CODE_EXEC';
      code: string;
      stdout: string;
      is_final: boolean;
      error?: string;
    };

export interface CodeTrace {
  id: string;
  code: string;
  stdout: string;
  is_final: boolean;
  error?: string;
  at: number;
}

export interface ToolCall {
  id: string;
  name: string;
  args: unknown;
  result?: unknown;
  error?: string;
  status: 'running' | 'done' | 'error';
  startedAt: number;
  completedAt?: number;
}
