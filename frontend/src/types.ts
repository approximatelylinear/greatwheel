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
}

export interface WidgetEvent {
  widget_id: string;
  surface_id: string;
  action: string;
  data: unknown;
}

export interface UiSurface {
  id: string;
  session_id: string;
  widget_order: string[];
  canvas_slot: string | null;
}

export interface UiSurfaceSnapshot {
  surface: UiSurface;
  widgets: Widget[];
}

// AG-UI outbound event shapes — the codec on the server emits these.
export type AgUiEvent =
  | { type: 'TEXT_MESSAGE_CONTENT'; message_id: string; delta: string }
  | { type: 'RUN_FINISHED'; run_id?: string }
  | { type: 'INPUT_REQUEST'; prompt: string }
  | { type: 'UI_EVENT'; surface_id: string; widget: Widget }
  | { type: 'STATE_DELTA'; surface_id: string; patch: StateDeltaPatch }
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

export type StateDeltaPatch =
  | { kind: 'supersede'; old: string; new: Widget }
  | { kind: 'resolve'; widget_id: string; data: unknown }
  | { kind: 'expire'; widget_id: string }
  | { kind: 'pin'; widget_id: string };
