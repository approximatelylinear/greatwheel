# greatwheel frontend

Throwaway Vite + React reference client for the AG-UI adapter in
[`gw-ui`](../crates/gw-ui/). The design notes are in
[`docs/design-gw-ui.md`](../docs/design-gw-ui.md).

## What's here

- Hand-rolled AG-UI client: browser `EventSource` for SSE, `fetch` for
  POSTs. No `@ag-ui/client` dependency.
- Minimal A2UI vocabulary (Column / Row / Text / Button). Swap in
  `json-render` once we need the full v0.9 catalog.
- MCP-UI placeholder: a single sandboxed `<iframe>`. For production,
  use `@mcp-ui/client`'s `UIResourceRenderer`.

## Running

Two processes — a backend that speaks AG-UI and a Vite dev server.

### 1. Backend (echo server)

```sh
cargo run -p gw-ui --example echo_server
```

Prints a `session_id` UUID on startup. Copy it.

### 2. Frontend

```sh
cd frontend
npm install
VITE_SESSION_ID=<paste-uuid-here> npm run dev
```

Or, without env var, open `http://localhost:5173/?session=<uuid>`.

The echo server mirrors each message and, on the first turn, emits an
A2UI "Yes / No" button widget. Clicking a button posts back through
`/widget-events`, which resolves the widget server-side and the
state delta round-trips over SSE.

## Scripts

```sh
npm run dev        # Vite dev server on :5173
npm run build      # type-check + production build into dist/
npm run preview    # serve dist/
npm run typecheck  # tsc --noEmit
```

## Configuration

| Env var | Default | Notes |
|---------|---------|-------|
| `VITE_API_BASE` | `http://127.0.0.1:8787` | AG-UI server base URL |
| `VITE_SESSION_ID` | _(none)_ | Session UUID; overridden by `?session=` in URL |
