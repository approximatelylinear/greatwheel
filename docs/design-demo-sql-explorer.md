# Design: SQL / data explorer demo

**Status:** Drafted 2026-04-24.

## 1. Goal

"How did retention change last quarter?" — user asks in plain
language, agent writes SQL, runs it against a sample database,
renders the result as an interactive table. Clicking a row reveals
drill-down widgets the agent composes on demand; filter widgets
refine without re-asking.

The goal is to showcase an agent that **computes** — a fundamentally
different surface area from the Frankenstein demo, which is mostly
read-and-summarise. Every number on the screen is a direct function
of data the agent queried.

## 2. What it proves

- **Agent-computed state.** Agent writes SQL, runs it, renders real
  data. Not hallucination, not summary — computation.
- **Progressive deepening.** Click a row → agent emits follow-up
  widgets specific to that row (related queries, time-series charts,
  comparable records). Showcases scope/visibility on a non-trivial
  dataset.
- **Schema-aware authoring.** Agent uses a `describe_schema` host fn
  to ground its SQL in real columns/tables. Great example of host
  fns that feed the agent context at runtime.
- **Filters as typed widgets.** Filter panel is a picker-style
  widget (multi_use). Clicking a filter emits a widget event; agent
  re-runs the query. Shows the widget-as-control-surface pattern
  beyond navigation.

## 3. Sample dataset

**Chinook** (SQLite, digital music store: customers, invoices,
tracks, albums, artists, genres). Public, widely known, bundled as a
single `.db` file, realistic enough for interesting queries, small
enough to ship in the repo. No credentials, no network.

Alternative: NYC Taxi sample, COVID cases — both have broader
relatability. Chinook wins on dev-familiarity and size.

## 4. Host functions

Built on current primitives — `emit_widget`, `supersede_widget`,
`resolve_widget`, `pin_to_canvas`, `highlight_button`. Plus one new
plugin (`SqlPlugin`, registered in the Frankenstein-style example):

```python
describe_schema() -> {tables: [{name, columns: [{name, type}], rowCount}]}
# Introduces the schema so the agent can write grounded SQL.

run_sql(sql: str, limit: int = 100) -> {columns: [str], rows: [[any]]}
# Executes read-only SQL. Rejects anything starting with a non-SELECT
# verb. Returns columns + rows.
```

No `set_state`, no `watch_state`. The agent's programming model is
the same as Frankenstein's: on each user/widget event, write code,
maybe emit widgets.

## 5. Agent prompt sketch

High-level turn structure:

1. First turn (session start): call `describe_schema`, emit a
   **schema explorer** widget (list of tables as Cards, scope=table),
   pin it to canvas primary. Emit a welcome message.
2. User types a question: two-iteration pattern.
   - Iteration 1: call `run_sql(generated_query, limit=100)`, print
     the result summary. Don't call FINAL.
   - Iteration 2: emit a **data table** widget with the rows; emit
     a **query card** widget showing the SQL for transparency; emit
     2–3 follow-up suggestion buttons (`follow_up=True`). FINAL with
     a one-sentence narration of what the numbers show.
3. User clicks a row (scope=row, data={rowId, rowValues}): emit
   drill-down widgets — a "related records" data table, a
   time-series card if there's a date column, etc. Scope all new
   widgets to the clicked row.
4. User clicks a filter value: agent re-runs the query with the
   filter, supersedes the previous data table widget.

## 6. Widget catalog additions

Beyond the existing 5 tags (Column/Row/Text/Button/Card), we need
two new ones. Both are data-rendering; neither needs dynamic
behaviour beyond what we already support.

### `DataTable`
Renders a table given `columns: string[]` and `rows: any[][]`. Each
row is clickable (emits `{action: "select", data: {rowId, rowValues}}`).
Header row sortable locally on the client (no round-trip for sort).

Zod prop shape:
```ts
props: z.object({
  columns: z.array(z.string()),
  rows: z.array(z.array(z.unknown())),
  rowKey: z.string().optional(),   // column name to use as row id
})
```

### `QueryCard`
Shows the SQL the agent wrote, syntax-highlighted (use
`react-syntax-highlighter` or `shiki`), optionally collapsible. No
interaction. Transparency widget — user can always see what the
agent ran. Scope to the current query via focusedScope.

```ts
props: z.object({
  sql: z.string(),
  summary: z.string().optional(),  // one-line narration
  error: z.string().optional(),
})
```

### `FilterPanel` (could be built from existing primitives)
Actually doable with Column/Row/Card. Each filter value is a Card
with `action="filter"`, `data={column, value}`. Multi-use. Skip the
custom widget.

## 7. State shape (widget-payload-based)

Since we're not using `set_state`, data lives inside widget payloads:

- Active query's SQL lives in the `QueryCard` widget payload.
- Current result rows live in the `DataTable` widget payload.
- Filters live in the `FilterPanel` payload (and in the
  `focusedScope.filter` map, once the server's
  `extract_scope_update` picks up `data.filter` clicks).

Re-running the query means `supersede_widget` on both the QueryCard
and the DataTable. A touch awkward but mirrors the existing
Frankenstein pattern (characters widget gets superseded on
chapter change).

## 8. User flow

1. Land on the demo. Canvas shows schema explorer (tables as Cards).
   Chat empty-state prompts "Ask a question about this data."
2. Type "Which genres drove the most revenue last year?"
3. Assistant: typing → schema explorer highlights `Genre`, `Invoice`,
   `InvoiceLine`. Table appears inline:

   | Genre    | Revenue |
   |----------|---------|
   | Rock     | $826.65 |
   | Latin    | $382.14 |
   | Metal    | $261.36 |
   | …        | …       |

   Follow-up buttons: "break down by country", "show me monthly
   trend", "which tracks in Rock?".
4. Click a row (Rock) → scope=`row:Rock`. Aux canvas shows a
   "Top Rock tracks" table and a "Rock revenue by month" line chart.
5. Click "break down by country" → new query, new table.

## 9. Deliberately out of scope

- **Write SQL.** All queries are SELECT; no inserts/updates. Keep
  the demo safe by construction.
- **Joins across user-supplied sources.** Chinook only.
- **Streaming results.** Whole-result-set then render. Streaming
  plugs in later via phase-5's forward-compatible TEXT_MESSAGE
  bracketing model.
- **Charts.** First cut renders tables only. Charts (line/bar) add
  real visual variety but need a charting library (recharts /
  visx) and more widget surface. Next pass.

## 10. Open questions

- **Row id.** When the agent emits a DataTable, does it always
  provide `rowKey`? For many queries there's no natural primary key
  (`SELECT genre, SUM(total) …`). **Proposal:** make `rowKey`
  optional; when absent, the client generates a synthetic
  `row-<index>`. Click events still work; drill-down lookups are
  lossy but that's OK for the demo.
- **SQL validation.** Agent can generate invalid SQL; `run_sql`
  returns an error. Where does the error surface? **Proposal:**
  emit an error variant of QueryCard; iteration 2 sees the error in
  stdout and can try again.
- **Schema size.** Chinook has ~11 tables; fits easily. For larger
  schemas, `describe_schema` output would bloat the prompt.
  **Proposal:** defer; first cut is Chinook-sized.

## 11. Acceptance

1. User asks a question in plain language; DataTable appears with
   real rows within 2–3 seconds.
2. Clicking a row reveals at least one drill-down widget.
3. QueryCard displays the exact SQL that ran (transparency).
4. Follow-up buttons land under the assistant message and, when
   clicked, refine the query.
5. No hallucinated column names or row values (checked by spot-
   comparison against the same query in sqlite CLI).
6. No writes to the database.

## 12. Scope estimate

2–3 days including: one plugin, one widget catalog addition
(DataTable), one widget (QueryCard), a tuned agent prompt, and a
Chinook SQLite file baked in. No frontend state-shape changes.
