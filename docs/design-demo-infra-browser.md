# Design: infra browser / generative ops dashboard

**Status:** Drafted 2026-04-24.

## 1. Goal

"How's prod?" — user asks, agent assembles a dashboard that answers
*this particular question right now*. Service cards with health and
replicas, inline alerts, latency/error time series in the canvas aux
slot for the service the user has focused. Click a service card → the
aux content swaps to that service; relevant alerts filter to it.

Pitches greatwheel as **generative Grafana**: the dashboard is
authored per-question, not built ahead of time. Same system
primitives as Frankenstein, but the domain is ops and the widgets
are denser.

## 2. What it proves

- **Scope-driven navigation at scale.** The canvas-aux auto-swap on
  click, which we built for chapter/characters, scales up to
  service/metrics with no protocol change. The same visibility
  primitive that hid a characters widget on nav hides a metrics
  chart, an alert list, etc.
- **Resolvable widgets beyond pickers.** An agent-proposed "scale
  this up?" widget is a single-use widget — user clicks, backend
  resolves, the widget's terminal state visibly shows "Answered."
  Pattern we have but haven't demonstrated outside chapter pickers.
- **Tool-palette widgets do real work.** A "refresh all services"
  button in the header is a multi-use widget wired to a host fn
  that re-reads state. Shows the tool-palette pattern for
  continuous operations.

## 3. Data source

First cut: **mock backend.** A `MockClusterPlugin` with a
hand-written set of services, baked-in time-series data, and a few
alerts. Makes the demo runnable with no cloud credentials and
reproducible for video recordings. Interactions are real — clicking
a service really does re-query the mock and get current (static)
data.

Follow-on: a `Kubectl` or `CloudWatch` plugin with real credentials
for a live demo. Out of scope for the first cut.

## 4. Host functions

All new, gated behind a new `InfraPlugin`:

```python
list_services() -> [
  {id: str, name: str, health: "ok" | "degraded" | "down",
   replicas: {desired: int, ready: int},
   version: str, lastDeploy: timestamp}
]

get_metrics(service_id: str, metric: str, range: "1h" | "6h" | "24h")
  -> {timestamps: [str], values: [number]}
# metric ∈ {"p50_ms", "p99_ms", "error_rate", "rps"}

list_alerts(range: "1h" | "6h" | "24h")
  -> [{id, severity: "info"|"warn"|"page", service, at, message, resolved: bool}]

get_logs(service_id: str, limit: int = 100)
  -> [{at, level, message}]
```

No `set_state` — all state is in widget payloads (same as
Frankenstein).

## 5. Agent prompt sketch

1. Session start:
   - Call `list_services()`, `list_alerts(range="1h")` in parallel.
   - Emit `ServiceGrid` widget (Column of service Cards, scope=none
     at the widget level — each Card has `data.scope = {kind:
     "service", key: service_id}`).
   - Emit `AlertSummary` widget in the aux canvas slot (initially).
   - Pin ServiceGrid to canvas primary.
   - FINAL welcome message.
2. User clicks a service Card:
   - focusedScope update fires server-side from `data.scope`; the
     aux canvas auto-swaps to any widget already scoped to that
     service (or the aux stays empty if this is the first click).
   - Agent (same turn, triggered by WidgetInteraction):
     - `get_metrics(service_id, "p99_ms", range="1h")` →
       emit `MetricsChart` widget with scope `{service: <id>}`.
     - `list_alerts(range="1h")` filtered to this service → emit
       `AlertList` scope `{service: <id>}`.
     - Pin MetricsChart below canvas (aux slot).
3. User types "scale svc-api to 10 replicas":
   - Agent emits a `ProposedAction` widget — "Scale svc-api to 10
     replicas. Confirm?" — with Yes/No buttons. Not `multi_use`;
     resolves on click.
   - On Yes: agent calls (not yet existing) `scale_service(id, n)`
     host fn; widget resolves; a fresh `ServiceGrid` supersedes.
   - On No: widget resolves to Cancel; nothing happens.
4. User clicks "refresh all" in a tool-palette header widget
   (multi-use): same as session start, new data.

## 6. Widget catalog additions

### `MetricsChart`
Line chart, typically p99 over time. Props:
```ts
props: z.object({
  title: z.string(),
  metric: z.string(),
  timestamps: z.array(z.string()),
  values: z.array(z.number()),
  threshold: z.number().optional(), // optional SLO line
})
```
Renderer uses SVG (simple line chart, 300×100). `recharts` or
`visx` if we want polish — but SVG-by-hand keeps the bundle small
for the demo.

### `ServiceCard`
Specialised variant of Card with a status indicator pip. Could also
be built from existing Card + Text. **Proposal:** build from
existing primitives; add a `subtitle` convention like
`"healthy · v1.4.2 · 10/10 ready"` and colour via the Card's
existing `pressed` visual tied to a health status derived by the
agent. Avoids new widget.

### `AlertList`
Column of Cards where each alert is a Card with a severity-coloured
stripe. Agent emits them; click alert → scope shifts, aux canvas
shows logs for the alert's timestamp range.

### `ProposedAction`
Binary confirm/cancel Column with two Buttons. Already expressible —
no new widget needed.

Net new widgets: **one** (`MetricsChart`). The rest reuses what
we already have.

## 7. Scope taxonomy

Two scope kinds, both inferred from `data.scope` on click:

- `{kind: "service", key: service_id}` — filters everything to that
  service.
- `{kind: "alertRange", key: alert_id}` — filters logs/metrics to
  the time range around an alert.

Multiple scope kinds coexist in `focusedScope`; they're independent
axes. A widget can bind to one or the other via `visible`.

## 8. User flow

1. Land on demo. Canvas shows a grid of 8 service cards. Aux shows
   an alert summary. Most cards have ok status; two are yellow.
2. Click a yellow service. Aux swaps to p99 chart for that service
   showing the recent spike, alerts filter to this service.
3. Click an alert. Aux now shows logs from the alert's time range.
4. Type "what happened?" → agent emits a follow-up widget with
   "show logs", "show related traces", "show the last deploy"
   buttons.
5. Type "scale svc-api to 10 replicas" → confirm widget appears →
   click Yes → service grid refreshes with new replica count.

## 9. Deliberately out of scope

- **Real cloud/k8s credentials.** Mock cluster only. The plugin
  interface is shaped so a real implementation is a drop-in.
- **Writing to production.** `scale_service` in the mock updates the
  in-memory state only. The real-auth version is a separate story.
- **Continuous updates.** No `watch_state` → the dashboard doesn't
  auto-refresh. User clicks "refresh" or a service Card to re-query.
  Adding real-time updates is a `watch_state` / timer-driven loop
  story, not v1.
- **Multi-cluster.** One cluster.

## 10. Open questions

- **Metrics freshness.** When a user clicks a service 10 seconds
  after the last refresh, does the agent re-query metrics or reuse
  the cached data? **Proposal:** always re-query on click. Cheap
  for the mock; the real version would need caching.
- **Alert deduplication.** If 5 alerts fire for the same service,
  render them all or collapse into a count? **Proposal:** show all;
  the rendering surface is dense enough that 5 alerts fit. Revisit
  if dashboards with 50 services/200 alerts come up.
- **Approval flows.** `ProposedAction` is a one-shot resolver today.
  For real infra, we'd want audit/approver context/PR-style
  approval. Defer.

## 11. Acceptance

1. Session start: service grid with at least 5 services renders
   within 1 second.
2. Clicking a service: metrics chart appears in aux canvas within
   1 second; alerts filter correctly via scope visibility.
3. Clicking an alert: logs widget appears, replacing the metrics
   chart.
4. Propose/resolve "scale" flow: widget appears, Yes triggers the
   host fn, service grid supersedes with new replica count, widget's
   terminal banner says "Answered."
5. No stale widgets: when a new service is focused, previously-
   focused service's scoped widgets auto-hide (not superseded — just
   hidden via visibility).

## 12. Scope estimate

3–4 days including: one plugin with mock data (including some
realistic-enough synthetic time series), one new widget catalog
entry (MetricsChart), a tuned agent prompt, and the scope taxonomy
wired on both sides. The biggest engineering unknown is polishing
the MetricsChart to look credibly dashboard-y.
