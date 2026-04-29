# Design: Semantic Spine + Entity Sidebar (`gw-ui`)

**Status:** Drafted 2026-04-28; shipped end-to-end 2026-04-29
(Issues #1–#6 plus clickable entity cards — see
[`design-demo-literature-assistant.md` §14](design-demo-literature-assistant.md)
for how it integrates with the literature-assistant surface).
**Depends on:** [`design-gw-ui.md`](design-gw-ui.md) (widget store, AG-UI
adapter), [`design-kb-entities.md`](design-kb-entities.md) (entity tables
+ host fns — already landed), `feat(literature-assistant): runtime
entity extraction off the chat path` (commit f9f0be9, the extraction
pattern this reuses).

## 1. Motivation

The default chat UI is a single vertical scroll of message bubbles.
That's fine for short conversations, but research-style sessions —
which is what the literature assistant and BrowseComp work look like —
hit three problems:

1. **"Where did we discuss X?"** Users scroll-search by keyword. The
   conversation has structure (BM25 vs ColBERT, recall vs precision,
   pipeline construction) but the chat surface doesn't expose it.
2. **"What is this conversation actually about?"** No compressed,
   glanceable summary. The closest thing is the conversation title,
   which is set once and never updates.
3. **"What entities have we committed to?"** The agent has a working
   ontology — methods chosen, tradeoffs accepted, datasets fixed — but
   it lives implicitly in prose. There's no separation between
   "mentioned in passing" and "load-bearing for the rest of the
   session."

A **Semantic Spine** — a vertical rail next to chat, with entity-cluster
markers tied to turn ranges — solves all three at once if it's
**interactive** (clickable, navigable, can generate prompts) rather
than a passive minimap. The right-hand **Entity Sidebar** is the
structured view that the spine pivots: click a cluster, sidebar shows
the entities + relations + sample turns from that segment.

The data this needs already exists. `gw-kb` extracts and persists
typed entities (`kb_entities`, `kb_chunk_entity_links`). The
literature assistant already runs entity extraction off the chat path.
What's missing is (a) per-turn entity attribution stored alongside the
session, (b) a spine projection over those attributions, (c) a
sidebar widget that consumes the projection, and (d) wiring the two
into a bidirectional control surface.

## 2. Design Principles

**Spine is derived state, not a new entity type.** The spine is a
projection over `(SessionEntry × kb_entities)`. We don't store "spine
markers" — we store entry-level entity links (mirroring
`kb_chunk_entity_links`) and project them on read. This keeps the spine
truthful by construction: it cannot drift from the underlying KB.

**Mentioned vs committed is a status, not a type.** Every entity
appearance starts as `Mentioned`. A user (or agent) action promotes it
to `Committed` — it joins the session workspace. Faint vs solid markers
are a render-time treatment of this status field, not two parallel
graphs.

**Clusters are computed, labeled, and cached — not authored.**
A contiguous run of entries sharing entities collapses into a cluster.
Labels come from a cheap LLM call ("Comparison", "Pipeline Design",
"Decision") triggered on cluster formation; failures fall back to "N
entities" with the dominant kind. The agent never writes spine state
directly; the agent's only contribution is calling `kb` host fns
during a turn, which the runtime observes.

**Spine is a widget.** Re-using the `gw-ui` widget protocol means the
spine survives history replay, hydrates from `/surface` on reconnect,
and gets persisted with the session. It's `multi_use: true` (a
persistent navigation surface, not a one-shot form). Frontend renders
it in a fixed third pane next to chat, but the position is a frontend
concern — backend just emits a widget.

**Reverse interaction is a widget event, not a host fn.** Clicking a
spine marker opens an action menu (revisit / expand / compare).
Selecting an action sends a `WidgetInteraction` whose `data` carries
the chosen action and the cluster ID. The loop translates that into
the agent's next turn (synthetic user message + structured context),
the same path `WidgetInteraction` already uses for forms.

**Sidebar is a separate widget, pinned to canvas.** The Entity
Sidebar is its own A2UI widget with `pin_to_canvas` set. It listens
for spine selection state (via the `focusedScope` mechanism from the
json-render migration design) and re-renders its content. No new state
machinery — it reuses the scope/focus pattern.

## 3. Data Model

### 3.1 Per-entry entity attribution

Mirror `kb_chunk_entity_links` for session entries. Store in `gw-loop`'s
session DB rather than `gw-kb` — the link is between a *session entry*
and a *kb entity*, and `gw-loop` already owns the entries table.

```sql
CREATE TABLE session_entry_entities (
    entry_id    UUID NOT NULL REFERENCES session_entries(entry_id),
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id),
    surface     TEXT NOT NULL,         -- the text span as it appeared
    role        TEXT NOT NULL,         -- "introduced" | "referenced" | "decided" | "compared"
    status      TEXT NOT NULL,         -- "mentioned" | "committed"
    confidence  REAL NOT NULL,
    span_start  INT,                   -- char offset within entry text
    span_end    INT,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id, entity_id, span_start)
);

CREATE INDEX session_entry_entities_entity ON session_entry_entities(entity_id);
CREATE INDEX session_entry_entities_status ON session_entry_entities(status)
    WHERE status = 'committed';
```

Spans are optional but nice — they enable in-message highlighting later
without another extraction pass.

### 3.2 Per-entry entity relations

A chat message can *establish* a relationship between two entities
("BM25 vs ColBERT", "ColBERT then cross-encoder rerank", "BM25 is
faster than ColBERT"). These are different from `kb_entity_links`:

- `kb_entity_links` is the **global, symmetric, undirected**
  co-mention/cosine graph computed by `linking.rs` over the whole
  corpus. Predicate is just `'related'`. It's good for "what's
  near this entity in the KB."
- Chat relations are **per-entry, typed, often directional, with a
  surface form**. They're what the user and agent are actually
  asserting in this conversation. Confidence comes from extraction,
  not aggregation.

Stored next to the entity links, with the same ownership rationale
(`gw-loop` owns the entry FK):

```sql
CREATE TABLE session_entry_relations (
    relation_id   UUID PRIMARY KEY,
    entry_id      UUID NOT NULL REFERENCES session_entries(entry_id),
    subject_id    UUID NOT NULL REFERENCES kb_entities(entity_id),
    object_id     UUID NOT NULL REFERENCES kb_entities(entity_id),
    predicate     TEXT NOT NULL,        -- "compared_with" | "tradeoff_in" | "composes" | "outperforms" | "is_a" | "uses" | "evaluated_on" | ...
    directed      BOOL NOT NULL,        -- false for symmetric predicates like "compared_with"
    surface       TEXT NOT NULL,        -- the span that asserted it, for trace + sidebar
    confidence    REAL NOT NULL,
    span_start    INT,
    span_end      INT,
    extracted_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX session_entry_relations_entry   ON session_entry_relations(entry_id);
CREATE INDEX session_entry_relations_subject ON session_entry_relations(subject_id);
CREATE INDEX session_entry_relations_object  ON session_entry_relations(object_id);
```

**Predicate vocabulary.** Free-form `TEXT` with a recommended set —
same convention as `kb_entities.kind`. Recommended starting set
covers the cases the literature/BrowseComp work actually surfaces:

| Predicate | Directed? | Example surface |
|---|---|---|
| `compared_with` | no | "BM25 vs ColBERT" |
| `tradeoff_in`   | no | "tradeoff between recall and precision" |
| `composes`      | yes (subject feeds object) | "ColBERT → cross-encoder rerank" |
| `outperforms`   | yes | "ColBERT beats BM25 on recall" |
| `is_a`          | yes | "BM25 is a sparse lexical retriever" |
| `uses`          | yes | "RAG uses dense retrieval" |
| `evaluated_on`  | yes | "evaluated on BrowseComp" |
| `decided`       | no  | (no second entity — see below) |

Single-entity assertions ("we'll go with ColBERT") are not relations
— they're commitments, handled by the `status` field on
`session_entry_entities` (§3.1).

**Promotion to the global graph.** When the same relation pattern
appears across many chunks/entries (`{subject_id, object_id,
predicate}` repeats with high cumulative confidence), a periodic job
can write it back as a `kb_entity_links` row with the predicate as
`relation`. This gives `linking.rs` a richer signal than pure
co-occurrence over time, without forcing every chat-level assertion
to commit to the global graph immediately. Defer the promotion job
until we have data to tune the threshold against.

### 3.3 Cluster cache

Clusters are derivable from the link table, but labeling is expensive
enough (one LLM call per new cluster) that we cache:

```sql
CREATE TABLE session_segments (
    segment_id  UUID PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES sessions(session_id),
    label       TEXT NOT NULL,         -- "Comparison", "Pipeline Design", ...
    kind        TEXT NOT NULL,         -- "comparison" | "decision" | "deep_dive" | "construction" | "other"
    entry_first UUID NOT NULL REFERENCES session_entries(entry_id),
    entry_last  UUID NOT NULL REFERENCES session_entries(entry_id),
    entity_ids  UUID[] NOT NULL,       -- top entities in segment, ranked by mentions
    summary     TEXT,                  -- optional LLM-generated, ≤140 chars
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    invalidated_at TIMESTAMPTZ         -- set when re-segmentation supersedes
);

CREATE INDEX session_segments_session ON session_segments(session_id)
    WHERE invalidated_at IS NULL;
```

Re-segmentation (after enough new entries arrive) marks old segments
invalidated and writes new ones. We keep the old rows for history
replay — refreshing a past view of the spine should show the old
labels.

### 3.4 Workspace commitment

`status = 'committed'` is the workspace state. Promoting an entity
from `mentioned` → `committed` is a single UPDATE on
`session_entry_entities`. We also need the inverse view ("what
entities are committed in this session, regardless of which entry
introduced them") for the sidebar's workspace tab; that's a
straightforward query with `DISTINCT entity_id WHERE status =
'committed'`, no separate table needed.

## 4. Backend Pipeline

### 4.1 Extraction (off the chat path)

Hook into the existing entry-write path in `gw-loop/src/conversation.rs`.
After an entry is persisted, fire a background task that does **joint
entity + relation extraction** — one LLM call returns both:

```rust
// crates/gw-loop/src/spine/extract.rs
pub struct EntryExtraction {
    pub entities:  Vec<EntryEntityLink>,
    pub relations: Vec<EntryRelation>,
}

pub async fn extract_entry(
    entry: &SessionEntry,
    kb: &KbHandle,
    llm: &LlmClient,
) -> Result<EntryExtraction, SpineError>;
```

Joint extraction matters because the LLM needs the entity list in
working context to assert relations between them — splitting into two
calls would either double the cost or force the second call to
re-extract entities anyway. The prompt asks for both, structured as:

```json
{
  "entities": [
    {"label": "BM25",    "kind": "method", "role": "compared", "surface": "BM25"},
    {"label": "ColBERT", "kind": "method", "role": "compared", "surface": "ColBERT"}
  ],
  "relations": [
    {"subject": "BM25", "predicate": "compared_with", "object": "ColBERT",
     "directed": false, "surface": "compare BM25 and ColBERT"}
  ]
}
```

Pipeline:

1. Prompt → JSON (reuse the `llm_parse.rs` discipline from
   `kb-extract`).
2. Canonicalise each entity against `kb_entities` by slug (create on
   miss), same as the literature assistant does today.
3. Resolve each relation's `subject`/`object` strings to the
   `entity_id` of an entity in the same extraction's entity list.
   Drop relations whose endpoints didn't make it through
   canonicalisation — better to lose the relation than to point at
   the wrong entity.
4. Persist entities to `session_entry_entities`, relations to
   `session_entry_relations`.

Off the chat path means it does not block `handle_turn`; the spine
catches up asynchronously and emits an event when ready.

Failure mode: if extraction fails, the entry is still saved, the spine
just lacks markers and edges for that entry until the next retry.
Don't propagate extraction errors into the chat surface.

### 4.2 Segmentation

Triggered after each successful extraction, or on a debounce (every N
new entries, or T seconds idle — pick whichever is cheaper to
implement first; debounce is fine).

```rust
// crates/gw-loop/src/spine/segment.rs
pub async fn resegment(
    session_id: SessionId,
    db: &PgPool,
    llm: &LlmClient,
) -> Result<Vec<Segment>, SpineError>;
```

Algorithm:

1. Fetch all entries + their entity links for the session.
2. Greedy contiguous grouping: extend a segment while the next entry
   shares ≥2 entities with the current one OR has cosine ≥ 0.6 to the
   centroid of the current segment's entity vectors.
3. Cap segments at 8 entries (avoid one giant segment dominating).
4. For each new segment, prompt the LLM with the segment's entity list
   + first/last entry text for a 1-3 word `label` and a `kind`.
5. Diff against existing segments: invalidate any whose entry range
   has shifted, write the new ones.

Cost target: one LLM call per *new* segment after debounce, not per
turn. For a typical 30-turn conversation that produces ~5 segments,
that's 5 calls amortised over the whole session.

### 4.3 New `LoopEvent` variants

```rust
pub enum LoopEvent {
    // ... existing ...
    EntryEntitiesExtracted {
        entry_id: EntryId,
        entities:  Vec<EntryEntityLink>,
        relations: Vec<EntryRelation>,
    },
    SegmentsUpdated {
        session_id: SessionId,
        segments: Vec<SegmentSnapshot>,
    },
    EntityCommitted {
        session_id: SessionId,
        entity_id: EntityId,
        committed: bool,    // false = uncommitted
    },
}
```

All three are outbound. The AG-UI adapter projects them as
`STATE_DELTA` on the spine widget (see §5.3). Existing exhaustive
match sites add no-op arms.

### 4.4 Host functions

Two new agent-facing host fns under `kb` (they belong with the entity
fns, not the UI ones — the UI just renders what the agent commits):

| Name | Signature | Effect |
|---|---|---|
| `kb.commit_entity` | `(slug: str, entry_id: str | None = None) -> None` | Set `status='committed'` on all matching links (or just the given entry's). |
| `kb.uncommit_entity` | `(slug: str) -> None` | Reverse the above. |

The agent uses these when it makes a decision ("we'll go with ColBERT
top-200") — that's a commitment to ColBERT. The user can also commit
via the sidebar (§5.4); that path goes through a `WidgetInteraction`
that the loop translates into the same DB write.

Capability gate: `kb:write` (already exists for `kb_topic` writes).

## 5. Frontend

### 5.1 Layout

The mock-up is the target. Three vertical panes between the left nav
and the right edge:

```
[ Nav ] [ ChatPane ] [ SpinePane ] [ SidebarPane ]
         flex-grow    240px         360px
```

The `App.tsx` layout currently has `ChatPane | CanvasPane`. Two changes:

- Insert `SpinePane` between chat and canvas. Always visible when
  there's at least one segment; collapsed-to-rail when none yet.
- Keep `CanvasPane` as the rightmost slot and let `SidebarPane` ride
  there as a pinned widget. The sidebar IS a canvas widget — no new
  pane primitive needed.

### 5.2 `SpinePane` component

```
frontend/src/components/SpinePane.tsx
```

Inputs (from a new `useSpine(sessionId)` selector over the surface
store):

```ts
type SpineState = {
  segments: SegmentSnapshot[];
  activeSegmentId: string | null;     // synced from scroll
  focusedSegmentId: string | null;    // user-clicked, drives sidebar
};

type SegmentSnapshot = {
  id: string;
  label: string;
  kind: 'comparison' | 'decision' | 'deep_dive' | 'construction' | 'other';
  entryFirst: string;
  entryLast: string;
  entityCount: number;
  topEntities: Array<{ id: string; label: string; kind: string; status: 'mentioned' | 'committed' }>;
  timestamp: string;                  // first entry's timestamp, for the rail label
};
```

Rendering:

- Vertical line with timestamp groupings (`10:21 AM`, `10:24 AM`, ...).
- Each segment is a marker (circle, color-coded by `kind`) + a label
  card. Clicking the card sets `focusedSegmentId` and emits a
  `WidgetInteraction` to the backend.
- Top-3 entities show as small dots beneath the segment label, colored
  by entity kind (concept / method / dataset / author / venue, matching
  the existing entity-cloud color scale).
- Faint vs solid: `opacity: 0.5` for `mentioned`, `1.0` for
  `committed`. CSS variable, not two render paths.
- Active highlight: the segment containing the entry currently
  intersecting the chat viewport's vertical center gets a subtle
  background. Use `IntersectionObserver` against entry `<div>`s, not
  scroll math.

### 5.3 Spine widget protocol

The spine is one A2UI widget per session. The agent never emits it
directly; `gw-loop` emits it once at session start and supersedes it
on `SegmentsUpdated`.

```rust
// In gw-loop, at session bootstrap:
ui_plugin.emit_widget(
    WidgetKind::A2ui,
    WidgetPayload::Inline(spine_widget(&segments)),
    None,
).with_scope("spine").with_multi_use(true).await?;
```

Payload shape (a small A2UI subtree with custom data):

```json
{
  "type": "SemanticSpine",
  "segments": [ ... SegmentSnapshot ... ],
  "actions": ["focus", "expand", "compare", "revisit"]
}
```

`SemanticSpine` is a custom A2UI component name we register in the
catalog — it falls outside json-render's stock set, which is fine
because the design-json-render-migration doc already accounts for
custom components.

Updates ride `WidgetSuperseded` events. The `STATE_DELTA` AG-UI
projection delivers them efficiently — only the changed segment IDs go
over the wire, not the whole spine.

### 5.4 `EntitySidebar` component

```
frontend/src/widgets/EntitySidebarWidget.tsx
```

Pinned to canvas via `ui.pin_to_canvas`. Listens (via the surface
store's selector) to the spine's `focusedSegmentId`. When focus
changes, it requests segment detail:

- `kb_entity(slug)` for each top entity in the segment (cached
  client-side).
- A new lightweight host fn or AG-UI endpoint
  `GET /sessions/:id/segments/:segment_id` that returns
  `{ entries, entities, relations, notes }` for the focused segment.
  This is purely a server-side join — no LLM — so it's cheap.

Tabs match the mock-up:

- **Entities** — cards with kind, summary, key properties (from
  `kb_entities.summary`), and a "Commit / Uncommit" toggle that fires
  a `WidgetInteraction`.
- **Relations** — typed edges asserted *in chat* during this segment,
  sourced from `session_entry_relations` joined to
  `session_entries`. Each row renders as `subject ←predicate→ object`
  with the surface span as a tooltip and a click-to-jump to the
  asserting entry. A "Show KB neighbors" toggle layers in the global
  `kb_entity_links` edges for the same entity set, dimmed, so users
  can see which chat-asserted relations match the global graph and
  which are novel to this conversation.
- **Notes** — free-text scratchpad. Stored in
  `session_segments.summary` initially; if users start using it
  heavily, split it out into `session_segment_notes`.

Footer:

- "Jump to first message in this segment" → scrolls chat.
- "View full graph" → opens the existing entity-cloud widget,
  pre-filtered to the segment's entities.

### 5.5 Reverse interactions

Clicking a marker offers actions. Each is one `WidgetInteraction` with
`action` set:

| Action | Synthetic prompt the loop injects |
|---|---|
| `revisit` | `"Revisit our discussion about {labels}. Summarize what we concluded and what's still open."` |
| `expand`  | `"Go deeper on {labels} — what didn't we cover?"` |
| `compare` | `"Compare {segment_entities} with the current direction ({committed_entities})."` |
| `focus`   | (no prompt — just sets sidebar focus) |

The loop builds the prompt server-side from the segment + workspace
state and feeds it as a `UserMessage`-shaped event so the agent
responds in the normal way. Doing this server-side keeps the prompts
tunable without a frontend release.

## 6. Implementation Order

Each step is independently shippable. The chat UI keeps working
through every step; the spine progressively acquires capability.

1. **Schema + extractor.** Migrations for `session_entry_entities`
   and `session_segments`. Port the literature-assistant extraction
   into `gw-loop/src/spine/extract.rs`. Run it on every entry write.
   No UI changes yet; verify by inspecting the DB.
2. **Segmentation.** `resegment()` + the cache table. Trigger on
   debounce. Add `SegmentsUpdated` `LoopEvent` and AG-UI projection.
   No UI yet; verify via SSE in the browser devtools.
3. **Spine pane (read-only).** Add `SpinePane` rendering segments
   from the surface store. Faint markers only — no commit yet.
   Active-segment scroll sync. At this point the spine is a useful
   passive minimap.
4. **Reverse interactions.** Click → focus, click → action menu →
   synthetic prompt. Sidebar listens to focus, shows entity detail.
   Spine becomes navigation rail.
5. **Commit/uncommit.** `kb.commit_entity` host fn, sidebar toggle,
   solid-marker rendering. Workspace tab in sidebar shows
   session-wide committed entities.
6. **Polish.** In-message entity highlighting (uses `span_start/end`
   from extraction), keyboard nav (j/k between segments), persist
   `focusedSegmentId` across reloads.

## 7. Tradeoffs

**Why store entry-entity links in `gw-loop` instead of a fourth
`kb_*_entity_links` table?** The link is between a *session entry*
(which `gw-loop` owns) and a *kb entity*. Putting it in `gw-kb` would
create a cross-crate FK and force `gw-kb` to know about session
entries, which it currently doesn't. The hop is symmetric — keep
ownership where the foreign key lives.

**Why is the spine a widget, not a first-class UI primitive?** Two
reasons. First, it gets persistence + replay + reconnect for free
through the existing `UiSurfaceStore`. Second, "the spine" is one
visualization of segment data; if we later want a horizontal timeline
or a graph view, those are different widgets over the same data — no
backend change. The widget protocol is exactly the right abstraction
height.

**Why not segment with topics instead of entities?** Topics
(`kb_topics`) are coarser than the spine wants. A research session
might stay inside one topic ("retrieval") for the whole conversation
while crossing through five distinct entity clusters (BM25 vs ColBERT,
recall/precision, pipeline construction, cross-encoder rerank,
evaluation). Entities give finer-grained segments. Topics are still
useful for the conversation-level summary, but not for spine markers.

**Why cache labels in `session_segments` instead of recomputing?**
LLM calls are expensive enough that recomputing on every spine render
would be visibly slow. Caching with explicit invalidation gives us
the cheap path while keeping the data path honest. The
`invalidated_at` column lets history replay show the labels that were
shown at the time, which matters for traces and bug reports.

**Why is "committed vs mentioned" a status on the link, not an
attribute on `kb_entities`?** Because commitment is *per-session*. The
same entity (BM25) might be committed in one conversation and merely
mentioned in another. Putting it on the link keeps it scoped.

## 8. Open Questions

- **Segmentation quality.** The greedy 2-shared-entities heuristic is
  almost certainly too crude. Likely needs at least one LLM-as-judge
  pass to merge or split segments after the heuristic runs. Defer
  until step 3 lands and we can eyeball real conversations.
- **Re-extraction on edit.** If a user edits a past message (does
  the chat UI even allow this today? — currently no), we'd need to
  re-extract. Punt until edits exist.
- **Cross-session spine.** Workspace tab is per-session. A user
  bouncing between conversations might want a project-level
  workspace view. Possible; needs a separate widget that aggregates
  `committed` entities across sessions. Out of scope for v1.
- **Privacy of synthetic prompts.** "Revisit" / "Compare" inject text
  into the agent context that the user didn't type. Probably fine
  because the user explicitly clicked the action, but flag this in
  trace events so the conversation log clearly shows the prompt
  origin.
- **Predicate vocabulary drift.** Free-form `predicate` plus a
  recommended set keeps us flexible early but invites a long tail of
  near-synonyms (`compared_with` vs `vs` vs `contrasted_with`). After
  step 1 has ingested real data, do a one-time clustering pass on
  observed predicates and either tighten the prompt or write a
  canonicalisation map. Don't over-design this up front.
- **A2UI custom-component story.** `SemanticSpine` and
  `EntitySidebar` are bespoke React components, not stock A2UI nodes.
  The json-render migration doc already accounts for custom
  components, but we should make sure spine + sidebar fall on the
  blessed side of "extend the catalog" rather than "fork the
  renderer."

## 9. Issues

One issue per step in §6. Each is sized to half-day to two-day work,
with a clear "done when" so they're independently mergeable.

### #1 — Spine: entry-entity extraction pipeline

**Scope:** schema + extractor only, no UI.

- [ ] Migration: `session_entry_entities` (entry_id, entity_id, surface, role, status, confidence, span_start, span_end, extracted_at).
- [ ] Migration: `session_entry_relations` (relation_id, entry_id, subject_id, object_id, predicate, directed, surface, confidence, span_start, span_end, extracted_at).
- [ ] Crate: `gw-loop/src/spine/extract.rs` with `extract_entry(entry, kb, llm) -> EntryExtraction { entities, relations }`.
- [ ] Joint prompt: returns both entities and relations; relations reference entities by label and are resolved to `entity_id` post-canonicalisation. Drop relations whose endpoints don't survive canonicalisation.
- [ ] Recommended predicate set documented in code as a const slice (free-form `TEXT` in the schema, but the prompt steers toward the recommended set).
- [ ] Wire into the entry-write path in `gw-loop/src/conversation.rs` as a `tokio::spawn`'d task — must not block `handle_turn`.
- [ ] Failure → log + continue; entries never blocked on extraction.
- [ ] Test: integration test that writes 3 entries asserting "BM25 vs ColBERT" and "ColBERT then rerank", asserts both entity links and the two relations land in DB with correct subject/object resolution.

**Done when:** running a chat session populates both `session_entry_entities` and `session_entry_relations` with sensible canonical entities and resolved typed edges; chat latency unchanged.

### #2 — Spine: segmentation + cache + outbound events

**Scope:** segment computation, no UI.

- [ ] Migration: `session_segments` (segment_id, session_id, label, kind, entry_first/last, entity_ids, summary, invalidated_at).
- [ ] `gw-loop/src/spine/segment.rs::resegment(session_id)` — greedy contiguous grouping (≥2 shared entities or cosine ≥ 0.6 to centroid), cap 8 entries/segment.
- [ ] LLM call for `label` + `kind` per *new* segment; fallback "N entities" on failure.
- [ ] Debounce trigger after each successful extraction (every 3 entries or 5s idle, whichever first).
- [ ] New `LoopEvent::EntryEntitiesExtracted` and `LoopEvent::SegmentsUpdated`; add no-op arms at existing match sites.
- [ ] AG-UI projection: both events → `STATE_DELTA` on the spine surface.

**Done when:** browser devtools shows `STATE_DELTA` events flowing during a live chat with sensible segment labels.

### #3 — Spine: read-only `SpinePane`

**Scope:** the visual rail, passive minimap behavior.

- [ ] `gw-loop` emits the spine widget at session bootstrap (`scope: "spine"`, `multi_use: true`); supersedes on `SegmentsUpdated`.
- [ ] Register `SemanticSpine` custom A2UI component in the frontend catalog.
- [ ] `frontend/src/components/SpinePane.tsx` — segment markers + label cards + top-3 entity dots, kind-colored.
- [ ] `useSpine(sessionId)` selector over the surface store.
- [ ] Active-segment scroll sync via `IntersectionObserver` on entry `<div>`s.
- [ ] Insert SpinePane into `App.tsx` between ChatPane and CanvasPane (240px fixed; collapses to rail when no segments yet).
- [ ] All markers render at `opacity: 0.5` (mentioned only — commit lands in #5).

**Done when:** opening a conversation shows a working spine that highlights the active segment as you scroll. No interactivity yet.

### #4 — Spine: reverse interactions + sidebar

**Scope:** clicks become navigation and prompts; sidebar shows segment detail.

- [ ] Marker click → `WidgetInteraction` with `action: "focus"`, sets `focusedSegmentId` in surface state.
- [ ] Action menu on second click: `revisit` / `expand` / `compare`.
- [ ] Server-side: `gw-loop` translates each non-focus action into a synthetic `UserMessage` using a templated prompt built from segment + workspace state. Tag the entry with `origin: "spine_action"` for trace clarity.
- [ ] `frontend/src/widgets/EntitySidebarWidget.tsx` — pinned to canvas via `ui.pin_to_canvas`, listens to `focusedSegmentId`.
- [ ] New endpoint `GET /sessions/:id/segments/:segment_id` → `{ entries, entities, relations, notes }`. Relations come from `session_entry_relations` joined to entries within the segment; pure server-side join, no LLM.
- [ ] Sidebar tabs: Entities, Relations, Notes. Relations tab renders `subject ←predicate→ object` rows with surface tooltip + click-to-jump-to-entry; toggle to overlay dimmed `kb_entity_links` neighbors for context.
- [ ] Footer: "Jump to first message" (scrolls chat), "View full graph" (opens existing entity-cloud filtered to segment).

**Done when:** clicking a marker focuses the sidebar and re-renders entity/relation cards; "revisit" produces a real agent turn grounded in that segment.

### #5 — Workspace: commit / uncommit

**Scope:** mentioned ↔ committed promotion, solid markers.

- [ ] Host fns in `gw-kb/src/plugin.rs`: `kb.commit_entity(slug, entry_id=None)` and `kb.uncommit_entity(slug)` under the existing `kb:write` capability.
- [ ] `LoopEvent::EntityCommitted` + AG-UI projection.
- [ ] Sidebar: per-entity Commit / Uncommit toggle → `WidgetInteraction` → server-side DB write (NOT direct host-fn call from the UI).
- [ ] CSS: `.spine-entity[data-status="committed"] { opacity: 1 }`; `mentioned` stays at 0.5.
- [ ] Sidebar Workspace tab: session-wide `DISTINCT entity_id WHERE status = 'committed'`.

**Done when:** committing an entity in the sidebar makes its dots solid across every spine segment that mentions it, and the Workspace tab lists everything committed in the session.

### #6 — Polish

**Scope:** the small things that make it feel like a primitive, not a demo.

- [ ] In-message entity highlighting using `span_start/span_end` from extraction (`<mark>` with kind-colored underline).
- [ ] Keyboard nav: `j`/`k` move between segments, `Enter` opens action menu, `Esc` clears focus.
- [ ] Persist `focusedSegmentId` per session in `localStorage` so reload restores focus.
- [ ] Trace event for synthetic spine prompts so the conversation log clearly shows agent prompts injected by spine actions.
- [ ] Empty state: "Spine appears once we've discussed a few topics" when no segments exist yet.

**Done when:** a power user can navigate a 30-turn research conversation entirely from the spine + keyboard.

---

## 10. What this doc is NOT

Not a commitment to a particular cluster-labeling model, not a design
for cross-session workspace, not a replacement for the existing
entity-cloud widget. It's the minimum design needed to land a useful
spine + sidebar that uses the gw-kb entity infrastructure already
built, and to do it in steps that each ship a runnable system.
