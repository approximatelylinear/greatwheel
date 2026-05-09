# Design: literature assistant with entity browser

**Status:** Drafted 2026-04-24 · Extended 2026-04-29 with the
Semantic Spine + Workspace (see §14) · Extended 2026-05-08 with the
KB wiki view, sessions API, multi-session manager, and transcript
backfill (see §15). Frontend sidebar plan in §16.

## 1. Goal

User asks for a survey of a research topic ("recent advances in
retrieval-augmented generation"). The agent fetches papers, extracts
named entities (authors, concepts, methods, datasets), computes
embeddings, projects to 2D via UMAP, and renders an interactive
**entity cloud**: a scatter plot where semantically related entities
cluster. Hover shows details; clicking an entity focuses it and
populates related panels (papers that reference it, co-occurring
entities, a short synthesised blurb).

The novel surface here is the **2D semantic map as a navigation
primitive**. Most research UIs are hierarchical lists; a projected
embedding lets users see structure they didn't know to ask for.

## 2. What it proves

- **Agent-computed latent structure.** Embeddings + UMAP is real
  computation performed inside the sandbox per question. Very
  different from "generate text about papers."
- **Custom rich widget on the existing protocol.** An `EntityCloud`
  widget is the most visually ambitious widget we'd add; proving the
  catalog handles it without protocol changes is itself a result.
- **Multi-axis scope.** Focus on a paper, an author, a concept, a
  year range — all expressible with existing `scope` + visibility.
  Makes the case that scope generalises beyond a single "current
  section" axis.
- **Progressive refinement.** User clicks a cluster region (even
  without a specific entity) to filter the view; agent responds with
  a region-scoped summary. Rare interaction pattern that only works
  because the agent owns UI state assembly.

## 3. Data sources

First cut: **arXiv API**. Public, no key, JSON or Atom feed,
realistic density, CS papers welcome. Cached locally to keep demos
reproducible.

## 4. Host functions

New `LiteraturePlugin`:

```python
arxiv_search(query: str, max_results: int = 50)
  -> [{arxiv_id, title, authors: [str], abstract, published, categories}]

extract_entities(text: str, types: list = ["author", "concept", "method", "dataset"])
  -> [{id, label, type, canonical_form, snippet: str}]
# LLM-based extraction via a dedicated small model, or spaCy-plus-rules
# as a cheap fallback. Canonical form dedupes "RAG" and "retrieval-
# augmented generation" to the same entity id.

embed_texts(texts: list[str]) -> [[float]]
# Sentence-transformer embeddings (local model, bundled). Vectors
# are deterministic for the same input so UMAP output is stable.

umap_project(vectors: list[list[float]], n_neighbors: int = 10, min_dist: float = 0.1)
  -> [[float, float]]
# 2D projection via umap-learn. Seeded for reproducibility.
```

All read-only from the agent's perspective. No writes outside the
widget payloads.

## 5. Agent prompt sketch

Per turn, bigger than Frankenstein's rLM loop because there's more
to compute:

1. User sends a topic query:
   - `papers = arxiv_search(topic, max_results=50)`
   - Flatten paper abstracts + title → `extract_entities` in
     batches. Dedupe by `canonical_form`. Build an entity list with
     back-references to papers.
   - `embed_texts(entity_snippets)` → one vector per entity.
   - `umap_project(vectors)` → one (x, y) per entity.
   - Emit the `EntityCloud` widget (see below) with the projected
     coords and type-coloured points.
   - Pin to canvas primary.
   - Emit a `PaperList` widget in aux canvas (Column of Cards, one
     per paper, scope=none initially).
   - FINAL: a one-paragraph narration — "I've laid out 124 entities
     across 50 papers. Concepts cluster top-left; authors bottom-
     right. Click any point to dig in."
2. User clicks an entity in the cloud:
   - `data.scope = {kind: "entity", key: entity_id}`. Server
     infers focusedScope update.
   - Aux swap: a `EntityDetailCard` (scope=entity) appears with
     the entity's label, canonical form, and the 3–5 papers that
     reference it.
   - Agent emits, on this same turn, a "Related entities" widget —
     Column of entity Cards co-occurring in the same papers.
3. User clicks a paper in the `PaperList`:
   - `data.scope = {kind: "paper", key: arxiv_id}`.
   - Aux swaps to a `PaperAbstract` card.
   - Entity cloud stays visible; points belonging to this paper
     highlight (see §7 on implementation).
4. User types a follow-up ("tell me more about RAG specifically"):
   - Agent treats this as a soft scope — emits a fresh
     `TopicSummary` widget without changing the cloud layout.

## 6. Widget catalog additions

### `EntityCloud` (the centrepiece)

2D scatter plot with pan/zoom, hover tooltips, click-to-focus. Props:

```ts
props: z.object({
  points: z.array(z.object({
    id: z.string(),
    label: z.string(),
    type: z.enum(["author", "concept", "method", "dataset", "paper"]),
    x: z.number(),
    y: z.number(),
    size: z.number().optional(),      // paper count / importance
  })),
  highlight: z.record(z.boolean()).optional(),  // ids to emphasise
})
```

**Rendering:**
- SVG for ≤500 points. Performance is fine; accessibility and CSS
  styling are easier.
- canvas / WebGL (`regl`, `deck.gl` ScatterplotLayer) for ≥1000
  points. Defer; v1 targets ≤500.
- Per-point: colour by type, area by size, label on hover.
- Pan/zoom: mouse wheel + drag. `d3-zoom` if we want the standard
  interaction; hand-rolled otherwise.
- Click emits `{action: "focus", data: {scope: {kind: "entity", key: id}}}`.

This is the only significantly custom widget in the demo.

### `EntityDetailCard`
Not really new — a Column of Text + small list of paper Cards. Can
be expressed with existing catalog entries.

### `PaperAbstract`
Column of Text nodes. Also existing catalog.

### `PaperList`
Column of Cards. Each Card has a scope-bearing click. Existing
catalog.

Net new: **one** custom widget (`EntityCloud`).

## 7. Highlight-by-scope

When a paper is focused, the cloud should visually emphasise its
entities. Implementation options:

- **Option A (naïve):** agent re-emits the cloud widget with a
  different `highlight` prop on each scope change. Supersedes the
  widget. Works, but the layout animation resets.
- **Option B (state-driven):** the cloud's `highlight` prop is a
  `{$state: "/focusedScope/paper_entities"}` binding; the agent
  updates that state bucket on scope change. Needs `set_state`.
- **Option C (client-side):** the cloud is aware of `focusedScope`
  itself, reads `state.focusedScope.paper`, and cross-references
  against a `highlightBy` prop (`{paperId: [entityId, ...]}` map).
  No agent action needed on scope change.

**Proposal: C.** The widget does client-side filtering against a
static lookup table the agent provided at emit time. Agent action
only required when the query itself changes. This is a general
pattern for "derived view" widgets and might deserve its own doc.

## 8. State shape (widget-payload based)

All data lives in the `EntityCloud` payload:

```json
{
  "type": "EntityCloud",
  "points": [...],
  "highlightBy": {
    "paper": {"arxiv_id_1": ["entity_1", "entity_2"], ...},
    "entity": {"entity_1": ["entity_2"], ...}   // co-occurrence map
  }
}
```

No server-side state beyond the usual widget store + focusedScope.

## 9. User flow

1. Type "survey retrieval-augmented generation." Wait ~10s while
   the agent fetches + embeds.
2. A 2D cloud fills the canvas: clusters of methods (RAG, HyDE,
   Self-RAG) in one region; authors (Lewis, Karpukhin) in another;
   datasets (MS MARCO, Natural Questions) in a third. Paper list in
   aux.
3. Hover: tooltip reveals full entity label and paper-count.
4. Click "RAG" point: aux swap to an EntityDetailCard showing
   canonical form + 4 papers. Cloud highlights entities
   co-occurring with RAG.
5. Click a paper: aux swap to PaperAbstract. Cloud highlights
   entities *in that paper*.
6. Pan/zoom to a cluster of concepts, click into a few of them to
   build a mental map.
7. Type "what's the difference between HyDE and Self-RAG?" — agent
   emits a comparison card using existing Column/Row/Text.

## 10. Deliberately out of scope

- **Live updates as new papers publish.** Static snapshot per query.
- **User-uploaded papers.** arXiv only.
- **Full-text search inside abstracts.** Query is the topic; search
  is implicit in arXiv's relevance.
- **Persistent state across queries.** Each topic query resets the
  cloud; no history browser.
- **Ontology disambiguation.** "RAG" might refer to the
  retrieval-augmented generation method or the rust analyzer gang.
  Relies on `extract_entities` getting it right; no explicit
  disambiguation UI.

## 11. Open questions

- **Embedding + UMAP runtime.** For 500 entities, sentence-
  transformer embeddings are ~5s on CPU, UMAP is ~3s. Total ~8s on
  top of arXiv fetch. Acceptable for a demo but the user will stare
  at a spinner. **Proposal:** show the raw paper list first (arXiv
  returns fast), then stream the cloud in when ready. Phase-5
  bracketing makes this natural.
- **Custom widget performance.** SVG for 500 points with pan/zoom is
  fine but borderline. **Proposal:** measure; upgrade to canvas only
  if needed.
- **Accessibility of 2D browsing.** Screen readers can't browse a
  scatter plot meaningfully. **Proposal:** emit a linearised
  `PaperList` + `EntityList` as a parallel surface so keyboard /
  screen-reader users have a path that doesn't require the cloud.
  Not demoed visually but present for correctness.
- **Entity type taxonomy.** `author | concept | method | dataset |
  paper` is ad-hoc. Real surveys might want `task`, `metric`,
  `institution`. **Proposal:** start with the 5; make `type` a
  free string in the widget schema so additions don't break things.

## 12. Acceptance

1. Typing a topic produces a cloud with ≥50 labelled entities within
   15 seconds.
2. Clicking an entity focuses it; aux canvas updates; ≥3 related
   entities highlight in the cloud.
3. Pan + zoom work smoothly on a modern laptop (60fps interaction).
4. Screen-reader linear path (PaperList + EntityList) is navigable
   via keyboard without requiring cloud interaction.
5. Running the same query twice produces the same cloud layout (UMAP
   is seeded).
6. No network beyond arXiv (no OpenAI / Anthropic calls for
   embeddings — bundled model).

## 13. Scope estimate

5–7 days including: the plugin (arxiv search, entity extraction
pipeline, embedding + UMAP), bundling the embedding model (and
weighing its size carefully against repo bloat — probably a separate
download step for first run), the `EntityCloud` widget with pan/zoom
and hover, tuning the agent prompt to compose the full view, plus
accessibility fallback.

Biggest engineering risks: entity extraction quality (bad entities →
messy cloud), UMAP reproducibility (seed it; test it), and the
bundled-model question (spaCy vs LLM-prompted extraction is a fork
in the road worth resolving in its own prototype).

## 14. Extension: Semantic Spine + Workspace (2026-04-29)

The original goal — entity cloud as 2D navigation primitive — held
up well, but once users started running long exploration sessions
(typed query → 30+ pin clicks → pivot → more pins) two limitations
showed up that the cloud couldn't address:

- **Temporal vs spatial.** The cloud shows *what's related to what*
  across all papers, but not *what we covered when*. As a session
  grew, scrolling chat became the only way to find earlier branches,
  and scroll-search by label is fragile.
- **No curated reading list.** Pin acks in chat accumulate but don't
  consolidate into anything. The user needs to mark "this thread is
  worth coming back to" without leaving the conversation.

Both motivated the **Semantic Spine** — a vertical rail next to chat
with one marker per typed user turn (widget-event chains absorbed
into the parent), and an entity-rich sidebar pinned to the canvas
column. Full design lives in
[`design-semantic-spine.md`](design-semantic-spine.md). What follows
is how it slots into the literature-assistant surface.

### 14.1 What it adds (briefly)

- **Per-entry entity attribution.** Every assistant narration (the
  resolved `FINAL("...")` prose, persisted as a separate
  `EntryType::AssistantNarration` so the spine reads from prose, not
  Python source) gets typed-entity extraction off the chat path.
  Rows land in `session_entry_entities` keyed to the same
  `kb_entities` table the EntityCloud already uses. **Same canonical
  ids across both surfaces.**
- **Turn-based segmentation.** Each typed user message opens a
  segment; widget-event-driven turns (point clicks, pin acks) chain
  into the parent so a click-driven exploration collapses under the
  question that started it. Persisted in `session_segments`.
- **Timeline rail.** Markers absolute-positioned at the y-offset of
  each segment's first chat row; the rail's scroll mirrors the chat
  pane's so markers track the user's reading position. Distinct
  visual states for *focused* (clicked, sidebar open) vs *current*
  (in viewport, scroll-driven).
- **Canvas-pinned sidebar.** Click a marker → sidebar appears in
  the canvas column above the EntityCloud (capped at 50% height so
  the cloud stays usable). Tabs: Entities / Relations / Notes.
  Action row at top: Revisit / Go deeper / Compare — each fires a
  synthetic prompt + new turn through the standard `WidgetInteraction`
  path. Footer has Jump-to-message + Close.
- **Workspace.** ★ Save toggle on the sidebar header; ★ Workspace
  button in the app header opens a drawer listing every saved
  segment (label, top entities, summary, Open / Jump). Survives
  resegment churn — invalidated-but-saved rows keep showing with a
  "(superseded)" tag.
- **Entity ↔ chat coupling.** When a segment is focused, every
  surface form of its entities (label + aliases) gets wrapped in
  `<mark>` inside the segment's chat rows; a left-border tick marks
  the segment range. Clicking an entity card narrows the highlight
  to that single entity across the *whole conversation* and scrolls
  to the first match.
- **Keyboard nav.** `↑` / `↓` move between markers (auto-scrolls
  chat to the new segment); `Esc` dismisses the focused sidebar.
  Skipped while typing into an input so the message composer stays
  usable.

### 14.2 How it complements the EntityCloud

The cloud remains the canvas centerpiece. Spine and cloud are
complementary projections of the same `kb_entities` graph:

| Surface          | Axis              | Anchored to                       |
|------------------|-------------------|-----------------------------------|
| `EntityCloud`    | spatial / semantic| UMAP coords across the corpus     |
| Semantic Spine   | temporal          | session entries in chronological order |

Same entities, two axes. Clicking an entity in the spine sidebar
already highlights its mentions in the chat. The mirror direction —
clicking a cloud point and having the spine focus the segment(s)
that mentioned it — isn't wired yet but is a natural follow-up
(see §14.5).

### 14.3 Schema additions since v1

| Table / migration                        | Purpose                                                     |
|------------------------------------------|-------------------------------------------------------------|
| `EntryType::AssistantNarration` (Rust)   | Resolved FINAL prose stored alongside the raw assistant entry so the extractor sees what the user saw. |
| `014_spine_entry_extraction.sql`         | `session_entry_entities`, `session_entry_relations`.        |
| `015_session_segments.sql`               | Segment cache (label, kind, range, entity_ids, invalidated_at). |
| `016_session_segment_commits.sql`        | `committed_at` flag → drives the workspace listing.         |

The cloud's data path is unchanged — it still reads `kb_entities` /
`kb_chunk_entity_links` directly. The spine's data path joins the
same `kb_entities` rows from the *session* side
(`session_entry_entities → kb_entities`).

### 14.4 Out-of-scope updates

§10's "Persistent state across queries. Each topic query resets the
cloud; no history browser" still holds *for the cloud*. The
workspace adds session-scoped persistence over a different axis —
saved segments survive new typed queries within the session. **Cross-
session persistence remains a non-goal**: each `?session=<uuid>` URL
is its own conversation; no global library of saved segments yet.

### 14.5 Deferred

Items left on the spine punch list, ordered by user-visible impact:

- **Cloud → spine linkage.** Click a cloud point → focus the
  segment(s) where that entity was discussed. Closes the only
  remaining one-way coupling between the two surfaces.
- **Coordinator-race serialization.** Multiple flushes can spawn
  concurrent `run_resegment` coordinators; `Mutex` per session would
  serialize cleanly. Empirically OK because Postgres serializes the
  writes and `supersede` is idempotent.
- **Label staleness refresh.** A segment's label is set on first
  creation; as it grows entity counts the label can drift. Cheap fix
  is re-prompt when entity_ids changes by ≥N.
- **GC for invalidated segment rows.** Accumulate forever today.

## 15. Extension: KB wiki + multi-user sessions (2026-05-08)

Four shipped in one push, on branch `browsecomp/apr19`. They turn
the demo from "single-shot canvas" into a multi-user app you can
leave and come back to.

### 15.1 What landed

#### a) `KbDocWiki` — Wikipedia-style view per ingested paper

A new long-form drawer surface that renders one KB source as a real
doc page: serif title + author byline, infobox right rail
(URL, ingest date, source format, short id), TOC left rail derived
from chunk `heading_path`, body in the middle, mentioned entities
grouped by kind + referenced topics as click-through chips.

  - Backend: new `wiki_slot` field on `UiSurface` (alongside
    `canvas_slot` / `canvas_aux_slot`), `WikiPinned` /
    `WikiUnpinned` notifications, `pin_to_wiki(widget_id)` host fn,
    `clear_wiki_slot(session_id)`. AG-UI codec mirrors `wikiSlot`
    in canonical state and emits `replace /wikiSlot ...` patches on
    pin/unpin. Close button fires `action: "close_wiki"` which the
    AG-UI adapter short-circuits like `focus` — clears the slot
    without involving the agent.
  - `kb_get_wiki(source_ref)` host fn (new `crates/gw-kb/src/wiki.rs`
    module) builds the WikiDoc payload: source meta + TOC +
    chunk-grouped sections + entities + topics. `source_ref`
    accepts UUID, UUID prefix (≥4 chars), full source URL, or
    arXiv id (`2504.13684`). The arXiv path matches both
    `metadata->>'arxiv_id'` and the canonical
    `https://arxiv.org/abs/<id>` URL so older rows still resolve.
  - Frontend: `KbDocWiki` widget in the json-render catalog (with
    matching `translate.ts` case + `registry.tsx` render),
    `frontend/src/widgets/KbDocWiki.tsx` for the layout,
    `frontend/src/components/WikiPane.tsx` for the slide-in drawer
    (80vw / max 1280px, backdrop, Esc-to-close), CSS in
    `frontend/src/styles.css` keyed `kb-wiki-*`. Entity / topic chip
    clicks fire `interact` with action `open_kb_entity` /
    `open_kb_topic`.
  - Trigger paths in the agent: a "📖 Open as wiki" button on the
    paper detail card (the existing turn-2 drill-down) and NL
    recognition of "show wiki for X" / "open the wiki". Single-
    iteration handler in `SYSTEM_PROMPT` runs
    `kb_get_wiki(arxiv_id)` → `emit_widget(KbDocWiki, multi_use=True)`
    → `pin_to_wiki(widget_id)`. **`multi_use=True` is required** so
    chip clicks don't terminate the widget on first interaction.

#### b) Sessions API — Phase 1 of the multi-user plan

Migration `017_session_titles.sql` adds `title` / `summary` /
`archived_at` to `sessions` plus an index on
`(user_id, archived_at, last_active_at DESC)` for the sidebar's hot
path. New `crates/gw-ui/src/sessions_api.rs` exposes:

  - `POST /users` — idempotent upsert. Body `{user_id?, name?,
    email?}`. Lets the frontend stamp a localStorage-generated UUID
    into PG on first load.
  - `GET /users/{uid}/sessions?include_archived=0` — newest first,
    bounded to 200 rows. Carries `(session_id, title, summary,
    created_at, last_active_at, archived_at, message_count)`.
  - `POST /users/{uid}/sessions` — body `{title?}`. Returns
    `{session_id}`. 404s when the user row doesn't exist (the
    frontend POSTs `/users` first).
  - `PATCH /sessions/{sid}` — body `{title?, archived?}`. Used by
    the auto-titler and the archive button.

Mounted by `literature_assistant` next to the spine routes when KB
is configured. Takes a `SessionsApiConfig{default_org_id,
default_agent_id}` so users + sessions created here pin to the same
FK chain `flush_to_pg` expects (`LIT_ORG_ID` / `LIT_AGENT_ID` for
the demo). No auth: trust-the-client `?user=<uuid>` model — real
auth is layered on later without changing this surface.

#### c) `LitSessionManager` — multi-session per process

`literature_assistant` is no longer single-session-per-process. New
`LitSessionManager` holds `RwLock<HashMap<SessionId, SessionEntry>>`;
an axum `from_fn_with_state` middleware on `/sessions/{uuid}/*`
calls `manager.ensure(session_id)` before each route handler runs.
First reference to an unknown id triggers a lazy spawn:

  - `lookup_session_user(pg, sid)` reads the existing row's
    `user_id` (so sessions created via the API inherit their owner);
    falls back to `LIT_USER_ID` for URL-pasted ids that don't yet
    have a row.
  - `ensure_literature_session(pg, sid, user_id)` is now idempotent
    and parameterised (was hard-coded to `LIT_USER_ID`). Seeds the
    full FK chain on every spawn — repeat calls are no-ops via
    `ON CONFLICT DO NOTHING`.
  - `spawn_session_loop(session_id, &deps)` builds the per-session
    machinery (channels, `register_session`, tap forwarder,
    `ConversationBridge`, `ReplAgent` with `gw_session_id` pre-set,
    `SessionTree::with_pg`, `ConversationLoop` + optional
    `SpineExtractor`) and spawns a dedicated `std::thread` running
    a fresh `tokio::Runtime` to host `conv_loop.run`.
  - `SessionDeps` struct bundles the process-global handles
    (`Arc<AgUiAdapter>`, `Option<Arc<KbStores>>`, `OllamaClient`
    by clone, `Arc<HostFnRouter>`, `Arc<LoopConfigTemplate>`).

Plugin caches (`paper_cache` / `vector_cache` / `text_cache` on
`LiteraturePlugin`) stay process-global and shared across sessions —
they're keyed by arxiv_id, so two users searching "GNN chemistry"
correctly see the same ingested papers.

Backward compat: `main()` still seeds one session at boot via
`manager.ensure(...)` and prints its URL. The existing
`?session=<uuid>` flow is untouched. No idle eviction in v1 —
sessions live for the process lifetime.

#### d) Chat transcript backfill on resume

The AG-UI `STATE_SNAPSHOT` rehydrates widgets on `/stream`
resubscribe but doesn't replay prior chat messages — those live in
`session_entries` (PG) and the frontend's `useSessionStore` starts
empty on each page load. So reloading a session brought back the
canvas + follow-up widgets while the user/assistant history above
them stayed blank.

  - New `GET /sessions/{sid}/transcript` handler in
    `literature_assistant.rs` filters `session_entries` to the
    chat-visible rows (`user_message` + `assistant_narration` —
    deliberately *not* `assistant_message`, which is mostly Python)
    and returns them oldest-first as `[{entry_id, role, content,
    created_at}]`.
  - `frontend/src/api/client.ts` exports `fetchTranscript`. A new
    `useEffect` in `App.tsx` runs it alongside the SSE-stream open
    and dispatches `hydrate-transcript` to seed `useSessionStore`.
    Code blocks, host calls, and snapshots stay out of the chat UI
    by design — a wider endpoint can serve the debug pane later.

### 15.2 Schema changes since §14

```
-- 017_session_titles.sql
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS title TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS summary TEXT;
ALTER TABLE sessions ADD COLUMN IF NOT EXISTS archived_at TIMESTAMPTZ;
CREATE INDEX IF NOT EXISTS sessions_user_active
    ON sessions (user_id, archived_at, last_active_at DESC);
```

(Existing `users`, `orgs`, `agent_defs`, `sessions`, `session_entries`
schemas were already there from §14 / earlier. No migration touches
were needed for the wiki view — `kb_sources.metadata->>'arxiv_id'`
already exists.)

### 15.3 New host functions

  - `kb_get_wiki(source_ref) -> dict` — under `kb.read` capability.
  - `pin_to_wiki(widget_id)` — under `ui:write`.

Updated tool_permissions list in `literature_assistant.rs` adds
both alongside the existing `pin_to_canvas` / `pin_below_canvas` /
`emit_widget` / etc.

### 15.4 What §15 deliberately leaves out

  - **Cluster wiki** (a global `KbClusterWiki` rendering N related
    sources together). Slot + drawer pattern reuses cleanly when we
    add it.
  - **REPL state persistence on resume.** The conversation loop's
    `SessionTree::with_pg` reloads prior entries into LLM context,
    and the transcript backfill restores chat UI; but Python REPL
    variables and the `LiteraturePlugin` in-memory caches still
    rebuild lazily. Acceptable trade for v1 — the persistent KB
    means most "I already saw this paper" facts come back via
    `get_paper(...)` on demand.
  - **REST-level cross-session activity.** Workspace still scopes
    to one session (committed segments per session). The user-level
    "all my saved segments / papers / queries" view (Phase 4 of the
    data plan) is deferred until the sidebar lands.
  - **Auth.** `?user=<uuid>` is trust-the-client.

## 16. Plan: frontend sidebar (next session)

What the rest of Phase 1 looks like — written so a fresh Claude
session can pick this up without reading the conversation log.

### 16.1 Goal

Replace the `?session=<uuid>` paste-the-URL flow with a real
left-rail sidebar that lists the user's prior sessions, lets them
switch between them, archive ones they don't want, and create new
ones with a single click. Plus a minimal user-identity hook so the
sidebar knows which sessions to fetch.

The backend that supports this already exists (§15.1.b, §15.1.c).
This is a frontend-only piece.

### 16.2 What's already there to lean on

  - **APIs.** `POST /users`, `GET /users/{uid}/sessions`,
    `POST /users/{uid}/sessions`, `PATCH /sessions/{sid}` — see
    §15.1.b for shapes. Lazy-spawn middleware (§15.1.c) means
    *any* session_id the frontend navigates to will cause its
    `ConversationLoop` to spin up on first `/stream` connect. The
    sidebar doesn't need to "ask the binary to spawn" — clicking a
    session and updating the URL is enough.
  - **Drawer pattern.** `frontend/src/components/WorkspaceDrawer.tsx`
    is a working example of an overlay panel that fetches via the
    API client, renders a list, and dispatches actions on click.
    Mirror its skeleton for the session list interactions
    (selection, archive, rename).
  - **Layout grid.** `App.tsx` already has a layout switch
    (`chat-primary` / `canvas-primary`). The sidebar adds a fixed
    left column that lives outside both, as a sibling of `app-main`.
  - **Identity placeholder.** The "no session configured" empty
    state in `App.tsx:81` becomes the bootstrap path for users with
    no localStorage UUID yet.

### 16.3 Decisions already made (per the planning conversation)

  - **(A)** Multi-session-per-process refactor — done in §15.1.c.
  - **(a)** User identity = silent UUID auto-mint into localStorage,
    no login UI. Optional display name set via header click later.
  - **First-6-words auto-titler** — derive a session title from the
    first user message; LLM-titler optional follow-up if titles
    look bad.

### 16.4 Implementation plan, file-level

#### 16.4.1 User identity hook

New: `frontend/src/api/users.ts` (api wrappers for `POST /users` and
the sessions endpoints) + a `useUserId()` hook that reads in this
order:

  1. `?user=<uuid>` URL param (so the user can paste a profile
     URL).
  2. `localStorage.gw_user_id`.
  3. Generate fresh UUID, persist to localStorage, fire
     `POST /users` once to create the row server-side.

Returns `{userId: string, displayName: string | null,
setDisplayName(name)}`. The display name is read from the response
of `POST /users` (server defaults to `user-<8hex>`); a future
header click can `PATCH` it.

#### 16.4.2 Session routing

`App.tsx`'s `resolveSessionId` becomes `useSessionRouting()` that
returns `{userId, sessionId, switchSession(id)}`. URL becomes
`?user=<u>&session=<s>`. `switchSession` updates the URL via
`history.pushState` (no full reload — the existing SSE / transcript
effects re-fire on `sessionId` change, which they already key on).

When `userId` is set but `sessionId` is null, the app should
auto-fetch `GET /users/{uid}/sessions` and either:

  - Land on the most recently active session (`last_active_at` is
    descending in the response).
  - Or, if there are no sessions yet, auto-create one via
    `POST /users/{uid}/sessions` and navigate to it.

This replaces today's "no session configured" placeholder.

#### 16.4.3 `SessionSidebar` component

New: `frontend/src/components/SessionSidebar.tsx`. Layout, top to
bottom:

  - **Header:** display name (truncated, hover for full uid). Click
    to rename via PATCH. Default names are `user-<8hex>`.
  - **`+ New session` button:** primary CTA. Calls
    `POST /users/{uid}/sessions` with empty body, then
    `switchSession(new_id)`.
  - **List:** scrollable, each row: title (or first-6-words
    fallback for untitled), relative timestamp ("3m ago", "yesterday"),
    hover-reveal archive icon. Active session has a left-border
    accent. Click → `switchSession(id)`.
  - **Empty state:** "No sessions yet — start one." with the
    `+ New session` button only.
  - **Archived toggle (optional v1):** a "Show archived" checkbox
    that flips the fetch to `?include_archived=1`. Archived rows
    render dimmer with a small "(archived)" badge.

State management: the sidebar manages its own list via local
`useState` + `fetchSessionsForUser`. Refetches when:

  - Component mounts.
  - `switchSession` is called (so a fresh session that just got
    `last_active_at` bumped re-sorts to the top).
  - `archive` / `rename` actions complete.

A small `bumpReloadKey()` pattern (see `WorkspaceDrawer`) avoids
state coupling with the rest of the app.

#### 16.4.4 Layout shift

`App.tsx`'s top-level grid currently has `app-header` /
`app-main` / `app-footer`. Add the sidebar as a left rail:

```
┌─────────────────────────────────────┐
│ app-header (existing)               │
├──────────┬──────────────────────────┤
│          │                          │
│ sidebar  │ app-main (existing grid) │
│ (240px)  │                          │
│          │                          │
├──────────┴──────────────────────────┤
│ app-footer (existing input)         │
└─────────────────────────────────────┘
```

Sidebar width is fixed at first (240px); a drag splitter could
follow `DragSplitter` (frontend/src/components/) if desired.

#### 16.4.5 Auto-titler

Server-side, in `LitSessionManager::ensure` (or right after the
first user message lands — `ConversationLoop` already emits the
`UserMessageAnchor` event). The cheapest hook: when the first
`session_entries` row of `entry_type = 'user_message'` is inserted
and the `sessions.title IS NULL`, run:

```rust
let title = msg.split_whitespace().take(6).collect::<Vec<_>>().join(" ");
sqlx::query("UPDATE sessions SET title = $1 WHERE id = $2 AND title IS NULL")
    .bind(title)
    .bind(session_id)
    .execute(pg)
    .await?;
```

Where to wire it: the easiest spot is the tap-forwarder spawned by
`spawn_session_loop` — it already sees every `LoopEvent`. When the
event is `UserMessage(content)` AND the title check is needed, fire
the update. Alternatively defer this to the LLM-titler follow-up.

For v1 the sidebar should just show the title field if set,
otherwise fall back to "Untitled session" or the first 30 chars of
the most-recent user message — this keeps the UI working even if
the auto-titler is shipped later.

### 16.5 Acceptance criteria

  - Open `localhost:5173` with no `?user` / `?session` — auto-mints
    a user, creates a first session, lands on it. Existing search
    flow works.
  - Click `+ New session` — new session appears at top of sidebar,
    URL updates to `?user=...&session=<new>`, chat is empty,
    canvas is empty. EntityCloud renders fresh on next search.
  - Click a different session in the sidebar — chat history,
    EntityCloud, and any pinned wiki rehydrate (transcript backfill
    + STATE_SNAPSHOT do this already; just verify no regressions).
  - Archive a session — disappears from default listing, shows up
    under "Show archived". Unarchive flips it back.
  - Open the same user in two browser windows — both see the same
    session list. Archiving in one updates the other on next
    refresh / sidebar reload.

### 16.6 Out of scope for this slice

  - **Real auth.** Still trust-the-client. A login flow can wrap
    `useUserId` later without touching the sidebar.
  - **Cross-user sharing of sessions.** Each session has one owner;
    no "share" affordance.
  - **Cross-session workspace.** Workspace stays per-session for
    now; the user-level "everything I've saved" view is Phase 4 of
    the data plan.
  - **Search within the sidebar.** A growing list eventually wants
    a filter input; punt until it's needed.

### 16.7 Files to touch (cheat sheet)

```
NEW   frontend/src/api/users.ts
NEW   frontend/src/components/SessionSidebar.tsx
NEW   frontend/src/components/UserBadge.tsx        (optional split)
EDIT  frontend/src/App.tsx                          (layout + routing)
EDIT  frontend/src/main.tsx                         (only if URL parsing
                                                     moves there — likely
                                                     stays in App.tsx)
EDIT  frontend/src/styles.css                       (sidebar CSS)
EDIT  frontend/src/api/client.ts                    (re-export TranscriptEntry
                                                     etc; or split into users.ts)
```

If the auto-titler is bundled in:

```
EDIT  crates/gw-ui/examples/literature_assistant.rs (tap forwarder hook
                                                     in spawn_session_loop)
```

### 16.8 Likely first commit

Single feature commit:
`feat(frontend): user-aware session sidebar (Phase 1 finish)`
covering identity hook + routing + sidebar component + auto-create-
on-first-load + layout. Auto-titler can be a follow-up if it grows.

### 16.9 Reading order for a fresh Claude session

  1. `docs/design-demo-literature-assistant.md` §15 (this document,
     the section above) for context on what already shipped.
  2. `crates/gw-ui/src/sessions_api.rs` for the API shapes.
  3. `frontend/src/components/WorkspaceDrawer.tsx` as a working
     example of fetch + render + actions.
  4. `frontend/src/App.tsx` `resolveSessionId` (~line 30) and the
     "no session configured" empty state (~line 81) for the routing
     touchpoints.
  5. `frontend/src/store/session.ts` `hydrate-transcript` action
     for how the chat seeds on session change — confirms
     `sessionId` is the right re-render key.
