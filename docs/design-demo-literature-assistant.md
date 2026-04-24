# Design: literature assistant with entity browser

**Status:** Drafted 2026-04-24.

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
