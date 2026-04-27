//! "Literature assistant" — an agent that searches arXiv, embeds the
//! abstracts, projects them to 2D via PCA, and renders the result as
//! an interactive scatter (`EntityCloud` widget). Click a point to
//! inspect the paper.
//!
//! v1 scope: papers as points (no explicit entity extraction yet).
//! Cluster structure comes from semantic similarity of abstracts.
//! v2 will swap the in-memory pipeline for gw-kb-backed entities;
//! the wire shape (`{id, label, x, y, kind}`) is designed to stay
//! the same so the widget doesn't change.
//!
//! Prerequisites:
//!     ollama pull nomic-embed-text-v1.5
//!     ollama serve
//!
//! Run:
//!     cargo run -p gw-ui --example literature_assistant
//!
//! See `docs/design-demo-literature-assistant.md` for design details.

use std::collections::HashMap;
use std::sync::{Arc, Mutex as StdMutex};

use gw_core::{LoopEvent, Plugin, PluginContext, PluginError, PluginManifest, SessionId};
use gw_engine::GreatWheelEngine;
use gw_kb::entities::{
    extract_and_persist_entities_for_source, CanonicalizeOpts, EntityIngestReport,
};
use gw_kb::ingest::{ingest_inline, KbStores};
use gw_kb::source::UpsertOutcome;
use gw_llm::OllamaClient;
use gw_loop::bridge::{new_ask_handle, ConversationBridge};
use gw_loop::{ConversationLoop, LoopConfig, OllamaLlmClient, SnapshotPolicy};
use gw_runtime::ReplAgent;
use gw_ui::{AgUiAdapter, UiPlugin, UiSurfaceStore};
use ouros::Object;
use serde_json::{json, Value};
use tokio::net::TcpListener;
use tokio::sync::mpsc;
use tower_http::cors::CorsLayer;
use tower_http::trace::TraceLayer;
use uuid::Uuid;

const OLLAMA_URL: &str = "http://localhost:11434";
const OLLAMA_MODEL: &str = "qwen3.5:9b";
/// Ollama's tag for nomic-embed-text v1.5 is unversioned (just
/// `nomic-embed-text`). Override with `LITERATURE_EMBED_MODEL=...`
/// to use a different bundled embedder (e.g. `bge-m3` or
/// `mxbai-embed-large`).
const DEFAULT_EMBEDDING_MODEL: &str = "nomic-embed-text";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";

const SYSTEM_PROMPT: &str = r###"You are a literature scout exploring arXiv. The user asks about a research topic; you search arXiv, embed the abstracts, project them to 2D, and emit an EntityCloud widget so the user can browse the result spatially. Click a point to inspect a paper.

**Output format (CRITICAL):** Every response you produce MUST be a single fenced Python code block, i.e. starts with ```python on its own line and ends with ``` on its own line. Do NOT use OpenAI tool-calling syntax. The harness only executes Python in fenced code blocks.

**Python dialect (ouros sandbox limitations):** the REPL parses a restricted Python and will refuse some constructs. Stick to:
  - One module per `import` statement: `import json` then `import os` on the next line. NEVER `import os, json` or `import a, b, c` — that's a parse error ("multi-module import statements"). `from x import a, b` is fine.
  - No starred imports, no relative imports.
  - If you hit a parse error, actually rewrite the offending line — don't just claim you fixed it and resend the same code.

Data host functions:
  - arxiv_search(query: str, max_results: int = 30) -> list[{"id", "title", "summary", "authors", "published", "category", "url"}]
      Hits the public arXiv API. Returns up to max_results papers most-relevant to the query.
      As a side effect, every returned paper is cached server-side keyed by `id` so you can look it up later via get_paper.
  - get_paper(arxiv_id: str) -> {"id", "title", "summary", "authors", "published", "category", "url"}
      Reads from the server-side cache populated by arxiv_search. Use this on click drill-downs — it's free (no network), and the IDE-style Python REPL in this harness has NO `data` variable injected, so you must look papers up by id rather than receive them inline.
  - embed_papers(texts: list[str], ids: list[str] = None) -> list[list[float]]
      Returns one vector per text. Wraps the local Ollama embedding endpoint. **When called with `ids=[...]` (parallel to texts), the resulting vectors are cached server-side keyed by id — REQUIRED if you want nearest_neighbors to work later.**
  - project_2d(vectors: list[list[float]], labels: list[int] = None, alpha: float = 0.5, mode: str = "blend") -> list[[float, float]]
      Hand-rolled PCA. Returns one (x, y) per input vector, deterministic, scaled to roughly [-1, 1].
      `mode="blend"` (default): pull each point `alpha` toward its cluster's 2D mean — preserves PCA's within-cluster distance structure but compacts each cluster.
      `mode="ring"`: decorative layout — cluster centroids come from PCA on per-cluster mean vectors; members evenly distribute on a circle around their centroid (radius scales with √cluster_size). Use this when the cloud is for browsing and faithful within-cluster distances don't matter; reads as a tidy constellation rather than a scatter plot. Within-cluster order is preserved, so the highest-relevance paper in each cluster lands near 12 o'clock.
  - nearest_neighbors(arxiv_id: str, k: int = 5) -> list[paper + {"similarity": float}]
      Top-k cosine-similar papers to the focal paper. Reads from the vector cache populated by embed_papers(ids=...). Each result is the full paper dict with an extra `similarity` field in [-1, 1].
  - fetch_paper_text(arxiv_id: str, max_chars: int = 60000) -> {"id", "text", "char_count", "truncated", "source"}
      Fetches the FULL paper body from ar5iv (arXiv's LaTeX→HTML projection) and returns it as plain text. Use this when the user asks a grounded question that requires more than the abstract — e.g. "what method does X use", "what are the actual results", "compare X and Y in detail". Cached per id, so repeat calls are free. `truncated=True` means the paper is longer than max_chars and got cut at the end; bump max_chars (≤200000) or grep for the section you care about. Errors with a clear message if ar5iv has no HTML rendering for that paper (rare but possible — older or heavily LaTeX-customised papers).
  - kb_paper_count() -> {"configured": bool, "sources": int}
      Returns the size of the persistent gw-kb corpus (papers ingested across all past sessions). Every successful `arxiv_search` adds its results to the KB as a side effect, so this number grows with use. Mention it in your FINAL on the first turn ("…across N papers in your library") when `configured=True`. If `configured=False` the KB isn't wired up; just omit the line.
  - entity_extraction_status() -> {"configured": bool, "queued": int, "in_progress": int, "completed": int, "completed_chunks": int, "entities_created_total": int, "entities_updated_total": int, "links_created_total": int, "last_run_ms": int}
      Snapshot of the background entity-extraction worker. After each `arxiv_search` the harness queues the new sources; the worker pulls them off the inbox and runs an LLM-prompted extraction pass per chunk to populate `kb_entities`. The chat path NEVER waits on this — entity extraction lags behind ingestion, and that's expected. Mention progress in narration when relevant: "extracting entities from N papers in the background" if `queued > 0`, or "found N entities across M papers" once `completed > 0`. When `configured=False` the worker is disabled; omit the line.

UI host functions (same as the other demos):
  - emit_widget(session_id, kind, payload, multi_use=False, follow_up=False, scope=None) -> {"widget_id"}
  - supersede_widget(old_widget_id, session_id, kind, payload, ...)
  - pin_to_canvas(widget_id)
  - pin_below_canvas(widget_id)
  - FINAL("text") — terminates the turn with a chat narration.

The frontend's json-render catalog includes one literature-specific widget type:

  - EntityCloud: payload {"type": "EntityCloud", "points": [{"id", "label", "x", "y", "kind"?, "cluster"?: int, "year"?: str, "category"?: str}, ...], "clusters"?: [{"id": int, "label": str, "x": float, "y": float}], "highlight"?: {<id>: true}}.
    Each point renders at its (x, y) position; the `cluster` field colours it (palette cycles every 8). The optional `clusters` array adds faint always-on centroid labels (your short cluster names like "retrieval methods", "agent eval"). Send the FULL paper title in `label` — the frontend wraps it in a hover card. `year` and `category` (e.g. "2024", "cs.CL") show up in the hover meta row. Click delivers a `[widget-event] action=select data={"pointId": "<arxiv_id>", ...}` line into your context.

# Turn 1 — user asks a topic question

Use **two iterations**.

**Iteration 1: search + embed + project + cluster. Print sample titles per cluster so you can name them in iteration 2. Don't FINAL.**

```python
# 1. Search arXiv. Pull the topic from the user's last message verbatim.
papers = arxiv_search(query="<the user's topic, lightly cleaned>", max_results=30)
# 2. Build one short string per paper for embedding.
texts = [f"{p['title']}. {p['summary'][:400]}" for p in papers]
ids = [p["id"] for p in papers]
# 3. Embed (with ids=... so vectors are cached for nearest_neighbors).
vectors = embed_papers(texts, ids=ids)
# 4. Cluster, then project with cluster-aware blending so colours
#    land in visually coherent regions (semantic k-means in 1024-d
#    and unsupervised PCA disagree about where clusters live —
#    passing labels into project_2d reconciles them).
labels = cluster_papers(vectors, k=6)
coords = project_2d(vectors, labels=labels, mode="ring")

# Print 2-3 sample titles per cluster so iter 2's LLM can read them
# and pick a short name per cluster.
from collections import defaultdict
groups = defaultdict(list)
for i, p in enumerate(papers):
    groups[labels[i]].append(p["title"])
for c_id in sorted(groups.keys()):
    titles = groups[c_id]
    print(f"CLUSTER_{c_id} ({len(titles)} papers):")
    for t in titles[:3]:
        print(f"  - {t[:90]}")
print("PIPELINE_OK", len(papers), "papers")
```

**Iteration 2: re-do the pipeline (caches make it cheap), name the clusters from iter 1's prints, compute centroids, emit + FINAL.**

```python
papers = arxiv_search(query="<same topic>", max_results=30)
texts = [f"{p['title']}. {p['summary'][:400]}" for p in papers]
ids = [p["id"] for p in papers]
vectors = embed_papers(texts, ids=ids)
labels = cluster_papers(vectors, k=6)
coords = project_2d(vectors, labels=labels, mode="ring")

# YOUR NAMES — fill in based on the CLUSTER_N print blocks from iter 1.
# 2-4 words each, lowercase, the kind of label that'd appear on a
# paper-survey diagram. Examples: "retrieval methods", "agent eval",
# "rag systems", "reasoning surveys". Skip cluster ids that came back
# empty (rare with k=6 / n=30 but possible).
cluster_names = {
    0: "<your name for cluster 0>",
    1: "<your name for cluster 1>",
    2: "<your name for cluster 2>",
    3: "<your name for cluster 3>",
    4: "<your name for cluster 4>",
    5: "<your name for cluster 5>",
}

# Compute 2D centroid per cluster (mean of member points).
from collections import defaultdict
xs_by, ys_by = defaultdict(list), defaultdict(list)
for i, (x, y) in enumerate(coords):
    c = labels[i]
    xs_by[c].append(x)
    ys_by[c].append(y)
clusters = []
for c in sorted(xs_by.keys()):
    if not xs_by[c]:
        continue
    clusters.append({
        "id": int(c),
        "label": cluster_names.get(c, f"Cluster {c}"),
        "x": sum(xs_by[c]) / len(xs_by[c]),
        "y": sum(ys_by[c]) / len(ys_by[c]),
    })

points = []
for p, (x, y), c in zip(papers, coords, labels):
    # Note: send the FULL title (not truncated) — the hover card
    # wraps it to a 4-line clamp on the frontend. `year` and
    # `category` show up in the hover meta row.
    points.append({
        "id": p["id"],
        "label": p["title"],
        "x": x,
        "y": y,
        "kind": "paper",
        "cluster": int(c),
        "year": (p.get("published") or "")[:4],
        "category": p.get("category", ""),
    })

result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    multi_use=True,
    payload={
        "type": "EntityCloud",
        "points": points,
        "clusters": clusters,
    },
)
pin_to_canvas(widget_id=result["widget_id"])

cluster_summary = ", ".join(c["label"] for c in clusters)
FINAL(f"Plotted {len(points)} papers across {len(clusters)} clusters: {cluster_summary}. Click a point to drill in.")
```

# Turn 2+ — user clicked a point in the cloud

You'll see a line in your conversation context like:

  [widget-event] action=select data={"pointId": "2504.13684", "point": {...}, "scope": {...}}

**Read the `pointId` value** out of that text and substitute it as a literal below. There is NO `data` variable in the Python REPL — you must hardcode the arxiv id you saw in the message.

Single iteration. The paper is in the cache from arxiv_search; vectors are in the cache from embed_papers(ids=...).

```python
arxiv_id = "<paste-the-pointId-here>"  # e.g. "2504.13684"
paper = get_paper(arxiv_id)
neighbors = nearest_neighbors(arxiv_id, k=5)  # 5 cosine-similar papers

# Authors line
authors = paper.get("authors", [])
authors_line = ", ".join(authors[:6]) + (" et al." if len(authors) > 6 else "")

# Build neighbor cards. Each card click sends `data = {"pointId": <neighbor_id>, ...}`
# which routes through the SAME drill-down flow you're in right now.
neighbor_cards = []
for n in neighbors:
    neighbor_cards.append({
        "type": "Card",
        "id": f"nbr-{n['id']}",
        # Send the FULL title — Card CSS wraps long titles.
        "title": n["title"],
        "subtitle": f"sim={n['similarity']:.2f} · " + ", ".join(n.get("authors", [])[:2]),
        "action": "select",
        "data": {"pointId": n["id"], "scope": {"kind": "paper", "key": n["id"]}},
    })

# Build 3 follow-up question buttons (follow_up=True, anchored to chat).
# Make them grounded in this paper specifically, not generic.
followups = {
    "type": "Column",
    "children": [
        {"type": "Text", "text": "Explore from here:"},
        {"type": "Row", "children": [
            {"type": "Button", "id": "fup-method",
             "label": "Method details",
             "action": "submit",
             "data": {"ask": f"What method does {paper['title'][:50]} actually use?"}},
            {"type": "Button", "id": "fup-compare",
             "label": "Vs neighbors",
             "action": "submit",
             "data": {"ask": f"How does {paper['title'][:40]} compare to its neighbors in the cloud?"}},
            {"type": "Button", "id": "fup-newer",
             "label": "What followed",
             "action": "submit",
             "data": {"ask": f"What more recent work builds on or supersedes {paper['title'][:40]}?"}},
        ]},
    ],
}

# Detail Column. Order matters — title/authors/meta on top, abstract,
# then a Text divider, then the neighbor list.
detail = {
    "type": "Column",
    "children": [
        # Heading (not Text) for the title — gets a serif title style
        # in the canvas-aux slot instead of inheriting the small-caps
        # picker-label cascade. Always the FIRST child.
        {"type": "Heading", "text": paper["title"]},
        {"type": "Text", "text": authors_line},
        {"type": "Text",
         "text": f"arXiv {arxiv_id} · {paper.get('published', '')[:10]} · {paper.get('category', '')}"},
        {"type": "Text", "text": paper.get("summary", "")},
        {"type": "Link", "url": paper.get("url", ""), "label": "View on arXiv ↗"},
        {"type": "Text", "text": "Nearest neighbors:"},
        {"type": "Column", "children": neighbor_cards},
    ],
}
result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    scope={"kind": "paper", "key": arxiv_id},  # auto-hides when user clicks a different paper
    payload=detail,
)
pin_below_canvas(widget_id=result["widget_id"])

# Anchor the follow-up question buttons to the chat (not the canvas).
emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    follow_up=True,
    payload=followups,
)

# Frontend parses this prefix to render the log line as a clickable
# re-pin button. Keep the "arxiv:" tag literal — that's the marker.
FINAL(f"Pinned · arxiv:{arxiv_id} · {paper['title']}")
```

# Turn N — user types a follow-up text question

If the user types something like "show me the most recent ones" or "what cluster is bottom-left?", run a fresh search or describe the layout — same two-iteration pattern, FINAL with prose.

# General rules

  - The EntityCloud widget always pins to the **primary** canvas (`pin_to_canvas`) with `multi_use=True`. It's the persistent workspace; clicks on points should not terminate it.
  - Per-paper detail widgets pin to the **aux** slot (`pin_below_canvas`) with a `scope={"kind": "paper", "key": <id>}` so they auto-hide when the user navigates away.
  - Do NOT scope the EntityCloud itself.
  - Cap max_results at 50; arXiv pagination kicks in past that.
  - If arxiv_search returns 0 papers, FINAL gracefully: "No papers found for that query — try a broader topic."
  - Never write Python that prints large blobs (full abstracts × 30 = a lot of stdout). Print summary lines only.
  - **When to fetch the full paper.** Abstracts are great for "show me the landscape" but lie by omission about methods and results. If the user asks a grounded follow-up — "what method does X use", "what is the result of X", "compare X and Y in detail", any of the click-button follow-ups (Method details / Vs neighbors / What followed) — call `fetch_paper_text(arxiv_id)` first, then write a 2–3 sentence answer that quotes or paraphrases the actual paper body. Don't fetch the full text for landscape questions; it's wasted context.
"###;

// ── Literature plugin ───────────────────────────────────────────────

struct LiteraturePlugin {
    http: Arc<reqwest::Client>,
    embedder: Arc<OllamaClient>,
    /// Per-session paper cache. Every `arxiv_search` call inserts
    /// each returned paper keyed by arxiv_id. `get_paper(arxiv_id)`
    /// reads from here so the agent doesn't need to round-trip a
    /// huge `meta` blob through click events. Never cleared — at
    /// our demo scale (≤50 papers per query) the cache stays small.
    paper_cache: Arc<StdMutex<HashMap<String, serde_json::Value>>>,
    /// Per-session vector cache, keyed by arxiv_id. Populated by
    /// `embed_papers` when called with an `ids` kwarg. Used by
    /// `nearest_neighbors` to compute cosine-similarity drill-downs
    /// without re-embedding.
    vector_cache: Arc<StdMutex<HashMap<String, Vec<f32>>>>,
    /// Per-session full-text cache, keyed by arxiv_id. Populated by
    /// `fetch_paper_text` after a successful ar5iv fetch + HTML strip.
    /// ar5iv conversion is slow on first hit; we cache the plaintext so
    /// the agent can ask multiple grounded questions about the same
    /// paper without re-fetching.
    text_cache: Arc<StdMutex<HashMap<String, String>>>,
    /// Optional gw-kb stores. When present, every successful
    /// `arxiv_search` ingests the discovered papers as kb_sources +
    /// kb_chunks so the corpus persists across sessions. None when
    /// `DATABASE_URL` etc. aren't configured — demo still runs without
    /// KB, just stateless.
    kb: Option<Arc<KbStores>>,
    /// Optional handle to a background worker that pulls per-source
    /// entity extraction jobs off an inbox and runs them sequentially
    /// against the LLM. `arxiv_search` enqueues new/updated source
    /// ids and returns immediately so the chat path never waits on
    /// per-chunk LLM calls. None when entity extraction is disabled
    /// (env `LITERATURE_ENTITY_EXTRACTION` unset/falsey, or `kb` None).
    entity_worker: Option<EntityWorkerHandle>,
}

impl LiteraturePlugin {
    fn new(embedder: Arc<OllamaClient>) -> Self {
        Self {
            http: Arc::new(
                reqwest::Client::builder()
                    .user_agent("greatwheel-literature-demo/0.1")
                    .build()
                    .expect("reqwest client"),
            ),
            embedder,
            paper_cache: Arc::new(StdMutex::new(HashMap::new())),
            vector_cache: Arc::new(StdMutex::new(HashMap::new())),
            text_cache: Arc::new(StdMutex::new(HashMap::new())),
            kb: None,
            entity_worker: None,
        }
    }

    fn with_kb(mut self, kb: Arc<KbStores>) -> Self {
        self.kb = Some(kb);
        self
    }

    fn with_entity_worker(mut self, worker: EntityWorkerHandle) -> Self {
        self.entity_worker = Some(worker);
        self
    }
}

impl Plugin for LiteraturePlugin {
    fn name(&self) -> &str {
        "literature"
    }

    fn manifest(&self) -> PluginManifest {
        PluginManifest {
            provides: vec![
                "literature".into(),
                "host_fn:literature.arxiv_search".into(),
                "host_fn:literature.embed_papers".into(),
                "host_fn:literature.project_2d".into(),
                "host_fn:literature.get_paper".into(),
                "host_fn:literature.nearest_neighbors".into(),
                "host_fn:literature.cluster_papers".into(),
                "host_fn:literature.fetch_paper_text".into(),
                "host_fn:literature.kb_paper_count".into(),
                "host_fn:literature.entity_extraction_status".into(),
            ],
            requires: vec![],
            priority: 0,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let http = self.http.clone();
        let cache = self.paper_cache.clone();
        let kb = self.kb.clone();
        let entity_worker = self.entity_worker.clone();
        ctx.register_host_fn_async("arxiv_search", None, move |args, kwargs| {
            let http = http.clone();
            let cache = cache.clone();
            let kb = kb.clone();
            let entity_worker = entity_worker.clone();
            async move {
                // Accept positional or kwarg `query`, same convention
                // as run_sql in the data explorer.
                let query = args
                    .first()
                    .and_then(|v| v.as_str())
                    .or_else(|| kwargs.get("query").and_then(|v| v.as_str()))
                    .ok_or_else(|| PluginError::HostFunction("query required (string)".into()))?
                    .trim()
                    .to_string();
                let max_results = args
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .or_else(|| kwargs.get("max_results").and_then(|v| v.as_u64()))
                    .unwrap_or(30)
                    .min(50) as usize;
                let result = arxiv_search(&http, &query, max_results)
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("arxiv_search: {e}")))?;
                // Populate the per-session paper cache so subsequent
                // get_paper(arxiv_id) calls can pull metadata without
                // a round trip.
                if let Value::Array(papers) = &result {
                    if let Ok(mut cache) = cache.lock() {
                        for paper in papers {
                            if let Some(id) = paper.get("id").and_then(|v| v.as_str()) {
                                cache.insert(id.to_string(), paper.clone());
                            }
                        }
                    }
                }

                // Best-effort gw-kb ingest. Each paper's title +
                // abstract becomes a kb_source / kb_chunk so the
                // corpus persists across sessions; later phases will
                // graduate the cloud to render from the KB instead of
                // re-fetching arXiv. Failures are logged but never
                // bubble up — the demo must keep working without the
                // KB. Fire each ingest sequentially: arXiv returns ≤50
                // papers per query and ingest_inline is fast enough
                // (no HTTP fetch — text is in hand).
                if let (Some(kb), Value::Array(papers)) = (kb.as_ref(), &result) {
                    let mut new_count = 0usize;
                    let mut existing_count = 0usize;
                    let mut failed = 0usize;
                    for paper in papers {
                        let id = paper
                            .get("id")
                            .and_then(|v| v.as_str())
                            .unwrap_or_default();
                        if id.is_empty() {
                            continue;
                        }
                        let title = paper
                            .get("title")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        let summary = paper
                            .get("summary")
                            .and_then(|v| v.as_str())
                            .unwrap_or("")
                            .to_string();
                        if summary.trim().is_empty() {
                            continue;
                        }
                        // Canonical arXiv URL — `ingest_inline`'s
                        // upsert_source dedups by URL, so re-running
                        // the same search is a no-op (just bumps
                        // last_seen).
                        let url = format!("https://arxiv.org/abs/{id}");
                        // Frontmatter lets `ingest_inline` parse the
                        // title cleanly; the abstract becomes the body.
                        let body = format!("---\ntitle: {title}\n---\n\n{summary}\n");
                        let metadata = json!({
                            "arxiv_id": id,
                            "category": paper.get("category").cloned().unwrap_or(Value::Null),
                            "published": paper.get("published").cloned().unwrap_or(Value::Null),
                            "authors": paper.get("authors").cloned().unwrap_or(Value::Null),
                        });
                        match ingest_inline(kb, &url, &body, Some(metadata)).await {
                            Ok(report) => {
                                match report.outcome {
                                    UpsertOutcome::Inserted => {
                                        new_count += 1;
                                        // Queue freshly-ingested sources for
                                        // background entity extraction. Worker
                                        // is None when extraction is disabled
                                        // via env, in which case we just
                                        // accumulate sources without entity
                                        // work (Phase A behaviour).
                                        if let Some(w) = entity_worker.as_ref() {
                                            w.enqueue_source(report.source.source_id);
                                        }
                                    }
                                    UpsertOutcome::Updated => {
                                        existing_count += 1;
                                        // Re-chunked content invalidates prior
                                        // entity links; re-extract.
                                        if let Some(w) = entity_worker.as_ref() {
                                            w.enqueue_source(report.source.source_id);
                                        }
                                    }
                                    UpsertOutcome::Unchanged => {
                                        existing_count += 1;
                                        // Already-known source whose hash
                                        // didn't change. Skip extraction —
                                        // either we did it on a previous run
                                        // or extraction is disabled.
                                    }
                                }
                            }
                            Err(e) => {
                                failed += 1;
                                tracing::warn!(arxiv_id = id, error = %e, "kb ingest failed");
                            }
                        }
                    }
                    tracing::info!(
                        new = new_count,
                        existing = existing_count,
                        failed,
                        "kb ingest summary"
                    );
                }

                Ok(result)
            }
        });

        let cache = self.paper_cache.clone();
        ctx.register_host_fn_sync("get_paper", None, move |args, kwargs| {
            let id = args
                .first()
                .and_then(|v| v.as_str())
                .or_else(|| kwargs.get("arxiv_id").and_then(|v| v.as_str()))
                .ok_or_else(|| PluginError::HostFunction("arxiv_id required (string)".into()))?
                .to_string();
            let cache = cache
                .lock()
                .map_err(|_| PluginError::HostFunction("paper cache poisoned".into()))?;
            match cache.get(&id) {
                Some(paper) => Ok(paper.clone()),
                None => Err(PluginError::HostFunction(format!(
                    "no paper cached for {id}; run arxiv_search first"
                ))),
            }
        });

        let embedder = self.embedder.clone();
        let vector_cache_w = self.vector_cache.clone();
        ctx.register_host_fn_async("embed_papers", None, move |args, kwargs| {
            let embedder = embedder.clone();
            let vector_cache = vector_cache_w.clone();
            async move {
                let texts_value = args
                    .first()
                    .or_else(|| kwargs.get("texts"))
                    .ok_or_else(|| PluginError::HostFunction("texts required (list[str])".into()))?;
                let texts = texts_value
                    .as_array()
                    .ok_or_else(|| {
                        PluginError::HostFunction("texts must be a list of strings".into())
                    })?
                    .iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>();
                // Optional parallel `ids` kwarg: when present, the
                // resulting vectors are cached server-side keyed by
                // id, ready for `nearest_neighbors` lookups.
                let ids: Option<Vec<String>> = kwargs.get("ids").and_then(|v| v.as_array()).map(
                    |arr| {
                        arr.iter()
                            .map(|v| v.as_str().unwrap_or("").to_string())
                            .collect()
                    },
                );
                if texts.is_empty() {
                    return Ok(Value::Array(vec![]));
                }
                let vectors = embedder
                    .embed(&texts)
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("embed: {e}")))?;
                let all_zero = !vectors.is_empty()
                    && vectors
                        .iter()
                        .all(|v| v.iter().all(|&x| x.abs() < 1e-9));
                if all_zero {
                    return Err(PluginError::HostFunction(
                        "embed_papers: every vector is zero — embedding model probably not pulled. Try `ollama pull nomic-embed-text` (or set LITERATURE_EMBED_MODEL to a model you have)."
                            .into(),
                    ));
                }
                if let Some(ids) = ids.as_ref() {
                    if ids.len() == vectors.len() {
                        if let Ok(mut cache) = vector_cache.lock() {
                            for (id, vec) in ids.iter().zip(vectors.iter()) {
                                cache.insert(id.clone(), vec.clone());
                            }
                        }
                    }
                }
                let json_vectors: Vec<Value> = vectors
                    .into_iter()
                    .map(|v| Value::Array(v.into_iter().map(num).collect()))
                    .collect();
                Ok(Value::Array(json_vectors))
            }
        });

        ctx.register_host_fn_sync("cluster_papers", None, move |args, kwargs| {
            let raw = args
                .first()
                .or_else(|| kwargs.get("vectors"))
                .ok_or_else(|| {
                    PluginError::HostFunction("vectors required (list[list[float]])".into())
                })?
                .as_array()
                .ok_or_else(|| {
                    PluginError::HostFunction("vectors must be a list of lists of floats".into())
                })?;
            let vectors: Vec<Vec<f32>> = raw
                .iter()
                .map(|row| {
                    row.as_array()
                        .map(|cells| {
                            cells
                                .iter()
                                .map(|c| c.as_f64().unwrap_or(0.0) as f32)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect();
            let k = args
                .get(1)
                .and_then(|v| v.as_u64())
                .or_else(|| kwargs.get("k").and_then(|v| v.as_u64()))
                .unwrap_or(6)
                .clamp(1, 20) as usize;
            let labels = kmeans(&vectors, k);
            let out: Vec<Value> = labels
                .into_iter()
                .map(|c| Value::Number(serde_json::Number::from(c as u64)))
                .collect();
            Ok(Value::Array(out))
        });

        let vector_cache_n = self.vector_cache.clone();
        let paper_cache_n = self.paper_cache.clone();
        ctx.register_host_fn_sync("nearest_neighbors", None, move |args, kwargs| {
            let id = args
                .first()
                .and_then(|v| v.as_str())
                .or_else(|| kwargs.get("arxiv_id").and_then(|v| v.as_str()))
                .ok_or_else(|| PluginError::HostFunction("arxiv_id required (string)".into()))?
                .to_string();
            let k = args
                .get(1)
                .and_then(|v| v.as_u64())
                .or_else(|| kwargs.get("k").and_then(|v| v.as_u64()))
                .unwrap_or(5)
                .min(20) as usize;
            let vectors = vector_cache_n
                .lock()
                .map_err(|_| PluginError::HostFunction("vector cache poisoned".into()))?;
            let papers = paper_cache_n
                .lock()
                .map_err(|_| PluginError::HostFunction("paper cache poisoned".into()))?;
            let focal = vectors.get(&id).cloned().ok_or_else(|| {
                PluginError::HostFunction(format!(
                    "no vector cached for {id}; call embed_papers with ids=[...] first"
                ))
            })?;
            let mut scored: Vec<(String, f32)> = vectors
                .iter()
                .filter(|(other, _)| **other != id)
                .map(|(other, vec)| (other.clone(), cosine(&focal, vec)))
                .collect();
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k);
            let result: Vec<Value> = scored
                .into_iter()
                .filter_map(|(nid, score)| {
                    let paper = papers.get(&nid)?.clone();
                    let mut entry = paper;
                    if let Value::Object(ref mut map) = entry {
                        map.insert("similarity".into(), num(score));
                    }
                    Some(entry)
                })
                .collect();
            Ok(Value::Array(result))
        });

        ctx.register_host_fn_sync("project_2d", None, move |args, kwargs| {
            let raw = args
                .first()
                .or_else(|| kwargs.get("vectors"))
                .ok_or_else(|| {
                    PluginError::HostFunction("vectors required (list[list[float]])".into())
                })?
                .as_array()
                .ok_or_else(|| {
                    PluginError::HostFunction("vectors must be a list of lists of floats".into())
                })?;
            let vectors: Vec<Vec<f32>> = raw
                .iter()
                .map(|row| {
                    row.as_array()
                        .map(|cells| {
                            cells
                                .iter()
                                .map(|c| c.as_f64().unwrap_or(0.0) as f32)
                                .collect::<Vec<_>>()
                        })
                        .unwrap_or_default()
                })
                .collect();
            let labels: Option<Vec<usize>> = kwargs.get("labels").and_then(|v| v.as_array()).map(
                |arr| {
                    arr.iter()
                        .map(|c| c.as_u64().unwrap_or(0) as usize)
                        .collect()
                },
            );
            let alpha = kwargs
                .get("alpha")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.5)
                .clamp(0.0, 1.0) as f32;
            // Layout mode: "blend" (default) preserves geometry —
            // PCA + pull-toward-cluster-mean. "ring" is decorative —
            // cluster centroids come from PCA on per-cluster mean
            // vectors, then members evenly distribute on a circle
            // around their centroid. Use ring when the cloud is for
            // browsing and faithful within-cluster distance doesn't
            // matter.
            let mode = kwargs
                .get("mode")
                .and_then(|v| v.as_str())
                .unwrap_or("blend")
                .to_string();

            let mut coords = if mode == "ring" {
                cluster_ring_layout(&vectors, labels.as_deref())
            } else {
                pca_project_2d(&vectors)
            };

            // Cluster-aware blending only applies in "blend" mode.
            if mode != "ring" {
                if let Some(labels) = labels.as_ref() {
                    if labels.len() == coords.len() && alpha > 0.0 {
                        let mut sums: std::collections::HashMap<usize, (f32, f32, usize)> =
                            std::collections::HashMap::new();
                        for (&c, &(x, y)) in labels.iter().zip(coords.iter()) {
                            let entry = sums.entry(c).or_insert((0.0, 0.0, 0));
                            entry.0 += x;
                            entry.1 += y;
                            entry.2 += 1;
                        }
                        let centroids: std::collections::HashMap<usize, (f32, f32)> = sums
                            .into_iter()
                            .map(|(c, (sx, sy, n))| {
                                let n = n.max(1) as f32;
                                (c, (sx / n, sy / n))
                            })
                            .collect();
                        for (i, (x, y)) in coords.iter_mut().enumerate() {
                            if let Some(&(cx, cy)) = centroids.get(&labels[i]) {
                                *x = (1.0 - alpha) * *x + alpha * cx;
                                *y = (1.0 - alpha) * *y + alpha * cy;
                            }
                        }
                    }
                }
            }

            let out: Vec<Value> = coords
                .into_iter()
                .map(|(x, y)| json!([f64::from(x).clamp(-2.0, 2.0), f64::from(y).clamp(-2.0, 2.0)]))
                .collect();
            Ok(Value::Array(out))
        });

        let http_t = self.http.clone();
        let text_cache = self.text_cache.clone();
        ctx.register_host_fn_async("fetch_paper_text", None, move |args, kwargs| {
            let http = http_t.clone();
            let text_cache = text_cache.clone();
            async move {
                let arxiv_id = args
                    .first()
                    .and_then(|v| v.as_str())
                    .or_else(|| kwargs.get("arxiv_id").and_then(|v| v.as_str()))
                    .ok_or_else(|| {
                        PluginError::HostFunction("arxiv_id required (string)".into())
                    })?
                    .trim()
                    .to_string();
                // Default 60K chars ≈ ~15K tokens — fits comfortably
                // alongside the rest of the agent context. Agent can
                // override but we hard-cap at 200K so a single fetch
                // can never blow the window on a long survey paper.
                let max_chars = args
                    .get(1)
                    .and_then(|v| v.as_u64())
                    .or_else(|| kwargs.get("max_chars").and_then(|v| v.as_u64()))
                    .unwrap_or(60_000)
                    .min(200_000) as usize;

                if let Some(cached) = text_cache.lock().ok().and_then(|m| m.get(&arxiv_id).cloned())
                {
                    let truncated = cached.chars().count() > max_chars;
                    let text: String = cached.chars().take(max_chars).collect();
                    return Ok(json!({
                        "id": arxiv_id,
                        "text": text,
                        "char_count": cached.chars().count(),
                        "truncated": truncated,
                        "source": "cache",
                    }));
                }

                let url = format!("https://ar5iv.labs.arxiv.org/html/{arxiv_id}");
                let resp = http.get(&url).send().await.map_err(|e| {
                    PluginError::HostFunction(format!("ar5iv fetch failed: {e}"))
                })?;
                if !resp.status().is_success() {
                    return Err(PluginError::HostFunction(format!(
                        "ar5iv returned HTTP {} for {arxiv_id} (paper may not have an HTML rendering — try a different paper)",
                        resp.status()
                    )));
                }
                let html = resp.text().await.map_err(|e| {
                    PluginError::HostFunction(format!("ar5iv body read failed: {e}"))
                })?;
                let plain = html_to_plaintext(&html);
                if plain.trim().is_empty() {
                    return Err(PluginError::HostFunction(format!(
                        "ar5iv returned an empty body for {arxiv_id}"
                    )));
                }

                if let Ok(mut m) = text_cache.lock() {
                    m.insert(arxiv_id.clone(), plain.clone());
                }
                let total = plain.chars().count();
                let truncated = total > max_chars;
                let text: String = plain.chars().take(max_chars).collect();
                Ok(json!({
                    "id": arxiv_id,
                    "text": text,
                    "char_count": total,
                    "truncated": truncated,
                    "source": "ar5iv",
                }))
            }
        });

        // kb_paper_count() -> {"sources": int, "configured": bool}
        // Quick readout of the persistent KB's paper count. The agent
        // calls this in its FINAL so the user can see the corpus
        // growing across sessions. When KB isn't configured, returns
        // {"configured": false} and the agent can omit the line.
        let kb_count = self.kb.clone();
        ctx.register_host_fn_async("kb_paper_count", None, move |_args, _kwargs| {
            let kb = kb_count.clone();
            async move {
                let Some(kb) = kb.as_ref() else {
                    return Ok(json!({ "configured": false, "sources": 0 }));
                };
                let row: Result<(i64,), _> =
                    sqlx::query_as("SELECT COUNT(*)::bigint FROM kb_sources")
                        .fetch_one(&kb.pg)
                        .await;
                match row {
                    Ok((n,)) => Ok(json!({ "configured": true, "sources": n })),
                    Err(e) => {
                        tracing::warn!(error = %e, "kb_paper_count query failed");
                        Ok(json!({ "configured": true, "sources": 0, "error": e.to_string() }))
                    }
                }
            }
        });

        // entity_extraction_status() -> {"configured", "queued",
        //   "in_progress", "completed", "completed_chunks",
        //   "entities_created_total", "entities_updated_total",
        //   "links_created_total", "last_run_ms"}
        //
        // Snapshot of the background worker's progress. The agent
        // calls this when it wants to mention entity-extraction
        // progress in narration ("3 papers still being analysed in
        // the background"). Returns {"configured": false} when
        // extraction is disabled; agent can omit the line.
        let worker = self.entity_worker.clone();
        ctx.register_host_fn_async(
            "entity_extraction_status",
            None,
            move |_args, _kwargs| {
                let worker = worker.clone();
                async move {
                    let Some(w) = worker else {
                        return Ok(json!({ "configured": false }));
                    };
                    let s = w.status.lock().await.clone();
                    Ok(json!({
                        "configured": true,
                        "queued": s.queued,
                        "in_progress": s.in_progress,
                        "completed": s.completed,
                        "completed_chunks": s.completed_chunks,
                        "entities_created_total": s.entities_created_total,
                        "entities_updated_total": s.entities_updated_total,
                        "links_created_total": s.links_created_total,
                        "last_run_ms": s.last_run_ms,
                    }))
                }
            },
        );

        Ok(())
    }
}

fn num(f: f32) -> Value {
    serde_json::Number::from_f64(f as f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na > 0.0 && nb > 0.0 {
        dot / (na * nb)
    } else {
        0.0
    }
}

/// Cosine k-means on the input vectors. Returns one cluster id per
/// vector in 0..k. Pure Rust, no deps.
///
/// Pipeline: L2-normalise each row → k-means++ init (first centroid
/// is point 0; each subsequent is the point furthest from any
/// existing centroid by cosine distance) → Lloyd iteration with
/// re-normalised centroids (cosine k-means).
fn kmeans(vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
    let n = vectors.len();
    if n == 0 {
        return vec![];
    }
    let k = k.min(n).max(1);
    let d = vectors[0].len();
    if d == 0 {
        return vec![0; n];
    }

    let normed: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-12 {
                v.clone()
            } else {
                v.iter().map(|x| x / norm).collect()
            }
        })
        .collect();

    // K-means++ init.
    let mut centroid_indices: Vec<usize> = vec![0];
    while centroid_indices.len() < k {
        let mut best_idx = 0usize;
        let mut best_min_dist = -1.0_f32;
        for (i, v) in normed.iter().enumerate() {
            if centroid_indices.contains(&i) {
                continue;
            }
            let min_d = centroid_indices
                .iter()
                .map(|&c_idx| 1.0 - cosine(v, &normed[c_idx]))
                .fold(f32::INFINITY, f32::min);
            if min_d > best_min_dist {
                best_min_dist = min_d;
                best_idx = i;
            }
        }
        centroid_indices.push(best_idx);
    }
    let mut centroids: Vec<Vec<f32>> = centroid_indices
        .iter()
        .map(|&i| normed[i].clone())
        .collect();

    let mut labels = vec![0usize; n];
    let max_iter = 50;
    for _ in 0..max_iter {
        // Assign each point to the nearest centroid (highest cosine).
        let mut changed = false;
        for (i, v) in normed.iter().enumerate() {
            let new_label = (0..k)
                .max_by(|&a, &b| {
                    let da = cosine(v, &centroids[a]);
                    let db = cosine(v, &centroids[b]);
                    da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal)
                })
                .unwrap_or(0);
            if labels[i] != new_label {
                labels[i] = new_label;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update centroids = mean of members, then re-normalise.
        for (c, centroid) in centroids.iter_mut().enumerate().take(k) {
            let mut new_c = vec![0.0f32; d];
            let mut count = 0usize;
            for (i, v) in normed.iter().enumerate() {
                if labels[i] == c {
                    for (j, x) in v.iter().enumerate() {
                        new_c[j] += x;
                    }
                    count += 1;
                }
            }
            if count == 0 {
                continue;
            }
            for x in &mut new_c {
                *x /= count as f32;
            }
            let norm: f32 = new_c.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-12 {
                for x in &mut new_c {
                    *x /= norm;
                }
            }
            *centroid = new_c;
        }
    }
    labels
}

// ── arXiv search ─────────────────────────────────────────────────────

#[derive(Debug)]
struct ArxivPaper {
    id: String,
    title: String,
    summary: String,
    authors: Vec<String>,
    published: String,
    category: String,
    url: String,
}

async fn arxiv_search(
    http: &reqwest::Client,
    query: &str,
    max_results: usize,
) -> Result<Value, Box<dyn std::error::Error + Send + Sync>> {
    // arXiv treats `+` as a space in search_query. We percent-encode
    // anything that's not unreserved per RFC 3986; common research
    // queries are mostly ASCII letters / digits / spaces.
    let q = percent_encode_query(query);
    let url = format!(
        "http://export.arxiv.org/api/query?search_query=all:{q}&max_results={max_results}&sortBy=relevance&sortOrder=descending"
    );
    let body = http
        .get(&url)
        .send()
        .await?
        .error_for_status()?
        .text()
        .await?;
    let papers = parse_arxiv_atom(&body);
    let json_papers: Vec<Value> = papers
        .into_iter()
        .map(|p| {
            json!({
                "id": p.id,
                "title": p.title,
                "summary": p.summary,
                "authors": p.authors,
                "published": p.published,
                "category": p.category,
                "url": p.url,
            })
        })
        .collect();
    Ok(Value::Array(json_papers))
}

/// Minimal Atom feed parser tailored to arXiv's response shape. We
/// don't pull in `quick-xml` because the format is regular and our
/// extraction targets are well-defined; saving a dep keeps the build
/// surface small. Anything we miss is recoverable — the agent uses
/// the result as a hint, not as ground truth.
fn parse_arxiv_atom(xml: &str) -> Vec<ArxivPaper> {
    let mut out = Vec::new();
    for entry in split_tags(xml, "entry") {
        let id = first_inner(&entry, "id").unwrap_or_default();
        // Strip trailing version (`http://arxiv.org/abs/2504.13684v1`
        // → `2504.13684`).
        let id_short = id
            .rsplit('/')
            .next()
            .map(|s| s.split('v').next().unwrap_or(s).to_string())
            .unwrap_or_default();
        let title = first_inner(&entry, "title").unwrap_or_default();
        let summary = first_inner(&entry, "summary").unwrap_or_default();
        let published = first_inner(&entry, "published").unwrap_or_default();
        let category = first_attr(&entry, "category", "term").unwrap_or_default();
        let url = first_attr(&entry, "link", "href").unwrap_or_else(|| id.clone());
        let authors: Vec<String> = split_tags(&entry, "author")
            .into_iter()
            .filter_map(|a| first_inner(&a, "name"))
            .collect();
        out.push(ArxivPaper {
            id: id_short,
            title: collapse_ws(&title),
            summary: collapse_ws(&summary),
            authors,
            published,
            category,
            url,
        });
    }
    out
}

/// Return the textual content of the first `<tag>...</tag>` in `xml`.
fn first_inner(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}");
    let close = format!("</{tag}>");
    let start = xml.find(&open)?;
    let after_open = xml[start..].find('>')? + start + 1;
    let end = xml[after_open..].find(&close)? + after_open;
    Some(xml[after_open..end].trim().to_string())
}

/// Return the value of `attr` on the first occurrence of `<tag ...>`.
fn first_attr(xml: &str, tag: &str, attr: &str) -> Option<String> {
    let open = format!("<{tag} ");
    let start = xml.find(&open)?;
    let close_tag = xml[start..].find('>')? + start;
    let inside = &xml[start..close_tag];
    let needle = format!("{attr}=\"");
    let attr_start = inside.find(&needle)? + needle.len();
    let attr_end = inside[attr_start..].find('"')? + attr_start;
    Some(inside[attr_start..attr_end].to_string())
}

/// Return every `<tag>...</tag>` block in `xml` as a Vec of substrings
/// (including the opening / closing tags). Naive — assumes no nested
/// tags of the same name within an entry, which holds for arXiv's
/// `<entry>` and `<author>` blocks.
fn split_tags(xml: &str, tag: &str) -> Vec<String> {
    let open = format!("<{tag}>");
    let open_attr = format!("<{tag} ");
    let close = format!("</{tag}>");
    let mut out = Vec::new();
    let mut cursor = 0usize;
    while cursor < xml.len() {
        let rest = &xml[cursor..];
        let i_plain = rest.find(&open);
        let i_attr = rest.find(&open_attr);
        let start_off = match (i_plain, i_attr) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => break,
        };
        let block_start = cursor + start_off;
        let block_close = match xml[block_start..].find(&close) {
            Some(p) => block_start + p + close.len(),
            None => break,
        };
        out.push(xml[block_start..block_close].to_string());
        cursor = block_close;
    }
    out
}

fn collapse_ws(s: &str) -> String {
    decode_xml_entities(&s.split_whitespace().collect::<Vec<_>>().join(" "))
}

/// Decode the handful of XML/HTML entities arXiv's Atom feed actually
/// uses in `<title>` and `<summary>` text. Without this, an ampersand
/// in a paper title arrives as the literal four characters `&amp;`.
/// Covers named entities (amp/lt/gt/quot/apos) and numeric escapes
/// (`&#NN;`, `&#xHH;`).
fn decode_xml_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut rest = s;
    while let Some(amp) = rest.find('&') {
        out.push_str(&rest[..amp]);
        let after = &rest[amp + 1..];
        match after.find(';') {
            Some(end) => {
                let entity = &after[..end];
                let decoded: Option<char> = if let Some(rest_e) = entity.strip_prefix('#') {
                    let (radix, digits) = match rest_e.strip_prefix(['x', 'X']) {
                        Some(hex) => (16u32, hex),
                        None => (10u32, rest_e),
                    };
                    u32::from_str_radix(digits, radix).ok().and_then(char::from_u32)
                } else {
                    match entity {
                        "amp" => Some('&'),
                        "lt" => Some('<'),
                        "gt" => Some('>'),
                        "quot" => Some('"'),
                        "apos" => Some('\''),
                        _ => None,
                    }
                };
                match decoded {
                    Some(c) => {
                        out.push(c);
                        rest = &after[end + 1..];
                    }
                    None => {
                        out.push('&');
                        rest = after;
                    }
                }
            }
            None => {
                out.push('&');
                rest = after;
            }
        }
    }
    out.push_str(rest);
    out
}

/// Strip HTML tags from an ar5iv document and return readable plain
/// text. Skips the contents of `<script>`, `<style>`, `<head>`, `<nav>`,
/// `<footer>`, and ar5iv's `<div class="ltx_bibliography">` block (the
/// agent rarely needs the full bibliography and it's most of the page).
/// Block-level tags become single newlines so paragraph structure
/// survives; inline entities are decoded via `decode_xml_entities`.
/// This is a deliberately simple lexer — ar5iv's output is well-formed
/// XHTML, so we don't need a full DOM.
fn html_to_plaintext(html: &str) -> String {
    let bytes = html.as_bytes();
    let mut out = String::with_capacity(html.len() / 2);
    let mut i = 0;
    let drop_tags: &[&[u8]] = &[
        b"script",
        b"style",
        b"head",
        b"nav",
        b"footer",
        b"figure",
        b"math",
    ];
    let block_tags: &[&[u8]] = &[
        b"p", b"div", b"li", b"h1", b"h2", b"h3", b"h4", b"h5", b"h6", b"br", b"section",
        b"article", b"tr", b"table", b"blockquote",
    ];
    while i < bytes.len() {
        if bytes[i] == b'<' {
            // Skip comments.
            if bytes[i..].starts_with(b"<!--") {
                if let Some(end) = html[i..].find("-->") {
                    i += end + 3;
                    continue;
                } else {
                    break;
                }
            }
            // Find tag end.
            let Some(end_rel) = html[i..].find('>') else {
                break;
            };
            let tag_block = &html[i + 1..i + end_rel];
            let tag_block_bytes = tag_block.as_bytes();
            // Skip the bibliography section by sniffing the class attr.
            if tag_block_bytes.first() == Some(&b'd')
                && tag_block.starts_with("div")
                && tag_block.contains("ltx_bibliography")
            {
                if let Some(close_rel) = html[i..].find("</div>") {
                    i += close_rel + "</div>".len();
                    continue;
                }
            }
            // Determine tag name (strip leading '/' and read until
            // whitespace or '>').
            let (is_close, name_start) = if tag_block_bytes.first() == Some(&b'/') {
                (true, 1)
            } else {
                (false, 0)
            };
            let name_end = tag_block_bytes[name_start..]
                .iter()
                .position(|&b| b == b' ' || b == b'\t' || b == b'/')
                .map(|p| name_start + p)
                .unwrap_or(tag_block_bytes.len());
            let name: Vec<u8> = tag_block_bytes[name_start..name_end]
                .iter()
                .map(|b| b.to_ascii_lowercase())
                .collect();

            // For drop_tags, fast-forward to the matching close tag.
            if !is_close && drop_tags.iter().any(|t| t == &name.as_slice()) {
                let close = format!("</{}", std::str::from_utf8(&name).unwrap_or(""));
                if let Some(close_rel) = html[i..].find(&close) {
                    if let Some(gt) = html[i + close_rel..].find('>') {
                        i += close_rel + gt + 1;
                        continue;
                    }
                }
                break;
            }

            if block_tags.iter().any(|t| t == &name.as_slice()) {
                if !out.ends_with('\n') {
                    out.push('\n');
                }
            } else {
                // inline tag — emit a space so adjacent words don't fuse.
                if !out.ends_with(char::is_whitespace) {
                    out.push(' ');
                }
            }
            i += end_rel + 1;
        } else {
            // Append text run up to the next '<'.
            let next = html[i..].find('<').map(|p| i + p).unwrap_or(bytes.len());
            out.push_str(&decode_xml_entities(&html[i..next]));
            i = next;
        }
    }

    // Collapse runs of whitespace, but preserve blank-line paragraph
    // boundaries: turn 3+ newlines into 2, and trim leading/trailing.
    let mut collapsed = String::with_capacity(out.len());
    let mut last_was_space = false;
    let mut newlines = 0usize;
    for c in out.chars() {
        if c == '\n' {
            newlines += 1;
            last_was_space = true;
            continue;
        }
        if newlines > 0 {
            collapsed.push_str(if newlines >= 2 { "\n\n" } else { "\n" });
            newlines = 0;
            last_was_space = true;
        }
        if c.is_whitespace() {
            if !last_was_space {
                collapsed.push(' ');
                last_was_space = true;
            }
        } else {
            collapsed.push(c);
            last_was_space = false;
        }
    }
    collapsed.trim().to_string()
}

// ─── Background entity-extraction worker ────────────────────────────
//
// One tokio task per binary instance. Owns a tokio mpsc inbox of
// source ids and a shared status struct readable from any host fn.
// arxiv_search pushes ids and returns; the worker drains the inbox,
// running `extract_and_persist_entities_for_source` per id. Sequential
// because the local Ollama path serialises anyway and a single worker
// also serialises any future link-rebuild step.
//
// Failure semantics: per-source errors are logged at warn level and
// don't take down the worker. Channel close (binary shutting down)
// exits the loop cleanly.

#[derive(Debug, Clone, Default)]
struct EntityWorkerStatus {
    /// Sources sitting in the inbox waiting for the worker to pick
    /// them up.
    queued: usize,
    /// Sources currently mid-extract. 0 or 1 in the single-worker
    /// model — surfaced as a counter so the same shape would extend
    /// to a worker pool later.
    in_progress: usize,
    /// Sources finished since process start (success only).
    completed: usize,
    /// Total chunks extracted across all completed sources.
    completed_chunks: usize,
    /// Aggregate of `EntityIngestReport` fields across the run.
    entities_created_total: usize,
    entities_updated_total: usize,
    links_created_total: usize,
    /// Wall time of the most recent extract pass.
    last_run_ms: u64,
}

#[derive(Clone)]
struct EntityWorkerHandle {
    tx: tokio::sync::mpsc::UnboundedSender<Uuid>,
    status: Arc<tokio::sync::Mutex<EntityWorkerStatus>>,
}

impl EntityWorkerHandle {
    fn enqueue_source(&self, source_id: Uuid) {
        // try_lock instead of blocking on the mutex inside the chat
        // path — if we somehow deadlock with the worker, drop the
        // counter increment rather than the chat. The worker also
        // increments queued atomically when it pulls.
        if let Ok(mut s) = self.status.try_lock() {
            s.queued += 1;
        }
        // Channel closure means the worker has exited — log once
        // here rather than crashing the chat.
        if self.tx.send(source_id).is_err() {
            tracing::warn!(%source_id, "entity worker inbox closed; source dropped");
        }
    }
}

fn spawn_entity_worker(stores: Arc<KbStores>) -> EntityWorkerHandle {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Uuid>();
    let status = Arc::new(tokio::sync::Mutex::new(EntityWorkerStatus::default()));
    let status_w = status.clone();
    tokio::spawn(async move {
        let opts = CanonicalizeOpts::default();
        while let Some(source_id) = rx.recv().await {
            {
                let mut s = status_w.lock().await;
                // The chat path may or may not have decremented queued
                // (try_lock); reconcile by clamping to ≥ 0 implicitly
                // via saturating_sub.
                s.queued = s.queued.saturating_sub(1);
                s.in_progress += 1;
            }
            let started = std::time::Instant::now();
            let result: Result<EntityIngestReport, _> =
                extract_and_persist_entities_for_source(&stores, source_id, &opts).await;
            let elapsed_ms = started.elapsed().as_millis() as u64;
            let mut s = status_w.lock().await;
            s.in_progress = s.in_progress.saturating_sub(1);
            s.last_run_ms = elapsed_ms;
            match result {
                Ok(report) => {
                    s.completed += 1;
                    s.completed_chunks += report.mentions_in;
                    s.entities_created_total += report.entities_created;
                    s.entities_updated_total += report.entities_updated;
                    s.links_created_total += report.links_created;
                    tracing::info!(
                        %source_id,
                        mentions = report.mentions_in,
                        new_entities = report.entities_created,
                        updated = report.entities_updated,
                        links = report.links_created,
                        ms = elapsed_ms,
                        "entity extraction complete"
                    );
                }
                Err(e) => {
                    tracing::warn!(%source_id, error = %e, "entity extraction failed");
                }
            }
        }
        tracing::info!("entity worker exiting (channel closed)");
    });
    EntityWorkerHandle { tx, status }
}

/// Build the gw-kb store bundle from environment, returning Ok(None)
/// when the user hasn't configured a database (so the demo can keep
/// running stateless). Defaults match the `gw_kb` CLI: lance under
/// `data/kb-lancedb`, tantivy under `data/kb-tantivy`, embedding model
/// `nomic-ai/nomic-embed-text-v1.5` (768-dim).
async fn build_kb_stores() -> Result<Option<KbStores>, Box<dyn std::error::Error>> {
    use gw_kb::embed::Embedder;
    use gw_kb::index::{KbLanceStore, KbTantivyStore};
    use sqlx::postgres::PgPoolOptions;

    let Ok(database_url) = std::env::var("DATABASE_URL") else {
        return Ok(None);
    };
    let lance_path =
        std::env::var("KB_LANCE_PATH").unwrap_or_else(|_| "data/kb-lancedb".into());
    let tantivy_path =
        std::env::var("KB_TANTIVY_PATH").unwrap_or_else(|_| "data/kb-tantivy".into());
    let embedding_model = std::env::var("KB_EMBEDDING_MODEL")
        .unwrap_or_else(|_| "nomic-ai/nomic-embed-text-v1.5".into());
    let embedding_dim: i32 = std::env::var("KB_EMBEDDING_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(768);
    let ollama_url = std::env::var("KB_OLLAMA_URL").unwrap_or_else(|_| OLLAMA_URL.into());

    let pg = PgPoolOptions::new()
        .max_connections(4)
        .connect(&database_url)
        .await?;
    // Apply gw-kb's schema migrations against this Postgres. The
    // sqlx::migrate! macro embeds the *.sql files at compile time and
    // tracks applied versions in a `_sqlx_migrations` table, so this
    // is idempotent: a fresh database gets all 12 migrations, an
    // already-bootstrapped one (e.g. from BrowseComp work) is a no-op.
    sqlx::migrate!("../../migrations").run(&pg).await?;
    let lance = Arc::new(KbLanceStore::open(&lance_path, embedding_dim).await?);
    let tantivy = Arc::new(KbTantivyStore::open(std::path::Path::new(&tantivy_path))?);
    let embedder = Arc::new(Embedder::new(embedding_model));
    // The chat LLM here is only used by gw-kb's organize/synthesize
    // pipelines, which the literature demo doesn't run. Plumb a
    // placeholder so the type-checks stay satisfied.
    let llm = Arc::new(OllamaClient::new(
        ollama_url.clone(),
        ollama_url,
        OLLAMA_MODEL.into(),
        "unused".into(),
    ));
    Ok(Some(KbStores {
        pg,
        lance,
        tantivy,
        embedder,
        llm,
    }))
}

fn percent_encode_query(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'a'..=b'z' | b'A'..=b'Z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char);
            }
            b' ' => out.push('+'),
            _ => out.push_str(&format!("%{b:02X}")),
        }
    }
    out
}

// ── PCA (top-2 components via power iteration) ──────────────────────

/// Project each row of `vectors` onto the top-2 principal components.
/// Returns one `(x, y)` per input row, with the layout centred and
/// **whitened** so each axis has unit standard deviation across
/// points (then re-scaled so max |coord| ≈ 1).
///
/// Pipeline: L2-normalise each row → centre by mean → power-iterate
/// on the n×n Gram matrix for top-2 eigenvectors → whiten.
///
/// L2-normalisation matters: text embeddings encode meaning as
/// direction (cosine similarity), so raw PCA on un-normalised
/// vectors is dominated by length variance — typical symptom is
/// 29 of 30 papers collapsing to a single dot near origin while
/// one outlier defines the long axis. Normalising puts everything
/// on the unit sphere so PCA captures real angular spread.
///
/// Whitening matters: the top eigenvalue often eats most of the
/// variance, leaving axis-2 with sub-pixel spread. Dividing each
/// projected axis by its std-dev gives a balanced layout.
/// Decorative cluster-ring layout. Each cluster's mean vector goes
/// through PCA to get a 2D centroid, then members distribute evenly on
/// a circle around that centroid. Sacrifices within-cluster distance
/// fidelity for visual cleanliness — useful when the cloud is for
/// browsing and the actual within-cluster geometry doesn't matter.
///
/// Ring radius scales with √cluster_size so a 12-paper cluster reads
/// visibly larger than a 3-paper one without the ring eating its
/// neighbours. Within-cluster ordering is preserved (12 o'clock = first
/// member); for arxiv_search results this means the highest-relevance
/// paper in each cluster lands at the top of its ring.
fn cluster_ring_layout(vectors: &[Vec<f32>], labels: Option<&[usize]>) -> Vec<(f32, f32)> {
    let n = vectors.len();
    let Some(labels) = labels else {
        return pca_project_2d(vectors);
    };
    if labels.len() != n || n == 0 {
        return pca_project_2d(vectors);
    }
    let dim = vectors.first().map(Vec::len).unwrap_or(0);
    let k = labels.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    if k == 0 || dim == 0 {
        return pca_project_2d(vectors);
    }

    // Per-cluster mean vector (in the original embedding space).
    let mut means = vec![vec![0.0f32; dim]; k];
    let mut counts = vec![0usize; k];
    for (i, v) in vectors.iter().enumerate() {
        let c = labels[i];
        if v.len() == dim {
            for (j, x) in v.iter().enumerate() {
                means[c][j] += x;
            }
        }
        counts[c] += 1;
    }
    for (c, m) in means.iter_mut().enumerate() {
        if counts[c] > 0 {
            for x in m.iter_mut() {
                *x /= counts[c] as f32;
            }
        }
    }

    // PCA-place centroids in 2D. With small k (typically 6) PCA on
    // means is cheap and gives a sensible spread that mirrors topical
    // similarity between clusters.
    let mut cluster_centers = pca_project_2d(&means);

    // Per-cluster ring radius (also used by the repulsion pass below
    // so the min-separation honours the actual rendered rings).
    let radii: Vec<f32> = (0..k).map(|c| ring_radius(counts[c])).collect();

    // Pairwise repulsion — when two cluster means are topical
    // neighbours, PCA puts them close enough that their rings stack
    // visually. Iterate up to ~80 passes pushing any pair closer than
    // `radius_i + radius_j + padding` apart along their connecting
    // axis. Converges fast (typically 5-15 passes) for k=6.
    let padding = 0.08;
    for _ in 0..80 {
        let mut moved = false;
        for i in 0..cluster_centers.len() {
            for j in (i + 1)..cluster_centers.len() {
                let dx = cluster_centers[j].0 - cluster_centers[i].0;
                let dy = cluster_centers[j].1 - cluster_centers[i].1;
                let dist = (dx * dx + dy * dy).sqrt();
                let min_dist = radii[i] + radii[j] + padding;
                if dist < min_dist {
                    // Pick a push direction: use the connecting axis
                    // when it exists, otherwise nudge along x so two
                    // exactly-coincident centers separate at all.
                    let (nx, ny) = if dist > 1e-6 {
                        (dx / dist, dy / dist)
                    } else {
                        let theta = (i + j) as f32 * 1.7;
                        (theta.cos(), theta.sin())
                    };
                    let half_push = (min_dist - dist.max(1e-6)) * 0.5 + 1e-4;
                    cluster_centers[i].0 -= nx * half_push;
                    cluster_centers[i].1 -= ny * half_push;
                    cluster_centers[j].0 += nx * half_push;
                    cluster_centers[j].1 += ny * half_push;
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }

    let mut indices_by_cluster: Vec<Vec<usize>> = vec![Vec::new(); k];
    for (i, &c) in labels.iter().enumerate() {
        indices_by_cluster[c].push(i);
    }

    let mut coords = vec![(0.0f32, 0.0f32); n];
    for (c, indices) in indices_by_cluster.iter().enumerate() {
        let center = cluster_centers.get(c).copied().unwrap_or((0.0, 0.0));
        let m = indices.len();
        if m == 0 {
            continue;
        }
        if m == 1 {
            coords[indices[0]] = center;
            continue;
        }
        let radius = radii[c];
        // Offset start angle by cluster id so adjacent clusters don't
        // all have a member at 12 o'clock — a small visual flourish.
        let phase = (c as f32) * 0.37;
        for (k_idx, &point_idx) in indices.iter().enumerate() {
            let angle =
                phase + (k_idx as f32 / m as f32) * std::f32::consts::TAU;
            coords[point_idx] = (
                center.0 + radius * angle.cos(),
                center.1 + radius * angle.sin(),
            );
        }
    }
    coords
}

/// Visible ring radius for a cluster of `m` papers. Singletons get a
/// zero-radius ring (just the centroid). Sqrt scaling keeps small
/// clusters tight and big clusters legible without exploding.
fn ring_radius(m: usize) -> f32 {
    if m <= 1 {
        0.0
    } else {
        0.10 + (m as f32).sqrt() * 0.045
    }
}

fn pca_project_2d(vectors: &[Vec<f32>]) -> Vec<(f32, f32)> {
    let n = vectors.len();
    if n == 0 {
        return vec![];
    }
    if n < 3 {
        // Not enough points for meaningful PCA — spread them on a line.
        return (0..n).map(|i| (i as f32 * 0.5 - 0.5, 0.0)).collect();
    }
    let d = vectors[0].len();
    if d == 0 {
        return vec![(0.0, 0.0); n];
    }

    // L2-normalise every row in-place. After this each vector lives on
    // the unit hypersphere; cosine similarity = dot product.
    let normed: Vec<Vec<f32>> = vectors
        .iter()
        .map(|v| {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-12 {
                v.clone()
            } else {
                v.iter().map(|x| x / norm).collect()
            }
        })
        .collect();

    // Centre each column.
    let mut means = vec![0.0f32; d];
    for v in &normed {
        for (j, x) in v.iter().enumerate() {
            means[j] += x;
        }
    }
    for m in &mut means {
        *m /= n as f32;
    }
    let mut centred: Vec<Vec<f32>> = normed
        .iter()
        .map(|v| {
            v.iter()
                .enumerate()
                .map(|(j, x)| x - means[j])
                .collect::<Vec<_>>()
        })
        .collect();

    // Gram matrix G = X X^T (n × n).
    let mut gram = vec![vec![0.0f32; n]; n];
    for i in 0..n {
        for j in i..n {
            let dot: f32 = centred[i]
                .iter()
                .zip(centred[j].iter())
                .map(|(a, b)| a * b)
                .sum();
            gram[i][j] = dot;
            gram[j][i] = dot;
        }
    }

    // Power iteration for the top eigenvector of `gram`.
    let pc1 = power_iterate(&gram, 100, 1e-6);
    // Deflate: G' = G - λ₁ v₁ v₁ᵀ. With orthonormal v₁ that just
    // subtracts the rank-1 projection.
    let lambda1 = rayleigh(&gram, &pc1);
    let mut deflated = gram.clone();
    for i in 0..n {
        for j in 0..n {
            deflated[i][j] -= lambda1 * pc1[i] * pc1[j];
        }
    }
    let pc2 = power_iterate(&deflated, 100, 1e-6);

    // Whiten each projected axis: divide by its std-dev across
    // points so axis-1 and axis-2 have comparable spread. Without
    // this the top eigenvalue often dwarfs the second by 10-100x
    // and the second axis collapses to sub-pixel range.
    let mut xs: Vec<f32> = pc1.clone();
    let mut ys: Vec<f32> = pc2.clone();
    let stddev = |v: &[f32]| -> f32 {
        let mean: f32 = v.iter().sum::<f32>() / v.len() as f32;
        let var: f32 = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
        var.sqrt().max(1e-9)
    };
    let sx = stddev(&xs);
    let sy = stddev(&ys);
    for x in &mut xs {
        *x /= sx;
    }
    for y in &mut ys {
        *y /= sy;
    }
    // Final pass: scale so max |coord| ≈ 1 (the SVG widget assumes
    // roughly that range).
    let mut coords: Vec<(f32, f32)> = xs.into_iter().zip(ys).collect();
    let max_abs = coords
        .iter()
        .flat_map(|(x, y)| [x.abs(), y.abs()])
        .fold(0.0f32, f32::max);
    if max_abs > 0.0 {
        for c in &mut coords {
            c.0 /= max_abs;
            c.1 /= max_abs;
        }
    }
    // Variables `lambda1`, `centred` are intermediate; touch them so
    // future refactors keep them easy to debug from this site.
    let _ = lambda1;
    centred.clear();
    coords
}

fn power_iterate(m: &[Vec<f32>], max_iter: usize, tol: f32) -> Vec<f32> {
    let n = m.len();
    if n == 0 {
        return vec![];
    }
    // Deterministic init: 1/sqrt(n) on every component.
    let init = 1.0 / (n as f32).sqrt();
    let mut v = vec![init; n];
    let mut last_lambda = 0.0f32;
    for _ in 0..max_iter {
        // w = M v
        let mut w = vec![0.0f32; n];
        for (i, row) in m.iter().enumerate() {
            let mut acc = 0.0f32;
            for (j, x) in row.iter().enumerate() {
                acc += x * v[j];
            }
            w[i] = acc;
        }
        let norm: f32 = w.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-12 {
            break;
        }
        for x in &mut w {
            *x /= norm;
        }
        let lambda = rayleigh(m, &w);
        if (lambda - last_lambda).abs() < tol {
            return w;
        }
        last_lambda = lambda;
        v = w;
    }
    v
}

fn rayleigh(m: &[Vec<f32>], v: &[f32]) -> f32 {
    // v^T M v / v^T v, assuming v is already unit length.
    let mut acc = 0.0f32;
    for (i, row) in m.iter().enumerate() {
        let mut row_acc = 0.0f32;
        for (j, x) in row.iter().enumerate() {
            row_acc += x * v[j];
        }
        acc += v[i] * row_acc;
    }
    acc
}

// ── Server ──────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let _ = dotenvy::dotenv();

    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,gw_ui=debug".into()),
        )
        .init();

    // Embedder: always go via Ollama. The chat LLM can still be
    // OpenAI if OPENAI_API_KEY is set.
    let embedding_model =
        std::env::var("LITERATURE_EMBED_MODEL").unwrap_or_else(|_| DEFAULT_EMBEDDING_MODEL.into());
    println!("embed model: {embedding_model}");
    let embedder = Arc::new(OllamaClient::new(
        OLLAMA_URL.into(),
        OLLAMA_URL.into(),
        OLLAMA_MODEL.into(),
        embedding_model.clone(),
    ));

    let (chat_client, model_label) = if let Ok(key) = std::env::var("OPENAI_API_KEY") {
        let model = std::env::var("OPENAI_MODEL").unwrap_or_else(|_| DEFAULT_OPENAI_MODEL.into());
        let label = format!("openai:{model}");
        (OllamaClient::new_openai(key, model), label)
    } else {
        let ollama = OllamaClient::new(
            OLLAMA_URL.into(),
            OLLAMA_URL.into(),
            OLLAMA_MODEL.into(),
            embedding_model.clone(),
        );
        (ollama, format!("ollama:{OLLAMA_MODEL}"))
    };
    let loop_llm: Box<dyn gw_loop::LlmClient> =
        Box::new(OllamaLlmClient::new(chat_client).with_think(Some(false)));

    // Best-effort gw-kb setup. When DATABASE_URL is set we wire up
    // Postgres + LanceDB + tantivy + a sentence-transformers embedder
    // and pass them to LiteraturePlugin. Every arxiv_search then
    // persists discovered papers to the KB. If anything fails, log
    // and continue stateless — the demo doesn't depend on KB at this
    // phase.
    let kb_stores: Option<Arc<KbStores>> = match build_kb_stores().await {
        Ok(Some(s)) => {
            tracing::info!("gw-kb stores ready — arxiv_search will persist results");
            Some(Arc::new(s))
        }
        Ok(None) => {
            tracing::info!("gw-kb not configured (DATABASE_URL unset); running stateless");
            None
        }
        Err(e) => {
            tracing::warn!(error = %e, "gw-kb setup failed; running stateless");
            None
        }
    };

    let mut lit = LiteraturePlugin::new(embedder);
    if let Some(kb) = kb_stores.clone() {
        lit = lit.with_kb(kb.clone());
        // Background entity extraction is opt-in: the LLM cost is
        // ~one chat call per chunk per query, so plain demo runs stay
        // cheap unless the user asks for it.
        let enable = std::env::var("LITERATURE_ENTITY_EXTRACTION")
            .map(|v| matches!(v.as_str(), "1" | "true" | "on" | "yes"))
            .unwrap_or(false);
        if enable {
            tracing::info!("entity worker enabled (LITERATURE_ENTITY_EXTRACTION)");
            let handle = spawn_entity_worker(kb);
            lit = lit.with_entity_worker(handle);
        } else {
            tracing::info!(
                "entity extraction disabled — set LITERATURE_ENTITY_EXTRACTION=on to enable"
            );
        }
    }

    let engine = GreatWheelEngine::new()
        .add_plugin(UiPlugin)
        .add_plugin(lit)
        .init(&HashMap::new())?;
    let plugin_router = engine.host_fn_router_arc();
    let store: Arc<UiSurfaceStore> = engine
        .registry
        .shared()
        .get::<Arc<UiSurfaceStore>>()
        .cloned()
        .ok_or("UiPlugin did not provide UiSurfaceStore")?;

    let adapter = Arc::new(AgUiAdapter::new(&store));
    adapter.set_branding("ArxivCloud", "search · embed · explore");
    adapter.set_layout("canvas-primary");
    adapter.set_welcome(
        "A spatial map of arXiv, drawn for you on demand.",
        "Name a research topic in plain language. The agent searches arXiv, embeds the abstracts, clusters them, and lays them out as a 2D cloud you can hover and click. Each cluster is auto-named; each point opens a paper synthesis with its nearest neighbours.",
        [
            "Survey of recent work on retrieval-augmented generation",
            "What's new in LLM agent evaluation?",
            "Diffusion models for code generation",
            "Mechanistic interpretability of transformers",
        ],
    );
    let session_id = SessionId(Uuid::new_v4());

    let (tap_tx, mut tap_rx) = mpsc::unbounded_channel::<LoopEvent>();
    let (loop_tx, loop_rx) = mpsc::unbounded_channel::<LoopEvent>();
    adapter.register_session(session_id, tap_tx.clone()).await;

    let adapter_for_tap = adapter.clone();
    tokio::spawn(async move {
        while let Some(ev) = tap_rx.recv().await {
            adapter_for_tap.dispatch(session_id, &ev).await;
            if loop_tx.send(ev).is_err() {
                break;
            }
        }
    });

    let ask_handle = new_ask_handle();
    let conv_bridge = ConversationBridge::with_plugin_router(
        tap_tx.clone(),
        ask_handle,
        None,
        Some(plugin_router),
    );

    let external_fns = vec![
        "FINAL".into(),
        "emit_widget".into(),
        "supersede_widget".into(),
        "resolve_widget".into(),
        "pin_to_canvas".into(),
        "pin_below_canvas".into(),
        "highlight_button".into(),
        "arxiv_search".into(),
        "embed_papers".into(),
        "project_2d".into(),
        "get_paper".into(),
        "nearest_neighbors".into(),
        "cluster_papers".into(),
        "fetch_paper_text".into(),
        "kb_paper_count".into(),
        "entity_extraction_status".into(),
    ];
    let mut repl = ReplAgent::new(external_fns, Box::new(conv_bridge));
    repl.set_variable("gw_session_id", Object::String(session_id.0.to_string()))
        .ok();

    let config = LoopConfig {
        system_prompt: SYSTEM_PROMPT.to_string(),
        recency_window: 30,
        max_iterations: 4,
        include_code_output: true,
        repl_output_max_chars: 4000,
        strip_think_tags: true,
        answer_validator: None,
        iteration_callback: None,
        snapshot_policy: SnapshotPolicy {
            every_n_turns: 0,
            before_compaction: false,
        },
        compaction_keep_count: 0,
        auto_compact_after_turns: None,
    };
    let mut conv_loop = ConversationLoop::new(session_id, repl, loop_llm, config, tap_tx);

    std::thread::Builder::new()
        .name("gw-loop".into())
        .spawn(move || {
            let rt = tokio::runtime::Builder::new_multi_thread()
                .enable_all()
                .worker_threads(2)
                .build()
                .expect("failed to build loop runtime");
            rt.block_on(async move {
                if let Err(e) = conv_loop.run(loop_rx).await {
                    tracing::error!(error = %e, "conversation loop exited");
                }
            });
        })?;

    let app = adapter
        .router()
        .layer(TraceLayer::new_for_http())
        .layer(CorsLayer::permissive());
    let listener = TcpListener::bind("127.0.0.1:8787").await?;
    println!("literature assistant listening on http://127.0.0.1:8787");
    println!("model: {model_label}");
    println!("session_id: {}", session_id.0);
    println!("open http://localhost:5173/?session={}", session_id.0);

    axum::serve(listener, app).await?;
    Ok(())
}
