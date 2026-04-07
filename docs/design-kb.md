# Design: Knowledge Base Ingestor (`gw-kb`)

**Status:** Draft
**Date:** 2026-04-06
**Motivation:** A personal knowledge base that ingests web documents and local files, organizes them into discoverable topic structures, and synthesizes actionable summaries — purpose-built for driving software development, ML research, and learning.

---

## 0. Problem Statement

Good search is insufficient when you don't know what to search for. The use
cases driving this system all share a common pattern: you're ingesting a stream
of information and need the system to **surface structure, connections, and gaps**
that you wouldn't find by querying.

**Target use cases:**
- Research a topic (e.g. LLM memory architectures) to inform implementation decisions
- Track a publication feed (e.g. HuggingFace blog) and map emerging research topics
- Follow trends across ML research as learning resources
- Build comprehensive understanding of a field to identify greenfield project opportunities

**Key property:** the knowledge base is not archival — it exists to help you
*act*. Every design decision should be evaluated against: "does this help the
user build software, train models, learn a field, or stay on top of current
events?"

---

## 1. Architecture Overview

```
Sources (URLs, files, feeds)
    │
    ▼
┌─────────────────────────────────┐
│  Extract (Python)               │
│  trafilatura (web/HTML)         │
│  marker (PDF)                   │
│  pymupdf4llm (lightweight PDF)  │
│  ─────────────────────────────  │
│  Output: markdown + metadata    │
└─────────────┬───────────────────┘
              │  markdown files + JSON metadata
              ▼
┌─────────────────────────────────┐
│  Ingest (Rust, gw-kb)          │
│  chunk → embed → store          │
│  deduplicate, link to source    │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Organize (Rust + LLM)         │
│  cluster → topics → domains     │
│  detect hierarchy, link topics  │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Synthesize (LLM)              │
│  per-topic summaries            │
│  cross-topic connections        │
│  delta detection (what changed) │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Surface (CLI + API)           │
│  landscape view, what's new     │
│  "relevant to X" queries        │
│  topic browse, full-text search │
└─────────────────────────────────┘
```

### Separation of concerns

**Python** handles format extraction only — converting source formats into clean
markdown. The Python code lives as a small package inside the `gw-kb` crate
(`crates/gw-kb/python/gw_kb_extract/`) and is invoked from Rust via **PyO3
embedding** (`pyo3` with the `auto-initialize` feature). This means:

- Rust is the call site — `gw-kb` is a normal Rust crate, not a maturin-built
  Python module
- The CPython interpreter is embedded in the binary at runtime
- A virtualenv with `trafilatura`, `pymupdf4llm`, `marker-pdf` is set up once
  and pointed at via `PYO3_PYTHON` at build time
- No staging files / no subprocess shelling — calls cross the Rust/Python
  boundary in-process via the GIL

**Rust (`gw-kb`)** handles chunking, embedding, storage, and the topic/domain
data model. It calls into `gw-memory`'s existing LanceDB and tantivy
infrastructure for retrieval.

**LLM** (via `gw-llm`) handles topic labeling, clustering refinement, summary
generation, and connection discovery. These are batch operations run at
organize/synthesize time, not on the query path.

---

## 2. Data Model

Three-level hierarchy: **Chunk → Topic → Domain**.

### 2.1 Source

A source is an ingested document — a URL, a local file, an RSS entry.

```sql
CREATE TABLE kb_sources (
    source_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    url             TEXT,                   -- NULL for local files
    file_path       TEXT,                   -- NULL for URLs
    title           TEXT NOT NULL,
    author          TEXT,
    published_at    TIMESTAMPTZ,            -- when the source was published
    ingested_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    content_hash    BYTEA NOT NULL,         -- SHA-256 of extracted markdown
    format          TEXT NOT NULL,           -- 'html', 'pdf', 'markdown', etc.
    metadata        JSONB DEFAULT '{}'      -- flexible: tags, feed name, etc.
);

CREATE UNIQUE INDEX idx_kb_sources_url ON kb_sources (url) WHERE url IS NOT NULL;
CREATE UNIQUE INDEX idx_kb_sources_path ON kb_sources (file_path) WHERE file_path IS NOT NULL;
```

The `content_hash` enables incremental re-ingestion: if the hash hasn't changed,
skip. If it has, re-chunk and update.

### 2.2 Chunk

A chunk is a passage from a source document — the atomic retrieval unit.

```sql
CREATE TABLE kb_chunks (
    chunk_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_id       UUID NOT NULL REFERENCES kb_sources(source_id) ON DELETE CASCADE,
    ordinal         INT NOT NULL,           -- position within source
    content         TEXT NOT NULL,
    char_offset     INT NOT NULL,           -- offset in source markdown
    char_length     INT NOT NULL,
    heading_path    TEXT[],                 -- e.g. ['## Architecture', '### Memory']
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX idx_kb_chunks_source ON kb_chunks (source_id, ordinal);
```

Chunks are also stored in:
- **LanceDB** — embedding vector for semantic retrieval
- **Tantivy** — full-text index for BM25 retrieval

Both reference `chunk_id` as the foreign key back to Postgres.

### 2.3 Topic

A topic is a cluster of related chunks across sources — the unit of understanding.

```sql
CREATE TABLE kb_topics (
    topic_id        UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    domain_id       UUID REFERENCES kb_domains(domain_id),
    label           TEXT NOT NULL,          -- short label, e.g. "KV-cache compression"
    slug            TEXT NOT NULL UNIQUE,    -- URL-safe identifier
    summary         TEXT,                   -- LLM-generated synthesis
    summary_at      TIMESTAMPTZ,            -- when summary was last generated
    chunk_count     INT NOT NULL DEFAULT 0,
    first_seen      TIMESTAMPTZ NOT NULL,   -- earliest chunk publication date
    last_seen       TIMESTAMPTZ NOT NULL,   -- most recent chunk publication date
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

### 2.4 Topic–Chunk membership

```sql
CREATE TABLE kb_topic_chunks (
    topic_id        UUID NOT NULL REFERENCES kb_topics(topic_id) ON DELETE CASCADE,
    chunk_id        UUID NOT NULL REFERENCES kb_chunks(chunk_id) ON DELETE CASCADE,
    relevance       REAL NOT NULL DEFAULT 1.0,  -- clustering confidence
    PRIMARY KEY (topic_id, chunk_id)
);
```

A chunk can belong to multiple topics (soft clustering).

### 2.5 Topic links

Connections between topics — "see also", "builds on", "contradicts".

```sql
CREATE TYPE kb_link_kind AS ENUM ('related', 'builds_on', 'contradicts', 'subtopic_of');

CREATE TABLE kb_topic_links (
    from_topic_id   UUID NOT NULL REFERENCES kb_topics(topic_id) ON DELETE CASCADE,
    to_topic_id     UUID NOT NULL REFERENCES kb_topics(topic_id) ON DELETE CASCADE,
    kind            kb_link_kind NOT NULL DEFAULT 'related',
    confidence      REAL NOT NULL DEFAULT 0.5,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (from_topic_id, to_topic_id)
);
```

### 2.6 Domain

A domain is a group of related topics — the orientation layer.

```sql
CREATE TABLE kb_domains (
    domain_id       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    label           TEXT NOT NULL,          -- e.g. "Agent memory systems"
    slug            TEXT NOT NULL UNIQUE,
    description     TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
```

---

## 3. Extraction Pipeline (Python)

### 3.1 Extractors

| Format | Tool | Notes |
|---|---|---|
| HTML / web pages | trafilatura | Boilerplate removal, metadata extraction (title, author, date) |
| PDF (born-digital) | pymupdf4llm | Fast, no ML deps, markdown output |
| PDF (scanned/complex) | marker | ML layout analysis, handles tables/equations. Known surya memory leaks — run as subprocess per batch |
| Markdown / plaintext | passthrough | Copy with minimal normalization |

### 3.2 In-process extraction via PyO3

There is no staging directory. Rust calls Python directly through PyO3 and
receives extracted content as a struct in memory:

```rust
pub struct Extracted {
    pub markdown: String,
    pub title: Option<String>,
    pub author: Option<String>,
    pub published_at: Option<DateTime<Utc>>,
    pub source_format: String,
    pub extractor: String,
}

pub fn extract_html(url: &str, html: &str) -> Result<Extracted>;
pub fn extract_pdf(path: &Path) -> Result<Extracted>;
pub fn extract_markdown(path: &Path) -> Result<Extracted>;
```

Each call acquires the GIL, invokes the matching function in
`gw_kb_extract.{html,pdf,markdown}`, and converts the returned Python dict into
the `Extracted` struct.

### 3.3 Python helper package layout

```
crates/gw-kb/python/
├── gw_kb_extract/
│   ├── __init__.py
│   ├── html.py          # trafilatura wrapper
│   ├── pdf.py           # pymupdf4llm + marker fallback
│   └── markdown.py      # passthrough + frontmatter parsing
├── requirements.txt     # trafilatura, pymupdf4llm, marker-pdf
└── README.md            # venv setup instructions
```

The Python package directory is added to `sys.path` once at startup (during
KB initialization) so that `py.import("gw_kb_extract.html")` works without
installing the package globally.

### 3.4 Build and runtime requirements

- System Python 3.10+ with development headers
- A virtualenv with `requirements.txt` installed
- `PYO3_PYTHON` environment variable pointing at the venv's `python` binary
  during `cargo build`
- A setup script (`crates/gw-kb/setup.sh`) creates the venv and prints the
  required exports

---

## 4. Ingest Pipeline (Rust)

### 4.1 Chunking strategy

Markdown-aware recursive splitting:

1. Split on `## ` headings (H2) — these are natural topic boundaries
2. If a section exceeds the target chunk size (~512 tokens), split on `### ` (H3)
3. If still too large, split on paragraph boundaries (double newline)
4. If still too large, split on sentence boundaries

Each chunk retains its **heading path** (e.g. `["## Architecture", "### Memory"]`)
for context when displaying results.

**Target chunk size:** 512 tokens with 64-token overlap between adjacent chunks.

Why markdown-aware over semantic chunking: we control the upstream output
(extraction always produces markdown), so we can exploit document structure
directly. Semantic chunking adds an embedding call per split decision — useful
when structure is unknown, overkill here.

### 4.2 Embedding

Use the same embedding model as `gw-memory` (currently via LanceDB's built-in
embedding functions or Ollama). Chunks are embedded and stored in a dedicated
LanceDB table (`kb_chunks`), separate from agent memory vectors.

### 4.3 Indexing

- **LanceDB:** chunk embedding + chunk_id + source_id + heading_path
- **Tantivy:** chunk text + chunk_id, with field boosts on title/headings
- **Postgres:** source metadata, chunk metadata, relationships

### 4.4 Deduplication

Before inserting, check `content_hash` against `kb_sources`:
- **New source:** insert source + chunks normally
- **Same hash:** skip entirely (already ingested)
- **Same URL/path, different hash:** re-extract, delete old chunks, insert new ones, mark affected topics for re-synthesis

---

## 5. Organization Pipeline

This is the layer that transforms a flat bag of chunks into a navigable knowledge
structure. It runs as a batch process after ingestion, and incrementally when
new documents are added.

### 5.1 Topic extraction

**Step 1: Chunk-level tagging**

For each new chunk, use the LLM to extract:
- 1-3 topic labels (short noun phrases, e.g. "KV-cache compression", "LoRA fine-tuning")
- Named entities (people, orgs, models, datasets)

Prompt is constrained to prefer existing topic labels when appropriate (provided
as context), creating new ones only when no existing label fits.

**Step 2: Topic assignment**

Match extracted labels to existing topics using embedding similarity
(label → existing topic label embeddings). Threshold: cosine > 0.85 = same topic,
0.70-0.85 = candidate (LLM confirms), < 0.70 = new topic.

**Step 3: Topic merging**

Periodically (or when topic count grows), run a merge pass:
- Find topic pairs with high chunk overlap (Jaccard > 0.3) or high label similarity
- LLM confirms whether they should merge
- Merge: combine chunks, regenerate label and summary

### 5.2 Domain assignment

Domains are coarser — dozens of topics map to a handful of domains. Assignment
is LLM-driven:

- Present all topic labels to the LLM, ask it to group into 5-15 domains
- New topics are assigned to existing domains by label similarity
- Domain list is regenerated periodically as the knowledge base grows

### 5.3 Topic linking

For each topic, find related topics via:
1. **Chunk co-occurrence:** topics that share sources tend to be related
2. **Embedding proximity:** topics whose centroid embeddings are close
3. **LLM judgment:** for top candidates, ask the LLM to classify the relationship type (related, builds_on, contradicts, subtopic_of)

### 5.4 Incremental updates

When new documents are ingested:
1. Tag and assign chunks to topics (step 5.1)
2. For each affected topic, check if summary is stale (new chunks since last synthesis)
3. Re-link affected topics (step 5.3)
4. Do NOT regenerate all domains/merges — these run on a schedule or manual trigger

---

## 6. Synthesis Pipeline

### 6.1 Topic summaries

For each topic (or when marked stale), generate a synthesis:

**Input to LLM:**
- Topic label
- All member chunks (or top-k by relevance if too many)
- Related topic labels (for cross-referencing)

**Output:**
- 2-4 paragraph summary integrating information across sources
- Key findings / state of the art
- Source attribution (which claims come from which sources)
- Cross-references to related topics

**Key property:** summaries should note contradictions and evolution, not just
aggregate. "Smith et al. found X, but Jones et al. showed Y with a larger
dataset" is more useful than "studies show X".

### 6.2 Delta reports

When re-synthesizing after new ingestion:

```
Topic: KV-cache compression
Last updated: 2026-03-20
New sources: 3 (since last synthesis)

What changed:
- New approach: per-layer budget allocation (Chen et al., 2026-03)
- Previous SOTA (uniform quantization) now shown to underperform by 2-5%
- New benchmark released: LongBench-v2
```

### 6.3 Landscape views

On-demand synthesis across a domain:

**Input:** all topics in a domain, their summaries, their link structure
**Output:** a map showing the major themes, how they relate, what's active vs.
stable, and where the gaps are.

This is the "I don't know what I don't know" interface — you ask "show me the
landscape of agent memory research" and get back a structured overview.

---

## 7. Graph Traversal

The topic graph is the primary navigation structure. The core traversal mechanism
is **spreading activation** — the same algorithm used in `gw-memory`'s hindsight
graph retrieval (`crates/gw-memory/src/graph.rs`), adapted from memory nodes to
topic nodes.

### 7.1 Why spreading activation

The alternatives and why they fall short for this use case:

- **Direct neighbors only** — shows you topics 1 hop away, but misses important
  topics that are well-connected through multiple paths. If topic A links to B
  and C, and B and C both link to D, then D is highly relevant to A — but a
  1-hop traversal never sees it.

- **Shortest-path / BFS** — finds reachable topics but treats all paths equally.
  A topic reachable through 3 strong links should rank higher than one reachable
  through 1 weak link, but BFS can't express this.

- **Embedding similarity** — finds topics that are *described* similarly, but
  misses structural relationships. Two topics can be strongly related (one builds
  on the other) without having similar embeddings.

Spreading activation handles all three: it follows the link structure, decays
with distance, accumulates through multiple paths, and respects edge weights.
A well-connected hub 3 hops away can outrank a weakly-linked direct neighbor.

### 7.2 Algorithm (adapted from gw-memory)

The existing `spreading_activation()` in `gw-memory/src/graph.rs` operates over
`memory_edges` with `(from_id, to_id, weight)` triples. The KB adaptation
operates over `kb_topic_links` with the same shape, plus typed edges.

```
Input:
  seeds        — starting topic(s), each with an initial activation score
  max_hops     — traversal depth (default: 3)
  decay        — per-hop decay factor (default: 0.5)
  link_weights — weight multipliers per link kind

Algorithm:
  1. Initialize activation map: seed topics → their initial scores
  2. frontier ← seed topic IDs
  3. For each hop (up to max_hops):
     a. Fetch all edges from frontier nodes in kb_topic_links
     b. For each edge (from → to, kind, confidence):
        - activation = parent_activation × decay × confidence × link_weight[kind]
        - If activation < threshold (0.001), skip
        - If activation > current activation for target, update and add to next frontier
     c. frontier ← next frontier
  4. Remove seeds from results
  5. Return topics sorted by activation score (highest first)
```

**Link kind weights** allow the traversal to prefer certain relationship types:

| Link kind | Default weight | Rationale |
|---|---|---|
| `builds_on` | 1.0 | Strong signal — if you're reading about X, things X builds on are directly relevant |
| `subtopic_of` | 0.9 | Navigate up/down the hierarchy |
| `related` | 0.7 | Useful but less directional |
| `contradicts` | 0.5 | Important to surface, but shouldn't dominate traversal |

These are configurable per query — a user exploring broadly might weight
`related` higher, while one drilling into a specific technique might boost
`builds_on` and `subtopic_of`.

### 7.3 Seed selection

How traversal starts depends on the entry point:

**From search:** Run hybrid search (vector + BM25) over chunks → map hit chunks
to their topics → use those topics as seeds, with activation proportional to
search score. This is the "I searched for X, show me the neighborhood" path.

**From a topic:** Single seed with activation 1.0. Pure graph exploration.

**From a query (no specific topic):** Embed the query → find nearest topic label
embeddings → top-k become seeds. This handles the "show me everything about
agent memory" case where the user hasn't landed on a specific topic yet.

**Multi-seed:** When starting from multiple topics (e.g. "what connects KV-cache
compression and quantization?"), activation propagates from all seeds
simultaneously. Topics reachable from *both* seeds accumulate activation from
both paths, naturally surfacing the bridge concepts.

### 7.4 Temporal filtering

Reuse the same pattern as `gw-memory`'s temporal-constrained graph traversal:
optionally restrict traversal to topics whose `last_seen` falls within a time
window. This supports queries like "what's related to X in the last month?"
without needing a separate query mechanism.

```sql
-- Build the temporal set: topics active in the window
SELECT topic_id FROM kb_topics
WHERE last_seen >= $1 AND first_seen <= $2
```

Pass this set as a filter to spreading activation — nodes outside the set are
not visited.

### 7.5 Traversal in practice

The traversal results feed into every surface-layer operation:

| Operation | Seeds | Hops | What's returned |
|---|---|---|---|
| `gw-kb topic <slug>` | Single topic | 2 | Topic summary + "Related topics" sidebar ranked by activation |
| `gw-kb search <query>` | Search hits → topics | 2 | Search results + "You might also explore" section |
| `gw-kb explore <query>` | Query → nearest topics | 3 | Ranked topic list with activation scores, grouped by domain |
| `gw-kb domain <slug>` | All topics in domain | 1 | Intra-domain connectivity map |
| `gw-kb bridge <topicA> <topicB>` | Both topics | 3 | Topics with high activation from both seeds (bridge concepts) |

The `explore` command is the primary discovery interface — it's how you answer
"what do I have in my knowledge base about X, and what's adjacent to it that I
might not have thought of?"

### 7.6 Implementation notes

The `gw-memory` implementation currently issues N+1 queries (one per hop). The
TODO in `graph.rs` notes this can be collapsed into a recursive CTE. For the KB
topic graph (likely hundreds to low thousands of nodes, not millions), the
per-hop approach is fine. If the graph grows, the CTE optimization applies
identically to `kb_topic_links`:

```sql
WITH RECURSIVE spread AS (
    SELECT to_topic_id, confidence, 1 AS hop
    FROM kb_topic_links WHERE from_topic_id = ANY($1)
    UNION ALL
    SELECT e.to_topic_id, s.confidence * $2 * e.confidence, s.hop + 1
    FROM spread s JOIN kb_topic_links e ON e.from_topic_id = s.to_topic_id
    WHERE s.hop < $3
)
SELECT DISTINCT ON (to_topic_id) to_topic_id, confidence
FROM spread ORDER BY to_topic_id, confidence DESC
```

---

## 8. Surface Layer (CLI)

### 8.1 Commands

```bash
# Ingest pipeline (extract + ingest + organize)
gw-kb ingest --url "https://..."
gw-kb ingest --url-list urls.txt
gw-kb ingest --dir ./papers/
gw-kb ingest --glob "./docs/**/*.md"

# Browse the knowledge base
gw-kb topics                        # list all topics, grouped by domain
gw-kb topic "kv-cache-compression"  # show topic summary + sources + links
gw-kb domain "agent-memory"         # show domain landscape

# Search
gw-kb search "attention mechanism"  # hybrid search over chunks
gw-kb search --topic "attention mechanism"  # find matching topics

# Explore (spreading activation from query)
gw-kb explore "agent memory"        # discover topics + neighbors via graph traversal
gw-kb bridge "kv-cache" "quantization"  # find bridge concepts between two topics

# What's new
gw-kb digest                        # delta report across all topics since last digest
gw-kb digest --domain "ml-research" # scoped to a domain
gw-kb digest --since 7d             # explicit time window

# Re-organize
gw-kb organize                      # re-run clustering, merge, re-link
gw-kb synthesize                    # regenerate stale summaries
gw-kb synthesize --topic "kv-cache" # regenerate specific topic

# Feeds (recurring sources)
gw-kb feed add --name "hf-blog" --url "https://huggingface.co/blog/feed.xml"
gw-kb feed sync                     # fetch new entries from all feeds
gw-kb feed list
```

### 8.2 Output format

Default: terminal-friendly markdown (rendered with formatting).
Optional: `--json` for programmatic use, `--markdown` for file output.

### 8.3 API endpoints

For integration with the rest of greatwheel (agents querying the KB):

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/kb/search` | Hybrid search over chunks |
| GET | `/api/kb/topics` | List topics (filterable by domain) |
| GET | `/api/kb/topics/:slug` | Topic detail + summary |
| GET | `/api/kb/domains` | List domains |
| GET | `/api/kb/digest` | Delta report |
| POST | `/api/kb/ingest` | Trigger ingestion from URL/path |

---

## 9. Storage Mapping

Reuses existing greatwheel infrastructure:

| Store | KB usage | Shared with |
|---|---|---|
| **Postgres** | Source metadata, chunk metadata, topic graph (nodes + edges + links), summaries, domains, feeds | Agent sessions, memories |
| **LanceDB** | Chunk embeddings (table: `kb_chunks`), topic label embeddings (table: `kb_topics`) | Agent memory vectors |
| **Tantivy** | Full-text chunk index (index: `kb`) | Agent memory BM25 |

The `gw-kb` crate depends on `gw-memory` for LanceDB and tantivy access, and
on `sqlx` directly for Postgres (KB-specific tables).

---

## 10. Crate Structure

```
crates/gw-kb/
├── Cargo.toml
└── src/
    ├── lib.rs           # public API
    ├── source.rs        # Source type, CRUD, content_hash logic
    ├── chunk.rs         # chunking strategy, Chunk type
    ├── topic.rs         # Topic, Domain types, topic graph operations
    ├── ingest.rs        # ingest pipeline: staging dir → chunks → store
    ├── organize.rs      # clustering, topic extraction, linking
    ├── synthesize.rs    # summary generation, delta reports, landscape
    ├── search.rs        # hybrid search over KB (wraps gw-memory)
    ├── feed.rs          # RSS/Atom feed tracking and sync
    ├── extract.rs       # Python extraction subprocess management
    └── cli.rs           # CLI subcommands
```

Dependencies:
- `gw-core` — shared types
- `gw-memory` — LanceDB + tantivy access
- `gw-llm` — LLM calls for organize + synthesize
- `sqlx` — Postgres for KB-specific tables
- `clap` — CLI

---

## 11. Implementation Plan

### Phase 1: Extract + Ingest (foundation)
- Python extraction script (trafilatura + marker + pymupdf4llm)
- Staging format (markdown + JSON sidecar)
- Rust ingest: chunking, embedding, storage in LanceDB + tantivy + Postgres
- CLI: `gw-kb ingest` and `gw-kb search`

### Phase 2: Organize (topic structure)
- LLM-based chunk tagging → topic extraction
- Topic assignment (embedding similarity + LLM confirmation)
- Topic linking (co-occurrence + embedding + LLM)
- Domain assignment
- CLI: `gw-kb topics`, `gw-kb topic <slug>`

### Phase 3: Synthesize (summaries + discovery)
- Per-topic summary generation
- Delta reports
- Domain landscape views
- CLI: `gw-kb digest`, `gw-kb domain <slug>`, `gw-kb synthesize`

### Phase 4: Feeds + incremental
- RSS/Atom feed tracking
- `gw-kb feed add/sync/list`
- Incremental re-organization on new ingestion
- Stale summary detection and regeneration

### Phase 5: Agent integration
- API endpoints in `gw-server`
- Host functions for rLM agents: `kb_search()`, `kb_topics()`, `kb_ingest()`
- Connect to autoresearch pipeline (Layer 1 from design-autoresearch.md)

---

## 12. Open Questions

1. **Embedding model:** Use the same model as gw-memory, or a specialized
   retrieval model (e.g. nomic-embed-text, BGE-M3)? Separate models allow
   optimization but add operational complexity.

2. **Chunk size tuning:** 512 tokens is a starting point. May want to experiment
   with larger chunks (1024) for synthesis quality vs. smaller chunks (256) for
   retrieval precision.

3. **Topic granularity:** How aggressively should we merge? Too few topics = each
   is a grab-bag. Too many = hard to browse. Probably needs manual tuning
   initially, with merge/split commands.

4. **Synthesis cost:** Generating summaries for hundreds of topics with an LLM is
   not free (even local). Should synthesis be on-demand (generate when viewed)
   rather than batch? Trade-off: batch is better for delta detection.

5. **Overlap with design-autoresearch.md:** The autoresearch design proposes
   `web_search()` and `web_fetch()` host functions, plus PDF ingestion. The KB
   ingestor should share infrastructure (extraction tools, embedding pipeline)
   but the KB adds the organization/synthesis layer on top. Need to decide if
   web_fetch → KB ingest is a first-class path.

6. **Marker memory leaks:** Monitor surya memory behavior. Fallback plan:
   subprocess isolation per document or per-batch, or switch to pymupdf4llm for
   non-complex PDFs and only use marker for scanned/complex layouts.
