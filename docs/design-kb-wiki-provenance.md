# Design: provenance-aware wiki view with entity highlights

**Status:** Drafted 2026-05-22. Two-milestone plan. M1 ships entity
highlights on the normalized markdown the wiki already renders. M2
swaps in original-source rendering (pdf.js for PDFs, replayed HTML
for web pages) with bounding-box / DOM-anchor overlays. The schema
in §3 is designed so M1 writes data that M2 reads without
backfill.

## 1. Why

The KB wiki view today (`KbDocWiki.tsx`, backed by
`crates/gw-kb/src/wiki.rs:344`) renders the normalized markdown we
stitched together from `kb_chunks`. It lists the entities the
extractor found in a sidebar, but offers no way to:

1. See *where* in the document an entity appears.
2. Click an entity-in-context to pull up structured info ("what is
   this thing, what other things does it relate to").
3. Read the document the way the author wrote it — equations,
   figures, tables, layout. `pymupdf4llm` is lossy by design.

(1) and (2) are a UX win at low cost. (3) is the real prize for a
research-assistant product: a researcher wants to read the *paper*,
not a markdown approximation of it.

The blocker: today's data path destroys the link from "entity E"
back to "byte range R in the original PDF". We extract markdown,
chunk it, run NER on the chunk text, and store `(chunk_id,
entity_id)` edges with no character spans (`migrations/013_kb_entities.sql:79-85`).
The original file is thrown away (`crates/gw-kb/src/ingest.rs:151-160`).

This doc fixes that in a forward-compatible way.

## 2. Span model

The central design decision is what coordinate system entity
mentions live in. The chunked normalized-markdown string is the
*only* universal substrate — every source format passes through it
on the way to entity extraction — so that's what we anchor to. We
then layer per-format provenance on top to translate normalized
coordinates back to original-source coordinates.

```
                     ┌──────────────────────┐
   original PDF/HTML │ source bytes         │   stored as blob (§4)
                     └──────────┬───────────┘
                                │ pymupdf / trafilatura
                                │ + provenance atoms (§3.2)
                                ▼
                     ┌──────────────────────┐
                     │ normalized markdown  │   kb_chunks.content
                     └──────────┬───────────┘
                                │ NER (with spans, §3.1)
                                ▼
                     ┌──────────────────────┐
                     │ entity mentions      │   kb_entity_mentions
                     └──────────────────────┘
```

Two new concepts:

- **Mention** — one occurrence of an entity in a chunk. Carries
  `(norm_start, norm_end)` offsets into `kb_chunks.content`.
  Multiple mentions of the same entity in the same chunk produce
  multiple rows. This is what the highlight overlay reads.
- **Provenance atom** — a minimal contiguous range of normalized
  text whose mapping to the original source is monotone. Each atom
  carries `(norm_start, norm_end)` plus a format-tagged
  `source_locator` (PDF page + bbox, HTML DOM path, plain-text
  line range). The renderer composes atoms to translate any
  mention span back to source regions.

Mentions and atoms are decoupled. M1 only writes mentions and
ignores atoms; M2 populates atoms at extraction time and the
renderer joins through them. No mention row needs to change between
milestones.

## 3. Schema

### 3.1 `kb_entity_mentions` — one row per occurrence

Replaces (or sits alongside — see §6) the existing
`kb_chunk_entity_links`. The old table is (chunk, entity) → one row
which throws away per-occurrence position. We want per-occurrence.

```sql
CREATE TABLE kb_entity_mentions (
    mention_id  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    chunk_id    UUID NOT NULL REFERENCES kb_chunks(chunk_id)    ON DELETE CASCADE,
    entity_id   UUID NOT NULL REFERENCES kb_entities(entity_id) ON DELETE CASCADE,
    norm_start  INT  NOT NULL,                  -- char offset within chunk.content (UTF-8 byte? see §7)
    norm_end    INT  NOT NULL,                  -- exclusive
    surface     TEXT NOT NULL,                  -- exact matched surface form (for verification + display)
    role        TEXT,                           -- "subject" | "object" | "mention" | NULL (carried over from old link table)
    confidence  REAL NOT NULL DEFAULT 1.0,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (norm_end > norm_start)
);

CREATE INDEX idx_kb_entity_mentions_chunk  ON kb_entity_mentions (chunk_id, norm_start);
CREATE INDEX idx_kb_entity_mentions_entity ON kb_entity_mentions (entity_id);
```

`(chunk_id, norm_start)` is the natural reading order for highlight
rendering — given a chunk, pull all mentions sorted by position.

`surface` is redundant with `chunk.content[norm_start..norm_end]`
but stored anyway: it (a) lets us validate spans haven't drifted
after re-extraction, (b) shows up in entity-card "appears as: …"
without re-fetching chunk text.

### 3.2 `kb_chunk_provenance` — atoms back to original

Optional, populated only when the extractor has source-coordinate
data. M1 leaves this empty.

```sql
CREATE TABLE kb_chunk_provenance (
    chunk_id        UUID NOT NULL REFERENCES kb_chunks(chunk_id) ON DELETE CASCADE,
    atom_ordinal    INT  NOT NULL,
    norm_start      INT  NOT NULL,    -- char offset within chunk.content
    norm_end        INT  NOT NULL,
    locator_kind    TEXT NOT NULL,    -- "pdf_bbox" | "html_dom" | "text_lines"
    locator         JSONB NOT NULL,   -- format-specific (see below)
    PRIMARY KEY (chunk_id, atom_ordinal),
    CHECK (norm_end > norm_start)
);

CREATE INDEX idx_kb_chunk_provenance_chunk_range
    ON kb_chunk_provenance (chunk_id, norm_start);
```

Locator JSON shapes (typed in Rust as an enum, serialized to JSONB):

```jsonc
// locator_kind = "pdf_bbox"
{ "page": 12, "bbox": [x0, y0, x1, y1] }    // 1-indexed page, PDF user-space coords

// locator_kind = "html_dom"
{ "css": "main > article > p:nth-of-type(4)", "text_offset": 137, "text_length": 42 }

// locator_kind = "text_lines"
{ "line_start": 88, "line_end": 91 }
```

A single contiguous mention may span multiple atoms (multi-line
text, column wrap). The renderer fetches the atoms whose `[norm_start,
norm_end)` intersects the mention range and emits one highlight
rect / DOM range per atom.

### 3.3 `kb_source_blobs` — original bytes

```sql
CREATE TABLE kb_source_blobs (
    source_id    UUID PRIMARY KEY REFERENCES kb_sources(source_id) ON DELETE CASCADE,
    content_type TEXT NOT NULL,           -- "application/pdf" | "text/html" | ...
    storage_kind TEXT NOT NULL,           -- "inline" | "fs" | "s3"
    inline_bytes BYTEA,                   -- populated iff storage_kind = "inline"
    fs_path      TEXT,                    -- populated iff storage_kind = "fs"
    s3_key       TEXT,                    -- populated iff storage_kind = "s3"
    byte_length  BIGINT NOT NULL,
    stored_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    CHECK (
        (storage_kind = 'inline' AND inline_bytes IS NOT NULL) OR
        (storage_kind = 'fs'     AND fs_path      IS NOT NULL) OR
        (storage_kind = 's3'     AND s3_key       IS NOT NULL)
    )
);
```

Three storage backends, single table, picked by config. Default to
`inline` until corpora exceed a few hundred MB; switch to `fs` for
the BrowseComp-style corpora; reserve `s3` for production. The blob
is keyed on `source_id` (not `content_hash`) so it cascades on
source deletion. `kb_sources.content_hash` already exists for dedup
checks.

## 4. Original-source storage

The user leaned toward storing originals. Concrete plan:

- **PDF sources:** Always store. PDFs are bounded in size, and
  refetching is unreliable (arxiv link rot, paywalled mirrors).
- **HTML sources:** Store the fetched bytes. Re-rendering live HTML
  is a security and stability nightmare (broken CSS, mixed content,
  JS, tracking) — much better to snapshot at ingest time and serve
  the snapshot. We'll need to consider whether to also store
  referenced assets (images, CSS) for full fidelity; out of scope
  for M2 v1 — render text-only with the original DOM structure.
- **Plain-text / markdown sources:** Store iff it differs from the
  normalized form. Usually it doesn't, so skip.

Config knob (`config/greatwheel.toml`):

```toml
[kb.blobs]
storage_kind   = "fs"             # "inline" | "fs" | "s3"
fs_root        = "./data/kb-blobs"
max_inline_kb  = 256              # fall back to fs above this even when storage_kind = inline
```

Open question: GDPR-style takedowns. `ON DELETE CASCADE` from
`kb_sources` handles row deletion; fs/s3 cleanup needs a sweeper.
Acceptable to defer; flagged in §7.

## 5. Per-format extraction changes

### 5.1 PDFs (M2)

Today: `pymupdf4llm.to_markdown(path)` (`crates/gw-kb/python/gw_kb_extract/pdf.py:15-39`)
returns markdown only. We replace with a direct `fitz` (PyMuPDF)
walk that emits both the markdown *and* a parallel list of atoms:

```python
# pseudocode
doc = fitz.open(path)
md_buf = StringIO()
atoms = []
for page in doc:
    for block in page.get_text("dict")["blocks"]:
        for line in block["lines"]:
            for span in line["spans"]:
                start = md_buf.tell()
                md_buf.write(span["text"])
                end = md_buf.tell()
                atoms.append({
                    "norm_start": start, "norm_end": end,
                    "locator_kind": "pdf_bbox",
                    "locator": {"page": page.number + 1, "bbox": span["bbox"]},
                })
        md_buf.write("\n")
```

Real implementation needs to match pymupdf4llm's heading/list/table
heuristics so existing chunking still works — non-trivial but
mechanical. Atoms are chunk-relative after chunking: the chunker
already produces `(char_offset, char_length)` within the source
markdown, so we slice the atom list to each chunk and rebase
offsets.

### 5.2 HTML (M2)

`trafilatura` already returns extracted text. We need it to also
return DOM paths for each extracted text node. Trafilatura's
internal `extract_text` walks the parsed tree — we fork the extract
call to keep `(node_path, text_offset, text_length)` per emitted
text segment. Atom locator becomes `{"css": <path>, ...}`.

For rendering, the blob is reparsed client-side and we use
`document.evaluate` (or a CSS selector lib) + `Range` API to
highlight.

### 5.3 Plain text / markdown (M1 + M2)

Atom = identity mapping. Often skip the table entirely; renderer
falls back to "normalized == original" when no atoms exist.

### 5.4 Entity extractor (M1)

`entities.rs:94` currently sends chunk text to an LLM and parses
JSON back with `{label, canonical_form, confidence}`. We extend the
prompt to ask for spans:

```jsonc
{
  "label": "ColBERT",
  "canonical_form": "ColBERT",
  "confidence": 0.95,
  "occurrences": [
    { "surface": "ColBERT", "char_start": 142, "char_end": 149 },
    { "surface": "ColBERT-v2", "char_start": 420, "char_end": 430 }
  ]
}
```

LLM-reported offsets drift (especially around whitespace and
unicode). After parsing we *verify* each occurrence by checking
`chunk.content[char_start..char_end] == surface`; if mismatch, we
fall back to a string search for `surface` within the chunk and use
the first unclaimed match. If no match, drop the occurrence with a
warn-level trace event but keep the entity (it still appears in the
sidebar, just unhighlighted).

This means **M1 ships even if span quality is imperfect** —
mentions degrade gracefully into "entity present, no highlight".

## 6. Migration from `kb_chunk_entity_links`

Two options:

- **(a) Drop the old table.** `kb_chunk_entity_links` becomes a
  view over `kb_entity_mentions` (`SELECT DISTINCT chunk_id,
  entity_id, role FROM kb_entity_mentions`). Existing callers
  (`linking.rs`, wiki sidebar) read the view unchanged.
- **(b) Keep both.** Old table stays as the chunk×entity boolean;
  new table is the per-occurrence detail. Writers populate both.

Recommend **(a)** — the old table is just a deduplicated projection
of the new one. Less drift. The CASCADE semantics are identical
either way. Risk: any future caller that wants to write to the link
table directly would have to be rewritten — acceptable given the
current call sites are all in `entities.rs`.

## 7. Open questions

1. **Char offsets: UTF-8 bytes or chars?** Postgres `TEXT` is bytes
   under the hood, but Rust `&str` indexing is bytes, and JS string
   indexing is UTF-16 code units. Recommend: store *byte offsets
   into the UTF-8 encoding of `chunk.content`*, document this in
   the migration comment, and have the frontend convert when
   rendering. Avoids the JS surrogate-pair trap. (`tiktoken`-style
   token offsets are tempting but tie us to a tokenizer.)
2. **Re-extraction & span drift.** When a chunk is re-extracted
   (e.g. extractor upgrade), how do we re-link mentions? Easiest:
   `DELETE FROM kb_entity_mentions WHERE chunk_id = $1` and
   repopulate. Loses any human-edited annotations — none today, so
   fine.
3. **Mentions in summaries / non-chunk text.** Topic summaries and
   entity summaries can also mention entities. Out of scope —
   `kb_entity_mentions` is chunk-anchored.
4. **Blob takedown / GC.** Need a sweeper that reconciles
   `kb_source_blobs` against fs/s3. Deferrable.
5. **Multi-column PDFs / RTL text.** Atom-per-span handles columns
   naturally (each span has its own bbox). RTL is untested but
   pymupdf reports bboxes correctly; verify with one Arabic PDF.

## 8. Milestones

### M1 — highlights on normalized markdown

Scope:

- Add `kb_entity_mentions` migration. (§3.1)
- Add `kb_chunk_entity_links` view replacing the table. (§6 option a)
- Extend entity-extraction LLM prompt to return occurrences with
  spans; verify and persist. (§5.4)
- Backfill: one-shot CLI command that re-runs extraction over
  existing chunks. (Acceptable to require this; corpus sizes are
  small.)
- Backend: extend `fetch_wiki_doc` to return per-section
  `mentions: [{entity_id, norm_start, norm_end, surface}]` aligned
  to section offsets.
- Frontend: in `KbDocWiki.tsx`, render each section's markdown with
  highlights overlaid (inject `<mark data-entity-id=…>` spans
  during markdown render). Click a mark → open entity card pane.
- Entity card pane: new widget reading `entity_id` → calls a new
  `kb_entity_detail` host function returning
  `{label, kind, aliases, summary, related: [...], mentions_in_doc}`.

Acceptance: open a wiki doc, see entity names highlighted inline
where they actually appear, click one, see a card with that
entity's structured info. No original PDF.

### M2 — original-source rendering

Scope:

- `kb_chunk_provenance` migration. (§3.2)
- `kb_source_blobs` migration + storage layer. (§3.3, §4)
- PDF extractor swap: pymupdf4llm → direct fitz with atoms. (§5.1)
- HTML extractor: trafilatura with DOM paths. (§5.2)
- Backend: new endpoint `kb_source_blob(source_id)` streaming
  bytes; new endpoint `kb_source_overlays(source_id, [entity_id])`
  returning per-page (or per-DOM-node) highlight rects derived from
  mentions + provenance atoms.
- Frontend: PDF view using pdf.js with an SVG overlay layer
  driven by the overlays endpoint. HTML view using an iframe
  (sandboxed) over the blob with overlay using DOM Ranges.
- Wiki view gets a toggle: "normalized" (M1 view) vs "original"
  (M2 view). Entity card and highlight interactions identical in
  both.

Acceptance: open a paper, see the original PDF with the same
entities highlighted on the rendered pages, click → entity card.
Toggle to normalized view falls back to M1 behavior.

## 9. Out of scope

- Editing entities or annotations from the wiki view.
- Multi-document wiki (cluster view, `KbClusterWiki.tsx`) — same
  mechanism applies but defer the UI work.
- Co-reference resolution ("it", "the model") — extractor returns
  surface-form mentions only.
- Entity-card relationship graph visualization — card shows a list,
  not a graph; defer.
