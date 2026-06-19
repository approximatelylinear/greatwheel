# Design: manual paper upload for the literature assistant

**Status:** Drafted 2026-05-23. Extends
[`design-demo-literature-assistant.md`](design-demo-literature-assistant.md)
§10 ("Deliberately out of scope: user-uploaded papers, arXiv only") —
revisited now that the KB persists across sessions and `gw-kb`'s
ingest primitives already cover the formats we'd need.

## 1. Goal

Let a user add a paper to their library without going through
`arxiv_search`. Three motivating cases:

- **Paper they already know they want.** They have an arXiv id or URL
  in hand; running a topical search to "discover" it is wasted motion
  and is rate-limited by arXiv besides.
- **Paper not on arXiv.** Conference PDFs, lab tech reports, blog
  posts. The agent currently has no way to read these.
- **Paper they have a local copy of.** Paywalled work, drafts, papers
  shared by collaborators.

The non-goal is *changing how the agent explores* — see §3.

## 2. What we lean on (already exists)

- **`gw-kb::ingest::ingest_url(stores, url)`** — fetches HTML or PDF,
  runs the right extractor (pymupdf4llm for PDFs, html→md for HTML),
  rewrites `arxiv.org/abs/<id>` to `/pdf/<id>` so the full paper is
  ingested, not just the abstract page.
- **`gw-kb::ingest::ingest_file(stores, path)`** — local file ingest;
  detects format by extension (`pdf | md | markdown | txt`).
- **`KbStores` + per-source dedup by URL** — re-running the same URL
  is a no-op (just bumps `last_seen`). Re-uploading a file under the
  same canonical URL collapses cleanly.
- **`EntityExtractionWorker.enqueue_source(source_id)`** — every
  current `arxiv_search` ingest already queues here. Manual uploads
  reuse the same handle.
- **`LitSessionManager`** — per-session loop already exists, and
  `lookup_session_user(pg, sid)` lets a request mounted on
  `/sessions/{sid}/...` find the owning `user_id` without auth.
- **Drawer / popover patterns** — `WikiPane` (slide-in) and
  `WorkspaceDrawer` (overlay list with fetch + actions) are the
  templates. The composer affordance is small enough to be inline.

## 3. Scope decisions

Resolved up-front so the doc stays narrow:

1. **Inputs supported:** arXiv id/URL paste, arbitrary URL (HTML or
   PDF), local PDF upload, **bulk** (multi-file drop or newline-
   separated id list).
2. **Affordance:** paperclip button left of the message input, opens
   a small popover anchored above the composer (paste field +
   drop-zone). See §4.
3. **Post-ingest behaviour:** background ingest only. Show a toast
   ("Added 'RAG: a survey' — 47 papers in library") and queue for
   entity extraction. **No automatic widget emission, no synthetic
   turn.** Paper becomes available to `kb_search` / `kb_get_wiki` /
   the next typed query. This keeps ingest as a library operation
   and exploration as the agent's job.

## 4. UX

### 4.1 Composer affordance

```
┌─ app-footer ───────────────────────┐
│ [📎] Type a topic or paste...  [↵] │
└────────────────────────────────────┘
          ↓ click 📎
  ┌─ Add paper ────────────────────────────┐
  │ Paste arXiv ID, URL, or multiple       │
  │ (one per line):                        │
  │ ┌────────────────────────────────────┐ │
  │ │ 2504.13684                         │ │
  │ │ https://arxiv.org/abs/2310.06825   │ │
  │ │                                    │ │
  │ └────────────────────────────────────┘ │
  │                                        │
  │   — or —                               │
  │                                        │
  │ ┌────────────────────────────────────┐ │
  │ │       Drop PDFs here               │ │
  │ │       or click to choose           │ │
  │ └────────────────────────────────────┘ │
  │                                        │
  │ Queue:                                 │
  │   ✓ 2504.13684  "RAG: a survey"        │
  │   ⏳ paper2.pdf (extracting…)          │
  │   ✗ broken-url.com  (HTTP 404)         │
  │                                        │
  │                       [ Add ] [Close]  │
  └────────────────────────────────────────┘
```

- Single popover, single textarea (paste arXiv ids, URLs, or a mix —
  one per line). Drop-zone is the file input. **No tab UI** — the
  two paths coexist visually.
- Click outside or `Esc` closes; in-flight uploads continue in the
  background (popover state mirrors a small per-session queue kept
  in the existing session store).
- Drag-drop activates a full-popover highlight when files enter; if
  the user drops files onto the canvas instead, nothing happens
  (drag-anywhere is explicitly out — see §10).

### 4.2 Status surfaces

Three concentric status views, in order of how prominently they
notify the user:

- **In-popover queue** — line per item with state (pending → fetching
  → extracting → indexed | failed). Sticks around for the popover's
  lifetime; cleared on close.
- **Toast** — one per terminal result. "Added *RAG: a survey* —
  N papers in library" on success; "Couldn't add *2999.99999* —
  arXiv returned 404" on failure. Toast component already lives in
  `frontend/src/components/` (we have toasts for the cancel button)
  — reuse.
- **Library counter** — the existing
  `arxiv_search` narration line ("…across N papers in your library")
  picks up the new total on the next turn. No proactive update.

### 4.3 What the agent sees

Nothing immediately. The next time the user types a topic, the
already-ingested rows surface naturally:

- `kb_search` and `kb_get_wiki` find them.
- `arxiv_search`'s deduping `ingest_inline` re-encounters them as
  *Unchanged* (URL match) — the full-text version persisted by manual
  upload takes precedence over the abstract that `arxiv_search`
  would otherwise write. See §6.3 on the asymmetry.
- The entity worker has already extracted entities by the time the
  agent runs (or is partway through; the `kb_extractor_status` host
  fn already exists to narrate progress).

## 5. Backend

### 5.1 New HTTP endpoints

All mounted on the literature_assistant router, alongside the
existing `/sessions/{sid}/*` routes (so the lazy-spawn middleware
gives us `user_id` for free via `lookup_session_user`):

```
POST   /sessions/{sid}/uploads/url
       Body: {"url": "https://..." | "<arxiv_id>"}
       Response: 202 {"upload_id": "<uuid>", "status": "queued"}

POST   /sessions/{sid}/uploads/file
       multipart/form-data: file=<binary>
       Response: 202 {"upload_id": "<uuid>", "status": "queued"}

POST   /sessions/{sid}/uploads/batch
       Body: {"urls": ["...", "..."]}
       Response: 202 {"upload_ids": [...]}

GET    /sessions/{sid}/uploads
       Response: [{"upload_id", "kind", "ref", "status",
                   "source_id?", "error?", "created_at"}]
```

The endpoints are session-scoped because that's where the routing
middleware lives, but **the KB itself is global** — a paper uploaded
in session A is visible to all of user X's sessions and (today)
every other user's sessions too. That's consistent with current
arxiv_search behaviour; cross-user partitioning is a separate
question handled by the broader auth roadmap, not this design.

### 5.2 New host functions (optional, deferred)

We can ship the UI without exposing upload as a host fn, since the
agent has no reason to call it. But a small fn keeps the door open
for "agent reads a URL the user mentioned in chat":

```python
kb_ingest_url(url: str) -> {"source_id": str, "outcome": "inserted" | "updated" | "unchanged"}
# Under kb.write capability. Deferred to v2.
```

Not in v1 — adds a capability bit we'd need to design (the only
existing kb capability is `kb.read`).

### 5.3 Wiring

Single new module `crates/gw-ui/src/uploads_api.rs`:

```rust
pub struct UploadsApiConfig {
    pub max_file_bytes: usize,      // default 50 MiB
    pub allowed_mime_prefixes: Vec<&'static str>,  // ["application/pdf", "text/html", "text/markdown", "text/plain"]
}

pub fn router(
    pg: PgPool,
    kb: Arc<KbStores>,
    entity_worker: Option<Arc<EntityExtractionWorker>>,
    cfg: UploadsApiConfig,
) -> Router;
```

State held in a single struct passed via `axum::Extension`. Uploads
register a row in a new lightweight table (§5.4) and spawn a
`tokio::task` that calls `ingest_url` or `ingest_file` and updates
the row's status.

Why a task and not inline: PDF extraction can take 10–30s for big
papers; we don't want the POST hanging. The `202 queued` + polling
pattern fits the existing `WorkspaceDrawer` reload-key idiom.

### 5.4 Schema

New table; one row per upload attempt:

```sql
-- 018_kb_uploads.sql
CREATE TABLE kb_uploads (
    upload_id    UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    session_id   UUID REFERENCES sessions(id) ON DELETE SET NULL,
    user_id      UUID REFERENCES users(id) ON DELETE SET NULL,
    kind         TEXT NOT NULL CHECK (kind IN ('url', 'file')),
    ref          TEXT NOT NULL,           -- url, or original filename
    source_id    UUID REFERENCES kb_sources(source_id) ON DELETE SET NULL,
    status       TEXT NOT NULL CHECK (status IN ('queued','fetching','extracting','indexed','failed')),
    outcome      TEXT,                    -- 'inserted' | 'updated' | 'unchanged' on success
    error        TEXT,                    -- non-null iff status='failed'
    bytes        BIGINT,                  -- present for kind='file' or after fetch
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE INDEX kb_uploads_session_recent
    ON kb_uploads (session_id, created_at DESC);
CREATE INDEX kb_uploads_user_recent
    ON kb_uploads (user_id, created_at DESC);
```

Rationale for *not* folding into `kb_sources.metadata`:

- We want to record **failed attempts** (so the user sees them in
  the popover queue) — failures don't produce a `kb_source` row.
- We want a per-user upload history view eventually (§9), which a
  jsonb metadata sub-field would force into expensive scans.
- The cost is tiny (~1KB per upload) and it's append-only.

### 5.5 Stash on `kb_sources.metadata`

When ingest succeeds, set:

```json
{
  "uploaded_by_user_id": "<uuid>",
  "upload_method": "manual_url" | "manual_file",
  "upload_session_id": "<uuid>"
}
```

These are *additive* — `arxiv_search`'s metadata writes
`{"arxiv_id", "category", "published", "authors"}` and we want both
sets when both paths see the same paper. Use `metadata = metadata || $1`
(jsonb concat) instead of overwriting.

## 6. Edge cases

### 6.1 Duplicate uploads

`upsert_source` dedups by URL. The interesting cases:

- **Same arXiv id, uploaded twice via the URL path:** second is
  `Unchanged`. Toast says "Already in library." Status row is
  `indexed` with `outcome='unchanged'`. No re-enqueue for the entity
  worker.
- **File upload of a paper already there from arxiv_search:**
  `ingest_inline` (from arxiv_search) wrote only the abstract under
  `https://arxiv.org/abs/<id>`. `ingest_file` doesn't know the file
  came from arXiv unless the user tells us, so it'll go in under a
  fresh URL (`file://<tempname>` from `ingest_file`'s path
  serialization). **This is bad** — two source rows for the same
  paper. **Mitigation:** when handling file uploads, accept an
  optional `?arxiv_id=<id>` query param the frontend sets when the
  filename matches `^(\d{4}\.\d{4,5})\.pdf$`; the handler then calls
  a small helper that constructs the canonical arXiv URL and routes
  through `ingest_url` to fetch text from PDF bytes already in hand
  (extract path), but persisting under the canonical URL. See §7 on
  the small refactor this needs in `gw-kb`.
- **Same URL, both manual and from arxiv_search:** see §6.3.

### 6.2 Big PDFs / extraction failures

- Cap multipart body at 50 MiB (config; arXiv papers are usually
  ≤5 MB). Reject larger with 413.
- `extract_pdf` errors land in `status='failed'`, `error=<message>`.
  Popover queue shows the error; toast shows a truncated version.
- Password-protected PDFs: pymupdf4llm raises, we surface as
  `extraction_failed: encrypted`.

### 6.3 The full-text vs abstract asymmetry

Current `arxiv_search` writes only `title + abstract` to the KB.
`ingest_url(arxiv_url)` writes the **full paper**. After this
change, a paper can be in the KB in two states:

| Discovery path           | Stored text             |
|--------------------------|-------------------------|
| arxiv_search (today)     | title + abstract        |
| manual URL paste         | title + full body       |
| manual file upload       | title + full body       |

When the same paper is encountered by both paths, the chunk-hash
check in `finish_ingest` correctly treats the full-text version as
*Updated* and re-chunks. Order matters:

- `arxiv_search` first, then manual upload → upgrades to full text.
  ✓ Re-extraction is queued.
- Manual upload first, then `arxiv_search` → `ingest_inline` would
  *downgrade* to just the abstract. ✗ Bad.

**Fix:** make `ingest_inline` (the path arxiv_search uses) detect
that the existing `kb_sources` row has a chunk_count >> what an
abstract would produce, and skip the upsert in that case. Cheap
guard: `if existing_chunks > 3 and incoming_text_chars < 5000:
treat as Unchanged`. Comment it as "abstract-vs-full-text safety
net" so it doesn't look like dead code.

Worth flagging that the cleaner fix is to make `arxiv_search` call
`ingest_url` instead of `ingest_inline` — then everything is
full-text and the asymmetry vanishes. But that's a bigger change
(adds N PDF fetches to every search) and a separate decision.

### 6.4 Non-paper URLs

`ingest_url` doesn't care whether a URL points to a paper. A user
who pastes a blog post will get a KB source with the blog text.
Acceptable: the EntityCloud will fold those entities in like any
other paper. The wiki view (`KbDocWiki`) renders them as a doc page
without a paper-specific layout, which is also fine.

We *don't* try to detect "is this a paper" — heuristics here
(check for arxiv.org host, check for `<title>` patterns) would be
fragile and the failure mode is benign.

### 6.5 Bulk: partial failures

Each item gets its own `kb_uploads` row and is processed in
parallel up to a small concurrency cap (4 — bounded by PDF
extraction CPU). Failures don't cancel siblings. The popover queue
shows mixed states cleanly.

### 6.6 Cancellation

`Esc` in the popover doesn't cancel in-flight ingests — they keep
running. The user sees the result in the toast even if the popover
is closed. A future iteration could add per-row cancel; not in v1.

## 7. Small refactor needed in gw-kb

`extract_pdf_bytes` (private in `ingest.rs:148`) is what we need for
the "uploaded PDF, but persist under the canonical arXiv URL" case
from §6.1. Expose it:

```rust
pub fn extract_pdf_bytes(bytes: &[u8]) -> Result<Extracted, KbError>;
pub async fn ingest_bytes_as_pdf(
    stores: &KbStores,
    url: &str,
    bytes: &[u8],
) -> Result<IngestReport, KbError>;
```

`ingest_bytes_as_pdf` is the new entry point: it calls
`extract_pdf_bytes`, then `finish_ingest(stores, Some(url), None,
extracted)`. Cheap, ~15 lines.

## 8. Frontend changes

### 8.1 Files

```
NEW   frontend/src/components/AddPaperPopover.tsx
NEW   frontend/src/api/uploads.ts
EDIT  frontend/src/components/MessageInput.tsx     (add 📎 slot)
EDIT  frontend/src/App.tsx                          (wire popover state)
EDIT  frontend/src/styles.css                       (.add-paper-* classes)
EDIT  frontend/src/store/session.ts                 (upload queue slice)
```

### 8.2 `AddPaperPopover` shape

```ts
interface Props {
  sessionId: string;
  onClose: () => void;
  // optional anchor element for positioning
  anchor?: HTMLElement | null;
}
```

State:
- `text: string` — textarea content for the paste field.
- `queue: UploadItem[]` — local mirror of in-flight uploads
  (populated by `useUploadQueue(sessionId)` hook, which polls
  `GET /sessions/{sid}/uploads?since=<ts>` every 1.5s while the
  popover is open).
- Drop-zone state via `onDragEnter`/`onDragLeave`/`onDrop`.

Submit logic on **Add**:
1. Parse textarea: split on newlines, trim, drop empties.
2. Classify each line:
   - arXiv id pattern (`^\d{4}\.\d{4,5}$`) → `https://arxiv.org/abs/<id>`
   - http(s) URL → as-is
   - anything else → mark as bad input locally, don't POST.
3. If >1 item, POST `/uploads/batch`. If 1, POST `/uploads/url`.
4. For each file selected, POST `/uploads/file` individually
   (multipart per file, cap=4 parallel).
5. Don't clear the textarea on submit — let the user see what they
   added; clear on Close.

### 8.3 `MessageInput` change

Add a `leadingSlot` prop or just inline the paperclip:

```tsx
<form className="message-input">
  <button type="button" className="message-input-attach"
    onClick={() => setAddPaperOpen(true)} aria-label="Add paper">
    📎
  </button>
  <input ... />
  <button>Send</button>
</form>
```

The popover is rendered by `App.tsx` (so it can survive
`MessageInput` re-mounts) and positioned via `position: absolute`
above the footer.

### 8.4 Toast plumbing

Once a queue item transitions to `indexed` or `failed`, fire a
toast. The hook keeps the last-seen status per upload_id to avoid
re-toasting on every poll tick.

## 9. Out of scope (v1)

- **Drag-drop onto the canvas / EntityCloud.** Discoverability is
  worse than the explicit button and we'd need a different drop-
  handler hierarchy. Add later if user-tested.
- **Agent-driven upload (`kb_ingest_url` host fn).** Deferred per
  §5.2; needs a new capability bit.
- **Cross-user partitioning of the KB.** Manual uploads land in the
  same global corpus as arxiv_search results today. This is a
  property of the existing system, not something this design
  changes.
- **DOI / Semantic Scholar / OpenReview connectors.** Just URLs and
  files; a metadata-aware connector layer is its own design.
- **OCR for scanned PDFs.** pymupdf4llm needs a text layer; scans
  fail with the same error as encrypted PDFs and we don't try to
  recover.
- **Reprocessing UI.** "Re-extract this paper" / "Replace with a
  better PDF" — both possible by re-uploading the same URL, but no
  bespoke UI.
- **User-level upload history view.** The `kb_uploads` table is
  there to support this later, but the UI is just the in-popover
  queue for v1.
- **Agent prompt updates.** No changes to `SYSTEM_PROMPT`. The
  agent's `kb_search` / `kb_get_wiki` paths already cover the
  surface that matters; the explicit-arxiv-id short-circuit
  (§408 of `literature_assistant.rs`) keeps working for both
  arxiv-search-ingested and manual-upload-ingested papers.

## 10. Acceptance

1. Click 📎 → popover opens above the composer.
2. Paste `2504.13684` → click Add → within ~3s the queue shows
   `indexed`, a toast appears, library count goes up. Closing the
   popover and asking "show wiki for 2504.13684" works on the next
   turn.
3. Paste a non-arXiv URL (e.g., a conference PDF) → ingest succeeds,
   `kb_search` finds chunks from it.
4. Drop a 4-page PDF into the drop zone → queue cycles through
   `fetching → extracting → indexed`. EntityCloud on the next
   topical search includes entities from it.
5. Paste 5 ids on separate lines → batch submit; ≤4 process in
   parallel; partial failures show per-row.
6. Paste a malformed URL → row shows `failed` with the underlying
   error text; toast surfaces it.
7. Re-upload the same arXiv id → second attempt shows `unchanged`;
   library count unchanged; no second entity-worker enqueue.
8. PDF >50 MiB → rejected with 413; popover shows
   "file too large (max 50 MiB)".
9. Reload the page mid-upload → POST already returned 202; the
   server task continues; on next popover open the queue endpoint
   returns the final state. (The toast is lost — acceptable.)
10. Open the same user in two browser windows; upload in one →
    library count in the other updates on the next typed turn
    (no live broadcast in v1).

## 11. Scope estimate

2–3 days:

- ~0.5d: `kb_uploads` migration + `ingest_bytes_as_pdf` helper +
  the `ingest_inline` "abstract-vs-full-text" safety net.
- ~0.5d: `uploads_api.rs` (router, multipart handler, task
  spawner, status row updates).
- ~1d: `AddPaperPopover` + `useUploadQueue` + toast plumbing +
  paperclip wiring in `MessageInput`.
- ~0.5d: acceptance pass + manual QA on each input type.

Biggest engineering risks:

- **PDF extraction stability.** pymupdf4llm has surprised us before
  (memory, encrypted files). Cap body size, wrap with a timeout,
  surface the error cleanly — don't try to recover.
- **The asymmetry in §6.3.** Easy to forget the safety net and
  re-introduce abstract-downgrade later. The comment matters.
- **Polling cost.** A 1.5s poll while the popover is open is fine
  for a single user; if we add server-pushed events later, fold it
  into the existing SSE stream rather than a new channel.

## 12. Files to touch (cheat sheet)

```
NEW   crates/gw-ui/src/uploads_api.rs
NEW   crates/gw-kb/migrations/018_kb_uploads.sql
EDIT  crates/gw-kb/src/ingest.rs                    (expose extract_pdf_bytes,
                                                     add ingest_bytes_as_pdf,
                                                     add abstract-vs-full-text guard)
EDIT  crates/gw-kb/src/lib.rs                       (re-export)
EDIT  crates/gw-ui/examples/literature_assistant.rs (mount uploads router)

NEW   frontend/src/components/AddPaperPopover.tsx
NEW   frontend/src/api/uploads.ts
EDIT  frontend/src/components/MessageInput.tsx
EDIT  frontend/src/App.tsx
EDIT  frontend/src/styles.css
EDIT  frontend/src/store/session.ts                 (optional: upload queue slice)
```

## 13. Reading order for a fresh Claude session

1. This doc.
2. `design-demo-literature-assistant.md` §10 (the original "out of
   scope" line that this revisits) and §15 (sessions API +
   LitSessionManager) for the mount-point context.
3. `crates/gw-kb/src/ingest.rs` — `ingest_url`, `ingest_file`,
   `ingest_inline`, `extract_pdf_bytes`.
4. `crates/gw-ui/src/sessions_api.rs` as the closest pattern for
   the new `uploads_api.rs` (router shape, axum extractors,
   error mapping).
5. `frontend/src/components/WorkspaceDrawer.tsx` for the
   fetch-poll-render-action loop the popover queue mirrors.
