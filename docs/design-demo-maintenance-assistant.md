# Demo — Maintenance Assistant (paper figures → 3D + text linking)

A research-assistant demo that ingests a hand-picked PDF (think maintenance
manual / engineering paper), extracts its figures, mints a 3D model per
figure via [meshwell](../../meshwell), and renders a side-by-side
"figure ↔ text" view where mentions of named parts highlight bboxes on
the figure. The 3D mesh is shown alongside as a rotatable artifact —
not part-segmented (see §"Out of scope").

Parallel in structure to `literature_assistant.rs` / EntityCloud, but
the widget is a paper viewer instead of a scatter plot.

## v1 scope

- Corpus: hand-picked PDFs in `examples/maintenance_assistant/papers/`
  (3–5 files). No retrieval, no arXiv search, no KB ingest.
- One agent system prompt; the user names a paper (by filename or
  index) and the agent renders the viewer widget.
- Figure extraction: PyMuPDF, run server-side as a host function.
- Per-figure VLM annotation: bboxes for named parts in **figure-local
  normalized coords** (`[x0, y0, x1, y1]` in `[0,1]`). Ollama VLM —
  see §VLM bake-off for candidates.
- 3D reconstruction: HTTP call to meshwell at `:8420`, new route
  `POST /api/generate-from-image` (see §meshwell changes).
- Text linking: simple substring match between part labels and the
  paper's extracted body text. Click a label → highlight matching
  spans in text; hover a text mention → highlight the figure bbox.
- Widget: AG-UI custom widget, three panels:
  - Left: paper body text, with part mentions wrapped in a span the
    UI can highlight on hover.
  - Top-right: figure image with overlaid bbox rectangles.
  - Bottom-right: `<model-viewer>` for the GLB.

## Out of scope (v1)

- **3D part segmentation.** SF3D outputs a single fused mesh. No
  part-level highlighting on the 3D model; the mesh is "look, we can
  generate it" eye candy. Stretch goal in §v2.
- Multi-paper corpora / KB integration / arXiv search.
- OCR — only digital-PDF text (PyMuPDF text extraction). Scanned PDFs
  out of scope.
- Caption-extraction quality — use the simplest "text below figure
  bbox" heuristic; don't fight it.

## Architecture

```
PDF (fixture)
  │
  ├─▶ extract_figures(pdf_path) ───────▶  [{figure_id, png_bytes, caption, page}, ...]
  │       (PyMuPDF host fn)
  │
  ├─▶ extract_text(pdf_path) ──────────▶  full body text
  │
  ▼
For each figure:
  ├─▶ annotate_figure(png, caption) ──▶  [{label, bbox: [x0,y0,x1,y1]}, ...]
  │       (Ollama VLM host fn)
  │
  └─▶ generate_mesh(png) ──────────────▶  glb_url
          (POST meshwell /api/generate-from-image, poll until ready)

         ▼
  emit PaperViewer widget {
    text: str,
    text_mentions: [{label, char_ranges: [[start, end], ...]}],
    figures: [
      {figure_id, image_url, caption, parts: [{label, bbox}], mesh_url}
    ],
  }
```

### Host functions (new, in the agent example)

| name | purpose |
|---|---|
| `list_papers() -> list[{id, filename, title}]` | enumerate fixtures |
| `extract_figures(paper_id) -> list[{figure_id, image_url, caption, page}]` | PyMuPDF, cached on disk |
| `extract_text(paper_id) -> str` | PyMuPDF text |
| `annotate_figure(figure_id) -> list[{label, bbox}]` | Ollama VLM, cached |
| `generate_mesh(figure_id) -> {mesh_url, status}` | POST meshwell, cached by figure hash |
| `match_mentions(text, labels) -> list[{label, char_ranges}]` | simple substring match |

All cached server-side under `examples/maintenance_assistant/cache/<paper_id>/`
so re-runs are instant and we don't burn GPU minutes per demo.

### Meshwell changes (small)

Add one route to `meshwell/meshwell/web/routes.py`:

```python
@router.post("/api/generate-from-image")
async def generate_from_image(
    request: Request,
    file: UploadFile = File(...),
    description: str = Form(""),
):
    """Image-to-3D: skip SDXL concept gen, feed image straight to SF3D.

    Creates a new asset, writes the upload as concept.png, then runs
    the reconstruction stage onward (multi-view → SF3D → cleanup).
    """
    config = _get_config(request)
    asset = AssetMeta(prompt=description or "imported figure",
                      stage=PipelineStage.CONCEPT_DONE)
    asset_dir = asset.dir(config.data_dir)
    asset_dir.mkdir(parents=True, exist_ok=True)
    concept_path = asset.concept_path(config.data_dir)
    with open(concept_path, "wb") as f:
        f.write(await file.read())
    asset.meta_path(config.data_dir).write_text(asset.model_dump_json(indent=2))

    async def run():
        async for _ in run_pipeline(asset, config, start_after_concept=True):
            pass

    task = asyncio.create_task(run())
    request.app.state.active_jobs[asset.id] = task
    return {"asset_id": asset.id, "status": "started"}
```

`run_pipeline` already iterates stages from an arg-controlled starting
point in `pipeline.py`; if not, add a `start_after_concept: bool` kwarg
that skips the `IMAGE_GEN` stage when set.

### Widget

New AG-UI widget `PaperViewer`. Wire-shape mirrors the
`EntityCloud` pattern (stable JSON contract so the agent can rebuild
without UI churn):

```jsonc
{
  "type": "paper_viewer",
  "paper_id": "manual-pump-2024",
  "text": "...full body text...",
  "text_mentions": [
    { "label": "impeller",        "char_ranges": [[412, 420], [1893, 1901]] },
    { "label": "discharge flange", "char_ranges": [[521, 537]] }
  ],
  "figures": [
    {
      "figure_id": "fig-3",
      "image_url": "/cache/manual-pump-2024/fig-3.png",
      "caption": "Fig. 3 — Cutaway view of the centrifugal pump assembly.",
      "parts": [
        { "label": "impeller",        "bbox": [0.31, 0.42, 0.55, 0.68] },
        { "label": "discharge flange", "bbox": [0.74, 0.18, 0.94, 0.35] }
      ],
      "mesh_url": "/meshwell/assets/<asset_id>/mesh.glb"
    }
  ]
}
```

Frontend: React component in `frontend/src/components/PaperViewer.tsx`
(parallel to whatever EntityCloud has). Uses `<model-viewer>` for the
GLB. Bbox overlay is plain SVG over the image.

## File layout

```
crates/gw-ui/examples/maintenance_assistant.rs       # agent + host fns
examples/maintenance_assistant/papers/*.pdf          # fixtures
examples/maintenance_assistant/cache/                # gitignored
frontend/src/components/PaperViewer.tsx              # widget
docs/design-demo-maintenance-assistant.md            # this doc
```

## VLM bake-off

Bbox grounding via LLM is the weakest part of the pipeline (mesh
quality aside) — different VLMs vary by 10–20pp on grounding
benchmarks. We will spot-check 2–3 fixtures by hand against this
shortlist before committing. Findings recorded back in this doc.

**Tier 1 (primary candidates, both on Ollama, May 2026):**

| model | size | notes |
|---|---|---|
| `qwen3.5:9b` | 6.6GB | Already used elsewhere in the project (BrowseComp benchmark). Multimodal at every size; no separate `-vl` tag. Native part of the stack — zero new infra. |
| `qwen3-vl:8b` | ~8GB | Dedicated VL line. Qwen VL family has the strongest bbox-grounding track record among open VLMs (trained with explicit `<bbox>` tokens). Slightly older base model than qwen3.5 but VL-specialized. |

**Tier 2 (worth a glance if Tier 1 disappoints):**

- `qwen3-vl:30b` — larger sibling of Tier 1, only if 8b's bbox accuracy is borderline.
- `mistral-small3.1:24b` — vision + 128k ctx, recent. Untested for bbox grounding.
- `kimi-k2.5` — "native multimodal agentic" framing, very new.
- `minicpm-v:8b` — older but punches above its weight on small-detail VQA.

**Dropped:**

- `llama3.2-vision` — ~1yr old, no bbox-grounding specialization, superseded.
- `qwen2.5vl` — Qwen3 supersedes it; no reason to test.

**Fallback if all VLMs are too sloppy: Grounding DINO 1.6 Pro.**
Not an LLM — a dedicated open-vocabulary detector that takes a text
query ("impeller, discharge flange, ...") and produces bboxes
directly. Achieves 55.4% AP on COCO zero-shot, materially better than
any VLM at this task. Tradeoff: extra Python service (not on Ollama),
and the VLM is still needed to *generate* the part list from the
caption. Architecture stays compatible — only the `annotate_figure`
host function changes internally.

## Open questions

1. **VLM choice** — answered by §VLM bake-off, pending empirical
   spot-check on 2–3 fixtures.
2. **Mesh quality on technical diagrams.** SF3D was trained on natural
   images / characters. Cutaway engineering drawings may reconstruct
   poorly. Acceptable for demo if even *some* figures produce
   recognisable meshes; if all figures fail, the demo collapses to
   "2D bbox highlighting" and we drop the 3D panel. See §SF3D
   fine-tuning sketch for a longer-horizon fix.
3. **Sync vs. async generation.** Mesh gen takes ~30s on a 4090.
   Decision: **pre-bake all fixture meshes on first load** of each
   paper, cache to disk, serve instantly thereafter.

## SF3D fine-tuning sketch (out of v1, but worth a north-star note)

SF3D underperforms on engineering figures because its training
distribution is natural images and stylized 3D characters, not
line-art / cutaways / exploded views. Two plausible fine-tune paths,
neither cheap but both tractable on a 4090:

1. **LoRA on the image-conditioning tower.** Freeze the triplane
   decoder; train low-rank adapters on the DINOv2 image encoder using
   pairs of (engineering figure, ground-truth CAD render). Smallest
   intervention, may close half the gap.
2. **Full fine-tune with a synthetic CAD-render dataset.** Use a
   public CAD corpus (ABC dataset, Fusion 360 Gallery, GrabCAD) →
   render each part from a randomized camera → resulting (image, GLB)
   pairs are SF3D's native training format. The harder lift but the
   one that actually moves the needle.

Either way, the demo's wire shape doesn't change — only the GLB at
the other end of `/api/generate-from-image` gets sharper.

## v2 sketch (not in scope, but design-compatible)

- 2D bbox → 3D surface highlight via camera ray-cast from the SF3D
  front-view onto the mesh. Approximate but cheap. Adds one optional
  field `parts[].mesh_uvs` or `parts[].mesh_face_ids` to the wire shape.
- Swap the fixture corpus for a real KB-backed retrieval (gw-kb
  ingest of a folder of PDFs).
- Entity extraction via `gw-kb` instead of substring match, so part
  *aliases* and *coreferences* link to the same bbox.
