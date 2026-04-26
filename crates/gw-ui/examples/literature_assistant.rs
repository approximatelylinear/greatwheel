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
use std::sync::Arc;

use gw_core::{LoopEvent, Plugin, PluginContext, PluginError, PluginManifest, SessionId};
use gw_engine::GreatWheelEngine;
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
const EMBEDDING_MODEL: &str = "nomic-embed-text-v1.5";
const DEFAULT_OPENAI_MODEL: &str = "gpt-5.4";

const SYSTEM_PROMPT: &str = r###"You are a literature scout exploring arXiv. The user asks about a research topic; you search arXiv, embed the abstracts, project them to 2D, and emit an EntityCloud widget so the user can browse the result spatially. Click a point to inspect a paper.

**Output format (CRITICAL):** Every response you produce MUST be a single fenced Python code block, i.e. starts with ```python on its own line and ends with ``` on its own line. Do NOT use OpenAI tool-calling syntax. The harness only executes Python in fenced code blocks.

Data host functions:
  - arxiv_search(query: str, max_results: int = 30) -> list[{"id", "title", "summary", "authors", "published", "category", "url"}]
      Hits the public arXiv API. Returns up to max_results papers most-relevant to the query.
  - embed_papers(texts: list[str]) -> list[list[float]]
      Returns one vector per text. Wraps the local Ollama embedding endpoint.
  - project_2d(vectors: list[list[float]]) -> list[[float, float]]
      Hand-rolled PCA. Returns one (x, y) per input vector, deterministic, scaled to roughly [-1, 1].

UI host functions (same as the other demos):
  - emit_widget(session_id, kind, payload, multi_use=False, follow_up=False, scope=None) -> {"widget_id"}
  - supersede_widget(old_widget_id, session_id, kind, payload, ...)
  - pin_to_canvas(widget_id)
  - pin_below_canvas(widget_id)
  - FINAL("text") — terminates the turn with a chat narration.

The frontend's json-render catalog includes one literature-specific widget type:

  - EntityCloud: payload {"type": "EntityCloud", "points": [{"id", "label", "x", "y", "kind"?}, ...], "highlight"?: {<id>: true}}.
    Each point is rendered at its (x, y) position. Click emits an interaction with `data = {"pointId": <id>}` so you can drill in.

# Turn 1 — user asks a topic question

Use **two iterations**.

**Iteration 1: search + embed + project. Don't FINAL.**

```python
# 1. Search arXiv. Pull the topic from the user's last message verbatim.
papers = arxiv_search(query="<the user's topic, lightly cleaned>", max_results=30)
# 2. Build one short string per paper for embedding (title + first sentence of abstract is enough).
texts = [f"{p['title']}. {p['summary'][:400]}" for p in papers]
# 3. Embed.
vectors = embed_papers(texts)
# 4. Project.
coords = project_2d(vectors)
print("PIPELINE_OK", len(papers), "papers")
```

**Iteration 2: build the widget payload + emit + FINAL.**

```python
papers = arxiv_search(query="<same topic>", max_results=30)
texts = [f"{p['title']}. {p['summary'][:400]}" for p in papers]
vectors = embed_papers(texts)
coords = project_2d(vectors)

points = []
for p, (x, y) in zip(papers, coords):
    points.append({
        "id": p["id"],
        "label": p["title"][:80],
        "x": x,
        "y": y,
        "kind": "paper",
    })

result = emit_widget(
    session_id=gw_session_id,
    kind="a2ui",
    multi_use=True,  # the cloud is a persistent palette — clicks don't terminate it
    payload={"type": "EntityCloud", "points": points},
)
pin_to_canvas(widget_id=result["widget_id"])  # PRIMARY slot — the cloud is the workspace

FINAL(f"Plotted {len(points)} arXiv papers on the topic. Click a point to read its abstract; nearby points are semantically related.")
```

# Turn 2+ — user clicked a point in the cloud

Each click delivers a WidgetInteraction with `data = {"pointId": <arxiv_id>}`. Treat it as "show me this paper."

```python
arxiv_id = "<from data.pointId>"
# Re-search to get the current paper record. (For v1 we don't persist
# anything between turns; v2 will swap this for kb_topic / kb_entity.)
hits = arxiv_search(query=f"id:{arxiv_id}", max_results=1)
paper = hits[0] if hits else None
if paper is None:
    FINAL(f"Couldn't refind paper {arxiv_id}.")
else:
    # Build a "paper detail" Column with title, authors, published, summary, link.
    detail = {
        "type": "Column",
        "children": [
            {"type": "Text", "text": paper["title"]},
            {"type": "Text", "text": ", ".join(paper["authors"][:6]) + (" et al." if len(paper["authors"]) > 6 else "")},
            {"type": "Text", "text": f"arXiv {paper['id']} · {paper['published'][:10]} · {paper['category']}"},
            {"type": "Text", "text": paper["summary"]},
        ],
    }
    result = emit_widget(
        session_id=gw_session_id,
        kind="a2ui",
        scope={"kind": "paper", "key": arxiv_id},
        payload=detail,
    )
    pin_below_canvas(widget_id=result["widget_id"])  # AUX slot — the detail sits below the cloud
    FINAL(f"Pinned {paper['title'][:60]}…")
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
"###;

// ── Literature plugin ───────────────────────────────────────────────

struct LiteraturePlugin {
    http: Arc<reqwest::Client>,
    embedder: Arc<OllamaClient>,
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
        }
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
            ],
            requires: vec![],
            priority: 0,
        }
    }

    fn init(&self, ctx: &mut PluginContext) -> Result<(), PluginError> {
        let http = self.http.clone();
        ctx.register_host_fn_async("arxiv_search", None, move |_args, kwargs| {
            let http = http.clone();
            async move {
                let query = kwargs
                    .get("query")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| PluginError::HostFunction("query required (string)".into()))?
                    .trim()
                    .to_string();
                let max_results = kwargs
                    .get("max_results")
                    .and_then(|v| v.as_u64())
                    .unwrap_or(30)
                    .min(50) as usize;
                arxiv_search(&http, &query, max_results)
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("arxiv_search: {e}")))
            }
        });

        let embedder = self.embedder.clone();
        ctx.register_host_fn_async("embed_papers", None, move |_args, kwargs| {
            let embedder = embedder.clone();
            async move {
                let texts = kwargs
                    .get("texts")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| PluginError::HostFunction("texts required (list[str])".into()))?
                    .iter()
                    .map(|v| v.as_str().unwrap_or("").to_string())
                    .collect::<Vec<_>>();
                if texts.is_empty() {
                    return Ok(Value::Array(vec![]));
                }
                let vectors = embedder
                    .embed(&texts)
                    .await
                    .map_err(|e| PluginError::HostFunction(format!("embed: {e}")))?;
                let json_vectors: Vec<Value> = vectors
                    .into_iter()
                    .map(|v| Value::Array(v.into_iter().map(num).collect()))
                    .collect();
                Ok(Value::Array(json_vectors))
            }
        });

        ctx.register_host_fn_sync("project_2d", None, move |_args, kwargs| {
            let raw = kwargs
                .get("vectors")
                .and_then(|v| v.as_array())
                .ok_or_else(|| {
                    PluginError::HostFunction("vectors required (list[list[float]])".into())
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
            let coords = pca_project_2d(&vectors);
            let out: Vec<Value> = coords
                .into_iter()
                .map(|(x, y)| json!([f64::from(x).clamp(-2.0, 2.0), f64::from(y).clamp(-2.0, 2.0)]))
                .collect();
            Ok(Value::Array(out))
        });

        Ok(())
    }
}

fn num(f: f32) -> Value {
    serde_json::Number::from_f64(f as f64)
        .map(Value::Number)
        .unwrap_or(Value::Null)
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
    s.split_whitespace().collect::<Vec<_>>().join(" ")
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

    // Embedder: always go via Ollama (nomic-embed-text-v1.5), because
    // OpenAI text-embedding-3-small produces 1536-dim vectors which
    // would slow PCA down for no demo benefit. The chat LLM can still
    // be OpenAI if OPENAI_API_KEY is set.
    let embedder = Arc::new(OllamaClient::new(
        OLLAMA_URL.into(),
        OLLAMA_URL.into(),
        OLLAMA_MODEL.into(),
        EMBEDDING_MODEL.into(),
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
            EMBEDDING_MODEL.into(),
        );
        (ollama, format!("ollama:{OLLAMA_MODEL}"))
    };
    let loop_llm: Box<dyn gw_loop::LlmClient> =
        Box::new(OllamaLlmClient::new(chat_client).with_think(Some(false)));

    let lit = LiteraturePlugin::new(embedder);

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
