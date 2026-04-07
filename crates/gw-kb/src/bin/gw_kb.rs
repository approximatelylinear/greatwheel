//! gw-kb CLI: ingest documents and search the knowledge base.

use std::path::PathBuf;
use std::sync::Arc;

use clap::{Parser, Subcommand};
use gw_llm::OllamaClient;
use pyo3::types::PyAnyMethods;
use sqlx::postgres::PgPoolOptions;
use sqlx::PgPool;
use tracing_subscriber::EnvFilter;

use gw_kb::classify::{classify_edges, ClassifyOpts};
use gw_kb::embed::Embedder;
use gw_kb::index::{KbLanceStore, KbTantivyStore};
use gw_kb::ingest::{ingest_file, ingest_url, KbStores};
use gw_kb::linking::{
    link, nearest_topics_to_query, neighbors_of, spread_from_seeds, EdgeDirection, LinkOpts,
    LinkedNeighbor, SpreadOpts,
};
use gw_kb::merge::{merge_topics, MergeOpts};
use gw_kb::organize::{organize, OrganizeOpts};
use gw_kb::synthesize::{fetch_summary, synthesize_topics, SynthesizeOpts};
use gw_kb::search::hybrid_search;
use gw_kb::source::{
    fetch_source, list_chunks_for_source, list_sources, resolve_source_id,
};
use gw_kb::topics::{
    fetch_topic_by_slug, list_chunks_for_topic, list_topic_summaries,
};
use gw_kb::KbError;

#[derive(Parser, Debug)]
#[command(name = "gw-kb", about = "Greatwheel knowledge base ingestor")]
struct Cli {
    /// Postgres connection URL.
    #[arg(long, global = true, env = "DATABASE_URL")]
    database_url: Option<String>,

    /// LanceDB directory.
    #[arg(long, global = true, env = "GW_KB_LANCE_PATH", default_value = "data/kb-lancedb")]
    lance_path: String,

    /// Tantivy index directory.
    #[arg(long, global = true, env = "GW_KB_TANTIVY_PATH", default_value = "data/kb-tantivy")]
    tantivy_path: String,

    /// Ollama base URL (used for embeddings).
    #[arg(long, global = true, env = "OLLAMA_URL", default_value = "http://localhost:11434")]
    ollama_url: String,

    /// Embedding model identifier. Passed to sentence-transformers (loaded
    /// in-process via PyO3, NOT through Ollama — Ollama's nomic wrapper is
    /// broken on short label inputs).
    #[arg(long, global = true, env = "GW_KB_EMBEDDING_MODEL", default_value = "nomic-ai/nomic-embed-text-v1.5")]
    embedding_model: String,

    /// Embedding dimension (must match the model).
    /// 768 for nomic-embed-text-v1.5, 1024 for mxbai-embed-large / bge-m3.
    #[arg(long, global = true, env = "GW_KB_EMBEDDING_DIM", default_value_t = 768)]
    embedding_dim: i32,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand, Debug)]
enum Command {
    /// Ingest a document from a URL or local file.
    Ingest {
        #[arg(long, conflicts_with = "file")]
        url: Option<String>,
        #[arg(long, conflicts_with = "url")]
        file: Option<PathBuf>,
    },

    /// Hybrid search (BM25 + vector) over the knowledge base.
    Search {
        /// Query string.
        query: String,
        /// Max results to return.
        #[arg(short = 'k', long, default_value_t = 5)]
        k: usize,
    },

    /// List all ingested sources, newest first.
    Sources {
        /// Maximum number of sources to show.
        #[arg(long, default_value_t = 50)]
        limit: i64,
    },

    /// Show one source's metadata and (optionally) its chunks.
    Source {
        /// Full UUID or any unique prefix (>= 4 chars).
        id: String,

        /// List all chunks for this source.
        #[arg(long)]
        chunks: bool,

        /// Show the full content of one chunk by ordinal.
        #[arg(long)]
        chunk: Option<i32>,
    },

    /// Tag untagged chunks with topics + entities (Phase 2).
    Organize {
        /// Process at most this many chunks.
        #[arg(long)]
        limit: Option<usize>,
        /// Restrict to one source (full UUID or unique prefix).
        #[arg(long)]
        source: Option<String>,
        /// Re-tag chunks even if already tagged.
        #[arg(long)]
        retag: bool,
    },

    /// List topics, ordered by chunk count.
    Topics {
        #[arg(long, default_value_t = 50)]
        limit: i64,
    },

    /// Show one topic and its member chunks.
    Topic {
        /// Topic slug.
        slug: String,
        /// Max member chunks to show.
        #[arg(long, default_value_t = 20)]
        chunks: i64,
    },

    /// Generate 2-4 paragraph LLM summaries for rich topics (Phase 3).
    Synthesize {
        /// Process at most N topics.
        #[arg(long)]
        limit: Option<usize>,
        /// Only topics with chunk_count >= N are summarized.
        #[arg(long, default_value_t = 5)]
        min_chunks: i32,
        /// Regenerate existing summaries instead of skipping them.
        #[arg(long)]
        regenerate: bool,
        /// Regenerate only summaries whose summary_at < topic updated_at.
        #[arg(long)]
        stale_only: bool,
        /// Synthesize only one specific topic by slug.
        #[arg(long)]
        topic: Option<String>,
    },

    /// Find and collapse duplicate topics. Detects candidates by
    /// label-vector cosine, then auto-merges high-confidence pairs and
    /// asks the LLM to confirm medium-confidence pairs.
    Merge {
        /// Pairs at or above this label-cosine merge without LLM confirmation.
        #[arg(long, default_value_t = 0.92)]
        auto_threshold: f32,
        /// Pairs in [ask_threshold, auto_threshold) get an LLM yes/no.
        #[arg(long, default_value_t = 0.85)]
        ask_threshold: f32,
        /// Process at most N candidate pairs (highest cosine first).
        #[arg(long)]
        limit: Option<usize>,
        /// Print what would happen but don't persist anything.
        #[arg(long)]
        dry_run: bool,
    },

    /// Type the existing topic edges using the LLM (subtopic_of /
    /// builds_on / contradicts / related). Idempotent by default —
    /// only re-types edges currently marked `related`.
    Classify {
        /// Process at most N edges (highest-confidence first).
        #[arg(long)]
        limit: Option<usize>,
        /// Re-classify edges that already have a non-`related` kind.
        #[arg(long)]
        reclassify: bool,
    },

    /// Build the topic link graph from co-occurrence + embedding similarity.
    Link {
        /// Co-occurrence: minimum shared chunks for a pair to link.
        #[arg(long, default_value_t = 2)]
        min_shared: i64,
        /// Embedding: minimum content-vector cosine for a pair to link.
        #[arg(long, default_value_t = 0.65)]
        min_cosine: f32,
        /// Drop edges with confidence below this floor.
        #[arg(long, default_value_t = 0.20)]
        min_confidence: f32,
    },

    /// Spreading-activation discovery from a free-text query.
    /// Picks the nearest topics as seeds, then walks the link graph.
    Explore {
        /// Free-text query.
        query: String,
        /// Number of seed topics from the query embedding.
        #[arg(long, default_value_t = 3)]
        seeds: usize,
        /// Max hops to walk from each seed.
        #[arg(long, default_value_t = 3)]
        hops: usize,
        /// Per-hop activation decay.
        #[arg(long, default_value_t = 0.5)]
        decay: f32,
        /// Number of activated topics to display.
        #[arg(short = 'k', long, default_value_t = 15)]
        limit: usize,
    },

    /// Smoke test the embedded Python interpreter.
    PyPing,
}

#[tokio::main]
async fn main() -> Result<(), KbError> {
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")))
        .init();

    let cli = Cli::parse();

    // Build stores up-front for commands that need them. Cheap construction;
    // py-ping doesn't need them but it's the only exception.
    let need_stores = !matches!(cli.command, Command::PyPing);
    let stores = if need_stores { Some(build_stores(&cli).await?) } else { None };

    match cli.command {
        Command::PyPing => {
            gw_kb::extract::init_python_path()?;
            pyo3::Python::with_gil(|py| -> Result<(), KbError> {
                let module = py.import_bound("gw_kb_extract")?;
                let doc: String = module.getattr("__doc__")?.extract().unwrap_or_default();
                println!("gw_kb_extract loaded ok");
                println!("---\n{}", doc.trim());
                Ok(())
            })?;
        }
        Command::Ingest { url, file } => {
            let stores = stores.as_ref().expect("stores built above");
            let report = if let Some(u) = url {
                ingest_url(stores, &u).await?
            } else if let Some(f) = file {
                ingest_file(stores, &f).await?
            } else {
                return Err(KbError::Other("must provide --url or --file".into()));
            };
            println!(
                "ingest: {} (outcome: {:?}, chunks: {})",
                report.source.title, report.outcome, report.chunks_written
            );
            println!("  source_id   = {}", report.source.source_id);
            println!("  format      = {}", report.source.source_format);
            println!("  extractor   = {}", report.source.extractor);
            if let Some(author) = &report.source.author {
                println!("  author      = {}", author);
            }
            if let Some(pub_at) = &report.source.published_at {
                println!("  published   = {}", pub_at);
            }
        }
        Command::Sources { limit } => {
            let stores = stores.as_ref().expect("stores built above");
            let rows = list_sources(&stores.pg, limit).await?;
            if rows.is_empty() {
                println!("no sources ingested yet");
                return Ok(());
            }
            // Header
            println!(
                "{:<10} {:<6} {:>6}  {:<19}  {}",
                "ID", "FORMAT", "CHUNKS", "INGESTED", "TITLE"
            );
            for s in &rows {
                let short = s.source_id.simple().to_string()[..8].to_string();
                let when = s.ingested_at.format("%Y-%m-%d %H:%M:%S").to_string();
                let title = if s.title.chars().count() > 60 {
                    let short_title: String = s.title.chars().take(57).collect();
                    format!("{}…", short_title)
                } else {
                    s.title.clone()
                };
                println!(
                    "{:<10} {:<6} {:>6}  {:<19}  {}",
                    short, s.source_format, s.chunk_count, when, title
                );
            }
            println!("\n{} source(s)", rows.len());
        }
        Command::Source { id, chunks, chunk } => {
            let stores = stores.as_ref().expect("stores built above");
            let source_id = resolve_source_id(&stores.pg, &id).await?;
            let source = fetch_source(&stores.pg, source_id).await?;

            println!("source_id    = {}", source.source_id);
            println!("title        = {}", source.title);
            if let Some(url) = &source.url {
                println!("url          = {}", url);
            }
            if let Some(path) = &source.file_path {
                println!("file_path    = {}", path);
            }
            println!("format       = {}", source.source_format);
            println!("extractor    = {}", source.extractor);
            if let Some(author) = &source.author {
                println!("author       = {}", author);
            }
            if let Some(p) = &source.published_at {
                println!("published_at = {}", p);
            }
            println!("ingested_at  = {}", source.ingested_at);
            println!("updated_at   = {}", source.updated_at);
            let hash_hex: String = source
                .content_hash
                .iter()
                .map(|b| format!("{:02x}", b))
                .collect();
            println!("content_hash = sha256:{}", hash_hex);

            let all_chunks = list_chunks_for_source(&stores.pg, source.source_id).await?;
            println!("chunks       = {}", all_chunks.len());

            if let Some(target_ord) = chunk {
                match all_chunks.iter().find(|c| c.ordinal == target_ord) {
                    Some(c) => {
                        let path = if c.heading_path.is_empty() {
                            "(root)".to_string()
                        } else {
                            c.heading_path.join(" > ")
                        };
                        println!("\n--- chunk {} ---", c.ordinal);
                        println!("chunk_id    = {}", c.chunk_id);
                        println!("char_offset = {}", c.char_offset);
                        println!("char_length = {}", c.char_length);
                        println!("heading     = {}", path);
                        println!("\n{}", c.content);
                    }
                    None => {
                        println!("\nno chunk with ordinal {}", target_ord);
                    }
                }
            } else if chunks {
                println!();
                println!(
                    "{:>4} {:>7} {:>6}  {:<60}  {}",
                    "ORD", "OFFSET", "LEN", "HEADING", "PREVIEW"
                );
                for c in &all_chunks {
                    let heading = if c.heading_path.is_empty() {
                        "(root)".to_string()
                    } else {
                        c.heading_path.join(" > ")
                    };
                    let heading_short = if heading.chars().count() > 60 {
                        let short: String = heading.chars().take(57).collect();
                        format!("{}…", short)
                    } else {
                        heading
                    };
                    let preview: String = c
                        .content
                        .chars()
                        .take(60)
                        .collect::<String>()
                        .replace('\n', " ");
                    println!(
                        "{:>4} {:>7} {:>6}  {:<60}  {}",
                        c.ordinal, c.char_offset, c.char_length, heading_short, preview
                    );
                }
            }
        }
        Command::Organize { limit, source, retag } => {
            let stores = stores.as_ref().expect("stores built above");
            let source_filter = if let Some(s) = source {
                Some(resolve_source_id(&stores.pg, &s).await?)
            } else {
                None
            };
            let opts = OrganizeOpts {
                limit,
                source_filter,
                retag,
            };
            let report = organize(stores, opts).await?;
            println!("organize report:");
            println!("  chunks_processed = {}", report.chunks_processed);
            println!("  chunks_skipped   = {}", report.chunks_skipped);
            println!("  topics_created   = {}", report.topics_created);
            println!("  topics_updated   = {}", report.topics_updated);
            println!("  assignments_made = {}", report.assignments_made);
            println!("  llm_failures     = {}", report.llm_failures);
        }
        Command::Topics { limit } => {
            let stores = stores.as_ref().expect("stores built above");
            let rows = list_topic_summaries(&stores.pg, limit).await?;
            if rows.is_empty() {
                println!("no topics yet — run `gw-kb organize` first");
                return Ok(());
            }
            println!(
                "{:>6} {:>6}  {:<30}  {}",
                "CHUNKS", "SRCS", "SLUG", "LABEL"
            );
            for t in &rows {
                let slug_short = if t.slug.chars().count() > 30 {
                    let s: String = t.slug.chars().take(28).collect();
                    format!("{}…", s)
                } else {
                    t.slug.clone()
                };
                println!(
                    "{:>6} {:>6}  {:<30}  {}",
                    t.chunk_count, t.source_count, slug_short, t.label
                );
            }
            println!("\n{} topic(s)", rows.len());
        }
        Command::Topic { slug, chunks } => {
            let stores = stores.as_ref().expect("stores built above");
            let topic = fetch_topic_by_slug(&stores.pg, &slug).await?;
            println!("topic_id     = {}", topic.topic_id);
            println!("label        = {}", topic.label);
            println!("slug         = {}", topic.slug);
            println!("chunk_count  = {}", topic.chunk_count);
            println!("first_seen   = {}", topic.first_seen);
            println!("last_seen    = {}", topic.last_seen);

            // Summary (Phase 3) — prominent placement right after metadata.
            if let Some((summary, summary_at)) =
                fetch_summary(&stores.pg, topic.topic_id).await?
            {
                println!("summary_at   = {}", summary_at);
                println!();
                println!("{}", summary);
            }
            println!();

            let members = if chunks > 0 {
                list_chunks_for_topic(&stores.pg, topic.topic_id, chunks).await?
            } else {
                Vec::new()
            };
            for (i, c) in members.iter().enumerate() {
                let path = if c.heading_path.is_empty() {
                    "(root)".to_string()
                } else {
                    c.heading_path.join(" > ")
                };
                println!(
                    "[{}] rel={:.3}  {}  ({})",
                    i + 1,
                    c.relevance,
                    c.source_title,
                    path
                );
                let preview: String = c.content.chars().take(220).collect::<String>().replace('\n', " ");
                println!("    {}", preview);
                if let Some(url) = &c.source_url {
                    println!("    {}", url);
                }
                println!();
            }
            if chunks > 0 && (members.len() as i64) == chunks {
                println!("(showing first {chunks}, use --chunks to fetch more)");
            }

            // Linked topics (if the graph has been built)
            let neigh = neighbors_of(&stores.pg, topic.topic_id, 25).await?;
            if !neigh.is_empty() {
                println!();
                print_neighbors_grouped(&neigh);
            }
        }
        Command::Synthesize {
            limit,
            min_chunks,
            regenerate,
            stale_only,
            topic,
        } => {
            let stores = stores.as_ref().expect("stores built above");

            let only_topic_id = if let Some(slug) = topic.as_ref() {
                let row = gw_kb::topics::fetch_topic_by_slug(&stores.pg, slug).await?;
                Some(row.topic_id)
            } else {
                None
            };

            let report = synthesize_topics(
                stores,
                SynthesizeOpts {
                    limit,
                    min_chunks,
                    regenerate: regenerate || only_topic_id.is_some(),
                    stale_only,
                    only_topic: only_topic_id,
                },
            )
            .await?;

            println!("synthesize report:");
            println!("  topics_considered = {}", report.topics_considered);
            println!("  summaries_written = {}", report.summaries_written);
            println!("  topics_skipped    = {}", report.topics_skipped);
            println!("  llm_failures      = {}", report.llm_failures);

            // In single-topic mode, also print the generated summary.
            if let Some(id) = only_topic_id {
                if let Some((summary, _)) = fetch_summary(&stores.pg, id).await? {
                    let row = sqlx::query_scalar::<_, String>(
                        "SELECT label FROM kb_topics WHERE topic_id = $1",
                    )
                    .bind(id)
                    .fetch_one(&stores.pg)
                    .await?;
                    println!("\n--- {} ---\n{}", row, summary);
                }
            }
        }
        Command::Merge { auto_threshold, ask_threshold, limit, dry_run } => {
            let stores = stores.as_ref().expect("stores built above");
            let report = merge_topics(
                stores,
                MergeOpts { auto_threshold, ask_threshold, limit, dry_run },
            )
            .await?;
            println!("merge report:");
            println!("  topics_seen           = {}", report.topics_seen);
            println!("  candidates_considered = {}", report.candidates_considered);
            println!("  auto_merges           = {}", report.auto_merges);
            println!("  llm_confirmed         = {}", report.llm_confirmed);
            println!("  llm_rejected          = {}", report.llm_rejected);
            println!("  llm_failures          = {}", report.llm_failures);
            println!("  merges_executed       = {}", report.merges_executed);
            if !report.examples.is_empty() {
                println!();
                println!("merges:");
                for ex in &report.examples {
                    let tag = if ex.auto { "auto" } else { "llm " };
                    println!(
                        "  [{}] cos={:.3}  {}  ←  {}",
                        tag, ex.cosine, ex.winner, ex.loser
                    );
                }
            }
            if report.merges_executed > 0 && !dry_run {
                println!();
                println!("note: topic graph is stale — run `gw-kb link` and `gw-kb classify` to rebuild edges");
            }
        }
        Command::Classify { limit, reclassify } => {
            let stores = stores.as_ref().expect("stores built above");
            let report = classify_edges(stores, ClassifyOpts { limit, reclassify }).await?;
            println!("classify report:");
            println!("  edges_seen        = {}", report.edges_seen);
            println!("  edges_classified  = {}", report.edges_classified);
            println!("  edges_skipped     = {}", report.edges_skipped);
            println!("  llm_failures      = {}", report.llm_failures);
            println!("  by kind:");
            let mut kinds: Vec<(&String, &usize)> = report.by_kind.iter().collect();
            kinds.sort_by(|a, b| b.1.cmp(a.1));
            for (k, n) in kinds {
                println!("    {:14} {}", k, n);
            }
        }
        Command::Link { min_shared, min_cosine, min_confidence } => {
            let stores = stores.as_ref().expect("stores built above");
            let report = link(
                &stores.pg,
                LinkOpts { min_shared_chunks: min_shared, min_cosine, min_confidence },
            )
            .await?;
            println!("link report:");
            println!("  topics_seen        = {}", report.topics_seen);
            println!("  cooccurrence_pairs = {}", report.cooccurrence_pairs);
            println!("  embedding_pairs    = {}", report.embedding_pairs);
            println!("  edges_written      = {}", report.edges_written);
        }
        Command::Explore { query, seeds, hops, decay, limit } => {
            let stores = stores.as_ref().expect("stores built above");

            // Embed the query and pick the nearest topics as seeds.
            let qvec = stores.embedder.embed_one(&query)?;
            let seed_pairs = nearest_topics_to_query(&stores.pg, &qvec, seeds).await?;
            if seed_pairs.is_empty() {
                println!("no topics in the knowledge base — run `gw-kb organize` first");
                return Ok(());
            }

            println!("seeds (nearest topics to query):");
            for (id, sim) in &seed_pairs {
                let label: String = sqlx::query_scalar(
                    "SELECT label FROM kb_topics WHERE topic_id = $1",
                )
                .bind(id)
                .fetch_one(&stores.pg)
                .await?;
                println!("  {:.4}  {}", sim, label);
            }
            println!();

            let activated = spread_from_seeds(
                &stores.pg,
                &seed_pairs,
                SpreadOpts { max_hops: hops, decay, limit },
            )
            .await?;

            if activated.is_empty() {
                println!("no neighbors reached — try increasing --hops or lowering link thresholds");
                return Ok(());
            }
            println!("activated topics (top {}):", activated.len());
            for (i, t) in activated.iter().enumerate() {
                println!(
                    "  [{:>2}] score={:.4}  ({} chunks)  {}  ({})",
                    i + 1,
                    t.score,
                    t.chunk_count,
                    t.label,
                    t.slug
                );
            }
        }
        Command::Search { query, k } => {
            let stores = stores.as_ref().expect("stores built above");
            let hits = hybrid_search(stores, &query, k).await?;
            if hits.is_empty() {
                println!("no results");
                return Ok(());
            }
            for (i, hit) in hits.iter().enumerate() {
                let path = if hit.heading_path.is_empty() {
                    "(root)".to_string()
                } else {
                    hit.heading_path.join(" > ")
                };
                println!(
                    "\n[{}] score={:.4}  {}  ({})",
                    i + 1,
                    hit.score,
                    hit.source_title,
                    path
                );
                if let Some(url) = &hit.source_url {
                    println!("    {}", url);
                }
                let preview: String = hit.content.chars().take(280).collect();
                let preview = preview.replace('\n', " ");
                println!("    {}", preview);
            }
        }
    }

    Ok(())
}

/// Render a topic's neighbors grouped by edge kind. For directional kinds
/// (`subtopic_of`, `builds_on`) the section heading reflects the direction
/// from the query topic's perspective.
fn print_neighbors_grouped(neigh: &[LinkedNeighbor]) {
    fn line(n: &LinkedNeighbor) {
        println!(
            "  conf={:.3}  ({} chunks)  {}  ({})",
            n.confidence, n.chunk_count, n.label, n.slug
        );
    }

    // Bucket by (kind, direction-relative-to-query-topic)
    let mut subtopics_of_query: Vec<&LinkedNeighbor> = vec![]; // query subtopic_of neighbor → neighbor is broader
    let mut parents_of_query: Vec<&LinkedNeighbor> = vec![];   // neighbor subtopic_of query → neighbor is narrower
    let mut query_builds_on: Vec<&LinkedNeighbor> = vec![];    // query builds_on neighbor → neighbor is prerequisite
    let mut builds_on_query: Vec<&LinkedNeighbor> = vec![];    // neighbor builds_on query → neighbor is dependent
    let mut contradicts: Vec<&LinkedNeighbor> = vec![];
    let mut related: Vec<&LinkedNeighbor> = vec![];

    for n in neigh {
        match (n.kind.as_str(), n.direction) {
            ("subtopic_of", EdgeDirection::OutgoingFrom) => subtopics_of_query.push(n),
            ("subtopic_of", EdgeDirection::IncomingTo) => parents_of_query.push(n),
            ("builds_on", EdgeDirection::OutgoingFrom) => query_builds_on.push(n),
            ("builds_on", EdgeDirection::IncomingTo) => builds_on_query.push(n),
            ("contradicts", _) => contradicts.push(n),
            _ => related.push(n),
        }
    }

    let sections: &[(&str, &[&LinkedNeighbor])] = &[
        ("Broader topics (this is a subtopic of):", &subtopics_of_query),
        ("Narrower topics (subtopics of this):", &parents_of_query),
        ("Prerequisites (this builds on):", &query_builds_on),
        ("Built on this:", &builds_on_query),
        ("Contradicts:", &contradicts),
        ("Related:", &related),
    ];

    for (title, items) in sections {
        if items.is_empty() {
            continue;
        }
        println!("{}", title);
        for n in *items {
            line(n);
        }
    }
}

async fn build_stores(cli: &Cli) -> Result<KbStores, KbError> {
    let pg = connect_pg(cli.database_url.as_deref()).await?;
    let lance = Arc::new(KbLanceStore::open(&cli.lance_path, cli.embedding_dim).await?);
    let tantivy = Arc::new(KbTantivyStore::open(std::path::Path::new(&cli.tantivy_path))?);
    let embedder = Arc::new(Embedder::new(cli.embedding_model.clone()));
    let llm = Arc::new(OllamaClient::new(
        cli.ollama_url.clone(),
        cli.ollama_url.clone(),
        // Chat model used for the tagger LLM in organize. embedding_model
        // here is unused by gw-llm because we route embeddings through
        // `embedder`, but gw-llm requires a string.
        "qwen3.5:9b".to_string(),
        "unused".to_string(),
    ));
    Ok(KbStores { pg, lance, tantivy, embedder, llm })
}

async fn connect_pg(database_url: Option<&str>) -> Result<PgPool, KbError> {
    let url = database_url
        .map(|s| s.to_string())
        .or_else(|| std::env::var("DATABASE_URL").ok())
        .ok_or_else(|| KbError::Other("DATABASE_URL not set".into()))?;
    let pool = PgPoolOptions::new()
        .max_connections(4)
        .connect(&url)
        .await?;
    Ok(pool)
}
