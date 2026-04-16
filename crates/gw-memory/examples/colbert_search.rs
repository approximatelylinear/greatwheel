//! Full end-to-end ColBERT pipeline in Rust.
//!
//! Composes:
//!   - `CandleColbertEncoder` (loads Reason-ModernColBERT from a HF cache dir)
//!   - `UsearchRetriever` (mmap'd HNSW over flattened token vectors)
//!   - `BlobReranker` (Lance blob store of precomputed passage tensors)
//!
//! into a single `ColbertStore`. Reads queries from stdin (one per line),
//! prints top-K results as JSON to stdout.
//!
//! Usage:
//!   cargo run -p gw-memory --example colbert_search --release -- \
//!       --model-dir <hf_cache_snapshot_dir> \
//!       --usearch-dir data/usearch-passages \
//!       --blob-store data/passage-blobs \
//!       [-k 10]
//!
//!   echo "Who won the Nobel Prize in Physics 2024?" | \
//!       cargo run -p gw-memory --example colbert_search --release -- ...
//!
//! Designed to mirror Python's `voyager_searcher.py` + `BlobReranker` so we
//! can compare quality and latency apples-to-apples.

use std::io::{self, BufRead, Write};
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use gw_memory::colbert::candle_encoder::CandleColbertEncoder;
use gw_memory::colbert::usearch_retriever::UsearchRetriever;
use gw_memory::colbert::ColbertStore;
use gw_memory::colbert_blobs::BlobReranker;
use serde::Serialize;

#[derive(Parser, Debug)]
struct Args {
    /// Path to the local HuggingFace cache snapshot directory containing
    /// `config.json`, `tokenizer.json`, `model.safetensors`, and `1_Dense/`.
    #[arg(long)]
    model_dir: String,

    /// Path to the directory containing `index.usearch` + `passage_to_docid.bin`.
    #[arg(long)]
    usearch_dir: String,

    /// Path to the Lance blob store directory (passage_blobs table).
    #[arg(long)]
    blob_store: String,

    /// Top-K results to return per query.
    #[arg(short, default_value_t = 10)]
    k: usize,

    /// First-stage candidate set size before reranking.
    #[arg(long, default_value_t = 200)]
    first_stage_k: usize,
}

#[derive(Serialize)]
struct ResultJson {
    query: String,
    n_candidates: usize,
    encode_ms: u128,
    first_stage_ms: u128,
    rerank_ms: u128,
    total_ms: u128,
    results: Vec<Hit>,
}

#[derive(Serialize)]
struct Hit {
    docid: String,
    score: f32,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    eprintln!("Loading encoder from {}...", args.model_dir);
    let t0 = Instant::now();
    let encoder = CandleColbertEncoder::from_dir(&args.model_dir)?;
    eprintln!("  encoder loaded in {:.1}s", t0.elapsed().as_secs_f32());

    eprintln!("Loading usearch index from {}...", args.usearch_dir);
    let t0 = Instant::now();
    let retriever = UsearchRetriever::open(&args.usearch_dir)?;
    eprintln!(
        "  usearch loaded in {:.1}s ({} token vectors)",
        t0.elapsed().as_secs_f32(),
        retriever.num_tokens()
    );

    eprintln!("Loading blob store from {}...", args.blob_store);
    let t0 = Instant::now();
    let reranker = BlobReranker::open(&args.blob_store).await?;
    eprintln!("  blob store loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let mut store = ColbertStore::new(Arc::new(encoder), Arc::new(retriever), reranker);
    store.first_stage_k = args.first_stage_k;

    eprintln!("Ready. Reading queries from stdin (one per line)...");
    let stdin = io::stdin();
    let mut stdout = io::stdout();
    for line in stdin.lock().lines() {
        let query = line?;
        if query.is_empty() {
            continue;
        }

        // Run the three stages individually so we can time each one.
        let t_total = Instant::now();
        let t_enc = Instant::now();
        let q_tokens = store.encoder().encode_query(&query).await?;
        let encode_ms = t_enc.elapsed().as_millis();

        let t_fs = Instant::now();
        let candidates = store
            .first_stage()
            .candidates(&query, q_tokens.view(), store.first_stage_k)
            .await?;
        let first_stage_ms = t_fs.elapsed().as_millis();

        let t_rr = Instant::now();
        let mut scored = store
            .reranker()
            .rerank(q_tokens.view(), &candidates)
            .await?;
        let rerank_ms = t_rr.elapsed().as_millis();

        scored.truncate(args.k);

        let total_ms = t_total.elapsed().as_millis();
        let out = ResultJson {
            query,
            n_candidates: candidates.len(),
            encode_ms,
            first_stage_ms,
            rerank_ms,
            total_ms,
            results: scored
                .into_iter()
                .map(|s| Hit {
                    docid: s.docid,
                    score: s.score,
                })
                .collect(),
        };
        writeln!(stdout, "{}", serde_json::to_string(&out)?)?;
        stdout.flush()?;
    }

    Ok(())
}
