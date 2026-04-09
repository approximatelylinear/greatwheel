//! Standalone blob-rerank example for parity testing against the Python
//! `BlobReranker`. Reads a JSON request from stdin, runs the rerank, prints
//! a JSON result to stdout.
//!
//! Request format:
//! ```json
//! {
//!   "blob_store": "data/passage-blobs",
//!   "query_tokens": [[f32; 128], ...],
//!   "candidates": ["docid1", "docid2", ...]
//! }
//! ```
//!
//! Result format:
//! ```json
//! {
//!   "scored": [{"docid": "...", "score": 12.34}, ...],
//!   "elapsed_ms": 187
//! }
//! ```
//!
//! Run with:
//! ```bash
//! cargo run -p gw-memory --example blob_rerank --release < request.json
//! ```

use std::io::{self, Read};
use std::time::Instant;

use gw_memory::colbert_blobs::{BlobReranker, ScoredDoc, TOKEN_DIM};
use ndarray::Array2;
use serde::{Deserialize, Serialize};

#[derive(Deserialize)]
struct Request {
    blob_store: String,
    query_tokens: Vec<Vec<f32>>,
    candidates: Vec<String>,
}

#[derive(Serialize)]
struct ScoredOut {
    docid: String,
    score: f32,
}

#[derive(Serialize)]
struct Response {
    scored: Vec<ScoredOut>,
    elapsed_ms: u128,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input)?;
    let req: Request = serde_json::from_str(&input)?;

    let nq = req.query_tokens.len();
    assert!(nq > 0, "empty query");
    assert_eq!(req.query_tokens[0].len(), TOKEN_DIM, "query token dim must be {TOKEN_DIM}");
    let mut q = Array2::<f32>::zeros((nq, TOKEN_DIM));
    for (i, row) in req.query_tokens.iter().enumerate() {
        for (j, v) in row.iter().enumerate() {
            q[[i, j]] = *v;
        }
    }

    let reranker = BlobReranker::open(&req.blob_store).await?;

    let t0 = Instant::now();
    let scored: Vec<ScoredDoc> = reranker.rerank(q.view(), &req.candidates).await?;
    let elapsed_ms = t0.elapsed().as_millis();

    let out = Response {
        scored: scored
            .into_iter()
            .map(|s| ScoredOut { docid: s.docid, score: s.score })
            .collect(),
        elapsed_ms,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
