//! Encode a query with the candle ColBERT encoder, print the resulting
//! token tensor as JSON. Used by the parity test against Python.
//!
//! Usage:
//!   cargo run -p gw-memory --example encode_query --release -- \
//!       <model_dir> <query>

use std::env;

use gw_memory::colbert::candle_encoder::CandleColbertEncoder;
use gw_memory::colbert::ColbertEncoder;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: encode_query <model_dir> <query>");
        std::process::exit(1);
    }
    let model_dir = &args[1];
    let query = &args[2];

    let encoder = CandleColbertEncoder::from_dir(model_dir)?;
    let arr = encoder.encode_query(query).await?;

    // shape: (n_tokens, 128)
    let (n, d) = arr.dim();
    let rows: Vec<Vec<f32>> = arr.outer_iter().map(|r| r.to_vec()).collect();
    let out = serde_json::json!({
        "n_tokens": n,
        "dim": d,
        "tokens": rows,
    });
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
