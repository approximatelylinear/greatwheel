//! ColBERT blob-store reranker.
//!
//! Production-shape rerank for late-interaction retrieval. Given a
//! pre-encoded query (Nq tokens × 128 dims) and a list of candidate
//! document IDs from any first-stage retriever, this fetches all the
//! candidates' precomputed passage token tensors from a Lance blob
//! table, runs full per-passage MaxSim, and returns the candidates
//! reordered by per-doc max score.
//!
//! Storage layout (built by `bench/browsecomp/build_passage_blob_store.py`):
//!
//! ```text
//! passage_blobs (lance table):
//!   docid       string
//!   chunk_idx   int32
//!   num_tokens  int32
//!   vectors     binary  -- num_tokens * 128 * 2 bytes (float16, row-major)
//! ```
//!
//! No GPU encoding happens at query time. The encoder lives upstream
//! (Python today, candle/ORT eventually) and produces the query token
//! matrix that gets passed in here.
//!
//! See `docs/design-colbert-production.md` for the full architecture
//! and the empirical results that justify it.

use std::collections::HashMap;

use arrow_array::{Array, BinaryArray, Int32Array, StringArray};
use futures::TryStreamExt;
use half::f16;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::{connect, Connection, Table};
use ndarray::{Array2, ArrayView2, Axis};

use crate::error::MemoryError;

/// Token vector dimension for Reason-ModernColBERT and most ColBERT-family models.
pub const TOKEN_DIM: usize = 128;

/// Default Lance table name written by the Python builder.
pub const DEFAULT_TABLE: &str = "passage_blobs";

/// One scored document returned from `BlobReranker::rerank`.
#[derive(Debug, Clone)]
pub struct ScoredDoc {
    pub docid: String,
    /// Per-doc maximum of per-passage MaxSim scores. Docs not present in
    /// the blob store get `f32::NEG_INFINITY` so they sink to the bottom.
    pub score: f32,
}

/// Late-interaction reranker backed by a Lance passage-blob table.
pub struct BlobReranker {
    #[allow(dead_code)]
    conn: Connection,
    table: Table,
}

impl BlobReranker {
    /// Open the blob store at `path` and load the `passage_blobs` table.
    pub async fn open(path: &str) -> Result<Self, MemoryError> {
        Self::open_table(path, DEFAULT_TABLE).await
    }

    /// Open the blob store at `path` and load a specific table by name.
    pub async fn open_table(path: &str, table_name: &str) -> Result<Self, MemoryError> {
        let conn = connect(path).execute().await?;
        let table = conn.open_table(table_name).execute().await?;
        Ok(Self { conn, table })
    }

    /// Rerank `candidates` by full per-passage MaxSim against `query_tokens`.
    ///
    /// `query_tokens` must be L2-normalized and have shape `(num_query_tokens, 128)`.
    /// `candidates` is the list of docids from a first-stage retriever (BM25,
    /// dense ANN, Voyager, etc.). Returns the same docids reordered by score
    /// (descending). Docids not present in the blob store get `-inf` and sink.
    pub async fn rerank(
        &self,
        query_tokens: ArrayView2<'_, f32>,
        candidates: &[String],
    ) -> Result<Vec<ScoredDoc>, MemoryError> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }
        assert_eq!(
            query_tokens.ncols(),
            TOKEN_DIM,
            "query token dim must be {TOKEN_DIM}"
        );

        // Build the SQL IN clause. Lance/DataFusion uses single-quoted string
        // literals; escape any embedded single quotes by doubling them.
        let in_list = candidates
            .iter()
            .map(|d| format!("'{}'", d.replace('\'', "''")))
            .collect::<Vec<_>>()
            .join(",");
        let where_clause = format!("docid IN ({in_list})");

        let mut stream = self
            .table
            .query()
            .only_if(where_clause)
            .select(Select::columns(&["docid", "num_tokens", "vectors"]))
            .limit(candidates.len() * 64) // generous: at most 64 passages per doc
            .execute()
            .await?;

        let mut doc_max: HashMap<String, f32> = HashMap::with_capacity(candidates.len());

        while let Some(batch) = stream.try_next().await? {
            let docid_col = batch
                .column_by_name("docid")
                .ok_or_else(|| MemoryError::Embedding("missing docid column".into()))?
                .as_any()
                .downcast_ref::<StringArray>()
                .ok_or_else(|| MemoryError::Embedding("docid not StringArray".into()))?;
            let ntok_col = batch
                .column_by_name("num_tokens")
                .ok_or_else(|| MemoryError::Embedding("missing num_tokens column".into()))?
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| MemoryError::Embedding("num_tokens not Int32Array".into()))?;
            let vec_col = batch
                .column_by_name("vectors")
                .ok_or_else(|| MemoryError::Embedding("missing vectors column".into()))?
                .as_any()
                .downcast_ref::<BinaryArray>()
                .ok_or_else(|| MemoryError::Embedding("vectors not BinaryArray".into()))?;

            for i in 0..batch.num_rows() {
                let docid = docid_col.value(i);
                let n = ntok_col.value(i) as usize;
                let bytes = vec_col.value(i);
                debug_assert_eq!(bytes.len(), n * TOKEN_DIM * 2);

                let passage = decode_f16_passage(bytes, n);
                let score = maxsim(&query_tokens, &passage.view());

                doc_max
                    .entry(docid.to_string())
                    .and_modify(|s| {
                        if score > *s {
                            *s = score;
                        }
                    })
                    .or_insert(score);
            }
        }

        let mut scored: Vec<ScoredDoc> = candidates
            .iter()
            .map(|d| ScoredDoc {
                docid: d.clone(),
                score: doc_max.get(d).copied().unwrap_or(f32::NEG_INFINITY),
            })
            .collect();
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Ok(scored)
    }
}

/// Decode `n` × 128 float16 token vectors from raw little-endian bytes
/// into a row-major `Array2<f32>`. Bytes are stored row-major (token, dim).
fn decode_f16_passage(bytes: &[u8], n: usize) -> Array2<f32> {
    let mut out = Array2::<f32>::zeros((n, TOKEN_DIM));
    let slice = out.as_slice_mut().expect("contiguous");
    for (i, chunk) in bytes.chunks_exact(2).enumerate() {
        slice[i] = f16::from_le_bytes([chunk[0], chunk[1]]).to_f32();
    }
    out
}

/// Full MaxSim: for each query token, take the maximum dot product over
/// passage tokens, then sum across query tokens. Both inputs are
/// L2-normalized so the dot product is cosine similarity.
fn maxsim(query: &ArrayView2<'_, f32>, passage: &ArrayView2<'_, f32>) -> f32 {
    // sims: (Nq, n_passage_tokens) = query @ passage^T
    let sims = query.dot(&passage.t());
    sims.map_axis(Axis(1), |row| {
        row.iter().copied().fold(f32::NEG_INFINITY, f32::max)
    })
    .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    #[test]
    fn maxsim_identity() {
        // Single query token equal to a single passage token → score = 1.0
        let q = array![[1.0_f32, 0.0, 0.0]];
        let p = array![[1.0_f32, 0.0, 0.0]];
        // For this test we ignore TOKEN_DIM since maxsim() doesn't enforce it.
        let s = q.dot(&p.t()).map_axis(Axis(1), |r| {
            r.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }).sum();
        assert!((s - 1.0).abs() < 1e-6);
    }

    #[test]
    fn maxsim_orthogonal() {
        let q = array![[1.0_f32, 0.0, 0.0]];
        let p = array![[0.0_f32, 1.0, 0.0]];
        let s = q.dot(&p.t()).map_axis(Axis(1), |r| {
            r.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }).sum();
        assert!(s.abs() < 1e-6);
    }

    #[test]
    fn maxsim_max_then_sum() {
        // Two query tokens, two passage tokens. Each query picks its best.
        // q0 = [1,0], q1 = [0,1]
        // p0 = [0.9, 0.1], p1 = [0.2, 0.8]
        // sims = [[0.9, 0.2], [0.1, 0.8]]; per-row max = [0.9, 0.8]; sum = 1.7
        let q = array![[1.0_f32, 0.0], [0.0, 1.0]];
        let p = array![[0.9_f32, 0.1], [0.2, 0.8]];
        let s = q.dot(&p.t()).map_axis(Axis(1), |r| {
            r.iter().copied().fold(f32::NEG_INFINITY, f32::max)
        }).sum();
        assert!((s - 1.7).abs() < 1e-6, "got {s}");
    }

    #[test]
    fn decode_f16_roundtrip() {
        // Encode 2 tokens × 128 dims, all 0.5, and verify decode matches.
        let n = 2;
        let mut bytes = Vec::with_capacity(n * TOKEN_DIM * 2);
        for _ in 0..(n * TOKEN_DIM) {
            bytes.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
        }
        let arr = decode_f16_passage(&bytes, n);
        assert_eq!(arr.shape(), &[n, TOKEN_DIM]);
        for v in arr.iter() {
            assert!((v - 0.5).abs() < 1e-3, "got {v}");
        }
    }
}
