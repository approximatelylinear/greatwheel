//! Usearch first-stage retriever for ColBERT-style late interaction.
//!
//! Uses a `usearch` HNSW index over flattened token vectors. For each
//! query token, we ANN-query the index for the top-N nearest passage
//! tokens; then aggregate token hits by docid (per-query-token max,
//! summed across query tokens) → top-K candidate docids.
//!
//! ## Storage layout
//!
//! Built once by a Python builder.
//! The on-disk artifacts are:
//!
//! - `index.usearch`            — usearch HNSW file (mmap'd via `view()`)
//! - `passage_to_docid.bin`     — packed binary map: `[u64 count]` then
//!   `[u64 passage_id, u32 docid_len, docid_bytes]*`
//!
//! Each token vector in the index is keyed by:
//!
//! ```text
//!   key u64 = (passage_id << 16) | token_idx
//! ```
//!
//! `passage_id` indexes into the docid map; `token_idx` is the token's
//! position within the passage (0..num_tokens). The high bits let us
//! recover the docid via a single `>> 16` and a HashMap lookup.

use std::collections::HashMap;
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

use async_trait::async_trait;
use ndarray::ArrayView2;
use usearch::{ffi::IndexOptions, ffi::MetricKind, ffi::ScalarKind, Index};

use crate::colbert::FirstStageRetriever;
use crate::error::MemoryError;

const TOKEN_DIM: usize = 128;
const PASSAGE_BITS: u32 = 16;
const TOKEN_MASK: u64 = (1u64 << PASSAGE_BITS) - 1;

pub struct UsearchRetriever {
    index: Index,
    /// passage_id → docid. Loaded fully into RAM (small: ~1M entries × ~30 bytes).
    passage_to_docid: HashMap<u64, String>,
    /// How many neighbors to ask the HNSW for per query token. Higher =
    /// more recall, slower. Empirically 1000–2000 works well at our scale.
    pub tokens_per_query: usize,
}

impl UsearchRetriever {
    /// Open a usearch index in read-only (`view`) mode and load the
    /// passage→docid map from `passage_to_docid.bin` next to it.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, MemoryError> {
        let dir = dir.as_ref();

        // Open index. We use F32 storage for compatibility; the index file
        // itself records the actual on-disk scalar kind.
        let opts = IndexOptions {
            dimensions: TOKEN_DIM,
            metric: MetricKind::Cos,
            quantization: ScalarKind::F32,
            connectivity: 0,     // 0 = use file's value
            expansion_add: 0,    // 0 = use file's value
            expansion_search: 0, // 0 = use file's value
            multi: false,
        };
        let index =
            Index::new(&opts).map_err(|e| MemoryError::Embedding(format!("usearch new: {e}")))?;

        let index_path = dir.join("index.usearch");
        let path_str = index_path
            .to_str()
            .ok_or_else(|| MemoryError::Embedding("non-utf8 index path".into()))?;
        index
            .view(path_str)
            .map_err(|e| MemoryError::Embedding(format!("usearch view {path_str}: {e}")))?;

        let map_path = dir.join("passage_to_docid.bin");
        let passage_to_docid = load_passage_map(&map_path)?;

        Ok(Self {
            index,
            passage_to_docid,
            tokens_per_query: 2000,
        })
    }

    /// Number of token vectors in the underlying index.
    pub fn num_tokens(&self) -> usize {
        self.index.size()
    }
}

#[async_trait]
impl FirstStageRetriever for UsearchRetriever {
    async fn candidates(
        &self,
        _text: &str,
        query_tokens: ArrayView2<'_, f32>,
        k: usize,
    ) -> Result<Vec<String>, MemoryError> {
        assert_eq!(query_tokens.ncols(), TOKEN_DIM);
        let nq = query_tokens.nrows();

        // For each query token, take the per-doc max similarity across that
        // token's top-N neighbors. Sum across query tokens → final doc score.
        let mut doc_scores: HashMap<String, f32> = HashMap::new();

        for qi in 0..nq {
            let row = query_tokens.row(qi);
            let row_slice = row
                .as_slice()
                .ok_or_else(|| MemoryError::Embedding("non-contiguous query row".into()))?;

            let matches = self
                .index
                .search(row_slice, self.tokens_per_query)
                .map_err(|e| MemoryError::Embedding(format!("usearch search: {e}")))?;

            // Per-query-token, find best similarity per docid (the "max" in MaxSim).
            let mut best_per_doc: HashMap<&str, f32> = HashMap::new();
            for (key, dist) in matches.keys.iter().zip(matches.distances.iter()) {
                let passage_id = key >> PASSAGE_BITS;
                let _token_idx = key & TOKEN_MASK;
                let docid = match self.passage_to_docid.get(&passage_id) {
                    Some(d) => d.as_str(),
                    None => continue,
                };
                // Cosine distance → similarity (usearch returns 1-cos for Cos metric)
                let sim = 1.0 - dist;
                best_per_doc
                    .entry(docid)
                    .and_modify(|s| {
                        if sim > *s {
                            *s = sim;
                        }
                    })
                    .or_insert(sim);
            }

            for (docid, sim) in best_per_doc {
                *doc_scores.entry(docid.to_string()).or_insert(0.0) += sim;
            }
        }

        let mut ranked: Vec<(String, f32)> = doc_scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        ranked.truncate(k);
        Ok(ranked.into_iter().map(|(d, _)| d).collect())
    }
}

/// Load `passage_to_docid.bin`. Format:
///
/// ```text
///   u64 little-endian: count
///   repeated count times:
///     u64 little-endian: passage_id
///     u32 little-endian: docid_len
///     [u8; docid_len]:    docid bytes (UTF-8)
/// ```
pub fn load_passage_map(path: &Path) -> Result<HashMap<u64, String>, MemoryError> {
    let f = File::open(path).map_err(|e| MemoryError::Embedding(format!("open {path:?}: {e}")))?;
    let mut r = BufReader::new(f);

    let mut buf8 = [0u8; 8];
    let mut buf4 = [0u8; 4];

    r.read_exact(&mut buf8)
        .map_err(|e| MemoryError::Embedding(format!("read count: {e}")))?;
    let count = u64::from_le_bytes(buf8) as usize;

    let mut map = HashMap::with_capacity(count);
    for _ in 0..count {
        r.read_exact(&mut buf8)
            .map_err(|e| MemoryError::Embedding(format!("read passage_id: {e}")))?;
        let passage_id = u64::from_le_bytes(buf8);

        r.read_exact(&mut buf4)
            .map_err(|e| MemoryError::Embedding(format!("read docid_len: {e}")))?;
        let len = u32::from_le_bytes(buf4) as usize;

        let mut docid_buf = vec![0u8; len];
        r.read_exact(&mut docid_buf)
            .map_err(|e| MemoryError::Embedding(format!("read docid: {e}")))?;
        let docid = String::from_utf8(docid_buf)
            .map_err(|e| MemoryError::Embedding(format!("docid utf8: {e}")))?;
        map.insert(passage_id, docid);
    }
    Ok(map)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_encoding_roundtrip() {
        let passage_id: u64 = 12345;
        let token_idx: u64 = 42;
        let key = (passage_id << PASSAGE_BITS) | token_idx;
        assert_eq!(key >> PASSAGE_BITS, passage_id);
        assert_eq!(key & TOKEN_MASK, token_idx);
    }

    #[test]
    fn key_encoding_max_token_idx() {
        // Max token_idx is 2^16 - 1 = 65535. Should be plenty for any single passage.
        let passage_id: u64 = 1;
        let token_idx: u64 = TOKEN_MASK;
        let key = (passage_id << PASSAGE_BITS) | token_idx;
        assert_eq!(key >> PASSAGE_BITS, passage_id);
        assert_eq!(key & TOKEN_MASK, token_idx);
    }
}
