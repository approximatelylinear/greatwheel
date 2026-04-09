"""Brute-force MaxSim searcher — exact recall ceiling, no index.

For each query, scans every passage in the blob store, computes the full
MaxSim score, takes per-doc max across passages, returns top-k. There's
no approximation anywhere — this is what every HNSW-based backend is
trying to approximate, and the ceiling all of them are bounded by.

## Performance

For BrowseComp's ~1M passages × ~500 tokens × 128 dims:

- Sequential CPU torch matmul:        ~3 sec / query
- GPU torch matmul (cached blob):     ~150 ms / query
- I/O bound from cold blob store:     ~30 sec / query (122 GB)

So this is **a research tool**, not a production path. We use it to:

1. Establish the exact recall ceiling on BrowseComp (the number every
   HNSW backend should be measured against).
2. Sanity-check that approximate backends aren't dropping recall by more
   than expected (~1% for HNSW with f16 quantization).

## Caching

The first query is I/O bound (full blob scan, ~30 s). Subsequent queries
hit the OS page cache and run at compute speed (~3 s CPU, ~150 ms GPU).
For benchmarks we want consistent numbers, so the first call is a
free warmup — discard its latency or run it twice.

## Optional in-RAM caching

Pass `cache_in_ram=True` to load all passages into a single contiguous
float32 tensor on construction. Cost: ~244 GB RAM (1M × 500 × 128 × 4).
Way over budget on most machines but useful if you have it. Default is
to stream from Lance per query.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any

import lancedb
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from searchers.base import EncoderClient, ScoredDoc

TOKEN_DIM = 128


class BruteForceSearcher:
    name = "brute_force"

    def __init__(
        self,
        blob_store_dir: str,
        encoder: EncoderClient,
        device: str | None = None,
        scan_batch_rows: int = 8192,
        max_passages: int | None = None,
    ):
        """Args:
            blob_store_dir: path to the Lance passage blob store
            encoder: shared HTTP encoder client
            device: 'cuda' or 'cpu' (default: cuda if available)
            scan_batch_rows: how many passage rows to fetch per Lance batch.
                Smaller = less peak RAM, more Lance overhead. 8192 is a
                good default at ~1.5 GB peak per batch.
            max_passages: cap total passages scanned. None = full corpus.
                Useful for smoke testing or debugging.
        """
        self.encoder = encoder
        self.scan_batch_rows = scan_batch_rows
        self.max_passages = max_passages
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Opening blob store: {blob_store_dir}", flush=True)
        db = lancedb.connect(blob_store_dir)
        self.table = db.open_table("passage_blobs")
        n = self.table.count_rows()
        print(f"  {n} passages", flush=True)

    def _iter_batches(self):
        """Yield (docids, ntoks, vec_bytes_list) batches from Lance.

        Uses offset+limit chunking — works with the lancedb high-level
        API without requiring pylance. Each batch is `scan_batch_rows`
        passages, well under any RAM limit.
        """
        offset = 0
        total = self.max_passages if self.max_passages else self.table.count_rows()
        while offset < total:
            limit = min(self.scan_batch_rows, total - offset)
            arrow_tbl = (
                self.table.search()
                .select(["docid", "num_tokens", "vectors"])
                .limit(limit)
                .offset(offset)
                .to_arrow()
            )
            if arrow_tbl.num_rows == 0:
                break
            yield (
                arrow_tbl.column("docid").to_pylist(),
                arrow_tbl.column("num_tokens").to_pylist(),
                arrow_tbl.column("vectors").to_pylist(),
            )
            offset += arrow_tbl.num_rows

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        # Encode query
        q_np = self.encoder.encode_query(query)  # (Nq, 128) float32
        q_t = torch.from_numpy(q_np).to(self.device)

        doc_max: dict[str, float] = {}
        n_seen = 0
        t0 = time.monotonic()

        for docids, ntoks, vec_bytes_list in self._iter_batches():
            for docid, n, vec_bytes in zip(docids, ntoks, vec_bytes_list):
                arr = np.frombuffer(vec_bytes, dtype=np.float16).reshape(n, TOKEN_DIM)
                pt = torch.from_numpy(arr.astype(np.float32)).to(self.device)
                sims = q_t @ pt.T  # (Nq, n_passage_tokens)
                score = float(sims.max(dim=1).values.sum().item())
                prev = doc_max.get(docid)
                if prev is None or score > prev:
                    doc_max[docid] = score
            n_seen += len(docids)
            if n_seen % (self.scan_batch_rows * 8) == 0:
                rate = n_seen / (time.monotonic() - t0)
                print(f"  scanned {n_seen} passages ({rate:.0f}/s)", flush=True)

        ranked = sorted(doc_max.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [ScoredDoc(docid=d, score=s) for d, s in ranked]
