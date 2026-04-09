"""Standalone blob-store-backed ColBERT reranker.

Loads only the encoder + Lance blob store. Given a query and a list of
candidate docids (from any first-stage retriever — BM25, dense, ANN,
whatever), encodes the query, fetches the candidates' precomputed
passage token tensors from the blob store, and runs full per-passage
MaxSim with per-doc max-pooling.

This is the production-shape rerank: no GPU encoding at query time, no
HNSW index, no Voyager. Just a fast key-value lookup + matmul.
"""

import os
import sys
import time
from typing import Any, Dict, List, Optional

import lancedb
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder


class BlobReranker:
    def __init__(
        self,
        blob_store_dir: str,
        model_name: str = "lightonai/Reason-ModernColBERT",
        device: Optional[str] = None,
    ):
        print(f"Loading blob store: {blob_store_dir}", flush=True)
        t0 = time.monotonic()
        db = lancedb.connect(blob_store_dir)
        self.table = db.open_table("passage_blobs")
        print(f"  loaded in {time.monotonic()-t0:.1f}s, {self.table.count_rows()} passages", flush=True)

        print(f"Loading encoder: {model_name}", flush=True)
        self.encoder = ColBERTEncoder(model_name, device=device)
        print(f"  device: {self.encoder.device}", flush=True)

    def rerank(
        self,
        query: str,
        candidates: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Rerank a list of {docid, ...} candidates by full per-passage MaxSim.

        Returns the same list reordered, with `score` set to the per-doc
        max-pooled MaxSim. Docids not present in the blob store sink to -inf.
        """
        if not candidates:
            return candidates

        # Encode the query (cheap, ~few ms)
        q = self.encoder._encode([query], max_length=128, is_query=True)[0]  # (Nq, 128) torch CPU

        docids = [c["docid"] for c in candidates]
        quoted = ",".join(f"'{d}'" for d in docids)
        arrow_tbl = (
            self.table.search()
            .where(f"docid IN ({quoted})")
            .select(["docid", "num_tokens", "vectors"])
            .limit(len(docids) * 64)
            .to_arrow()
        )

        col_docid = arrow_tbl.column("docid").to_pylist()
        col_ntok = arrow_tbl.column("num_tokens").to_pylist()
        col_vecs = arrow_tbl.column("vectors").to_pylist()

        doc_max: Dict[str, float] = {}
        for docid, n, vec_bytes in zip(col_docid, col_ntok, col_vecs):
            arr = np.frombuffer(vec_bytes, dtype=np.float16).reshape(n, 128)
            pt = torch.from_numpy(arr.astype(np.float32))
            sims = q @ pt.T
            score = float(sims.max(dim=1).values.sum())
            prev = doc_max.get(docid)
            if prev is None or score > prev:
                doc_max[docid] = score

        scored = [
            {**c, "score": doc_max.get(c["docid"], float("-inf"))}
            for c in candidates
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored
