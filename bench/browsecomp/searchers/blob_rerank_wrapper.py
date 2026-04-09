"""Decorator that adds passage-blob-store reranking on top of any Searcher.

Lets us turn any first-stage retriever (BM25, dense ANN, even brute-force)
into a "first stage → late-interaction rerank" pipeline using the existing
Lance blob store of precomputed token tensors. The wrapped Searcher
provides candidate docids; this wrapper does the MaxSim reordering.

Usage:

    base = TantivySearcher(index_path)
    wrapped = BlobRerankWrapper(
        base,
        encoder=EncoderClient(),
        blob_store_dir="data/passage-blobs",
        first_stage_k=200,
    )
    results = wrapped.search("Who won the Nobel Prize?", k=10)

The base searcher is asked for `first_stage_k` candidates (default 200);
those are reranked via real per-passage MaxSim against the query and the
top-k are returned. R@k for k > first_stage_k is bounded by the base
searcher's R@first_stage_k.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from blob_reranker import BlobReranker
from searchers.base import EncoderClient, ScoredDoc, Searcher


class BlobRerankWrapper:
    """Searcher decorator that adds blob-store MaxSim rerank."""

    def __init__(
        self,
        base: Searcher,
        encoder: EncoderClient,
        blob_store_dir: str,
        first_stage_k: int = 200,
    ):
        self.base = base
        self.encoder = encoder
        self.first_stage_k = first_stage_k
        # BlobReranker loads its own encoder by default; we don't need that
        # since we're getting tokens via HTTP. Build it via __new__ + manual
        # field assignment to skip the model load.
        self._reranker = BlobReranker.__new__(BlobReranker)
        import lancedb
        self._reranker.table = lancedb.connect(blob_store_dir).open_table("passage_blobs")
        self._reranker.encoder = None  # we never call self._reranker.encoder

    @property
    def name(self) -> str:
        return f"{self.base.name}+rerank"

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        # 1. First stage — get candidates
        candidates = self.base.search(query, self.first_stage_k)
        if not candidates:
            return []

        # 2. Encode query (HTTP, no GPU model in this process)
        q = self.encoder.encode_query(query)  # (Nq, 128) numpy float32

        # 3. Real MaxSim rerank against the blob store. The BlobReranker's
        # rerank() expects to encode the query itself, so we call its
        # internal rerank-from-tensor path directly.
        candidate_dicts = [{"docid": c.docid} for c in candidates]
        reranked = _rerank_from_blobs_with_tensor(
            self._reranker, q, candidate_dicts
        )

        # 4. Convert back to ScoredDoc, preserving any payload from the base
        payload_by_docid = {c.docid: c.payload for c in candidates}
        out: list[ScoredDoc] = []
        for r in reranked[:k]:
            out.append(ScoredDoc(
                docid=r["docid"],
                score=float(r["score"]),
                payload=payload_by_docid.get(r["docid"], {}),
            ))
        return out


def _rerank_from_blobs_with_tensor(
    reranker: BlobReranker,
    q: np.ndarray,
    candidates: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Like BlobReranker.rerank_from_blobs but takes a pre-encoded query.

    BlobReranker.rerank_from_blobs hardcodes a `self.encoder._encode(...)`
    call to get the query tokens. We already have them from the HTTP
    encoder, so we replicate the rest of the rerank logic here.
    """
    import torch
    from collections import defaultdict

    if not candidates:
        return []

    docids = [c["docid"] for c in candidates]
    quoted = ",".join(f"'{d}'" for d in docids)
    arrow_tbl = (
        reranker.table.search()
        .where(f"docid IN ({quoted})")
        .select(["docid", "num_tokens", "vectors"])
        .limit(len(docids) * 64)
        .to_arrow()
    )

    col_docid = arrow_tbl.column("docid").to_pylist()
    col_ntok = arrow_tbl.column("num_tokens").to_pylist()
    col_vecs = arrow_tbl.column("vectors").to_pylist()

    q_t = torch.from_numpy(q)  # (Nq, 128)
    doc_max: dict[str, float] = {}
    for docid, n, vec_bytes in zip(col_docid, col_ntok, col_vecs):
        arr = np.frombuffer(vec_bytes, dtype=np.float16).reshape(n, 128)
        pt = torch.from_numpy(arr.astype(np.float32))
        sims = q_t @ pt.T  # (Nq, n)
        score = float(sims.max(dim=1).values.sum())
        prev = doc_max.get(docid)
        if prev is None or score > prev:
            doc_max[docid] = score

    scored = [
        {"docid": c["docid"], "score": doc_max.get(c["docid"], float("-inf"))}
        for c in candidates
    ]
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored
