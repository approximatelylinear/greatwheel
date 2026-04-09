"""Qdrant native multi-vector searcher.

Uses Qdrant 1.10+'s `MultiVectorConfig` with `MAX_SIM` comparator. The
collection is built by `build_qdrant_index.py` with one point per corpus
doc and all passage tokens flattened into a single multi-vector field.
Qdrant computes ColBERT MaxSim natively in the C++ core.
"""

from __future__ import annotations

import os
import sys

from qdrant_client import QdrantClient

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from searchers.base import EncoderClient, ScoredDoc


class QdrantSearcher:
    name = "qdrant"

    def __init__(
        self,
        encoder: EncoderClient,
        qdrant_url: str = "http://localhost:6333",
        collection: str = "colbert_mv",
    ):
        self.encoder = encoder
        self.client = QdrantClient(url=qdrant_url, timeout=60)
        self.collection = collection
        n = self.client.count(collection).count
        print(f"Qdrant collection {collection!r}: {n} points", flush=True)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        # Encode → list of 128-dim float lists
        q_arr = self.encoder.encode_query(query)
        q_list = q_arr.tolist()

        # Multi-vector query — Qdrant accepts a list of vectors directly.
        results = self.client.query_points(
            collection_name=self.collection,
            query=q_list,
            limit=k,
            with_payload=True,
        )
        return [
            ScoredDoc(
                docid=str(p.payload.get("docid", p.id)),
                score=float(p.score),
            )
            for p in results.points
        ]
