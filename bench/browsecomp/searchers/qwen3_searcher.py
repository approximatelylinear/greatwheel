"""Qwen3-Embedding single-vector searcher (LanceDB cosine).

Mirrors the BrowseComp-Plus retrieval recipe: query gets the instruction
prefix and is encoded by the qwen3_embed_server (max_length=512, EOS pool,
L2-normalized). Index built by `build_qwen3_index.py` holds one normalized
vector per doc, so cosine == dot product.
"""

from __future__ import annotations

import os
import sys

import lancedb
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from searchers.base import ScoredDoc

TABLE = "qwen3_docs"


class Qwen3EncoderClient:
    """Tiny HTTP client for the Qwen3 embed service. Returns a single dense vector."""

    def __init__(self, base_url: str = "http://127.0.0.1:8003", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> dict:
        r = self._session.get(f"{self.base_url}/", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def encode_query(self, text: str) -> list[float]:
        r = self._session.post(
            f"{self.base_url}/encode_query",
            json={"text": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json()["vector"]


class Qwen3Searcher:
    name = "qwen3"

    def __init__(
        self,
        index_dir: str,
        encoder_url: str = "http://127.0.0.1:8003",
    ):
        self.encoder = Qwen3EncoderClient(encoder_url)
        info = self.encoder.health()
        print(f"Qwen3 encoder: {info.get('model')} ({info.get('dim')}-d) on {info.get('device')}", flush=True)

        print(f"Opening Qwen3 LanceDB index: {index_dir}", flush=True)
        db = lancedb.connect(index_dir)
        self.table = db.open_table(TABLE)
        n = self.table.count_rows()
        print(f"  {n} docs", flush=True)

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        q = self.encoder.encode_query(query)
        df = (
            self.table.search(q)
            .select(["docid", "_distance"])
            .limit(k)
            .to_pandas()
        )
        results: list[ScoredDoc] = []
        for _, row in df.iterrows():
            results.append(ScoredDoc(
                docid=str(row["docid"]),
                score=float(-row["_distance"]),
            ))
        return results
