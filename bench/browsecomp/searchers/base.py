"""Common interface for all retrieval backends.

Every searcher (BM25, dense ANN, late interaction, brute force, etc.)
implements the `Searcher` protocol so the benchmark harness can swap
backends with one CLI flag. Backends that need encoded query token
tensors get them from the shared encoder service over HTTP — they should
NOT load their own copy of the model.

The protocol is intentionally minimal:

    search(query, k) -> list[ScoredDoc]

Anything more elaborate (per-passage scores, snippet extraction, payload
filters) is a backend-specific extension and lives on the concrete class.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import requests


@dataclass
class ScoredDoc:
    """One result row from a Searcher."""
    docid: str
    score: float
    # Optional payload for downstream consumers (text snippet, passage_id, etc).
    # Backends are free to populate or ignore this.
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"docid": self.docid, "score": self.score, **self.payload}


class Searcher(Protocol):
    """All retrieval backends implement this."""

    name: str

    def search(self, query: str, k: int) -> list[ScoredDoc]:
        """Return the top-k documents for the query, ranked by score (desc)."""
        ...


# ---------------------------------------------------------------------------
# Shared encoder client — backends use this to get query / doc token vectors
# from the colbert_server.py HTTP service.
# ---------------------------------------------------------------------------


class EncoderClient:
    """Thin HTTP client for the shared ColBERT encoder service.

    All Searcher and index-builder code should call this rather than
    instantiating ColBERTEncoder directly. That way the model loads exactly
    once per benchmark session.
    """

    def __init__(self, base_url: str = "http://127.0.0.1:8002", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()

    def health(self) -> dict[str, Any]:
        r = self._session.get(f"{self.base_url}/", timeout=self.timeout)
        r.raise_for_status()
        return r.json()

    def encode_query(self, text: str) -> np.ndarray:
        """Encode a single query → (n_tokens, 128) float32 array."""
        r = self._session.post(
            f"{self.base_url}/encode_query",
            json={"text": text},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return np.asarray(r.json()["tokens"], dtype=np.float32)

    def encode_query_batch(self, texts: list[str]) -> list[np.ndarray]:
        """Encode many queries in one round-trip."""
        r = self._session.post(
            f"{self.base_url}/encode_query_batch",
            json={"texts": texts},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return [
            np.asarray(item["tokens"], dtype=np.float32)
            for item in r.json()["results"]
        ]

    def encode_doc_batch(self, texts: list[str], max_length: int = 512) -> list[np.ndarray]:
        """Encode many documents in one round-trip. Used by index builders."""
        r = self._session.post(
            f"{self.base_url}/encode_doc_batch",
            json={"texts": texts, "max_length": max_length},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return [
            np.asarray(item["tokens"], dtype=np.float32)
            for item in r.json()["results"]
        ]
