"""Embedding via sentence-transformers (canonical loading path).

Used by gw-kb for both chunk and label embeddings. We bypass Ollama for
embeddings because Ollama's nomic-embed-text wrapper produces collapsed
(numerically identical) vectors for short label inputs. The same model
loaded via sentence-transformers works correctly.

The model is loaded lazily on first use and cached at module scope, so
subsequent calls are fast. The PyO3 host process holds the model resident.
"""

from __future__ import annotations

import os
import threading
from typing import Optional

_MODEL = None
_MODEL_NAME: Optional[str] = None
_LOCK = threading.Lock()


def _get_model(model_name: str):
    """Lazy-load (and cache) the sentence-transformers model.

    Honours `GW_KB_EMBED_DEVICE` ("cpu" / "cuda" / "cuda:N") when set
    so the caller can force a device when the GPU is busy with other
    workloads (e.g. Ollama holding a chat model resident).
    sentence-transformers' default is to pick CUDA if available.
    """
    global _MODEL, _MODEL_NAME
    if _MODEL is not None and _MODEL_NAME == model_name:
        return _MODEL
    with _LOCK:
        if _MODEL is None or _MODEL_NAME != model_name:
            from sentence_transformers import SentenceTransformer
            device = os.environ.get("GW_KB_EMBED_DEVICE") or None
            _MODEL = SentenceTransformer(
                model_name, trust_remote_code=True, device=device
            )
            _MODEL_NAME = model_name
    return _MODEL


def embed_texts(
    texts: list[str],
    model_name: str = "nomic-ai/nomic-embed-text-v1.5",
    batch_size: int = 32,
) -> list[list[float]]:
    """Embed a list of texts and return a list of normalized float vectors.

    Vectors are L2-normalized so cosine similarity reduces to a dot product.
    """
    if not texts:
        return []
    model = _get_model(model_name)
    arr = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    # Convert to plain Python lists for clean PyO3 marshalling
    return arr.tolist()


def embedding_dim(model_name: str = "nomic-ai/nomic-embed-text-v1.5") -> int:
    """Return the embedding dimension for `model_name`."""
    model = _get_model(model_name)
    return int(model.get_sentence_embedding_dimension())
