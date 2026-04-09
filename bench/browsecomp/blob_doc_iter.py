"""Streaming iterator over (docid, capped_token_list) from a passage blob store.

The blob store is built in corpus order, so all passages of one doc are
contiguous. This iterator emits one doc at a time as soon as the docid
changes — peak RAM is one doc, not the whole corpus.

Used by all `build_*_index.py` scripts so they can run safely without
holding ~58 GB of token data in memory at once.

Usage:
    src = lancedb.connect(blob_dir).open_table("passage_blobs")
    for docid, tokens in iter_docs(src, max_tokens_per_doc=2000):
        # tokens is list[list[float]] of length <= max_tokens_per_doc
        ...
"""

from __future__ import annotations

import lancedb
import numpy as np

TOKEN_DIM = 128
SCAN_BATCH = 8192


def iter_passage_batches(src):
    """Yield Arrow batches from the blob store via offset+limit chunking."""
    total = src.count_rows()
    offset = 0
    while offset < total:
        limit = min(SCAN_BATCH, total - offset)
        batch = (
            src.search()
            .select(["docid", "num_tokens", "vectors"])
            .limit(limit)
            .offset(offset)
            .to_arrow()
        )
        if batch.num_rows == 0:
            break
        yield batch
        offset += batch.num_rows


def iter_docs(src, max_tokens_per_doc: int, max_docs: int | None = None):
    """Yield (docid, capped_token_list_f32) for each doc in the blob store.

    Args:
        src: opened lancedb Table for `passage_blobs`
        max_tokens_per_doc: hard cap on tokens per doc (truncates rest)
        max_docs: optional cap on total docs to yield

    Peak RAM is one doc's worth of tokens (~few MB), independent of
    corpus size.
    """
    cur_docid: str | None = None
    cur_tokens: list[list[float]] = []
    docs_yielded = 0

    for batch in iter_passage_batches(src):
        for docid, n_tok, vec_bytes in zip(
            batch.column("docid").to_pylist(),
            batch.column("num_tokens").to_pylist(),
            batch.column("vectors").to_pylist(),
        ):
            if docid != cur_docid:
                if cur_docid is not None:
                    yield (cur_docid, cur_tokens)
                    docs_yielded += 1
                    if max_docs and docs_yielded >= max_docs:
                        return
                cur_docid = docid
                cur_tokens = []

            if len(cur_tokens) >= max_tokens_per_doc:
                continue
            arr = np.frombuffer(vec_bytes, dtype=np.float16).reshape(n_tok, TOKEN_DIM)
            arr_f32 = arr.astype(np.float32)
            remaining = max_tokens_per_doc - len(cur_tokens)
            cur_tokens.extend(arr_f32[:remaining].tolist())

    if cur_docid is not None:
        yield (cur_docid, cur_tokens)
