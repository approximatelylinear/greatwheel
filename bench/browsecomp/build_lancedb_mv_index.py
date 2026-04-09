#!/usr/bin/env python3
"""Build a LanceDB native multi-vector index from the existing passage blob store.

LanceDB 0.29+ supports multi-vector columns of type
`List<FixedSizeList<f32, dim>>` and `tbl.search(q)` over them computes
sum-of-max-of-cosine natively — i.e. exactly the MaxSim we want, in C++,
no Python loop. By storing one row per **doc** (not per passage) with the
concatenation of all the doc's passage tokens, we get the per-doc-max
behavior for free, since `max(max(p1), max(p2)) == max(p1 ∪ p2)`.

This script reformats the existing `data/passage-blobs` Lance store into
a new `data/lancedb-mv` table without re-encoding anything. ~5-10 min vs
the 80 min the encoder build took.

Output schema:
    docid     : string
    n_tokens  : int32       -- total tokens across all of this doc's passages
    tokens    : list<fixed_size_list<float32, 128>>

Usage:
    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/build_lancedb_mv_index.py \\
        data/passage-blobs data/lancedb-mv
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import lancedb
import numpy as np
import pyarrow as pa

TOKEN_DIM = 128
DOC_BATCH = 256  # docs per write batch
MAX_TOKENS_PER_DOC = 2000  # standardized across all backends for fair comparison


SCAN_BATCH = 8192  # passage rows per Lance fetch


def _iter_passage_batches(src):
    """Yield passage batches from the blob store via offset/limit chunking."""
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


def _doc_token_iter(src, max_docs: int | None):
    """Stream (docid, capped_token_list_f32) tuples in doc order.

    Relies on the fact that the blob store is built in corpus order, so
    all passages of one doc are contiguous. We accumulate tokens for the
    current doc until the docid changes, then yield it. Peak memory is
    one doc, not the whole corpus.
    """
    cur_docid: str | None = None
    cur_tokens: list[list[float]] = []
    docs_yielded = 0

    def _emit():
        nonlocal cur_docid, cur_tokens, docs_yielded
        if cur_docid is None:
            return None
        out = (cur_docid, cur_tokens)
        cur_docid = None
        cur_tokens = []
        docs_yielded += 1
        return out

    for batch in _iter_passage_batches(src):
        for docid, n_tok, vec_bytes in zip(
            batch.column("docid").to_pylist(),
            batch.column("num_tokens").to_pylist(),
            batch.column("vectors").to_pylist(),
        ):
            if docid != cur_docid:
                emitted = _emit()
                if emitted is not None:
                    yield emitted
                    if max_docs and docs_yielded >= max_docs:
                        return
                cur_docid = docid
                cur_tokens = []

            if len(cur_tokens) >= MAX_TOKENS_PER_DOC:
                continue
            arr = np.frombuffer(vec_bytes, dtype=np.float16).reshape(n_tok, TOKEN_DIM)
            arr_f32 = arr.astype(np.float32)
            remaining = MAX_TOKENS_PER_DOC - len(cur_tokens)
            cur_tokens.extend(arr_f32[:remaining].tolist())

    emitted = _emit()
    if emitted is not None:
        yield emitted


def build(blob_dir: str, out_dir: str, max_docs: int | None = None) -> None:
    print(f"Opening source blob store: {blob_dir}", flush=True)
    src_db = lancedb.connect(blob_dir)
    src = src_db.open_table("passage_blobs")
    n_passages = src.count_rows()
    print(f"  {n_passages} passages", flush=True)

    # Output schema: list of fixed-size token vectors per doc.
    schema = pa.schema([
        pa.field("docid", pa.string()),
        pa.field("n_tokens", pa.int32()),
        pa.field("tokens", pa.list_(pa.list_(pa.float32(), TOKEN_DIM))),
    ])

    os.makedirs(out_dir, exist_ok=True)
    out_db = lancedb.connect(out_dir)

    print("Streaming docs (peak RAM = 1 doc) and writing...", flush=True)
    write_buf: list[dict] = []
    written = 0
    t0 = time.monotonic()

    for docid, all_tokens in _doc_token_iter(src, max_docs):
        write_buf.append({
            "docid": docid,
            "n_tokens": len(all_tokens),
            "tokens": all_tokens,
        })

        if len(write_buf) >= DOC_BATCH:
            tbl_data = pa.Table.from_pylist(write_buf, schema=schema)
            try:
                out_db.open_table("docs_mv").add(tbl_data)
            except (ValueError, FileNotFoundError):
                out_db.create_table("docs_mv", tbl_data)
            written += len(write_buf)
            write_buf = []
            elapsed = time.monotonic() - t0
            rate = written / elapsed if elapsed > 0 else 0
            print(f"  wrote {written} docs ({rate:.0f} docs/s)", flush=True)

    if write_buf:
        tbl_data = pa.Table.from_pylist(write_buf, schema=schema)
        try:
            out_db.open_table("docs_mv").add(tbl_data)
        except (ValueError, FileNotFoundError):
            out_db.create_table("docs_mv", tbl_data)
        written += len(write_buf)

    elapsed = time.monotonic() - t0
    print(f"\nDone: wrote {written} docs in {elapsed:.0f}s", flush=True)
    final = out_db.open_table("docs_mv")
    print(f"  table rows: {final.count_rows()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blob_dir", nargs="?", default="data/passage-blobs")
    parser.add_argument("out_dir", nargs="?", default="data/lancedb-mv")
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap docs (for smoke testing)")
    args = parser.parse_args()
    build(args.blob_dir, args.out_dir, args.max_docs)


if __name__ == "__main__":
    main()
