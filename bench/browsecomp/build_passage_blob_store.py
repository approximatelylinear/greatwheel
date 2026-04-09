#!/usr/bin/env python3
"""Build a passage-keyed token-tensor blob store from the BrowseComp corpus.

This is the read-side of the production rerank architecture (Option B in
docs/design-colbert-production.md). It encodes every passage of every doc
ONCE with Reason-ModernColBERT and stores the resulting token tensors as
raw float16 bytes in a Lance table, keyed by (docid, chunk_idx).

At query time, the searcher fetches the precomputed tensors for the
candidate docs (no GPU encoding) and runs MaxSim on them — turning the
~90s/query rerank cost into milliseconds.

Why bytes instead of Lance's multi-vector type:
  - Lance multi-vector blew up to 391 GB on this corpus (earlier abandoned
    attempt). Float16-as-bytes is ~110 GB.
  - We don't need ANN search over the tensors. We need point lookups by
    docid. Bytes is the simplest and fastest representation for that.

Schema:
  docid       (string)
  chunk_idx   (int32)
  num_tokens  (int32)
  vectors     (binary, num_tokens * 128 * 2 bytes)

Usage:
    uv run --project bench/browsecomp --extra colbert \\
        python -u bench/browsecomp/build_passage_blob_store.py \\
        vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \\
        data/passage-blobs

    # Resume after interrupt — already-encoded docids are skipped
    python -u bench/browsecomp/build_passage_blob_store.py --resume
"""

import argparse
import json
import os
import sys
import time

import lancedb
import numpy as np
import pyarrow as pa

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder
from build_voyager_index import split as split_passages, title as extract_title

TABLE = "passage_blobs"
ENCODE_BATCH = 32   # passages per encoder call
WRITE_EVERY = 4096  # rows per Lance append


def existing_docids(db) -> set:
    """Return set of docids already in the blob store (for resume)."""
    try:
        tbl = db.open_table(TABLE)
    except (ValueError, FileNotFoundError):
        return set()
    # Column-only scan via the search() builder — works without pylance
    arrow_tbl = tbl.search().select(["docid"]).limit(0).to_arrow()
    return set(arrow_tbl.column("docid").to_pylist())


def build(corpus_path: str, out_dir: str, max_docs: int | None = None, resume: bool = False):
    os.makedirs(out_dir, exist_ok=True)
    db = lancedb.connect(out_dir)

    skip_docids: set = set()
    if resume:
        print("Loading existing docids for resume...", flush=True)
        t0 = time.monotonic()
        skip_docids = existing_docids(db)
        print(f"  {len(skip_docids)} docs already encoded ({time.monotonic()-t0:.1f}s)", flush=True)

    enc = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    print(f"Device: {enc.device}", flush=True)

    encode_buf_texts: list[str] = []
    encode_buf_meta: list[tuple[str, int]] = []  # (docid, chunk_idx)
    write_buf_rows: list[dict] = []
    doc_count = 0
    passage_count = 0
    token_count = 0
    t0 = time.monotonic()

    def flush_encode():
        nonlocal token_count
        if not encode_buf_texts:
            return
        tensors = enc._encode(encode_buf_texts, max_length=512, is_query=False)
        for (docid, chunk_idx), tensor in zip(encode_buf_meta, tensors):
            if tensor.numel() == 0:
                continue
            arr = tensor.numpy().astype(np.float16)  # (num_tokens, 128) float16
            n = arr.shape[0]
            token_count += n
            write_buf_rows.append({
                "docid": docid,
                "chunk_idx": chunk_idx,
                "num_tokens": n,
                "vectors": arr.tobytes(),
            })
        encode_buf_texts.clear()
        encode_buf_meta.clear()

    def flush_write():
        if not write_buf_rows:
            return
        tbl_data = pa.table({
            "docid": pa.array([r["docid"] for r in write_buf_rows], pa.string()),
            "chunk_idx": pa.array([r["chunk_idx"] for r in write_buf_rows], pa.int32()),
            "num_tokens": pa.array([r["num_tokens"] for r in write_buf_rows], pa.int32()),
            "vectors": pa.array([r["vectors"] for r in write_buf_rows], pa.binary()),
        })
        try:
            db.open_table(TABLE).add(tbl_data)
        except (ValueError, FileNotFoundError):
            db.create_table(TABLE, tbl_data)
        write_buf_rows.clear()

    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docid = str(obj["docid"])

            if docid in skip_docids:
                continue

            doc_count += 1
            text = obj["text"]
            t = extract_title(text)
            pfx = f"[Title: {t}]\n" if t else ""

            for chunk_idx, chunk in enumerate(split_passages(text)):
                encode_buf_texts.append(pfx + chunk)
                encode_buf_meta.append((docid, chunk_idx))
                passage_count += 1

                if len(encode_buf_texts) >= ENCODE_BATCH:
                    flush_encode()

                if len(write_buf_rows) >= WRITE_EVERY:
                    flush_write()

            if doc_count % 2000 == 0:
                elapsed = time.monotonic() - t0
                rate = passage_count / elapsed if elapsed > 0 else 0
                print(f"  {doc_count} docs | {passage_count} passages | "
                      f"{token_count} tokens | {rate:.0f} p/s", flush=True)

            if max_docs and doc_count >= max_docs:
                break

    # Final flush
    flush_encode()
    flush_write()

    elapsed = time.monotonic() - t0
    print(f"\nDone: {doc_count} new docs, {passage_count} passages, "
          f"{token_count} tokens, {elapsed:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?",
                        default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    parser.add_argument("out_dir", nargs="?", default="data/passage-blobs")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test on 100 docs")
    args = parser.parse_args()

    if args.test:
        args.max_docs = 100
        args.out_dir = "/tmp/passage-blobs-test"

    build(args.corpus, args.out_dir, max_docs=args.max_docs, resume=args.resume)


if __name__ == "__main__":
    main()
