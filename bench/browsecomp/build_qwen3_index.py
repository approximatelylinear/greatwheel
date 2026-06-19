#!/usr/bin/env python3
"""Build a Qwen3-Embedding LanceDB index from the BrowseComp corpus.

Mirrors the BrowseComp-Plus indexing recipe: encode each document as a
single vector at max_length=4096 with no instruction prefix, EOS pooling,
L2-normalized. Queries get the instruction prefix at search time inside
the encoder service. One row per docid.

Schema:
    docid   (string)
    vector  (fixed_size_list<float32, dim>)

Usage:
    # 1. Start the encoder service
    uv run --project bench/browsecomp --extra qwen3-embed \\
        python bench/browsecomp/qwen3_embed_server.py --port 8003

    # 2. Build the index
    uv run --project bench/browsecomp --extra qwen3-embed \\
        python -u bench/browsecomp/build_qwen3_index.py \\
        vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \\
        data/qwen3-embed --encoder-url http://127.0.0.1:8003

    # Resume after interrupt — already-encoded docids are skipped
    python -u bench/browsecomp/build_qwen3_index.py --resume
"""

import argparse
import json
import os
import sys
import time

import lancedb
import pyarrow as pa
import requests

TABLE = "qwen3_docs"
DEFAULT_BATCH = 8
WRITE_EVERY = 256


def existing_docids(db) -> set:
    try:
        tbl = db.open_table(TABLE)
    except (ValueError, FileNotFoundError):
        return set()
    arrow_tbl = tbl.search().select(["docid"]).limit(0).to_arrow()
    return set(arrow_tbl.column("docid").to_pylist())


def encoder_dim(encoder_url: str) -> int:
    r = requests.get(f"{encoder_url.rstrip('/')}/", timeout=30)
    r.raise_for_status()
    info = r.json()
    return int(info["dim"])


def encode_batch(encoder_url: str, texts: list[str], max_length: int) -> list[list[float]]:
    r = requests.post(
        f"{encoder_url.rstrip('/')}/encode_doc_batch",
        json={"texts": texts, "max_length": max_length},
        timeout=600,
    )
    r.raise_for_status()
    return r.json()["vectors"]


def build(
    corpus_path: str,
    out_dir: str,
    encoder_url: str,
    batch_size: int,
    max_length: int,
    max_docs: int | None,
    resume: bool,
    sort_by_length: bool,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    db = lancedb.connect(out_dir)

    skip_docids: set = set()
    if resume:
        print("Loading existing docids for resume...", flush=True)
        t0 = time.monotonic()
        skip_docids = existing_docids(db)
        print(f"  {len(skip_docids)} docs already encoded ({time.monotonic()-t0:.1f}s)", flush=True)

    dim = encoder_dim(encoder_url)
    print(f"Encoder dim: {dim}", flush=True)

    schema = pa.schema([
        pa.field("docid", pa.string()),
        pa.field("vector", pa.list_(pa.float32(), dim)),
    ])

    pending_texts: list[str] = []
    pending_docids: list[str] = []
    write_buf: list[dict] = []
    written = 0
    seen = 0
    t0 = time.monotonic()

    def flush_encode():
        nonlocal pending_texts, pending_docids
        if not pending_texts:
            return
        vecs = encode_batch(encoder_url, pending_texts, max_length)
        for docid, vec in zip(pending_docids, vecs):
            write_buf.append({"docid": docid, "vector": vec})
        pending_texts = []
        pending_docids = []

    def flush_write():
        nonlocal write_buf, written
        if not write_buf:
            return
        tbl_data = pa.Table.from_pylist(write_buf, schema=schema)
        try:
            db.open_table(TABLE).add(tbl_data)
        except (ValueError, FileNotFoundError):
            db.create_table(TABLE, tbl_data)
        written += len(write_buf)
        write_buf.clear()
        elapsed = time.monotonic() - t0
        rate = written / elapsed if elapsed > 0 else 0.0
        print(f"  wrote {written} docs ({rate:.1f} docs/s)", flush=True)

    print(f"Loading corpus: {corpus_path}", flush=True)
    docs: list[tuple[str, str]] = []
    with open(corpus_path) as f:
        for line in f:
            obj = json.loads(line)
            docid = str(obj["docid"])
            text = obj.get("text", "")
            seen += 1
            if docid in skip_docids:
                continue
            if not text:
                continue
            docs.append((docid, text))
            if max_docs and len(docs) >= max_docs:
                break
    print(f"  {len(docs)} docs to encode ({seen} scanned, {len(skip_docids)} skipped)", flush=True)

    if sort_by_length:
        # Char length is a cheap proxy for token length. Grouping similar-length
        # docs into adjacent batches slashes padding waste on this long-tailed corpus
        # (p50=1300 tokens but max=714K, so random batching pads most items to 4096).
        docs.sort(key=lambda d: len(d[1]))
        print(f"  sorted by length (char proxy)", flush=True)

    for docid, text in docs:
        pending_texts.append(text)
        pending_docids.append(docid)

        if len(pending_texts) >= batch_size:
            flush_encode()

        if len(write_buf) >= WRITE_EVERY:
            flush_write()

    flush_encode()
    flush_write()

    elapsed = time.monotonic() - t0
    print(f"\nDone: encoded {written} new docs ({seen} scanned) in {elapsed:.0f}s", flush=True)
    final = db.open_table(TABLE)
    print(f"  table rows: {final.count_rows()}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "corpus_path",
        nargs="?",
        default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl",
    )
    parser.add_argument("out_dir", nargs="?", default="data/qwen3-embed")
    parser.add_argument("--encoder-url", default="http://127.0.0.1:8003")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-docs", type=int, default=None,
                        help="Cap docs (for smoke testing)")
    parser.add_argument("--resume", action="store_true",
                        help="Skip docids already in the output table")
    parser.add_argument("--sort-by-length", action="store_true", default=True,
                        help="Sort docs by length before batching (default; cuts padding waste on long-tailed corpora)")
    parser.add_argument("--no-sort-by-length", dest="sort_by_length", action="store_false")
    args = parser.parse_args()

    build(
        corpus_path=args.corpus_path,
        out_dir=args.out_dir,
        encoder_url=args.encoder_url,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_docs=args.max_docs,
        resume=args.resume,
        sort_by_length=args.sort_by_length,
    )


if __name__ == "__main__":
    main()
