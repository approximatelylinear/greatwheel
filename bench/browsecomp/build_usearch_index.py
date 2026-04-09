#!/usr/bin/env python3
"""Build a usearch HNSW index of flattened ColBERT token vectors.

This is the Rust-friendly replacement for `build_voyager_index.py`. Same
encoder, same chunking, same flat-token-vector layout — but the on-disk
format is `usearch` instead of Voyager, which has a first-class Rust
client (`crates/gw-memory/src/colbert/usearch_retriever.rs`).

## On-disk artifacts

Output directory contains:

- `index.usearch`           — usearch HNSW file (mmap-able from Rust via `view()`)
- `passage_to_docid.bin`    — packed binary map; format documented in
                              `colbert/usearch_retriever.rs::load_passage_map`
- `progress.json`           — `{doc_count, passage_count, token_count}` for resume

## Key encoding

Each token vector is keyed by:

    key u64 = (passage_id << 16) | token_idx

`passage_id` indexes into the docid map; `token_idx` is the token's
position within the passage (capped at 2^16-1 = 65535, far above any real
ColBERT max length). The Rust retriever decodes this to recover the docid
via `key >> 16`.

## Why f16 quantization?

usearch supports i8/f16/f32 storage. f16 cuts memory ~2× vs f32 with
negligible recall loss for ColBERT cosine similarity (we already validated
this empirically with float16 token blobs in the rerank path: see
docs/design-colbert-production.md). i8 would shrink another 2× but at
some quality risk we haven't measured. Default: f16.

## Resume

Pass `--resume` to skip docids already indexed. The script reads
`progress.json` to know where to restart.

## Usage

    uv run --project bench/browsecomp --extra colbert \\
        python -u bench/browsecomp/build_usearch_index.py \\
        vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \\
        data/usearch-passages

    # Resume after interrupt
    python -u bench/browsecomp/build_usearch_index.py --resume
"""

import argparse
import json
import os
import struct
import sys
import time
from typing import Iterable

import numpy as np
from usearch.index import Index, MetricKind, ScalarKind

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder
from build_voyager_index import split as split_passages, title as extract_title

TOKEN_DIM = 128
PASSAGE_BITS = 16
TOKEN_MASK = (1 << PASSAGE_BITS) - 1

ENCODE_BATCH = 32
CHECKPOINT_EVERY = 5000  # docs

# usearch HNSW params (mirror voyager defaults: M=12, ef_construction=50)
M = 12
EF_CONSTRUCTION = 50


def encode_key(passage_id: int, token_idx: int) -> int:
    if token_idx > TOKEN_MASK:
        raise ValueError(f"token_idx {token_idx} exceeds {PASSAGE_BITS}-bit cap")
    return (passage_id << PASSAGE_BITS) | token_idx


def write_passage_map(path: str, passage_to_docid: dict[int, str]) -> None:
    """Write the binary docid map atomically. Format documented in
    colbert/usearch_retriever.rs::load_passage_map."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(passage_to_docid)))
        # Sort by passage_id for deterministic output
        for pid in sorted(passage_to_docid.keys()):
            docid = passage_to_docid[pid]
            docid_bytes = docid.encode("utf-8")
            f.write(struct.pack("<Q", pid))
            f.write(struct.pack("<I", len(docid_bytes)))
            f.write(docid_bytes)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_checkpoint(out_dir: str, idx: Index, passage_to_docid: dict[int, str],
                    doc_count: int, passage_count: int, token_count: int) -> None:
    """Save index + map + progress atomically."""
    idx_path = os.path.join(out_dir, "index.usearch")
    map_path = os.path.join(out_dir, "passage_to_docid.bin")
    progress_path = os.path.join(out_dir, "progress.json")

    idx_tmp = idx_path + ".tmp"
    progress_tmp = progress_path + ".tmp"

    idx.save(idx_tmp)
    with open(progress_tmp, "w") as f:
        json.dump({
            "doc_count": doc_count,
            "passage_count": passage_count,
            "token_count": token_count,
            "n_passages": len(passage_to_docid),
        }, f)
        f.flush()
        os.fsync(f.fileno())

    write_passage_map(map_path, passage_to_docid)
    os.replace(idx_tmp, idx_path)
    os.replace(progress_tmp, progress_path)

    size_gb = os.path.getsize(idx_path) / 1e9
    print(f"  [checkpoint] {doc_count} docs / {passage_count} passages / "
          f"{token_count} tokens / {size_gb:.1f} GB", flush=True)


def load_progress(out_dir: str) -> tuple[int, int, int, set[str]]:
    """Read progress.json and the docid map to enable resume.
    Returns (doc_count, passage_count, token_count, seen_docids)."""
    progress_path = os.path.join(out_dir, "progress.json")
    if not os.path.exists(progress_path):
        return 0, 0, 0, set()

    with open(progress_path) as f:
        progress = json.load(f)
    doc_count = progress.get("doc_count", 0)
    passage_count = progress.get("passage_count", 0)
    token_count = progress.get("token_count", 0)

    # We can't recover the docid set from the binary file alone (it's
    # passage_id → docid). To know which docs are done, we just count from
    # the corpus file in order — same as build_voyager_index.py uses
    # corpus order as the canonical iteration. The caller skips docs by
    # index, not by lookup.
    return doc_count, passage_count, token_count, set()


def build(corpus_path: str, out_dir: str, max_docs: int | None = None, resume: bool = False):
    os.makedirs(out_dir, exist_ok=True)

    # Load encoder
    enc = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    print(f"Device: {enc.device}", flush=True)

    # Init index
    idx = Index(
        ndim=TOKEN_DIM,
        metric=MetricKind.Cos,
        dtype=ScalarKind.F16,
        connectivity=M,
        expansion_add=EF_CONSTRUCTION,
        expansion_search=EF_CONSTRUCTION,
    )
    print(f"Index: dim={TOKEN_DIM}, metric=Cos, dtype=f16, M={M}, ef={EF_CONSTRUCTION}", flush=True)

    doc_count = 0
    passage_count = 0
    token_count = 0
    next_passage_id = 0
    passage_to_docid: dict[int, str] = {}
    skip_until = 0

    if resume:
        doc_count, passage_count, token_count, _ = load_progress(out_dir)
        if doc_count > 0:
            print(f"Resuming from doc {doc_count} ({passage_count} passages)", flush=True)
            skip_until = doc_count

            # Load existing index
            idx_path = os.path.join(out_dir, "index.usearch")
            if os.path.exists(idx_path):
                idx.load(idx_path)
                print(f"  loaded existing index ({idx.size} vectors)", flush=True)

            # Load existing passage map (we need it to set next_passage_id)
            map_path = os.path.join(out_dir, "passage_to_docid.bin")
            if os.path.exists(map_path):
                with open(map_path, "rb") as f:
                    count = struct.unpack("<Q", f.read(8))[0]
                    for _ in range(count):
                        pid = struct.unpack("<Q", f.read(8))[0]
                        dlen = struct.unpack("<I", f.read(4))[0]
                        d = f.read(dlen).decode("utf-8")
                        passage_to_docid[pid] = d
                next_passage_id = max(passage_to_docid.keys()) + 1 if passage_to_docid else 0
                print(f"  loaded {len(passage_to_docid)} passages from map, "
                      f"next_passage_id={next_passage_id}", flush=True)

    # Buffers
    encode_buf_texts: list[str] = []
    encode_buf_meta: list[int] = []  # passage_id for each text in buf

    def flush_encode():
        nonlocal token_count
        if not encode_buf_texts:
            return
        tensors = enc._encode(encode_buf_texts, max_length=512, is_query=False)
        # Vectorized batch construction: avoid per-token Python loops.
        # For each passage: keys = (pid << 16) | arange(n) and the matching
        # rows of the token tensor. Concatenate across the whole batch and
        # do ONE multi-threaded idx.add().
        key_chunks: list[np.ndarray] = []
        vec_chunks: list[np.ndarray] = []
        for pid, tensor in zip(encode_buf_meta, tensors):
            arr = tensor.numpy().astype(np.float32, copy=False)  # (n_tokens, 128)
            n = arr.shape[0]
            if n == 0:
                continue
            keys = (np.uint64(pid) << np.uint64(PASSAGE_BITS)) | np.arange(n, dtype=np.uint64)
            key_chunks.append(keys)
            vec_chunks.append(arr)
            token_count += n
        if key_chunks:
            keys_np = np.concatenate(key_chunks)
            vecs_np = np.concatenate(vec_chunks, axis=0)
            # threads=0 → use all available cores for HNSW insert
            idx.add(keys_np, vecs_np, threads=0)
        encode_buf_texts.clear()
        encode_buf_meta.clear()

    t0 = time.monotonic()
    line_no = 0
    with open(corpus_path) as f:
        for line in f:
            line_no += 1
            line = line.strip()
            if not line:
                continue

            # Resume: skip docs already encoded
            if line_no <= skip_until:
                continue

            obj = json.loads(line)
            docid = str(obj["docid"])
            doc_count += 1

            text = obj["text"]
            t = extract_title(text)
            pfx = f"[Title: {t}]\n" if t else ""

            for chunk in split_passages(text):
                pid = next_passage_id
                next_passage_id += 1
                passage_to_docid[pid] = docid
                encode_buf_texts.append(pfx + chunk)
                encode_buf_meta.append(pid)
                passage_count += 1

                if len(encode_buf_texts) >= ENCODE_BATCH:
                    flush_encode()

            if doc_count % 2000 == 0:
                elapsed = time.monotonic() - t0
                rate = passage_count / elapsed if elapsed > 0 else 0
                print(f"  {doc_count} docs | {passage_count} passages | "
                      f"{token_count} tokens | {rate:.0f} p/s", flush=True)

            if doc_count % CHECKPOINT_EVERY == 0:
                flush_encode()
                save_checkpoint(out_dir, idx, passage_to_docid,
                                doc_count, passage_count, token_count)

            if max_docs and doc_count >= max_docs:
                break

    # Final flush + save (skip if loop already saved at this boundary)
    flush_encode()
    if doc_count % CHECKPOINT_EVERY != 0:
        save_checkpoint(out_dir, idx, passage_to_docid,
                        doc_count, passage_count, token_count)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {doc_count} docs, {passage_count} passages, "
          f"{token_count} tokens, {elapsed:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?",
                        default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    parser.add_argument("out_dir", nargs="?", default="data/usearch-passages")
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--test", action="store_true", help="Test on 50 docs")
    args = parser.parse_args()

    if args.test:
        args.max_docs = 50
        args.out_dir = "/tmp/usearch-test"

    build(args.corpus, args.out_dir, max_docs=args.max_docs, resume=args.resume)


if __name__ == "__main__":
    main()
