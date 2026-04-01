#!/usr/bin/env python3
"""Build a Voyager HNSW index from ColBERT passage embeddings.

Flattens all token vectors into one Voyager index, maintains
token→docid mapping. Uses our ColBERTEncoder (no PyLate needed).

Usage:
    uv run --project bench/browsecomp --extra colbert \
        python -u bench/browsecomp/build_voyager_index.py \
        vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \
        data/voyager-passages

Test first:
    python -u bench/browsecomp/build_voyager_index.py --test
"""

import argparse, json, os, sys, time, pickle
import numpy as np
import voyager

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder

CHUNK = 4096
OVERLAP = 800


def split(text):
    if len(text) <= CHUNK:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + CHUNK, len(text))
        out.append(text[start:end])
        if end == len(text):
            break
        start = end - OVERLAP
    return out


def title(text):
    in_fm = False
    for line in text.split("\n"):
        s = line.strip()
        if s == "---":
            if in_fm: break
            in_fm = True; continue
        if in_fm and s.startswith("title:"):
            return s.split(":", 1)[1].strip()
    return ""


def build(corpus_path, out_dir, batch_size=8, max_docs=None):
    os.makedirs(out_dir, exist_ok=True)

    enc = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    print(f"Device: {enc.device}", flush=True)

    # E4M3 quantization: ~5x smaller than Float32, preserves MaxSim rankings
    idx = voyager.Index(voyager.Space.Cosine, num_dimensions=128,
                        storage_data_type=voyager.StorageDataType.E4M3)
    token_to_docid = {}  # token_vector_id → docid string

    t0 = time.monotonic()
    doc_count = 0
    passage_count = 0
    token_count = 0

    # Buffer for batch encoding
    buf_docids = []
    buf_texts = []

    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            docid = str(obj["docid"])
            text = obj["text"]
            t = title(text)
            pfx = f"[Title: {t}]\n" if t else ""

            for chunk in split(text):
                buf_docids.append(docid)
                buf_texts.append(pfx + chunk)

                if len(buf_texts) >= batch_size:
                    _encode_and_add(enc, idx, token_to_docid, buf_docids, buf_texts)
                    token_count += sum(1 for _ in buf_texts)  # approximate
                    passage_count += len(buf_texts)
                    buf_docids, buf_texts = [], []

            doc_count += 1
            if doc_count % 2000 == 0:
                elapsed = time.monotonic() - t0
                rate = passage_count / elapsed if elapsed > 0 else 0
                print(f"  {doc_count} docs | {passage_count} passages | {len(token_to_docid)} tokens | {rate:.0f} p/s", flush=True)

            if max_docs and doc_count >= max_docs:
                break

    # Flush
    if buf_texts:
        _encode_and_add(enc, idx, token_to_docid, buf_docids, buf_texts)
        passage_count += len(buf_texts)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {doc_count} docs, {passage_count} passages, {len(token_to_docid)} tokens, {elapsed:.0f}s", flush=True)

    # Save
    idx_path = os.path.join(out_dir, "index.voyager")
    map_path = os.path.join(out_dir, "token_to_docid.pkl")

    idx.save(idx_path)
    with open(map_path, "wb") as f:
        pickle.dump(token_to_docid, f)

    print(f"Saved: {idx_path} ({os.path.getsize(idx_path) / 1e9:.1f} GB)", flush=True)
    print(f"Saved: {map_path} ({os.path.getsize(map_path) / 1e6:.1f} MB)", flush=True)


def _encode_and_add(enc, idx, token_to_docid, docids, texts):
    # Use _encode directly to get tensors, skip tolist() overhead
    tensors = enc._encode(texts, max_length=512, is_query=False)
    for docid, tensor in zip(docids, tensors):
        if tensor.numel() == 0:
            continue
        arr = tensor.numpy()  # already on CPU from _encode
        ids = idx.add_items(arr)
        for tid in ids:
            token_to_docid[tid] = docid


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    parser.add_argument("out_dir", nargs="?", default="data/voyager-passages")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--test", action="store_true", help="Test on 100 docs")
    args = parser.parse_args()

    if args.test:
        args.max_docs = 100
        args.out_dir = "/tmp/voyager-test"

    build(args.corpus, args.out_dir, args.batch_size, args.max_docs)


if __name__ == "__main__":
    main()
