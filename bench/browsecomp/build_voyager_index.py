#!/usr/bin/env python3
"""Build a Voyager HNSW index from ColBERT passage embeddings.

Flattens all token vectors into one Voyager index, maintains
token→docid mapping. Uses our ColBERTEncoder (no PyLate needed).
Checkpoints every 10K docs for resume support.

Usage:
    uv run --project bench/browsecomp --extra colbert \
        python -u bench/browsecomp/build_voyager_index.py \
        vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \
        data/voyager-passages

    # Resume after interrupt:
    python -u bench/browsecomp/build_voyager_index.py --resume

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
CHECKPOINT_EVERY = 5000  # docs


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


def save_checkpoint(out_dir, idx, token_to_docid, doc_count, passage_count):
    """Save index + mapping + progress atomically.

    Writes to .tmp files first, then renames. This prevents partial
    writes from corrupting an existing checkpoint if interrupted.
    """
    idx_path = os.path.join(out_dir, "index.voyager")
    map_path = os.path.join(out_dir, "token_to_docid.pkl")
    progress_path = os.path.join(out_dir, "progress.json")

    # Write to temp files
    idx_tmp = idx_path + ".tmp"
    map_tmp = map_path + ".tmp"
    progress_tmp = progress_path + ".tmp"

    idx.save(idx_tmp)
    with open(map_tmp, "wb") as f:
        pickle.dump(token_to_docid, f, protocol=pickle.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    with open(progress_tmp, "w") as f:
        json.dump({"doc_count": doc_count, "passage_count": passage_count,
                    "token_count": len(token_to_docid)}, f)
        f.flush()
        os.fsync(f.fileno())

    # Atomic renames (last)
    os.replace(idx_tmp, idx_path)
    os.replace(map_tmp, map_path)
    os.replace(progress_tmp, progress_path)

    size_gb = os.path.getsize(idx_path) / 1e9
    print(f"  [checkpoint] {doc_count} docs, {len(token_to_docid)} tokens, {size_gb:.1f} GB", flush=True)


def load_checkpoint(out_dir):
    """Load existing checkpoint if available. Returns (idx, token_to_docid, doc_count, passage_count)."""
    idx_path = os.path.join(out_dir, "index.voyager")
    map_path = os.path.join(out_dir, "token_to_docid.pkl")
    progress_path = os.path.join(out_dir, "progress.json")

    if not all(os.path.exists(p) for p in [idx_path, map_path, progress_path]):
        return None, None, 0, 0

    idx = voyager.Index.load(idx_path)
    with open(map_path, "rb") as f:
        token_to_docid = pickle.load(f)
    with open(progress_path) as f:
        progress = json.load(f)

    print(f"  [resume] Loaded checkpoint: {progress['doc_count']} docs, "
          f"{progress['token_count']} tokens", flush=True)
    return idx, token_to_docid, progress["doc_count"], progress["passage_count"]


def build(corpus_path, out_dir, batch_size=32, max_docs=None, resume=False):
    os.makedirs(out_dir, exist_ok=True)

    enc = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    print(f"Device: {enc.device}", flush=True)

    # Resume or fresh start
    skip_docs = 0
    if resume:
        idx, token_to_docid, skip_docs, passage_count = load_checkpoint(out_dir)
        if idx is None:
            print("  No checkpoint found, starting fresh", flush=True)

    if not resume or skip_docs == 0:
        idx = voyager.Index(voyager.Space.Cosine, num_dimensions=128,
                            storage_data_type=voyager.StorageDataType.E4M3,
                            M=12, ef_construction=50)
        token_to_docid = {}
        passage_count = 0

    t0 = time.monotonic()
    doc_count = 0

    buf_docids = []
    buf_texts = []
    vec_buffer = []
    docid_buffer = []

    with open(corpus_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            doc_count += 1

            # Skip already-processed docs on resume
            if doc_count <= skip_docs:
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
                    _encode_and_add(enc, idx, token_to_docid, buf_docids, buf_texts, vec_buffer, docid_buffer)
                    passage_count += len(buf_texts)
                    buf_docids, buf_texts = [], []

            if doc_count % 2000 == 0:
                elapsed = time.monotonic() - t0
                docs_done = doc_count - skip_docs
                rate = (passage_count - (0 if not resume else 0)) / elapsed if elapsed > 0 else 0
                print(f"  {doc_count} docs | {passage_count} passages | {len(token_to_docid)} tokens | {rate:.0f} p/s", flush=True)

            if doc_count % CHECKPOINT_EVERY == 0:
                # Flush vectors before checkpoint
                _flush_buffer(idx, token_to_docid, vec_buffer, docid_buffer)
                save_checkpoint(out_dir, idx, token_to_docid, doc_count, passage_count)

            if max_docs and doc_count >= max_docs:
                break

    # Flush remaining
    if buf_texts:
        _encode_and_add(enc, idx, token_to_docid, buf_docids, buf_texts, vec_buffer, docid_buffer)
        passage_count += len(buf_texts)
    _flush_buffer(idx, token_to_docid, vec_buffer, docid_buffer)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {doc_count} docs, {passage_count} passages, {len(token_to_docid)} tokens, {elapsed:.0f}s", flush=True)

    # Final save (skip if loop already checkpointed at this doc_count)
    if doc_count % CHECKPOINT_EVERY != 0:
        save_checkpoint(out_dir, idx, token_to_docid, doc_count, passage_count)


def _encode_and_add(enc, idx, token_to_docid, docids, texts, vec_buffer, docid_buffer):
    tensors = enc._encode(texts, max_length=512, is_query=False)
    for docid, tensor in zip(docids, tensors):
        if tensor.numel() == 0:
            continue
        arr = tensor.numpy()
        vec_buffer.append(arr)
        docid_buffer.extend([docid] * len(arr))

    if sum(len(v) for v in vec_buffer) >= 50000:
        _flush_buffer(idx, token_to_docid, vec_buffer, docid_buffer)


def _flush_buffer(idx, token_to_docid, vec_buffer, docid_buffer):
    if not vec_buffer:
        return
    all_vecs = np.concatenate(vec_buffer, axis=0)
    ids = idx.add_items(all_vecs)
    for tid, docid in zip(ids, docid_buffer):
        token_to_docid[tid] = docid
    vec_buffer.clear()
    docid_buffer.clear()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", nargs="?", default="vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl")
    parser.add_argument("out_dir", nargs="?", default="data/voyager-passages")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--test", action="store_true", help="Test on 100 docs")
    args = parser.parse_args()

    if args.test:
        args.max_docs = 100
        args.out_dir = "/tmp/voyager-test"

    build(args.corpus, args.out_dir, args.batch_size, args.max_docs, args.resume)


if __name__ == "__main__":
    main()
