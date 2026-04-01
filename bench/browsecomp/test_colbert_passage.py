#!/usr/bin/env python3
"""Test ColBERT passage encoding: 1 passage, then 2, then 3, etc.

Usage:
    uv run --project bench/browsecomp --extra colbert \
        python -u bench/browsecomp/test_colbert_passage.py
"""

import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder

def main():
    # Make passages of ~4096 chars each
    base = "The quick brown fox jumps over the lazy dog. " * 91  # ~4095 chars
    passages = [base] * 20  # pool of 20 identical passages

    enc = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    print(f"Device: {enc.device}", flush=True)

    for n in [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]:
        batch = passages[:n]
        t0 = time.monotonic()
        vecs = enc.encode_docs(batch, max_length=512)
        elapsed = time.monotonic() - t0
        print(f"{n} passages ({n*4095} chars): {elapsed:.2f}s ({len(vecs[0])} tok each)", flush=True)

        if elapsed > 10:
            print(f"SLOW at {n} passages. Stopping.", flush=True)
            break

    # Now test with real doc text at increasing sizes
    import json
    CORPUS = "vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl"
    with open(CORPUS) as f:
        for line in f:
            obj = json.loads(line)
            if len(obj["text"]) > 50000:
                break

    full = obj["text"]
    print(f"\nReal doc {obj['docid']}: {len(full)} chars", flush=True)

    CHUNK, OVERLAP = 4096, 800
    for label, size in [("10K", 10000), ("15K", 15000), ("Q", len(full)//4), ("20K", 20000), ("H", len(full)//2), ("full", len(full))]:
        text = full[:size]
        chunks, start = [], 0
        while start < len(text):
            end = min(start + CHUNK, len(text))
            chunks.append(text[start:end])
            if end == len(text):
                break  # last chunk, don't overlap
            start = end - OVERLAP

        t0 = time.monotonic()
        vecs = enc.encode_docs(chunks, max_length=512)
        elapsed = time.monotonic() - t0
        print(f"{label} ({len(text)} chars, {len(chunks)} passages): {elapsed:.2f}s", flush=True)

        if elapsed > 30:
            print(f"SLOW at {label}. Stopping.", flush=True)
            break

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
