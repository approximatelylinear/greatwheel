#!/usr/bin/env python3
"""Build ColBERT passage-level LanceDB index. Streaming, minimal."""

import json, os, sys, time, math
import pyarrow as pa
import lancedb

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder

CHUNK = 4096
OVERLAP = 800
BATCH = 16
WRITE_EVERY = 200


def split(text, chunk=CHUNK, overlap=OVERLAP):
    if len(text) <= chunk:
        return [text]
    out, start = [], 0
    while start < len(text):
        end = min(start + chunk, len(text))
        out.append(text[start:end])
        if end == len(text):
            break  # last chunk, don't overlap
        start = end - overlap
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


def make_array(vectors):
    dim = 128
    floats, offsets = [], [0]
    for vecs in vectors:
        for v in vecs:
            floats.extend(v)
        offsets.append(offsets[-1] + len(vecs))
    flat = pa.array(floats, type=pa.float32())
    inner = pa.FixedSizeListArray.from_arrays(flat, dim)
    return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), inner)


def main():
    corpus = sys.argv[1] if len(sys.argv) > 1 else "vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl"
    dbpath = sys.argv[2] if len(sys.argv) > 2 else "data/colbert-passages-lance"

    print("Loading encoder...", flush=True)
    enc = ColBERTEncoder("lightonai/Reason-ModernColBERT")
    print(f"Device: {enc.device}", flush=True)

    os.makedirs(dbpath, exist_ok=True)
    db = lancedb.connect(dbpath)

    docids_buf, texts_buf, write_buf = [], [], []
    total_p, total_tok, t0 = 0, 0, time.monotonic()

    with open(corpus) as f:
        for lineno, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            did = str(obj["docid"])
            text = obj["text"]
            t = title(text)
            pfx = f"[Title: {t}]\n" if t else ""

            for chunk in split(text):
                docids_buf.append(did)
                texts_buf.append(pfx + chunk)

                if len(texts_buf) >= BATCH:
                    vecs = enc.encode_docs(texts_buf, max_length=512)
                    for d, v in zip(docids_buf, vecs):
                        total_tok += len(v)
                        write_buf.append({"docid": d, "vector": v})
                    docids_buf, texts_buf = [], []
                    total_p += BATCH

                    if len(write_buf) >= WRITE_EVERY:
                        tbl_data = pa.table({
                            "docid": pa.array([x["docid"] for x in write_buf]),
                            "vector": make_array([x["vector"] for x in write_buf]),
                        })
                        try:
                            db.open_table("colbert_passages").add(tbl_data)
                        except Exception:
                            db.create_table("colbert_passages", tbl_data)
                        write_buf = []

            if (lineno + 1) % 2000 == 0:
                elapsed = time.monotonic() - t0
                rate = total_p / elapsed if elapsed > 0 else 0
                eta = (843000 - total_p) / rate / 60 if rate > 0 else 0
                print(f"  {lineno+1} docs | {total_p} passages | {total_tok} tokens | {rate:.0f} p/s | ETA {eta:.0f}m", flush=True)

    # flush
    if texts_buf:
        vecs = enc.encode_docs(texts_buf, max_length=512)
        for d, v in zip(docids_buf, vecs):
            total_tok += len(v)
            write_buf.append({"docid": d, "vector": v})
        total_p += len(texts_buf)

    if write_buf:
        tbl_data = pa.table({
            "docid": pa.array([x["docid"] for x in write_buf]),
            "vector": make_array([x["vector"] for x in write_buf]),
        })
        try:
            db.open_table("colbert_passages").add(tbl_data)
        except Exception:
            db.create_table("colbert_passages", tbl_data)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {total_p} passages, {total_tok} tokens, {elapsed:.0f}s", flush=True)

    # Build index
    print("Creating IVF-HNSW-SQ index...", flush=True)
    tbl = db.open_table("colbert_passages")
    n = tbl.count_rows()
    npart = min(max(int(math.sqrt(total_tok)), 256), 8192)
    print(f"  {n} rows, {npart} partitions", flush=True)
    tbl.create_index(
        metric="cosine", num_partitions=npart,
        vector_column_name="vector", index_type="IVF_HNSW_SQ",
        m=20, ef_construction=300, replace=True,
    )
    print("Index done!", flush=True)


if __name__ == "__main__":
    main()
