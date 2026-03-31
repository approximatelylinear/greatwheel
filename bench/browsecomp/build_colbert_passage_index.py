"""Build a ColBERT passage-level LanceDB index.

Reads the corpus JSONL, splits into passages (with title metadata),
encodes each passage with Reason-ModernColBERT, and stores per-token
embeddings in LanceDB.

This gives ColBERT passage-level retrieval: search returns the most
relevant passage (not document), with MaxSim scoring over the
passage's token embeddings.

Usage:
    uv run --project bench/browsecomp --extra colbert --extra lancedb \
        python bench/browsecomp/build_colbert_passage_index.py \
        --corpus-jsonl vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \
        --db-path data/colbert-passages-lance \
        --chunk-bytes 4096 --overlap-bytes 800
"""

import argparse
import json
import os
import re
import sys
import time

import pyarrow as pa
import lancedb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder


def extract_title(text: str) -> str:
    """Extract title from YAML frontmatter."""
    in_fm = False
    for line in text.split("\n"):
        stripped = line.strip()
        if stripped == "---":
            if in_fm:
                break
            in_fm = True
            continue
        if in_fm and stripped.startswith("title:"):
            return stripped.split(":", 1)[1].strip()
    return ""


def split_passages(text: str, chunk_bytes: int, overlap_bytes: int) -> list[str]:
    """Split text into overlapping passages."""
    if len(text) <= chunk_bytes:
        return [text]
    passages = []
    start = 0
    while start < len(text):
        end = min(start + chunk_bytes, len(text))
        # Try to break at sentence boundary in last 20%
        break_zone = max(start, end - chunk_bytes // 5)
        best_break = end
        for sep in [". ", ".\n", "\n\n", "\n"]:
            pos = text.rfind(sep, break_zone, end)
            if pos > break_zone:
                best_break = pos + len(sep)
                break
        passages.append(text[start:best_break])
        start = best_break - overlap_bytes
        if start >= len(text):
            break
    return passages


def build_index(
    corpus_jsonl: str,
    db_path: str,
    table_name: str = "colbert_passages",
    model_name: str = "lightonai/Reason-ModernColBERT",
    batch_size: int = 32,
    max_passage_tokens: int = 512,
    chunk_bytes: int = 4096,
    overlap_bytes: int = 800,
    resume: bool = False,
):
    print(f"Loading ColBERT model: {model_name} ...", flush=True)
    encoder = ColBERTEncoder(model_name)

    # Read corpus and split into passages
    print(f"Reading corpus and splitting into passages (chunk={chunk_bytes}, overlap={overlap_bytes}) ...", flush=True)
    passages = []  # (docid, passage_text_with_title)
    with open(corpus_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            docid = parsed["docid"]
            text = parsed["text"]
            title = extract_title(text)
            title_prefix = f"[Title: {title}]\n" if title else ""

            chunks = split_passages(text, chunk_bytes, overlap_bytes)
            for chunk in chunks:
                passages.append((str(docid), f"{title_prefix}{chunk}"))

    print(f"  {len(passages)} passages from corpus", flush=True)

    # Handle resume
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)

    existing_count = 0
    if resume:
        try:
            existing = db.open_table(table_name)
            existing_count = existing.count_rows()
            print(f"  Resuming: {existing_count} passages already indexed", flush=True)
        except Exception:
            pass

    if existing_count >= len(passages):
        print("All passages already indexed!", flush=True)
        return

    # Skip already-indexed passages
    passages = passages[existing_count:]
    print(f"  {len(passages)} passages to encode", flush=True)

    # Encode and write in batches
    t0 = time.monotonic()
    total_tokens = 0
    batch_data = []

    for batch_start in tqdm(range(0, len(passages), batch_size), desc="Encoding"):
        batch = passages[batch_start:batch_start + batch_size]
        docids = [did for did, _ in batch]
        texts = [text for _, text in batch]

        token_vecs_list = encoder.encode_docs(texts, max_length=max_passage_tokens)

        for docid, token_vecs in zip(docids, token_vecs_list):
            total_tokens += len(token_vecs)
            batch_data.append({
                "docid": docid,
                "vector": token_vecs,
            })

        if len(batch_data) >= 1000:
            _write_batch(db, table_name, batch_data)
            batch_data = []

    if batch_data:
        _write_batch(db, table_name, batch_data)

    elapsed = time.monotonic() - t0
    avg_tokens = total_tokens / len(passages) if passages else 0
    print(f"\nEncoding complete: {len(passages)} passages, {total_tokens} tokens "
          f"(avg {avg_tokens:.0f}/passage), {elapsed:.0f}s", flush=True)

    # Create index
    print("Creating IVF-HNSW-SQ index ...", flush=True)
    table = db.open_table(table_name)
    num_rows = table.count_rows()
    import math
    est_vectors = int(num_rows * avg_tokens)
    num_partitions = min(max(int(math.sqrt(est_vectors)), 256), 8192)
    print(f"  {num_rows} rows, ~{est_vectors} token vectors, {num_partitions} partitions", flush=True)

    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
        vector_column_name="vector",
        index_type="IVF_HNSW_SQ",
        m=20,
        ef_construction=300,
        replace=True,
    )
    print("Index created!", flush=True)


def _write_batch(db, table_name: str, batch_data: list[dict]):
    docids = [d["docid"] for d in batch_data]
    vectors = [d["vector"] for d in batch_data]

    docid_array = pa.array(docids, type=pa.string())
    vector_array = _build_multivec_array(vectors)

    table_data = pa.table({"docid": docid_array, "vector": vector_array})

    try:
        existing = db.open_table(table_name)
        existing.add(table_data)
    except Exception:
        db.create_table(table_name, table_data)

    print(f"  Wrote {len(batch_data)} passages ({sum(len(v) for v in vectors)} token vectors)", flush=True)


def _build_multivec_array(vectors: list[list[list[float]]]) -> pa.Array:
    dim = 128
    all_floats = []
    offsets = [0]
    for doc_vecs in vectors:
        for token_vec in doc_vecs:
            assert len(token_vec) == dim
            all_floats.extend(token_vec)
        offsets.append(offsets[-1] + len(doc_vecs))

    flat_values = pa.array(all_floats, type=pa.float32())
    inner_array = pa.FixedSizeListArray.from_arrays(flat_values, dim)
    offset_array = pa.array(offsets, type=pa.int32())
    return pa.ListArray.from_arrays(offset_array, inner_array)


def main():
    parser = argparse.ArgumentParser(description="Build ColBERT passage-level LanceDB index")
    parser.add_argument("--corpus-jsonl", required=True)
    parser.add_argument("--db-path", default="data/colbert-passages-lance")
    parser.add_argument("--table-name", default="colbert_passages")
    parser.add_argument("--model", default="lightonai/Reason-ModernColBERT")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-passage-tokens", type=int, default=512)
    parser.add_argument("--chunk-bytes", type=int, default=4096)
    parser.add_argument("--overlap-bytes", type=int, default=800)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    build_index(
        corpus_jsonl=args.corpus_jsonl,
        db_path=args.db_path,
        table_name=args.table_name,
        model_name=args.model,
        batch_size=args.batch_size,
        max_passage_tokens=args.max_passage_tokens,
        chunk_bytes=args.chunk_bytes,
        overlap_bytes=args.overlap_bytes,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
