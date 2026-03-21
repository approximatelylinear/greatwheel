"""Build a LanceDB multi-vector index from BrowseComp-Plus corpus using ColBERT.

Encodes all documents with Reason-ModernColBERT (768→128 projection) and stores
per-token embeddings in a LanceDB table with schema:
    {docid: Utf8, vector: List(FixedSizeList(Float32, 128))}

After writing, creates an IVF-PQ index for ANN search with MaxSim scoring.

The Rust side (gw-memory) reads this table directly — no Python in the search path.

Usage:
    uv run --project bench/browsecomp --extra colbert \
        python bench/browsecomp/build_colbert_index.py \
        --corpus-jsonl vendor/BrowseComp-Plus/data/bm25s-index/corpus_meta.jsonl \
        --db-path data/colbert-lance

Estimated time: 2-4 hours on a single GPU for 100K documents.
"""

import argparse
import json
import os
import sys
import time

import pyarrow as pa
import lancedb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from colbert_encode import ColBERTEncoder


def build_index(
    corpus_jsonl: str,
    db_path: str,
    table_name: str = "colbert_docs",
    model_name: str = "lightonai/Reason-ModernColBERT",
    batch_size: int = 16,
    max_doc_tokens: int = 512,
    resume: bool = False,
):
    # Load encoder
    print(f"Loading ColBERT model: {model_name} ...", flush=True)
    encoder = ColBERTEncoder(model_name)

    # Read corpus
    print(f"Reading corpus from {corpus_jsonl} ...", flush=True)
    docs = []
    with open(corpus_jsonl) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parsed = json.loads(line)
            docs.append((parsed["docid"], parsed["text"]))
    print(f"  {len(docs)} documents loaded", flush=True)

    # Open LanceDB
    os.makedirs(db_path, exist_ok=True)
    db = lancedb.connect(db_path)

    # Check for existing table (resume support)
    existing_docids = set()
    if resume:
        try:
            existing_table = db.open_table(table_name)
            # Read all existing docids
            existing_df = existing_table.to_pandas(columns=["docid"])
            existing_docids = set(existing_df["docid"].tolist())
            print(f"  Resuming: {len(existing_docids)} documents already indexed", flush=True)
        except Exception:
            pass

    # Filter out already-indexed docs
    if existing_docids:
        docs = [(did, text) for did, text in docs if did not in existing_docids]
        print(f"  {len(docs)} documents remaining to index", flush=True)

    if not docs:
        print("All documents already indexed!", flush=True)
        return

    # Process in batches
    t0 = time.monotonic()
    total_tokens = 0
    batch_data = []

    for batch_start in tqdm(range(0, len(docs), batch_size), desc="Encoding"):
        batch_docs = docs[batch_start : batch_start + batch_size]
        texts = [text for _, text in batch_docs]
        docids = [did for did, _ in batch_docs]

        # Encode documents
        token_vecs_list = encoder.encode_docs(texts, max_length=max_doc_tokens)

        for docid, token_vecs in zip(docids, token_vecs_list):
            total_tokens += len(token_vecs)
            batch_data.append({
                "docid": docid,
                "vector": token_vecs,  # list of list[float], variable length
            })

        # Write every 500 docs to avoid memory buildup
        if len(batch_data) >= 500:
            _write_batch(db, table_name, batch_data)
            batch_data = []

    # Write remaining
    if batch_data:
        _write_batch(db, table_name, batch_data)

    elapsed = time.monotonic() - t0
    avg_tokens = total_tokens / len(docs) if docs else 0
    print(f"\nEncoding complete: {len(docs)} docs, {total_tokens} total tokens "
          f"(avg {avg_tokens:.0f}/doc), {elapsed:.0f}s", flush=True)

    # Create IVF-PQ index
    print("Creating IVF-PQ index (this may take a while) ...", flush=True)
    table = db.open_table(table_name)
    num_rows = table.count_rows()
    # num_partitions ~ sqrt(total_vectors), capped reasonably
    est_vectors = int(num_rows * avg_tokens)
    num_partitions = min(max(int(est_vectors ** 0.5), 256), 8192)
    print(f"  {num_rows} rows, ~{est_vectors} token vectors, {num_partitions} partitions", flush=True)

    table.create_index(
        metric="cosine",
        num_partitions=num_partitions,
        num_sub_vectors=8,   # 128 / 16 = 8
        num_bits=8,
        vector_column_name="vector",
        index_type="IVF_PQ",
        replace=True,
    )
    print("Index created successfully!", flush=True)


def _write_batch(db, table_name: str, batch_data: list[dict]):
    """Write a batch of documents to LanceDB."""
    # Build PyArrow arrays
    docids = [d["docid"] for d in batch_data]
    vectors = [d["vector"] for d in batch_data]

    # Build the multi-vector column: List(FixedSizeList(Float32, 128))
    # Each document has a variable number of 128-dim vectors
    docid_array = pa.array(docids, type=pa.string())
    vector_array = _build_multivec_array(vectors)

    table_data = pa.table({
        "docid": docid_array,
        "vector": vector_array,
    })

    try:
        existing = db.open_table(table_name)
        existing.add(table_data)
    except Exception:
        db.create_table(table_name, table_data)

    print(f"  Wrote {len(batch_data)} docs ({sum(len(v) for v in vectors)} token vectors)", flush=True)


def _build_multivec_array(vectors: list[list[list[float]]]) -> pa.Array:
    """Build a PyArrow List(FixedSizeList(Float32, 128)) array from nested lists."""
    dim = 128
    # Flatten all token vectors into a single float array
    all_floats = []
    offsets = [0]
    for doc_vecs in vectors:
        for token_vec in doc_vecs:
            assert len(token_vec) == dim, f"Expected {dim}-dim, got {len(token_vec)}"
            all_floats.extend(token_vec)
        offsets.append(offsets[-1] + len(doc_vecs))

    # Inner: FixedSizeList(Float32, 128)
    flat_values = pa.array(all_floats, type=pa.float32())
    inner_type = pa.list_(pa.float32(), dim)
    inner_array = pa.FixedSizeListArray.from_arrays(flat_values, dim)

    # Outer: List of FixedSizeList
    offset_array = pa.array(offsets, type=pa.int32())
    outer_array = pa.ListArray.from_arrays(offset_array, inner_array)

    return outer_array


def main():
    parser = argparse.ArgumentParser(description="Build ColBERT LanceDB multi-vector index")
    parser.add_argument("--corpus-jsonl", required=True,
                        help="Path to corpus_meta.jsonl")
    parser.add_argument("--db-path", default="data/colbert-lance",
                        help="LanceDB database path")
    parser.add_argument("--table-name", default="colbert_docs",
                        help="LanceDB table name")
    parser.add_argument("--model", default="lightonai/Reason-ModernColBERT",
                        help="ColBERT model name")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Encoding batch size")
    parser.add_argument("--max-doc-tokens", type=int, default=512,
                        help="Max tokens per document")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing table")
    args = parser.parse_args()

    build_index(
        corpus_jsonl=args.corpus_jsonl,
        db_path=args.db_path,
        table_name=args.table_name,
        model_name=args.model,
        batch_size=args.batch_size,
        max_doc_tokens=args.max_doc_tokens,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
