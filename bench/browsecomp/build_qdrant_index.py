#!/usr/bin/env python3
"""Build a Qdrant native multi-vector collection from the passage blob store.

Qdrant 1.10+ supports `MultiVectorConfig` with `Comparator.MAX_SIM`, which
gives ColBERT MaxSim natively at query time. We mirror the LanceDB MV /
Elasticsearch shape: one Qdrant point per corpus doc, with all of its
passage tokens flattened into a single multi-vector field. Per-doc max
across passages happens automatically because max-pool is associative.

Usage:
    docker compose -f docker/docker-compose.bench.yml up -d qdrant
    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/build_qdrant_index.py \\
        data/passage-blobs --qdrant-url http://localhost:6333 --collection colbert_mv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import lancedb
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blob_doc_iter import iter_docs

TOKEN_DIM = 128
DOC_BATCH = 8  # docs per Qdrant upsert (multi-vector upserts are slow)
MAX_TOKENS_PER_DOC = 2000  # Qdrant has a 1 MB / 2048-vector hard limit per point


def ensure_collection(client: QdrantClient, name: str) -> None:
    if client.collection_exists(name):
        print(f"  collection {name} exists; deleting", flush=True)
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config=qm.VectorParams(
            size=TOKEN_DIM,
            distance=qm.Distance.COSINE,
            multivector_config=qm.MultiVectorConfig(
                comparator=qm.MultiVectorComparator.MAX_SIM,
            ),
        ),
    )
    print(f"  created collection {name}", flush=True)


def build(blob_dir: str, qdrant_url: str, collection: str, max_docs: int | None = None) -> None:
    print(f"Connecting to Qdrant: {qdrant_url}", flush=True)
    # Generous timeout: occasional upserts stall when Qdrant is doing background
    # HNSW maintenance, especially after the first few thousand multi-vectors.
    client = QdrantClient(url=qdrant_url, timeout=600)
    print(f"  version: {client.info().version if hasattr(client, 'info') else 'unknown'}", flush=True)

    ensure_collection(client, collection)

    print(f"Opening source blob store: {blob_dir}", flush=True)
    src_db = lancedb.connect(blob_dir)
    src = src_db.open_table("passage_blobs")
    print(f"  {src.count_rows()} passages", flush=True)

    print(f"Streaming docs into Qdrant (peak RAM = 1 doc)...", flush=True)
    t0 = time.monotonic()
    n_done = 0
    points: list[qm.PointStruct] = []

    def flush():
        nonlocal points, n_done
        if not points:
            return
        # Use the raw HTTP API directly so we can see exactly what Qdrant
        # complains about — qdrant_client.upsert masks 400 errors as
        # generic "ResponseHandlingException: timed out" which is unhelpful.
        import json as _json
        import urllib.request, urllib.error
        body = _json.dumps({
            "points": [
                {"id": p.id, "vector": p.vector, "payload": p.payload}
                for p in points
            ]
        }).encode()
        req = urllib.request.Request(
            f"{qdrant_url}/collections/{collection}/points?wait=false",
            data=body,
            method="PUT",
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=600) as resp:
                _ = resp.read()
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            print(f"  Qdrant HTTP {e.code} at n_done={n_done}: {err_body[:500]}", flush=True)
            print(f"  failing batch ids: {[p.id for p in points]}", flush=True)
            print(f"  failing batch vector lens: {[len(p.vector) for p in points]}", flush=True)
            raise
        except urllib.error.URLError as e:
            print(f"  Qdrant URLError at n_done={n_done}: {e}", flush=True)
            print(f"  failing batch ids: {[p.id for p in points]}", flush=True)
            print(f"  failing batch vector lens: {[len(p.vector) for p in points]}", flush=True)
            print(f"  failing batch payload size: {len(body)} bytes", flush=True)
            # Also try a single-point upsert to find the exact bad point
            for p in points:
                single_body = _json.dumps({"points": [{"id": p.id, "vector": p.vector, "payload": p.payload}]}).encode()
                req2 = urllib.request.Request(
                    f"{qdrant_url}/collections/{collection}/points?wait=false",
                    data=single_body, method="PUT",
                    headers={"Content-Type": "application/json"},
                )
                try:
                    with urllib.request.urlopen(req2, timeout=120) as r2:
                        r2.read()
                    print(f"    point id={p.id} len={len(p.vector)} OK", flush=True)
                except urllib.error.HTTPError as he:
                    print(f"    point id={p.id} len={len(p.vector)} HTTP {he.code}: {he.read().decode('utf-8', errors='replace')[:300]}", flush=True)
                except urllib.error.URLError as ue:
                    print(f"    point id={p.id} len={len(p.vector)} URLError: {ue}", flush=True)
            raise
        n_done += len(points)
        points = []
        elapsed = time.monotonic() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f"  upserted {n_done} ({rate:.0f} docs/s)", flush=True)
        n_done += len(points)
        points = []
        elapsed = time.monotonic() - t0
        rate = n_done / elapsed if elapsed > 0 else 0
        print(f"  upserted {n_done} ({rate:.0f} docs/s)", flush=True)

    for docid, all_tokens in iter_docs(src, MAX_TOKENS_PER_DOC, max_docs=max_docs):
        try:
            point_id = int(docid)
        except ValueError:
            import hashlib
            point_id = int(hashlib.md5(docid.encode()).hexdigest()[:16], 16)
        points.append(qm.PointStruct(
            id=point_id,
            vector=all_tokens,
            payload={"docid": docid, "n_tokens": len(all_tokens)},
        ))
        if len(points) >= DOC_BATCH:
            flush()

    flush()
    elapsed = time.monotonic() - t0
    print(f"\nDone: upserted {n_done} docs in {elapsed:.0f}s", flush=True)
    print(f"  collection point count: {client.count(collection).count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blob_dir", nargs="?", default="data/passage-blobs")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--collection", default="colbert_mv")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()
    build(args.blob_dir, args.qdrant_url, args.collection, args.max_docs)


if __name__ == "__main__":
    main()
