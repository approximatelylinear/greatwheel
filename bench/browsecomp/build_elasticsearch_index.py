#!/usr/bin/env python3
"""Build an Elasticsearch rank_vectors index from the existing passage blob store.

ES 8.18+ supports `rank_vectors` for ColBERT-style late interaction. We
mirror the LanceDB MV approach: one document per corpus doc, with all of
its passage tokens flattened into a single rank_vectors field. ES then
computes MaxSim natively at query time via the `maxSim` script function.

License: requires Platinum or trial. The trial gives 30 days of all
features and is started in setup with:
  curl -X POST localhost:9200/_license/start_trial?acknowledge=true

Usage:
    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/build_elasticsearch_index.py \\
        data/passage-blobs --es-url http://localhost:9200 --index colbert_mv
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import lancedb
from elasticsearch import Elasticsearch, helpers

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from blob_doc_iter import iter_docs

TOKEN_DIM = 128
DOC_BATCH = 4  # docs per ES bulk write (each doc is ~5-15 MB JSON-encoded)
MAX_TOKENS_PER_DOC = 2000  # standardized across all backends for fair comparison


def ensure_index(es: Elasticsearch, name: str) -> None:
    if es.indices.exists(index=name):
        print(f"  index {name} already exists; deleting", flush=True)
        es.indices.delete(index=name)
    body = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "refresh_interval": "30s",
        },
        "mappings": {
            "properties": {
                "docid": {"type": "keyword"},
                "n_tokens": {"type": "integer"},
                "tokens": {
                    "type": "rank_vectors",
                    "dims": TOKEN_DIM,
                    "element_type": "float",
                },
            }
        },
    }
    es.indices.create(index=name, body=body)
    print(f"  created index {name}", flush=True)


def build(blob_dir: str, es_url: str, index: str, max_docs: int | None = None) -> None:
    print(f"Connecting to ES: {es_url}", flush=True)
    es = Elasticsearch(es_url)
    print(f"  cluster: {es.info().body['cluster_name']}", flush=True)

    ensure_index(es, index)

    print(f"Opening source blob store: {blob_dir}", flush=True)
    src_db = lancedb.connect(blob_dir)
    src = src_db.open_table("passage_blobs")
    print(f"  {src.count_rows()} passages", flush=True)

    print(f"Streaming docs into ES (peak RAM = 1 doc)...", flush=True)
    t0 = time.monotonic()

    def gen_actions():
        for docid, all_tokens in iter_docs(src, MAX_TOKENS_PER_DOC, max_docs=max_docs):
            yield {
                "_index": index,
                "_id": docid,
                "_source": {
                    "docid": docid,
                    "n_tokens": len(all_tokens),
                    "tokens": all_tokens,
                },
            }

    n_ok = 0
    n_err = 0
    for ok, item in helpers.streaming_bulk(
        es,
        gen_actions(),
        chunk_size=DOC_BATCH,
        max_retries=3,
        request_timeout=120,
        raise_on_error=False,
    ):
        if ok:
            n_ok += 1
        else:
            n_err += 1
            if n_err <= 5:
                print(f"  ERR: {item}", flush=True)
        if (n_ok + n_err) % 1000 == 0:
            elapsed = time.monotonic() - t0
            rate = (n_ok + n_err) / elapsed if elapsed > 0 else 0
            print(f"  indexed {n_ok + n_err} ({rate:.0f} docs/s, {n_err} errs)", flush=True)

    elapsed = time.monotonic() - t0
    print(f"\nDone: {n_ok} indexed, {n_err} errors, {elapsed:.0f}s", flush=True)
    es.indices.refresh(index=index)
    print(f"  index doc count: {es.count(index=index).body['count']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("blob_dir", nargs="?", default="data/passage-blobs")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--index", default="colbert_mv")
    parser.add_argument("--max-docs", type=int, default=None)
    args = parser.parse_args()
    build(args.blob_dir, args.es_url, args.index, args.max_docs)


if __name__ == "__main__":
    main()
