#!/usr/bin/env python3
"""Multi-backend retrieval benchmark for BrowseComp.

Runs any subset of `Searcher` backends against a fixed query set,
reports R@k and per-query latency in a uniform table. Backends share a
single ColBERT encoder via the HTTP service in `colbert_server.py`.

Usage
-----

Start the encoder service first (one process, hot model):

    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/colbert_server.py --port 8002

Run all backends:

    uv run --project bench/browsecomp --extra colbert \\
        python bench/browsecomp/retrieval_benchmark_v2.py \\
        --searchers brute_force lancedb_mv elasticsearch qdrant

Run only a few:

    python bench/browsecomp/retrieval_benchmark_v2.py --searchers lancedb_mv qdrant

Add the BlobRerankWrapper to any backend by name:

    python bench/browsecomp/retrieval_benchmark_v2.py --searchers tantivy+rerank
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "BrowseComp-Plus"

sys.path.insert(0, str(Path(__file__).parent))
from searchers.base import EncoderClient, ScoredDoc

# k values to report
DEFAULT_K_VALUES = [5, 10, 20, 50, 100, 200]


# ---------------------------------------------------------------------------
# Backend factory
# ---------------------------------------------------------------------------


def make_searcher(name: str, encoder: EncoderClient, args: argparse.Namespace):
    """Build a Searcher by name. Suffix `+rerank` wraps it with BlobRerankWrapper."""
    add_rerank = name.endswith("+rerank")
    base_name = name[: -len("+rerank")] if add_rerank else name

    if base_name == "brute_force":
        from searchers.brute_force_searcher import BruteForceSearcher
        s = BruteForceSearcher(args.blob_store, encoder=encoder, max_passages=args.brute_force_passages)
    elif base_name == "lancedb_mv":
        from searchers.lancedb_mv_searcher import LanceDbMvSearcher
        s = LanceDbMvSearcher(args.lancedb_mv_dir, encoder=encoder)
    elif base_name == "elasticsearch":
        from searchers.elasticsearch_searcher import ElasticsearchSearcher
        s = ElasticsearchSearcher(encoder, es_url=args.es_url, index=args.es_index)
    elif base_name == "qdrant":
        from searchers.qdrant_searcher import QdrantSearcher
        s = QdrantSearcher(encoder, qdrant_url=args.qdrant_url, collection=args.qdrant_collection)
    elif base_name == "tantivy":
        # Lightweight wrapper around the existing TantivySearcher in retrieval_benchmark.py
        from retrieval_benchmark import TantivySearcher
        ts = TantivySearcher(args.passage_index, "passage")

        class TantivyAdapter:
            name = "tantivy"
            def search(self, query: str, k: int) -> list[ScoredDoc]:
                hits = ts.search(query, k)
                return [ScoredDoc(docid=h["docid"], score=h["score"]) for h in hits]
        s = TantivyAdapter()
    else:
        raise ValueError(f"unknown searcher: {base_name}")

    if add_rerank:
        from searchers.blob_rerank_wrapper import BlobRerankWrapper
        s = BlobRerankWrapper(s, encoder=encoder, blob_store_dir=args.blob_store, first_stage_k=args.first_stage_k)
    return s


# ---------------------------------------------------------------------------
# Ground truth + queries (mirrors retrieval_benchmark.py)
# ---------------------------------------------------------------------------


def load_ground_truth() -> dict[str, dict]:
    gt_path = VENDOR_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
    gt = {}
    with open(gt_path) as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["query_id"])
            gold = set()
            for doc in obj.get("gold_docs", []):
                gold.add(str(doc["docid"]))
            for doc in obj.get("evidence_docs", []):
                gold.add(str(doc["docid"]))
            gt[qid] = {"query": obj["query"], "answer": obj["answer"], "gold_docids": gold}
    return gt


def load_sample_queries(path: str | None) -> list[tuple[str, str]]:
    p = Path(path) if path else REPO_ROOT / "bench" / "browsecomp" / "sample30.tsv"
    queries = []
    with open(p) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries.append((parts[0], parts[1]))
    return queries


def measure_recall(hits: list[ScoredDoc], gold: set[str], k_values: list[int]) -> dict[int, bool]:
    out = {}
    for k in k_values:
        topk = {h.docid for h in hits[:k]}
        out[k] = bool(gold & topk)
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(args):
    print(f"Connecting to encoder service: {args.encoder_url}", flush=True)
    encoder = EncoderClient(args.encoder_url)
    health = encoder.health()
    print(f"  encoder ready: {health.get('model')} on {health.get('device')}", flush=True)

    gt = load_ground_truth()
    queries = load_sample_queries(args.queries)
    print(f"Loaded {len(queries)} queries from sample set", flush=True)

    # Open all backends
    print(f"\nOpening {len(args.searchers)} backends...", flush=True)
    searchers = []
    for name in args.searchers:
        print(f"  [{name}]", flush=True)
        searchers.append((name, make_searcher(name, encoder, args)))

    # Recall + latency
    k_values = args.k_values
    recall_counts: dict[str, dict[int, int]] = {n: {k: 0 for k in k_values} for n, _ in searchers}
    latencies_ms: dict[str, list[float]] = {n: [] for n, _ in searchers}
    n_queries = 0

    for qid, query_text in queries:
        if qid not in gt:
            continue
        n_queries += 1
        gold = gt[qid]["gold_docids"]

        for name, s in searchers:
            t0 = time.monotonic()
            try:
                hits = s.search(query_text, max(k_values))
            except Exception as e:
                print(f"  ERR {name} q{qid}: {type(e).__name__}: {e}", file=sys.stderr)
                hits = []
            dt = (time.monotonic() - t0) * 1000
            latencies_ms[name].append(dt)

            r = measure_recall(hits, gold, k_values)
            for k in k_values:
                if r[k]:
                    recall_counts[name][k] += 1

    # ----- print results -----
    print(f"\n{'=' * 90}")
    print(f"BENCHMARK RESULTS — {n_queries} queries × {len(searchers)} backends")
    print(f"{'=' * 90}\n")

    header = f"{'Backend':<25s}"
    for k in k_values:
        header += f" {'R@'+str(k):>7s}"
    header += f" {'p50_ms':>9s} {'p95_ms':>9s}"
    print(header)
    print("-" * len(header))

    for name, _ in searchers:
        row = f"{name:<25s}"
        for k in k_values:
            row += f" {recall_counts[name][k]:>3d}/{n_queries:<3d}"
        lats = sorted(latencies_ms[name])
        if lats:
            p50 = lats[len(lats) // 2]
            p95 = lats[max(0, int(len(lats) * 0.95) - 1)]
            row += f" {p50:>9.0f} {p95:>9.0f}"
        print(row)

    # JSON dump for downstream analysis
    if args.json_out:
        with open(args.json_out, "w") as f:
            json.dump({
                "n_queries": n_queries,
                "k_values": k_values,
                "recall_counts": recall_counts,
                "latencies_ms": latencies_ms,
            }, f, indent=2)
        print(f"\nWrote JSON to {args.json_out}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--searchers", nargs="+", required=True,
                        help="Backends to benchmark. Suffix any with +rerank to wrap with BlobRerankWrapper.")
    parser.add_argument("--queries", default=None)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)

    parser.add_argument("--encoder-url", default="http://127.0.0.1:8002")
    parser.add_argument("--blob-store", default="data/passage-blobs")

    parser.add_argument("--lancedb-mv-dir", default="data/lancedb-mv")
    parser.add_argument("--es-url", default="http://localhost:9200")
    parser.add_argument("--es-index", default="colbert_mv")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--qdrant-collection", default="colbert_mv")
    parser.add_argument("--passage-index", default=str(VENDOR_ROOT / "data" / "tantivy-passages"))

    parser.add_argument("--brute-force-passages", type=int, default=None,
                        help="Cap brute-force scan to N passages (for fast smoke runs)")
    parser.add_argument("--first-stage-k", type=int, default=200,
                        help="Candidate count for +rerank wrapper")
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
