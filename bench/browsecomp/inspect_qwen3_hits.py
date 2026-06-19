#!/usr/bin/env python3
"""Per-query inspection of qwen3 retrieval results on sample30.

For each query, reports: gold_rank (rank of first gold doc within top-200,
or 'miss'), top-3 retrieved docids, and a snippet of each. Lets us sanity
check the aggregate R@k numbers — are the finds plausible, are the misses
explainable, are we actually measuring semantic recall vs something weird.
"""

import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(Path(__file__).parent))

from searchers.qwen3_searcher import Qwen3Searcher

VENDOR_ROOT = REPO_ROOT / "vendor" / "BrowseComp-Plus"
GT_PATH = VENDOR_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
CORPUS_PATH = VENDOR_ROOT / "data" / "bm25s-index" / "corpus_meta.jsonl"
SAMPLE_PATH = REPO_ROOT / "bench" / "browsecomp" / "sample30.tsv"


def load_ground_truth():
    gt = {}
    with open(GT_PATH) as f:
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


def load_sample():
    out = []
    with open(SAMPLE_PATH) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                out.append((parts[0], parts[1]))
    return out


def load_corpus_snippets(docids: set[str]) -> dict[str, str]:
    """Load only the docs we actually need (saves memory)."""
    snippets = {}
    with open(CORPUS_PATH) as f:
        for line in f:
            obj = json.loads(line)
            docid = str(obj["docid"])
            if docid in docids:
                snippets[docid] = obj.get("text", "")[:200].replace("\n", " ")
            if len(snippets) == len(docids):
                break
    return snippets


def main():
    s = Qwen3Searcher("data/qwen3-embed", encoder_url="http://127.0.0.1:8003")
    gt = load_ground_truth()
    queries = load_sample()

    # First pass: search, collect per-query results
    per_query: list[dict] = []
    all_docids_seen: set[str] = set()

    for qid, qtext in queries:
        if qid not in gt:
            continue
        gold = gt[qid]["gold_docids"]
        hits = s.search(qtext, k=200)
        hit_docids = [h.docid for h in hits]
        all_docids_seen.update(hit_docids[:5])
        all_docids_seen.update(gold)

        gold_rank = None
        gold_docid_found = None
        for i, d in enumerate(hit_docids, start=1):
            if d in gold:
                gold_rank = i
                gold_docid_found = d
                break

        per_query.append({
            "qid": qid,
            "query": qtext,
            "answer": gt[qid]["answer"],
            "gold": gold,
            "gold_rank": gold_rank,
            "gold_docid_found": gold_docid_found,
            "top5": hit_docids[:5],
        })

    # Fetch snippets for top-5 hits + all gold docids (for misses)
    snippets = load_corpus_snippets(all_docids_seen)

    # Print summary
    found = [r for r in per_query if r["gold_rank"] is not None]
    missed = [r for r in per_query if r["gold_rank"] is None]

    print(f"\n{'=' * 80}")
    print(f"PER-QUERY RESULTS — qwen3-0.6B on sample30")
    print(f"{'=' * 80}\n")

    print(f"Gold found within top-200: {len(found)}/{len(per_query)}")
    rank_hist = {"1": 0, "2-5": 0, "6-10": 0, "11-50": 0, "51-200": 0}
    for r in found:
        rk = r["gold_rank"]
        if rk == 1: rank_hist["1"] += 1
        elif rk <= 5: rank_hist["2-5"] += 1
        elif rk <= 10: rank_hist["6-10"] += 1
        elif rk <= 50: rank_hist["11-50"] += 1
        else: rank_hist["51-200"] += 1
    print(f"Gold-rank distribution: {rank_hist}\n")

    print(f"{'qid':<6s} {'rank':>5s}  {'answer':<30s}  {'query':<40s}")
    print("-" * 100)
    for r in sorted(per_query, key=lambda x: (x["gold_rank"] or 999, x["qid"])):
        rank_str = str(r["gold_rank"]) if r["gold_rank"] else "MISS"
        ans = r["answer"][:28]
        q = r["query"][:50]
        print(f"{r['qid']:<6s} {rank_str:>5s}  {ans:<30s}  {q}")

    # Spot check a few queries
    print(f"\n{'=' * 80}")
    print(f"SPOT CHECK — top-5 retrievals for 3 sample queries")
    print(f"{'=' * 80}\n")

    # Pick: best rank, a mid-range find, and a miss
    found_sorted = sorted(found, key=lambda r: r["gold_rank"])
    spot = []
    if found_sorted:
        spot.append(("BEST RANK", found_sorted[0]))
        if len(found_sorted) > 5:
            spot.append(("MID RANK", found_sorted[len(found_sorted) // 2]))
    if missed:
        spot.append(("MISS", missed[0]))

    for label, r in spot:
        print(f"\n[{label}] q{r['qid']} — answer: {r['answer']!r}")
        print(f"  query: {r['query']}")
        print(f"  gold:  {sorted(r['gold'])[:5]}")
        print(f"  gold_rank: {r['gold_rank']}")
        print(f"  top-5 retrieved:")
        for i, docid in enumerate(r["top5"], 1):
            marker = " ← GOLD" if docid in r["gold"] else ""
            snippet = snippets.get(docid, "(no snippet)")[:120]
            print(f"    {i}. {docid:<8s} {snippet}{marker}")


if __name__ == "__main__":
    main()
