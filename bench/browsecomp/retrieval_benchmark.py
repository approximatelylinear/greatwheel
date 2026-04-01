#!/usr/bin/env python3
"""Retrieval-only benchmark for BrowseComp-Plus.

Measures retrieval recall@k against gold docids WITHOUT running the agent loop.
This isolates retrieval quality from LLM reasoning, allowing fast iteration
on search configuration changes.

Tests multiple retrieval strategies:
  1. Doc-level BM25 (tantivy)
  2. Passage-level BM25 (tantivy-passages)
  3. Doc+passage RRF fusion
  4. Oracle queries (upper bound)

Usage:
    uv run --project bench/browsecomp python bench/browsecomp/retrieval_benchmark.py
    uv run --project bench/browsecomp python bench/browsecomp/retrieval_benchmark.py --passage-index vendor/BrowseComp-Plus/data/tantivy-passages
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "BrowseComp-Plus"

# Import sanitization from retrieval_diagnostic
sys.path.insert(0, str(Path(__file__).parent))
from retrieval_diagnostic import _sanitize_query


def load_ground_truth() -> dict[str, dict]:
    gt_path = VENDOR_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
    gt = {}
    with open(gt_path) as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["query_id"])
            gold_docids = set()
            for doc in obj.get("gold_docs", []):
                gold_docids.add(str(doc["docid"]))
            for doc in obj.get("evidence_docs", []):
                gold_docids.add(str(doc["docid"]))
            gt[qid] = {
                "query": obj["query"],
                "answer": obj["answer"],
                "gold_docids": gold_docids,
            }
    return gt


def load_sample_queries(sample_path: str | None = None) -> list[tuple[str, str]]:
    path = sample_path or str(REPO_ROOT / "bench" / "browsecomp" / "sample30.tsv")
    queries = []
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                queries.append((parts[0], parts[1]))
    return queries


class TantivySearcher:
    """Wrapper around tantivy-py for retrieval benchmarking.

    Uses boolean_query with individual term queries instead of the query
    parser. This avoids all syntax issues (colons, hyphens, parentheses)
    and gives proper BM25 scoring.
    """

    def __init__(self, index_path: str, name: str = "doc"):
        import tantivy
        self.tantivy = tantivy
        self.index = tantivy.Index.open(index_path)
        # Register tokenizer so the index can be read
        analyzer = tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple()).build()
        self.index.register_tokenizer("en_stopwords", analyzer)
        self.searcher = self.index.searcher()
        self.schema = self.index.schema
        self.name = name
        print(f"  {name}: {self.searcher.num_docs} docs", file=sys.stderr)

    def _tokenize(self, text: str) -> list[str]:
        """Simple tokenization: lowercase, split on non-alphanumeric, filter short."""
        tokens = re.findall(r'[a-zA-Z0-9]+', text.lower())
        return [t for t in tokens if len(t) >= 2]

    def search(self, query: str, k: int = 200) -> list[dict]:
        tokens = self._tokenize(query)
        if not tokens:
            return []

        # Build boolean OR query from individual terms — no parser needed
        tantivy = self.tantivy
        clauses = []
        for token in tokens:
            term_q = tantivy.Query.term_query(self.schema, "text", token)
            clauses.append((tantivy.Occur.Should, term_q))

        if not clauses:
            return []

        query_obj = tantivy.Query.boolean_query(clauses)
        results = self.searcher.search(query_obj, k)

        hits = []
        for score, addr in results.hits:
            doc = self.searcher.doc(addr)
            docid = doc["docid"][0]
            hits.append({"docid": str(docid), "score": score, "rank": len(hits)})
        return hits


def rrf_fusion(result_lists: list[list[dict]], rrf_k: int = 60) -> list[dict]:
    """Reciprocal Rank Fusion across multiple result lists."""
    scores: dict[str, float] = {}
    for results in result_lists:
        for rank, hit in enumerate(results):
            docid = hit["docid"]
            scores[docid] = scores.get(docid, 0.0) + 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores.items(), key=lambda x: -x[1])
    return [{"docid": docid, "score": score, "rank": i} for i, (docid, score) in enumerate(ranked)]


def measure_recall(hits: list[dict], gold_docids: set[str], k_values: list[int]) -> dict[int, bool]:
    """Measure recall@k for multiple k values."""
    retrieved_at_k = {}
    for k in k_values:
        top_k_docids = set(h["docid"] for h in hits[:k])
        retrieved_at_k[k] = bool(gold_docids & top_k_docids)
    return retrieved_at_k


def gold_doc_rank(hits: list[dict], gold_docids: set[str]) -> int | None:
    """Find the rank of the first gold document in the result list."""
    for i, hit in enumerate(hits):
        if hit["docid"] in gold_docids:
            return i
    return None


class ColBERTSearcher:
    """ColBERT multi-vector search via LanceDB."""

    def __init__(self, lance_path: str, encode_url: str | None = None):
        import lancedb
        self.db = lancedb.connect(lance_path)
        # Try passage table first, fall back to doc table
        try:
            self.table = self.db.open_table("colbert_passages")
        except ValueError:
            self.table = self.db.open_table("colbert_docs")
        self.encode_url = encode_url

        # Load encoder locally if no server URL
        self._encoder = None
        if not encode_url:
            from colbert_encode import ColBERTEncoder
            self._encoder = ColBERTEncoder("lightonai/Reason-ModernColBERT")

        print(f"  colbert: {self.table.count_rows()} rows (LanceDB)", file=sys.stderr)

    def _encode_query(self, query: str) -> list[list[float]]:
        if self._encoder:
            return self._encoder.encode_query(query)
        else:
            import requests
            resp = requests.post(
                f"{self.encode_url}/encode",
                json={"text": query},
                timeout=30,
            )
            return resp.json()["tokens"]

    def search(self, query: str, k: int = 200, nprobes: int = 1, refine_factor: int = 2) -> list[dict]:
        q_vecs = self._encode_query(query)
        results = (
            self.table.search(q_vecs)
            .nprobes(nprobes)
            .refine_factor(refine_factor)
            .limit(k)
            .to_pandas()
        )
        hits = []
        for _, row in results.iterrows():
            hits.append({"docid": str(row["docid"]), "score": float(-row["_distance"]), "rank": len(hits)})
        return hits


def run_benchmark(
    doc_index_path: str,
    passage_index_path: str | None,
    colbert_lance_path: str | None = None,
    colbert_encode_url: str | None = None,
    queries: list[tuple[str, str]] = [],
    gt: dict[str, dict] = {},
    k_values: list[int] = [5, 10, 20, 50, 100, 200],
    colbert_nprobes: list[int] = [1, 5, 20],
):
    print("Opening indexes...", file=sys.stderr)
    doc_searcher = TantivySearcher(doc_index_path, "doc")
    passage_searcher = TantivySearcher(passage_index_path, "passage") if passage_index_path else None
    colbert_searcher = ColBERTSearcher(colbert_lance_path, colbert_encode_url) if colbert_lance_path else None

    strategies = ["doc_bm25"]
    if passage_searcher:
        strategies.extend(["passage_bm25", "doc_passage_rrf"])
    if colbert_searcher:
        for np in colbert_nprobes:
            strategies.append(f"colbert_np{np}")
        if passage_searcher:
            strategies.append("all_3_rrf")

    # Results: {strategy: {k: count_recalled}}
    recall_counts = {s: {k: 0 for k in k_values} for s in strategies}
    per_query = []
    total = 0

    for qid, query_text in queries:
        if qid not in gt:
            continue
        total += 1
        gold = gt[qid]["gold_docids"]

        # Strategy 1: Doc BM25
        doc_hits = doc_searcher.search(query_text, max(k_values))
        doc_recall = measure_recall(doc_hits, gold, k_values)
        doc_rank = gold_doc_rank(doc_hits, gold)

        query_result = {
            "query_id": qid,
            "answer": gt[qid]["answer"][:60],
            "n_gold_docs": len(gold),
            "doc_bm25_rank": doc_rank,
        }

        for k in k_values:
            if doc_recall[k]:
                recall_counts["doc_bm25"][k] += 1

        # Strategy 2: Passage BM25
        passage_hits = []
        if passage_searcher:
            passage_hits = passage_searcher.search(query_text, max(k_values))
            passage_recall = measure_recall(passage_hits, gold, k_values)
            passage_rank = gold_doc_rank(passage_hits, gold)
            query_result["passage_bm25_rank"] = passage_rank

            for k in k_values:
                if passage_recall[k]:
                    recall_counts["passage_bm25"][k] += 1

            # Strategy 3: Doc+Passage RRF
            rrf_hits = rrf_fusion([doc_hits, passage_hits])
            rrf_recall = measure_recall(rrf_hits, gold, k_values)
            rrf_rank = gold_doc_rank(rrf_hits, gold)
            query_result["rrf_rank"] = rrf_rank

            for k in k_values:
                if rrf_recall[k]:
                    recall_counts["doc_passage_rrf"][k] += 1

        # ColBERT strategies
        if colbert_searcher:
            for np in colbert_nprobes:
                colbert_hits = colbert_searcher.search(query_text, max(k_values), nprobes=np)
                colbert_recall = measure_recall(colbert_hits, gold, k_values)
                colbert_rank = gold_doc_rank(colbert_hits, gold)
                query_result[f"colbert_np{np}_rank"] = colbert_rank

                for k in k_values:
                    if colbert_recall[k]:
                        recall_counts[f"colbert_np{np}"][k] += 1

            # 3-channel RRF: doc + passage + ColBERT(nprobes=1)
            if passage_searcher:
                colbert_hits_1 = colbert_searcher.search(query_text, max(k_values), nprobes=1)
                all3_hits = rrf_fusion([doc_hits, passage_hits, colbert_hits_1])
                all3_recall = measure_recall(all3_hits, gold, k_values)
                all3_rank = gold_doc_rank(all3_hits, gold)
                query_result["all_3_rrf_rank"] = all3_rank

                for k in k_values:
                    if all3_recall[k]:
                        recall_counts["all_3_rrf"][k] += 1

        per_query.append(query_result)

    return {
        "total": total,
        "k_values": k_values,
        "strategies": strategies,
        "recall_counts": recall_counts,
        "per_query": per_query,
    }


def print_results(results: dict):
    total = results["total"]
    k_values = results["k_values"]
    strategies = results["strategies"]
    recall = results["recall_counts"]

    print(f"\n{'=' * 80}")
    print(f"RETRIEVAL BENCHMARK — {total} queries")
    print(f"{'=' * 80}\n")

    # Table header
    header = f"{'Strategy':<20s}"
    for k in k_values:
        header += f" {'R@'+str(k):>7s}"
    print(header)
    print("-" * len(header))

    for strategy in strategies:
        row = f"{strategy:<20s}"
        for k in k_values:
            count = recall[strategy][k]
            pct = count / total * 100
            row += f" {count:>3d}/{total:<3d}"
        print(row)

    # Per-query detail
    print(f"\nPer-query gold doc rank (lower = better, None = not found):")
    for q in results["per_query"]:
        doc_r = q.get("doc_bm25_rank")
        pass_r = q.get("passage_bm25_rank")
        rrf_r = q.get("rrf_rank")

        doc_str = f"#{doc_r+1}" if doc_r is not None else "miss"
        pass_str = f"#{pass_r+1}" if pass_r is not None else "miss"
        rrf_str = f"#{rrf_r+1}" if rrf_r is not None else "miss"

        best = min(
            r for r in [doc_r, pass_r, rrf_r] if r is not None
        ) if any(r is not None for r in [doc_r, pass_r, rrf_r]) else None

        mark = "📄" if best is not None and best < 10 else ("  " if best is not None else "✗ ")

        line = f"  {mark} Q{q['query_id']:>5s}  doc={doc_str:>6s}"
        if pass_r is not None or "passage_bm25_rank" in q:
            line += f"  pass={pass_str:>6s}  rrf={rrf_str:>6s}"
        line += f"  {q['answer']}"
        print(line)


def main():
    parser = argparse.ArgumentParser(description="Retrieval-only benchmark")
    parser.add_argument("--doc-index", default=str(REPO_ROOT / "vendor" / "BrowseComp-Plus" / "data" / "tantivy-corpus"))
    parser.add_argument("--passage-index", default=None)
    parser.add_argument("--colbert-lance", default=None, help="Path to ColBERT LanceDB index")
    parser.add_argument("--colbert-encode-url", default=None, help="ColBERT encode server URL")
    parser.add_argument("--query-file", default=None, help="TSV query file (default: sample30.tsv)")
    parser.add_argument("--json", action="store_true", help="Output JSON instead of table")
    args = parser.parse_args()

    gt = load_ground_truth()
    queries = load_sample_queries(args.query_file)

    results = run_benchmark(
        doc_index_path=args.doc_index,
        passage_index_path=args.passage_index,
        colbert_lance_path=args.colbert_lance,
        colbert_encode_url=args.colbert_encode_url,
        queries=queries,
        gt=gt,
    )

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results)


if __name__ == "__main__":
    main()
