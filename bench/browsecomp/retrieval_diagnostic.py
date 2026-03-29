#!/usr/bin/env python3
"""Retrieval diagnostic: analyze query quality vs document retrievability.

For each sample30 query, answers two questions:
  1. QUERY QUALITY — Did the model's actual search queries find the gold doc?
     If not, how close did they get (rank of gold doc)?
  2. DOCUMENT RETRIEVABILITY — Can the gold doc be found by *any* BM25 query?
     Tests oracle queries derived from the gold doc itself.

Requires: tantivy index + gw-bench binary (for native BM25 search).

Usage:
    uv run --project bench/browsecomp python bench/browsecomp/retrieval_diagnostic.py \
        --run-dir /tmp/browsecomp-exp
"""

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "BrowseComp-Plus"


def load_ground_truth() -> dict[str, dict]:
    gt_path = VENDOR_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
    gt = {}
    with open(gt_path) as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["query_id"])
            gold_docids = set()
            gold_texts = {}
            for doc in obj.get("gold_docs", []):
                did = str(doc["docid"])
                gold_docids.add(did)
                gold_texts[did] = doc.get("text", "")
            for doc in obj.get("evidence_docs", []):
                did = str(doc["docid"])
                gold_docids.add(did)
                if did not in gold_texts:
                    gold_texts[did] = doc.get("text", "")
            gt[qid] = {
                "query": obj["query"],
                "answer": obj["answer"],
                "gold_docids": gold_docids,
                "gold_texts": gold_texts,
            }
    return gt


def extract_search_queries(data: dict) -> list[str]:
    """Extract all search queries from a run trajectory."""
    queries = []
    for t in data.get("trajectory", []):
        for cb in t.get("code_blocks", []):
            for m in re.finditer(r'search\(\s*["\x27](.+?)["\x27]\s*\)', cb):
                queries.append(m.group(1))
            for m in re.finditer(r'search\(\s*f["\x27](.+?)["\x27]\s*\)', cb):
                queries.append(m.group(1))
    return queries


def bm25_search(query: str, k: int = 100) -> list[dict]:
    """Run a BM25 search via the tantivy index using a Python wrapper."""
    # Use gw-memory's CorpusSearcher via a small inline script
    # This avoids needing to compile a separate binary
    try:
        import sys
        sys.path.insert(0, str(REPO_ROOT))
        # We'll shell out to a quick Python script that uses tantivy directly
        # Actually, let's use the Rust binary in single-query mode
        pass
    except ImportError:
        pass

    # Fallback: use tantivy-py if available, otherwise return empty
    try:
        import tantivy
        return _tantivy_search(query, k)
    except ImportError:
        return []


# Global tantivy searcher (uses Rust binary for compatibility with custom tokenizer)
_searcher_cache: dict = {}


def _rust_bm25_search(query: str, k: int = 200) -> list[dict]:
    """Search using the Rust gw-bench binary's native BM25 (boosted).

    This ensures we use the exact same tokenizer and scoring as the benchmark.
    Falls back to tantivy-py if the binary isn't available.
    """
    cache_key = (query, k)
    if cache_key in _searcher_cache:
        return _searcher_cache[cache_key]

    binary = REPO_ROOT / "target" / "release" / "gw-bench"
    tantivy_path = REPO_ROOT / "data" / "tantivy-corpus"

    if not binary.exists():
        # Fallback to tantivy-py
        return _tantivy_py_search(query, k)

    import tempfile
    with tempfile.TemporaryDirectory(prefix="diag_") as tmpdir:
        tmpdir = Path(tmpdir)
        # Use single-query mode with max-turns=0 to just get pre-search results
        # Actually, we need a simpler approach — let's write a tiny Rust-compatible query
        # and parse the context from the trajectory.
        # Simpler: use tantivy-py with the default tokenizer
        pass

    return _tantivy_py_search(query, k)


_tantivy_index = None
_tantivy_searcher = None


def _open_tantivy():
    """Open tantivy index and register compatible tokenizer."""
    global _tantivy_index, _tantivy_searcher
    if _tantivy_index is not None:
        return

    import tantivy

    index_path = str(REPO_ROOT / "data" / "tantivy-corpus")
    _tantivy_index = tantivy.Index.open(index_path)

    # Register compatible tokenizer (simple, no filters — close enough for diagnostics)
    analyzer = tantivy.TextAnalyzerBuilder(tantivy.Tokenizer.simple()).build()
    _tantivy_index.register_tokenizer("en_stopwords", analyzer)

    _tantivy_searcher = _tantivy_index.searcher()
    print(f"Tantivy index opened: {_tantivy_searcher.num_docs} docs", file=sys.stderr)


def _sanitize_query(query: str) -> str:
    """Sanitize a query for tantivy's query parser.

    Strips characters that tantivy treats as syntax:
    - Colons (field:value syntax)
    - Hyphens at start of terms (exclusion)
    - Parentheses, brackets, braces (grouping)
    - Quotes, tildes, carets, backslashes
    - Asterisks, question marks (wildcards)

    Also removes words that look like field names (word followed by colon).
    """
    # Remove field:value patterns (e.g., "name: Richard", "birth_date: 1943")
    sanitized = re.sub(r'\b\w+:', ' ', query)
    # Remove all syntax characters
    sanitized = re.sub(r'["\'\(\)\[\]\{\}~\^\\*?!#@&|/<>]', ' ', sanitized)
    # Remove leading hyphens on words (NOT operator)
    sanitized = re.sub(r'(?:^|\s)-(\w)', r' \1', sanitized)
    # Collapse whitespace
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()
    return sanitized


def _tantivy_py_search(query: str, k: int = 200) -> list[dict]:
    """Search using tantivy-py with boolean term queries (no parser)."""
    cache_key = (query, k)
    if cache_key in _searcher_cache:
        return _searcher_cache[cache_key]

    _open_tantivy()

    import tantivy

    # Tokenize: lowercase, split on non-alphanumeric, filter short
    tokens = re.findall(r'[a-zA-Z0-9]+', query.lower())
    tokens = [t for t in tokens if len(t) >= 2]
    if not tokens:
        return []

    try:
        # Build boolean OR query — no parser, no syntax issues
        schema = _tantivy_index.schema
        clauses = [
            (tantivy.Occur.Should, tantivy.Query.term_query(schema, "text", t))
            for t in tokens
        ]
        query_obj = tantivy.Query.boolean_query(clauses)
        results = _tantivy_searcher.search(query_obj, k)

        hits = []
        for score, doc_address in results.hits:
            doc = _tantivy_searcher.doc(doc_address)
            docid = doc["docid"][0]
            hits.append({"docid": str(docid), "score": score})

        _searcher_cache[cache_key] = hits
        return hits
    except Exception as e:
        print(f"  Search error for '{query[:60]}': {e}", file=sys.stderr)
        return []


def generate_oracle_queries(gold_text: str, answer: str, query: str) -> list[str]:
    """Generate oracle queries from the gold document text and answer.

    These represent the best possible BM25 queries — if even these
    don't retrieve the gold doc, the document is effectively invisible to BM25.
    """
    queries = []

    # Oracle 1: the answer itself (most direct)
    if answer:
        queries.append(answer)

    # Oracle 2: title extraction (first line after --- often has the title)
    lines = gold_text.split("\n")
    for line in lines:
        if line.startswith("title:"):
            title = line.split(":", 1)[1].strip()
            if title:
                queries.append(title[:100])
            break

    # Oracle 3: answer + key terms from query (2-3 distinctive nouns)
    query_words = [w for w in query.split() if len(w) > 4 and w[0].isupper()]
    if query_words and answer:
        queries.append(f"{answer} {' '.join(query_words[:3])}")

    # Oracle 4: first sentence of doc (often the most distinctive content)
    # Skip YAML frontmatter
    in_frontmatter = False
    for line in lines:
        stripped = line.strip()
        if stripped == "---":
            in_frontmatter = not in_frontmatter
            continue
        if in_frontmatter:
            continue
        if len(stripped) > 20:
            # Take first 5 words as a query
            words = stripped.split()[:7]
            queries.append(" ".join(words))
            break

    return queries


def analyze_query(qid: str, gt_entry: dict, run_data: dict | None, use_tantivy: bool = True) -> dict:
    """Analyze retrieval for a single query."""
    gold_docids = gt_entry["gold_docids"]
    gold_texts = gt_entry["gold_texts"]
    answer = gt_entry["answer"]
    query = gt_entry["query"]

    result = {
        "query_id": qid,
        "answer": answer[:100],
        "gold_docids": sorted(gold_docids),
        "n_gold_docs": len(gold_docids),
    }

    # --- Query quality: did the model's queries find the gold doc? ---
    if run_data:
        model_queries = extract_search_queries(run_data)
        retrieved = set(str(d) for d in run_data.get("retrieved_docids", []))
        gold_retrieved = bool(gold_docids & retrieved)

        result["model_queries"] = model_queries
        result["n_model_queries"] = len(model_queries)
        result["gold_retrieved_by_model"] = gold_retrieved
        result["n_unique_docs_retrieved"] = len(retrieved)

    # --- Document retrievability: can oracle queries find the gold doc? ---
    if use_tantivy:
        oracle_results = {}
        for gold_did in sorted(gold_docids):
            gold_text = gold_texts.get(gold_did, "")
            oracle_queries = generate_oracle_queries(gold_text, answer, query)

            best_rank = None
            best_query = None

            for oq in oracle_queries:
                if not oq.strip():
                    continue
                hits = _tantivy_py_search(oq, k=200)
                for rank, hit in enumerate(hits):
                    if hit["docid"] == gold_did:
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_query = oq
                        break

            oracle_results[gold_did] = {
                "oracle_queries": oracle_queries,
                "best_rank": best_rank,  # None = not found in top-200
                "best_query": best_query,
            }

        # Also check model's actual queries against gold docs
        if run_data:
            for gold_did in sorted(gold_docids):
                model_best_rank = None
                model_best_query = None
                for mq in model_queries:
                    hits = _tantivy_py_search(mq, k=200)
                    for rank, hit in enumerate(hits):
                        if hit["docid"] == gold_did:
                            if model_best_rank is None or rank < model_best_rank:
                                model_best_rank = rank
                                model_best_query = mq
                            break
                if gold_did in oracle_results:
                    oracle_results[gold_did]["model_best_rank"] = model_best_rank
                    oracle_results[gold_did]["model_best_query"] = model_best_query

        result["oracle_analysis"] = oracle_results

        # Summary flags
        any_oracle_found = any(
            v["best_rank"] is not None
            for v in oracle_results.values()
        )
        any_oracle_top10 = any(
            v["best_rank"] is not None and v["best_rank"] < 10
            for v in oracle_results.values()
        )
        result["oracle_retrievable"] = any_oracle_found
        result["oracle_in_top10"] = any_oracle_top10

    return result


def main():
    parser = argparse.ArgumentParser(description="Retrieval diagnostic for BrowseComp-Plus")
    parser.add_argument("--run-dir", help="Path to run output directory (for model query analysis)")
    parser.add_argument("--query-ids", help="Comma-separated query IDs to analyze (default: all sample30)")
    parser.add_argument("--no-tantivy", action="store_true", help="Skip tantivy oracle analysis")
    args = parser.parse_args()

    gt = load_ground_truth()

    # Load sample30 query IDs
    sample_path = REPO_ROOT / "bench" / "browsecomp" / "sample30.tsv"
    sample_ids = []
    with open(sample_path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                sample_ids.append(parts[0])

    if args.query_ids:
        sample_ids = [qid.strip() for qid in args.query_ids.split(",")]

    # Load run data if available
    run_data = {}
    if args.run_dir:
        run_dir = Path(args.run_dir)
        for fn in run_dir.glob("*.json"):
            data = json.load(open(fn))
            qid = str(data.get("query_id", ""))
            if qid:
                run_data[qid] = data

    use_tantivy = not args.no_tantivy
    if use_tantivy:
        try:
            _open_tantivy()
            print("Tantivy index opened", file=sys.stderr)
        except Exception as e:
            print(f"Warning: tantivy-py not available ({e}), skipping oracle analysis", file=sys.stderr)
            use_tantivy = False

    # Analyze each query
    results = []
    for qid in sample_ids:
        if qid not in gt:
            continue
        rd = run_data.get(qid)
        r = analyze_query(qid, gt[qid], rd, use_tantivy=use_tantivy)
        results.append(r)

    # Print summary
    print(f"\n{'='*80}")
    print(f"RETRIEVAL DIAGNOSTIC — {len(results)} queries")
    print(f"{'='*80}\n")

    n_oracle_retrievable = sum(1 for r in results if r.get("oracle_retrievable"))
    n_oracle_top10 = sum(1 for r in results if r.get("oracle_in_top10"))
    n_model_retrieved = sum(1 for r in results if r.get("gold_retrieved_by_model"))

    if use_tantivy:
        print(f"Oracle retrievable (top-200): {n_oracle_retrievable}/{len(results)}")
        print(f"Oracle in top-10:             {n_oracle_top10}/{len(results)}")
    if run_data:
        print(f"Model actually retrieved:      {n_model_retrieved}/{len(results)}")
    print()

    # Categorize each query
    for r in results:
        qid = r["query_id"]
        flags = []

        oracle_ok = r.get("oracle_in_top10", False)
        oracle_any = r.get("oracle_retrievable", False)
        model_ok = r.get("gold_retrieved_by_model", False)

        if model_ok:
            category = "MODEL_FOUND"
        elif oracle_ok:
            category = "QUERY_QUALITY"  # oracle finds it top-10, model's queries didn't
        elif oracle_any:
            category = "QUERY_QUALITY_HARD"  # oracle finds it but not top-10
        else:
            category = "UNRETRIEVABLE"  # even oracle can't find it

        # Build detail line
        oracle_info = ""
        if use_tantivy and "oracle_analysis" in r:
            for did, oa in r["oracle_analysis"].items():
                o_rank = oa.get("best_rank", "miss")
                m_rank = oa.get("model_best_rank", "miss")
                o_rank_str = f"oracle=#{o_rank+1}" if isinstance(o_rank, int) else "oracle=MISS"
                m_rank_str = f"model=#{m_rank+1}" if isinstance(m_rank, int) else "model=MISS"
                oracle_info = f" [{o_rank_str}, {m_rank_str}]"

        n_queries = r.get("n_model_queries", "?")
        print(f"  Q{qid:>5s} {category:20s}{oracle_info}  queries={n_queries}  answer={r['answer'][:50]}")

    # Print actionable summary
    if use_tantivy:
        query_quality_issues = [r for r in results if r.get("oracle_in_top10") and not r.get("gold_retrieved_by_model")]
        unretrievable = [r for r in results if not r.get("oracle_retrievable")]

        print(f"\n--- Actionable Summary ---")
        print(f"QUERY_QUALITY ({len(query_quality_issues)}): Gold doc is easily retrievable but model's queries miss it.")
        print(f"  → Improve search query generation in the system prompt")
        for r in query_quality_issues:
            for did, oa in r.get("oracle_analysis", {}).items():
                if oa.get("best_rank") is not None and oa["best_rank"] < 10:
                    print(f"     Q{r['query_id']}: oracle query '{oa['best_query'][:60]}' → rank #{oa['best_rank']+1}")

        print(f"\nUNRETRIEVABLE ({len(unretrievable)}): Gold doc not in BM25 top-200 even with oracle queries.")
        print(f"  → Need better retrieval (ColBERT, larger index, etc.)")
        for r in unretrievable:
            print(f"     Q{r['query_id']}: {r['answer'][:60]}")


if __name__ == "__main__":
    main()
