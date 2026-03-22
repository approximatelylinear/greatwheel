#!/usr/bin/env python3
"""Classify BrowseComp-Plus run failures into actionable categories (GEPA Phase 1).

Reads trajectory JSONs from a run directory, cross-references with ground truth,
and produces per-query ASI (Actionable Side Information) and an aggregated summary.

Failure modes:
  - correct:          Answer matches ground truth
  - retrieval_miss:   Gold document never appeared in any search result
  - extraction_error: Gold document was retrieved but wrong answer extracted
  - hedge:            Model answered with a refusal/hedge phrase
  - timeout:          Hit turn/time limit without calling FINAL()

Usage:
    python classify_failures.py --run-dir /tmp/browsecomp-exp
    python classify_failures.py --run-dir /tmp/browsecomp-exp --output asi_report.json
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor" / "BrowseComp-Plus"


def load_ground_truth() -> dict[str, dict]:
    """Load ground truth with answers and gold docids."""
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


def extract_agent_answer(data: dict) -> str:
    """Extract the agent's final answer from a run record."""
    for entry in reversed(data.get("result", [])):
        out = entry.get("output", "")
        if isinstance(out, str) and "Exact Answer:" in out:
            return out.split("Exact Answer:", 1)[1].strip()
    return ""


def score_answer(agent_answer: str, ground_truth: str) -> bool:
    """Check if ground truth appears in agent answer (substring match)."""
    if not agent_answer or not ground_truth:
        return False
    a = agent_answer.lower().strip()
    g = ground_truth.lower().strip()
    if g in a:
        return True
    g_clean = re.sub(r"[^\w\s]", "", g)
    a_clean = re.sub(r"[^\w\s]", "", a)
    return g_clean in a_clean


def is_hedge(answer: str) -> bool:
    """Check if the answer is a refusal/hedge."""
    lower = answer.lower()
    if len(answer) > 150:
        return True
    hedges = [
        "unable to", "not found", "insufficient", "cannot determine",
        "could not find", "could not determine", "no exact match",
        "no specific", "no matching", "not enough information",
        "based on the research", "based on the provided", "n/a",
    ]
    return any(h in lower for h in hedges)


def extract_retrieved_docids(data: dict) -> set[str]:
    """Extract all docids the agent encountered during the run."""
    docids = set()

    for t in data.get("trajectory", []):
        # From code blocks: get_document("12345")
        for cb in t.get("code_blocks", []):
            for m in re.finditer(r'get_document\([^)]*?["\x27](\d+)["\x27]', cb):
                docids.add(m.group(1))

        # From REPL output: printed docids
        output = t.get("repl_output") or ""
        # Patterns: "0: 69893 —", "docid=69893", "'docid': '69893'"
        for m in re.finditer(r"(?:^\d+:\s+|docid[=:\s]+['\"]?)(\d{3,6})", output, re.MULTILINE):
            docids.add(m.group(1))
        for m in re.finditer(r"['\"]docid['\"]:\s*['\"](\d+)['\"]", output):
            docids.add(m.group(1))

    # Also check result entries
    for entry in data.get("result", []):
        out = str(entry.get("output", ""))
        for m in re.finditer(r"['\"]docid['\"]:\s*['\"](\d+)['\"]", out):
            docids.add(m.group(1))

    return docids


def extract_search_queries(data: dict) -> list[str]:
    """Extract all search queries issued by the agent."""
    queries = []
    for t in data.get("trajectory", []):
        for cb in t.get("code_blocks", []):
            for m in re.finditer(r'search\(\s*["\x27](.+?)["\x27]\s*\)', cb):
                queries.append(m.group(1))
            # Also f-string patterns: search(f"...")
            for m in re.finditer(r'search\(\s*f["\x27](.+?)["\x27]\s*\)', cb):
                queries.append(m.group(1))
    return queries


def classify_query(data: dict, gt_entry: dict) -> dict:
    """Classify a single query result into a failure mode with ASI."""
    qid = data.get("query_id", "?")
    answer = extract_agent_answer(data)
    gold_answer = gt_entry["answer"]
    gold_docids = gt_entry["gold_docids"]
    status = data.get("status", "?")

    correct = score_answer(answer, gold_answer)
    retrieved = extract_retrieved_docids(data)
    searches = extract_search_queries(data)
    gold_retrieved = bool(gold_docids & retrieved)

    n_turns = sum(1 for t in data.get("trajectory", []) if t.get("role") == "assistant")
    tokens = data.get("usage", {}).get("total_tokens", 0)
    n_docs_read = len([
        cb for t in data.get("trajectory", [])
        for cb in t.get("code_blocks", [])
        if "get_document(" in cb
    ])

    # Classify failure mode
    if correct:
        failure_mode = "correct"
    elif "fallback" in status or "timeout" in status:
        failure_mode = "timeout"
    elif is_hedge(answer):
        failure_mode = "hedge"
    elif gold_retrieved:
        failure_mode = "extraction_error"
    else:
        failure_mode = "retrieval_miss"

    return {
        "query_id": qid,
        "correct": correct,
        "failure_mode": failure_mode,
        "agent_answer": answer[:200],
        "gold_answer": gold_answer[:200],
        "gold_docids": sorted(gold_docids),
        "gold_docid_retrieved": gold_retrieved,
        "retrieved_docids": sorted(retrieved),
        "n_unique_docids": len(retrieved),
        "searches_issued": searches,
        "n_unique_searches": len(set(searches)),
        "n_docs_read": n_docs_read,
        "turns_used": n_turns,
        "token_cost": tokens,
        "status": status,
    }


def aggregate_asi(per_query: list[dict]) -> dict:
    """Build aggregated ASI summary from per-query results."""
    n = len(per_query)
    failure_dist = {}
    correct_ids = []
    wrong_ids = []
    hedge_ids = []
    timeout_ids = []
    retrieval_miss_ids = []
    extraction_error_ids = []
    total_tokens = 0
    total_searches = 0

    for q in per_query:
        mode = q["failure_mode"]
        failure_dist[mode] = failure_dist.get(mode, 0) + 1
        total_tokens += q["token_cost"]
        total_searches += q["n_unique_searches"]

        qid = q["query_id"]
        if mode == "correct":
            correct_ids.append(qid)
        elif mode == "hedge":
            hedge_ids.append(qid)
        elif mode == "timeout":
            timeout_ids.append(qid)
        elif mode == "retrieval_miss":
            retrieval_miss_ids.append(qid)
        elif mode == "extraction_error":
            extraction_error_ids.append(qid)
            wrong_ids.append(qid)

    return {
        "accuracy": len(correct_ids),
        "total": n,
        "accuracy_pct": round(len(correct_ids) / n, 4) if n else 0,
        "failure_distribution": failure_dist,
        "correct_ids": correct_ids,
        "retrieval_miss_ids": retrieval_miss_ids,
        "extraction_error_ids": extraction_error_ids,
        "hedge_ids": hedge_ids,
        "timeout_ids": timeout_ids,
        "avg_token_cost": round(total_tokens / n) if n else 0,
        "avg_unique_searches": round(total_searches / n, 1) if n else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify BrowseComp-Plus failures (GEPA Phase 1)")
    parser.add_argument("--run-dir", required=True, help="Path to run output directory")
    parser.add_argument("--output", help="Path to write ASI JSON (default: stdout)")
    parser.add_argument("--verbose", action="store_true", help="Print per-query details")
    args = parser.parse_args()

    gt = load_ground_truth()
    run_dir = Path(args.run_dir)

    per_query = []
    for fn in sorted(run_dir.glob("*.json")):
        data = json.load(open(fn))
        qid = str(data.get("query_id", ""))
        if qid not in gt:
            continue
        result = classify_query(data, gt[qid])
        per_query.append(result)

    summary = aggregate_asi(per_query)

    report = {
        "summary": summary,
        "per_query": per_query,
    }

    if args.verbose:
        # Print human-readable summary
        s = summary
        print(f"=== ASI Failure Classification ===")
        print(f"Accuracy: {s['accuracy']}/{s['total']} ({s['accuracy_pct']:.1%})")
        print(f"Avg tokens: {s['avg_token_cost']}, Avg searches: {s['avg_unique_searches']}")
        print(f"\nFailure distribution:")
        for mode, count in sorted(s["failure_distribution"].items(), key=lambda x: -x[1]):
            pct = count / s["total"] * 100
            ids = [q["query_id"] for q in per_query if q["failure_mode"] == mode]
            print(f"  {mode:20s} {count:3d} ({pct:5.1f}%)  {ids}")

        print(f"\nPer-query details:")
        for q in per_query:
            mark = "✓" if q["correct"] else "✗"
            gold_flag = "📄" if q["gold_docid_retrieved"] else "  "
            print(f"  {mark} Q{q['query_id']:>5s} {gold_flag} {q['failure_mode']:20s} "
                  f"turns={q['turns_used']:2d} docs={q['n_docs_read']:2d} "
                  f"searches={q['n_unique_searches']:2d} tokens={q['token_cost']:6d}")
            if not q["correct"]:
                print(f"         Gold: {q['gold_answer'][:70]}")
                print(f"         Got:  {q['agent_answer'][:70]}")
        print()

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"Written to {args.output}", file=sys.stderr)
    elif not args.verbose:
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
