#!/usr/bin/env python3
"""
Quick evaluation for BrowseComp-Plus runs — no vLLM/judge needed.

Uses simple string matching against ground truth for fast iteration.
Reports: accuracy (exact substring match), answer extraction rate, avg searches.

Usage:
    # Evaluate a run directory
    python quick_eval.py --run-dir runs/bm25s/ollama-qwen2.5:7b

    # Run a quick sample (N queries) and evaluate
    python quick_eval.py --sample 30 --run-cmd "python ollama_client.py --searcher-type bm25s --index-path ..."
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path

VENDOR_ROOT = Path(__file__).resolve().parent.parent.parent / "vendor" / "BrowseComp-Plus"


def load_ground_truth() -> dict[str, dict[str, str]]:
    gt_path = VENDOR_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
    gt = {}
    with open(gt_path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            gt[str(obj["query_id"])] = {
                "query": obj["query"],
                "answer": obj["answer"],
            }
    return gt


def extract_agent_answer(result_entries: list[dict]) -> str:
    """Extract the agent's final answer from result entries."""
    # Get the last output_text entry
    final_text = ""
    for entry in reversed(result_entries):
        if entry.get("type") == "output_text":
            final_text = entry.get("output", "")
            break

    if not final_text:
        return ""

    # Try to extract "Exact Answer:" line
    for line in final_text.split("\n"):
        lower = line.lower().strip()
        if lower.startswith("exact answer:"):
            return line.split(":", 1)[1].strip()
        if lower.startswith("**exact answer"):
            # Handle **Exact Answer:** format
            cleaned = re.sub(r"\*\*", "", line)
            if ":" in cleaned:
                return cleaned.split(":", 1)[1].strip()

    # Fallback: return the full final text
    return final_text


def score_answer(agent_answer: str, ground_truth: str) -> bool:
    """Check if the ground truth answer appears in the agent's response."""
    if not agent_answer or not ground_truth:
        return False

    # Normalize both
    agent_lower = agent_answer.lower().strip()
    gt_lower = ground_truth.lower().strip()

    # Exact substring match
    if gt_lower in agent_lower:
        return True

    # Try without punctuation
    gt_clean = re.sub(r"[^\w\s]", "", gt_lower)
    agent_clean = re.sub(r"[^\w\s]", "", agent_lower)
    if gt_clean and gt_clean in agent_clean:
        return True

    return False


def evaluate_run_dir(run_dir: Path, gt: dict[str, dict[str, str]]) -> dict:
    """Evaluate all JSON files in a run directory."""
    json_files = sorted(run_dir.glob("*.json"))
    if not json_files:
        return {"error": f"No JSON files in {run_dir}"}

    correct = 0
    total = 0
    answered = 0
    total_searches = 0
    total_tokens = 0
    errors = 0

    for fp in json_files:
        with open(fp) as f:
            d = json.load(f)

        qid = str(d.get("query_id", ""))
        if qid not in gt:
            continue

        total += 1

        if d.get("status") == "error":
            errors += 1
            continue

        # Count searches
        tc = d.get("tool_call_counts", {})
        total_searches += tc.get("search", 0)
        total_tokens += d.get("usage", {}).get("total_tokens", 0)

        # Extract and score answer
        agent_answer = extract_agent_answer(d.get("result", []))
        if agent_answer:
            answered += 1

        gt_answer = gt[qid]["answer"]
        if score_answer(agent_answer, gt_answer):
            correct += 1

    return {
        "total": total,
        "correct": correct,
        "accuracy": correct / total if total else 0.0,
        "answered": answered,
        "answer_rate": answered / total if total else 0.0,
        "errors": errors,
        "avg_searches": total_searches / total if total else 0.0,
        "avg_tokens": total_tokens / total if total else 0.0,
    }


def print_results(results: dict):
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"--- Quick Eval Results ---")
    print(f"accuracy:      {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    print(f"answer_rate:   {results['answer_rate']:.4f} ({results['answered']}/{results['total']})")
    print(f"avg_searches:  {results['avg_searches']:.1f}")
    print(f"avg_tokens:    {results['avg_tokens']:.0f}")
    print(f"errors:        {results['errors']}")


def get_sample_query_ids(n: int, seed: int = 42) -> list[str]:
    """Get a deterministic sample of query IDs for quick testing."""
    gt = load_ground_truth()
    import random
    rng = random.Random(seed)
    all_ids = sorted(gt.keys())
    return rng.sample(all_ids, min(n, len(all_ids)))


def write_sample_tsv(query_ids: list[str], output_path: Path):
    """Write a TSV of sampled queries for use with ollama_client.py."""
    gt = load_ground_truth()
    with open(output_path, "w", encoding="utf-8") as f:
        for qid in query_ids:
            query = gt[qid]["query"].replace("\t", " ")
            f.write(f"{qid}\t{query}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quick BrowseComp-Plus evaluation")
    parser.add_argument("--run-dir", required=True, help="Directory with run JSON files")
    parser.add_argument(
        "--write-sample-tsv",
        metavar="N",
        type=int,
        help="Write a sample TSV of N queries to stdout and exit",
    )
    args = parser.parse_args()

    if args.write_sample_tsv:
        ids = get_sample_query_ids(args.write_sample_tsv)
        gt = load_ground_truth()
        for qid in ids:
            query = gt[qid]["query"].replace("\t", " ")
            print(f"{qid}\t{query}")
        sys.exit(0)

    gt = load_ground_truth()
    results = evaluate_run_dir(Path(args.run_dir), gt)
    print_results(results)
