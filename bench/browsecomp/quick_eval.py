#!/usr/bin/env python3
"""
Quick evaluation for BrowseComp-Plus runs.

Three scoring tiers:
  1. Exact: ground truth is a substring of the agent's answer (original)
  2. Fuzzy: normalized edit distance < 0.3, or token overlap > 0.7
  3. LLM:   Ollama judge decides if the agent's answer is equivalent

Usage:
    # Exact only (fast, default)
    python quick_eval.py --run-dir /tmp/browsecomp-exp

    # Exact + fuzzy
    python quick_eval.py --run-dir /tmp/browsecomp-exp --fuzzy

    # Exact + fuzzy + LLM judge
    python quick_eval.py --run-dir /tmp/browsecomp-exp --llm-judge

    # LLM judge with custom model
    python quick_eval.py --run-dir /tmp/browsecomp-exp --llm-judge --judge-model qwen3.5:9b
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


def normalize_answer(text: str) -> str:
    """Normalize an answer for fuzzy comparison."""
    text = text.lower().strip()
    # Remove common prefixes/suffixes
    for prefix in ["the ", "a ", "an "]:
        if text.startswith(prefix):
            text = text[len(prefix):]
    # Remove punctuation
    text = re.sub(r"[^\w\s]", "", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def edit_distance(a: str, b: str) -> int:
    """Levenshtein edit distance."""
    if len(a) < len(b):
        return edit_distance(b, a)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a):
        curr = [i + 1]
        for j, cb in enumerate(b):
            cost = 0 if ca == cb else 1
            curr.append(min(curr[j] + 1, prev[j + 1] + 1, prev[j] + cost))
        prev = curr
    return prev[-1]


def score_answer_fuzzy(agent_answer: str, ground_truth: str) -> bool:
    """Fuzzy match: normalized edit distance or token overlap."""
    if not agent_answer or not ground_truth:
        return False

    # First try exact (superset of fuzzy)
    if score_answer(agent_answer, ground_truth):
        return True

    a_norm = normalize_answer(agent_answer)
    g_norm = normalize_answer(ground_truth)

    if not a_norm or not g_norm:
        return False

    # Edit distance relative to gold length
    dist = edit_distance(a_norm, g_norm)
    max_len = max(len(a_norm), len(g_norm))
    if max_len > 0 and dist / max_len < 0.3:
        return True

    # Check if gold is a subsequence of agent (handles extra words)
    # e.g. "Richard Larson" contains all tokens of "Richard C. Larson" minus "C."
    g_tokens = set(g_norm.split())
    a_tokens = set(a_norm.split())
    if g_tokens and len(g_tokens & a_tokens) / len(g_tokens) >= 0.7:
        return True

    # Check if agent answer is contained in gold (handles truncation)
    # e.g. "Gingras Trading Post" in "The Gingras Trading Post State Historic Site"
    if len(a_norm) >= 4 and a_norm in g_norm:
        return True

    return False


def score_answer_llm(
    agent_answer: str,
    ground_truth: str,
    query: str,
    ollama_url: str = "http://localhost:11434",
    model: str = "qwen3.5:9b",
) -> bool:
    """LLM judge: ask the model if the answers are equivalent."""
    if not agent_answer or not ground_truth:
        return False

    # Skip LLM call if exact match
    if score_answer(agent_answer, ground_truth):
        return True

    import requests

    # Quick reject: obviously garbage answers
    a_norm = normalize_answer(agent_answer)
    if not a_norm or len(a_norm) < 2 or a_norm in ("answer", "unknown", "none", "null"):
        return False

    prompt = (
        f"You are a strict factual answer evaluator.\n\n"
        f"Question: {query[:500]}\n"
        f"Expected answer: {ground_truth}\n"
        f"Agent's answer: {agent_answer[:200]}\n\n"
        f"Is the agent's answer correct? It is correct ONLY if it refers to the "
        f"SAME specific entity, date, number, or fact as the expected answer. "
        f"Minor variations are OK: missing middle initials, different transliterations, "
        f"abbreviations, partial but unambiguous name matches.\n\n"
        f"It is WRONG if: it names a different entity, gives a vague/generic response, "
        f"or the answer text is not a real factual answer.\n\n"
        f"Reply with ONLY 'YES' or 'NO'.\n\n/no_think"
    )

    try:
        resp = requests.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "think": False,
            },
            timeout=30,
        )
        if resp.ok:
            content = resp.json().get("message", {}).get("content", "").strip().upper()
            return content.startswith("YES")
    except Exception:
        pass

    return False


def evaluate_run_dir(run_dir: Path, gt: dict[str, dict[str, str]], fuzzy: bool = False, llm_judge: bool = False, judge_model: str = "qwen3.5:9b", ollama_url: str = "http://localhost:11434") -> dict:
    """Evaluate all JSON files in a run directory."""
    json_files = sorted(run_dir.glob("*.json"))
    if not json_files:
        return {"error": f"No JSON files in {run_dir}"}

    exact_correct = 0
    fuzzy_correct = 0
    llm_correct = 0
    total = 0
    answered = 0
    total_searches = 0
    total_tokens = 0
    errors = 0
    per_query = []

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
        query_text = gt[qid]["query"]

        is_exact = score_answer(agent_answer, gt_answer)
        is_fuzzy = score_answer_fuzzy(agent_answer, gt_answer) if fuzzy or llm_judge else False
        is_llm = False

        if is_exact:
            exact_correct += 1
            fuzzy_correct += 1
            llm_correct += 1
            is_fuzzy = True
            is_llm = True
        elif is_fuzzy:
            fuzzy_correct += 1
            llm_correct += 1
            is_llm = True
        elif llm_judge:
            is_llm = score_answer_llm(agent_answer, gt_answer, query_text, ollama_url, judge_model)
            if is_llm:
                llm_correct += 1

        per_query.append({
            "query_id": qid,
            "agent_answer": agent_answer[:200],
            "gold_answer": gt_answer[:200],
            "exact": is_exact,
            "fuzzy": is_fuzzy,
            "llm": is_llm,
        })

    result = {
        "total": total,
        "correct": exact_correct,
        "accuracy": exact_correct / total if total else 0.0,
        "answered": answered,
        "answer_rate": answered / total if total else 0.0,
        "errors": errors,
        "avg_searches": total_searches / total if total else 0.0,
        "avg_tokens": total_tokens / total if total else 0.0,
    }

    if fuzzy or llm_judge:
        result["fuzzy_correct"] = fuzzy_correct
        result["fuzzy_accuracy"] = fuzzy_correct / total if total else 0.0
    if llm_judge:
        result["llm_correct"] = llm_correct
        result["llm_accuracy"] = llm_correct / total if total else 0.0

    result["per_query"] = per_query
    return result


def print_results(results: dict, verbose: bool = False):
    if "error" in results:
        print(f"Error: {results['error']}")
        return

    print(f"--- Quick Eval Results ---")
    print(f"exact:         {results['accuracy']:.4f} ({results['correct']}/{results['total']})")
    if "fuzzy_correct" in results:
        print(f"fuzzy:         {results['fuzzy_accuracy']:.4f} ({results['fuzzy_correct']}/{results['total']})")
    if "llm_correct" in results:
        print(f"llm_judge:     {results['llm_accuracy']:.4f} ({results['llm_correct']}/{results['total']})")
    print(f"answer_rate:   {results['answer_rate']:.4f} ({results['answered']}/{results['total']})")
    print(f"avg_searches:  {results['avg_searches']:.1f}")
    print(f"avg_tokens:    {results['avg_tokens']:.0f}")
    print(f"errors:        {results['errors']}")

    if verbose and "per_query" in results:
        print(f"\nPer-query breakdown:")
        for q in results["per_query"]:
            flags = []
            if q["exact"]: flags.append("exact")
            elif q.get("fuzzy"): flags.append("fuzzy")
            elif q.get("llm"): flags.append("llm")
            mark = "✓" if flags else "✗"
            flag_str = f" [{','.join(flags)}]" if flags else ""
            print(f"  {mark} Q{q['query_id']:>5s}{flag_str}")
            if not q["exact"] and (q.get("fuzzy") or q.get("llm")):
                print(f"         Gold:  {q['gold_answer'][:70]}")
                print(f"         Agent: {q['agent_answer'][:70]}")


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
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching (edit distance + token overlap)")
    parser.add_argument("--llm-judge", action="store_true", help="Enable LLM judge (requires Ollama)")
    parser.add_argument("--judge-model", default="qwen3.5:9b", help="Model for LLM judge")
    parser.add_argument("--ollama-url", default="http://localhost:11434", help="Ollama URL for LLM judge")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-query breakdown")
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
    results = evaluate_run_dir(
        Path(args.run_dir), gt,
        fuzzy=args.fuzzy or args.llm_judge,
        llm_judge=args.llm_judge,
        judge_model=args.judge_model,
        ollama_url=args.ollama_url,
    )
    print_results(results, verbose=args.verbose)
