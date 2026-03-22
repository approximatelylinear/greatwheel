#!/usr/bin/env python3
"""GEPA evaluator wrapper for BrowseComp-Plus benchmark (Phase 3).

Wraps the gw-bench Rust binary as a Python callable matching the
`optimize_anything` evaluator signature:

    evaluate(candidate: dict, example: dict) -> tuple[float, dict]

The candidate dict contains:
    - "system_prompt": str — the full system prompt text
    - "config": dict (optional) — BenchConfig overrides

The example dict contains:
    - "query_id": str
    - "query": str
    - "answer": str
    - "gold_docids": list[str] (optional)

Usage standalone (test single query):
    python gepa_evaluator.py --query-id 159 --candidate-config configs/baseline.toml

Usage with GEPA:
    from gepa_evaluator import BrowseCompEvaluator
    evaluator = BrowseCompEvaluator()
    score, asi = evaluator(candidate, example)
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

# Repo root (two levels up from bench/browsecomp/)
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
VENDOR_ROOT = REPO_ROOT / "vendor" / "BrowseComp-Plus"
BINARY = REPO_ROOT / "target" / "release" / "gw-bench"
DEFAULT_TANTIVY = REPO_ROOT / "data" / "tantivy-corpus"
DEFAULT_MODEL = "qwen3.5:9b"


def load_ground_truth() -> dict[str, dict]:
    """Load ground truth with answers and gold docids."""
    gt_path = VENDOR_ROOT / "data" / "browsecomp_plus_decrypted.jsonl"
    gt = {}
    with open(gt_path) as f:
        for line in f:
            obj = json.loads(line)
            qid = str(obj["query_id"])
            gold_docids = []
            for doc in obj.get("gold_docs", []):
                gold_docids.append(str(doc["docid"]))
            for doc in obj.get("evidence_docs", []):
                if str(doc["docid"]) not in gold_docids:
                    gold_docids.append(str(doc["docid"]))
            gt[qid] = {
                "query_id": qid,
                "query": obj["query"],
                "answer": obj["answer"],
                "gold_docids": gold_docids,
            }
    return gt


def load_sample30() -> list[dict]:
    """Load the 30-query sample as a list of example dicts."""
    gt = load_ground_truth()
    sample_path = REPO_ROOT / "bench" / "browsecomp" / "sample30.tsv"
    examples = []
    with open(sample_path) as f:
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                qid = parts[0]
                if qid in gt:
                    examples.append(gt[qid])
    return examples


def score_answer(agent_answer: str, ground_truth: str) -> bool:
    """Check if ground truth appears in agent answer."""
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
    """Check if the answer is a refusal."""
    lower = answer.lower()
    if len(answer) > 150:
        return True
    hedges = [
        "unable to", "not found", "insufficient", "cannot determine",
        "could not find", "no exact match", "no specific", "n/a",
    ]
    return any(h in lower for h in hedges)


def classify_failure(result: dict, gold_answer: str, gold_docids: list[str]) -> str:
    """Classify a single query result into a failure mode."""
    answer = extract_answer(result)
    correct = score_answer(answer, gold_answer)
    if correct:
        return "correct"

    status = result.get("status", "")
    if "fallback" in status or "timeout" in status:
        return "timeout"

    if is_hedge(answer):
        return "hedge"

    retrieved = set(str(d) for d in result.get("retrieved_docids", []))
    if retrieved & set(gold_docids):
        return "extraction_error"

    return "retrieval_miss"


def extract_answer(result: dict) -> str:
    """Extract the agent's final answer from a result record."""
    for entry in reversed(result.get("result", [])):
        out = entry.get("output", "")
        if isinstance(out, str) and "Exact Answer:" in out:
            return out.split("Exact Answer:", 1)[1].strip()
    return ""


class BrowseCompEvaluator:
    """Evaluator matching the GEPA optimize_anything signature.

    Invokes the gw-bench Rust binary for each (candidate, example) pair.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        tantivy_index: str = str(DEFAULT_TANTIVY),
        binary: str = str(BINARY),
        max_turns: int = 12,
        k: int = 10,
        timeout: int = 300,
    ):
        self.model = model
        self.tantivy_index = tantivy_index
        self.binary = binary
        self.max_turns = max_turns
        self.k = k
        self.timeout = timeout

        if not Path(self.binary).exists():
            raise FileNotFoundError(
                f"gw-bench binary not found at {self.binary}. "
                f"Run: cargo build --release --bin gw-bench"
            )

    def __call__(self, candidate: dict, example: dict) -> tuple[float, dict]:
        """Evaluate a candidate on a single example.

        Args:
            candidate: {"system_prompt": str, "config": dict (optional)}
            example: {"query_id": str, "query": str, "answer": str, "gold_docids": list}

        Returns:
            (score, asi_dict) where score is 0.0 or 1.0
        """
        return self.evaluate(candidate, example)

    def evaluate(self, candidate: dict, example: dict) -> tuple[float, dict]:
        """Run one query with the candidate config and return (score, ASI)."""
        qid = str(example["query_id"])
        query_text = example["query"]
        gold_answer = example["answer"]
        gold_docids = example.get("gold_docids", [])

        with tempfile.TemporaryDirectory(prefix="gepa_eval_") as tmpdir:
            tmpdir = Path(tmpdir)

            # Write system prompt to temp file
            prompt_path = tmpdir / "system_prompt.txt"
            prompt_text = candidate.get("system_prompt", "")
            if prompt_text:
                prompt_path.write_text(prompt_text)

            # Build config TOML
            config = candidate.get("config", {}).copy()
            if prompt_text:
                config["system_prompt_path"] = str(prompt_path)
            config_path = tmpdir / "config.toml"
            self._write_toml(config_path, config)

            # Write single-query TSV (more reliable than passing query text via CLI)
            query_tsv = tmpdir / "query.tsv"
            query_tsv.write_text(f"{qid}\t{query_text}\n")

            output_dir = tmpdir / "output"
            output_dir.mkdir()

            # Invoke gw-bench
            cmd = [
                self.binary,
                "--search-backend", "native",
                "--tantivy-index", self.tantivy_index,
                "--model", self.model,
                "--config", str(config_path),
                "--query", str(query_tsv),
                "--output-dir", str(output_dir),
                "--k", str(self.k),
                "--max-turns", str(self.max_turns),
            ]

            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    cwd=str(REPO_ROOT),
                )
                elapsed = time.monotonic() - t0
                if proc.returncode != 0:
                    return 0.0, {
                        "query_id": qid,
                        "failure_mode": "crash",
                        "error": proc.stderr[-500:] if proc.stderr else "unknown",
                        "elapsed_s": round(elapsed, 1),
                    }
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                return 0.0, {
                    "query_id": qid,
                    "failure_mode": "timeout",
                    "error": f"Process timeout after {self.timeout}s",
                    "elapsed_s": round(elapsed, 1),
                }

            # Read result JSON
            result_files = list(output_dir.glob("*.json"))
            if not result_files:
                return 0.0, {
                    "query_id": qid,
                    "failure_mode": "crash",
                    "error": "No output JSON produced",
                    "elapsed_s": round(elapsed, 1),
                }

            result = json.loads(result_files[0].read_text())

            # Score and classify
            answer = extract_answer(result)
            correct = score_answer(answer, gold_answer)
            failure_mode = classify_failure(result, gold_answer, gold_docids)

            # Extract search queries from trajectory
            searches = []
            for t in result.get("trajectory", []):
                for cb in t.get("code_blocks", []):
                    for m in re.finditer(r'search\(\s*["\x27](.+?)["\x27]\s*\)', cb):
                        searches.append(m.group(1))

            n_turns = sum(1 for t in result.get("trajectory", []) if t.get("role") == "assistant")
            tokens = result.get("usage", {}).get("total_tokens", 0)
            retrieved = result.get("retrieved_docids", [])

            asi = {
                "query_id": qid,
                "correct": correct,
                "failure_mode": failure_mode,
                "final_answer": answer[:200],
                "expected_answer": gold_answer[:200],
                "gold_docid_retrieved": bool(set(str(d) for d in gold_docids) & set(str(d) for d in retrieved)),
                "retrieved_docids": retrieved,
                "searches_issued": searches,
                "n_unique_searches": len(set(searches)),
                "n_docs_read": sum(
                    1 for t in result.get("trajectory", [])
                    for cb in t.get("code_blocks", [])
                    if "get_document(" in cb
                ),
                "turns_used": n_turns,
                "token_cost": tokens,
                "elapsed_s": round(elapsed, 1),
                "status": result.get("status", ""),
            }

            return float(correct), asi

    def evaluate_batch(self, candidate: dict, examples: list[dict]) -> tuple[float, list[dict]]:
        """Evaluate a candidate on multiple examples. Returns (avg_score, asi_list)."""
        total_score = 0.0
        asi_list = []
        for example in examples:
            score, asi = self.evaluate(candidate, example)
            total_score += score
            asi_list.append(asi)
        avg_score = total_score / len(examples) if examples else 0.0
        return avg_score, asi_list

    @staticmethod
    def _write_toml(path: Path, config: dict):
        """Write a dict as TOML (simple flat key=value format)."""
        lines = []
        for k, v in sorted(config.items()):
            if isinstance(v, str):
                lines.append(f'{k} = "{v}"')
            elif isinstance(v, bool):
                lines.append(f"{k} = {'true' if v else 'false'}")
            elif isinstance(v, int):
                lines.append(f"{k} = {v}")
            elif isinstance(v, float):
                lines.append(f"{k} = {v}")
        path.write_text("\n".join(lines) + "\n")


def main():
    """Standalone test: evaluate a single query with the current config."""
    parser = argparse.ArgumentParser(description="GEPA evaluator for BrowseComp-Plus")
    parser.add_argument("--query-id", help="Query ID to evaluate (from sample30)")
    parser.add_argument("--candidate-config", help="Path to candidate TOML config")
    parser.add_argument("--candidate-prompt", help="Path to candidate system prompt text file")
    parser.add_argument("--all", action="store_true", help="Evaluate all 30 queries")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    # Load ground truth
    gt = load_ground_truth()
    examples = load_sample30()

    # Build candidate
    candidate = {}
    if args.candidate_prompt:
        candidate["system_prompt"] = Path(args.candidate_prompt).read_text()
    if args.candidate_config:
        # Load config as dict (for passing overrides)
        import tomllib
        with open(args.candidate_config, "rb") as f:
            candidate["config"] = tomllib.load(f)

    evaluator = BrowseCompEvaluator(
        model=args.model,
        max_turns=args.max_turns,
        k=args.k,
    )

    if args.all:
        score, asi_list = evaluator.evaluate_batch(candidate, examples)
        print(f"\n=== Batch Evaluation ===")
        print(f"Accuracy: {score:.4f} ({int(score * len(examples))}/{len(examples)})")
        for asi in asi_list:
            mark = "✓" if asi["correct"] else "✗"
            print(f"  {mark} Q{asi['query_id']:>5s} {asi['failure_mode']:20s} "
                  f"turns={asi['turns_used']:2d} tokens={asi['token_cost']:6d} "
                  f"{asi['elapsed_s']:.0f}s")
            if not asi["correct"]:
                print(f"         Gold: {asi['expected_answer'][:70]}")
                print(f"         Got:  {asi['final_answer'][:70]}")
    elif args.query_id:
        if args.query_id not in gt:
            print(f"Query ID {args.query_id} not found in ground truth", file=sys.stderr)
            sys.exit(1)
        example = gt[args.query_id]
        score, asi = evaluator.evaluate(candidate, example)
        print(json.dumps(asi, indent=2))
        print(f"\nScore: {score}")
    else:
        # Default: evaluate first query in sample30
        example = examples[0]
        score, asi = evaluator.evaluate(candidate, example)
        print(json.dumps(asi, indent=2))
        print(f"\nScore: {score}")


if __name__ == "__main__":
    main()
