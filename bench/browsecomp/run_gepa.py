#!/usr/bin/env python3
"""Run GEPA prompt optimization for BrowseComp-Plus (Phase 4).

Optimizes the system prompt using GEPA's optimize_anything with our
gw-bench evaluator. The reflection LM proposes prompt variants based
on per-query ASI feedback.

Usage:
    # Optimize on 8-query representative subset (fast iteration)
    uv run --project bench/browsecomp python bench/browsecomp/run_gepa.py \
        --subset 8 --max-metric-calls 20

    # Optimize on full sample30 (slower, more reliable)
    uv run --project bench/browsecomp python bench/browsecomp/run_gepa.py \
        --max-metric-calls 50

    # Use a specific reflection model
    uv run --project bench/browsecomp python bench/browsecomp/run_gepa.py \
        --reflection-lm anthropic/claude-sonnet-4-20250514 --max-metric-calls 30

Environment:
    ANTHROPIC_API_KEY — for Claude reflection models
    OPENAI_API_KEY — for OpenAI reflection models
    (or use --reflection-lm ollama/qwen3.5:9b for local)
"""

import argparse
import json
import sys
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()  # Load OPENAI_API_KEY etc. from .env

sys.path.insert(0, str(Path(__file__).parent))
from gepa_evaluator import BrowseCompEvaluator, load_sample30, load_ground_truth

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def select_representative_subset(examples: list[dict], n: int = 8) -> list[dict]:
    """Select a representative subset spanning difficulty levels.

    Based on ASI analysis:
    - 3 consistently correct queries (easy — verify we don't regress)
    - 3 high-variance queries (optimization opportunity)
    - 2 consistently wrong queries (stretch goals)
    """
    # Categorized from historical runs
    easy_ids = {"159", "191", "464", "797"}       # correct in >80% of runs
    medium_ids = {"572", "643", "853", "1128", "1144", "689", "894"}  # correct in some runs
    hard_ids = {"1034", "1036", "237", "706", "469", "152"}  # rarely correct

    easy = [e for e in examples if e["query_id"] in easy_ids][:3]
    medium = [e for e in examples if e["query_id"] in medium_ids][:3]
    hard = [e for e in examples if e["query_id"] in hard_ids][:max(0, n - 6)]

    subset = easy + medium + hard
    # Pad if needed
    remaining = [e for e in examples if e not in subset]
    while len(subset) < n and remaining:
        subset.append(remaining.pop(0))

    print(f"Selected {len(subset)} queries: {[e['query_id'] for e in subset]}")
    return subset[:n]


def build_evaluator_fn(evaluator: BrowseCompEvaluator, examples: list[dict]):
    """Build a GEPA-compatible evaluator function.

    For optimize_anything, the evaluator receives (candidate_str) and
    returns (score, side_info_dict). We evaluate across all examples
    and return the aggregate score + per-query ASI.
    """
    def evaluate(candidate: str) -> tuple[float, dict]:
        candidate_dict = {"system_prompt": candidate}
        total_score = 0.0
        per_query = []

        for example in examples:
            score, asi = evaluator.evaluate(candidate_dict, example)
            total_score += score
            per_query.append(asi)

        accuracy = total_score / len(examples) if examples else 0.0

        # Build aggregated side info for the reflection LM
        failure_dist = {}
        for q in per_query:
            mode = q["failure_mode"]
            failure_dist[mode] = failure_dist.get(mode, 0) + 1

        correct_ids = [q["query_id"] for q in per_query if q.get("correct", False)]
        wrong_queries = [q for q in per_query if not q.get("correct", False)]

        # Build concise feedback string for reflection
        feedback_lines = [
            f"Accuracy: {int(total_score)}/{len(examples)} ({accuracy:.0%})",
            f"Failures: {failure_dist}",
            f"Correct: {correct_ids}",
            "",
        ]
        for q in wrong_queries[:5]:  # Show up to 5 failures in detail
            feedback_lines.append(
                f"Q{q.get('query_id', '?')} [{q.get('failure_mode', '?')}]: "
                f"expected '{q.get('expected_answer', '?')[:50]}', "
                f"got '{q.get('final_answer', '?')[:50]}'"
            )
            searches = q.get("searches_issued", [])
            if searches:
                feedback_lines.append(f"  searches: {searches[:3]}")

        side_info = {
            "accuracy": accuracy,
            "n_correct": int(total_score),
            "n_total": len(examples),
            "failure_distribution": failure_dist,
            "correct_ids": correct_ids,
            "per_query_summary": "\n".join(feedback_lines),
        }

        return accuracy, side_info

    return evaluate


def main():
    parser = argparse.ArgumentParser(description="Run GEPA prompt optimization")
    parser.add_argument("--subset", type=int, default=0,
                        help="Use N-query representative subset (0 = all 30)")
    parser.add_argument("--max-metric-calls", type=int, default=20,
                        help="Max evaluator invocations")
    parser.add_argument("--reflection-lm", default="openai/gpt-4o",
                        help="LiteLLM model string for reflection/proposal")
    parser.add_argument("--model", default="qwen3.5:9b",
                        help="Agent LLM model (via Ollama)")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--run-dir", default=None,
                        help="Directory to save GEPA state (default: auto)")
    args = parser.parse_args()

    # Load examples
    examples = load_sample30()
    if args.subset > 0:
        examples = select_representative_subset(examples, args.subset)
    print(f"Evaluating on {len(examples)} queries")

    # Load seed prompt
    prompt_path = REPO_ROOT / "bench" / "browsecomp" / "configs" / "system_prompt.txt"
    seed_prompt = prompt_path.read_text()
    print(f"Seed prompt: {len(seed_prompt)} chars")

    # Create evaluator
    bench_evaluator = BrowseCompEvaluator(
        model=args.model,
        max_turns=args.max_turns,
        k=args.k,
    )

    # Build GEPA-compatible evaluator
    evaluate_fn = build_evaluator_fn(bench_evaluator, examples)

    # Run optimization
    from gepa.optimize_anything import (
        optimize_anything, GEPAConfig, EngineConfig,
        ReflectionConfig, TrackingConfig,
    )

    run_dir = args.run_dir or f"bench/browsecomp/gepa_runs/{time.strftime('%Y%m%d_%H%M%S')}"

    config = GEPAConfig(
        engine=EngineConfig(
            max_metric_calls=args.max_metric_calls,
            run_dir=run_dir,
            display_progress_bar=True,
            candidate_selection_strategy="pareto",
            parallel=False,  # gw-bench is already GPU-bound
        ),
        reflection=ReflectionConfig(
            reflection_lm=args.reflection_lm,
            skip_perfect_score=True,
            perfect_score=1.0,
        ),
        tracking=TrackingConfig(),
    )

    objective = (
        "Optimize the system prompt for an rLM agent that answers factoid queries "
        "by searching a 100K web document corpus. The agent works in a Python REPL "
        "with search(), get_document(), llm_query(), and FINAL() tools.\n\n"
        "Key findings from prior experiments:\n"
        "- 30% of failures are extraction errors (right doc found, wrong answer extracted)\n"
        "- 23% are retrieval misses (right doc never found via BM25 keyword search)\n"
        "- 10% are hedges (model says 'unable to determine' instead of guessing)\n"
        "- The model must call FINAL('answer') to submit — it often runs out of turns\n"
        "- BM25 search matches keywords, so search queries should be 2-5 specific nouns\n"
        "- Answers are short factual phrases (names, dates, numbers)\n\n"
        "Focus on: (1) making the agent submit concrete answers via FINAL(), "
        "(2) improving search query strategy for BM25, "
        "(3) better fact extraction from documents via llm_query()."
    )

    background = (
        "The agent gets 12 iterations in a Python REPL. Pre-search decomposes the query "
        "into 5 keyword queries. The agent has access to search(), get_document(), "
        "llm_query(), batch_llm_query(), and FINAL(). The system prompt defines the "
        "workflow, tool descriptions, examples, and rules. The LLM is qwen3.5:9b (local, "
        "small but capable). Each query takes ~2-3 minutes."
    )

    print(f"\nStarting GEPA optimization:")
    print(f"  Reflection LM: {args.reflection_lm}")
    print(f"  Max metric calls: {args.max_metric_calls}")
    print(f"  Queries: {len(examples)}")
    print(f"  Run dir: {run_dir}")
    print()

    result = optimize_anything(
        seed_candidate=seed_prompt,
        evaluator=evaluate_fn,
        objective=objective,
        background=background,
        config=config,
    )

    # Report results
    print("\n" + "=" * 60)
    print("GEPA Optimization Complete")
    print("=" * 60)
    best = result.best_candidate
    if isinstance(best, dict):
        best_text = best.get("system_prompt", str(best))
    else:
        best_text = str(best)
    print(f"Best candidate index: {result.best_idx}")
    print(f"Best candidate length: {len(best_text)} chars")
    print(f"Total metric calls: {result.total_metric_calls}")

    # Save best prompt
    best_path = Path(run_dir) / "best_prompt.txt"
    best_path.parent.mkdir(parents=True, exist_ok=True)
    best_path.write_text(best_text)
    print(f"Best prompt saved to: {best_path}")

    # Save full result
    result_path = Path(run_dir) / "result.json"
    result_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    print(f"Full result saved to: {result_path}")

    # Evaluate best on full sample30 if we used a subset
    if args.subset > 0:
        print(f"\nValidating best prompt on full sample30...")
        all_examples = load_sample30()
        full_eval = build_evaluator_fn(bench_evaluator, all_examples)
        full_score, full_info = full_eval(best_text)
        print(f"Full sample30 accuracy: {full_info['n_correct']}/{full_info['n_total']} ({full_score:.1%})")
        print(f"Failures: {full_info['failure_distribution']}")


if __name__ == "__main__":
    main()
