from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic
from dynamic_rubric.providers.vllm import normalized_yes_probability
from dynamic_rubric.services.vllm_score_proxy import ProxyState


TARGETS = ["YES", "NO"]


def _real_reward_batches(
    rollouts_path: Path, rubrics_path: Path, limit: int
) -> list[list[str]]:
    rubrics = {str(row["prompt_id"]): row for row in read_jsonl(rubrics_path)}
    batches: list[list[str]] = []
    for rollout in read_jsonl(rollouts_path)[:limit]:
        prompt_id = str(rollout["prompt_id"])
        criteria = rubrics[prompt_id]["criteria"]
        if not isinstance(criteria, list) or len(criteria) != 8:
            raise ValueError(f"static rubric must have eight criteria: {prompt_id}")
        response = str(rollout["output"])
        batches.append(
            [
                f"Criterion: {criterion['text']}\nResponse: {response}\nAnswer:"
                for criterion in criteria
            ]
        )
    if len(batches) != limit:
        raise ValueError(f"requested {limit} batches, found {len(batches)}")
    return batches


def _benchmark(
    state: ProxyState, batches: list[list[str]], workers: int
) -> tuple[list[list[dict[str, float]]], dict[str, float | int]]:
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda batch: state.score(batch, TARGETS), batches))
    elapsed = time.perf_counter() - started
    criterion_count = sum(len(batch) for batch in batches)
    return results, {
        "batches": len(batches),
        "criteria": criterion_count,
        "target_sequences": criterion_count * len(TARGETS),
        "workers": workers,
        "elapsed_seconds": elapsed,
        "criteria_per_second": criterion_count / elapsed,
    }


def _comparison(
    reference: list[list[dict[str, float]]],
    candidate: list[list[dict[str, float]]],
) -> dict[str, float]:
    if len(reference) != len(candidate):
        raise ValueError("benchmark result batch count mismatch")
    raw_deltas: list[float] = []
    probability_deltas: list[float] = []
    for reference_batch, candidate_batch in zip(reference, candidate):
        if len(reference_batch) != len(candidate_batch):
            raise ValueError("benchmark result criterion count mismatch")
        for reference_row, candidate_row in zip(reference_batch, candidate_batch):
            raw_deltas.extend(
                abs(reference_row[target] - candidate_row[target]) for target in TARGETS
            )
            reference_probability = normalized_yes_probability(
                reference_row["YES"], reference_row["NO"]
            )
            candidate_probability = normalized_yes_probability(
                candidate_row["YES"], candidate_row["NO"]
            )
            probability_deltas.append(
                abs(reference_probability - candidate_probability)
            )
    return {
        "max_abs_target_logprob_delta": max(raw_deltas, default=0.0),
        "mean_abs_target_logprob_delta": (
            sum(raw_deltas) / len(raw_deltas) if raw_deltas else 0.0
        ),
        "max_abs_probability_yes_delta": max(probability_deltas, default=0.0),
        "mean_abs_probability_yes_delta": (
            sum(probability_deltas) / len(probability_deltas)
            if probability_deltas
            else 0.0
        ),
    }


def _state(args: argparse.Namespace, tokenizer: Any, upstream: str | list[str]) -> ProxyState:
    return ProxyState(
        upstream=upstream,
        served_model=args.served_model,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer=tokenizer,
    )


def main() -> None:
    from transformers import AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--rubrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batches", type=int, default=16)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if len(args.upstream) < 2:
        raise ValueError("at least two upstreams are required for replica comparison")
    if args.batches < 1 or args.workers < 1:
        raise ValueError("batches and workers must be positive")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    batches = _real_reward_batches(args.rollouts, args.rubrics, args.batches)

    endpoint_results: dict[str, list[list[dict[str, float]]]] = {}
    endpoint_benchmarks: dict[str, Mapping[str, float | int]] = {}
    for upstream in args.upstream:
        state = _state(args, tokenizer, upstream)
        state.validate_upstreams()
        scores, benchmark = _benchmark(state, batches, args.workers)
        endpoint_results[upstream] = scores
        endpoint_benchmarks[upstream] = benchmark

    reference_url = args.upstream[0]
    comparisons = {
        upstream: _comparison(endpoint_results[reference_url], endpoint_results[upstream])
        for upstream in args.upstream[1:]
    }
    pooled_state = _state(args, tokenizer, args.upstream)
    pooled_state.validate_upstreams()
    _, pooled_benchmark = _benchmark(pooled_state, batches, args.workers)

    write_json_atomic(
        args.output,
        {
            "schema_version": 1,
            "served_model": args.served_model,
            "model_revision": args.model_revision,
            "tokenizer_revision": args.tokenizer_revision,
            "rollouts": str(args.rollouts),
            "rubrics": str(args.rubrics),
            "reference_upstream": reference_url,
            "endpoint_benchmarks": endpoint_benchmarks,
            "comparisons": comparisons,
            "pooled_benchmark": pooled_benchmark,
        },
    )


if __name__ == "__main__":
    main()
