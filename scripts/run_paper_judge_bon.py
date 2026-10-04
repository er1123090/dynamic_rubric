#!/usr/bin/env python3
"""Run the isolated paper-style Qwen/GPT-5 BoN rejudge experiment."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from dynamic_rubric.minimum_gold_streaming import sync_streaming_gold
from dynamic_rubric.minimum_interim import (
    ordered_prompt_subset,
    target_gold_complete,
    target_groups,
)
from dynamic_rubric.paper_judge_experiment import (
    analyze_subset,
    prepare_or_submit_gold_subset,
    prepare_paper_run,
    score_and_select_subset,
)


DEFAULT_SOURCE = Path("artifacts/runs/pilot-static-r0-100step-20260821")
DEFAULT_RUN = Path("artifacts/runs/pilot-static-r0-100step-20260821-paper-judge-v1")


def _status(run_root: Path, stage: str, **details: Any) -> dict[str, Any]:
    value = {
        "stage": stage,
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "updated_at_unix": time.time(),
        **details,
    }
    write_json_atomic(
        run_root / "paper-judge-experiment" / "progress.json",
        value,
        immutable=False,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return value


def _routing_contract(score_endpoint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "paper-style-qwen-bon-rejudge",
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "served_model": "Qwen/Qwen3-32B",
        "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "score_proxy_endpoint": score_endpoint,
        "strategy": "deterministic-length-balanced-largest-first",
        "weights": [5, 2, 2],
        "replicas": [
            {
                "host": "trainer",
                "gpu_indices": [0],
                "endpoint": "http://127.0.0.1:8004",
                "weight": 5,
            },
            {
                "host": "inference_a",
                "gpu_indices": [0],
                "tunnel_endpoint": "http://127.0.0.1:18047",
                "weight": 2,
            },
            {
                "host": "inference_a",
                "gpu_indices": [1],
                "tunnel_endpoint": "http://127.0.0.1:18057",
                "weight": 2,
            },
        ],
        "excluded_gpus": [
            {"host": "trainer", "gpu_indices": [1]},
            {"host": "inference_a", "gpu_indices": [2, 3]},
        ],
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    result = prepare_paper_run(args.source_run_root, args.run_root)
    contract = _routing_contract(args.score_endpoint)
    write_json_atomic(args.run_root / "paper-judge-qwen-routing.json", contract)
    return {**result, "routing": contract}


def _sync(args: argparse.Namespace) -> dict[str, Any]:
    prompt_ids = ordered_prompt_subset(args.run_root, args.prompt_count)
    expected = len(target_groups(prompt_ids))
    while True:
        result = sync_streaming_gold(args.run_root, args.gold_schema)
        completed = target_gold_complete(args.run_root, prompt_ids)
        _status(
            args.run_root,
            "gold-sync",
            completed_target_groups=completed,
            expected_target_groups=expected,
            submitted_groups=result["submitted_groups"],
        )
        if completed == expected or not args.wait:
            return result
        time.sleep(args.poll_seconds)


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    _status(
        args.run_root,
        "prepared",
        prompt_count=args.prompt_count,
        source_run_root=str(args.source_run_root),
    )
    if args.phase == "prepare":
        return prepared

    score_result: dict[str, Any] | None = None
    if args.phase in {"score", "all"}:
        score_result = score_and_select_subset(
            args.run_root,
            args.source_run_root,
            args.score_endpoint,
            prompt_count=args.prompt_count,
            workers=args.workers,
        )
        _status(
            args.run_root,
            "qwen-score-and-selection-complete",
            prompt_count=args.prompt_count,
            groups=len(target_groups(score_result["prompt_ids"])),
        )
        if args.phase == "score":
            return score_result

    if args.phase in {"gold", "submit", "all"}:
        submit = args.phase == "submit" or (args.phase == "all" and args.submit_gold)
        gold_result = prepare_or_submit_gold_subset(
            args.run_root,
            args.source_run_root,
            args.private_gt,
            args.gold_schema,
            prompt_count=args.prompt_count,
            submit=submit,
        )
        _status(
            args.run_root,
            "gold-submitted" if submit else "gold-input-prepared",
            requests=gold_result["requests"],
            groups=len(gold_result["groups"]),
        )
        if args.phase == "gold" or (args.phase == "all" and not submit):
            return {"score": score_result, "gold": gold_result}
        if args.phase == "submit" and not args.wait:
            return gold_result

    if args.phase in {"sync", "submit", "all"}:
        synced = _sync(args)
        if not args.wait:
            return synced

    result = analyze_subset(
        args.run_root,
        prompt_count=args.prompt_count,
    )
    _status(
        args.run_root,
        "complete",
        curves=str(args.run_root / f"interim-bon-{args.prompt_count}prompt" / "n_curves.png"),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=("prepare", "score", "gold", "submit", "sync", "analyze", "all"),
        default="all",
    )
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8104")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--submit-gold", action="store_true")
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument(
        "--private-gt",
        type=Path,
        default=Path("data/private_gt/healthbench_gold_rubrics.jsonl"),
    )
    parser.add_argument(
        "--gold-schema",
        type=Path,
        default=Path("configs/schemas/paper_gold_grader_v1.json"),
    )
    args = parser.parse_args()
    if args.prompt_count < 1 or args.workers < 1 or args.poll_seconds < 10:
        parser.error("prompt-count/workers must be positive and poll-seconds >= 10")
    if args.phase == "analyze":
        _prepare(args)
        result = analyze_subset(
            args.run_root,
            prompt_count=args.prompt_count,
        )
    else:
        result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
