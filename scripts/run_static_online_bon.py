#!/usr/bin/env python3
"""Run Static R0 vs OnlineRubric for either pi_ref or pi_old control."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.minimum_gold_streaming import sync_streaming_gold
from dynamic_rubric.minimum_staleness import FOCAL_STEPS
from dynamic_rubric.paper_judge_experiment import (
    analyze_subset,
    prepare_or_submit_gold_subset,
    score_and_select_subset,
)
from dynamic_rubric.static_online_bon import CONTROL_STAGES, prepare_static_online_run


DEFAULT_SOURCE = Path("artifacts/runs/pilot-static-r0-100step-20260821")
DEFAULT_RUNS = {
    "pi_ref": Path(
        "artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-piref-paper-judge-v1"
    ),
    "pi_old": Path(
        "artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-piold-paper-judge-v1"
    ),
}


def _status(run_root: Path, stage: str, **details: Any) -> dict[str, Any]:
    value = {"stage": stage, "updated_at_unix": time.time(), **details}
    write_json_atomic(
        run_root / "static-online-experiment" / "progress.json",
        value,
        immutable=False,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return value


def _routing_contract(score_endpoint: str) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "purpose": "paper-style-static-vs-onlinerubric-bon-rejudge",
        "served_model": "Qwen/Qwen3-32B",
        "model_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "tokenizer_revision": "9216db5781bf21249d130ec9da846c4624c16137",
        "score_proxy_endpoint": score_endpoint,
        "strategy": "deterministic-length-balanced-largest-first",
        "weights": [5, 2, 2, 1, 1],
        "replicas": [
            {"host": "trainer", "gpu_indices": [0], "endpoint": "http://127.0.0.1:8004"},
            {"host": "inference_a", "gpu_indices": [0], "endpoint": "http://127.0.0.1:18047"},
            {"host": "inference_a", "gpu_indices": [1], "endpoint": "http://127.0.0.1:18057"},
            {
                "host": "inference_b",
                "gpu_indices": [0, 1],
                "endpoint": "http://127.0.0.1:18067",
            },
            {
                "host": "inference_b",
                "gpu_indices": [2, 3],
                "endpoint": "http://127.0.0.1:18077",
            },
        ],
    }


def _prepare(args: argparse.Namespace) -> dict[str, Any]:
    result = prepare_static_online_run(args.source_run_root, args.run_root, args.online_control)
    write_json_atomic(
        args.run_root / "paper-judge-qwen-routing.json",
        _routing_contract(args.score_endpoint),
    )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = _prepare(args)
    if args.phase == "prepare":
        return prepared
    if args.phase == "score":
        _status(
            args.run_root,
            "qwen-score-started",
            online_control=args.online_control,
            prompt_count=args.prompt_count,
        )
        result = score_and_select_subset(
            args.run_root,
            args.source_run_root,
            args.score_endpoint,
            prompt_count=args.prompt_count,
            workers=args.workers,
        )
        _status(args.run_root, "qwen-score-and-selection-complete", **result)
        return result
    if args.phase in {"gold", "submit"}:
        submit = args.phase == "submit"
        result = prepare_or_submit_gold_subset(
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
            requests=result["requests"],
            groups=len(result["groups"]),
        )
        return result
    if args.phase == "sync":
        result = sync_streaming_gold(args.run_root, args.gold_schema)
        _status(
            args.run_root,
            "gold-synced",
            completed_groups=result["completed_groups"],
            submitted_groups=result["submitted_groups"],
        )
        return result
    if args.phase == "analyze":
        result = analyze_subset(args.run_root, prompt_count=args.prompt_count)
        interim_root = args.run_root / f"interim-bon-{args.prompt_count}prompt"
        subprocess.run(
            [
                sys.executable,
                "scripts/plot_paper_proxy_gt.py",
                "--run-root",
                str(args.run_root),
                "--interim-root",
                str(interim_root),
                "--steps",
                *(str(step) for step in FOCAL_STEPS),
            ],
            check=True,
        )
        _status(
            args.run_root,
            "complete",
            graph=str(interim_root / "combined_proxy_gt_curves.svg"),
        )
        return result
    raise ValueError(f"unsupported phase: {args.phase}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        required=True,
        choices=("prepare", "score", "gold", "submit", "sync", "analyze"),
    )
    parser.add_argument("--online-control", required=True, choices=tuple(CONTROL_STAGES))
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8104")
    parser.add_argument("--workers", type=int, default=12)
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
    if args.run_root is None:
        args.run_root = DEFAULT_RUNS[args.online_control]
    if args.prompt_count < 1 or args.workers < 1:
        parser.error("prompt-count/workers must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
