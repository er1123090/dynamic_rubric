#!/usr/bin/env python3
"""Run π_ref-vs-π_old OnlineRubric BoN with paper-style Qwen and GPT-5 GT."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.minimum_gold_streaming import sync_streaming_gold
from dynamic_rubric.online_rubric_bon import (
    analyze_online_subset,
    prepare_online_run,
    prepare_or_submit_available_gold,
    score_and_select_online_subset,
)


DEFAULT_SOURCE = Path("artifacts/runs/pilot-static-r0-100step-20260821")
DEFAULT_RUN = Path("artifacts/runs/pilot-static-r0-100step-20260821-onlinerubric-paper-judge-v1")


def _status(run_root: Path, stage: str, **details: Any) -> dict[str, Any]:
    value = {"stage": stage, "updated_at_unix": time.time(), **details}
    write_json_atomic(
        run_root / "online-rubric-experiment" / "progress.json",
        value,
        immutable=False,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    prepared = prepare_online_run(args.source_run_root, args.run_root)
    if args.phase == "prepare":
        return prepared
    if args.phase == "score":
        _status(args.run_root, "qwen-score-started", prompt_count=args.prompt_count)
        result = score_and_select_online_subset(
            args.run_root,
            args.score_endpoint,
            prompt_count=args.prompt_count,
            workers=args.workers,
        )
        _status(args.run_root, "qwen-score-and-selection-complete", **result)
        return result
    if args.phase in {"gold", "submit"}:
        result = prepare_or_submit_available_gold(
            args.run_root,
            args.private_gt,
            args.gold_schema,
            prompt_count=args.prompt_count,
            submit=args.phase == "submit",
        )
        _status(
            args.run_root,
            "gold-submitted" if args.phase == "submit" else "gold-input-prepared",
            ready_groups=result["ready_groups"],
            expected_groups=result["expected_groups"],
            requests=result["requests"],
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
        result = analyze_online_subset(args.run_root, prompt_count=args.prompt_count)
        _status(args.run_root, "graph-updated", policies=result["policies"])
        return result
    raise ValueError(f"unsupported phase: {args.phase}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("prepare", "score", "gold", "submit", "sync", "analyze")
    )
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
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
    if args.prompt_count < 1 or args.workers < 1:
        parser.error("prompt-count/workers must be positive")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
