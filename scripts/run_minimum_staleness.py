#!/usr/bin/env python3
"""Operational CLI for the static-vs-current minimum staleness experiment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_rubric.minimum_staleness import (
    export_audit_package,
    replay_current_aligned,
    score_bon,
    select_bon,
)
from dynamic_rubric.minimum_gold import (
    collect_gold_batch,
    gold_batch_status,
    prepare_gold_batch,
    submit_gold_batch,
)
from dynamic_rubric.minimum_analysis import analyze_minimum
from dynamic_rubric.rubrics.replay import ReplayMode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command",
        choices=(
            "replay",
            "score-bon",
            "select",
            "export-audit",
            "prepare-gold",
            "submit-gold",
            "gold-status",
            "collect-gold",
            "analyze",
        ),
    )
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8102")
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/bge-m3",
    )
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--mode",
        choices=(
            ReplayMode.DYNAMIC_FIXED_BUDGETED.value,
            ReplayMode.DYNAMIC_PREV_BUDGETED.value,
        ),
        default=ReplayMode.DYNAMIC_FIXED_BUDGETED.value,
    )
    parser.add_argument(
        "--private-gt",
        type=Path,
        default=Path("data/private_gt/healthbench_gold_rubrics.jsonl"),
    )
    parser.add_argument(
        "--gold-schema",
        type=Path,
        default=Path("configs/schemas/hidden_gold_grader_v1.json"),
    )
    args = parser.parse_args()
    if args.command == "replay":
        result = replay_current_aligned(
            args.run_root,
            args.score_endpoint,
            args.embedding_model_path,
            embedding_device=args.embedding_device,
            workers=args.workers,
            mode=args.mode,
        )
    elif args.command == "score-bon":
        result = score_bon(
            args.run_root,
            args.score_endpoint,
            workers=args.workers,
            mode=args.mode,
        )
    elif args.command == "select":
        result = select_bon(args.run_root)
    elif args.command == "export-audit":
        result = export_audit_package(args.run_root)
    elif args.command == "prepare-gold":
        result = prepare_gold_batch(args.run_root, args.private_gt, args.gold_schema)
    elif args.command == "submit-gold":
        result = submit_gold_batch(args.run_root)
    elif args.command == "gold-status":
        result = gold_batch_status(args.run_root)
    elif args.command == "collect-gold":
        result = collect_gold_batch(args.run_root, args.gold_schema)
    else:
        result = analyze_minimum(args.run_root)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
