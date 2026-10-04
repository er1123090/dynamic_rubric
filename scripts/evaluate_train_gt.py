#!/usr/bin/env python3
"""Prepare, submit, monitor, and collect GPT-5 mini GT scores for train rollouts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_rubric.train_gt_audit import (
    DEFAULT_MODEL,
    collect_train_gold_batch,
    prepare_train_gold_batch,
    stage_root,
    submit_train_gold_batch,
    train_gold_batch_status,
)


DEFAULT_RUN_ROOT = Path("artifacts/runs/pilot-static-r0-100step-20260821")
DEFAULT_PRIVATE_GT = Path("data/private_gt/healthbench_gold_rubrics.jsonl")
DEFAULT_SCHEMA = Path("configs/schemas/paper_gold_grader_v1.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("prepare", "submit", "status", "collect"))
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--private-gt", type=Path, default=DEFAULT_PRIVATE_GT)
    parser.add_argument("--schema", type=Path, default=DEFAULT_SCHEMA)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-step", type=int, default=50)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    approval = args.approval or stage_root(args.run_root.resolve(), args.model) / "egress-approval.json"
    if args.action == "prepare":
        result = prepare_train_gold_batch(
            args.run_root,
            args.private_gt,
            args.schema,
            approval,
            model=args.model,
            max_step=args.max_step,
        )
    elif args.action == "submit":
        result = submit_train_gold_batch(
            args.run_root, approval, model=args.model
        )
    elif args.action == "status":
        result = train_gold_batch_status(args.run_root, model=args.model)
    else:
        result = collect_train_gold_batch(args.run_root, args.schema, model=args.model)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
