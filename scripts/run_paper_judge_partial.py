#!/usr/bin/env python3
"""Submit and analyze completed paper-judge policy checkpoints early."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.judge_prompts import PAPER_JUDGE_PROMPT_VERSION
from dynamic_rubric.minimum_gold_streaming import (
    prepare_gold_selection_shard,
    submit_gold_selection_shard,
    sync_streaming_gold,
)
from dynamic_rubric.minimum_interim import (
    analyze_interim,
    ordered_prompt_subset,
    target_gold_complete,
    target_groups,
)
from dynamic_rubric.minimum_staleness import MinimumExperimentError, _shard_name
from dynamic_rubric.paper_judge_experiment import prepare_paper_run


DEFAULT_SOURCE = Path("artifacts/runs/pilot-static-r0-100step-20260821")
DEFAULT_RUN = Path("artifacts/runs/pilot-static-r0-100step-20260821-paper-judge-v1")


def _status(run_root: Path, steps: tuple[int, ...], stage: str, **details: Any) -> None:
    suffix = "-".join(str(step) for step in steps)
    value = {
        "stage": stage,
        "policy_steps": list(steps),
        "judge_prompt_version": PAPER_JUDGE_PROMPT_VERSION,
        "updated_at_unix": time.time(),
        **details,
    }
    write_json_atomic(
        run_root / "paper-judge-partial" / f"pi-{suffix}-progress.json",
        value,
        immutable=False,
    )
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def _selection_paths(
    run_root: Path, prompt_ids: tuple[str, ...], steps: tuple[int, ...]
) -> list[tuple[str, str, Path]]:
    paths = []
    for policy_id, prompt_id in sorted(target_groups(prompt_ids, steps)):
        stem = f"{policy_id}-{_shard_name(policy_id, prompt_id)}.jsonl"
        path = run_root / "select-bon-minimum" / "shards" / stem
        if not path.is_file():
            raise MinimumExperimentError(f"partial paper selection is incomplete: {path}")
        paths.append((policy_id, prompt_id, path))
    return paths


def run(args: argparse.Namespace) -> dict[str, Any]:
    steps = tuple(args.steps)
    prepare_paper_run(args.source_run_root, args.run_root)
    prompt_ids = ordered_prompt_subset(args.run_root, args.prompt_count)
    selection_paths = _selection_paths(args.run_root, prompt_ids, steps)
    approval_path = args.run_root / "paper-judge-gold-egress-approval.json"

    groups = []
    for policy_id, prompt_id, selection_path in selection_paths:
        manifest = prepare_gold_selection_shard(
            args.run_root,
            selection_path,
            args.private_gt,
            args.gold_schema,
            prompt_version=PAPER_JUDGE_PROMPT_VERSION,
        )
        receipt = submit_gold_selection_shard(
            args.run_root,
            selection_path,
            approval_path=approval_path,
        )
        groups.append(
            {
                "policy_id": policy_id,
                "prompt_id": prompt_id,
                "requests": manifest["requests"],
                "batch_id": receipt["batch_id"],
            }
        )
    _status(
        args.run_root,
        steps,
        "submitted",
        groups=len(groups),
        requests=sum(int(row["requests"]) for row in groups),
    )

    expected = len(groups)
    while True:
        synced = sync_streaming_gold(args.run_root, args.gold_schema)
        completed = target_gold_complete(args.run_root, prompt_ids, steps)
        _status(
            args.run_root,
            steps,
            "gold-sync",
            completed_target_groups=completed,
            expected_target_groups=expected,
            submitted_groups=synced["submitted_groups"],
        )
        if completed == expected or not args.wait:
            break
        time.sleep(args.poll_seconds)

    result: dict[str, Any] = {
        "prompt_ids": list(prompt_ids),
        "policy_steps": list(steps),
        "groups": groups,
        "completed_target_groups": completed,
    }
    if completed == expected:
        analysis = analyze_interim(args.run_root, prompt_ids, steps)
        result["analysis"] = analysis
        _status(
            args.run_root,
            steps,
            "complete",
            figure=analysis["figures"][0]["path"],
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-run-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--prompt-count", type=int, default=5)
    parser.add_argument("--steps", type=int, nargs="+", default=[3, 10])
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
    if args.prompt_count < 1 or args.poll_seconds < 10:
        parser.error("prompt-count must be positive and poll-seconds >= 10")
    result = run(args)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
