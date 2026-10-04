#!/usr/bin/env python3
"""Run a balanced prompt subset across all focal policies, then plot N-curves."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.minimum_gold_streaming import (
    stream_gold_selection_shard,
    sync_streaming_gold,
)
from dynamic_rubric.minimum_interim import (
    analyze_interim,
    ordered_prompt_subset,
    target_gold_complete,
    target_groups,
)
from dynamic_rubric.minimum_staleness import (
    FOCAL_STEPS,
    N_GRID,
    PERMUTATIONS,
    _shard_name,
    score_bon,
    select_bon_shard,
)


def _status(stage_root: Path, stage: str, **details: Any) -> None:
    value = {"stage": stage, "updated_at_unix": time.time(), **details}
    write_json_atomic(stage_root / "progress.json", value, immutable=False)
    print(json.dumps(value, ensure_ascii=False, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root: Path = args.run_root
    prompt_ids = ordered_prompt_subset(run_root, args.prompt_count)
    targets = target_groups(prompt_ids)
    stage_root = run_root / f"interim-bon-{args.prompt_count}prompt"
    write_json_atomic(
        stage_root / "manifest.json",
        {
            "schema_version": 1,
            "diagnostic_only": True,
            "prompt_selection": "first-N-in-immutable-pi_3-BoN-order",
            "prompt_ids": list(prompt_ids),
            "target_groups": [list(value) for value in sorted(targets)],
            "focal_steps": list(FOCAL_STEPS),
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "bon_pool_sha256": sha256_file(run_root / "generate-bon" / "bon_pool.jsonl"),
        },
    )

    def on_score_shard(policy_id: str, prompt_id: str, candidates: list[dict[str, Any]]) -> None:
        selection_path = select_bon_shard(run_root, policy_id, prompt_id, candidates)
        streamed = stream_gold_selection_shard(
            run_root, selection_path, args.private_gt, args.gold_schema
        )
        _status(
            stage_root,
            "stream-gold-submitted",
            policy_id=policy_id,
            prompt_id=prompt_id,
            completed_score_groups=sum(
                (
                    run_root
                    / "score-proxy-minimum"
                    / "shards"
                    / f"{key[0]}-{_shard_name(*key)}.jsonl"
                ).is_file()
                for key in targets
            ),
            batch_id=streamed["submission"]["batch_id"],
            requests=streamed["manifest"]["requests"],
        )

    score_result: dict[str, Any] | None = None
    for attempt in range(1, 25):
        try:
            score_result = score_bon(
                run_root,
                args.score_endpoint,
                workers=args.workers,
                on_shard=on_score_shard,
                target_groups=targets,
                progress_filename=f"interim-{args.prompt_count}prompt-progress.json",
            )
            break
        except Exception as error:
            _status(stage_root, "score-retry", attempt=attempt, error=repr(error))
            if attempt == 24:
                raise
            time.sleep(args.retry_seconds)
    if score_result is None:
        raise AssertionError("targeted score retry loop exited without a result")
    _status(stage_root, "score-complete", result=score_result)

    while True:
        try:
            synced = sync_streaming_gold(run_root, args.gold_schema)
        except Exception as error:
            _status(stage_root, "gold-sync-retry", error=repr(error))
            time.sleep(args.retry_seconds)
            continue
        completed = target_gold_complete(run_root, prompt_ids)
        _status(
            stage_root,
            "waiting-for-target-gold",
            completed_target_groups=completed,
            expected_target_groups=len(targets),
            all_submitted_groups=synced["submitted_groups"],
            all_collected_groups=synced["completed_groups"],
        )
        if completed == len(targets):
            break
        time.sleep(args.poll_seconds)
    result = analyze_interim(run_root, prompt_ids)
    _status(
        stage_root,
        "complete",
        result_path=str(stage_root / "result.json"),
        figure_path=str(stage_root / "n_curves.png"),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--prompt-count", type=int, default=10)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8103")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=300)
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
    if args.prompt_count < 1 or args.poll_seconds < 10 or args.retry_seconds < 10:
        parser.error("prompt count must be positive and poll/retry intervals at least 10 seconds")
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
