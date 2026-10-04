#!/usr/bin/env python3
"""Resumable supervisor for the minimum rubric-staleness experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any, Callable

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.minimum_analysis import analyze_minimum
from dynamic_rubric.minimum_gold import (
    prepare_gold_batch,
)
from dynamic_rubric.minimum_gold_streaming import (
    finalize_streaming_gold,
    stream_gold_selection_shard,
    submit_available_gold_shards,
    sync_streaming_gold,
)
from dynamic_rubric.minimum_staleness import (
    export_audit_package,
    replay_current_aligned,
    score_bon,
    select_bon,
    select_bon_shard,
)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _status(run_root: Path, stage: str, **details: Any) -> None:
    write_json_atomic(
        run_root / "minimum-pipeline-progress.json",
        {"stage": stage, "updated_at_unix": time.time(), **details},
        immutable=False,
    )
    print(json.dumps({"stage": stage, **details}, ensure_ascii=False, sort_keys=True), flush=True)


def _retry(
    run_root: Path,
    stage: str,
    operation: Callable[[], dict[str, Any]],
    *,
    delay_seconds: int,
    attempts: int = 24,
) -> dict[str, Any]:
    for attempt in range(1, attempts + 1):
        _status(run_root, stage, attempt=attempt)
        try:
            result = operation()
        except Exception as error:
            _status(run_root, f"{stage}-retry", attempt=attempt, error=repr(error))
            if attempt == attempts:
                raise
            time.sleep(delay_seconds)
        else:
            _status(run_root, f"{stage}-complete", result=result)
            return result
    raise AssertionError("retry loop exhausted without returning")


def run(args: argparse.Namespace) -> dict[str, Any]:
    run_root: Path = args.run_root
    if args.wait_pid and _pid_exists(args.wait_pid):
        _status(run_root, "waiting-for-active-replay", pid=args.wait_pid)
        while _pid_exists(args.wait_pid):
            time.sleep(args.poll_seconds)
    replay_result_path = run_root / "replay-dynamic-minimum" / "result.json"
    if replay_result_path.is_file():
        _status(
            run_root,
            "replay-already-complete",
            result=read_json(replay_result_path),
        )
    else:
        _retry(
            run_root,
            "replay",
            lambda: replay_current_aligned(
                run_root,
                args.score_endpoint,
                args.embedding_model_path,
                embedding_device=args.embedding_device,
                workers=args.workers,
            ),
            delay_seconds=args.retry_seconds,
        )
    def on_score_shard(
        policy_id: str, prompt_id: str, candidates: list[dict[str, Any]]
    ) -> None:
        selection_path = select_bon_shard(
            run_root, policy_id, prompt_id, candidates
        )
        if args.submit_gold:
            streamed = stream_gold_selection_shard(
                run_root, selection_path, args.private_gt, args.gold_schema
            )
            _status(
                run_root,
                "stream-gold-submitted",
                group=streamed["submission"]["group"],
                batch_id=streamed["submission"]["batch_id"],
                requests=streamed["manifest"]["requests"],
            )

    _retry(
        run_root,
        "score-bon",
        lambda: score_bon(
            run_root,
            args.score_endpoint,
            workers=args.workers,
            on_shard=on_score_shard,
        ),
        delay_seconds=args.retry_seconds,
    )
    _retry(run_root, "select", lambda: select_bon(run_root), delay_seconds=args.retry_seconds)
    _retry(
        run_root,
        "export-audit",
        lambda: export_audit_package(run_root),
        delay_seconds=args.retry_seconds,
    )
    if not args.submit_gold:
        _retry(
            run_root,
            "prepare-gold",
            lambda: prepare_gold_batch(run_root, args.private_gt, args.gold_schema),
            delay_seconds=args.retry_seconds,
        )
        result = {
            "status": "awaiting-gold-egress-approval",
            "prepared_manifest": str(run_root / "audit-gold-minimum-private" / "manifest.json"),
        }
        _status(run_root, "awaiting-gold-egress-approval", result=result)
        return result
    submission = _retry(
        run_root,
        "stream-gold-catch-up",
        lambda: submit_available_gold_shards(
            run_root, args.private_gt, args.gold_schema
        ),
        delay_seconds=args.retry_seconds,
    )
    _status(
        run_root,
        "waiting-for-streaming-gold-batches",
        jobs=submission["submitted_groups"],
    )
    while True:
        current = sync_streaming_gold(run_root, args.gold_schema)
        _status(
            run_root,
            "streaming-gold-batch-status",
            submitted_groups=current["submitted_groups"],
            completed_groups=current["completed_groups"],
        )
        if current["all_completed"]:
            break
        time.sleep(args.poll_seconds)
    _retry(
        run_root,
        "finalize-streaming-gold",
        lambda: finalize_streaming_gold(run_root, args.gold_schema),
        delay_seconds=args.retry_seconds,
    )
    result = _retry(
        run_root,
        "analyze",
        lambda: analyze_minimum(run_root),
        delay_seconds=args.retry_seconds,
    )
    _status(run_root, "complete", result=result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--wait-pid", type=int, default=0)
    parser.add_argument("--score-endpoint", default="http://127.0.0.1:8102")
    parser.add_argument(
        "--embedding-model-path",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "models/bge-m3",
    )
    parser.add_argument("--embedding-device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument(
        "--submit-gold",
        action="store_true",
        help="upload selected medical responses and private physician rubrics to OpenAI Batch",
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
    if args.poll_seconds < 10 or args.retry_seconds < 10:
        parser.error("poll/retry intervals must be at least 10 seconds")
    print(json.dumps(run(args), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
