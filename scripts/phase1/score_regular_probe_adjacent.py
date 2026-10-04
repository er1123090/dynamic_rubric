#!/usr/bin/env python3
"""Score fixed-train Pool B using adjacent defaults or an explicit cell plan."""

from __future__ import annotations

import argparse
import fcntl
from pathlib import Path
import time

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.probe_adjacent_scoring import score_regular_adjacent
from dynamic_rubric.providers.vllm_chat import VLLMChatError


def run_with_restarts(args):
    for attempt in range(args.max_client_restarts + 1):
        try:
            return score_regular_adjacent(
                run_dir=args.run_dir, artifact_root=args.artifact_root,
                output_root=args.output_root, steps=args.steps, judge_urls=args.judge_urls,
                concurrency=args.concurrency, wait_timeout_seconds=args.wait_timeout,
                cell_plan_path=args.cell_plan,
                bounded_grading_whitespace=args.bounded_grading_whitespace,
            )
        except Exception as error:
            retry = isinstance(error, VLLMChatError) and attempt < args.max_client_restarts
            path = args.output_root / 'status.json'
            status = read_json(path) if path.exists() else {}
            status.update(state='retrying' if retry else 'failed', error=repr(error),
                          client_restart_attempt=attempt, updated_at_epoch=time.time())
            write_json_atomic(path, status, immutable=False)
            if not retry:
                raise
            time.sleep(args.restart_delay)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "--cell-plan schema: JSON object with schema_version=1, increasing integer "
            "steps, and cells=[{evaluator_step: int, policy_step: int}, ...]. The exact "
            "listed cells are scored; no triangular expansion is performed."
        ),
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--steps",
        nargs="+",
        type=int,
        default=(),
        help="Regular checkpoints for adjacent mode, or exact plan steps for cross-checking.",
    )
    parser.add_argument(
        "--cell-plan",
        type=Path,
        help="Optional schema-v1 JSON plan; permits explicitly listed non-regular checkpoints.",
    )
    parser.add_argument("--judge-urls", required=True, nargs="+")
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--wait-timeout", type=float, default=0.0)
    parser.add_argument('--bounded-grading-whitespace', action='store_true',
                        help='Prevent whitespace-only token exhaustion using an equivalent binary-grade grammar.')
    parser.add_argument('--max-client-restarts', type=int, default=0)
    parser.add_argument('--restart-delay', type=float, default=30.0)
    args = parser.parse_args()
    if not args.steps and args.cell_plan is None:
        parser.error("--steps is required unless --cell-plan is supplied")
    if args.max_client_restarts < 0 or args.restart_delay < 0:
        parser.error('restart count and delay must be non-negative')
    args.output_root.mkdir(parents=True, exist_ok=True)
    lock_path = args.output_root / "scorer.lock"
    with lock_path.open("a", encoding="utf-8") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"another scorer owns {lock_path}") from error
        result = run_with_restarts(args)
    print(result)


if __name__ == "__main__":
    main()
