#!/usr/bin/env python3
"""Keep resumable training state only in the latest horizon checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def _latest_step(checkpoint_root: Path) -> int | None:
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    try:
        return int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise RuntimeError(f"malformed checkpoint tracker: {tracker}") from error


def prune_stale_training_state(
    checkpoint_root: Path,
    *,
    retained_parameter_steps: frozenset[int] | None = None,
) -> tuple[Path, ...]:
    """Keep the latest checkpoint resumable and retain only selected old models.

    veRL updates ``latest_checkpointed_iteration.txt`` only after the checkpoint
    payload has been saved. Treating that tracker as the commit marker ensures
    that the previous checkpoint remains resumable until its successor is
    complete. Older checkpoints outside ``retained_parameter_steps`` are
    transient recovery checkpoints and are removed completely after a newer
    checkpoint is sealed.
    """

    latest_step = _latest_step(checkpoint_root)
    if latest_step is None:
        return ()
    latest = checkpoint_root / f"global_step_{latest_step}"
    if not latest.is_dir():
        raise RuntimeError(f"latest checkpoint directory is missing: {latest}")

    removed: list[Path] = []
    for checkpoint in sorted(checkpoint_root.glob("global_step_*")):
        if not checkpoint.is_dir() or checkpoint == latest:
            continue
        try:
            checkpoint_step = int(checkpoint.name.removeprefix("global_step_"))
        except ValueError:
            continue
        # A checkpoint newer than the tracker is still being written. veRL
        # writes optimizer state before advancing the tracker, so pruning it
        # here would make the new checkpoint non-resumable before it is sealed.
        if checkpoint_step >= latest_step:
            continue
        if (
            retained_parameter_steps is not None
            and checkpoint_step not in retained_parameter_steps
        ):
            shutil.rmtree(checkpoint)
            removed.append(checkpoint)
            continue

        candidates = [
            *checkpoint.glob("actor/optim_world_size_*_rank_*.pt"),
            *checkpoint.glob("actor/extra_state_world_size_*_rank_*.pt"),
            *checkpoint.glob("critic/optim_world_size_*_rank_*.pt"),
            *checkpoint.glob("critic/extra_state_world_size_*_rank_*.pt"),
            *checkpoint.glob("data.pt"),
            *checkpoint.glob("data_*.pt"),
        ]
        for path in candidates:
            if path.is_file():
                path.unlink()
                removed.append(path)
    return tuple(removed)


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _emit(checkpoint_root: Path, removed: tuple[Path, ...]) -> None:
    if not removed:
        return
    print(
        json.dumps(
            {
                "checkpoint_root": str(checkpoint_root),
                "event": "pruned_stale_training_state",
                "removed": [str(path) for path in removed],
            },
            sort_keys=True,
        ),
        flush=True,
    )


def _parse_retained_steps(raw: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    try:
        steps = frozenset(int(value.strip()) for value in raw.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "--retain-parameter-steps must be a comma-separated integer list"
        ) from error
    if any(step < 0 for step in steps):
        raise argparse.ArgumentTypeError(
            "--retain-parameter-steps must contain only non-negative steps"
        )
    return steps


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--while-pid", type=int)
    parser.add_argument(
        "--retain-parameter-steps",
        type=_parse_retained_steps,
        default=None,
        help="comma-separated audit steps whose model parameters must be retained",
    )
    args = parser.parse_args()

    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.while_pid is not None and not args.watch:
        parser.error("--while-pid requires --watch")

    while True:
        _emit(
            args.checkpoint_root,
            prune_stale_training_state(
                args.checkpoint_root,
                retained_parameter_steps=args.retain_parameter_steps,
            ),
        )
        if not args.watch:
            return 0
        if args.while_pid is not None and not _pid_exists(args.while_pid):
            _emit(
                args.checkpoint_root,
                prune_stale_training_state(
                    args.checkpoint_root,
                    retained_parameter_steps=args.retain_parameter_steps,
                ),
            )
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
