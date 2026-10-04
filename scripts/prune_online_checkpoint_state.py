#!/usr/bin/env python3
"""Retain resume state only in the latest dynamic-OnlineRubrics checkpoint."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from dynamic_rubric.training.checkpoint_retention import prune_stale_resume_state


def _pid_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _emit(checkpoint_root: Path, removed: tuple[Path, ...]) -> None:
    if removed:
        print(
            json.dumps(
                {
                    "checkpoint_root": str(checkpoint_root),
                    "event": "pruned_online_stale_resume_state",
                    "removed": [str(path) for path in removed],
                },
                sort_keys=True,
            ),
            flush=True,
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--while-pid", type=int)
    args = parser.parse_args()
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    if args.while_pid is not None and not args.watch:
        parser.error("--while-pid requires --watch")

    while True:
        _emit(args.checkpoint_root, prune_stale_resume_state(args.checkpoint_root))
        if not args.watch:
            return 0
        if args.while_pid is not None and not _pid_exists(args.while_pid):
            _emit(args.checkpoint_root, prune_stale_resume_state(args.checkpoint_root))
            return 0
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
