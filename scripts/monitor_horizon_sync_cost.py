from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from dynamic_rubric.horizon.sync_rubrics import _write_cost_summary


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuously aggregate sync Responses cost")
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--watch-pid", type=int, required=True)
    parser.add_argument("--interval-seconds", type=float, default=10.0)
    args = parser.parse_args()
    if args.interval_seconds <= 0:
        raise ValueError("interval-seconds must be positive")
    while True:
        summary = _write_cost_summary(args.state_root)
        print(
            json.dumps(
                {
                    "updated_at": summary["updated_at"],
                    "attempt_count": summary["attempt_count"],
                    "completed_attempts": summary["completed_attempts"],
                    "incomplete_attempts": summary["incomplete_attempts"],
                    "estimated_cost_usd": summary["cost"]["estimated_cost_usd"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not _alive(args.watch_pid):
            break
        time.sleep(args.interval_seconds)


if __name__ == "__main__":
    main()
