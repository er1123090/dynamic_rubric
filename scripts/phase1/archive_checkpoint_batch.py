#!/usr/bin/env python3
"""Sequential, resumable upload/cleanup queue for historical checkpoints."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import json
from pathlib import Path
from typing import Callable

from dynamic_rubric.artifacts import read_json, write_json_atomic
from scripts.phase1.archive_unused_checkpoints_hf import (
    ALLOWED_STEPS,
    archive,
    cleanup_verified,
)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def validate_cleanup_gate(path: Path, run: Path) -> dict:
    if path.is_symlink() or not path.is_file():
        raise ValueError("cleanup-ready file must be an existing regular file")
    gate = read_json(path)
    if (
        gate.get("schema_version") != 1
        or gate.get("state") != "cleanup_ready"
        or gate.get("run_id") != run.resolve().name
        or gate.get("protected_resume_step") != 45
        or gate.get("graders_use_archived_checkpoint_identity") is not True
    ):
        raise ValueError("cleanup-ready file does not authorize this run's historical cleanup")
    authorized = gate.get("authorized_steps")
    if not isinstance(authorized, list) or authorized != list(ALLOWED_STEPS):
        raise ValueError("cleanup-ready file does not authorize the exact historical step set")
    return gate


def run_batch(
    *,
    run: Path,
    export_root: Path,
    steps: tuple[int, ...] = ALLOWED_STEPS,
    workers: int = 1,
    upload_only: bool = False,
    cleanup_ready_file: Path | None = None,
    queue_root: Path | None = None,
    archive_fn: Callable = archive,
    cleanup_fn: Callable = cleanup_verified,
) -> dict:
    run = run.resolve()
    export_root = export_root.resolve()
    if workers != 1:
        raise ValueError("checkpoint batch queue is intentionally limited to one worker")
    if len(steps) != len(set(steps)) or not set(steps) <= set(ALLOWED_STEPS):
        raise ValueError("batch steps must be unique members of ALLOWED_STEPS")
    if 45 in steps:
        raise ValueError("protected resume checkpoint 45 cannot enter the archive queue")
    gate = None
    if not upload_only:
        if cleanup_ready_file is None:
            raise ValueError("archive-and-cleanup mode requires --cleanup-ready-file")
        gate = validate_cleanup_gate(cleanup_ready_file, run)

    queue = (queue_root or run / "verl-run/checkpoint_archive_queue").resolve()
    queue.mkdir(parents=True, exist_ok=True)
    lock_handle = (queue / ".archive-queue.lock").open("a+", encoding="utf-8")
    try:
        fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        lock_handle.close()
        raise ValueError("another checkpoint archive queue worker is active") from error
    completed: list[int] = []
    for step in steps:
        state_path = queue / f"global_step_{step}.json"
        previous = read_json(state_path) if state_path.is_file() else {}
        if previous.get("state") == "complete" or (
            upload_only and previous.get("state") == "archive_verified"
        ):
            completed.append(step)
            continue
        attempt = int(previous.get("attempt", 0)) + 1
        receipt_path = run / "verl-run/checkpoint_archives" / f"global_step_{step}.json"
        resume_verified = previous.get("state") == "archive_verified" and receipt_path.is_file()
        state = {
            "schema_version": 1,
            "state": "cleanup_resuming" if resume_verified else "archiving",
            "step": step,
            "attempt": attempt,
            "run_id": run.name,
            "export_root": str(export_root),
            "updated_at": _now(),
        }
        write_json_atomic(state_path, state, immutable=False)
        write_json_atomic(
            queue / "batch-status.json",
            {
                "schema_version": 1,
                "state": state["state"],
                "completed_steps": completed,
                "active_step": step,
                "pending_steps": [value for value in steps if value not in completed and value != step],
                "updated_at": _now(),
            },
            immutable=False,
        )
        try:
            if resume_verified:
                receipt = read_json(receipt_path)
            else:
                receipt = archive_fn(
                    run,
                    step,
                    export_root=export_root,
                    upload=True,
                    workers=workers,
                )
            state.update(
                state="archive_verified",
                repo_id=receipt["repo_id"],
                revision=receipt["revision"],
                updated_at=_now(),
            )
            write_json_atomic(state_path, state, immutable=False)
            if not upload_only:
                # Revalidate the gate immediately before each destructive cleanup.
                validate_cleanup_gate(cleanup_ready_file, run)
                cleanup_fn(run, step)
                state.update(
                    state="complete",
                    cleanup_gate=str(cleanup_ready_file.resolve()),
                    cleanup_gate_authorization=gate,
                    updated_at=_now(),
                )
                write_json_atomic(state_path, state, immutable=False)
            completed.append(step)
        except Exception as error:
            state.update(state="failed", error=repr(error), updated_at=_now())
            write_json_atomic(state_path, state, immutable=False)
            summary = {
                "schema_version": 1,
                "state": "failed",
                "completed_steps": completed,
                "failed_step": step,
                "pending_steps": [
                    value for value in steps if value not in completed and value != step
                ],
                "updated_at": _now(),
            }
            write_json_atomic(queue / "batch-status.json", summary, immutable=False)
            raise

    summary = {
        "schema_version": 1,
        "state": "uploads_complete" if upload_only else "complete",
        "completed_steps": completed,
        "pending_steps": [],
        "updated_at": _now(),
    }
    write_json_atomic(queue / "batch-status.json", summary, immutable=False)
    print(json.dumps(summary), flush=True)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--export-root", required=True, type=Path)
    parser.add_argument("--steps", nargs="+", type=int, default=list(ALLOWED_STEPS))
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--upload-only", action="store_true")
    parser.add_argument("--cleanup-ready-file", type=Path)
    parser.add_argument("--queue-root", type=Path)
    args = parser.parse_args()
    if args.upload_only and args.cleanup_ready_file is not None:
        parser.error("--upload-only cannot be combined with --cleanup-ready-file")
    if not args.upload_only and args.cleanup_ready_file is None:
        parser.error("archive-and-cleanup mode requires --cleanup-ready-file")
    run_batch(
        run=args.run,
        export_root=args.export_root,
        steps=tuple(args.steps),
        workers=args.workers,
        upload_only=args.upload_only,
        cleanup_ready_file=args.cleanup_ready_file,
        queue_root=args.queue_root,
    )


if __name__ == "__main__":
    main()
