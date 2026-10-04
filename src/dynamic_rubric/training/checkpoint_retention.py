"""Safe retention of resumable state for audit checkpoints.

Model parameter shards (including adapter parameters) are audit artifacts and are
never removed.  Optimizer, scheduler/RNG, and dataloader state are resume-only;
they are retained only for veRL's latest sealed checkpoint.
"""

from __future__ import annotations

import os
from pathlib import Path


_RESUME_ONLY_PATTERNS = (
    "actor/optim_world_size_*_rank_*.pt",
    "actor/extra_state_world_size_*_rank_*.pt",
    "critic/optim_world_size_*_rank_*.pt",
    "critic/extra_state_world_size_*_rank_*.pt",
    "data.pt",
    "data_*.pt",
)


def latest_sealed_step(checkpoint_root: Path) -> int | None:
    """Return veRL's latest fully-written checkpoint step.

    veRL advances this tracker only after synchronously saving the checkpoint
    payload, so it is the retention commit marker rather than directory presence.
    """

    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    if not tracker.is_file():
        return None
    try:
        step = int(tracker.read_text(encoding="utf-8").strip())
    except ValueError as error:
        raise RuntimeError(f"malformed checkpoint tracker: {tracker}") from error
    if step < 0:
        raise RuntimeError(f"checkpoint tracker must be non-negative: {tracker}")
    return step


def _checkpoint_step(path: Path) -> int | None:
    try:
        return int(path.name.removeprefix("global_step_"))
    except ValueError:
        return None


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def prune_stale_resume_state(checkpoint_root: Path) -> tuple[Path, ...]:
    """Remove resume-only files from checkpoints older than the sealed latest.

    If a checkpoint directory exists ahead of the tracker, a new checkpoint is
    still being written.  In that case nothing is pruned, leaving the previous
    checkpoint fully resumable until its successor is sealed.
    """

    latest_step = latest_sealed_step(checkpoint_root)
    if latest_step is None:
        return ()
    latest = checkpoint_root / f"global_step_{latest_step}"
    if not latest.is_dir():
        raise RuntimeError(f"latest checkpoint directory is missing: {latest}")
    if not (latest / "data.pt").is_file():
        raise RuntimeError(f"latest checkpoint has no dataloader state: {latest}")

    checkpoints: list[tuple[int, Path]] = []
    for checkpoint in checkpoint_root.glob("global_step_*"):
        if not checkpoint.is_dir():
            continue
        step = _checkpoint_step(checkpoint)
        if step is not None:
            checkpoints.append((step, checkpoint))
    if any(step > latest_step for step, _ in checkpoints):
        return ()

    removed: list[Path] = []
    for step, checkpoint in sorted(checkpoints):
        if step >= latest_step:
            continue
        for pattern in _RESUME_ONLY_PATTERNS:
            for path in checkpoint.glob(pattern):
                if path.is_file():
                    path.unlink()
                    removed.append(path)
        _fsync_directory(checkpoint)
    return tuple(removed)
