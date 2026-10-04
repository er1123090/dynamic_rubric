"""Map preregistered epoch progress to unique observed trainer checkpoints."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ObservedCheckpoint:
    global_step: int
    epoch_fraction: float
    processed_prompts: int
    checkpoint_hash: str


@dataclass(frozen=True, slots=True)
class ScheduledCheckpoint:
    target_epoch: float
    global_step: int
    observed_epoch: float
    processed_prompts: int
    checkpoint_hash: str


def map_checkpoint_schedule(
    targets: Sequence[float], observed: Iterable[ObservedCheckpoint]
) -> tuple[ScheduledCheckpoint, ...]:
    checkpoints = tuple(observed)
    if not targets or not checkpoints:
        raise ValueError("targets and observed checkpoints must be non-empty")
    if len(targets) > len(checkpoints):
        raise ValueError("not enough unique observed checkpoints for target schedule")
    if any(target < 0 for target in targets):
        raise ValueError("target epochs must be non-negative")
    if len({item.global_step for item in checkpoints}) != len(checkpoints):
        raise ValueError("observed global steps must be unique")
    remaining = set(range(len(checkpoints)))
    schedule: list[ScheduledCheckpoint] = []
    for target in targets:
        chosen = min(
            remaining,
            key=lambda index: (
                abs(checkpoints[index].epoch_fraction - target),
                checkpoints[index].global_step,
            ),
        )
        remaining.remove(chosen)
        item = checkpoints[chosen]
        schedule.append(
            ScheduledCheckpoint(
                target_epoch=float(target),
                global_step=item.global_step,
                observed_epoch=item.epoch_fraction,
                processed_prompts=item.processed_prompts,
                checkpoint_hash=item.checkpoint_hash,
            )
        )
    schedule.sort(key=lambda item: item.target_epoch)
    if [item.global_step for item in schedule] != sorted(item.global_step for item in schedule):
        raise ValueError("nearest unique checkpoint mapping is not monotonic")
    return tuple(schedule)


def checkpoint_schedule_json(
    schedule: Sequence[ScheduledCheckpoint], *, training_seed: int, domain: str
) -> Mapping[str, object]:
    return {
        "schema_version": 1,
        "domain": domain,
        "training_seed": training_seed,
        "checkpoints": [asdict(item) for item in schedule],
    }
