"""Restore online training's shuffled prompt stream without restarting its seed."""
from __future__ import annotations

from copy import deepcopy
from typing import Any


def restore_online_dataloader_state(dataloader: Any, state: dict, *, global_step: int) -> None:
    steps_per_epoch = len(dataloader)
    at_boundary = global_step > 0 and steps_per_epoch > 0 and global_step % steps_per_epoch == 0
    if at_boundary:
        # Online checkpoints are captured inside the last batch's loop body,
        # before StopIteration has advanced the loader to its next epoch. This
        # bounded path is validated for the run's single-process loader only.
        if dataloader.num_workers != 0:
            raise RuntimeError("Online epoch-boundary restore requires the validated num_workers=0 path")
        if state.get("_num_yielded") != steps_per_epoch or state.get("_iterator_finished") is not False:
            raise RuntimeError("Online checkpoint is not the expected final-batch dataloader snapshot")
    dataloader.load_state_dict(deepcopy(state))
    if at_boundary:
        # Consume ONLY the pending StopIteration. Restoring then exhausting the
        # old iterator keeps its sampler generator state; dropping the state
        # instead silently repeats the initial epoch's prompt permutation.
        try:
            next(iter(dataloader))
        except StopIteration:
            pass
        else:
            raise RuntimeError("Epoch-boundary restore unexpectedly yielded a prompt batch")
