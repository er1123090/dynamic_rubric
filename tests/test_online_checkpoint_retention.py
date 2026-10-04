from __future__ import annotations

from pathlib import Path

import pytest

from dynamic_rubric.training.checkpoint_retention import prune_stale_resume_state
from dynamic_rubric.training.verl_online_runtime import _actor_parameter_hash


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    adapter = actor / "adapter"
    adapter.mkdir(parents=True)
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"model")
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
    (actor / "optim_world_size_1_rank_0.pt").write_bytes(b"optimizer")
    (actor / "extra_state_world_size_1_rank_0.pt").write_bytes(b"extra")
    (checkpoint / "data.pt").write_bytes(b"dataloader")
    return checkpoint


def test_only_latest_online_checkpoint_remains_resumable(tmp_path: Path) -> None:
    older = _checkpoint(tmp_path, 3)
    latest = _checkpoint(tmp_path, 6)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6", encoding="utf-8")

    older_parameter_hash = _actor_parameter_hash(older / "actor")
    removed = prune_stale_resume_state(tmp_path)

    assert removed
    assert _actor_parameter_hash(older / "actor") == older_parameter_hash
    assert (older / "actor/model_world_size_1_rank_0.pt").is_file()
    assert (older / "actor/adapter/adapter_model.safetensors").is_file()
    assert not (older / "actor/optim_world_size_1_rank_0.pt").exists()
    assert not (older / "actor/extra_state_world_size_1_rank_0.pt").exists()
    assert not (older / "data.pt").exists()
    assert (latest / "actor/model_world_size_1_rank_0.pt").is_file()
    assert (latest / "actor/adapter/adapter_model.safetensors").is_file()
    assert (latest / "actor/optim_world_size_1_rank_0.pt").is_file()
    assert (latest / "actor/extra_state_world_size_1_rank_0.pt").is_file()
    assert (latest / "data.pt").is_file()


def test_retention_waits_while_successor_is_being_written(tmp_path: Path) -> None:
    sealed = _checkpoint(tmp_path, 3)
    in_progress = _checkpoint(tmp_path, 6)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")

    assert prune_stale_resume_state(tmp_path) == ()
    assert (sealed / "actor/optim_world_size_1_rank_0.pt").is_file()
    assert (in_progress / "actor/optim_world_size_1_rank_0.pt").is_file()


def test_retention_fails_closed_if_latest_checkpoint_is_not_full(tmp_path: Path) -> None:
    latest = _checkpoint(tmp_path, 3)
    (latest / "data.pt").unlink()
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("3", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no dataloader state"):
        prune_stale_resume_state(tmp_path)
