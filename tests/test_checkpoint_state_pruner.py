from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_pruner_module():
    path = Path(__file__).parents[1] / "scripts" / "prune_horizon_checkpoint_state.py"
    spec = importlib.util.spec_from_file_location("checkpoint_state_pruner", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkpoint(root: Path, step: int) -> Path:
    checkpoint = root / f"global_step_{step}"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"model")
    (actor / "optim_world_size_1_rank_0.pt").write_bytes(b"optimizer")
    (actor / "extra_state_world_size_1_rank_0.pt").write_bytes(b"extra")
    (checkpoint / "data.pt").write_bytes(b"data")
    return checkpoint


def test_pruner_keeps_only_latest_checkpoint_resumable(tmp_path: Path) -> None:
    module = _load_pruner_module()
    older = _checkpoint(tmp_path, 3)
    latest = _checkpoint(tmp_path, 6)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6", encoding="utf-8")

    removed = module.prune_stale_training_state(tmp_path)

    assert removed
    assert (older / "actor/model_world_size_1_rank_0.pt").is_file()
    assert not (older / "actor/optim_world_size_1_rank_0.pt").exists()
    assert not (older / "actor/extra_state_world_size_1_rank_0.pt").exists()
    assert not (older / "data.pt").exists()
    assert (latest / "actor/model_world_size_1_rank_0.pt").is_file()
    assert (latest / "actor/optim_world_size_1_rank_0.pt").is_file()
    assert (latest / "actor/extra_state_world_size_1_rank_0.pt").is_file()
    assert (latest / "data.pt").is_file()


def test_pruner_waits_for_committed_tracker(tmp_path: Path) -> None:
    module = _load_pruner_module()
    checkpoint = _checkpoint(tmp_path, 3)

    assert module.prune_stale_training_state(tmp_path) == ()
    assert (checkpoint / "actor/optim_world_size_1_rank_0.pt").is_file()


def test_pruner_does_not_touch_checkpoint_ahead_of_tracker(tmp_path: Path) -> None:
    module = _load_pruner_module()
    latest = _checkpoint(tmp_path, 6)
    in_progress = _checkpoint(tmp_path, 9)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("6", encoding="utf-8")

    assert module.prune_stale_training_state(tmp_path) == ()
    assert (latest / "actor/optim_world_size_1_rank_0.pt").is_file()
    assert (in_progress / "actor/optim_world_size_1_rank_0.pt").is_file()
    assert (in_progress / "actor/extra_state_world_size_1_rank_0.pt").is_file()
    assert (in_progress / "data.pt").is_file()


def test_pruner_removes_superseded_recovery_checkpoint(tmp_path: Path) -> None:
    module = _load_pruner_module()
    audit = _checkpoint(tmp_path, 39)
    recovery = _checkpoint(tmp_path, 40)
    latest = _checkpoint(tmp_path, 41)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("41", encoding="utf-8")

    removed = module.prune_stale_training_state(
        tmp_path,
        retained_parameter_steps=frozenset({39, 42, 45, 48}),
    )

    assert recovery in removed
    assert not recovery.exists()
    assert (audit / "actor/model_world_size_1_rank_0.pt").is_file()
    assert not (audit / "actor/optim_world_size_1_rank_0.pt").exists()
    assert (latest / "actor/optim_world_size_1_rank_0.pt").is_file()


def test_retained_recovery_step_becomes_parameter_only(tmp_path: Path) -> None:
    module = _load_pruner_module()
    retained = _checkpoint(tmp_path, 42)
    latest = _checkpoint(tmp_path, 43)
    (tmp_path / "latest_checkpointed_iteration.txt").write_text("43", encoding="utf-8")

    module.prune_stale_training_state(
        tmp_path,
        retained_parameter_steps=frozenset({42, 45, 48}),
    )

    assert (retained / "actor/model_world_size_1_rank_0.pt").is_file()
    assert not (retained / "actor/optim_world_size_1_rank_0.pt").exists()
    assert not (retained / "actor/extra_state_world_size_1_rank_0.pt").exists()
    assert not (retained / "data.pt").exists()
    assert (latest / "actor/optim_world_size_1_rank_0.pt").is_file()
