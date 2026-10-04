from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
SPEC = spec_from_file_location("pause_committed", ROOT / "scripts/phase1/pause_at_committed_checkpoint.py")
MODULE = module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def checkpoint(tmp_path, *, step=32, saved=True):
    root = tmp_path / "run"
    ckpt = root / f"checkpoints/global_step_{step}"
    for name in ("actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
                 "actor/extra_state_world_size_1_rank_0.pt", "actor/huggingface/config.json", "data.pt"):
        path = ckpt / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"test")
    (root / "checkpoints/latest_checkpointed_iteration.txt").write_text(str(step))
    record = {"optimizer_update_index": step, "checkpoint_saved": saved,
              "checkpoint": str(ckpt), "resume_checkpoint_hash": "digest"}
    (root / "latest_commit.json").write_text(json.dumps(record))
    return root, record


def test_waits_for_exact_full_commit(tmp_path):
    root, record = checkpoint(tmp_path, step=31, saved=False)
    assert MODULE.committed_checkpoint(root, 32) is None
    assert MODULE.committed_checkpoint(root, 31) is None


def test_signal_success_is_not_mistaken_for_stopped_process(monkeypatch):
    monkeypatch.setattr(MODULE, "process_state", lambda pid: "S")
    monkeypatch.setattr(MODULE.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="not retained"):
        MODULE.require_stopped([123])
    monkeypatch.setattr(MODULE, "process_state", lambda pid: "T")
    MODULE.require_stopped([123])


def test_accepts_only_matching_complete_checkpoint(tmp_path):
    root, record = checkpoint(tmp_path)
    assert MODULE.committed_checkpoint(root, 32) == record
    (root / "checkpoints/latest_checkpointed_iteration.txt").write_text("30")
    with pytest.raises(RuntimeError, match="disagree"):
        MODULE.committed_checkpoint(root, 32)


def test_missing_optimizer_fails_closed(tmp_path):
    root, record = checkpoint(tmp_path)
    (Path(record["checkpoint"]) / "actor/optim_world_size_1_rank_0.pt").unlink()
    with pytest.raises(RuntimeError, match="lacks actor/optim"):
        MODULE.committed_checkpoint(root, 32)


def test_passed_target_never_pauses_later_step(tmp_path):
    root, _ = checkpoint(tmp_path, step=33)
    with pytest.raises(RuntimeError, match="passed"):
        MODULE.committed_checkpoint(root, 32)
