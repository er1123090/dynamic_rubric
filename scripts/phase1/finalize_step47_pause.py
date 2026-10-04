"""Finalize the live step-47 pause without replaying update 48.

This is an operator-run finalizer.  It performs no process control: the
save-and-hold wrapper is responsible for holding the controller and writing
the live actor snapshot.  The script only validates, publishes, and verifies
the already-written step-47 state.
"""
from copy import deepcopy
import dataclasses
import io
import os
from pathlib import Path

import pyarrow.parquet as pq
import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

from dynamic_rubric.artifacts import read_json, write_bytes_atomic, write_json_atomic, write_text_atomic
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.phase1.full_run import latest_full_checkpoint
from dynamic_rubric.training.dataloader_resume import restore_online_dataloader_state
from dynamic_rubric.training.live_online import (
    _actor_parameter_tree_hash,
    _checkpoint_tree_hash,
    _validate_manifest_file_hashes,
    resolve_committed_resume,
)
from dynamic_rubric.training.online_contracts import validate_online_step_manifest


RUN = Path(__file__).resolve().parents[2] / "outputs/medicine/online_rubrics/seed-11/phase1-online-rubrics-medicine-full-20260905-seed11-final"
VRUN = RUN / "verl-run"
CHECKPOINT = VRUN / "checkpoints/global_step_47"


def make_loader():
    rows = list(range(1500))
    return StatefulDataLoader(
        rows,
        sampler=RandomSampler(rows, generator=torch.Generator().manual_seed(11)),
        batch_size=96,
        drop_last=False,
        num_workers=0,
    )


def _proc_identity(pid: int) -> tuple[str, str, str]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    return Path(f"/proc/{pid}/comm").read_text().strip(), fields[19], fields[0]


def _normalize_recorded_name(value: object) -> str:
    """Normalize save_and_hold.identity's ``pid (comm`` prefix."""
    name = str(value)
    if "(" in name:
        name = name.rsplit("(", 1)[-1]
    return name.rstrip(")").strip()


def _guard_held_processes(stopped: dict) -> None:
    """Fail closed unless the exact recorded controller/supervisor are held."""
    recorded = stopped.get("pids") or stopped.get("group_members")
    if isinstance(recorded, dict):
        entries = []
        for pid, value in recorded.items():
            if isinstance(value, (list, tuple)):
                entries.append({"pid": pid, "name": value[0], "starttime": value[1]})
            else:
                entries.append({"pid": pid, **value})
    else:
        entries = list(recorded or [])
    if not entries:
        raise RuntimeError("step47 stop receipt has no process identities")
    for entry in entries:
        pid = int(entry["pid"])
        path = Path(f"/proc/{pid}/stat")
        if not path.exists():
            continue
        comm, starttime, state = _proc_identity(pid)
        expected_name = _normalize_recorded_name(entry.get("name", entry.get("comm", "")))
        expected_start = str(entry.get("starttime", entry.get("start_time", "")))
        if expected_start and starttime != expected_start:
            raise RuntimeError(f"PID identity changed: {pid}")
        if expected_name and expected_name not in comm:
            raise RuntimeError(f"PID name changed: {pid} ({comm!r})")
        if state not in {"T", "t", "Z", "X"}:
            raise RuntimeError(f"held process is running: {pid} ({state})")


def main() -> None:
    receipt = read_json(RUN / "logs/save-step47-live-receipt.json")
    stopped = read_json(RUN / "logs/stop-after-step47.json")
    _guard_held_processes(stopped)
    assert receipt["after"]["optimizer_steps"] == [47]
    assert receipt["after"]["scheduler_last_epoch"] == 47
    latest_path = VRUN / "latest_commit.json"
    latest = read_json(latest_path)
    assert latest["optimizer_update_index"] == 47 and not latest["checkpoint_saved"]
    assert not (VRUN / "online_steps/step-000048/pre_update_seal.json").exists()
    assert not (VRUN / "online_steps/step-000048/commit.json").exists()
    assert CHECKPOINT.is_dir()

    optimizer = torch.load(CHECKPOINT / "actor/optim_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
    extra = torch.load(CHECKPOINT / "actor/extra_state_world_size_1_rank_0.pt", map_location="cpu", weights_only=False)
    model = torch.load(CHECKPOINT / "actor/model_world_size_1_rank_0.pt", map_location="cpu", weights_only=False, mmap=True)
    assert sorted({int(s["step"].item()) for s in optimizer["state"].values() if "step" in s}) == [47]
    assert extra["lr_scheduler"]["last_epoch"] == 47
    assert {"cpu", "cuda", "numpy", "random"} <= extra["rng"].keys()
    assert model and all(isinstance(value, torch.Tensor) for value in model.values())

    ids = pq.read_table(RUN / "verl-data/train-online-full.parquet", columns=["prompt_occurrence_id"]).column(0).to_pylist()
    source = VRUN / "checkpoints/global_step_45/data.pt"
    source_hash = sha256_file(source)
    state45 = torch.load(source, map_location="cpu", weights_only=False)
    loader = make_loader()
    restore_online_dataloader_state(loader, state45, global_step=45)
    iterator = iter(loader)
    actual46 = [ids[index] for index in next(iterator).tolist()]
    actual47 = [ids[index] for index in next(iterator).tolist()]
    assert actual46 == read_json(VRUN / "online_steps/step-000046/batch.json")["prompt_occurrence_ids"]
    assert actual47 == read_json(VRUN / "online_steps/step-000047/batch.json")["prompt_occurrence_ids"]
    state47 = deepcopy(loader.state_dict())
    oracle = read_json(RUN / "logs/epoch3-resume-order-oracle.json")["expected_prompt_occurrence_ids"]
    resumed = make_loader()
    restore_online_dataloader_state(resumed, state47, global_step=47)
    next_batches = [[ids[index] for index in batch.tolist()] for batch in resumed]
    assert next_batches == [oracle[str(step)] for step in range(48, 49)]
    payload = io.BytesIO()
    torch.save(state47, payload)
    write_bytes_atomic(CHECKPOINT / "data.pt", payload.getvalue())
    reloaded = make_loader()
    restore_online_dataloader_state(reloaded, torch.load(CHECKPOINT / "data.pt", weights_only=False), global_step=47)
    assert [[ids[index] for index in batch.tolist()] for batch in reloaded] == next_batches

    for path in CHECKPOINT.rglob("*"):
        if path.is_file():
            fd = os.open(path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
    resume_hash = _checkpoint_tree_hash(CHECKPOINT)
    parameter_hash = _actor_parameter_tree_hash(CHECKPOINT / "actor")
    commit_path = VRUN / "online_steps/step-000047/commit.json"
    manifest = validate_online_step_manifest(commit_path)
    assert manifest.artifacts["logical_policy_token"] == receipt["logical_policy_token"]
    _validate_manifest_file_hashes(commit_path.parent, manifest.artifacts)
    _guard_held_processes(stopped)
    write_json_atomic(RUN / "logs/manual-checkpoint47/commit.logical-only.json", read_json(commit_path))
    write_json_atomic(RUN / "logs/manual-checkpoint47/latest.logical-only.json", latest)
    promoted = dataclasses.replace(manifest, artifacts={**manifest.artifacts, "resume_checkpoint_hash": resume_hash, "actor_parameter_hash": parameter_hash})
    new_latest = {**latest, "checkpoint_saved": True, "checkpoint": str(CHECKPOINT), "resume_checkpoint_hash": resume_hash, "actor_parameter_hash": parameter_hash, "manifest_hash": promoted.content_hash}
    write_json_atomic(commit_path, dataclasses.asdict(promoted), immutable=False)
    write_json_atomic(latest_path, new_latest, immutable=False)
    write_text_atomic(VRUN / "checkpoints/latest_checkpointed_iteration.txt", "47", immutable=False)
    assert latest_full_checkpoint(RUN) == CHECKPOINT

    # This validates every historical committed actor, including public archive receipts.
    resolved, step = resolve_committed_resume(VRUN)
    assert resolved == CHECKPOINT and step == 47
    assert not (VRUN / "online_steps/step-000048/pre_update_seal.json").exists()
    assert not (VRUN / "online_steps/step-000048/commit.json").exists()
    assert read_json(latest_path) == new_latest
    _guard_held_processes(stopped)
    write_json_atomic(RUN / "logs/finalize-step47-pause-receipt.json", {
        "state": "step47_sealed_update48_blocked",
        "checkpoint": str(CHECKPOINT),
        "resume_step": 47,
        "resume_checkpoint_hash": resume_hash,
        "actor_parameter_hash": parameter_hash,
        "source_dataloader_checkpoint": 45,
        "source_dataloader_sha256": source_hash,
        "verified_next_batches": [48],
        "archive_chain_verified": True,
    })


if __name__ == "__main__":
    main()
