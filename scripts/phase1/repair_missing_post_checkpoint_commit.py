#!/usr/bin/env python3
"""Recover an OnlineRubrics commit interrupted after veRL sealed its checkpoint."""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.training.live_online import resolve_committed_resume
from dynamic_rubric.training.online_contracts import (
    StepState,
    manifest_from_mapping,
)
from dynamic_rubric.training.verl_online_runtime import (
    _actor_parameter_hash,
    _checkpoint_hash,
)


class RepairError(RuntimeError):
    pass


def repair(run_root: Path, step: int, *, apply: bool) -> dict[str, object]:
    run_root = run_root.resolve()
    online_root = run_root / "online_steps"
    checkpoint_root = run_root / "checkpoints"
    tracker = checkpoint_root / "latest_checkpointed_iteration.txt"
    step_root = online_root / f"step-{step:06d}"
    checkpoint = checkpoint_root / f"global_step_{step}"
    commit_path = step_root / "commit.json"
    latest_path = run_root / "latest_commit.json"

    if commit_path.exists():
        raise RepairError(f"refusing to replace existing commit: {commit_path}")
    try:
        tracked_step = int(tracker.read_text(encoding="utf-8").strip())
    except (OSError, ValueError) as error:
        raise RepairError(f"invalid checkpoint tracker: {tracker}") from error
    if tracked_step != step:
        raise RepairError(f"tracker step {tracked_step} does not equal repair step {step}")

    previous = read_json(latest_path)
    if int(previous.get("optimizer_update_index", -1)) != step - 1:
        raise RepairError("latest commit is not the immediately previous step")
    previous_commit_path = online_root / f"step-{step - 1:06d}" / "commit.json"
    previous_commit = read_json(previous_commit_path)
    if sha256_json(previous_commit) != str(previous.get("manifest_hash", "")):
        raise RepairError("previous latest pointer does not match its commit manifest")
    previous_token = str(previous.get("logical_policy_token", ""))
    if previous_commit.get("artifacts", {}).get("logical_policy_token") != previous_token:
        raise RepairError("previous logical policy token is inconsistent")

    sealed = manifest_from_mapping(read_json(step_root / "pre_update_seal.json"))
    if sealed.state is not StepState.PRE_UPDATE_SEALED:
        raise RepairError("repair input is not pre-update sealed")
    if sealed.optimizer_update_index != step:
        raise RepairError("sealed manifest is bound to a different step")
    for name, expected_hash in sealed.artifacts.items():
        artifact = step_root / name
        if artifact.parent != step_root or not artifact.is_file():
            raise RepairError(f"sealed artifact is missing or unsafe: {name}")
        actual_hash = sha256_file(artifact)
        if actual_hash != expected_hash:
            raise RepairError(
                f"sealed artifact hash mismatch: {name}: {actual_hash} != {expected_hash}"
            )

    resume_hash = _checkpoint_hash(checkpoint)
    actor_hash = _actor_parameter_hash(checkpoint / "actor")
    logical_policy_token = sha256_json(
        {
            "kind": "logical_policy_version",
            "run_id": sealed.run_id,
            "optimizer_update_index": step,
            "pre_update_policy_token": previous_token,
            "reward_manifest_hash": sealed.content_hash,
        }
    )
    committed = dataclasses.replace(
        sealed,
        state=StepState.COMMITTED,
        artifacts={
            **dict(sealed.artifacts),
            "logical_policy_token": logical_policy_token,
            "resume_checkpoint_hash": resume_hash,
            "actor_parameter_hash": actor_hash,
        },
    )
    latest = {
        "schema_version": 1,
        "optimizer_update_index": step,
        "manifest_hash": committed.content_hash,
        "logical_policy_token": logical_policy_token,
        "checkpoint_saved": True,
        "checkpoint": str(checkpoint),
        "resume_checkpoint_hash": resume_hash,
        "actor_parameter_hash": actor_hash,
    }

    if apply:
        write_json_atomic(commit_path, dataclasses.asdict(committed))
        write_json_atomic(latest_path, latest, immutable=False)
        resolved_checkpoint, resolved_step = resolve_committed_resume(run_root)
        if resolved_checkpoint.resolve() != checkpoint.resolve() or resolved_step != step:
            raise RepairError("post-repair resume resolution disagrees with repaired checkpoint")

    return {
        "applied": apply,
        "step": step,
        "checkpoint": str(checkpoint),
        "manifest_hash": committed.content_hash,
        "logical_policy_token": logical_policy_token,
        "resume_checkpoint_hash": resume_hash,
        "actor_parameter_hash": actor_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    if args.step < 2:
        parser.error("repair step must be at least 2")
    print(json.dumps(repair(args.run_root, args.step, apply=args.apply), sort_keys=True))


if __name__ == "__main__":
    main()
