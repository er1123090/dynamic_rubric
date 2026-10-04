"""Hold this run's controller after a fully committed checkpoint, without replay.

This is an operator tool, not a training hook. It never changes model servers or
checkpoint files. The stopped processes stay stopped for explicit verification
and restart; it does not kill a Ray cluster or signal a process group.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import time

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.training.live_online import _checkpoint_tree_hash


def process_identity(pid: int, expected_name: str) -> str:
    stat = Path(f"/proc/{pid}/stat").read_text()
    if expected_name not in stat.split(")", 1)[0]:
        raise RuntimeError(f"Unexpected identity for PID {pid}")
    return stat.rsplit(")", 1)[1].split()[19]  # starttime, field 22


def process_state(pid: int) -> str:
    return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]


def require_stopped(pids: list[int]) -> None:
    for _ in range(20):
        if all(process_state(pid) in {"T", "t"} for pid in pids):
            return
        time.sleep(0.05)
    raise RuntimeError("SIGSTOP was not retained; refuse to claim a verified hold")


def committed_checkpoint(root: Path, step: int) -> dict | None:
    latest = root / "latest_commit.json"
    tracker = root / "checkpoints/latest_checkpointed_iteration.txt"
    if not latest.is_file() or not tracker.is_file():
        return None
    record = json.loads(latest.read_text())
    committed_step = int(record["optimizer_update_index"])
    if committed_step > step:
        raise RuntimeError("Target checkpoint was passed; refusing to pause a different step")
    if committed_step != step or not record.get("checkpoint_saved"):
        return None
    checkpoint = root / f"checkpoints/global_step_{step}"
    if int(tracker.read_text()) != step or Path(record["checkpoint"]).resolve() != checkpoint.resolve():
        raise RuntimeError("Checkpoint tracker and logical commit disagree")
    for name in ("actor/model_world_size_1_rank_0.pt", "actor/optim_world_size_1_rank_0.pt",
                 "actor/extra_state_world_size_1_rank_0.pt", "actor/huggingface/config.json", "data.pt"):
        if not (checkpoint / name).is_file() or not (checkpoint / name).stat().st_size:
            raise RuntimeError(f"Committed checkpoint lacks {name}")
    if not record.get("resume_checkpoint_hash"):
        raise RuntimeError("Committed checkpoint lacks its full integrity hash")
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--controller-pid", type=int, required=True)
    parser.add_argument("--supervisor-pid", type=int, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--wait-for-rollout", action="store_true")
    args = parser.parse_args()
    targets = [(args.controller_pid, "TaskRunner"), (args.supervisor_pid, "bash")]
    identities = {pid: process_identity(pid, name) for pid, name in targets}
    print(f"Waiting for full committed checkpoint {args.step}; no training changes", flush=True)
    while True:
        for pid, name in targets:
            if process_identity(pid, name) != identities[pid]:
                raise RuntimeError("PID reused while waiting; refusing signal")
        record = committed_checkpoint(args.run_root, args.step)
        if record is not None:
            if not args.wait_for_rollout:
                break
            # Preserve the existing post-checkpoint rollout audit dump too. A
            # directory or partially written JSONL is not a completion marker.
            dump = args.run_root / f"rollouts/{args.step}.jsonl"
            if dump.is_file():
                payload = dump.read_bytes()
                expected = sum(1 for _ in (args.run_root / f"online_steps/step-{args.step:06d}/current_responses.jsonl").open())
                if payload.endswith(b"\n") and len(payload.splitlines()) == expected:
                    rows = [json.loads(line) for line in payload.splitlines()]
                    if all(row["step"] == args.step for row in rows):
                        break
        time.sleep(1)
    # External polling may allow a next rollout to start. Hold the controller
    # first, then prove below that no later reward seal (and hence no later
    # optimizer update in this serial trainer) was reached. Never infer this
    # safety property merely from an earlier observation of latest_commit.
    stopped = []
    try:
        for pid, name in targets:
            if process_identity(pid, name) != identities[pid]:
                raise RuntimeError("PID reused before signal")
            os.kill(pid, signal.SIGSTOP)
            stopped.append(pid)
        require_stopped(stopped)
        receipt = {"state": "held_pending_hash", "step": args.step, "commit": record,
                   "pids": identities, "stopped": stopped, "time": time.time()}
        write_json_atomic(args.receipt, receipt, immutable=False)
        print(f"Held supervisor/controller at checkpoint {args.step}; verifying full hash", flush=True)
        later = sorted(path for path in (args.run_root / "online_steps").glob("step-*")
                       if path.is_dir() and int(path.name.removeprefix("step-")) > args.step)
        if any((path / name).exists() for path in later for name in ("pre_update_seal.json", "commit.json")):
            raise RuntimeError("A later reward seal exists; cannot prove the requested optimizer boundary")
        receipt["uncommitted_later_directories"] = [str(path) for path in later]
        receipt["no_later_optimizer_update_verified"] = True
        actual = _checkpoint_tree_hash(Path(record["checkpoint"]))
        if actual != record["resume_checkpoint_hash"]:
            raise RuntimeError("Full checkpoint integrity verification failed; leaving processes held")
        if committed_checkpoint(args.run_root, args.step) != record:
            raise RuntimeError("Commit changed during pause verification")
        require_stopped(stopped)
        if any((path / name).exists()
               for path in (args.run_root / "online_steps").glob("step-*")
               if int(path.name.removeprefix("step-")) > args.step
               for name in ("pre_update_seal.json", "commit.json")):
            raise RuntimeError("A later reward seal appeared during checkpoint verification")
        receipt.update(state="held_verified", verified_at=time.time())
        write_json_atomic(args.receipt, receipt, immutable=False)
        print(f"Checkpoint {args.step} verified. Processes remain held for controlled restart.", flush=True)
    except BaseException as exc:
        write_json_atomic(args.receipt, {"state": "hold_error", "stopped": stopped,
                          "pids": identities, "step": args.step, "error": str(exc)}, immutable=False)
        raise


if __name__ == "__main__":
    main()
