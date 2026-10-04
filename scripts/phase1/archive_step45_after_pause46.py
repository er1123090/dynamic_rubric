#!/usr/bin/env python3
"""One-run archive authorization: publish/delete 45 while retaining sealed 46."""

import argparse
import fcntl
from pathlib import Path
import time

from dynamic_rubric.artifacts import read_json, write_json_atomic
from dynamic_rubric.phase1.full_run import latest_full_checkpoint
from dynamic_rubric.training.live_online import _checkpoint_tree_hash
from scripts.phase1.archive_unused_checkpoints_hf import archive, cleanup_verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    run, audit = args.run.resolve(), args.audit.resolve()
    journal = run / "logs/archive45-protect46-20260909"
    journal.mkdir(exist_ok=True)
    with (journal / "task.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def status(state, **extra):
            write_json_atomic(journal / "status.json", {
                "state": state, "time": time.time(), "step": 45,
                "protected_resume_step": 46, **extra,
            }, immutable=False)

        def verify_resume():
            protected = latest_full_checkpoint(run)
            seal = read_json(run / "logs/pause46-for-kl-20260909/sealed.json")
            if protected.name != "global_step_46" or str(protected) != seal["checkpoint"]:
                raise ValueError("Expected the sealed pause checkpoint 46")
            digest = _checkpoint_tree_hash(protected)
            if digest != seal["resume_checkpoint_hash"]:
                raise ValueError("Checkpoint46 model/optimizer/data integrity mismatch")
            return digest

        try:
            status("verifying_protected46")
            digest = verify_resume()
            receipt_path = run / "verl-run/checkpoint_archives/global_step_45.json"
            receipt = read_json(receipt_path) if receipt_path.exists() else None
            if not receipt or not receipt.get("local_deleted_at"):
                if not receipt or receipt.get("state") != "verified":
                    status("uploading45", protected_resume_hash=digest)
                    receipt = archive(run, 45, export_root=audit / "exports", upload=True, workers=1)
                status("verifying46_before_cleanup", revision=receipt["revision"])
                verify_resume()
                receipt = cleanup_verified(run, 45, protected_resume_step=46)
            status("verifying46_after_cleanup", revision=receipt["revision"])
            verify_resume()
            for path in (run / "verl-run/checkpoints/global_step_45",
                         audit / "exports/global_step_45",
                         run / "hf_archive_staging/global_step_45"):
                if path.exists():
                    raise ValueError(f"Cleanup target remains: {path}")
            status("complete", repo_id=receipt["repo_id"], revision=receipt["revision"],
                   protected_resume_hash=digest, local_deleted_at=receipt["local_deleted_at"])
        except Exception as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
