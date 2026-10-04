from __future__ import annotations

import json

import pytest

from scripts.phase1.archive_checkpoint_batch import run_batch
from scripts.phase1.archive_unused_checkpoints_hf import ALLOWED_STEPS


def write_gate(path, run):
    path.write_text(json.dumps({
        "schema_version": 1,
        "state": "cleanup_ready",
        "run_id": run.name,
        "protected_resume_step": 45,
        "graders_use_archived_checkpoint_identity": True,
        "authorized_steps": list(ALLOWED_STEPS),
    }))


def test_upload_only_queue_is_sequential_and_resumable(tmp_path):
    calls = []

    def fake_archive(run, step, **kwargs):
        calls.append((step, kwargs["workers"]))
        return {"repo_id": f"repo-{step}", "revision": str(step) * 40}

    kwargs = {
        "run": tmp_path,
        "export_root": tmp_path / "audit/exports",
        "steps": (0, 3),
        "upload_only": True,
        "queue_root": tmp_path / "queue",
        "archive_fn": fake_archive,
    }
    result = run_batch(**kwargs)
    assert result["state"] == "uploads_complete"
    assert calls == [(0, 1), (3, 1)]

    run_batch(**kwargs)
    assert calls == [(0, 1), (3, 1)]



def test_cleanup_requires_and_rechecks_exact_gate(tmp_path):
    gate = tmp_path / "cleanup-ready.json"
    write_gate(gate, tmp_path)
    cleaned = []

    result = run_batch(
        run=tmp_path,
        export_root=tmp_path / "audit/exports",
        steps=(3,),
        cleanup_ready_file=gate,
        queue_root=tmp_path / "queue",
        archive_fn=lambda run, step, **kwargs: {
            "repo_id": "repo",
            "revision": "a" * 40,
        },
        cleanup_fn=lambda run, step: cleaned.append(step),
    )
    assert result["state"] == "complete"
    assert cleaned == [3]

    gate.unlink()
    with pytest.raises(ValueError, match="cleanup-ready"):
        run_batch(
            run=tmp_path,
            export_root=tmp_path / "audit/exports",
            steps=(6,),
            cleanup_ready_file=gate,
            queue_root=tmp_path / "another-queue",
        )


def test_failure_is_durable_and_retry_increments_attempt(tmp_path):
    queue = tmp_path / "queue"

    def fail(*args, **kwargs):
        raise RuntimeError("upload interrupted")

    with pytest.raises(RuntimeError, match="interrupted"):
        run_batch(
            run=tmp_path,
            export_root=tmp_path / "audit/exports",
            steps=(6,),
            upload_only=True,
            queue_root=queue,
            archive_fn=fail,
        )
    failed = json.loads((queue / "global_step_6.json").read_text())
    assert failed["state"] == "failed"
    assert failed["attempt"] == 1
    assert json.loads((queue / "batch-status.json").read_text())["failed_step"] == 6

    run_batch(
        run=tmp_path,
        export_root=tmp_path / "audit/exports",
        steps=(6,),
        upload_only=True,
        queue_root=queue,
        archive_fn=lambda run, step, **kwargs: {
            "repo_id": "repo",
            "revision": "b" * 40,
        },
    )
    retried = json.loads((queue / "global_step_6.json").read_text())
    assert retried["state"] == "archive_verified"
    assert retried["attempt"] == 2


def test_archive_verified_state_resumes_cleanup_without_archiving_again(tmp_path):
    queue = tmp_path / "queue"
    queue.mkdir()
    (queue / "global_step_3.json").write_text(json.dumps({
        "schema_version": 1,
        "state": "archive_verified",
        "step": 3,
        "attempt": 1,
    }))
    receipt = tmp_path / "verl-run/checkpoint_archives/global_step_3.json"
    receipt.parent.mkdir(parents=True)
    receipt.write_text(json.dumps({"repo_id": "repo", "revision": "a" * 40}))
    gate = tmp_path / "cleanup-ready.json"
    write_gate(gate, tmp_path)
    cleaned = []

    run_batch(
        run=tmp_path,
        export_root=tmp_path / "audit/exports",
        steps=(3,),
        cleanup_ready_file=gate,
        queue_root=queue,
        archive_fn=lambda *args, **kwargs: pytest.fail("archive must not rerun"),
        cleanup_fn=lambda run, step: cleaned.append(step),
    )
    assert cleaned == [3]
    state = json.loads((queue / "global_step_3.json").read_text())
    assert state["state"] == "complete"
    assert state["attempt"] == 2


def test_queue_rejects_parallel_workers_and_protected_step(tmp_path):
    with pytest.raises(ValueError, match="one worker"):
        run_batch(
            run=tmp_path,
            export_root=tmp_path,
            steps=(3,),
            workers=2,
            upload_only=True,
        )
    with pytest.raises(ValueError, match="members of ALLOWED_STEPS"):
        run_batch(
            run=tmp_path,
            export_root=tmp_path,
            steps=(45,),
            upload_only=True,
        )
