from pathlib import Path
import json

import pytest

from scripts.phase1.archive_unused_checkpoints_hf import (
    HF_METADATA,
    check_source,
    cleanup_verified,
    export_records,
    initial_policy_provenance,
    original_records,
    stage_link,
)
from dynamic_rubric.hashing import sha256_file



def make_checkpoint(run: Path, step: int = 13, latest: int = 40) -> Path:
    root = run / "verl-run/checkpoints"
    actor = root / f"global_step_{step}" / "actor"
    (actor / "huggingface").mkdir(parents=True)
    (root / "latest_checkpointed_iteration.txt").write_text(str(latest))
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"weights")
    (actor / "fsdp_config.json").write_text('{"world_size": 1}')
    for name in HF_METADATA:
        (actor / "huggingface" / name).write_text("metadata")
    return actor.parent


@pytest.mark.parametrize("step", (1, 13, 14, 35, 45, 47))
def test_every_nonfinal_step_is_archivable_after_a_newer_seal(tmp_path, step):
    source = make_checkpoint(tmp_path, step=step, latest=step + 1)
    assert check_source(tmp_path, step) == source


def test_current_or_final_step_is_not_historical_archive_target(tmp_path):
    source = make_checkpoint(tmp_path, step=47, latest=47)
    with pytest.raises(ValueError):
        check_source(tmp_path, 47)
    source.rename(source.with_name("global_step_48"))
    (tmp_path / "verl-run/checkpoints/latest_checkpointed_iteration.txt").write_text("49")
    with pytest.raises(ValueError):
        check_source(tmp_path, 48)


def test_source_rejects_unexpected_private_files_and_symlinks(tmp_path):
    source = make_checkpoint(tmp_path)
    (source / "credentials.json").write_text("not-for-upload")
    with pytest.raises(ValueError, match="allowlist"):
        check_source(tmp_path, 13)
    (source / "credentials.json").unlink()
    (source / "linked").symlink_to(source / "actor/fsdp_config.json")
    with pytest.raises(ValueError, match="symlinks"):
        check_source(tmp_path, 13)


def test_step40_only_becomes_eligible_after_newer_sealed_checkpoint(tmp_path):
    source = make_checkpoint(tmp_path, step=40, latest=40)
    with pytest.raises(ValueError):
        check_source(tmp_path, 40)
    (tmp_path / "verl-run/checkpoints/latest_checkpointed_iteration.txt").write_text("42")
    assert check_source(tmp_path, 40) == source


def test_step45_eligible_only_after_newer_checkpoint(tmp_path):
    source = make_checkpoint(tmp_path, step=45, latest=45)
    with pytest.raises(ValueError):
        check_source(tmp_path, 45)
    (tmp_path / "verl-run/checkpoints/latest_checkpointed_iteration.txt").write_text("46")
    assert check_source(tmp_path, 45) == source
    with pytest.raises(ValueError):
        check_source(tmp_path, 46)


def make_export(run: Path, source: Path, step: int) -> Path:
    exported = run / "audit/exports" / f"global_step_{step}"
    exported.mkdir(parents=True)
    for name in HF_METADATA:
        (exported / name).write_text("metadata")
    (exported / "model.safetensors").write_bytes(b"bf16-export")
    artifacts = []
    for name in ("config.json", "model.safetensors"):
        path = (exported / name).resolve()
        artifacts.append({"path": str(path), "bytes": path.stat().st_size,
                          "sha256": sha256_file(path)})
    manifest = {
        "schema_version": 1,
        "artifact_kind": "phase1_policy_checkpoint_export",
        "checkpoint_step": step,
        "checkpoint_id": f"global_step_{step}",
        "run_id": run.name,
        "source_model_sha256": sha256_file(source / "actor/model_world_size_1_rank_0.pt"),
        "source_model_bytes": (source / "actor/model_world_size_1_rank_0.pt").stat().st_size,
        "config_sha256": "a" * 64,
        "launch_spec_sha256": "b" * 64,
        "artifacts": artifacts,
    }
    (exported / "audit_export_manifest.json").write_text(json.dumps(manifest))
    return exported


def test_prebuilt_export_is_bound_to_exact_checkpoint(tmp_path):
    source = make_checkpoint(tmp_path, step=13, latest=45)
    exported = make_export(tmp_path, source, 13)
    resolved, records, binding = export_records(exported.parent, tmp_path, 13, source)
    assert resolved == exported.resolve()
    assert {row["path"] for row in records} == HF_METADATA | {"model.safetensors"}
    assert binding["path"] == str(exported.resolve())

    (exported / "model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum"):
        export_records(exported.parent, tmp_path, 13, source)


def test_step_zero_uses_probe_provenance_instead_of_online_commit(tmp_path):
    source = make_checkpoint(tmp_path, step=0, latest=45)
    exported = make_export(tmp_path, source, 0)
    _, _, binding = export_records(exported.parent, tmp_path, 0, source)
    provenance = exported.parents[1] / "responses/checkpoint-000000/provenance.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({
        "schema_version": 1,
        "run_id": tmp_path.name,
        "global_step": 0,
        "checkpoint_hash": sha256_file(source / "actor/model_world_size_1_rank_0.pt"),
        "config_sha256": "a" * 64,
        "launch_spec_sha256": "b" * 64,
    }))
    record = initial_policy_provenance(exported.parent, tmp_path, source, binding)
    assert record["path"] == str(provenance.resolve())
    assert record["sha256"] == sha256_file(provenance)


def test_original_records_exactly_reproduce_actor_tree_hash(tmp_path):
    from dynamic_rubric.training.live_online import _actor_parameter_tree_hash

    source = make_checkpoint(tmp_path)
    records, digest = original_records(source)
    assert len(records) == 7
    assert digest == _actor_parameter_tree_hash(source / "actor")
    assert all(row["remote_path"] == f"original_checkpoint/{row['path']}" for row in records)


def test_stage_is_idempotent_hardlink_not_copy(tmp_path):
    source = tmp_path / "source"
    source.write_bytes(b"weights")
    target = tmp_path / "stage/model.pt"
    stage_link(source, target)
    stage_link(source, target)
    assert source.stat().st_ino == target.stat().st_ino
    other = tmp_path / "other"
    other.write_bytes(b"different")
    with pytest.raises(ValueError):
        stage_link(other, target)


def prepare_cleanup(run, monkeypatch, *, remote_matches=True):
    import scripts.phase1.archive_unused_checkpoints_hf as script
    import dynamic_rubric.phase1.full_run as full_run
    import dynamic_rubric.training.checkpoint_archive as remote

    source = make_checkpoint(run)
    exported = make_export(run, source, 13)
    _, public_export_files, export_binding = export_records(exported.parent, run, 13, source)
    records, digest = original_records(source)
    receipt = {"schema_version": 1, "state": "verified", "checkpoint_step": 13,
               "run_id": run.name, "repo_id": "HYU-NLP-EVAL/test", "revision": "a" * 40,
               "actor_parameter_hash": digest, "files": records,
               "public_export_files": public_export_files,
               "export_directory_binding": export_binding,
               "audit_export_manifest": json.loads(
                   (exported / "audit_export_manifest.json").read_text()
               )}
    receipt_path = run / "verl-run/checkpoint_archives/global_step_13.json"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(json.dumps(receipt))
    commit = run / "verl-run/online_steps/step-000013/commit.json"
    commit.parent.mkdir(parents=True)
    commit.write_text(json.dumps({"artifacts": {"actor_parameter_hash": digest}}))
    protected = run / "verl-run/checkpoints/global_step_45"
    protected.mkdir()
    (protected / "optimizer-kept").write_text("keep")
    monkeypatch.setattr(full_run, "latest_full_checkpoint", lambda _: protected)
    monkeypatch.setattr(remote, "verify_public_archive", lambda _: digest if remote_matches else "0" * 64)
    monkeypatch.setattr(script, "verify_remote_records", lambda *args, **kwargs: None)
    return source, protected, receipt_path, exported


def test_cleanup_removes_only_verified_historical_source_and_its_staging(tmp_path, monkeypatch):
    source, protected, receipt_path, exported = prepare_cleanup(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text())
    receipt["local_deleted_at"] = "previous-cleanup-before-restore"
    receipt_path.write_text(json.dumps(receipt))
    stage = tmp_path / "hf_archive_staging/global_step_13/public"
    stage.mkdir(parents=True)
    stage_link(source / "actor/model_world_size_1_rank_0.pt", stage / "weights")
    result = cleanup_verified(tmp_path, 13)
    assert not source.exists()
    assert not exported.exists()
    assert not stage.parent.exists()
    assert not exported.exists()
    assert (protected / "optimizer-kept").read_text() == "keep"
    assert result["local_deleted_at"]
    assert json.loads(receipt_path.read_text())["state"] == "verified"
    assert json.loads(receipt_path.read_text())["audit_export_manifest"]["checkpoint_step"] == 13
    assert cleanup_verified(tmp_path, 13) == result


def test_cleanup_resumes_after_export_deletion(tmp_path, monkeypatch):
    source, _, receipt_path, exported = prepare_cleanup(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text())
    receipt["deletion_authorized_at"] = "before-crash"
    receipt["export_deletion_started_at"] = "before-crash"
    receipt_path.write_text(json.dumps(receipt))
    import shutil

    shutil.rmtree(exported)
    result = cleanup_verified(tmp_path, 13)
    assert not source.exists()
    assert result["local_deleted_at"]


def test_remote_mismatch_never_deletes_local_checkpoint(tmp_path, monkeypatch):
    source, _, _, exported = prepare_cleanup(tmp_path, monkeypatch, remote_matches=False)
    with pytest.raises(ValueError, match="no longer matches"):
        cleanup_verified(tmp_path, 13)
    assert source.is_dir()
    assert exported.is_dir()


def test_cleanup_rejects_wrong_run_receipt(tmp_path, monkeypatch):
    source, _, receipt_path, _ = prepare_cleanup(tmp_path, monkeypatch)
    receipt = json.loads(receipt_path.read_text())
    receipt["run_id"] = "another-run"
    receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="not bound"):
        cleanup_verified(tmp_path, 13)
    assert source.is_dir()


def test_cleanup_requires_explicit_updated_protection(tmp_path, monkeypatch):
    import dynamic_rubric.phase1.full_run as full_run

    source, old, _, exported = prepare_cleanup(tmp_path, monkeypatch)
    protected = old.with_name("global_step_46")
    old.rename(protected)
    monkeypatch.setattr(full_run, "latest_full_checkpoint", lambda _: protected)
    with pytest.raises(ValueError, match="authorization"):
        cleanup_verified(tmp_path, 13)
    assert source.is_dir() and exported.is_dir()
    result = cleanup_verified(tmp_path, 13, protected_resume_step=46)
    assert result["protected_resume_checkpoint"] == "global_step_46"
    assert (protected / "optimizer-kept").read_text() == "keep"
    assert not source.exists() and not exported.exists()
