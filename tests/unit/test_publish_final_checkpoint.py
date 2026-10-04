import pytest

from dynamic_rubric.artifacts import read_json, write_json_atomic
from scripts.phase1.archive_unused_checkpoints_hf import (
    PARAMETER_FILES,
    RESUME_ONLY_FILES,
    check_source,
    cleanup_verified,
    original_records,
)


def saved_final(tmp_path):
    source = tmp_path / "verl-run/checkpoints/global_step_48"
    for name in PARAMETER_FILES | RESUME_ONLY_FILES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    (source.parent / "latest_checkpointed_iteration.txt").write_text("48")
    write_json_atomic(tmp_path / "verl-run/latest_commit.json", {
        "optimizer_update_index": 48, "checkpoint_saved": True, "checkpoint": str(source),
    })
    return source


def test_final_upload_inventory_excludes_optimizer_rng_and_data(tmp_path):
    source = saved_final(tmp_path)
    before = {p.relative_to(source).as_posix(): p.read_bytes()
              for p in source.rglob("*") if p.is_file()}
    assert check_source(tmp_path, 48, publish_latest=True) == source
    records, digest = original_records(source, parameters_only=True)
    assert {r["path"] for r in records} == PARAMETER_FILES
    assert len(digest) == 64
    assert all((source / name).read_bytes() == content for name, content in before.items())


def test_final_remains_forbidden_in_historical_cleanup(tmp_path):
    source = saved_final(tmp_path)
    with pytest.raises(ValueError, match="historical"):
        check_source(tmp_path, 48)
    write_json_atomic(tmp_path / "verl-run/checkpoint_archives/global_step_48.json", {
        "run_id": tmp_path.name, "checkpoint_step": 48, "state": "verified",
    })
    with pytest.raises(ValueError, match="historical"):
        cleanup_verified(tmp_path, 48, protected_resume_step=48)
    assert all((source / name).is_file() for name in RESUME_ONLY_FILES)


def test_final_upload_requires_saved_commit_and_complete_resume(tmp_path):
    source = saved_final(tmp_path)
    path = tmp_path / "verl-run/latest_commit.json"
    commit = read_json(path)
    write_json_atomic(path, {**commit, "checkpoint_saved": False}, immutable=False)
    with pytest.raises(ValueError, match="saved final"):
        check_source(tmp_path, 48, publish_latest=True)
    write_json_atomic(path, commit, immutable=False)
    (source / "data.pt").unlink()
    with pytest.raises(ValueError, match="allowlist"):
        check_source(tmp_path, 48, publish_latest=True)


def test_final_upload_rejects_extra_public_files_and_symlinks(tmp_path):
    source = saved_final(tmp_path)
    extra = source / "unexpected.txt"
    extra.write_text("not public")
    with pytest.raises(ValueError, match="allowlist"):
        check_source(tmp_path, 48, publish_latest=True)
    extra.unlink()
    extra.symlink_to(source / "data.pt")
    with pytest.raises(ValueError, match="symlinks"):
        check_source(tmp_path, 48, publish_latest=True)
