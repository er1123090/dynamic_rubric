from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.pipeline import PipelineContext, StageError


def _config(path: Path, public_data: str = "data/public") -> Path:
    path.write_text(
        json.dumps(
            {
                "paths": {"public_data": public_data, "artifacts": "artifacts"},
                "training": {"reward_source": "static_r0_only"},
                "execution": {"mode": "fake"},
            }
        ),
        encoding="utf-8",
    )
    return path


def test_public_root_rejects_symlink_to_private_mount(tmp_path: Path) -> None:
    private = tmp_path / "data" / "private_gt"
    private.mkdir(parents=True)
    public_alias = tmp_path / "data" / "public"
    public_alias.symlink_to(private, target_is_directory=True)
    context = PipelineContext.create(
        tmp_path, _config(tmp_path / "config.json"), "generate-static", "run"
    )
    with pytest.raises(PermissionError, match="private-GT"):
        _ = context.public_root


def test_config_and_stage_input_reject_project_escape(tmp_path: Path) -> None:
    outside = tmp_path.parent / f"{tmp_path.name}-outside.json"
    outside.write_text("{}", encoding="utf-8")
    alias = tmp_path / "outside-config.json"
    alias.symlink_to(outside)
    with pytest.raises(StageError, match="escapes"):
        PipelineContext.create(tmp_path, alias, "generate-static", "run")

    config = _config(tmp_path / "config.json")
    context = PipelineContext.create(tmp_path, config, "generate-static", "run")
    with pytest.raises(StageError, match="escapes"):
        context.begin_stage(inputs=(outside,))
    assert not (context.stage_root() / "manifest.json").exists()


@pytest.mark.parametrize(
    "run_id",
    ("../escape", "nested/run", "nested\\run", "..", ".hidden", "run..escape", "/absolute"),
)
def test_run_id_rejects_path_traversal_and_non_slug_values(tmp_path: Path, run_id: str) -> None:
    config = _config(tmp_path / "config.json")
    with pytest.raises(StageError, match="run_id"):
        PipelineContext.create(tmp_path, config, "preflight", run_id)
    assert not (tmp_path.parent / "escape").exists()


def test_run_id_accepts_versioned_slug(tmp_path: Path) -> None:
    context = PipelineContext.create(
        tmp_path, _config(tmp_path / "config.json"), "preflight", "pilot-v1.2_test"
    )
    assert context.run_root == tmp_path / "artifacts" / "runs" / "pilot-v1.2_test"
