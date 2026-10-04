from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.pipeline import PipelineContext, StageError


def test_live_stage_requires_successful_preflight_receipt(tmp_path: Path, monkeypatch) -> None:
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "paths": {"artifacts": "artifacts"},
                "training": {"reward_source": "static_r0_only"},
                "execution": {"mode": "live"},
            }
        ),
        encoding="utf-8",
    )
    lock = tmp_path / "environment" / "upstream-lock.json"
    lock.parent.mkdir()
    lock.write_text(json.dumps({"live_gate": {"ready": True}}), encoding="utf-8")
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("DYNAMIC_RUBRIC_VLLM_URL", "http://127.0.0.1:1")
    context = PipelineContext.create(tmp_path, config, "generate-static", "run")
    with pytest.raises(StageError, match="preflight receipt"):
        context.begin_stage()
    assert not (tmp_path / "artifacts").exists()
