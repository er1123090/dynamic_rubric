from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.live_preflight import run_preflight
from dynamic_rubric.pipeline import PipelineContext, StageError


def test_live_preflight_does_not_treat_identity_only_lock_as_ready(tmp_path: Path) -> None:
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
    lock.write_text(
        json.dumps(
            {
                "live_gate": {"ready": True},
                "healthbench": {"source_sha256": "pinned"},
                "smokes": {
                    "openai_static_rubric": {"status": "passed"},
                    "openai_dynamic_extraction": {"status": "passed"},
                },
            }
        ),
        encoding="utf-8",
    )
    context = PipelineContext.create(tmp_path, config, "preflight", "identity-only")
    with pytest.raises(StageError, match="capability smokes not passed"):
        run_preflight(context)
    assert not context.run_root.exists()
