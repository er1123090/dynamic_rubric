from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.pipeline import PipelineContext, StageError


def test_downstream_rejects_receipt_from_different_config_or_lock(
    tmp_path: Path, monkeypatch
) -> None:
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
                "healthbench": {"source_sha256": "dataset-pin"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("DYNAMIC_RUBRIC_VLLM_URL", "http://127.0.0.1:1")
    context = PipelineContext.create(tmp_path, config, "generate-static", "bound-run")
    receipt = context.run_root / "preflight" / "preflight.json"
    receipt.parent.mkdir(parents=True)
    manifest = receipt.parent / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "ready": True,
                "ready_stages": ["generate-static", "train-static"],
                "run_id": "bound-run",
                "config_hash": "wrong-config-hash",
                "lock_sha256": sha256_file(lock),
                "preflight_manifest_sha256": sha256_file(manifest),
                "dataset": {"sha256": "dataset-pin"},
                "providers": {"identity": "test"},
                "capabilities": {"all": {"status": "passed"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StageError, match="incompatible"):
        context.begin_stage()
    assert not (context.stage_root() / "manifest.json").exists()

    value = json.loads(receipt.read_text(encoding="utf-8"))
    value["config_hash"] = context.config.config_hash
    value["lock_sha256"] = "wrong-lock-hash"
    receipt.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(StageError, match="incompatible"):
        context.begin_stage()
    assert not (context.stage_root() / "manifest.json").exists()

    value["lock_sha256"] = sha256_file(lock)
    receipt.write_text(json.dumps(value), encoding="utf-8")
    context.begin_stage()
    downstream_manifest = json.loads(
        (context.stage_root() / "manifest.json").read_text(encoding="utf-8")
    )
    assert any(
        path.endswith("/preflight/preflight.json") for path in downstream_manifest["input_hashes"]
    )


def test_downstream_rejects_stage_absent_from_ready_stages(
    tmp_path: Path, monkeypatch
) -> None:
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
                "live_gate": {"ready": False},
                "healthbench": {"source_sha256": "dataset-pin"},
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    context = PipelineContext.create(tmp_path, config, "generate-bon", "bound-run")
    receipt = context.run_root / "preflight" / "preflight.json"
    receipt.parent.mkdir(parents=True)
    manifest = receipt.parent / "manifest.json"
    manifest.write_text("{}\n", encoding="utf-8")
    receipt.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "ready": True,
                "ready_stages": ["generate-static", "train-static"],
                "run_id": "bound-run",
                "config_hash": context.config.config_hash,
                "lock_sha256": sha256_file(lock),
                "preflight_manifest_sha256": sha256_file(manifest),
                "dataset": {"sha256": "dataset-pin"},
                "providers": {"identity": "test"},
                "capabilities": {"all": {"status": "passed"}},
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(StageError, match="incompatible"):
        context.begin_stage()
