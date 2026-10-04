from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.artifacts import read_json, write_json_atomic


SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "phase1" / "prepare_final_probe48.py"
)


def load_script():
    spec = importlib.util.spec_from_file_location("prepare_final_probe48", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_validate_completed_final_checkpoint_requires_saved_update_48(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    run = tmp_path / "run"
    checkpoint = run / "verl-run/checkpoints/global_step_48"
    checkpoint.mkdir(parents=True)
    output = tmp_path / "audit"
    export = output / "exports/global_step_48"
    export.mkdir(parents=True)
    latest = {
        "optimizer_update_index": 48,
        "checkpoint_saved": True,
        "checkpoint": str(checkpoint),
        "resume_checkpoint_hash": "r" * 64,
        "actor_parameter_hash": "a" * 64,
    }
    write_json_atomic(run / "verl-run/latest_commit.json", latest)
    contract = SimpleNamespace(run_dir=run)
    identity = SimpleNamespace(source_model_sha256="c" * 64)
    monkeypatch.setattr(module, "load_run_contract", lambda _path: contract)
    monkeypatch.setattr(module, "inspect_checkpoint", lambda *_args: identity)
    monkeypatch.setattr(module, "export_checkpoint", lambda *_args, **_kwargs: export)
    write_json_atomic(export / "audit_export_manifest.json", {"source_model_sha256": "c" * 64})

    result = module.validate_completed_final_checkpoint(run, output)
    assert result["checkpoint"] is identity

    latest["checkpoint_saved"] = False
    write_json_atomic(run / "verl-run/latest_commit.json", latest, immutable=False)
    with pytest.raises(module.FinalProbePreparationError, match="complete resumable"):
        module.validate_completed_final_checkpoint(run, output)


def test_fresh_rubric_command_preserves_production_parameters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    control = tmp_path / "pi0.json"
    control.write_text("{}\n", encoding="utf-8")
    launch = tmp_path / "launch.json"
    write_json_atomic(launch, {"control_cache": {"path": str(control)}})
    contract = SimpleNamespace(
        launch_spec_path=launch,
        run_dir=tmp_path / "run",
        run_id="production-run",
        train_path=tmp_path / "train.jsonl",
        probe_manifest_path=tmp_path / "probe.json",
        seed=11,
    )
    command = module.fresh_rubric_command(
        contract=contract,
        output_root=tmp_path / "audit",
        checkpoint_hash="h" * 64,
        endpoint="http://extractor:8001",
        prompt_workers=8,
        extractor_concurrency=8,
        max_in_flight=32,
        timeout_seconds=900,
    )
    joined = " ".join(command)
    assert "build_fixed_train_fresh_rubrics.py" in joined
    assert "--checkpoint-step 48" in joined
    assert "--seed 11" in joined
    assert "--prompt-workers 8" in joined
    assert "--extractor-concurrency 8" in joined
    assert "--max-in-flight 32" in joined
    assert "--timeout 900" in joined
    assert str(control) in command


def test_prepare_generates_a_before_overlapping_rubric_and_b(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = load_script()
    events: list[str] = []
    contract = SimpleNamespace(run_dir=tmp_path / "run")
    checkpoint = SimpleNamespace(source_model_sha256="c" * 64)
    monkeypatch.setattr(
        module,
        "validate_completed_final_checkpoint",
        lambda *_args, **_kwargs: {"contract": contract, "checkpoint": checkpoint},
    )
    monkeypatch.setattr(
        module,
        "validate_pool_provenance",
        lambda *_args, required_pools, **_kwargs: {
            "pool_counts": {"probe_A": 800, "probe_B": 1600},
            "pool_a_b_disjoint": len(required_pools) == 2,
        },
    )
    monkeypatch.setattr(module, "fresh_rubric_command", lambda **_kwargs: ["rubric-builder"])
    rubric_status = tmp_path / "audit/rubrics/checkpoint-000048/status.json"
    write_json_atomic(rubric_status, {"state": "complete", "prompt_count": 100})

    def generate(_contract, *, pools, **_kwargs):
        events.append(f"generate:{pools[0]}")
        return {"reused": False}

    class Process:
        def __init__(self, *_args, **_kwargs):
            events.append("rubric:start")

        def wait(self):
            events.append("rubric:wait")
            return 0

    monkeypatch.setattr(module.subprocess, "Popen", Process)
    result = module.prepare(
        run_dir=tmp_path / "run",
        output_root=tmp_path / "audit",
        policy_base_url="http://policy",
        extractor_endpoint="http://extractor",
        log_dir=tmp_path / "audit/logs/final-probe48",
        pool_generator=generate,
    )
    assert events == [
        "generate:probe_A",
        "rubric:start",
        "generate:probe_B",
        "rubric:wait",
    ]
    assert result["state"] == "complete"
    assert result["pool_a_b_disjoint"] is True
    assert read_json(tmp_path / "audit/logs/final-probe48/status.json")["state"] == "complete"
