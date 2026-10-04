from __future__ import annotations

import json
import shutil
import urllib.request
from pathlib import Path

import pytest
import yaml

from dynamic_rubric.data.sealing import SealedTrajectoryError
from dynamic_rubric.live_preflight import run_preflight
from dynamic_rubric.pipeline import (
    PipelineContext,
    run_freeze_updater,
    run_generate_static,
    run_prepare_data,
    run_replay_dynamic,
    run_train_static,
)
from dynamic_rubric.pipeline_audit import (
    run_analyze,
    run_audit_gold,
    run_export_audit_package,
    run_validate_inventory,
)
from dynamic_rubric.pipeline_evaluation import run_generate_bon, run_score_proxy, run_select_bon


def _write_project(tmp_path: Path) -> tuple[Path, Path]:
    schema_source = Path(__file__).resolve().parents[2] / "configs" / "schemas"
    shutil.copytree(schema_source, tmp_path / "configs" / "schemas")
    config = {
        "experiment": "credential_free_integration",
        "split_seed": 7,
        "bootstrap_seed": 11,
        "paths": {"public_data": "data/public", "artifacts": "artifacts", "results": "results"},
        "models": {
            "rubric_generator": {
                "requested_model": "gpt-5-mini",
                "reasoning_effort": "medium",
                "prompt_version": "rubric-generator-v1",
                "schema": "configs/schemas/rubric_generator_v1.json",
            },
            "hidden_gt_grader": {
                "requested_model": "gpt-5",
                "reasoning_effort": "medium",
                "prompt_version": "hidden-gold-grader-v1",
                "schema": "configs/schemas/hidden_gold_grader_v1.json",
            },
            "policy": {"model": "fake/qwen3-4b", "revision": "fake-v1"},
            "proxy_grader": {"model": "fake/qwen3-32b", "revision": "fake-v1"},
            "criterion_embedding": {"model": "fake/embedding", "revision": "fake-v1"},
        },
        "splits": {"pilot_train": 1, "pilot_probe": 1, "pilot_audit": 1},
        "training": {
            "max_steps": 3,
            "train_batch_size": 1,
            "rollout_n": 2,
            "checkpoint_steps": [0, 1, 2, 3],
            "reward_source": "static_r0_only",
            "artifact_inputs": ["artifacts/static"],
        },
        "probe": {"development_samples_per_family": 2, "final_samples_per_family": 2},
        "replay": {
            "modes": [
                "static",
                "dynamic_fixed_budgeted",
                "dynamic_prev_budgeted",
                "refresh_only_budgeted",
                "dynamic_fixed_cumulative",
            ],
            "max_dynamic_slots": 4,
            "satisfaction_min": 0.10,
            "satisfaction_max": 0.90,
            "separation_min": 0.15,
            "max_similarity_exclusive": 0.85,
            "parse_success_min": 0.95,
        },
        "bon": {
            "focal_steps": [1, 3],
            "rubric_steps": [1, 3],
            "pool_size": 4,
            "sizes": [1, 2, 4],
            "permutations": 2,
        },
        "bootstrap": {"iterations": 200},
        "execution": {"mode": "fake"},
    }
    config_path = tmp_path / "configs" / "test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source = tmp_path / "data" / "source" / "healthbench_consensus.jsonl"
    source.parent.mkdir(parents=True)
    rows = [
        {
            "prompt": f"Synthetic medical prompt {index}",
            "rubric": [{"criterion": f"Private synthetic gold criterion {index}", "points": 1}],
        }
        for index in range(3)
    ]
    source.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return config_path, source


def _context(root: Path, config: Path, stage: str, run_id: str = "offline-e2e") -> PipelineContext:
    return PipelineContext.create(root, config, stage, run_id)


def test_credential_free_pipeline_resume_sealing_and_inventory(tmp_path: Path, monkeypatch) -> None:
    config, source = _write_project(tmp_path)
    network_calls = 0

    def tripwire(*args, **kwargs):
        nonlocal network_calls
        network_calls += 1
        raise AssertionError("offline pipeline attempted network access")

    monkeypatch.setattr(urllib.request, "urlopen", tripwire)
    run_prepare_data(_context(tmp_path, config, "prepare-data"), source)
    run_preflight(_context(tmp_path, config, "preflight"))
    static_context = _context(tmp_path, config, "generate-static")
    run_generate_static(static_context)
    static_bytes = (static_context.stage_root() / "static_rubrics.jsonl").read_bytes()
    run_generate_static(static_context)
    assert (static_context.stage_root() / "static_rubrics.jsonl").read_bytes() == static_bytes
    run_train_static(_context(tmp_path, config, "train-static"))

    with pytest.raises(SealedTrajectoryError):
        run_replay_dynamic(_context(tmp_path, config, "replay-dynamic-final"), "final")

    run_replay_dynamic(_context(tmp_path, config, "replay-dynamic-development"), "development")
    run_freeze_updater(_context(tmp_path, config, "freeze-updater"))
    run_replay_dynamic(_context(tmp_path, config, "replay-dynamic-final"), "final")
    run_generate_bon(_context(tmp_path, config, "generate-bon"))
    run_score_proxy(_context(tmp_path, config, "score-proxy"))
    run_select_bon(_context(tmp_path, config, "select-bon"))
    run_export_audit_package(_context(tmp_path, config, "export-audit-package"))
    private_gt = tmp_path / "data" / "private_gt" / "healthbench_gold_rubrics.jsonl"
    audit = run_audit_gold(_context(tmp_path, config, "audit-gold"), private_gt)
    report = run_analyze(_context(tmp_path, config, "analyze"))
    inventory = run_validate_inventory(_context(tmp_path, config, "validate-inventory"))

    assert network_calls == 0
    assert audit["grader_calls"] == audit["selected_unique_responses"]
    assert audit["unselected_calls"] == 0
    assert inventory["status"] == "passed"
    assert report["interpretation"] in {
        "weak_policy_drift",
        "refresh_noise",
        "policy_adaptive_gain",
        "local_redundancy",
        "general_rubric_improvement",
        "textual_only_churn",
        "frequent_update_need",
        "updater_miss",
    }
    final_snapshots = json.loads(
        (tmp_path / "results" / "offline-e2e" / "pilot_report.json").read_text(encoding="utf-8")
    )
    assert "textual_vs_functional_change" in final_snapshots
    assert all(
        value["judge_repeat_stability"] is not None
        for value in final_snapshots["metrics"]["reward_resolution"].values()
    )
    assert all(
        "semantic_overlap_mean" in value and "mean_survival_steps" in value
        for value in final_snapshots["metrics"]["rubric_churn"].values()
    )
    assert final_snapshots["statistical_rules"]["bootstrap_samples"] == 200
