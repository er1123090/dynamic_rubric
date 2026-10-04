from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import yaml

from dynamic_rubric.artifacts import read_jsonl
from dynamic_rubric.pipeline import (
    PipelineContext,
    run_generate_static,
    run_prepare_data,
    run_replay_dynamic,
    run_train_static,
)


def _project(tmp_path: Path) -> tuple[Path, Path]:
    schemas = Path(__file__).resolve().parents[2] / "configs" / "schemas"
    shutil.copytree(schemas, tmp_path / "configs" / "schemas")
    config = {
        "experiment": "replay_contract",
        "split_seed": 17,
        "paths": {
            "public_data": "data/public",
            "artifacts": "artifacts",
            "results": "results",
        },
        "models": {
            "rubric_generator": {
                "requested_model": "gpt-5-mini",
                "reasoning_effort": "medium",
                "schema": "configs/schemas/rubric_generator_v1.json",
            },
            "policy": {"model": "fake/policy", "revision": "v1"},
            "proxy_grader": {"model": "fake/grader", "revision": "v1"},
            "criterion_embedding": {"model": "fake/embed", "revision": "v1"},
        },
        "splits": {"pilot_train": 1, "pilot_probe": 1, "pilot_audit": 1},
        "training": {
            "max_steps": 3,
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
            "replicate_fraction": 1.0,
        },
        "execution": {"mode": "fake"},
    }
    config_path = tmp_path / "configs" / "test.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    source = tmp_path / "data" / "source.jsonl"
    source.parent.mkdir(parents=True)
    source.write_text(
        "".join(
            json.dumps(
                {
                    "prompt": f"Synthetic prompt {index}",
                    "rubric": [{"criterion": f"private criterion {index}", "points": 1}],
                }
            )
            + "\n"
            for index in range(3)
        ),
        encoding="utf-8",
    )
    return config_path, source


def _context(root: Path, config: Path, stage: str) -> PipelineContext:
    return PipelineContext.create(root, config, stage, "replay-contract")


def _prepare_static(tmp_path: Path, config: Path, source: Path) -> None:
    run_prepare_data(_context(tmp_path, config, "prepare-data"), source)
    run_generate_static(_context(tmp_path, config, "generate-static"))


def test_training_mid_stage_resume_is_byte_identical(tmp_path: Path, monkeypatch) -> None:
    config, source = _project(tmp_path)
    _prepare_static(tmp_path, config, source)
    monkeypatch.setenv("DYNAMIC_RUBRIC_TRAIN_FAULT_AFTER_SHARDS", "2")
    with pytest.raises(RuntimeError, match="injected training interruption"):
        run_train_static(_context(tmp_path, config, "train-static"))
    shard_root = tmp_path / "artifacts" / "runs" / "replay-contract" / "train-static" / "shards"
    before = {path: path.read_bytes() for path in shard_root.rglob("*.jsonl")}
    assert before

    monkeypatch.delenv("DYNAMIC_RUBRIC_TRAIN_FAULT_AFTER_SHARDS")
    run_train_static(_context(tmp_path, config, "train-static"))
    assert {path: path.read_bytes() for path in before} == before
    trajectory = shard_root.parent / "trajectory_development.jsonl"
    completed = trajectory.read_bytes()
    run_train_static(_context(tmp_path, config, "train-static"))
    assert trajectory.read_bytes() == completed


def test_replay_uses_fixed_previous_refresh_and_blind_heldout_pools(tmp_path: Path) -> None:
    config, source = _project(tmp_path)
    _prepare_static(tmp_path, config, source)
    train_context = _context(tmp_path, config, "train-static")
    run_train_static(train_context)
    trajectory = read_jsonl(train_context.stage_root() / "trajectory_development.jsonl")
    assert all(
        hashlib.sha256(str(row["response_text"]).encode()).hexdigest()
        == row["provider_call"]["raw_response_hash"]
        for row in trajectory
    )
    context = _context(tmp_path, config, "replay-dynamic-development")
    run_replay_dynamic(context, "development")
    rows = read_jsonl(context.stage_root() / "replay_snapshots.jsonl")
    keyed = {(row["policy_step"], row["mode"]): row for row in rows}

    fixed = keyed[(2, "dynamic_fixed_budgeted")]
    cumulative = keyed[(2, "dynamic_fixed_cumulative")]
    previous = keyed[(2, "dynamic_prev_budgeted")]
    refresh = keyed[(2, "refresh_only_budgeted")]
    assert fixed["control_policy"] == "pi_0"
    assert fixed["current_response_used"] is True
    assert cumulative["discovery_pool_hash"] == fixed["discovery_pool_hash"]
    assert cumulative["validation_pool_hash"] == fixed["validation_pool_hash"]
    assert cumulative["pairing_hash"] == fixed["pairing_hash"]
    assert keyed[(1, "dynamic_prev_budgeted")]["control_policy"] == "pi_0"
    assert previous["control_policy"] == "pi_1"
    assert refresh["control_policy"] == "pi_0_refresh"
    assert refresh["current_response_used"] is False
    assert refresh["current_response_ids"] == []
    assert set(refresh["extraction_left_response_ids"]).isdisjoint(refresh["control_response_ids"])

    dynamic = [row for row in rows if row["mode"] != "static"]
    assert all(row["generator_payload_source_blind"] for row in dynamic)
    assert all(row["replicate_b"]["independent_call"] for row in dynamic)
    assert all(
        candidate["evidence"]["independent_validation"]
        for row in dynamic
        for candidate in row["candidate_evidence"]
    )


def test_replay_prompt_shard_resume_is_byte_identical(tmp_path: Path, monkeypatch) -> None:
    config, source = _project(tmp_path)
    _prepare_static(tmp_path, config, source)
    run_train_static(_context(tmp_path, config, "train-static"))
    context = _context(tmp_path, config, "replay-dynamic-development")
    monkeypatch.setenv("DYNAMIC_RUBRIC_REPLAY_FAULT_AFTER_SHARDS", "1")
    with pytest.raises(RuntimeError, match="injected replay interruption"):
        run_replay_dynamic(context, "development")
    shard = next((context.stage_root() / "shards").glob("*.jsonl"))
    before = shard.read_bytes()

    monkeypatch.delenv("DYNAMIC_RUBRIC_REPLAY_FAULT_AFTER_SHARDS")
    run_replay_dynamic(context, "development")
    assert shard.read_bytes() == before
    aggregate = (context.stage_root() / "replay_snapshots.jsonl").read_bytes()
    run_replay_dynamic(context, "development")
    assert (context.stage_root() / "replay_snapshots.jsonl").read_bytes() == aggregate
