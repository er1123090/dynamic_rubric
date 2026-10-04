from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.artifacts import artifact_record, read_json
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1 import probe_adjacent_scoring as target


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_step_zero_uses_only_original_prompt_specific_r0(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    row = {
        "prompt_id": "p1",
        "r0": {
            "criteria": [
                {"criterion_id": "p1:r0:0", "criterion": "Be correct", "weight_units": 10}
            ]
        },
    }
    _write_jsonl(train, [row])
    contract = SimpleNamespace(train_path=train, train_sha256="a" * 64)
    monkeypatch.setattr(target, "load_probe_prompts", lambda contract: [{"prompt_id": "p1"}])

    rubrics, source = target.load_evaluator_rubrics(contract, tmp_path, 0)

    assert rubrics == {
        "p1": [
            {"criterion_id": "p1:r0:0", "text": "Be correct", "weight": 10, "source": "r0"}
        ]
    }
    assert source["kind"] == "initial_prompt_specific_r0"
    assert not (tmp_path / "rubrics").exists()


def test_fresh_rubric_requires_bound_complete_inventory(tmp_path, monkeypatch):
    train = tmp_path / "train.jsonl"
    _write_jsonl(
        train,
        [
            {
                "prompt_id": "p1",
                "r0": {
                    "criteria": [
                        {"criterion_id": "p1:r0:0", "criterion": "R0", "weight_units": 10}
                    ]
                },
            }
        ],
    )
    contract = SimpleNamespace(train_path=train, train_sha256="a" * 64)
    monkeypatch.setattr(target, "load_probe_prompts", lambda contract: [{"prompt_id": "p1"}])
    rubric_path = tmp_path / "rubrics/checkpoint-000003/fresh_rubrics.jsonl"
    _write_jsonl(
        rubric_path,
        [
            {
                "state": "verified_complete",
                "prompt_id": "p1",
                "global_step": 3,
                "evaluator_checkpoint": 3,
                "fresh_or_stale": "fresh",
                "fresh_rubric": {
                    "offline_criteria": [
                        {"criterion_id": "p1:r0:0", "text": "R0", "weight": 10, "source": "r0"}
                    ],
                    "online_criteria": [
                        {"criterion_id": "online-1", "text": "New", "weight": 1, "source": "online"}
                    ],
                },
            }
        ],
    )
    _write_json(
        rubric_path.parent / "status.json",
        {
            "state": "complete",
            "checkpoint_step": 3,
            "prompt_count": 100,
        },
    )

    with pytest.raises(target.ProbeAdjacentScoringError, match="not ready"):
        target.load_evaluator_rubrics(contract, tmp_path, 3)


def test_production_schema_hydrates_pool_b_and_binds_e3(tmp_path, monkeypatch):
    run_dir = tmp_path / "run"
    artifact_root = tmp_path / "audit"
    train = tmp_path / "train.jsonl"
    prompt_rows = [
        {
            "prompt_id": f"p{i}",
            "messages": [{"role": "user", "content": f"question {i}"}],
            "r0": {
                "criteria": [
                    {"criterion_id": f"p{i}:r0:0", "criterion": "R0", "weight_units": 10}
                ]
            },
        }
        for i in range(100)
    ]
    _write_jsonl(train, prompt_rows)
    contract = SimpleNamespace(
        run_dir=run_dir,
        run_id="run",
        domain="medicine",
        method="online_rubrics",
        seed=11,
        train_path=train,
        train_sha256=sha256_file(train),
        probe_manifest_sha256="b" * 64,
        config_sha256="c" * 64,
        launch_spec_sha256="d" * 64,
    )
    monkeypatch.setattr(
        target,
        "load_probe_prompts",
        lambda contract: [
            {"prompt_id": row["prompt_id"], "messages": row["messages"]} for row in prompt_rows
        ],
    )
    checkpoint = SimpleNamespace(source_model_sha256="e" * 64)
    monkeypatch.setattr(target, "inspect_checkpoint", lambda contract, step: checkpoint)
    monkeypatch.setattr(target, "_validate_pool_rows", lambda *args, **kwargs: None)

    response_dir = artifact_root / "responses/checkpoint-000003"
    pool_a = [
        {
            "prompt_id": f"p{i}",
            "response_id": f"a{i}:{j}",
            "pool": "probe_A",
            "vllm_seed": 1000 + j,
        }
        for i in range(100)
        for j in range(8)
    ]
    pool_b = [
        {
            "prompt_id": f"p{i}",
            "response_id": f"b{i}:{j}",
            "pool": "probe_B",
            "vllm_seed": 2000 + j,
            "sample_index": j,
            "response_text": f"answer {i} {j}",
        }
        for i in range(100)
        for j in range(16)
    ]
    _write_jsonl(response_dir / "probe_A.jsonl", pool_a)
    _write_jsonl(response_dir / "probe_B.jsonl", pool_b)
    _write_json(
        response_dir / "provenance.json",
        {
            "artifact_kind": "phase1_fixed_train_probe_policy_pools",
            "domain": contract.domain,
            "method": contract.method,
            "seed": contract.seed,
            "run_id": contract.run_id,
            "global_step": 3,
            "checkpoint_id": "global_step_3",
            "policy_checkpoint": 3,
            "checkpoint_hash": checkpoint.source_model_sha256,
            "config_sha256": contract.config_sha256,
            "launch_spec_sha256": contract.launch_spec_sha256,
            "probe_manifest_sha256": contract.probe_manifest_sha256,
            "train_sha256": contract.train_sha256,
            "selected_pools": ["probe_A", "probe_B"],
            "pool_counts": {"probe_A": 800, "probe_B": 1600},
            "artifacts": [
                artifact_record(response_dir / "probe_A.jsonl"),
                artifact_record(response_dir / "probe_B.jsonl"),
            ],
        },
    )
    hydrated, _ = target.load_pool_b(contract, artifact_root, 3)
    assert len(hydrated) == 1600
    assert hydrated[0]["text"] == "answer 0 0"
    assert hydrated[0]["prompt_messages"] == prompt_rows[0]["messages"]
    assert hydrated[0]["rollout_index"] == 0

    rubric_dir = artifact_root / "rubrics/checkpoint-000003"
    _write_json(
        rubric_dir / "invocation.json",
        {
            "run_id": contract.run_id,
            "checkpoint_step": 3,
            "checkpoint_hash": checkpoint.source_model_sha256,
            "seed": 11,
            "train_sha256": contract.train_sha256,
            "probe_manifest_sha256": contract.probe_manifest_sha256,
            "prompt_count": 100,
            "pool_a_responses_per_prompt": 8,
            "pi0_controls_per_prompt": 8,
            "domain": contract.domain,
            "method": contract.method,
            "run_contract_dir": str(run_dir),
        },
    )
    fresh_rows = []
    for row in prompt_rows:
        prompt_id = row["prompt_id"]
        offline = [
            {"criterion_id": f"{prompt_id}:r0:0", "text": "R0", "weight": 10, "source": "r0"}
        ]
        online = [
            {"criterion_id": f"{prompt_id}:online:0", "text": "New", "weight": 1, "source": "online_pairwise"}
        ]
        invocation = {
            "run_id": contract.run_id,
            "checkpoint_step": 3,
            "checkpoint_hash": checkpoint.source_model_sha256,
            "prompt_id": prompt_id,
            "seed": 11,
            "r0_hash": sha256_json(offline),
        }
        fresh_rows.append(
            {
                "state": "verified_complete",
                "domain": contract.domain,
                "method": contract.method,
                "seed": 11,
                "global_step": 3,
                "evaluator_checkpoint": 3,
                "fresh_or_stale": "fresh",
                "prompt_id": prompt_id,
                "invocation": invocation,
                "invocation_hash": sha256_json(invocation),
                "fresh_rubric": {
                    "offline_criteria": offline,
                    "online_criteria": online,
                    "content_hash": sha256_json(offline + online),
                },
            }
        )
    _write_jsonl(rubric_dir / "fresh_rubrics.jsonl", fresh_rows)
    _write_json(
        rubric_dir / "status.json",
        {
            "state": "complete",
            "checkpoint_step": 3,
            "checkpoint_hash": checkpoint.source_model_sha256,
            "prompt_count": 100,
            "fresh_rubrics_sha256": sha256_file(rubric_dir / "fresh_rubrics.jsonl"),
        },
    )
    rubrics, _ = target.load_evaluator_rubrics(contract, artifact_root, 3)
    assert len(rubrics) == 100
    assert [item["source"] for item in rubrics["p0"]] == ["r0", "online_pairwise"]


def test_regular_scoring_runs_diagonal_and_adjacent_then_resumes(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    _write_json(
        config,
        {
            "models": {
                "judge": {
                    "model": "Qwen/Qwen3-32B",
                    "revision": "rev",
                    "max_output_tokens": 4096,
                }
            }
        },
    )
    contract = SimpleNamespace(
        config_path=config,
        domain="medicine",
        method="online_rubrics",
        seed=11,
    )
    monkeypatch.setattr(target, "load_run_contract", lambda path: contract)
    monkeypatch.setattr(
        target,
        "endpoint_identity",
        lambda url, model, revision: {
            "url": url,
            "model": {
                "id": model,
                "root": f"/models/{revision}",
                "max_model_len": 32768,
            },
            "version": {"v": 1},
        },
    )
    monkeypatch.setattr(target, "VLLMChatAdapter", lambda *args, **kwargs: object())
    monkeypatch.setattr(target, "_pool_b_readiness", lambda *args: (True, "ready"))
    monkeypatch.setattr(target, "_evaluator_readiness", lambda *args: (True, "ready"))

    pool_loads = []
    evaluator_loads = []

    def responses(contract, root, step):
        pool_loads.append(step)
        rows = [
            {
                "response_id": f"p{i // 16}:r{i % 16}",
                "prompt_id": f"p{i // 16}",
                "sample_index": i % 16,
                "policy_step": step,
            }
            for i in range(1600)
        ]
        return rows, {"path": f"pool-{step}", "sha256": "a" * 64, "bytes": 1}

    monkeypatch.setattr(target, "load_pool_b", responses)
    def evaluators(contract, root, step):
        evaluator_loads.append(step)
        return (
            {f"p{i}": [{"criterion_id": f"c{step}", "text": "criterion", "weight": 1}] for i in range(100)},
            {"kind": "r0"} if step == 0 else {"path": f"rubric-{step}"},
        )

    monkeypatch.setattr(target, "load_evaluator_rubrics", evaluators)
    calls = []

    def score_pool(rows, rubrics, **kwargs):
        calls.append((int(kwargs["evaluator_checkpoint"]), int(kwargs["policy_checkpoint"])))
        return [
            {
                "response_id": row["response_id"],
                "prompt_id": row["prompt_id"],
                "policy_step": int(kwargs["policy_checkpoint"]),
                "evaluator_step": int(kwargs["evaluator_checkpoint"]),
                "pool": "probe_B",
            }
            for row in rows
        ]

    monkeypatch.setattr(target, "score_pool", score_pool)
    output = tmp_path / "scores"
    result = target.score_regular_adjacent(
        run_dir=tmp_path,
        artifact_root=tmp_path / "artifacts",
        output_root=output,
        steps=(0, 3),
        judge_urls=("http://judge",),
    )

    assert calls == [(0, 0), (0, 3), (3, 3)]
    assert pool_loads == [0, 3]
    assert evaluator_loads == [0, 3]
    assert result["cell_count"] == 3
    assert result["score_count"] == 4800
    stale = read_json(output / "policy-000003/evaluator-000000/manifest.json")
    fresh = read_json(output / "policy-000003/evaluator-000003/manifest.json")
    assert stale["response_ids_sha256"] == fresh["response_ids_sha256"]
    assert stale["same_pool_b"] is True
    assert stale["initial_rubric_is_ground_truth"] is False

    monkeypatch.setattr(
        target,
        "score_pool",
        lambda *args, **kwargs: pytest.fail("completed cells must be reused without grading"),
    )
    resumed = target.score_regular_adjacent(
        run_dir=tmp_path,
        artifact_root=tmp_path / "artifacts",
        output_root=output,
        steps=(0, 3),
        judge_urls=("http://judge",),
    )
    assert resumed["cell_count"] == 3


def test_authorized_regular_inventory_is_exactly_31_cells():
    cells = target.authorized_cells(target.REGULAR_STEPS)

    expected = {(step, step) for step in target.REGULAR_STEPS}
    expected.update((step - 3, step) for step in target.REGULAR_STEPS if step > 0)
    assert len(cells) == 31
    assert set(cells) == expected


def test_explicit_historical_plan_is_exactly_100_cells(tmp_path):
    steps = [0, 3, 6, 9, 12, 13, 15, 16, 18, 21, 24, 27, 30, 32, 33, 34, 36, 39, 40, 42, 45, 48]
    cells = {(step, step) for step in steps}
    cells.update((steps[index - 1], step) for index, step in enumerate(steps) if index > 0)
    for anchor in (0, 9, 16, 32, 48):
        cells.update((anchor, step) for step in steps if step >= anchor)
    plan_path = tmp_path / "historical-100.json"
    _write_json(
        plan_path,
        {
            "schema_version": 1,
            "analysis": "historical_100_cell_reuse_matrix",
            "steps": steps,
            "cells": [
                {"evaluator_step": evaluator, "policy_step": policy}
                for evaluator, policy in sorted(cells, key=lambda cell: (cell[1], cell[0]))
            ],
        },
    )

    loaded_steps, loaded_cells, artifact = target.load_cell_plan(plan_path)

    assert loaded_steps == tuple(steps)
    assert len(loaded_cells) == 100
    assert set(loaded_cells) == cells
    assert (13, 13) in loaded_cells
    assert (13, 15) in loaded_cells
    assert (9, 45) in loaded_cells
    assert (48, 48) in loaded_cells
    assert artifact["sha256"] == sha256_file(plan_path)


@pytest.mark.parametrize(
    "cells, match",
    [
        ([{"evaluator_step": 3, "policy_step": 0}], "after policy_step"),
        ([{"evaluator_step": 0, "policy_step": 4}], "outside steps"),
        (
            [
                {"evaluator_step": 0, "policy_step": 3},
                {"evaluator_step": 0, "policy_step": 3},
            ],
            "non-empty and unique",
        ),
    ],
)
def test_explicit_cell_plan_rejects_invalid_cells(tmp_path, cells, match):
    plan_path = tmp_path / "invalid.json"
    _write_json(plan_path, {"schema_version": 1, "steps": [0, 3], "cells": cells})

    with pytest.raises(target.ProbeAdjacentScoringError, match=match):
        target.load_cell_plan(plan_path)


def test_missing_future_checkpoint_stays_explicitly_pending(tmp_path, monkeypatch):
    monkeypatch.setattr(
        target,
        "_pool_b_readiness",
        lambda root, step: (step != 48, "ready" if step != 48 else "pool_b_provenance_missing"),
    )
    monkeypatch.setattr(
        target,
        "_evaluator_readiness",
        lambda root, step: (step != 48, "ready" if step != 48 else "fresh_rubric_status_missing"),
    )

    ready, blocked = target._scan_ready_cells(tmp_path, [(45, 45), (45, 48), (48, 48)], set())

    assert ready == [(45, 45)]
    assert blocked == [
        {
            "evaluator_step": 45,
            "policy_step": 48,
            "pool_b": "pool_b_provenance_missing",
            "evaluator": "ready",
        },
        {
            "evaluator_step": 48,
            "policy_step": 48,
            "pool_b": "pool_b_provenance_missing",
            "evaluator": "fresh_rubric_status_missing",
        },
    ]


def test_unready_fresh_does_not_block_ready_stale(tmp_path, monkeypatch):
    monkeypatch.setattr(target, "_pool_b_readiness", lambda root, step: (True, "ready"))
    monkeypatch.setattr(
        target,
        "_evaluator_readiness",
        lambda root, step: (step == 0, "ready" if step == 0 else "fresh_rubric_running"),
    )

    ready, blocked = target._scan_ready_cells(tmp_path, [(3, 3), (0, 3)], set())

    assert ready == [(0, 3)]
    assert blocked == [
        {
            "evaluator_step": 3,
            "policy_step": 3,
            "pool_b": "ready",
            "evaluator": "fresh_rubric_running",
        }
    ]


def test_later_ready_cell_bypasses_earlier_blocked_cell(tmp_path, monkeypatch):
    monkeypatch.setattr(
        target,
        "_pool_b_readiness",
        lambda root, step: (step == 9, "ready" if step == 9 else "pool_b_not_committed"),
    )
    monkeypatch.setattr(
        target,
        "_evaluator_readiness",
        lambda root, step: (step == 6, "ready" if step == 6 else "fresh_rubric_running"),
    )

    ready, blocked = target._scan_ready_cells(tmp_path, [(0, 3), (9, 9), (6, 9)], set())

    assert ready == [(6, 9)]
    assert {(item["evaluator_step"], item["policy_step"]) for item in blocked} == {
        (0, 3),
        (9, 9),
    }


def test_readiness_fails_closed_for_committed_missing_inputs(tmp_path):
    response_dir = tmp_path / "responses/checkpoint-000003"
    _write_json(
        response_dir / "provenance.json",
        {"selected_pools": ["probe_B"], "pool_counts": {"probe_B": 1600}},
    )
    with pytest.raises(target.ProbeAdjacentScoringError, match="file is missing"):
        target._pool_b_readiness(tmp_path, 3)

    rubric_dir = tmp_path / "rubrics/checkpoint-000003"
    _write_json(rubric_dir / "status.json", {"state": "complete"})
    with pytest.raises(target.ProbeAdjacentScoringError, match="invocation.json is missing"):
        target._evaluator_readiness(tmp_path, 3)


@pytest.mark.parametrize("steps", [(), (1,), (-3,), (0, 3, 3)])
def test_regular_checkpoint_schedule_is_strict(tmp_path, monkeypatch, steps):
    monkeypatch.setattr(target, "load_run_contract", lambda path: None)
    with pytest.raises(target.ProbeAdjacentScoringError):
        target.score_regular_adjacent(
            run_dir=tmp_path,
            artifact_root=tmp_path,
            output_root=tmp_path / "out",
            steps=steps,
            judge_urls=("http://judge",),
        )


def _endpoint_observation(*, created=1, permission="old", root="/models/rev", version="0.19.1"):
    return {
        "url": "http://judge",
        "model": {
            "id": "Qwen/Qwen3-32B",
            "root": root,
            "max_model_len": 32768,
            "created": created,
            "permission": [{"id": permission}],
        },
        "version": {"version": version},
    }


def test_endpoint_receipt_resume_ignores_volatile_fields(tmp_path):
    path = tmp_path / "judge_endpoints.json"
    original = [_endpoint_observation()]
    target._publish_endpoint_identities(path, original)

    target._publish_endpoint_identities(
        path, [_endpoint_observation(created=999, permission="new")]
    )

    assert read_json(path) == original


@pytest.mark.parametrize(
    "changed",
    [
        _endpoint_observation(root="/models/other"),
        _endpoint_observation(version="0.20.0"),
    ],
)
def test_endpoint_receipt_resume_rejects_stable_change(tmp_path, changed):
    path = tmp_path / "judge_endpoints.json"
    target._publish_endpoint_identities(path, [_endpoint_observation()])

    with pytest.raises(target.ProbeAdjacentScoringError, match="stable identity changed"):
        target._publish_endpoint_identities(path, [changed])
