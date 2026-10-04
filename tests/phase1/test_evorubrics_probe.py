from __future__ import annotations

import pytest

from dynamic_rubric.phase1.evorubrics_probe import (
    EvoProbeError,
    UpstreamJudge,
    _judgments,
    _tensor_digest,
    compute_policy_kl_artifacts,
    parse_pairs,
    sampled_kl_summary,
    validate_live_smoke,
)


def test_pair_selection_adds_required_fresh_cell():
    available = ((0, 0), (0, 3), (3, 3))
    assert parse_pairs("0:3", available) == ((0, 0), (0, 3), (3, 3))
    assert parse_pairs("all", available) == ((0, 0), (0, 3), (3, 3))
    with pytest.raises(EvoProbeError, match="not committed"):
        parse_pairs("1:3", available)


def test_upstream_judge_uses_training_sized_four_answer_chunks():
    class Client:
        def __init__(self):
            self.calls = []

        def batch_evaluate_multiple_answers(self, **kwargs):
            self.calls.append(kwargs)
            return [{"answer": answer} for answer in kwargs["answers"]]

    judge = object.__new__(UpstreamJudge)
    judge.client = Client()
    answers = [f"answer-{index}" for index in range(16)]
    results = judge.grade(
        question="question",
        answers=answers,
        criteria=[{"criterion": "criterion", "weight": 1}],
    )

    assert [len(call["answers"]) for call in judge.client.calls] == [4, 4, 4, 4]
    assert [row["answer"] for row in results] == answers


def test_judge_receipt_requires_every_boolean_criterion():
    criteria = (
        {"criterion_id": "c0", "weight": 3},
        {"criterion_id": "c1", "weight": -2},
    )
    result = {
        "details": {
            "rubric_scores": [
                {"rubric_index": 0, "criteria_met": True},
                {"rubric_index": 1, "criteria_met": False},
            ]
        }
    }
    judgments = _judgments(result, criteria)
    assert [(row.criterion_id, row.weight, row.grade) for row in judgments] == [
        ("c0", 3.0, True),
        ("c1", -2.0, False),
    ]
    result["details"]["rubric_scores"].pop()
    with pytest.raises(EvoProbeError, match="missing criterion"):
        _judgments(result, criteria)


def test_cached_grade_rejects_non_boolean_criterion():
    from dynamic_rubric.phase1.evorubrics_probe import _record_from_json

    with pytest.raises(EvoProbeError, match="must be boolean"):
        _record_from_json(
            {
                "policy_step": 1,
                "evaluator_step": 0,
                "policy_checkpoint": "theta-1",
                "evaluator_checkpoint": "psi-0",
                "prompt_id": "p1",
                "response_id": "r0",
                "rubric_id": "rubric-0",
                "judgments": [{"criterion_id": "c0", "weight": 1, "grade": "true"}],
                "parse_ok": True,
            }
        )


def test_sampled_kl_masks_prompts_and_uses_prompt_clusters():
    report = sampled_kl_summary(
        (
            {
                "prompt_id": "p1",
                "current_logprob_sum": -2,
                "stale_logprob_sum": -4,
                "response_token_count": 2,
            },
            {
                "prompt_id": "p1",
                "current_logprob_sum": -3,
                "stale_logprob_sum": -4,
                "response_token_count": 1,
            },
            {
                "prompt_id": "p2",
                "current_logprob_sum": -8,
                "stale_logprob_sum": -8,
                "response_token_count": 4,
            },
        )
    )
    assert report["token_weighted_sampled_kl"] == pytest.approx(3 / 7)
    assert report["prompt_balanced_sampled_kl_mean"] == pytest.approx(0.5)
    assert report["prompt_count"] == 2
    assert report["prompt_tokens_excluded"] is True


def test_missing_cache_message_is_explicit(tmp_path):
    from dynamic_rubric.phase1.evorubrics_probe import _load_cached

    with pytest.raises(EvoProbeError, match="missing cached artifact"):
        _load_cached(tmp_path / "responses.json", expected=16)


def test_policy_kl_scores_identical_saved_tokens_and_caches(tmp_path):
    from dynamic_rubric.artifacts import write_json_atomic

    sequences = [
        {
            "prompt_id": "p1",
            "response_id": f"r{index}",
            "prompt_token_ids": [1, 2],
            "response_token_ids": [3, 4],
        }
        for index in range(16)
    ]
    write_json_atomic(tmp_path / "audit/fixed_probe/responses/theta_000002.json", sequences)

    class Backend:
        def __init__(self):
            self.calls = []

        def score_logprobs(self, *, adapter_path, adapter_name, sequences):
            self.calls.append((adapter_path, [row["response_id"] for row in sequences]))
            value = -2.0 if adapter_path == "current" else -4.0
            return [
                {
                    "prompt_id": row["prompt_id"],
                    "response_id": row["response_id"],
                    "logprob_sum": value,
                    "response_token_count": 2,
                }
                for row in sequences
            ]

    backend = Backend()
    checkpoints = {
        0: {"policy": {"adapter_path": "stale"}},
        2: {"policy": {"adapter_path": "current"}},
    }
    first = compute_policy_kl_artifacts(
        run_root=tmp_path,
        pairs=((0, 2),),
        checkpoint_rows=checkpoints,
        probe_prompt_count=1,
        backend=backend,
    )
    second = compute_policy_kl_artifacts(
        run_root=tmp_path,
        pairs=((0, 2),),
        checkpoint_rows=checkpoints,
        probe_prompt_count=1,
        backend=backend,
    )
    assert first == second
    assert first[0]["summary"]["prompt_balanced_sampled_kl_mean"] == 1
    assert len(backend.calls) == 2
    assert backend.calls[0][1] == backend.calls[1][1]


def test_live_smoke_validator_writes_only_after_all_real_proofs(tmp_path, monkeypatch):
    import json

    from dynamic_rubric.hashing import sha256_file, sha256_json

    def put(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    put("config.resolved.json", {"method": "evorubrics"})
    semantic_identity = {"domain": "medicine", "method": "evorubrics", "seed": 11}
    import hashlib

    from dynamic_rubric.hashing import canonical_json_bytes

    semantic_hash = hashlib.sha256(canonical_json_bytes(semantic_identity)).hexdigest()
    launch_scope = {
        "mode": "smoke",
        "upstream_config_sha256": "upstream-hash",
        "expected_steps": 1,
        "expected_prompt_exposures": 2,
    }
    put(
        "launch_spec.json",
        {
            "phase1_config_sha256": sha256_json({"method": "evorubrics"}),
            **launch_scope,
        },
    )
    put(
        "run_provenance.json",
        {
            "scope": launch_scope,
            "semantic_identity": semantic_identity,
            "semantic_identity_sha256": semantic_hash,
            "actual_training": True,
        },
    )
    put("training_complete.json", {"status": "training_passed", "actual_training": True})
    torch = pytest.importorskip("torch")
    save_file = pytest.importorskip("safetensors.torch").save_file

    required = ("responses", "response_mask", "advantages", "old_log_probs", "ref_log_prob")
    for adapter in ("policy_llm", "rubrics_generator"):
        tensors = {key: torch.tensor([1.0]) for key in required}
        relative = f"audit/advantages/step_000001_{adapter}.safetensors"
        binary = tmp_path / relative
        binary.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensors, binary)
        put(
            f"audit/advantages/step_000001_{adapter}.json",
            {
                "schema_version": 1,
                "global_step": 1,
                "input_step": 0,
                "adapter": adapter,
                "artifact": {
                    "path": relative,
                    "sha256": sha256_file(binary),
                    "bytes": binary.stat().st_size,
                    "format": "safetensors",
                },
                "tensors": {
                    key: {
                        "shape": list(tensor.shape),
                        "dtype": str(tensor.dtype),
                        "sha256": _tensor_digest(tensor),
                    }
                    for key, tensor in tensors.items()
                },
                "tensor_keys": sorted(tensors),
                "uid": ["u1"],
                "uid_count": 1,
                "actual_training": True,
            },
        )
    put("judge-preflight.json", {"status": "passed", "actual_remote_call": True})
    grade_records = [
        {
            "policy_step": 1,
            "evaluator_step": 0,
            "policy_checkpoint": "theta-1",
            "evaluator_checkpoint": "psi-0",
            "prompt_id": "p1",
            "response_id": f"r{response_index}",
            "rubric_id": f"rubric-{rubric_index}",
            "judgments": [{"criterion_id": "c0", "weight": 1, "grade": True}],
            "parse_ok": True,
        }
        for response_index in range(16)
        for rubric_index in range(4)
    ]
    raw_details = [
        {
            "prompt_id": "p1",
            "rubric_id": f"rubric-{rubric_index}",
            "response_ids": [f"r{index}" for index in range(16)],
            "criteria": [{"criterion_id": "c0", "weight": 1}],
            "results": [
                {"details": {"rubric_scores": [{"rubric_index": 0, "criteria_met": True}]}}
                for _ in range(16)
            ],
        }
        for rubric_index in range(4)
    ]
    put(
        "audit/fixed_probe/grades/psi_000000_theta_000001.json",
        {"records": grade_records, "raw_grading_details": raw_details},
    )
    put(
        "audit/fixed_probe/analysis.json",
        {
            "cells": [
                {
                    "evaluator_step": tau,
                    "policy_step": step,
                    "status": "complete",
                    "prompt_count": 1,
                }
                for tau, step in ((0, 0), (0, 1), (1, 1))
            ]
        },
    )
    kl_receipts = [
        {
            "prompt_id": "p1",
            "response_id": f"r{index}",
            "current_logprob_sum": -2.0,
            "stale_logprob_sum": -2.2,
            "response_token_count": 2,
        }
        for index in range(16)
    ]
    kl_summary = sampled_kl_summary(kl_receipts)
    put(
        "audit/fixed_probe/policy_kl/theta_000000_to_000001.json",
        {
            "stale_policy_step": 0,
            "current_policy_step": 1,
            "response_policy_step": 1,
            "summary": kl_summary,
            "receipts": kl_receipts,
        },
    )
    reload_roles = {}
    for role in ("policy", "generator"):
        adapter_path = tmp_path / f"checkpoints/{role}"
        adapter_path.mkdir(parents=True)
        weights = adapter_path / "adapter_model.safetensors"
        adapter_config = adapter_path / "adapter_config.json"
        optimizer = adapter_path / "optimizer.pt"
        weights.write_bytes(b"weights")
        adapter_config.write_text("{}")
        optimizer.write_bytes(b"optimizer")
        reload_roles[role] = {
            "adapter_path": str(adapter_path),
            "weights_sha256": sha256_file(weights),
            "config_sha256": sha256_file(adapter_config),
            "optimizer_path": str(optimizer),
            "optimizer_sha256": sha256_file(optimizer),
        }
    put(
        "resume_verified.json",
        {
            "status": "passed",
            "actual_reload": True,
            "completed_main_call": True,
            "strict_optimizer_reload": True,
            "checkpoint_step": 1,
            "final_checkpoint_step": 1,
            "semantic_identity_sha256": semantic_hash,
            "roles": reload_roles,
        },
    )
    monkeypatch.setattr(
        "dynamic_rubric.phase1.evorubrics_probe.discover_checkpoint_pairs",
        lambda root: {"steps": [0, 1]},
    )
    result = validate_live_smoke(tmp_path)
    assert result["status"] == "passed"
    assert (tmp_path / "smoke_complete.json").is_file()


def test_live_smoke_validator_rejects_synthetic_training(tmp_path):
    import json

    from dynamic_rubric.hashing import sha256_json

    (tmp_path / "config.resolved.json").write_text("{}")
    semantic_identity = {"method": "evorubrics"}
    import hashlib

    from dynamic_rubric.hashing import canonical_json_bytes

    semantic_hash = hashlib.sha256(canonical_json_bytes(semantic_identity)).hexdigest()
    scope = {
        "mode": "smoke",
        "upstream_config_sha256": "u",
        "expected_steps": 1,
        "expected_prompt_exposures": 2,
    }
    (tmp_path / "launch_spec.json").write_text(
        json.dumps(
            {
                "phase1_config_sha256": sha256_json({}),
                **scope,
            }
        )
    )
    (tmp_path / "run_provenance.json").write_text(
        json.dumps(
            {
                "scope": scope,
                "semantic_identity": semantic_identity,
                "semantic_identity_sha256": semantic_hash,
                "actual_training": True,
            }
        )
    )
    (tmp_path / "training_complete.json").write_text(
        json.dumps({"status": "training_passed", "actual_training": False})
    )
    with pytest.raises(EvoProbeError, match="actual training_passed"):
        validate_live_smoke(tmp_path)
    assert not (tmp_path / "smoke_complete.json").exists()
