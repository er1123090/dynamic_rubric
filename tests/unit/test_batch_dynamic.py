from __future__ import annotations

import json

import pytest

from dynamic_rubric.batch_dynamic import (
    DYNAMIC_MAX_OUTPUT_TOKENS,
    _normalize_batch_row,
    _replicate_ids,
    _responses_payload,
    _shard,
    control_policy_step,
    dynamic_batch_prefix,
    dynamic_batch_stage,
)
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.pipeline import StageError


def test_batch_responses_payload_contract() -> None:
    request = GenerationRequest(
        "p",
        ({"role": "user", "content": "x"},),
        "dynamic",
        7,
        json_schema={"type": "object"},
        schema_name="candidate",
        reasoning_effort="medium",
    )
    assert _responses_payload("gpt-5-mini", request) == {
        "model": "gpt-5-mini",
        "input": [{"role": "user", "content": "x"}],
        "max_output_tokens": 1024,
        "reasoning": {"effort": "medium"},
        "text": {
            "format": {
                "type": "json_schema",
                "name": "candidate",
                "schema": {"type": "object"},
                "strict": True,
            }
        },
    }


def test_batch_sharding_preserves_order_and_all_requests(monkeypatch) -> None:
    monkeypatch.setattr("dynamic_rubric.batch_dynamic.MAX_BATCH_FILE_BYTES", 220)
    lines = [{"custom_id": str(index), "body": {"value": "x" * 50}} for index in range(7)]
    shards = _shard(lines)
    assert len(shards) > 1
    assert [row for shard in shards for row in shard] == lines
    assert all(
        sum(
            len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()) + 1
            for row in shard
        )
        <= 220
        for shard in shards
    )


def test_batch_sharding_accepts_explicit_per_workflow_limits() -> None:
    lines = [{"custom_id": str(index), "body": {"value": "x" * 20}} for index in range(5)]
    shards = _shard(lines, max_file_bytes=10_000, max_requests=2)
    assert [len(shard) for shard in shards] == [2, 2, 1]
    assert [row for shard in shards for row in shard] == lines


def test_replicate_subset_is_deterministic() -> None:
    assert _replicate_ids("prompt-a", 0.20) == _replicate_ids("prompt-a", 0.20)
    assert _replicate_ids("prompt", 0.0) == ("A",)
    assert _replicate_ids("prompt", 1.0) == ("A", "B")


def test_normalize_batch_output_checks_model_and_schema() -> None:
    identity = {"custom_id": "c", "prompt_id": "p", "policy_step": 3}
    row = {
        "id": "batch-request",
        "custom_id": "c",
        "error": None,
        "response": {
            "status_code": 200,
            "request_id": "header-id",
            "body": {
                "id": "resp-id",
                "model": "gpt-5-mini-snapshot",
                "output_text": json.dumps(
                    {"criteria": [{"text": "A sufficiently long criterion", "rationale": "r"}]}
                ),
                "usage": {"input_tokens": 10},
            },
        },
    }
    normalized = _normalize_batch_row(row, identity, "gpt-5-mini")
    assert normalized["criteria"][0]["text"] == "A sufficiently long criterion"
    assert normalized["provider_call"]["returned_model"] == "gpt-5-mini-snapshot"
    assert normalized["provider_call"]["request_id"] == "resp-id"


def test_dynamic_prev_uses_immediately_previous_policy_control() -> None:
    assert control_policy_step("dynamic_prev_budgeted", 1) == 0
    assert control_policy_step("dynamic_prev_budgeted", 2) == 1
    assert control_policy_step("dynamic_prev_budgeted", 50) == 49
    assert control_policy_step("dynamic_fixed_budgeted", 50) == 0


def test_dynamic_prev_has_isolated_batch_namespace() -> None:
    assert dynamic_batch_stage("dynamic_fixed_budgeted") == "dynamic-batch"
    assert dynamic_batch_stage("dynamic_prev_budgeted") == "dynamic-prev-batch"
    assert dynamic_batch_prefix("dynamic_prev_budgeted") == "dynamic-prev"


def test_batch_dynamic_output_budget_leaves_room_for_reasoning() -> None:
    assert DYNAMIC_MAX_OUTPUT_TOKENS == 8192


def test_normalize_batch_output_rejects_incomplete_response() -> None:
    identity = {"custom_id": "c", "prompt_id": "p", "policy_step": 3}
    row = {
        "id": "batch-request",
        "custom_id": "c",
        "error": None,
        "response": {
            "status_code": 200,
            "body": {
                "id": "resp-id",
                "model": "gpt-5-mini-snapshot",
                "status": "incomplete",
                "incomplete_details": {"reason": "max_output_tokens"},
                "output_text": '{"criteria":[]}',
            },
        },
    }
    with pytest.raises(StageError, match="max_output_tokens"):
        _normalize_batch_row(row, identity, "gpt-5-mini")
