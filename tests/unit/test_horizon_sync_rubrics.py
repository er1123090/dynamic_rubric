from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any

import pytest

from dynamic_rubric.artifacts import ImmutableArtifactError, read_json, read_jsonl
from dynamic_rubric.cli import build_parser, dispatch
from dynamic_rubric.horizon.sync_rubrics import build_horizon_rubrics_sync


SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "configs" / "schemas"


class FakeResponses:
    def __init__(self, *, incomplete_once: bool = False) -> None:
        self.calls: list[dict[str, Any]] = []
        self.incomplete_once = incomplete_once
        self._lock = threading.Lock()

    def create(self, **kwargs):
        with self._lock:
            self.calls.append(kwargs)
            index = len(self.calls)
        assert kwargs["model"] == "gpt-5-mini"
        assert len(kwargs["extra_headers"]["Idempotency-Key"]) == 64
        schema_name = kwargs["text"]["format"]["name"]
        if self.incomplete_once and index == 1:
            return {
                "id": "response-incomplete-1",
                "status": "incomplete",
                "model": "gpt-5-mini-2025-08-07",
                "incomplete_details": {"reason": "max_output_tokens"},
                "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            }
        if schema_name == "horizon_extraction_v1":
            value = {"analysis": "no grounded addition", "new_criteria": []}
        else:
            assert schema_name == "horizon_dedup_v1"
            value = {"analysis": "nothing to deduplicate", "final_criteria": []}
        return {
            "id": f"response-{index}",
            "status": "completed",
            "model": "gpt-5-mini-2025-08-07",
            "output_text": json.dumps(value),
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        }


class FakeClient:
    def __init__(self, *, incomplete_once: bool = False) -> None:
        self.responses = FakeResponses(incomplete_once=incomplete_once)


def prompt() -> dict[str, Any]:
    return {
        "prompt_id": "p1",
        "messages": [{"role": "user", "content": "Question"}],
        "r0": {
            "criteria": [
                {
                    "criterion_id": "r0-1",
                    "criterion": "Answers correctly",
                    "importance_class": "essential",
                    "criterion_type": "quality",
                    "weight_units": 10,
                }
            ]
        },
    }


def test_two_stage_sync_is_resumable_and_records_usage(tmp_path: Path) -> None:
    client = FakeClient()
    rows = [
        {"prompt_id": "p1", "response_text": f"alpha response {index}"}
        for index in range(8)
    ]
    output = tmp_path / "rubrics.jsonl"
    state_root = tmp_path / "state"
    kwargs = {
        "client": client,
        "run_id": "sync-run-1",
        "prompts": [prompt()],
        "current_rows": rows,
        "control_rows": rows,
        "checkpoint_id": "step3",
        "pairing_seed": 7,
        "model": "gpt-5-mini",
        "extraction_schema": read_json(SCHEMA_ROOT / "horizon_extraction_v1.json"),
        "dedup_schema": read_json(SCHEMA_ROOT / "horizon_dedup_v1.json"),
        "output_path": output,
        "state_root": state_root,
        "max_workers": 4,
    }
    result = build_horizon_rubrics_sync(**kwargs)
    assert result["generation_method"] == "openai_sync_responses_two_stage"
    assert len(client.responses.calls) == 9
    assert len(list((state_root / "extraction" / "responses").glob("*.json"))) == 8
    assert len(list((state_root / "dedup" / "responses").glob("*.json"))) == 1
    assert read_json(state_root / "extraction" / "status.json")["usage"] == {
        "input_tokens": 80,
        "output_tokens": 40,
        "total_tokens": 120,
    }
    assert read_json(state_root / "dedup" / "status.json")["status"] == "completed"
    cost_status = read_json(state_root / "cost_status.json")
    assert cost_status["attempt_count"] == 9
    assert cost_status["completed_attempts"] == 9
    assert cost_status["incomplete_attempts"] == 0
    assert cost_status["cost"]["estimated_cost_usd"] == pytest.approx(0.0001125)
    assert read_jsonl(output)[0]["extension"] == []
    assert "api_key" not in json.dumps(read_json(state_root / "result.json")).casefold()

    assert build_horizon_rubrics_sync(**kwargs) == result
    assert len(client.responses.calls) == 9

    changed = {**kwargs, "current_rows": [{**rows[0], "response_text": "changed"}, *rows[1:]]}
    with pytest.raises(ImmutableArtifactError, match="invocation.json"):
        build_horizon_rubrics_sync(**changed)


def test_sync_retries_only_max_output_incomplete_and_accounts_attempt(tmp_path: Path) -> None:
    client = FakeClient(incomplete_once=True)
    rows = [
        {"prompt_id": "p1", "response_text": f"alpha response {index}"}
        for index in range(8)
    ]
    state_root = tmp_path / "state"
    result = build_horizon_rubrics_sync(
        client=client,
        run_id="sync-incomplete-retry",
        prompts=[prompt()],
        current_rows=rows,
        control_rows=rows,
        checkpoint_id="step3",
        pairing_seed=7,
        model="gpt-5-mini",
        extraction_schema=read_json(SCHEMA_ROOT / "horizon_extraction_v1.json"),
        dedup_schema=read_json(SCHEMA_ROOT / "horizon_dedup_v1.json"),
        output_path=tmp_path / "rubrics.jsonl",
        state_root=state_root,
        max_workers=1,
    )
    assert result["generation_method"] == "openai_sync_responses_two_stage"
    assert len(client.responses.calls) == 10
    assert client.responses.calls[0]["max_output_tokens"] == 8192
    assert client.responses.calls[1]["max_output_tokens"] == 16384
    assert (
        client.responses.calls[0]["extra_headers"]["Idempotency-Key"]
        != client.responses.calls[1]["extra_headers"]["Idempotency-Key"]
    )
    assert len(list((state_root / "extraction" / "attempts").glob("*.json"))) == 9
    assert read_json(state_root / "extraction" / "status.json")["usage"] == {
        "input_tokens": 90,
        "output_tokens": 45,
        "total_tokens": 135,
    }
    cost_status = read_json(state_root / "cost_status.json")
    assert cost_status["attempt_count"] == 10
    assert cost_status["completed_attempts"] == 9
    assert cost_status["incomplete_attempts"] == 1
    assert cost_status["cost"]["estimated_cost_usd"] == pytest.approx(0.000125)


def test_dedup_schema_uses_only_supported_structured_output_keywords() -> None:
    schema = read_json(SCHEMA_ROOT / "horizon_dedup_v1.json")
    source_ids = schema["properties"]["final_criteria"]["items"]["properties"][
        "source_candidate_ids"
    ]
    assert "uniqueItems" not in source_ids


def test_cli_uses_sync_mode_from_horizon_config(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_sync_runner(**kwargs):
        captured.update(kwargs)
        return {"generation_method": "openai_sync_responses_two_stage"}

    monkeypatch.setattr(
        "dynamic_rubric.cli.build_horizon_rubrics_sync_from_files", fake_sync_runner
    )
    args = build_parser().parse_args(
        [
            "build-horizon-rubrics",
            "--config",
            str(Path(__file__).resolve().parents[2] / "configs" / "horizon_medicine.yaml"),
            "--run-id",
            "sync-run",
            "--prompts",
            str(tmp_path / "prompts.jsonl"),
            "--current-pool",
            str(tmp_path / "current.jsonl"),
            "--control-pool",
            str(tmp_path / "control.jsonl"),
            "--checkpoint-id",
            "step3",
            "--output",
            str(tmp_path / "rubrics.jsonl"),
            "--sync-concurrency",
            "7",
        ]
    )
    result = dispatch(args)
    assert result["generation_method"] == "openai_sync_responses_two_stage"
    assert captured["model"] == "gpt-5-mini"
    assert captured["max_workers"] == 7
    assert captured["state_root"].parts[-3:] == ("checkpoints", "step3", "sync")
