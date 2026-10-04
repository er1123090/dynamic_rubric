from __future__ import annotations

import json
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Any

import pytest

import dynamic_rubric.horizon.batch_rubrics as batch_rubrics
from dynamic_rubric.artifacts import ImmutableArtifactError, read_json, read_jsonl
from dynamic_rubric.cli import build_parser, dispatch
from dynamic_rubric.horizon.batch_rubrics import build_horizon_rubrics_batch
from dynamic_rubric.pipeline import StageError


SCHEMA_ROOT = Path(__file__).resolve().parents[2] / "configs" / "schemas"


class FakeFiles:
    def __init__(self) -> None:
        self.uploads: dict[str, bytes] = {}
        self.downloads: dict[str, bytes] = {}
        self.retrieves: dict[str, int] = {}

    def create(self, *, file, purpose: str):
        assert purpose == "batch"
        identity = f"file-{len(self.uploads) + 1}"
        self.uploads[identity] = file.read()
        return {"id": identity}

    def retrieve(self, file_id: str):
        count = self.retrieves.get(file_id, 0) + 1
        self.retrieves[file_id] = count
        return {"id": file_id, "status": "uploaded" if count == 1 else "processed"}

    def content(self, file_id: str):
        return SimpleNamespace(content=self.downloads[file_id])


class FakeBatches:
    def __init__(self, files: FakeFiles) -> None:
        self.files = files
        self.jobs: dict[str, dict[str, Any]] = {}
        self.creates: list[dict[str, Any]] = []
        self.retrieves: dict[str, int] = {}
        self.fail_first_visibility = False
        self.incomplete_first_extraction = False
        self.incomplete_first_dedup = False
        self.candidate_text = "States diagnosis"

    def create(self, **kwargs):
        self.creates.append(kwargs)
        assert kwargs["endpoint"] == "/v1/responses"
        assert kwargs["completion_window"] == "24h"
        input_rows = [
            json.loads(line)
            for line in self.files.uploads[kwargs["input_file_id"]].splitlines()
            if line.strip()
        ]
        phase = kwargs["metadata"]["phase"]
        output_rows = []
        for index, request in enumerate(input_rows):
            assert request["method"] == "POST"
            assert request["url"] == "/v1/responses"
            assert request["body"]["model"] == "gpt-5-mini"
            if phase.startswith("extraction"):
                value = {
                    "analysis": "grounded distinction",
                    "new_criteria": [
                        {
                            "candidate_id": "candidate-1",
                            "quote": "alpha",
                            "criterion": self.candidate_text,
                            "weight": 5,
                            "importance_class": "important",
                            "criterion_type": "quality",
                        }
                    ],
                }
            else:
                user_content = str(request["body"]["input"][-1]["content"])
                source_candidate_ids = re.findall(r'"candidate_id": "(hc-[^"]+)"', user_content)
                assert len(source_candidate_ids) == 8
                value = {
                    "analysis": "retain source wording",
                    "final_criteria": [
                        {
                            "criterion": "States diagnosis",
                            "source_candidate_ids": [
                                *source_candidate_ids,
                                source_candidate_ids[0],
                                "unknown-source-candidate",
                            ],
                        },
                        {
                            "criterion": "Hallucinated criterion",
                            "source_candidate_ids": ["unknown-source-candidate"],
                        },
                    ],
                }
            response_body = {
                "id": f"response-{phase}-{index}",
                "status": "completed",
                "model": "gpt-5-mini-2025-08-07",
                "output_text": json.dumps(value),
                "usage": {"input_tokens": 10, "output_tokens": 5},
            }
            if self.incomplete_first_extraction and phase == "extraction" and index == 0:
                response_body = {
                    "id": f"response-{phase}-{index}",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "max_output_tokens": request["body"]["max_output_tokens"],
                    "model": "gpt-5-mini-2025-08-07",
                    "usage": {"input_tokens": 10, "output_tokens": 4096},
                }
            if self.incomplete_first_dedup and phase == "dedup" and index == 0:
                response_body = {
                    "id": f"response-{phase}-{index}",
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                    "max_output_tokens": request["body"]["max_output_tokens"],
                    "model": "gpt-5-mini-2025-08-07",
                    "usage": {"input_tokens": 10, "output_tokens": 4096},
                }
            output_rows.append(
                {
                    "id": f"batch-request-{phase}-{index}",
                    "custom_id": request["custom_id"],
                    "response": {"status_code": 200, "body": response_body},
                    "error": None,
                }
            )
        batch_id = f"batch-{len(self.jobs) + 1}"
        output_id = f"output-{len(self.jobs) + 1}"
        self.files.downloads[output_id] = b"".join(
            json.dumps(row, sort_keys=True).encode() + b"\n" for row in output_rows
        )
        self.jobs[batch_id] = {
            "id": batch_id,
            "status": "in_progress",
            "output_file_id": output_id,
            "error_file_id": None,
        }
        return {"id": batch_id, "status": "validating"}

    def retrieve(self, batch_id: str):
        if self.fail_first_visibility and batch_id == "batch-1":
            return {
                "id": batch_id,
                "status": "failed",
                "errors": {
                    "data": [
                        {
                            "code": "invalid_request",
                            "param": "file_id",
                            "message": "Cannot find file file-1, or organization has no access",
                        }
                    ]
                },
            }
        self.retrieves[batch_id] = self.retrieves.get(batch_id, 0) + 1
        job = dict(self.jobs[batch_id])
        if self.retrieves[batch_id] >= 2:
            job["status"] = "completed"
        return job


class FakeClient:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches(self.files)


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


def test_openai_request_throttle_is_shared_through_a_lock_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    clock = [100.0]
    sleeps: list[float] = []

    def fake_sleep(delay: float) -> None:
        sleeps.append(delay)
        clock[0] += delay

    monkeypatch.setenv("DYNAMIC_RUBRIC_OPENAI_REQUESTS_PER_MINUTE", "120")
    monkeypatch.setenv("DYNAMIC_RUBRIC_OPENAI_RATE_LIMIT_PATH", str(tmp_path / "rate-limit.lock"))
    monkeypatch.setattr(batch_rubrics.time, "time", lambda: clock[0])
    monkeypatch.setattr(batch_rubrics.time, "sleep", fake_sleep)

    batch_rubrics._throttle_openai_request()
    batch_rubrics._throttle_openai_request()

    assert sleeps == [0.5]


def test_two_stage_batch_is_resumable_and_dedup_waits_for_extraction(tmp_path: Path) -> None:
    client = FakeClient()
    sleeps: list[float] = []
    rows = [{"prompt_id": "p1", "response_text": f"alpha response {index}"} for index in range(8)]
    output = tmp_path / "rubrics.jsonl"
    state_root = tmp_path / "state"
    kwargs = {
        "client": client,
        "run_id": "run-1",
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
        "poll_interval_seconds": 0.25,
        "sleep": sleeps.append,
    }
    result = build_horizon_rubrics_batch(**kwargs)
    assert result["generation_method"] == "openai_batch_responses_two_stage"
    assert [call["metadata"]["phase"] for call in client.batches.creates] == [
        "extraction",
        "dedup",
    ]
    assert sleeps == [1.0, 0.25, 1.0, 0.25]
    assert len(read_jsonl(state_root / "extraction" / "request_map.jsonl")) == 8
    assert len(read_jsonl(state_root / "dedup" / "request_map.jsonl")) == 1
    assert {
        row["body"]["max_output_tokens"]
        for row in read_jsonl(state_root / "dedup" / "inputs" / "dedup-001.jsonl")
    } == {16384}
    assert {
        row["body"]["max_output_tokens"]
        for row in read_jsonl(state_root / "extraction" / "inputs" / "extraction-001.jsonl")
    } == {8192}
    rubric = read_jsonl(output)[0]
    assert rubric["extension"][0]["text"] == "States diagnosis"
    assert rubric["extension"][0]["weight_units"] == 7
    assert len(rubric["extension"][0]["source_candidate_ids"]) == 8
    assert ["unresolved", "unknown_source_candidate_ids"] in rubric["rejected"]
    assert rubric["pool_b_inputs"] == []
    assert "api_key" not in json.dumps(read_json(state_root / "result.json")).casefold()

    # Completed result is a no-network resume boundary.
    assert build_horizon_rubrics_batch(**kwargs) == result
    assert len(client.batches.creates) == 2

    changed = {**kwargs, "current_rows": [{**rows[0], "response_text": "changed"}, *rows[1:]]}
    with pytest.raises(ImmutableArtifactError, match="invocation.json"):
        build_horizon_rubrics_batch(**changed)


def test_batch_retries_only_max_output_incomplete_requests(tmp_path: Path) -> None:
    client = FakeClient()
    client.batches.incomplete_first_extraction = True
    sleeps: list[float] = []
    rows = [{"prompt_id": "p1", "response_text": f"alpha response {index}"} for index in range(8)]
    result = build_horizon_rubrics_batch(
        client=client,
        run_id="run-max-output-retry",
        prompts=[prompt()],
        current_rows=rows,
        control_rows=rows,
        checkpoint_id="step3",
        pairing_seed=7,
        model="gpt-5-mini",
        extraction_schema=read_json(SCHEMA_ROOT / "horizon_extraction_v1.json"),
        dedup_schema=read_json(SCHEMA_ROOT / "horizon_dedup_v1.json"),
        output_path=tmp_path / "rubrics.jsonl",
        state_root=tmp_path / "state",
        poll_interval_seconds=0.25,
        sleep=sleeps.append,
    )
    assert result["generation_method"] == "openai_batch_responses_two_stage"
    assert [call["metadata"]["phase"] for call in client.batches.creates] == [
        "extraction",
        "extraction-max-output-16384",
        "dedup",
    ]
    retry_upload = client.batches.creates[1]["input_file_id"]
    retry_rows = [
        json.loads(line) for line in client.files.uploads[retry_upload].splitlines() if line.strip()
    ]
    assert len(retry_rows) == 1
    assert retry_rows[0]["body"]["max_output_tokens"] == 16384
    assert sleeps == [1.0, 0.25, 1.0, 0.25, 1.0, 0.25]


def test_default_change_reuses_legacy_prepared_stage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    client = FakeClient()
    rows = [{"prompt_id": "p1", "response_text": f"alpha response {index}"} for index in range(8)]
    output = tmp_path / "rubrics.jsonl"
    state_root = tmp_path / "state"
    kwargs = {
        "client": client,
        "run_id": "run-legacy-default",
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
        "poll_interval_seconds": 0.25,
        "sleep": lambda _: None,
    }
    monkeypatch.setattr(batch_rubrics, "MAX_OUTPUT_TOKENS", 4096)
    build_horizon_rubrics_batch(**kwargs)
    first_input = read_jsonl(state_root / "extraction" / "inputs" / "extraction-001.jsonl")
    assert first_input[0]["body"]["max_output_tokens"] == 4096

    output.unlink()
    (state_root / "result.json").unlink()
    monkeypatch.setattr(batch_rubrics, "MAX_OUTPUT_TOKENS", 8192)
    build_horizon_rubrics_batch(**kwargs)
    resumed_input = read_jsonl(state_root / "extraction" / "inputs" / "extraction-001.jsonl")
    assert resumed_input[0]["body"]["max_output_tokens"] == 4096
    assert len(client.batches.creates) == 2


def test_dedup_starts_at_16384_and_preserves_large_file_limit(tmp_path: Path) -> None:
    client = FakeClient()
    client.batches.candidate_text = "States diagnosis " + ("x" * 50_000)
    rows = [{"prompt_id": "p1", "response_text": f"alpha response {index}"} for index in range(8)]
    result = build_horizon_rubrics_batch(
        client=client,
        run_id="run-large-dedup",
        prompts=[prompt()],
        current_rows=rows,
        control_rows=rows,
        checkpoint_id="step3",
        pairing_seed=7,
        model="gpt-5-mini",
        extraction_schema=read_json(SCHEMA_ROOT / "horizon_extraction_v1.json"),
        dedup_schema=read_json(SCHEMA_ROOT / "horizon_dedup_v1.json"),
        output_path=tmp_path / "rubrics.jsonl",
        state_root=tmp_path / "state",
        poll_interval_seconds=0.25,
        sleep=lambda _: None,
    )
    assert result["generation_method"] == "openai_batch_responses_two_stage"
    assert [call["metadata"]["phase"] for call in client.batches.creates] == [
        "extraction",
        "dedup",
    ]
    dedup_upload = client.batches.creates[1]["input_file_id"]
    assert 300_000 < len(client.files.uploads[dedup_upload]) < 2_000_000


def test_file_visibility_failure_reuses_processed_upload_on_resume(tmp_path: Path) -> None:
    client = FakeClient()
    client.batches.fail_first_visibility = True
    sleeps: list[float] = []
    rows = [{"prompt_id": "p1", "response_text": f"alpha response {index}"} for index in range(8)]
    state_root = tmp_path / "state"
    kwargs = {
        "client": client,
        "run_id": "run-visibility-retry",
        "prompts": [prompt()],
        "current_rows": rows,
        "control_rows": rows,
        "checkpoint_id": "step3",
        "pairing_seed": 7,
        "model": "gpt-5-mini",
        "extraction_schema": read_json(SCHEMA_ROOT / "horizon_extraction_v1.json"),
        "dedup_schema": read_json(SCHEMA_ROOT / "horizon_dedup_v1.json"),
        "output_path": tmp_path / "rubrics.jsonl",
        "state_root": state_root,
        "poll_interval_seconds": 0.25,
        "sleep": sleeps.append,
    }

    with pytest.raises(StageError, match="terminal failure"):
        build_horizon_rubrics_batch(**kwargs)
    assert len(client.files.uploads) == 1
    assert len(client.batches.creates) == 1

    result = build_horizon_rubrics_batch(**kwargs)
    assert result["generation_method"] == "openai_batch_responses_two_stage"
    assert len(client.files.uploads) == 2
    assert len(client.batches.creates) == 3
    assert [call["metadata"]["phase"] for call in client.batches.creates] == [
        "extraction",
        "extraction",
        "dedup",
    ]
    receipt = read_json(state_root / "extraction" / "submission.json")
    assert receipt["submission_attempt"] == 2
    assert receipt["jobs"][0]["input_file_id"] == "file-1"
    assert sleeps == [1.0, 0.25, 1.0, 0.25]


def test_cli_build_horizon_rubrics_dispatches_to_batch_runner(monkeypatch, tmp_path: Path) -> None:
    captured = {}

    def fake_batch_runner(**kwargs):
        captured.update(kwargs)
        return {"generation_method": "openai_batch_responses_two_stage"}

    monkeypatch.setattr(
        "dynamic_rubric.cli.build_horizon_rubrics_batch_from_files", fake_batch_runner
    )
    parser = build_parser()
    args = parser.parse_args(
        [
            "build-horizon-rubrics",
            "--config",
            str(Path(__file__).resolve().parents[2] / "configs" / "horizon_medicine.yaml"),
            "--api-mode",
            "batch",
            "--run-id",
            "batch-run",
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
            "--batch-poll-interval-seconds",
            "2.5",
        ]
    )
    result = dispatch(args)
    assert result["generation_method"] == "openai_batch_responses_two_stage"
    assert captured["model"] == "gpt-5-mini"
    assert captured["poll_interval_seconds"] == 2.5
    assert captured["state_root"].parts[-3:] == ("checkpoints", "step3", "batch")
