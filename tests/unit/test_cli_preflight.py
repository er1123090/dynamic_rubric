from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pytest

from dynamic_rubric.cli import build_parser
from dynamic_rubric.config import config_from_mapping
from dynamic_rubric.inventory_validation import _expected_evaluation_rubrics
from dynamic_rubric.live_preflight import run_preflight
from dynamic_rubric.pipeline import PipelineContext, StageError
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.openai_responses import OpenAIResponsesAdapter, OpenAIResponsesError


def test_help_lists_all_canonical_stages_and_private_option_is_audit_only(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit) as help_exit:
        parser.parse_args(["--help"])
    assert help_exit.value.code == 0
    help_text = capsys.readouterr().out
    for stage in (
        "prepare-data",
        "preflight",
        "generate-static",
        "train-static",
        "replay-dynamic",
        "freeze-updater",
        "generate-bon",
        "score-proxy",
        "select-bon",
        "export-audit-package",
        "audit-gold",
        "analyze",
        "validate-inventory",
    ):
        assert stage in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["generate-static", "--config", "x", "--run-id", "r", "--private-gt", "secret"]
        )
    parsed = parser.parse_args(
        ["audit-gold", "--config", "x", "--run-id", "r", "--private-gt", "secret"]
    )
    assert parsed.private_gt == "secret"


def test_training_stage_validates_only_training_inputs() -> None:
    config = config_from_mapping(
        {
            "training": {"artifact_inputs": ["artifacts/static"]},
            "replay": {"modes": ["dynamic_fixed_budgeted", "refresh_only_budgeted"]},
        },
        stage="train-static",
    )
    assert config.training.reward_source == "static_r0_only"


def test_canonical_pilot_r0_is_static_only_in_evaluation_inventory() -> None:
    root = Path(__file__).resolve().parents[2]
    context = PipelineContext.create(
        root, root / "configs" / "pilot.yaml", "validate-inventory", "pilot-grid-contract"
    )
    expected = _expected_evaluation_rubrics(context, {"prompt"})["prompt"]
    assert "prompt:R_0" in expected
    assert len(expected) == 21
    assert not any(value != "prompt:R_0" and value.endswith(":R_0") for value in expected)


def test_failed_live_preflight_writes_no_durable_output(tmp_path: Path) -> None:
    config_path = tmp_path / "pilot.json"
    config_path.write_text(
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
            {"status": "blocked", "verl": {"commit": "UNRESOLVED"}, "live_gate": {"ready": False}}
        ),
        encoding="utf-8",
    )
    context = PipelineContext.create(tmp_path, config_path, "preflight", "blocked-run")
    with pytest.raises(StageError):
        run_preflight(context)
    assert not (tmp_path / "artifacts").exists()


class _HTTPResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value
        self.headers = {"x-request-id": str(value["id"])}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


def test_openai_returned_model_drift_hard_fails(tmp_path: Path, monkeypatch) -> None:
    responses = iter(
        [
            {
                "id": "req-1",
                "model": "gpt-5-mini-identity-a",
                "status": "completed",
                "output_text": "{}",
                "created": 1,
            },
            {
                "id": "req-2",
                "model": "gpt-5-mini-identity-b",
                "status": "completed",
                "output_text": "{}",
                "created": 2,
            },
        ]
    )
    monkeypatch.setattr(
        urllib.request, "urlopen", lambda *args, **kwargs: _HTTPResponse(next(responses))
    )
    adapter = OpenAIResponsesAdapter("secret", "gpt-5-mini", tmp_path / "cache", max_retries=0)
    first = GenerationRequest("p1", ({"role": "user", "content": "one"},), "test", 1)
    second = GenerationRequest("p2", ({"role": "user", "content": "two"},), "test", 2)
    assert adapter.generate(first).returned_model.endswith("identity-a")
    with pytest.raises(OpenAIResponsesError):
        adapter.generate(second)
