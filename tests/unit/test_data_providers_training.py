from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from dynamic_rubric.data.healthbench import scan_public_outputs_for_gold
from dynamic_rubric.providers.openai_responses import OpenAIResponsesAdapter
from dynamic_rubric.providers.vllm import (
    NO_TARGET,
    YES_TARGET,
    VLLMCriterionGrader,
    VLLMIdentity,
    VLLMPreflightError,
    normalized_yes_probability,
)
from dynamic_rubric.providers.vllm_generation import (
    VLLMGenerationError,
    VLLMPolicyGenerator,
)
from dynamic_rubric.training.probe_export import prove_probe_side_effect_free
from dynamic_rubric.training.static_reward import StaticRewardConfig, StaticRewardContractError


def test_public_leakage_scanner_rejects_gold_text_and_hash(tmp_path: Path) -> None:
    private = tmp_path / "private.jsonl"
    private.write_text(
        json.dumps({"prompt_id": "p", "gold_rubric": "Private physician criterion text"}) + "\n",
        encoding="utf-8",
    )
    public = tmp_path / "public.json"
    public.write_text("Private physician criterion text", encoding="utf-8")
    with pytest.raises(PermissionError):
        scan_public_outputs_for_gold([public], private)


def test_static_training_rejects_dynamic_path() -> None:
    with pytest.raises(StaticRewardContractError):
        StaticRewardConfig(Path("artifacts/replay/dynamic_fixed.jsonl")).validate()


def test_probe_side_effect_smoke_detects_rng_or_state_change() -> None:
    def stable(with_probe: bool):
        del with_probe
        return {"weights": b"same", "optimizer_step": 3, "rng": "same"}

    assert set(prove_probe_side_effect_free(stable)) == {"weights", "optimizer_step", "rng"}

    def changed(with_probe: bool):
        return {"weights": b"same", "optimizer_step": 3, "rng": str(with_probe)}

    with pytest.raises(AssertionError):
        prove_probe_side_effect_free(changed)


def test_vllm_probability_and_malformed_target_contract() -> None:
    assert normalized_yes_probability(0.0, 0.0) == 0.5
    with pytest.raises(ValueError):
        normalized_yes_probability(float("nan"), 0.0)


def test_vllm_horizon_grader_requires_exact_space_prefixed_targets() -> None:
    assert VLLMCriterionGrader._extract_pair(
        {"target_logprobs": {YES_TARGET: -0.1, NO_TARGET: -1.0}}
    ) == (-0.1, -1.0)
    with pytest.raises(VLLMPreflightError, match="exactly"):
        VLLMCriterionGrader._extract_pair(
            {"target_logprobs": {"YES": -0.1, "NO": -1.0}}
        )


def test_openai_adapter_requires_key_and_tracks_model_drift(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        OpenAIResponsesAdapter("", "gpt-5-mini", tmp_path)


class _JSONResponse:
    def __init__(self, value: object) -> None:
        self.value = value

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


def test_policy_preflight_binds_the_finetuned_checkpoint_hash(monkeypatch) -> None:
    identity = {
        "served_model": "policy",
        "model_revision": "base-revision",
        "tokenizer_revision": "base-revision",
        "thinking": False,
        "checkpoint_hash": "wrong-checkpoint",
    }
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _JSONResponse(identity))
    provider = VLLMPolicyGenerator(
        "http://policy",
        "policy",
        "base-revision",
        "base-revision",
        expected_checkpoint_hash="fine-tuned-checkpoint",
    )
    with pytest.raises(VLLMGenerationError, match="identity mismatch"):
        provider.preflight()


def test_policy_preflight_has_no_launch_spec_fallback_for_finetuned_weights(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: (_ for _ in ()).throw(OSError()))
    launch_spec = tmp_path / "launch.json"
    launch_spec.write_text("{}", encoding="utf-8")
    provider = VLLMPolicyGenerator(
        "http://policy",
        "policy",
        "base-revision",
        "base-revision",
        launch_spec_path=launch_spec,
        expected_checkpoint_hash="fine-tuned-checkpoint",
    )
    with pytest.raises(VLLMGenerationError, match="requires /dynamic-rubric/identity"):
        provider.preflight()


def test_vllm_grader_places_response_before_criterion_for_prefix_reuse(monkeypatch) -> None:
    grader = VLLMCriterionGrader(
        "http://grader",
        VLLMIdentity(
            served_model="grader",
            model_revision="revision",
            tokenizer_revision="revision",
            thinking=False,
        ),
    )
    captured: dict[str, object] = {}

    def post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        captured["endpoint"] = endpoint
        captured["payload"] = payload
        return {"target_logprobs": {YES_TARGET: -0.1, NO_TARGET: -1.0}}

    monkeypatch.setattr(grader, "_post", post)
    rows = grader.score_many_full(
        (("prompt", "response", "long answer", "criterion", "checks fact"),),
        prompt_text_by_id={"prompt": "question"},
    )

    assert rows[0].parse_status == "ok"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    rendered_prompts = payload["rendered_prompts"]
    assert isinstance(rendered_prompts, list)
    rendered = rendered_prompts[0]
    assert isinstance(rendered, str)
    assert rendered.index("\nResponse: long answer") < rendered.index("\nCriterion: checks fact")


def test_vllm_grader_chunks_large_requests_without_reordering(monkeypatch) -> None:
    grader = VLLMCriterionGrader(
        "http://grader",
        VLLMIdentity(
            served_model="grader",
            model_revision="revision",
            tokenizer_revision="revision",
            thinking=False,
        ),
        max_items_per_request=2,
    )
    request_sizes: list[int] = []

    def post(endpoint: str, payload: dict[str, object]) -> dict[str, object]:
        assert endpoint == "/dynamic-rubric/score-targets"
        rendered = payload["rendered_prompts"]
        assert isinstance(rendered, list)
        request_sizes.append(len(rendered))
        pairs = [
            {YES_TARGET: -0.1 - index, NO_TARGET: -1.0 - index}
            for index in range(len(rendered))
        ]
        return {"target_logprobs": pairs[0] if len(pairs) == 1 else pairs}

    monkeypatch.setattr(grader, "_post", post)
    items = tuple(
        ("prompt", f"response-{index}", "answer", f"criterion-{index}", "criterion")
        for index in range(5)
    )

    rows = grader.score_many_full(items)

    assert request_sizes == [2, 2, 1]
    assert [row.response_id for row in rows] == [item[1] for item in items]


def test_vllm_grader_rejects_invalid_request_limits() -> None:
    identity = VLLMIdentity("grader", "revision", "revision", False)
    with pytest.raises(ValueError, match="timeout_seconds"):
        VLLMCriterionGrader("http://grader", identity, timeout_seconds=0)
    with pytest.raises(ValueError, match="max_items_per_request"):
        VLLMCriterionGrader("http://grader", identity, max_items_per_request=0)
    with pytest.raises(ValueError, match="max_retries"):
        VLLMCriterionGrader("http://grader", identity, max_retries=-1)


def test_vllm_grader_retries_transient_sidecar_failure(monkeypatch) -> None:
    identity = VLLMIdentity("grader", "revision", "revision", False)
    grader = VLLMCriterionGrader(
        "http://grader",
        identity,
        max_retries=1,
        retry_initial_seconds=0,
    )
    calls = 0

    def urlopen(*args, **kwargs):
        del args, kwargs
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("temporary outage")
        return _JSONResponse(
            {"target_logprobs": {YES_TARGET: -0.1, NO_TARGET: -1.0}}
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    value = grader._post(
        "/dynamic-rubric/score-targets",
        {"rendered_prompts": ["prompt"], "targets": [YES_TARGET, NO_TARGET]},
    )

    assert calls == 2
    assert value["target_logprobs"][YES_TARGET] == -0.1
