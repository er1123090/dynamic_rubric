from __future__ import annotations

import json
import threading
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from dynamic_rubric.artifacts import ImmutableArtifactError, read_json, write_json_atomic
from dynamic_rubric.live_preflight import (
    LIVE_STAGE_CAPABILITIES,
    _existing_preflight_receipt,
    _locked_healthbench_source,
    run_preflight,
)
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.pipeline import (
    STATIC_RUBRIC_MAX_OUTPUT_TOKENS,
    STATIC_RUBRIC_VALIDATION_ATTEMPTS,
    PipelineContext,
    StageError,
    _static_validation_feedback,
)
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.fake import FakeGenerator
from dynamic_rubric.providers.openai_responses import OpenAIResponsesAdapter, OpenAIResponsesError
from dynamic_rubric.providers.vllm_generation import (
    VLLMGenerationError,
    VLLMPolicyGenerator,
)
from dynamic_rubric.seeds import SeedFamily, VLLM_SEED_MAX, derive_seed, vllm_seed


class _HTTPResponse:
    def __init__(self, value: dict[str, object]) -> None:
        self.value = value
        self.headers = {"x-request-id": str(value.get("id", ""))}

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(self.value).encode()


def test_immutable_concurrent_publication_never_overwrites(tmp_path: Path) -> None:
    destination = tmp_path / "winner.json"
    barrier = threading.Barrier(16)

    def publish(index: int) -> tuple[int, bool]:
        barrier.wait()
        try:
            return index, write_json_atomic(destination, {"writer": index})
        except ImmutableArtifactError:
            return index, False

    with ThreadPoolExecutor(max_workers=16) as executor:
        results = list(executor.map(publish, range(16)))
    winners = [index for index, created in results if created]
    assert len(winners) == 1
    assert read_json(destination) == {"writer": winners[0]}
    assert not list(tmp_path.glob("*.tmp"))


def test_openai_cache_separates_replicates_and_checks_cached_drift(
    tmp_path: Path, monkeypatch
) -> None:
    calls: list[urllib.request.Request] = []
    returned_models = iter(("snapshot-a", "snapshot-a", "snapshot-b"))

    def urlopen(request, **kwargs):
        del kwargs
        calls.append(request)
        model = next(returned_models)
        return _HTTPResponse(
            {
                "id": f"req-{len(calls)}",
                "model": model,
                "status": "completed",
                "output_text": "{}",
                "created": 1,
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    cache = tmp_path / "cache"
    adapter = OpenAIResponsesAdapter("secret", "gpt-5-mini", cache, max_retries=0)

    def replicate(replicate_id: str) -> GenerationRequest:
        return GenerationRequest(
            prompt_id="p",
            messages=({"role": "user", "content": "same"},),
            family="replicate",
            seed=7,
            metadata={"replicate_id": replicate_id},
        )

    adapter.generate(replicate("A"))
    adapter.generate(replicate("B"))
    assert len(calls) == 2
    assert len(list(cache.glob("*.json"))) == 2

    other = OpenAIResponsesAdapter("secret", "gpt-5-mini", cache, max_retries=0)
    other.generate(
        GenerationRequest("other", ({"role": "user", "content": "other"},), "replicate", 8)
    )
    with pytest.raises(OpenAIResponsesError, match="cached returned model drifted"):
        other.generate(replicate("A"))


def test_openai_incomplete_response_is_rejected_before_identity_and_cache(
    tmp_path: Path, monkeypatch
) -> None:
    cache = tmp_path / "cache"
    adapter = OpenAIResponsesAdapter("secret", "gpt-5-mini", cache, max_retries=0)
    response = {
        "id": "req-incomplete",
        "model": "snapshot-that-must-not-be-trusted",
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "output_text": '{"otherwise": "valid JSON"}',
        "created": 1,
    }
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kwargs: _HTTPResponse(response))

    request = GenerationRequest("p", ({"role": "user", "content": "prompt"},), "incomplete-test", 1)
    with pytest.raises(OpenAIResponsesError, match="not completed"):
        adapter.generate(request)

    assert adapter._returned_model is None
    assert not list(cache.glob("*.json"))


def test_vllm_retains_logical_and_provider_seed(tmp_path: Path, monkeypatch) -> None:
    del tmp_path
    payloads: list[dict[str, Any]] = []

    def urlopen(request, **kwargs):
        del kwargs
        payloads.append(json.loads(request.data))
        return _HTTPResponse(
            {
                "id": "req-vllm",
                "model": "policy",
                "choices": [{"message": {"content": "answer"}}],
                "usage": {},
            }
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    logical = derive_seed("run", SeedFamily.AUDIT_BON, "prompt", 100, 63)
    result = VLLMPolicyGenerator("http://local", "policy", "rev", "tok").generate_many(
        GenerationRequest("p", ({"role": "user", "content": "x"},), "audit_bon", logical),
        1,
    )[0]
    assert 0 <= payloads[0]["seed"] <= VLLM_SEED_MAX
    assert payloads[0]["seed"] == vllm_seed(logical)
    assert result.usage["dynamic_rubric_logical_seed"] == logical
    assert result.usage["dynamic_rubric_vllm_seed"] == vllm_seed(logical)


def test_standard_vllm_identity_requires_matching_launch_spec(tmp_path: Path, monkeypatch) -> None:
    snapshot = tmp_path / "snapshots" / "rev"
    snapshot.mkdir(parents=True)
    spec = tmp_path / "policy-launch.json"
    spec.write_text(
        json.dumps(
            {
                "served_model": "policy",
                "model_revision": "rev",
                "tokenizer_revision": "tok",
                "thinking": False,
                "model_path": str(snapshot),
            }
        ),
        encoding="utf-8",
    )

    def urlopen(request, **kwargs):
        del kwargs
        url = request if isinstance(request, str) else request.full_url
        if url.endswith("/dynamic-rubric/identity"):
            raise OSError("identity extension absent")
        return _HTTPResponse({"id": "models", "data": [{"id": "policy"}]})

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    identity = VLLMPolicyGenerator(
        "http://local", "policy", "rev", "tok", launch_spec_path=spec
    ).preflight()
    assert identity["identity_source"] == "v1/models+immutable_launch_spec"
    assert identity["launch_spec_sha256"]

    with pytest.raises(VLLMGenerationError, match="launch spec"):
        VLLMPolicyGenerator("http://local", "policy", "rev", "tok").preflight()


def test_fake_generator_honors_criteria_array_schema() -> None:
    schema = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "minItems": 0,
                "maxItems": 3,
            }
        },
    }
    result = FakeGenerator().generate(
        GenerationRequest(
            "prompt",
            ({"role": "user", "content": "x"},),
            "dynamic_extraction",
            7,
            json_schema=schema,
        )
    )
    assert len(json.loads(result.text)["criteria"]) == 3


def test_live_preflight_exposes_only_stage4_and_stage5() -> None:
    assert LIVE_STAGE_CAPABILITIES["generate-static"] is True
    assert LIVE_STAGE_CAPABILITIES["train-static"] is True
    assert not any(
        ready
        for stage, ready in LIVE_STAGE_CAPABILITIES.items()
        if stage not in {"generate-static", "train-static"}
    )


def test_static_rubric_output_budget_leaves_room_for_reasoning_and_json() -> None:
    assert STATIC_RUBRIC_MAX_OUTPUT_TOKENS == 8192
    assert STATIC_RUBRIC_VALIDATION_ATTEMPTS == 16
    semantic = _static_validation_feedback(ValueError("semantic duplicates"))
    atomic = _static_validation_feedback(ValueError("single and atomic"))
    assert "Represent their shared behavior only once" in semantic
    assert "differ only by diagnosis" in semantic
    assert "genuinely unrelated behavior" in semantic
    assert "exactly one short, positive clause" in atomic
    assert "including inside examples or parentheses" in atomic


def test_live_preflight_uses_locked_healthbench_source_path(tmp_path: Path) -> None:
    locked = _locked_healthbench_source(
        tmp_path,
        {"healthbench": {"source_file": "data/source/consensus.jsonl"}},
    )
    assert locked == tmp_path / "data" / "source" / "consensus.jsonl"
    with pytest.raises(StageError, match="source path is invalid"):
        _locked_healthbench_source(
            tmp_path,
            {"healthbench": {"source_file": "../private.jsonl"}},
        )


def test_live_preflight_reuses_only_a_fully_bound_receipt(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
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
    context = PipelineContext.create(tmp_path, config_path, "preflight", "resume-run")
    context.stage_root().mkdir(parents=True)
    manifest_path = context.stage_root() / "manifest.json"
    manifest_path.write_text('{"manifest":"bound"}\n', encoding="utf-8")
    receipt = {
        "schema_version": 2,
        "ready": True,
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "lock_sha256": "lock-pin",
        "dataset": {"sha256": "dataset-pin"},
        "preflight_manifest_sha256": sha256_file(manifest_path),
        "ready_stages": ["generate-static", "train-static"],
        "providers": {"openai": {"model": "snapshot"}},
        "capabilities": {"probe": {"status": "passed"}},
    }
    receipt_path = context.stage_root() / "preflight.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    assert (
        _existing_preflight_receipt(
            context,
            lock_sha256="lock-pin",
            dataset_sha256="dataset-pin",
        )
        == receipt
    )
    with pytest.raises(StageError, match="receipt is incompatible"):
        _existing_preflight_receipt(
            context,
            lock_sha256="different-lock",
            dataset_sha256="dataset-pin",
        )


def test_fake_preflight_receipt_binds_run_config_and_capabilities(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "paths": {"artifacts": "artifacts"},
                "training": {"reward_source": "static_r0_only"},
                "execution": {"mode": "fake"},
            }
        ),
        encoding="utf-8",
    )
    context = PipelineContext.create(tmp_path, config_path, "preflight", "bound-run")
    result = run_preflight(context)
    assert result["ready"] is True
    assert result["run_id"] == "bound-run"
    assert result["config_hash"] == context.config.config_hash
    assert "generate-static" in result["ready_stages"]
    assert "train-static" in result["ready_stages"]
    assert result["capabilities"]
    assert {item["status"] for item in result["capabilities"].values()} == {"passed"}
    assert read_json(context.stage_root() / "preflight.json") == result
