from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from dynamic_rubric.hashing import sha256_json
from dynamic_rubric.phase1.pi0_cache import (
    ImmutablePi0Cache,
    Pi0CacheError,
    build_pi0_cache,
)
from dynamic_rubric.providers.base import GenerationResult
from dynamic_rubric.training.online_contracts import PromptOccurrence
from dynamic_rubric.training.verl_online_runtime import (
    VerlOnlineRewardRuntime,
    _directory_hash,
    create_online_reward_runtime,
)


CHECKPOINT_HASH = "a" * 64


class FakeGenerator:
    model = "actor"
    revision = "rev"
    tokenizer_revision = "tok"

    def __init__(self) -> None:
        self.requests: list[Any] = []

    def preflight(self) -> dict[str, Any]:
        return {
            "served_model": self.model,
            "model_revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "checkpoint_hash": CHECKPOINT_HASH,
            "thinking": False,
        }

    def generate_many(self, request: Any, count: int) -> list[GenerationResult]:
        self.requests.append(request)
        return [
            GenerationResult(
                text=f"{request.prompt_id}-control-{index}",
                requested_model=self.model,
                returned_model=self.model,
                request_id=f"{request.prompt_id}-{index}",
                created_at=1,
                retry_count=0,
                usage={"logical_seed": request.seed + index},
                raw_response_hash=sha256_json([request.prompt_id, index]),
            )
            for index in range(count)
        ]


def prompts() -> list[dict[str, Any]]:
    return [
        {
            "prompt_id": f"p{index}",
            "source_row_id": f"source-{index}",
            "messages": [{"role": "user", "content": f"question {index}"}],
        }
        for index in range(2)
    ]


def make_cache(tmp_path: Path) -> tuple[Path, FakeGenerator]:
    generator = FakeGenerator()
    manifest = build_pi0_cache(
        prompts=prompts(),
        generator=generator,
        output_dir=tmp_path,
        model_revision="rev",
        checkpoint_hash=CHECKPOINT_HASH,
        expected_prompt_count=2,
        max_concurrency=1,
    )
    return manifest, generator


def occurrence(*, prompt_index: int = 0, occurrence_id: str = "occ", step: int = 1) -> PromptOccurrence:
    prompt = prompts()[prompt_index]
    return PromptOccurrence(
        run_id="run",
        optimizer_update_index=step,
        batch_uid=f"batch-{step}",
        source_row_id=prompt["source_row_id"],
        prompt_id=prompt["prompt_id"],
        prompt_occurrence_id=occurrence_id,
        prompt=tuple(prompt["messages"]),
    )


def test_builder_seals_exact_inventory_with_prompt_deterministic_seeds(tmp_path: Path) -> None:
    first_path, first_generator = make_cache(tmp_path / "first")
    second_path, second_generator = make_cache(tmp_path / "second")

    first = ImmutablePi0Cache(first_path, expected_prompt_count=2)
    second = ImmutablePi0Cache(second_path, expected_prompt_count=2)
    assert first.manifest_hash == second.manifest_hash
    assert first.manifest["response_count"] == 16
    assert [request.seed for request in first_generator.requests] == [
        request.seed for request in second_generator.requests
    ]
    assert len({request.seed for request in first_generator.requests}) == 2
    assert all(
        "optimizer_update_index" not in request.metadata
        and request.metadata["control_policy"] == "pi0_precomputed_immutable"
        for request in first_generator.requests
    )


def test_cache_rebinds_source_verified_rows_to_current_occurrence(tmp_path: Path) -> None:
    manifest, _ = make_cache(tmp_path)
    cache = ImmutablePi0Cache(manifest, expected_prompt_count=2)

    first_records, first_receipts = cache.bind(
        occurrence(occurrence_id="train-occ-1", step=1), step=1
    )
    later_records, later_receipts = cache.bind(
        occurrence(occurrence_id="train-occ-9", step=9), step=9
    )

    assert len(first_records) == len(first_receipts) == 8
    assert [row.rollout_index for row in first_records] == list(range(8))
    assert [row.text for row in first_records] == [row.text for row in later_records]
    assert {row.prompt_occurrence_id for row in first_records} == {"train-occ-1"}
    assert {row.prompt_occurrence_id for row in later_records} == {"train-occ-9"}
    assert [row.response_id for row in first_records] != [
        row.response_id for row in later_records
    ]
    assert {row["control_source"] for row in later_receipts} == {
        "precomputed_immutable"
    }
    assert {row["optimizer_update_index"] for row in later_receipts} == {9}


def test_cache_rejects_source_identity_mismatch_and_tampering(tmp_path: Path) -> None:
    manifest, _ = make_cache(tmp_path)
    cache = ImmutablePi0Cache(manifest, expected_prompt_count=2)
    wrong = occurrence()
    wrong = replace(wrong, source_row_id="wrong-source")
    with pytest.raises(Pi0CacheError, match="source identity mismatch"):
        cache.bind(wrong, step=1)

    response_path = tmp_path / cache.manifest["response_jsonl"]
    response_path.write_bytes(response_path.read_bytes() + b"\n")
    with pytest.raises(Pi0CacheError, match="hash mismatch"):
        ImmutablePi0Cache(manifest, expected_prompt_count=2)


def test_runtime_uses_cache_without_live_generator(tmp_path: Path) -> None:
    manifest, _ = make_cache(tmp_path / "cache")
    cache = ImmutablePi0Cache(manifest, expected_prompt_count=2)
    runtime = VerlOnlineRewardRuntime(
        trainer=SimpleNamespace(),
        coordinator=object(),
        control_generator=None,
        control_cache=cache,
        artifact_root=tmp_path / "steps",
        run_id="run",
        actor_model="actor",
        actor_revision="rev",
        control_hash=CHECKPOINT_HASH,
        control_concurrency=1,
        seed=11,
    )

    controls, receipts = runtime._control_responses((occurrence(),), step=1)

    assert len(controls["occ"]) == 8
    assert len(receipts) == 8
    assert runtime.control_generator is None


def test_factory_cache_path_does_not_require_online_control_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"actor")
    checkpoint_hash = _directory_hash(snapshot)

    class FakeCache:
        model = "actor"
        revision = "rev"
        tokenizer_revision = "tok"
        manifest_hash = "b" * 64

        def __init__(self, path: Path) -> None:
            assert path == tmp_path / "pi0-manifest.json"
            self.checkpoint_hash = checkpoint_hash

    monkeypatch.setattr(
        "dynamic_rubric.training.verl_online_runtime.ImmutablePi0Cache", FakeCache
    )
    for name, value in {
        "OPENAI_API_KEY": "test",
        "ONLINE_EXTRACTOR_MODEL": "extractor",
        "ONLINE_GRADER_MODEL": "grader",
        "ONLINE_ACTOR_MODEL": "actor",
        "ONLINE_ACTOR_REVISION": "rev",
        "ONLINE_CONTROL_REVISION": "rev",
        "ONLINE_CONTROL_TOKENIZER_REVISION": "tok",
        "ONLINE_CONTROL_CHECKPOINT_HASH": checkpoint_hash,
        "ONLINE_CONTROL_CACHE": str(tmp_path / "pi0-manifest.json"),
        "ONLINE_RUN_ID": "run",
        "MODEL_PATH": str(snapshot),
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("ONLINE_CONTROL_URL", raising=False)
    config = SimpleNamespace(
        reward={
            "online_step_runtime": {
                "control_policy": "pi_ref",
                "frozen_control": True,
                "artifact_root": str(tmp_path / "steps"),
            }
        }
    )

    runtime = create_online_reward_runtime(
        config=config,
        trainer=SimpleNamespace(),
    )

    assert runtime.control_generator is None
    assert isinstance(runtime.control_cache, FakeCache)


def test_builder_resumes_after_interruption_and_assembles_identical_artifacts(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class InterruptingGenerator(FakeGenerator):
        def generate_many(self, request: Any, count: int) -> list[GenerationResult]:
            if request.prompt_id == "p1":
                self.requests.append(request)
                raise RuntimeError("simulated interruption")
            return super().generate_many(request, count)

    interrupted = InterruptingGenerator()
    with pytest.raises(RuntimeError, match="simulated interruption"):
        build_pi0_cache(
            prompts=prompts(),
            generator=interrupted,
            output_dir=tmp_path / "resumed",
            model_revision="rev",
            checkpoint_hash=CHECKPOINT_HASH,
            expected_prompt_count=2,
            max_concurrency=1,
            progress_every=1,
        )
    shards = list((tmp_path / "resumed" / ".staging").rglob("*.json"))
    assert len(shards) == 1

    resumed = FakeGenerator()
    resumed_manifest = build_pi0_cache(
        prompts=prompts(),
        generator=resumed,
        output_dir=tmp_path / "resumed",
        model_revision="rev",
        checkpoint_hash=CHECKPOINT_HASH,
        expected_prompt_count=2,
        max_concurrency=1,
        progress_every=1,
    )
    assert [request.prompt_id for request in resumed.requests] == ["p1"]
    progress = capsys.readouterr().out
    assert "pi0-cache progress 2/2 (reused=1 generated=1)" in progress

    clean = FakeGenerator()
    clean_manifest = build_pi0_cache(
        prompts=prompts(),
        generator=clean,
        output_dir=tmp_path / "clean",
        model_revision="rev",
        checkpoint_hash=CHECKPOINT_HASH,
        expected_prompt_count=2,
        max_concurrency=1,
        progress_every=2,
    )
    assert resumed_manifest.name == clean_manifest.name
    resumed_cache = ImmutablePi0Cache(resumed_manifest, expected_prompt_count=2)
    clean_cache = ImmutablePi0Cache(clean_manifest, expected_prompt_count=2)
    assert resumed_cache.manifest == clean_cache.manifest


def test_builder_regenerates_only_malformed_staging_shard(tmp_path: Path) -> None:
    manifest, _ = make_cache(tmp_path)
    original = ImmutablePi0Cache(manifest, expected_prompt_count=2)
    shards = sorted((tmp_path / ".staging").rglob("*.json"))
    assert len(shards) == 2
    shards[0].write_text('{"partial":true}', encoding="utf-8")

    resumed = FakeGenerator()
    rebuilt_manifest = build_pi0_cache(
        prompts=prompts(),
        generator=resumed,
        output_dir=tmp_path,
        model_revision="rev",
        checkpoint_hash=CHECKPOINT_HASH,
        expected_prompt_count=2,
        max_concurrency=1,
        progress_every=2,
    )

    assert len(resumed.requests) == 1
    assert rebuilt_manifest.name == manifest.name
    rebuilt = ImmutablePi0Cache(rebuilt_manifest, expected_prompt_count=2)
    assert rebuilt.manifest == original.manifest
