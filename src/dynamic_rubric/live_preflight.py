from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

from .artifacts import read_json, write_json_atomic
from .hashing import sha256_file
from .pipeline import LIVE_READY_STAGES, PipelineContext, StageError, _unresolved
from .providers.local_embedding import LocalBGEEmbeddingProvider
from .providers.base import GenerationRequest
from .providers.openai_responses import OpenAIResponsesAdapter
from .providers.vllm import VLLMCriterionGrader, VLLMIdentity
from .providers.vllm_generation import VLLMPolicyGenerator
from .rubrics.static import (
    STATIC_RUBRIC_INSTRUCTIONS,
    UNIVERSAL_CRITERIA,
    CriterionValidationError,
    build_static_rubric,
    validate_criteria,
)
from .training.verl_adapter import dependency_gate


_STAGE_LOCK_SMOKES = {
    "generate-static": (),
    "train-static": (
        "proxy_grader_ten_pairs",
        "policy_one_step_probe",
        "checkpoint_resume_retention",
    ),
}

LIVE_STAGE_CAPABILITIES = {
    "generate-static": True,
    "train-static": True,
    "replay-dynamic": False,
    "generate-bon": False,
    "score-proxy": False,
    "audit-gold": False,
}


def _passed(detail: Any) -> dict[str, Any]:
    return {"status": "passed", "detail": detail}


def _required_lock_smokes() -> tuple[str, ...]:
    return tuple(
        sorted(
            {
                smoke
                for stage in LIVE_READY_STAGES
                for smoke in _STAGE_LOCK_SMOKES[stage]
            }
        )
    )


def _require_locked_smokes(lock: Mapping[str, Any]) -> dict[str, Any]:
    required = _required_lock_smokes()
    smokes = lock.get("smokes")
    if not isinstance(smokes, Mapping):
        raise StageError("live preflight failed: capability smoke records are absent")
    missing = [
        name
        for name in required
        if not isinstance(smokes.get(name), Mapping) or smokes[name].get("status") != "passed"
    ]
    if missing:
        raise StageError(f"live preflight failed: capability smokes not passed: {missing}")
    return {name: dict(smokes[name]) for name in required}


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise StageError(f"live preflight failed: {name} is absent")
    return value


def _locked_healthbench_source(root: Path, lock: Mapping[str, Any]) -> Path:
    source_file = lock.get("healthbench", {}).get("source_file")
    if not isinstance(source_file, str) or not source_file:
        raise StageError("live preflight failed: HealthBench source path is absent")
    source_relative = Path(source_file)
    if source_relative.is_absolute() or ".." in source_relative.parts:
        raise StageError("live preflight failed: HealthBench source path is invalid")
    return root / source_relative


def _existing_preflight_receipt(
    context: PipelineContext,
    *,
    lock_sha256: str,
    dataset_sha256: str,
) -> dict[str, Any] | None:
    receipt_path = context.stage_root() / "preflight.json"
    manifest_path = context.stage_root() / "manifest.json"
    if not receipt_path.exists() and not manifest_path.exists():
        return None
    if not receipt_path.is_file() or not manifest_path.is_file():
        raise StageError("live preflight resume failed: receipt or manifest is missing")
    receipt = read_json(receipt_path)
    capabilities = receipt.get("capabilities")
    providers = receipt.get("providers")
    compatible = (
        receipt.get("schema_version") == 2
        and receipt.get("ready") is True
        and receipt.get("run_id") == context.run_id
        and receipt.get("config_hash") == context.config.config_hash
        and receipt.get("lock_sha256") == lock_sha256
        and receipt.get("dataset", {}).get("sha256") == dataset_sha256
        and receipt.get("preflight_manifest_sha256") == sha256_file(manifest_path)
        and receipt.get("ready_stages") == list(LIVE_READY_STAGES)
        and isinstance(providers, Mapping)
        and bool(providers)
        and isinstance(capabilities, Mapping)
        and bool(capabilities)
        and all(
            isinstance(item, Mapping) and item.get("status") == "passed"
            for item in capabilities.values()
        )
    )
    if not compatible:
        raise StageError(
            "live preflight resume failed: receipt is incompatible with the run, config, lock, dataset, or manifest"
        )
    return dict(receipt)


def _openai_static_rubric_probe(
    cache_root: Path,
    api_key: str,
    model: str,
    reasoning_effort: str,
    schema: Mapping[str, Any],
    embedding: LocalBGEEmbeddingProvider,
) -> dict[str, Any]:
    adapter = OpenAIResponsesAdapter(api_key, model, cache_root / model)
    prompt = (
        "A patient reports a blood pressure of 180/120 with chest pain and asks whether "
        "waiting until tomorrow is safe."
    )
    candidates = (
        "Seek emergency care now because chest pain with this blood pressure may be dangerous.",
        "Rest at home and recheck tomorrow.",
        "Call emergency services now. Bring the medication list if it is readily available.",
    )
    last_error: Exception | None = None
    for attempt in range(3):
        result = adapter.generate(
            GenerationRequest(
                prompt_id="stage4-static-rubric-preflight",
                messages=(
                    {"role": "developer", "content": STATIC_RUBRIC_INSTRUCTIONS},
                    {"role": "user", "content": prompt},
                    {
                        "role": "user",
                        "content": "Candidate responses:\n"
                        + json.dumps(candidates, ensure_ascii=False),
                    },
                ),
                family="openai_static_rubric_preflight",
                seed=attempt,
                max_output_tokens=2048,
                json_schema=schema,
                schema_name="rubric_generator_v1",
                reasoning_effort=reasoning_effort,
                metadata={"replicate_id": attempt, "validation_attempt": attempt},
            )
        )
        try:
            parsed = json.loads(result.text)
            texts = validate_criteria(
                (item["text"] for item in parsed["criteria"]), expected_count=6
            )
            all_texts = (*texts, *UNIVERSAL_CRITERIA)
            vectors = embedding.embed(all_texts)
            vector_by_text = dict(zip(all_texts, vectors))

            def similarity(left: str, right: str) -> float:
                return sum(
                    a * b for a, b in zip(vector_by_text[left], vector_by_text[right])
                )

            rubric = build_static_rubric(
                "stage4-static-rubric-preflight", texts, similarity=similarity
            )
        except (CriterionValidationError, KeyError, TypeError, ValueError) as error:
            last_error = error
            continue
        return {
            "requested_model": result.requested_model,
            "returned_model": result.returned_model,
            "request_id": result.request_id,
            "created_at": result.created_at,
            "retry_count": result.retry_count,
            "reasoning_effort": reasoning_effort,
            "criteria_count": len(rubric.criteria),
            "content_hash": rubric.content_hash,
            "validation_attempt": attempt,
            "structural_validation": "passed",
            "semantic_admission": "passed",
        }
    raise StageError(f"{model} static-rubric preflight failed: {last_error}")


def run_preflight(context: PipelineContext) -> dict[str, Any]:
    if context.mode == "fake":
        capabilities = {
            name: _passed({"provider": "fake", "external_calls": 0})
            for name in _required_lock_smokes()
        }
        capabilities["openai_static_rubric"] = _passed(
            {"provider": "fake", "external_calls": 0}
        )
        result = {
            "schema_version": 2,
            "ready": all(item["status"] == "passed" for item in capabilities.values()),
            "mode": "fake",
            "ready_stages": list(LIVE_STAGE_CAPABILITIES),
            "run_id": context.run_id,
            "config_hash": context.config.config_hash,
            "lock_sha256": None,
            "dataset": {"identity": "fake", "sha256": None},
            "providers": {"identity": "fake"},
            "capabilities": capabilities,
            "network_calls": 0,
            "gpu_calls": 0,
            "external_call_tripwire": "not_triggered",
        }
        context.begin_stage(
            metadata={"preflight": "passed", "external_calls": 0, "receipt_schema": 2}
        )
        result["preflight_manifest_sha256"] = sha256_file(context.stage_root() / "manifest.json")
        write_json_atomic(context.stage_root() / "preflight.json", result)
        return result

    lock_path = context.root / "environment" / "upstream-lock.json"
    if not lock_path.is_file():
        raise StageError("live preflight failed: environment/upstream-lock.json is missing")
    lock = read_json(lock_path)
    lock_sha256 = sha256_file(lock_path)
    if _unresolved(lock):
        raise StageError(
            "live preflight failed closed: exact source/environment revisions are unresolved"
        )
    locked_smokes = _require_locked_smokes(lock)
    if any(not LIVE_STAGE_CAPABILITIES.get(stage, False) for stage in LIVE_READY_STAGES):
        raise StageError(
            f"live preflight blocked: required stage orchestrators are unavailable: "
            f"{list(LIVE_READY_STAGES)}"
        )
    source_path = _locked_healthbench_source(context.root, lock)
    expected_source_hash = lock.get("healthbench", {}).get("source_sha256")
    if not source_path.is_file() or not expected_source_hash:
        raise StageError("live preflight failed: pinned HealthBench Consensus bytes are absent")
    if sha256_file(source_path) != expected_source_hash:
        raise StageError("live preflight failed: HealthBench Consensus SHA-256 mismatch")
    existing_receipt = _existing_preflight_receipt(
        context,
        lock_sha256=lock_sha256,
        dataset_sha256=str(expected_source_hash),
    )
    if existing_receipt is not None:
        return existing_receipt
    api_key = _required_environment("OPENAI_API_KEY")
    grader_url = _required_environment("DYNAMIC_RUBRIC_VLLM_URL")
    policy_url = _required_environment("DYNAMIC_RUBRIC_POLICY_URL")
    policy_launch_spec = Path(_required_environment("DYNAMIC_RUBRIC_POLICY_LAUNCH_SPEC"))
    embedding_path = Path(_required_environment("DYNAMIC_RUBRIC_EMBEDDING_MODEL_PATH"))
    models = context.config.models
    policy = models["policy"]
    grader = models["proxy_grader"]
    embedding = models["criterion_embedding"]
    locked_policy = lock.get("models", {}).get("policy", {})
    expected_policy_spec_hash = locked_policy.get("stage4_launch_spec_sha256")
    if (
        not policy_launch_spec.is_file()
        or not expected_policy_spec_hash
        or sha256_file(policy_launch_spec) != expected_policy_spec_hash
    ):
        raise StageError("live preflight failed: Stage 4 policy launch spec is unpinned")
    verl = dependency_gate(lock, context.root)
    with tempfile.TemporaryDirectory(prefix="dynamic-rubric-preflight-") as temporary:
        temporary_path = Path(temporary)
        embedder = LocalBGEEmbeddingProvider(
            embedding_path,
            str(embedding["model"]),
            str(embedding["revision"]),
            device=os.environ.get("DYNAMIC_RUBRIC_EMBEDDING_DEVICE", "cpu"),
        )
        embedding_result = embedder.preflight()
        rubric_generator = models["rubric_generator"]
        openai_static_rubric = _openai_static_rubric_probe(
            temporary_path,
            api_key,
            str(rubric_generator["requested_model"]),
            str(rubric_generator["reasoning_effort"]),
            read_json(context.root / str(rubric_generator["schema"])),
            embedder,
        )
        grader_identity = VLLMCriterionGrader(
            grader_url,
            VLLMIdentity(
                served_model=str(grader["model"]),
                model_revision=str(grader["revision"]),
                tokenizer_revision=str(grader["tokenizer_revision"]),
                thinking=False,
            ),
        ).preflight()
        policy_identity = VLLMPolicyGenerator(
            policy_url,
            str(policy["model"]),
            str(policy["revision"]),
            str(policy["tokenizer_revision"]),
            launch_spec_path=policy_launch_spec,
        ).preflight()
    providers = {
        "openai": {"rubric_generator": openai_static_rubric},
        "proxy_grader": grader_identity,
        "policy": policy_identity,
        "criterion_embedding": embedding_result,
        "verl": {
            "revision": verl.revision,
            "custom_reward_hook": verl.custom_reward_hook,
            "raw_validation_export": verl.raw_validation_export,
            "focal_checkpoint_retention": verl.focal_checkpoint_retention,
            "probe_patch_required": verl.probe_patch_required,
            "patch_sha256": verl.patch_sha256,
        },
    }
    capabilities = {
        **{name: _passed(detail) for name, detail in locked_smokes.items()},
        "dataset_checksum": _passed({"sha256": expected_source_hash}),
        "openai_static_rubric": _passed(openai_static_rubric),
        "proxy_grader_identity_and_targets": _passed(grader_identity),
        "policy_identity": _passed(policy_identity),
        "embedding_identity": _passed(embedding_result),
        "verl_hooks": _passed(providers["verl"]),
    }
    ready = all(item.get("status") == "passed" for item in capabilities.values())
    if not ready:
        raise StageError("live preflight failed: not all capability statuses passed")
    result = {
        "schema_version": 2,
        "ready": ready,
        "mode": "live",
        "ready_stages": list(LIVE_READY_STAGES),
        "run_id": context.run_id,
        "config_hash": context.config.config_hash,
        "lock_sha256": lock_sha256,
        "dataset": {
            "name": "healthbench_consensus",
            "path": str(source_path.relative_to(context.root)),
            "sha256": expected_source_hash,
        },
        "providers": providers,
        "capabilities": capabilities,
    }
    # All network/model/dataset/dependency probes above complete before the first
    # durable run artifact is created.
    context.begin_stage(
        metadata={
            "preflight": "passed",
            "all_probes_completed": True,
            "receipt_schema": 2,
            "lock_sha256": lock_sha256,
        }
    )
    result["preflight_manifest_sha256"] = sha256_file(context.stage_root() / "manifest.json")
    write_json_atomic(context.stage_root() / "preflight.json", result)
    return result
