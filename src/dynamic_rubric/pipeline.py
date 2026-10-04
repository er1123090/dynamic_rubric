from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .artifacts import (
    read_json,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
    write_manifest,
)
from .config import RunConfig, load_config
from .data.healthbench import prepare_healthbench_source, scan_public_outputs_for_gold
from .data.sealing import freeze_updater as write_updater_lock
from .data.splits import SplitSpec, assign_splits, validate_disjoint, write_splits
from .hashing import sha256_file, sha256_json
from .path_boundary import PathBoundaryError, resolve_project_path
from .providers.base import GenerationRequest
from .providers.fake import FakeEmbeddingProvider, FakeGenerator
from .rubrics.static import (
    STATIC_RUBRIC_INSTRUCTIONS,
    UNIVERSAL_CRITERIA,
    CriterionValidationError,
    StaticRubric,
    build_static_rubric,
    validate_criteria,
)
from .schemas import RunManifest
from .seeds import RESPONSE_FAMILIES, SEED_RANGES, SeedFamily, derive_seed, response_id
from .training.live_static import run_live_static_training


class StageError(RuntimeError):
    pass


LIVE_READY_STAGES = ("generate-static", "train-static")
STATIC_RUBRIC_MAX_OUTPUT_TOKENS = 8192
STATIC_RUBRIC_VALIDATION_ATTEMPTS = 16


def _canonical_hash_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _static_validation_feedback(error: Exception) -> str:
    detail = str(error)
    if "semantic" in detail.casefold():
        repair = (
            "The quoted pair is too similar. Represent their shared behavior only once; "
            "do not create mirrored criteria that differ only by diagnosis, severity, "
            "patient subgroup, or timepoint. Merge that behavior into one broader atomic "
            "criterion, then use the freed slot for a genuinely unrelated behavior supported "
            "by the candidate responses. Do not restate another generated criterion or "
            "either universal criterion. Keep every criterion free of standalone 'and' or "
            "'or', including examples and parentheses."
        )
    elif "atomic" in detail.casefold():
        repair = (
            "The quoted criterion caused the failure. Rewrite it as exactly one short, "
            "positive clause. Every criterion string must contain no standalone word "
            "'and' and no standalone word 'or', including inside examples or parentheses; "
            "it must also contain no semicolon or newline. Preserve six distinct concepts."
        )
    else:
        repair = "Follow every structural rule literally and return a complete JSON object."
    return f"Previous criteria failed validation: {detail}. Regenerate all six. {repair}"


def _relative_or_absolute(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _unresolved(value: Any) -> bool:
    if isinstance(value, Mapping):
        return any(_unresolved(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_unresolved(item) for item in value)
    return isinstance(value, str) and ("UNRESOLVED" in value.upper() or "UNPINNED" in value.upper())


@dataclass
class PipelineContext:
    root: Path
    config_path: Path
    stage: str
    run_id: str
    config: RunConfig

    @classmethod
    def create(cls, root: Path, config_path: Path, stage: str, run_id: str) -> "PipelineContext":
        root = root.resolve()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", run_id) or ".." in run_id:
            raise StageError(
                "run_id must be a single safe slug using letters, digits, dot, underscore, or hyphen"
            )
        path = config_path if config_path.is_absolute() else root / config_path
        try:
            path = resolve_project_path(root, path, allow_private=False)
        except PathBoundaryError as error:
            raise StageError(str(error)) from error
        return cls(root, path, stage, run_id, load_config(path, stage=stage))

    @property
    def raw(self) -> Mapping[str, Any]:
        return self.config.raw

    @property
    def mode(self) -> str:
        execution = self.raw.get("execution", {})
        return str(execution.get("mode", "live")) if isinstance(execution, Mapping) else "live"

    @property
    def artifacts_root(self) -> Path:
        return resolve_project_path(
            self.root, _relative_or_absolute(self.root, self.config.paths.artifacts)
        )

    @property
    def public_root(self) -> Path:
        return resolve_project_path(
            self.root, _relative_or_absolute(self.root, self.config.paths.public_data)
        )

    @property
    def results_root(self) -> Path:
        return resolve_project_path(
            self.root, _relative_or_absolute(self.root, self.config.paths.results)
        )

    @property
    def run_root(self) -> Path:
        return self.artifacts_root / "runs" / self.run_id

    def stage_root(self, stage: str | None = None) -> Path:
        return self.run_root / (stage or self.stage)

    def require_live_gate(self) -> Mapping[str, Any]:
        if self.mode == "fake":
            return {"ready": True, "mode": "fake", "network_calls": 0, "gpu_calls": 0}
        lock_path = self.root / "environment" / "upstream-lock.json"
        if not lock_path.is_file():
            raise StageError("live preflight failed: environment/upstream-lock.json is missing")
        lock = read_json(lock_path)
        if _unresolved(lock):
            raise StageError(
                "live preflight failed closed: exact revisions are unresolved"
            )
        return lock

    def begin_stage(
        self,
        *,
        inputs: Sequence[Path] = (),
        metadata: Mapping[str, Any] | None = None,
    ) -> RunManifest:
        if self.mode != "fake" and self.stage == "prepare-data":
            lock_path = self.root / "environment" / "upstream-lock.json"
            if not lock_path.is_file():
                raise StageError("data preparation failed: environment lock is missing")
            lock = read_json(lock_path)
            if _unresolved(lock.get("healthbench", {})):
                raise StageError("data preparation failed: HealthBench source lock is unresolved")
        else:
            lock = self.require_live_gate()
        stage_inputs = tuple(inputs)
        if self.mode != "fake" and self.stage == "train-online":
            receipt_path = self.stage_root() / "online_preflight.json"
            if not receipt_path.is_file():
                raise StageError("online preflight receipt is missing")
            receipt = read_json(receipt_path)
            if not (
                receipt.get("schema_version") == 1
                and receipt.get("ready") is True
                and receipt.get("run_id") == self.run_id
                and receipt.get("config_hash") == self.config.config_hash
                and receipt.get("stage") == "train-online"
            ):
                raise StageError("online preflight receipt is incompatible with this run")
            stage_inputs = (*stage_inputs, receipt_path)
        elif self.mode != "fake" and self.stage not in {"preflight", "prepare-data"}:
            receipt_path = self.run_root / "preflight" / "preflight.json"
            preflight_manifest_path = self.run_root / "preflight" / "manifest.json"
            if not receipt_path.is_file() or not preflight_manifest_path.is_file():
                raise StageError(
                    "live preflight receipt is incompatible: receipt or manifest is missing"
                )
            receipt = read_json(receipt_path)
            capabilities = receipt.get("capabilities", {})
            ready_stages = receipt.get("ready_stages")
            expected_lock_hash = sha256_file(self.root / "environment" / "upstream-lock.json")
            expected_dataset_hash = lock.get("healthbench", {}).get("source_sha256")
            expected_preflight_manifest_hash = sha256_file(preflight_manifest_path)
            identity_ok = (
                receipt.get("schema_version") == 2
                and receipt.get("ready") is True
                and receipt.get("run_id") == self.run_id
                and receipt.get("config_hash") == self.config.config_hash
                and receipt.get("lock_sha256") == expected_lock_hash
                and receipt.get("dataset", {}).get("sha256") == expected_dataset_hash
                and receipt.get("preflight_manifest_sha256") == expected_preflight_manifest_hash
                and isinstance(ready_stages, list)
                and self.stage in ready_stages
                and isinstance(receipt.get("providers"), Mapping)
                and bool(receipt.get("providers"))
                and isinstance(capabilities, Mapping)
                and bool(capabilities)
                and all(
                    isinstance(item, Mapping) and item.get("status") == "passed"
                    for item in capabilities.values()
                )
            )
            if not identity_ok:
                raise StageError(
                    "live preflight receipt is incompatible with the run, config, lock, dataset, or providers"
                )
            stage_inputs = (*stage_inputs, receipt_path)
        missing = [path for path in stage_inputs if not path.is_file()]
        if missing:
            raise StageError(f"required stage inputs are missing: {missing}")
        allow_private = self.stage in {"audit-gold", "audit_gold"}
        try:
            resolved_inputs = tuple(
                resolve_project_path(self.root, path, allow_private=allow_private)
                for path in stage_inputs
            )
        except PathBoundaryError as error:
            raise StageError(str(error)) from error
        input_hashes = {
            str(path.relative_to(self.root)): sha256_file(path) for path in resolved_inputs
        }
        schema_hashes: dict[str, str] = {}
        prompt_hashes: dict[str, str] = {}
        for role, model in sorted(self.config.models.items()):
            if not isinstance(model, Mapping):
                continue
            schema = model.get("schema")
            if schema:
                schema_path = _relative_or_absolute(self.root, str(schema))
                if schema_path.is_file():
                    schema_hashes[str(role)] = sha256_file(schema_path)
            prompt_version = model.get("prompt_version")
            if prompt_version:
                prompt_hashes[str(role)] = _canonical_hash_text(str(prompt_version))
        manifest = RunManifest(
            run_id=self.run_id,
            stage=self.stage,
            config_hash=self.config.config_hash,
            input_hashes=input_hashes,
            model_identities=dict(self.config.models),
            prompt_hashes=prompt_hashes,
            schema_hashes=schema_hashes,
            seed_namespaces={
                family.value: {"start": SEED_RANGES[family][0], "end": SEED_RANGES[family][1]}
                for family in RESPONSE_FAMILIES
            },
            metadata={"execution_mode": self.mode, **dict(metadata or {})},
        )
        write_manifest(self.stage_root() / "manifest.json", manifest)
        return manifest


def _configured_splits(raw: Mapping[str, Any]) -> tuple[SplitSpec, ...]:
    values = raw.get("splits", {})
    if not isinstance(values, Mapping) or not values:
        raise StageError("config.splits must be a non-empty mapping")
    return tuple(SplitSpec(str(name), int(count)) for name, count in values.items())


def _all_public_prompts(context: PipelineContext) -> list[dict[str, Any]]:
    prompts: list[dict[str, Any]] = []
    for spec in _configured_splits(context.raw):
        path = context.public_root / f"{spec.name}.jsonl"
        for row in read_jsonl(path):
            row = dict(row)
            row["split"] = spec.name
            prompts.append(row)
    return prompts


def run_prepare_data(context: PipelineContext, source: Path) -> dict[str, Any]:
    if context.mode != "fake":
        lock_path = context.root / "environment" / "upstream-lock.json"
        if not lock_path.is_file():
            raise StageError("data preparation failed: environment lock is missing")
        lock = read_json(lock_path)
        expected = lock.get("healthbench", {}).get("source_sha256")
        if not expected or sha256_file(source) != expected:
            raise StageError(
                "HealthBench source hash does not match the immutable environment lock"
            )
    context.begin_stage(inputs=(source,), metadata={"contains_private_ingestion": True})
    normalized = context.stage_root() / "healthbench_public_normalized.jsonl"
    private_output = context.root / "data" / "private_gt" / "healthbench_gold_rubrics.jsonl"
    source_manifest = prepare_healthbench_source(source, normalized, private_output)
    rows = read_jsonl(normalized)
    specs = _configured_splits(context.raw)
    assigned = assign_splits(rows, specs, context.config.split_seed)
    validate_disjoint(assigned)
    write_splits(assigned, context.public_root, context.config.split_seed)
    public_paths = [context.public_root / f"{spec.name}.jsonl" for spec in specs]
    public_paths.append(context.public_root / "split_manifest.json")
    scan_public_outputs_for_gold(public_paths, private_output)
    result = {
        **source_manifest,
        "split_counts": {name: len(values) for name, values in assigned.items()},
        "private_output": str(private_output.relative_to(context.root)),
        "public_leakage_scan": "passed",
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def run_preflight(context: PipelineContext) -> dict[str, Any]:
    from .live_preflight import run_preflight as run_live_preflight

    return run_live_preflight(context)


def _fake_generator_schema(context: PipelineContext) -> Mapping[str, Any]:
    path = context.root / "configs" / "schemas" / "rubric_generator_v1.json"
    return read_json(path)


def _rubric_dict(rubric: StaticRubric) -> dict[str, Any]:
    return {
        "prompt_id": rubric.prompt_id,
        "rubric_id": f"{rubric.prompt_id}:R_0",
        "policy_step": 0,
        "trajectory": "static",
        "criteria": [dataclasses.asdict(criterion) for criterion in rubric.criteria],
        "content_hash": rubric.content_hash,
        "provenance": {"gold_access": False, "generated": 6, "universal": 2},
    }


def _live_static_providers(context: PipelineContext) -> tuple[Any, Any, Any]:
    from .providers.local_embedding import LocalBGEEmbeddingProvider
    from .providers.openai_responses import OpenAIResponsesAdapter
    from .providers.vllm_generation import VLLMPolicyGenerator

    api_key = os.environ.get("OPENAI_API_KEY")
    policy_url = os.environ.get("DYNAMIC_RUBRIC_POLICY_URL")
    policy_launch_spec = os.environ.get("DYNAMIC_RUBRIC_POLICY_LAUNCH_SPEC")
    embedding_path = os.environ.get("DYNAMIC_RUBRIC_EMBEDDING_MODEL_PATH")
    if not api_key or not policy_url or not policy_launch_spec or not embedding_path:
        raise StageError(
            "live static generation requires OPENAI_API_KEY, DYNAMIC_RUBRIC_POLICY_URL, "
            "DYNAMIC_RUBRIC_POLICY_LAUNCH_SPEC, and DYNAMIC_RUBRIC_EMBEDDING_MODEL_PATH"
        )
    policy = context.config.models["policy"]
    rubric_generator = context.config.models["rubric_generator"]
    embedding = context.config.models["criterion_embedding"]
    generator = OpenAIResponsesAdapter(
        api_key,
        str(rubric_generator["requested_model"]),
        context.run_root / "provider_cache" / "openai" / "rubric_generator",
    )
    policy_generator = VLLMPolicyGenerator(
        policy_url,
        str(policy["model"]),
        str(policy["revision"]),
        str(policy["tokenizer_revision"]),
        launch_spec_path=Path(policy_launch_spec),
    )
    embedder = LocalBGEEmbeddingProvider(
        Path(embedding_path),
        str(embedding["model"]),
        str(embedding["revision"]),
        device=os.environ.get("DYNAMIC_RUBRIC_EMBEDDING_DEVICE", "cpu"),
    )
    return generator, policy_generator, embedder


def _static_candidate_rows(
    context: PipelineContext,
    prompt: Mapping[str, Any],
    policy_generator: Any,
) -> list[dict[str, Any]]:
    prompt_id = str(prompt["prompt_id"])
    shard = context.stage_root() / "candidate_shards" / f"{prompt_id}.json"
    if shard.is_file():
        cached = read_json(shard)
        rows = cached.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 12
            or any(row.get("prompt_id") != prompt_id for row in rows)
        ):
            raise StageError(f"invalid immutable static candidate shard: {shard}")
        return [dict(row) for row in rows]
    rows: list[dict[str, Any]] = []
    for sample_index in range(12):
        seed = derive_seed(
            context.run_id, SeedFamily.STATIC_CANDIDATE, prompt_id, 0, sample_index
        )
        request = GenerationRequest(
            prompt_id=prompt_id,
            messages=tuple(prompt["messages"]),
            family=SeedFamily.STATIC_CANDIDATE.value,
            seed=seed,
            temperature=0.7 if sample_index < 6 else 1.1,
            top_p=0.95,
            max_output_tokens=int(
                context.raw.get("training", {}).get("max_response_length", 1536)
            ),
            metadata={
                "run_id": context.run_id,
                "policy_step": 0,
                "sample_index": sample_index,
            },
        )
        response = policy_generator.generate(request)
        rows.append(
            {
                "prompt_id": prompt_id,
                "response_id": response_id(
                    context.run_id, SeedFamily.STATIC_CANDIDATE, prompt_id, 0, sample_index
                ),
                "sample_index": sample_index,
                "seed": seed,
                "temperature": request.temperature,
                "top_p": request.top_p,
                "response_text": response.text,
                "provider_call": {
                    "requested_model": response.requested_model,
                    "returned_model": response.returned_model,
                    "request_id": response.request_id,
                    "created_at": response.created_at,
                    "retry_count": response.retry_count,
                    "usage": dict(response.usage),
                    "raw_response_hash": response.raw_response_hash,
                },
            }
        )
    write_json_atomic(shard, {"prompt_id": prompt_id, "rows": rows}, immutable=True)
    return rows


def _pi0_reference_rows(
    context: PipelineContext,
    prompt: Mapping[str, Any],
    policy_generator: Any,
) -> list[dict[str, Any]]:
    prompt_id = str(prompt["prompt_id"])
    split = str(prompt["split"])
    shard = context.stage_root() / "reference_shards" / f"{prompt_id}.json"
    if shard.is_file():
        cached = read_json(shard)
        rows = cached.get("rows")
        if (
            not isinstance(rows, list)
            or len(rows) != 12
            or any(row.get("prompt_id") != prompt_id for row in rows)
        ):
            raise StageError(f"invalid immutable pi0 reference shard: {shard}")
        return [dict(row) for row in rows]
    rows: list[dict[str, Any]] = []
    for family, count in (
        (SeedFamily.REFERENCE_DISCOVERY, 8),
        (SeedFamily.REFERENCE_VALIDATION, 4),
    ):
        for sample_index in range(count):
            seed = derive_seed(context.run_id, family, prompt_id, 0, sample_index)
            request = GenerationRequest(
                prompt_id=prompt_id,
                messages=tuple(prompt["messages"]),
                family=family.value,
                seed=seed,
                temperature=1.0,
                top_p=0.95,
                max_output_tokens=int(
                    context.raw.get("training", {}).get("max_response_length", 1536)
                ),
                metadata={
                    "run_id": context.run_id,
                    "policy_step": 0,
                    "sample_index": sample_index,
                    "split": split,
                },
            )
            response = policy_generator.generate(request)
            rows.append(
                {
                    "run_id": context.run_id,
                    "prompt_id": prompt_id,
                    "split": split,
                    "policy_id": "pi_0",
                    "policy_step": 0,
                    "family": family.value,
                    "sample_index": sample_index,
                    "seed": seed,
                    "response_id": response_id(
                        context.run_id, family, prompt_id, 0, sample_index
                    ),
                    "response_text": response.text,
                    "provider_call": {
                        "requested_model": response.requested_model,
                        "returned_model": response.returned_model,
                        "request_id": response.request_id,
                        "created_at": response.created_at,
                        "retry_count": response.retry_count,
                        "usage": dict(response.usage),
                        "raw_response_hash": response.raw_response_hash,
                    },
                }
            )
    write_json_atomic(shard, {"prompt_id": prompt_id, "rows": rows}, immutable=True)
    return rows


def _static_rubric_row(
    context: PipelineContext,
    prompt: Mapping[str, Any],
    prompt_candidates: Sequence[str],
    generator: Any,
    embedding: Any,
    embedding_lock: threading.Lock,
) -> dict[str, Any]:
    prompt_id = str(prompt["prompt_id"])
    shard = context.stage_root() / "rubric_shards" / f"{prompt_id}.json"
    if shard.is_file():
        cached = read_json(shard)
        if cached.get("prompt_id") != prompt_id or len(cached.get("criteria", [])) != 8:
            raise StageError(f"invalid immutable static rubric shard: {shard}")
        return dict(cached)
    last_error: Exception | None = None
    for attempt in range(STATIC_RUBRIC_VALIDATION_ATTEMPTS):
        feedback = (
            ()
            if last_error is None
            else (
                {
                    "role": "developer",
                    "content": _static_validation_feedback(last_error),
                },
            )
        )
        rubric_request = GenerationRequest(
            prompt_id=prompt_id,
            messages=(
                {"role": "developer", "content": STATIC_RUBRIC_INSTRUCTIONS},
                *tuple(prompt["messages"]),
                {
                    "role": "user",
                    "content": "Candidate responses:\n"
                    + json.dumps(list(prompt_candidates), ensure_ascii=False),
                },
                *feedback,
            ),
            family="rubric_generator",
            seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, 0, attempt),
            max_output_tokens=STATIC_RUBRIC_MAX_OUTPUT_TOKENS,
            json_schema=_fake_generator_schema(context),
            schema_name="rubric_generator_v1",
            reasoning_effort=str(context.config.models["rubric_generator"]["reasoning_effort"]),
            metadata={"run_id": context.run_id, "attempt": attempt, "replicate_id": attempt},
        )
        try:
            generator_result = generator.generate(rubric_request)
            generated = json.loads(generator_result.text)
            texts = [item["text"] for item in generated["criteria"]]
            validate_criteria(texts, expected_count=6)
            all_texts = [*texts, *UNIVERSAL_CRITERIA]
            with embedding_lock:
                vectors = embedding.embed(all_texts)
            vector_by_text = dict(zip(all_texts, vectors))

            def similarity(left: str, right: str) -> float:
                return sum(
                    a * b for a, b in zip(vector_by_text[left], vector_by_text[right])
                )

            rubric = build_static_rubric(prompt_id, texts, similarity=similarity)
        except (CriterionValidationError, KeyError, TypeError, ValueError) as error:
            last_error = error
            continue
        rubric_row = _rubric_dict(rubric)
        rubric_row["provenance"]["semantic_embedding"] = dict(embedding.identity)
        rubric_row["generator_call"] = {
            "requested_model": generator_result.requested_model,
            "returned_model": generator_result.returned_model,
            "request_id": generator_result.request_id,
            "created_at": generator_result.created_at,
            "retry_count": generator_result.retry_count,
            "usage": dict(generator_result.usage),
            "raw_response_hash": generator_result.raw_response_hash,
            "reasoning_effort": rubric_request.reasoning_effort,
            "generation_parameters": {
                "temperature": rubric_request.temperature,
                "top_p": rubric_request.top_p,
                "max_output_tokens": rubric_request.max_output_tokens,
            },
            "prompt_hash": sha256_json([dict(message) for message in rubric_request.messages]),
            "schema_hash": sha256_json(rubric_request.json_schema),
            "validation_attempt": attempt,
        }
        write_json_atomic(shard, rubric_row, immutable=True)
        return rubric_row
    raise StageError(f"gpt-5-mini failed static rubric validation for {prompt_id}: {last_error}")


def run_generate_static(context: PipelineContext) -> dict[str, Any]:
    split_paths = [
        context.public_root / f"{spec.name}.jsonl" for spec in _configured_splits(context.raw)
    ]
    context.begin_stage(inputs=split_paths, metadata={"gold_access": False})
    if context.mode == "fake":
        generator = FakeGenerator("fake/gpt-5-mini-v1")
        policy_generator = FakeGenerator("fake/qwen3-4b-v1")
        embedding = FakeEmbeddingProvider()
    else:
        generator, policy_generator, embedding = _live_static_providers(context)
        prompts = _all_public_prompts(context)
        candidate_workers = int(os.environ.get("DYNAMIC_RUBRIC_POLICY_CONCURRENCY", "32"))
        with ThreadPoolExecutor(max_workers=candidate_workers) as pool:
            grouped_candidates = list(
                pool.map(
                    lambda prompt: _static_candidate_rows(
                        context, prompt, policy_generator
                    ),
                    prompts,
                )
            )
        candidates = [row for group in grouped_candidates for row in group]
        reference_prompts = [
            prompt
            for prompt in prompts
            if prompt.get("split") in {"pilot_probe", "pilot_audit"}
        ]
        with ThreadPoolExecutor(max_workers=candidate_workers) as pool:
            grouped_references = list(
                pool.map(
                    lambda prompt: _pi0_reference_rows(
                        context, prompt, policy_generator
                    ),
                    reference_prompts,
                )
            )
        references = [row for group in grouped_references for row in group]
        candidates_by_prompt = {
            str(prompt["prompt_id"]): [
                str(row["response_text"])
                for row in grouped_candidates[index]
            ]
            for index, prompt in enumerate(prompts)
        }
        embedding_lock = threading.Lock()
        rubric_workers = int(os.environ.get("DYNAMIC_RUBRIC_OPENAI_CONCURRENCY", "16"))
        with ThreadPoolExecutor(max_workers=rubric_workers) as pool:
            rubrics = list(
                pool.map(
                    lambda prompt: _static_rubric_row(
                        context,
                        prompt,
                        candidates_by_prompt[str(prompt["prompt_id"])],
                        generator,
                        embedding,
                        embedding_lock,
                    ),
                    prompts,
                )
            )
        write_jsonl_atomic(context.stage_root() / "static_candidates.jsonl", candidates)
        write_jsonl_atomic(context.stage_root() / "static_rubrics.jsonl", rubrics)
        write_jsonl_atomic(
            context.stage_root() / "pi0_reference_responses.jsonl", references
        )
        result = {
            "prompts": len(rubrics),
            "candidates": len(candidates),
            "pi0_reference_responses": len(references),
            "criteria_per_prompt": 8,
            "equal_weight": 1 / 8,
            "gold_access": False,
            "resumable_prompt_shards": True,
            "provider_calls": {
                "policy": policy_generator.calls,
                "rubric_generator": generator.calls,
            },
        }
        write_json_atomic(context.stage_root() / "result.json", result)
        return result
    candidates: list[dict[str, Any]] = []
    rubrics: list[dict[str, Any]] = []
    for prompt in _all_public_prompts(context):
        prompt_id = str(prompt["prompt_id"])
        prompt_candidates: list[str] = []
        for sample_index in range(12):
            seed = derive_seed(
                context.run_id, SeedFamily.STATIC_CANDIDATE, prompt_id, 0, sample_index
            )
            request = GenerationRequest(
                prompt_id=prompt_id,
                messages=tuple(prompt["messages"]),
                family=SeedFamily.STATIC_CANDIDATE.value,
                seed=seed,
                temperature=0.7 if sample_index < 6 else 1.1,
                top_p=0.95,
            )
            response = policy_generator.generate(request)
            prompt_candidates.append(response.text)
            candidates.append(
                {
                    "prompt_id": prompt_id,
                    "response_id": response_id(
                        context.run_id, SeedFamily.STATIC_CANDIDATE, prompt_id, 0, sample_index
                    ),
                    "sample_index": sample_index,
                    "seed": seed,
                    "temperature": request.temperature,
                    "top_p": request.top_p,
                    "response_text": response.text,
                    "provider_call": {
                        "requested_model": response.requested_model,
                        "returned_model": response.returned_model,
                        "request_id": response.request_id,
                        "created_at": response.created_at,
                        "retry_count": response.retry_count,
                    },
                }
            )
        rubric_request = GenerationRequest(
            prompt_id=prompt_id,
            messages=tuple(prompt["messages"])
            + (
                {
                    "role": "user",
                    "content": "Candidate responses:\n"
                    + json.dumps(prompt_candidates, ensure_ascii=False),
                },
            ),
            family="rubric_generator",
            seed=derive_seed(context.run_id, SeedFamily.PAIRING, prompt_id, 0, 0),
            json_schema=_fake_generator_schema(context),
            schema_name="rubric_generator_v1",
            reasoning_effort=str(context.config.models["rubric_generator"]["reasoning_effort"]),
        )
        generator_result = generator.generate(rubric_request)
        generated = json.loads(generator_result.text)
        rubric = build_static_rubric(
            prompt_id,
            [item["text"] for item in generated["criteria"]],
            similarity=lambda left, right: sum(
                a * b for a, b in zip(*embedding.embed([left, right]))
            ),
        )
        rubric_row = _rubric_dict(rubric)
        rubric_row["provenance"]["semantic_embedding"] = dict(embedding.identity)
        rubric_row["generator_call"] = {
            "requested_model": generator_result.requested_model,
            "returned_model": generator_result.returned_model,
            "request_id": generator_result.request_id,
            "created_at": generator_result.created_at,
            "retry_count": generator_result.retry_count,
            "reasoning_effort": rubric_request.reasoning_effort,
            "generation_parameters": {
                "temperature": rubric_request.temperature,
                "top_p": rubric_request.top_p,
                "max_output_tokens": rubric_request.max_output_tokens,
            },
            "prompt_hash": sha256_json([dict(message) for message in rubric_request.messages]),
            "schema_hash": sha256_json(rubric_request.json_schema),
        }
        rubrics.append(rubric_row)
    write_jsonl_atomic(context.stage_root() / "static_candidates.jsonl", candidates)
    write_jsonl_atomic(context.stage_root() / "static_rubrics.jsonl", rubrics)
    result = {
        "prompts": len(rubrics),
        "candidates": len(candidates),
        "criteria_per_prompt": 8,
        "equal_weight": 1 / 8,
        "gold_access": False,
        "provider_calls": {
            "policy": policy_generator.calls,
            "rubric_generator": generator.calls,
        },
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def replay_operator(context: PipelineContext) -> dict[str, Any]:
    replay = dict(context.raw.get("replay", {}))
    replay["criterion_embedding"] = context.config.models.get("criterion_embedding", {})
    replay["rubric_generator"] = context.config.models.get("rubric_generator", {})
    replay["proxy_grader"] = context.config.models.get("proxy_grader", {})
    replay["dynamic_schema_hash"] = sha256_file(
        context.root / "configs" / "schemas" / "dynamic_candidate_v1.json"
    )
    return replay


def run_freeze_updater(context: PipelineContext) -> dict[str, Any]:
    development_replay = context.run_root / "replay-dynamic-development" / "replay_snapshots.jsonl"
    context.begin_stage(inputs=(development_replay,), metadata={"source_split": "development_only"})
    lock_path = context.run_root / "updater_lock.json"
    lock = write_updater_lock(lock_path, replay_operator(context), context.run_id)
    write_json_atomic(context.stage_root() / "result.json", lock)
    return lock


# Public compatibility aliases: the credential-free lane exercises the same
# pairing, validation, and resumable-shard contracts as the live design.
from .offline_orchestration import (  # noqa: E402
    run_replay_dynamic as run_replay_dynamic,
    run_train_static as _run_train_static_offline,
)
from .training.live_online import (  # noqa: E402
    preflight_online_training,
    run_live_online_training,
)


def run_train_static(context: PipelineContext) -> dict[str, Any]:
    if context.mode == "fake":
        return _run_train_static_offline(context)
    rubric_path = context.run_root / "generate-static" / "static_rubrics.jsonl"
    reference_path = (
        context.run_root / "generate-static" / "pi0_reference_responses.jsonl"
    )
    split_paths = tuple(
        context.public_root / f"{spec.name}.jsonl" for spec in _configured_splits(context.raw)
    )
    context.begin_stage(
        inputs=(rubric_path, reference_path, *split_paths),
        metadata={
            "reward_source": "static_r0_only",
            "after_optimizer_update": True,
            "resume_unit": "veRL_checkpoint_and_immutable_response_cache",
            "policy_gpu": 0,
            "proxy_grader_gpu": 1,
        },
    )
    lock = read_json(context.root / "environment" / "upstream-lock.json")
    return run_live_static_training(context, rubric_path, reference_path, lock)


def run_train_online(context: PipelineContext, *, resume: bool = False) -> dict[str, Any]:
    """Run the separate causal online lane without weakening static training guards."""

    if context.mode == "fake":
        raise StageError(
            "train-online is a live veRL stage; use the fake online coordinator integration test"
        )
    if context.config.online_training is None:
        raise StageError("config has no online_training section")
    preflight = preflight_online_training(context, resume=resume)
    train_path = context.public_root / "train.jsonl"
    development_path = context.public_root / "development.jsonl"
    if not resume:
        context.begin_stage(
            inputs=(train_path, development_path),
            metadata={
                "runtime_claim": context.config.online_training.runtime_claim,
                "control_policy": context.config.online_training.control_policy,
                "same_step_causal": True,
                "reward_source": "online_r0_union_elicited_same_step",
                "criteria_scope": "prompt_step_ephemeral",
                "failure_policy": "fail_closed",
                "checkpoint_interval_steps": (
                    context.config.online_training.checkpoint_interval_steps
                ),
                "checkpoint_steps": list(context.config.online_training.checkpoint_steps),
            },
        )
    else:
        manifest = context.stage_root() / "manifest.json"
        if not manifest.is_file():
            raise StageError("resume-online requires the original train-online manifest")
    return run_live_online_training(context, preflight, resume=resume)
