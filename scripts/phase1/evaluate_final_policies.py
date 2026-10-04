#!/usr/bin/env python3
"""Resumable, fail-closed downstream evaluation for final Phase-1 policies.

The runner deliberately separates policy generation from rubric grading because the
single policy endpoint is expected to be restarted with one model at a time.  Every
successful response and criterion grade is stored as an immutable JSON artifact.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import canonical_json_bytes, sha256_file
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter, VLLMChatError


SCHEMA_VERSION = 1
GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "explanation": {"type": "string"},
        "criteria_met": {"type": "boolean"},
    },
    "required": ["explanation", "criteria_met"],
    "additionalProperties": False,
}
REQUIRED_MODEL_NAMES = {
    "static_base",
    "static_final",
    "online_base",
    "online_final",
}
ALLOWED_METHODS = {"static", "online"}
ALLOWED_MODEL_ROLES = {"base", "checkpoint", "final"}


class EvaluationError(RuntimeError):
    """The run cannot continue without violating its evaluation contract."""


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _resolve_path(root: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvaluationError(message)


def _literal_assignment(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                expression = node.value
                if (
                    isinstance(expression, ast.Call)
                    and isinstance(expression.func, ast.Attribute)
                    and expression.func.attr == "strip"
                    and not expression.args
                    and not expression.keywords
                ):
                    expression = expression.func.value
                value = ast.literal_eval(expression)
                _require(isinstance(value, str) and value.strip(), f"{name} is not a string")
                return value.strip()
    raise EvaluationError(f"could not find literal assignment {name}")


def load_official_grader_template(config: Mapping[str, Any], root: Path) -> tuple[str, dict]:
    upstream = config.get("healthbench_upstream")
    _require(isinstance(upstream, Mapping), "healthbench_upstream mapping is required")
    repository = _resolve_path(root, str(upstream.get("repository", "")))
    commit = str(upstream.get("commit", ""))
    source_path = str(upstream.get("source_path", "healthbench_eval.py"))
    _require(repository.is_dir() and commit, "pinned HealthBench repository and commit required")
    try:
        source = subprocess.run(
            ["git", "-C", str(repository), "show", f"{commit}:{source_path}"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except subprocess.CalledProcessError as exc:
        raise EvaluationError(f"cannot read pinned HealthBench source: {exc.stderr.strip()}") from exc
    template = _literal_assignment(source, "GRADER_TEMPLATE")
    expected = upstream.get("grader_template_sha256")
    actual = hashlib.sha256(template.encode()).hexdigest()
    if expected is not None:
        _require(actual == expected, "official HealthBench grader template hash mismatch")
    return template, {
        "repository": str(repository),
        "commit": commit,
        "source_path": source_path,
        "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "grader_template_sha256": actual,
    }


def _criteria_for_rar(row: Mapping[str, Any]) -> list[dict]:
    r0 = row.get("r0")
    _require(isinstance(r0, Mapping), "RaR row is missing r0")
    raw = r0.get("criteria")
    _require(isinstance(raw, list) and raw, "RaR r0 criteria must be non-empty")
    criteria = []
    for item in raw:
        _require(isinstance(item, Mapping), "RaR criterion must be a mapping")
        weight = item.get("weight_units")
        _require(
            isinstance(weight, int) and not isinstance(weight, bool) and weight > 0,
            "RaR criterion weights must be positive integers",
        )
        criteria.append(
            {
                "criterion_id": str(item.get("criterion_id", "")),
                "criterion": str(item.get("criterion", "")),
                "points": weight,
                "tags": [],
            }
        )
    return criteria


def _criteria_for_healthbench(row: Mapping[str, Any]) -> list[dict]:
    raw = row.get("rubrics")
    _require(isinstance(raw, list) and raw, "HealthBench rubrics must be non-empty")
    criteria = []
    for index, item in enumerate(raw):
        _require(isinstance(item, Mapping), "HealthBench rubric item must be a mapping")
        points = item.get("points")
        _require(
            isinstance(points, (int, float))
            and not isinstance(points, bool)
            and math.isfinite(float(points)),
            "HealthBench points must be finite numbers",
        )
        criteria.append(
            {
                "criterion_id": str(item.get("id") or item.get("criterion_id") or index),
                "criterion": str(item.get("criterion", "")),
                "points": points,
                "tags": list(item.get("tags") or []),
            }
        )
    _require(sum(item["points"] for item in criteria if item["points"] > 0) > 0, "HealthBench rubric has no positive points")
    return criteria


def normalize_dataset_row(kind: str, row: Mapping[str, Any], index: int) -> dict:
    messages = row.get("messages") if kind == "rar" else row.get("prompt")
    if kind == "healthbench" and messages is None:
        messages = row.get("messages")
    _require(isinstance(messages, list) and messages, f"{kind} row has no messages")
    normalized_messages = []
    for message in messages:
        _require(isinstance(message, Mapping), "message must be a mapping")
        role, content = message.get("role"), message.get("content")
        _require(isinstance(role, str) and isinstance(content, str), "invalid message role/content")
        normalized_messages.append({"role": role, "content": content})
    prompt_id = _source_prompt_id(kind, row, index)
    criteria = _criteria_for_rar(row) if kind == "rar" else _criteria_for_healthbench(row)
    criterion_ids = [item["criterion_id"] for item in criteria]
    _require(all(criterion_ids) and len(set(criterion_ids)) == len(criterion_ids), "criterion IDs must be unique and non-empty")
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_kind": kind,
        "prompt_id": prompt_id,
        "messages": normalized_messages,
        "criteria": criteria,
        "reference_answer": row.get("reference_answer") if kind == "rar" else None,
        "source_row_sha256": _digest(row),
    }


def _source_prompt_id(kind: str, row: Mapping[str, Any], index: int) -> str:
    return str(row.get("prompt_id") or row.get("id") or f"{kind}-{index:06d}")


def load_config(config_path: Path, limit: int | None, output_override: Path | None = None) -> dict:
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), "config must be a mapping")
    root = config_path.parent.resolve()
    _require(raw.get("schema_version") == SCHEMA_VERSION, "unsupported config schema_version")
    datasets = raw.get("datasets")
    models = raw.get("models")
    _require(isinstance(datasets, Mapping) and datasets, "datasets mapping is required")
    _require(isinstance(models, Mapping) and models, "models mapping is required")
    normalized_models = {}
    for name, value in models.items():
        _require(isinstance(value, Mapping), f"{name}: model spec must be a mapping")
        method, role = value.get("method"), value.get("role")
        _require(method in ALLOWED_METHODS, f"{name}: unsupported method {method!r}")
        _require(role in ALLOWED_MODEL_ROLES, f"{name}: unsupported role {role!r}")
        step = value.get("step")
        if step is None:
            step = 0 if role == "base" else 48 if role == "final" else None
        _require(
            isinstance(step, int) and not isinstance(step, bool) and step >= 0,
            f"{name}: non-negative integer step required",
        )
        _require(bool(str(value.get("served_model", "")).strip()), f"{name}: served_model required")
        _require(bool(str(value.get("artifact", "")).strip()), f"{name}: artifact required")
        normalized_models[str(name)] = {**value, "step": step}
    for method in sorted({str(value["method"]) for value in normalized_models.values()}):
        method_models = [value for value in normalized_models.values() if value["method"] == method]
        _require(
            sum(value["role"] == "base" for value in method_models) == 1,
            f"{method}: exactly one base model is required",
        )
        steps = [int(value["step"]) for value in method_models]
        _require(len(steps) == len(set(steps)), f"{method}: checkpoint steps must be unique")
    resolved = dict(raw)
    resolved["config_path"] = str(config_path.resolve())
    resolved_datasets = {}
    for name, value in datasets.items():
        spec = {**value, "path": str(_resolve_path(root, str(value["path"])))}
        include_path = value.get("include_prompt_ids_from")
        if include_path is not None:
            _require(
                isinstance(include_path, str) and include_path.strip(),
                f"{name}: include_prompt_ids_from must be a non-empty path",
            )
            spec["include_prompt_ids_from"] = str(_resolve_path(root, include_path))
        resolved_datasets[name] = spec
    resolved["datasets"] = resolved_datasets
    resolved["output_root"] = str(
        output_override.resolve()
        if output_override is not None
        else _resolve_path(root, str(raw["output_root"]))
    )
    resolved["models"] = normalized_models
    reuse_roots = raw.get("reuse_responses_from", [])
    _require(isinstance(reuse_roots, list), "reuse_responses_from must be a list")
    resolved["reuse_responses_from"] = [
        str(_resolve_path(root, str(value))) for value in reuse_roots
    ]
    reuse_grade_roots = raw.get("reuse_grades_from", [])
    _require(isinstance(reuse_grade_roots, list), "reuse_grades_from must be a list")
    resolved["reuse_grades_from"] = [
        str(_resolve_path(root, str(value))) for value in reuse_grade_roots
    ]
    resolved["limit"] = limit
    resolved.setdefault("seed", 11)
    resolved.setdefault("bootstrap_replicates", 10_000)
    resolved.setdefault("generation", {})
    resolved.setdefault("grading", {})
    _require(int(resolved["bootstrap_replicates"]) > 0, "bootstrap_replicates must be positive")
    return resolved


def ordered_model_names(config: Mapping[str, Any]) -> list[str]:
    return sorted(
        config["models"],
        key=lambda name: (
            0 if config["models"][name]["method"] == "static" else 1,
            int(config["models"][name]["step"]),
            str(name),
        ),
    )


def run_directory(config: Mapping[str, Any]) -> Path:
    identity = {key: value for key, value in config.items() if key != "config_path"}
    label = "full" if config["limit"] is None else f"smoke-{config['limit']}"
    return Path(str(config["output_root"])) / f"{label}-{_digest(identity)[:16]}"


def prepare(config: Mapping[str, Any]) -> Path:
    output = run_directory(config)
    template, upstream = load_official_grader_template(config, Path(config["config_path"]).parent)
    prepared = []
    source_records = {}
    for dataset_name, spec in config["datasets"].items():
        path = Path(spec["path"])
        _require(path.is_file(), f"missing dataset: {path}")
        rows = read_jsonl(path)
        expected = int(spec["expected_count"])
        _require(len(rows) == expected, f"{dataset_name}: expected {expected} rows, found {len(rows)}")
        expected_hash = spec.get("sha256")
        actual_hash = sha256_file(path)
        if expected_hash is not None:
            _require(actual_hash == expected_hash, f"{dataset_name}: source hash mismatch")
        sample_count = spec.get("sample_count")
        include_path = spec.get("include_prompt_ids_from")
        required_ids: list[str] = []
        required_indices: list[int] = []
        include_provenance = None
        if include_path is not None:
            _require(sample_count is not None, f"{dataset_name}: required IDs need sample_count")
            include_file = Path(str(include_path))
            _require(include_file.is_file(), f"{dataset_name}: missing required-ID manifest")
            include_rows = read_jsonl(include_file)
            required_ids = [
                str(row["prompt_id"])
                for row in include_rows
                if row.get("dataset") == dataset_name and row.get("prompt_id") is not None
            ]
            _require(required_ids, f"{dataset_name}: required-ID manifest has no matching prompts")
            _require(
                len(required_ids) == len(set(required_ids)),
                f"{dataset_name}: duplicate IDs in required-ID manifest",
            )
            source_ids = [
                _source_prompt_id(str(spec["kind"]), row, index)
                for index, row in enumerate(rows)
            ]
            _require(len(source_ids) == len(set(source_ids)), f"{dataset_name}: duplicate source prompt IDs")
            index_by_id = {prompt_id: index for index, prompt_id in enumerate(source_ids)}
            missing_ids = sorted(set(required_ids) - set(index_by_id))
            _require(not missing_ids, f"{dataset_name}: required prompt IDs missing from source")
            required_indices = [index_by_id[prompt_id] for prompt_id in required_ids]
            include_provenance = {
                "path": str(include_file.resolve()),
                "sha256": sha256_file(include_file),
                "required_count": len(required_ids),
                "required_prompt_ids_sha256": _digest(sorted(required_ids)),
            }
        if sample_count is not None:
            _require(
                isinstance(sample_count, int) and not isinstance(sample_count, bool)
                and 0 < sample_count <= len(rows),
                f"{dataset_name}: sample_count must be in [1, {len(rows)}]",
            )
            sample_seed = int(spec.get("sample_seed", config["seed"]))
            _require(
                len(required_indices) <= sample_count,
                f"{dataset_name}: required prompt count exceeds sample_count",
            )
            required_index_set = set(required_indices)
            remaining_indices = [index for index in range(len(rows)) if index not in required_index_set]
            additional_count = sample_count - len(required_indices)
            selected_indices = sorted(
                required_indices
                + random.Random(sample_seed).sample(remaining_indices, additional_count)
            )
            sampled = [rows[index] for index in selected_indices]
            sampling = {
                "kind": (
                    "seeded_random_without_replacement_with_required_ids"
                    if required_indices
                    else "seeded_random_without_replacement"
                ),
                "sample_count": sample_count,
                "sample_seed": sample_seed,
                "source_indices_sha256": _digest(selected_indices),
            }
            if include_provenance is not None:
                sampling["required_ids"] = include_provenance
                sampling["additional_random_count"] = additional_count
        else:
            sampled = rows
            sampling = {"kind": "all_rows", "sample_count": len(rows)}
        count = min(len(sampled), config["limit"]) if config["limit"] is not None else len(sampled)
        selected = sampled[:count]
        normalized = [normalize_dataset_row(str(spec["kind"]), row, index) for index, row in enumerate(selected)]
        ids = [row["prompt_id"] for row in normalized]
        _require(len(ids) == len(set(ids)), f"{dataset_name}: duplicate prompt IDs")
        for row in normalized:
            row["dataset"] = dataset_name
        prepared.extend(normalized)
        source_records[dataset_name] = {
            "path": str(path.resolve()),
            "sha256": actual_hash,
            "source_count": len(rows),
            "selected_count": count,
            "criterion_count": sum(len(row["criteria"]) for row in normalized),
            "sampling": sampling,
        }
    reused = _reuse_existing_responses(output, config, prepared)
    reused_grades = _reuse_existing_grades(output, config, prepared)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "resolved_config": config,
        "resolved_config_sha256": _digest(config),
        "run_kind": "full" if config["limit"] is None else "smoke",
        "sources": source_records,
        "healthbench_upstream": upstream,
        "grader_template": template,
        "models": config["models"],
        "reused_responses": reused,
        "reused_grades": reused_grades,
    }
    write_json_atomic(output / "manifest.json", manifest, immutable=True)
    write_jsonl_atomic(output / "prepared" / "prompts.jsonl", prepared, immutable=True)
    return output


def _reuse_existing_responses(
    output: Path, config: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    roots = [Path(value) for value in config.get("reuse_responses_from", [])]
    counts = {name: 0 for name in config["models"]}
    for model_name, model in config["models"].items():
        for row in prompts:
            destination = _response_path(output, model_name, row["dataset"], row["prompt_id"])
            if destination.is_file():
                _validate_response_record(read_json(destination), model_name=model_name, row=row)
                counts[model_name] += 1
                continue
            for root in roots:
                source = _response_path(root, model_name, row["dataset"], row["prompt_id"])
                if not source.is_file():
                    continue
                record = read_json(source)
                _validate_response_record(record, model_name=model_name, row=row)
                _require(
                    record.get("model_artifact") == model.get("artifact"),
                    f"reused response artifact mismatch: {model_name}/{row['prompt_id']}",
                )
                destination.parent.mkdir(parents=True, exist_ok=True)
                try:
                    os.link(source, destination)
                except OSError:
                    shutil.copy2(source, destination)
                counts[model_name] += 1
                break
    return {
        "source_roots": [str(root.resolve()) for root in roots],
        "counts_by_model": counts,
        "total": sum(counts.values()),
        "method": "hardlink_with_copy_fallback",
    }


def _reuse_existing_grades(
    output: Path, config: Mapping[str, Any], prompts: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    roots = [Path(value) for value in config.get("reuse_grades_from", [])]
    counts = {name: 0 for name in config["models"]}
    for model_name in config["models"]:
        for row in prompts:
            response_path = _response_path(output, model_name, row["dataset"], row["prompt_id"])
            if not response_path.is_file():
                continue
            response = read_json(response_path)
            for criterion in row["criteria"]:
                destination = _criterion_path(
                    output, model_name, row["dataset"], row["prompt_id"], criterion["criterion_id"]
                )
                if destination.is_file():
                    counts[model_name] += 1
                    continue
                for root in roots:
                    source = _criterion_path(
                        root, model_name, row["dataset"], row["prompt_id"], criterion["criterion_id"]
                    )
                    if not source.is_file():
                        continue
                    record = read_json(source)
                    _validate_grade_record(
                        record, model_name=model_name, row=row, response=response, criterion=criterion
                    )
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        os.link(source, destination)
                    except OSError:
                        shutil.copy2(source, destination)
                    counts[model_name] += 1
                    break
    return {
        "source_roots": [str(root.resolve()) for root in roots],
        "counts_by_model": counts,
        "total": sum(counts.values()),
        "method": "prompt_and_criterion_matched_hardlink_with_copy_fallback",
    }


def _load_run(config: Mapping[str, Any]) -> tuple[Path, dict, list[dict]]:
    output = run_directory(config)
    manifest_path = output / "manifest.json"
    _require(manifest_path.is_file(), "prepare stage has not completed")
    manifest = read_json(manifest_path)
    _require(manifest["resolved_config_sha256"] == _digest(config), "resolved config changed")
    prompts = read_jsonl(output / "prepared" / "prompts.jsonl")
    expected = sum(item["selected_count"] for item in manifest["sources"].values())
    _require(len(prompts) == expected, "prepared prompt inventory is incomplete")
    return output, manifest, prompts


def _adapter(spec: Mapping[str, Any], cache: Path) -> VLLMChatAdapter:
    return VLLMChatAdapter(
        spec["base_url"],
        str(spec["served_model"]),
        cache,
        api_key=spec.get("api_key"),
        timeout_seconds=float(spec.get("timeout_seconds", 180)),
        max_retries=int(spec.get("max_retries", 4)),
        max_in_flight=spec.get("max_in_flight"),
    )


def _response_path(output: Path, model_name: str, dataset: str, prompt_id: str) -> Path:
    return output / "responses" / model_name / dataset / f"{_digest(prompt_id)}.json"


def _validate_response_record(
    record: Mapping[str, Any], *, model_name: str, row: Mapping[str, Any]
) -> None:
    _require(record.get("dataset") == row["dataset"], "cached response dataset mismatch")
    _require(record.get("prompt_id") == row["prompt_id"], "cached response prompt mismatch")
    _require(record.get("model_name") == model_name, "cached response model mismatch")
    _require(isinstance(record.get("text"), str) and record["text"].strip(), "cached response text missing")
    _require(isinstance(record.get("response_id"), str), "cached response ID missing")
    _require(isinstance(record.get("finish_reason"), str), "cached response finish_reason missing")


def generate(config: Mapping[str, Any], model_name: str) -> None:
    _require(model_name in config["models"], f"unknown model {model_name}")
    output, _, prompts = _load_run(config)
    model = config["models"][model_name]
    generation = config["generation"]
    adapter = _adapter(
        {**generation, "served_model": model["served_model"]},
        output / "provider_cache" / "generation" / model_name,
    )
    workers = int(generation.get("workers", 1))
    _require(workers > 0, "generation workers must be positive")
    generation_seed = int(generation.get("seed", config["seed"]))
    generation_temperature = float(generation.get("temperature", 0.0))
    generation_top_p = float(generation.get("top_p", 1.0))
    _require(generation_temperature >= 0.0, "generation temperature must be non-negative")
    _require(0.0 < generation_top_p <= 1.0, "generation top_p must be in (0, 1]")

    def invoke(row: Mapping[str, Any]) -> None:
        destination = _response_path(output, model_name, row["dataset"], row["prompt_id"])
        if destination.is_file():
            _validate_response_record(read_json(destination), model_name=model_name, row=row)
            return
        request = GenerationRequest(
            prompt_id=row["prompt_id"],
            messages=tuple(row["messages"]),
            family="phase1_final_policy_eval_generation",
            seed=generation_seed,
            temperature=generation_temperature,
            top_p=generation_top_p,
            max_output_tokens=int(generation.get("max_output_tokens", 2048)),
            metadata={
                "dataset": row["dataset"],
                "model_name": model_name,
                "model_artifact": model.get("artifact"),
                "thinking": False,
                "single_answer": True,
            },
        )
        result = adapter.generate(request)
        usage = dict(result.usage)
        finish_reason = usage.get("finish_reason")
        _require(isinstance(finish_reason, str) and finish_reason, "generation finish_reason missing")
        record = {
            "schema_version": SCHEMA_VERSION,
            "response_id": _digest([row["dataset"], model_name, row["prompt_id"], result.text]),
            "dataset": row["dataset"],
            "prompt_id": row["prompt_id"],
            "model_name": model_name,
            "model_method": model["method"],
            "model_role": model["role"],
            "model_artifact": model.get("artifact"),
            "requested_model": result.requested_model,
            "returned_model": result.returned_model,
            "text": result.text,
            "finish_reason": finish_reason,
            "truncated": finish_reason == "length",
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "total_tokens": usage.get("total_tokens"),
            "request_id": result.request_id,
            "retry_count": result.retry_count,
            "raw_response_hash": result.raw_response_hash,
            "provider": adapter.request_provenance(request),
            "generation_contract": {
                "seed": generation_seed,
                "temperature": generation_temperature,
                "top_p": generation_top_p,
                "thinking": False,
                "n": 1,
            },
        }
        _validate_response_record(record, model_name=model_name, row=row)
        write_json_atomic(destination, record, immutable=True)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(invoke, prompts))

def _criterion_path(output: Path, model_name: str, dataset: str, prompt_id: str, criterion_id: str) -> Path:
    return output / "grades" / model_name / dataset / _digest(prompt_id) / f"{_digest(criterion_id)}.json"


def _grader_prompt(template: str, row: Mapping[str, Any], response: Mapping[str, Any], criterion: Mapping[str, Any]) -> str:
    conversation = row["messages"] + [{"role": "assistant", "content": response["text"]}]
    convo_str = "\n\n".join(f"{message['role']}: {message['content']}" for message in conversation)
    rubric_item = f"[{criterion['points']}] {criterion['criterion']}"
    return template.replace("<<conversation>>", convo_str).replace("<<rubric_item>>", rubric_item)


def _validate_grade_record(
    record: Mapping[str, Any],
    *,
    model_name: str,
    row: Mapping[str, Any],
    response: Mapping[str, Any],
    criterion: Mapping[str, Any],
) -> None:
    _require(record.get("dataset") == row["dataset"], "cached grade dataset mismatch")
    _require(record.get("model_name") == model_name, "cached grade model mismatch")
    _require(record.get("prompt_id") == row["prompt_id"], "cached grade prompt mismatch")
    _require(record.get("response_id") == response["response_id"], "cached grade response mismatch")
    _require(record.get("criterion_id") == criterion["criterion_id"], "cached grade criterion mismatch")
    _require(type(record.get("criteria_met")) is bool, "cached grade has no boolean criteria_met")
    _require(isinstance(record.get("explanation"), str), "cached grade has no explanation")


def _grade_one(
    output: Path,
    template: str,
    adapter: VLLMChatAdapter,
    model_name: str,
    row: Mapping[str, Any],
    response: Mapping[str, Any],
    criterion: Mapping[str, Any],
    seed: int,
    max_output_tokens: int,
    parse_retries: int,
) -> None:
    destination = _criterion_path(
        output, model_name, row["dataset"], row["prompt_id"], criterion["criterion_id"]
    )
    if destination.is_file():
        _validate_grade_record(
            read_json(destination), model_name=model_name, row=row, response=response, criterion=criterion
        )
        return
    last_error = ""
    for parse_attempt in range(parse_retries + 1):
        recovery_mode = parse_attempt > 0
        grader_content = _grader_prompt(template, row, response, criterion)
        if recovery_mode:
            grader_content += (
                "\n\n# Serialization recovery\n"
                "The prior structured response could not be parsed. Return the same judgment as a "
                "single valid JSON object. Keep explanation under 40 words and do not repeat text."
            )
        request = GenerationRequest(
            prompt_id=f"{row['prompt_id']}:{criterion['criterion_id']}:{response['response_id']}",
            messages=({"role": "user", "content": grader_content},),
            family="phase1_final_policy_eval_grading",
            seed=seed + parse_attempt,
            temperature=0.0,
            top_p=1.0,
            max_output_tokens=min(max_output_tokens, 384) if recovery_mode else max_output_tokens,
            json_schema=GRADE_SCHEMA,
            schema_name="healthbench_criterion_grade_v1",
            metadata={
                "dataset": row["dataset"], "model_name": model_name, "prompt_id": row["prompt_id"],
                "response_id": response["response_id"], "criterion_id": criterion["criterion_id"],
                "parse_attempt": parse_attempt, "serialization_recovery": recovery_mode,
            },
        )
        try:
            result = adapter.generate(request)
        except VLLMChatError as exc:
            last_error = str(exc)
            continue
        try:
            parsed = json.loads(result.text)
            _require(type(parsed.get("criteria_met")) is bool, "grader omitted boolean criteria_met")
            _require(isinstance(parsed.get("explanation"), str), "grader omitted explanation")
        except (json.JSONDecodeError, EvaluationError) as exc:
            last_error = str(exc)
            write_json_atomic(
                output / "grade_failures" / model_name / row["dataset"] / _digest(row["prompt_id"])
                / f"{_digest(criterion['criterion_id'])}-attempt-{parse_attempt}.json",
                {
                    "schema_version": SCHEMA_VERSION, "model_name": model_name, "dataset": row["dataset"],
                    "prompt_id": row["prompt_id"], "response_id": response["response_id"],
                    "criterion_id": criterion["criterion_id"], "parse_attempt": parse_attempt,
                    "raw_text": result.text, "error": last_error, "request_id": result.request_id,
                    "raw_response_hash": result.raw_response_hash, "provider": adapter.request_provenance(request),
                },
                immutable=True,
            )
            continue
        record = {
            "schema_version": SCHEMA_VERSION, "dataset": row["dataset"], "model_name": model_name,
            "prompt_id": row["prompt_id"], "response_id": response["response_id"],
            "criterion_id": criterion["criterion_id"], "criterion": criterion["criterion"],
            "points": criterion["points"], "tags": criterion["tags"], "criteria_met": parsed["criteria_met"],
            "explanation": parsed["explanation"], "requested_model": result.requested_model,
            "returned_model": result.returned_model, "request_id": result.request_id,
            "retry_count": result.retry_count, "usage": dict(result.usage),
            "raw_response_hash": result.raw_response_hash, "provider": adapter.request_provenance(request),
            "parse_attempt": parse_attempt, "serialization_recovery": recovery_mode,
        }
        _validate_grade_record(
            record, model_name=model_name, row=row, response=response, criterion=criterion
        )
        write_json_atomic(destination, record, immutable=True)
        return
    raise EvaluationError(
        f"grader produced no valid grade after {parse_retries + 1} attempts: {last_error}"
    )

def grade(
    config: Mapping[str, Any],
    model_name: str | None = None,
    runtime_base_urls: Sequence[str] | None = None,
    dataset_name: str | None = None,
) -> None:
    output, manifest, prompts = _load_run(config)
    if dataset_name is not None:
        _require(dataset_name in config["datasets"], f"unknown dataset {dataset_name}")
        prompts = [row for row in prompts if row["dataset"] == dataset_name]
    grading = dict(config["grading"])
    if runtime_base_urls:
        grading["base_url"] = list(runtime_base_urls)
    adapter = _adapter(grading, output / "provider_cache" / "grading")
    selected_models = [model_name] if model_name else ordered_model_names(config)
    tasks = []
    for selected in selected_models:
        _require(selected in config["models"], f"unknown model {selected}")
        for row in prompts:
            response_path = _response_path(output, selected, row["dataset"], row["prompt_id"])
            _require(response_path.is_file(), f"missing response: {selected}/{row['dataset']}/{row['prompt_id']}")
            response = read_json(response_path)
            for criterion in row["criteria"]:
                tasks.append((selected, row, response, criterion))
    workers = int(grading.get("workers", 1))
    _require(workers > 0, "grading workers must be positive")

    def invoke(task: tuple) -> None:
        selected, row, response, criterion = task
        _grade_one(
            output,
            manifest["grader_template"],
            adapter,
            selected,
            row,
            response,
            criterion,
            int(config["seed"]),
            int(grading.get("max_output_tokens", 1024)),
            int(grading.get("parse_retries", 2)),
        )

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(invoke, tasks))


def score_prompt(criteria: Sequence[Mapping[str, Any]], grades: Sequence[Mapping[str, Any]]) -> float:
    _require(len(criteria) == len(grades), "criterion grade inventory is incomplete")
    by_id = {str(grade["criterion_id"]): grade for grade in grades}
    _require(len(by_id) == len(grades), "duplicate criterion grades")
    expected = {str(item["criterion_id"]) for item in criteria}
    _require(set(by_id) == expected, "criterion grade identities do not match rubric")
    denominator = sum(float(item["points"]) for item in criteria if float(item["points"]) > 0)
    _require(denominator > 0, "rubric has no positive-weight denominator")
    numerator = sum(
        float(item["points"])
        for item in criteria
        if by_id[str(item["criterion_id"])]["criteria_met"] is True
    )
    return numerator / denominator


def _percentile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1 - fraction) + ordered[upper] * fraction


def paired_bootstrap(
    base: Mapping[str, float], final: Mapping[str, float], *, seed: int, replicates: int, clip_aggregate: bool
) -> dict:
    _require(set(base) == set(final) and base, "paired models do not have identical prompt coverage")
    ids = sorted(base)

    def aggregate(values: Iterable[float]) -> float:
        values = list(values)
        mean = sum(values) / len(values)
        return min(1.0, max(0.0, mean)) if clip_aggregate else mean

    delta = aggregate(final[key] for key in ids) - aggregate(base[key] for key in ids)
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sample = [ids[rng.randrange(len(ids))] for _ in ids]
        draws.append(
            aggregate(final[key] for key in sample) - aggregate(base[key] for key in sample)
        )
    return {
        "base_mean": aggregate(base.values()),
        "final_mean": aggregate(final.values()),
        "delta_final_minus_base": delta,
        "bootstrap_95_ci": [_percentile(draws, 0.025), _percentile(draws, 0.975)],
        "bootstrap_seed": seed,
        "bootstrap_replicates": replicates,
        "n_prompts": len(ids),
        "aggregate_clip_0_1": clip_aggregate,
    }


def summarize(config: Mapping[str, Any]) -> dict:
    output, manifest, prompts = _load_run(config)
    scores: dict[tuple[str, str], dict[str, float]] = {}
    records = []
    model_names = ordered_model_names(config)
    for model_name in model_names:
        model = config["models"][model_name]
        for row in prompts:
            response = read_json(_response_path(output, model_name, row["dataset"], row["prompt_id"]))
            grades = []
            for criterion in row["criteria"]:
                path = _criterion_path(output, model_name, row["dataset"], row["prompt_id"], criterion["criterion_id"])
                _require(path.is_file(), f"missing criterion grade: {model_name}/{row['dataset']}/{row['prompt_id']}/{criterion['criterion_id']}")
                grade_record = read_json(path)
                _require(grade_record["response_id"] == response["response_id"], "grade points to a different response")
                grades.append(grade_record)
            raw_score = score_prompt(row["criteria"], grades)
            records.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "dataset": row["dataset"],
                    "dataset_kind": row["dataset_kind"],
                    "model_name": model_name,
                    "model_method": model["method"],
                    "model_role": model["role"],
                    "global_step": model["step"],
                    "prompt_id": row["prompt_id"],
                    "response_id": response["response_id"],
                    "raw_score_unclipped": raw_score,
                    "criterion_count": len(grades),
                }
            )
            scores.setdefault((row["dataset"], model_name), {})[row["prompt_id"]] = raw_score
    comparisons = {}
    trajectories = {}
    methods = sorted({config["models"][name]["method"] for name in model_names})
    for dataset_name, source in manifest["sources"].items():
        kind = str(config["datasets"][dataset_name]["kind"])
        comparisons[dataset_name] = {}
        trajectories[dataset_name] = {}
        for method in methods:
            method_names = [
                name for name in model_names if config["models"][name]["method"] == method
            ]
            if not method_names:
                continue
            base = next(name for name in method_names if config["models"][name]["role"] == "base")
            rows = []
            for name in method_names:
                comparison = paired_bootstrap(
                    scores[(dataset_name, base)],
                    scores[(dataset_name, name)],
                    seed=int(config["seed"]) + int(config["models"][name]["step"]),
                    replicates=int(config["bootstrap_replicates"]),
                    clip_aggregate=kind == "healthbench",
                )
                rows.append(
                    {
                        "model_name": name,
                        "method": method,
                        "role": config["models"][name]["role"],
                        "global_step": int(config["models"][name]["step"]),
                        **comparison,
                    }
                )
            trajectories[dataset_name][method] = rows
            final_names = [
                name for name in method_names if config["models"][name]["role"] == "final"
            ]
            if final_names:
                final = max(final_names, key=lambda name: int(config["models"][name]["step"]))
                comparisons[dataset_name][method] = next(
                    row for row in rows if row["model_name"] == final
                )
        _require(
            all(len(scores[(dataset_name, name)]) == source["selected_count"] for name in model_names),
            f"{dataset_name}: incomplete model score inventory",
        )
    write_jsonl_atomic(output / "summary" / "prompt_scores.jsonl", records, immutable=True)
    trajectory_records = [
        {"schema_version": SCHEMA_VERSION, "dataset": dataset, **row}
        for dataset, method_rows in trajectories.items()
        for rows in method_rows.values()
        for row in rows
    ]
    write_jsonl_atomic(
        output / "summary" / "checkpoint_trajectory.jsonl", trajectory_records, immutable=True
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "run_kind": manifest["run_kind"],
        "resolved_config_sha256": manifest["resolved_config_sha256"],
        "comparisons": comparisons,
        "trajectories": trajectories,
        "notes": [
            "One response per model and prompt; no best-of-N.",
            "HealthBench prompt scores are not clipped; only each dataset aggregate mean is clipped to [0,1].",
            "Static and OnlineRubrics gains are own-base paired comparisons; their difference is not a rubric-only causal effect when backbones differ.",
        ],
    }
    write_json_atomic(output / "summary" / "summary.json", summary, immutable=True)
    return summary


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("prepare", "generate", "grade", "summarize"), required=True)
    parser.add_argument("--model")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    if args.stage == "generate" and args.model is None:
        parser.error("--stage generate requires --model because the policy endpoint is externally switched")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = load_config(args.config.resolve(), args.limit, args.output_dir)
    if args.stage == "prepare":
        output = prepare(config)
        print(output)
    elif args.stage == "generate":
        generate(config, args.model)
    elif args.stage == "grade":
        grade(config, args.model)
    else:
        print(json.dumps(summarize(config), ensure_ascii=False, sort_keys=True, indent=2))

if __name__ == "__main__":
    main()

