#!/usr/bin/env python3
"""Resumable held-out validation audit for Static and OnlineRubrics trajectories.

This is deliberately separate from the fixed-train-probe implementation.  It never
trains a model and never reads or writes optimizer state.  Policy checkpoints are
served one at a time from Hugging Face inference exports; the other stages only
create or consume immutable analysis artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import random
import shutil
import statistics
import subprocess
import tempfile
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from dynamic_rubric.artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import canonical_json_bytes, sha256_file
from dynamic_rubric.phase1.audit_scoring import AuditScoreConfig, score_pool
from dynamic_rubric.prompt_versions.onlinerubric_prompt import (
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter, VLLMChatError
from dynamic_rubric.rubrics.extractor import make_blind_pairing
from dynamic_rubric.training.online_contracts import WeightedCriterion
from dynamic_rubric.training.online_step import (
    DEDUP_SCHEMA,
    EXTRACTION_SCHEMA,
    _parse_dedup,
    _parse_extraction,
    _receipt,
)


SCHEMA_VERSION = 1
DATA_ROLE = "heldout_validation"
POOL_A_COUNT = 8
POOL_B_COUNT = 16
CONTROL_COUNT = 8
METHODS = ("static", "online")
STATIC_STEPS = (0, 3, 6, 9, 13, 16, 24, 32, 40, 48)
ONLINE_STEPS = (0, 3, 6, 9, 12, 13, 15, 16, 18, 21, 24, 27, 30, 32, 33, 34, 36, 39, 40, 42, 45, 48)
EXPECTED_STEPS_BY_METHOD = {"static": STATIC_STEPS, "online": ONLINE_STEPS}
APPROVED_JUDGE_BASE_URLS = (
    "http://127.0.0.1:28132/v1",
    "http://127.0.0.1:28133/v1",
)
APPROVED_RUNTIME_JUDGE_BASE_URLS = frozenset(
    (
        *APPROVED_JUDGE_BASE_URLS,
        "http://127.0.0.1:28134/v1",
        "http://127.0.0.1:28135/v1",
        "http://127.0.0.1:28136/v1",
        "http://127.0.0.1:28137/v1",
        "http://127.0.0.1:28138/v1",
    )
)


class HeldoutAuditError(RuntimeError):
    """An artifact would violate the held-out validation contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise HeldoutAuditError(message)


def _digest(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _resolve(base: Path, value: str) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def load_config(path: Path, output_override: Path | None = None) -> dict[str, Any]:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    _require(isinstance(raw, Mapping), "config must be a mapping")
    _require(raw.get("schema_version") == SCHEMA_VERSION, "unsupported schema_version")
    _require(raw.get("data_role") == DATA_ROLE, "data_role must be heldout_validation")
    schedules = raw.get("checkpoint_steps")
    _require(isinstance(schedules, Mapping), "checkpoint_steps must be method-specific")
    _require(tuple(schedules.get("static", ())) == STATIC_STEPS, "static checkpoint schedule drift")
    _require(tuple(schedules.get("online", ())) == ONLINE_STEPS, "online checkpoint schedule drift")
    _require(raw.get("training_enabled") is False, "this audit must not enable training")
    _require(raw.get("optimizer_required") is False, "this audit must not require optimizer state")
    _require(set(raw.get("methods", {})) == set(METHODS), "static and online methods are required")
    base = path.parent.resolve()
    config = dict(raw)
    config["config_path"] = str(path.resolve())
    config["output_root"] = str(
        output_override.resolve() if output_override else _resolve(base, str(raw["output_root"]))
    )
    config["datasets"] = {
        name: {**spec, "path": str(_resolve(base, str(spec["path"])))}
        for name, spec in raw["datasets"].items()
    }
    _require(int(config["validation"]["count"]) == 100, "validation count must be 100")
    _require(int(config["validation"]["sample_seed"]) == 11, "validation sample seed must be 11")
    _require(float(config["metrics"]["pairwise_tie_epsilon"]) >= 0, "invalid tie epsilon")
    judge_urls = config["grading"].get("base_urls")
    _require(
        tuple(judge_urls or ()) == APPROVED_JUDGE_BASE_URLS,
        "judge must use exactly the approved Inference B tunnel and Trainer local endpoints",
    )
    return config


def output_root(config: Mapping[str, Any]) -> Path:
    return Path(str(config["output_root"]))


def _load_source(config: Mapping[str, Any], name: str) -> list[dict[str, Any]]:
    spec = config["datasets"][name]
    path = Path(spec["path"])
    _require(path.is_file(), f"missing {name} dataset: {path}")
    _require(sha256_file(path) == spec["sha256"], f"{name} source hash mismatch")
    rows = read_jsonl(path)
    _require(len(rows) == int(spec["expected_count"]), f"{name} row count mismatch")
    ids = [str(row.get("prompt_id", "")) for row in rows]
    _require(all(ids) and len(ids) == len(set(ids)), f"{name} prompt IDs invalid")
    return rows


def _sample_validation(rows: Sequence[Mapping[str, Any]], count: int, seed: int) -> list[dict]:
    indexed = sorted((str(row["prompt_id"]), dict(row)) for row in rows)
    rng = random.Random(seed)
    selected_ids = set(rng.sample([item[0] for item in indexed], count))
    return [row for prompt_id, row in indexed if prompt_id in selected_ids]


def prepare(config: Mapping[str, Any]) -> Path:
    """Freeze one validation100 manifest and the complete HF-backed work plan."""
    development = _load_source(config, "development")
    train = _load_source(config, "train")
    test = _load_source(config, "heldout_test")
    selected = _sample_validation(
        development, int(config["validation"]["count"]), int(config["validation"]["sample_seed"])
    )
    selected_ids = {str(row["prompt_id"]) for row in selected}
    train_ids = {str(row["prompt_id"]) for row in train}
    test_ids = {str(row["prompt_id"]) for row in test}
    _require(not selected_ids & train_ids, "validation IDs overlap training IDs")
    _require(not selected_ids & test_ids, "validation IDs overlap held-out test IDs")
    expected_prompt_ids_sha256 = config["validation"].get("prompt_ids_sha256")
    _require(
        expected_prompt_ids_sha256 is None
        or str(expected_prompt_ids_sha256) == _digest(sorted(selected_ids)),
        "validation100 prompt ID manifest drift",
    )
    root = output_root(config)
    prompt_rows = []
    for row in selected:
        prompt_rows.append(
            {
                "schema_version": SCHEMA_VERSION,
                "data_role": DATA_ROLE,
                "domain": "medicine",
                "prompt_id": str(row["prompt_id"]),
                "messages": row["messages"],
                "r0": row["r0"],
                "source_split": "development",
                "source_row_sha256": _digest(row),
            }
        )
    write_jsonl_atomic(root / "manifests" / "validation_prompts.jsonl", prompt_rows)
    plan = []
    for method in METHODS:
        method_spec = config["methods"][method]
        for step in EXPECTED_STEPS_BY_METHOD[method]:
            repo = (
                method_spec["base_repo"]
                if step == 0
                else method_spec["checkpoint_repo_template"].format(step=step)
            )
            revisions = method_spec.get("checkpoint_revisions", {})
            revision = (
                method_spec["base_revision"]
                if step == 0
                else revisions.get(
                    step, revisions.get(str(step), method_spec.get("checkpoint_revision", "main"))
                )
            )
            plan.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "data_role": DATA_ROLE,
                    "method": method,
                    "policy_step": step,
                    "hf_repo_id": repo,
                    "hf_revision_requested": revision,
                    "inference_parameters_only": True,
                    "optimizer_required": False,
                    "pool_a_per_prompt": POOL_A_COUNT,
                    "pool_b_per_prompt": POOL_B_COUNT,
                    "fresh_rubric_mode": "r0" if step == 0 else "step_local_r0_union_new_criteria",
                }
            )
    write_jsonl_atomic(root / "manifests" / "checkpoint_work_plan.jsonl", plan)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "data_role": DATA_ROLE,
        "analysis_only": True,
        "training_enabled": False,
        "optimizer_required": False,
        "validation_source": "development",
        "validation_count": len(prompt_rows),
        "validation_prompt_ids_sha256": _digest(sorted(selected_ids)),
        "train_prompt_ids_sha256": _digest(sorted(train_ids)),
        "heldout_test_prompt_ids_sha256": _digest(sorted(test_ids)),
        "validation_train_overlap": 0,
        "validation_test_overlap": 0,
        "checkpoint_steps": {
            method: list(steps) for method, steps in EXPECTED_STEPS_BY_METHOD.items()
        },
        "methods": list(METHODS),
        "resolved_config": dict(config),
    }
    write_json_atomic(root / "manifest.json", identity)
    return root


def preflight(config: Mapping[str, Any]) -> dict[str, Any]:
    root = prepare(config)
    prompts = read_jsonl(root / "manifests" / "validation_prompts.jsonl")
    plan = read_jsonl(root / "manifests" / "checkpoint_work_plan.jsonl")
    report = {
        "offline_manifest_valid": len(prompts) == 100
        and len(plan) == sum(map(len, EXPECTED_STEPS_BY_METHOD.values())),
        "remote_endpoints_called": False,
        "gpu_required": False,
        "data_role": DATA_ROLE,
        "validation_prompts": len(prompts),
        "policy_checkpoint_tasks": len(plan),
        "matrix_cells_per_method": {
            method: len(steps) * (len(steps) + 1) // 2
            for method, steps in EXPECTED_STEPS_BY_METHOD.items()
        },
        "signs": {
            "mad_gain": "current_minus_stale",
            "zar_gain": "stale_minus_current",
            "ptr_gain": "stale_minus_current",
        },
        "pairwise_tie_epsilon": float(config["metrics"]["pairwise_tie_epsilon"]),
    }
    write_json_atomic(root / "preflight.json", report)
    return report


def _response_id(method: str, step: int, prompt_id: str, pool: str, sample_index: int) -> str:
    return _digest([DATA_ROLE, method, step, prompt_id, pool, sample_index])


def _adapter(spec: Mapping[str, Any], root: Path, model: str) -> VLLMChatAdapter:
    base_url = spec["base_url"]
    if isinstance(base_url, Sequence) and not isinstance(base_url, str):
        return _JudgeReplicaAdapter(spec, root, model)
    return VLLMChatAdapter(
        base_url,
        model,
        root / "provider_cache",
        timeout_seconds=float(spec.get("timeout_seconds", 600)),
        max_retries=int(spec.get("max_retries", 4)),
        max_in_flight=int(spec.get("max_in_flight", spec.get("workers", 16))),
    )


class _JudgeReplicaAdapter:
    """Deterministically shard judge calls and fail over to the other verified replica."""

    def __init__(self, spec: Mapping[str, Any], root: Path, model: str) -> None:
        base_urls = tuple(str(url) for url in spec["base_url"])
        _verify_runtime_judge_inventory(base_urls)
        common = {
            "timeout_seconds": float(spec.get("timeout_seconds", 600)),
            "max_retries": int(spec.get("max_retries", 4)),
            "max_in_flight": int(spec.get("max_in_flight", spec.get("workers", 16))),
        }
        self._router = VLLMChatAdapter(base_urls, model, root / "router", **common)
        self._replicas = {
            url: VLLMChatAdapter(url, model, root / f"replica-{index}", **common)
            for index, url in enumerate(base_urls)
        }
        self._base_urls = base_urls
        self._transport_by_request: dict[int, dict[str, Any]] = {}
        self._transport_lock = threading.Lock()

    def generate(self, request: GenerationRequest) -> Any:
        primary = self._router.request_provenance(request)["selected_base_url"]
        ordered = (primary, *(url for url in self._base_urls if url != primary))
        last_error: VLLMChatError | None = None
        for attempt, url in enumerate(ordered):
            adapter = self._replicas[url]
            try:
                result = adapter.generate(request)
            except VLLMChatError as error:
                last_error = error
                continue
            transport = {
                **adapter.request_provenance(request),
                "configured_base_urls": list(self._base_urls),
                "primary_base_url": primary,
                "failover_used": attempt > 0,
            }
            with self._transport_lock:
                self._transport_by_request[id(request)] = transport
            return result
        raise VLLMChatError(f"all verified judge replicas failed: {last_error}") from last_error

    def request_provenance(self, request: GenerationRequest) -> dict[str, Any]:
        with self._transport_lock:
            transport = self._transport_by_request.pop(id(request), None)
        _require(transport is not None, "judge transport provenance missing after generation")
        return transport


def _generate_one(
    adapter: Any,
    config: Mapping[str, Any],
    method: str,
    step: int,
    prompt: Mapping[str, Any],
    pool: str,
    index: int,
) -> dict[str, Any]:
    generation = config["policy_generation"]
    seed = (
        int(config["seed"]) * 10_000_000
        + (0 if method == "static" else 1) * 1_000_000
        + step * 10_000
        + (0 if pool == "probe_A" else 1000)
        + index
    )
    result = adapter.generate(
        GenerationRequest(
            prompt_id=str(prompt["prompt_id"]),
            messages=tuple(prompt["messages"]),
            family="heldout_validation_policy_generation",
            seed=seed,
            temperature=float(generation["temperature"]),
            top_p=float(generation["top_p"]),
            max_output_tokens=int(generation["max_output_tokens"]),
            metadata={
                "data_role": DATA_ROLE,
                "method": method,
                "policy_step": step,
                "pool": pool,
                "sample_index": index,
                "no_gradient": True,
            },
        )
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "data_role": DATA_ROLE,
        "domain": "medicine",
        "method": method,
        "global_step": step,
        "checkpoint_id": str(step),
        "policy_checkpoint": str(step),
        "evaluator_checkpoint": None,
        "fresh_or_stale": None,
        "prompt_id": str(prompt["prompt_id"]),
        "response_id": _response_id(method, step, str(prompt["prompt_id"]), pool, index),
        "pool": pool,
        "sample_index": index,
        "text": result.text,
        "seed": seed,
        "generation": {
            "requested_model": result.requested_model,
            "returned_model": result.returned_model,
            "request_id": result.request_id,
            "usage": dict(result.usage),
            "raw_response_hash": result.raw_response_hash,
        },
        "used_for_gradient": False,
    }


def _validate_pool_rows(
    rows: Sequence[Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
    method: str,
    step: int,
    pool: str,
    count: int,
) -> None:
    expected_prompts = {str(row["prompt_id"]) for row in prompts}
    _require(len(rows) == len(prompts) * count, f"{pool} count mismatch")
    _require({str(row["prompt_id"]) for row in rows} == expected_prompts, f"{pool} prompts drift")
    _require(
        all(
            row.get("data_role") == DATA_ROLE
            and row.get("method") == method
            and int(row.get("global_step", -1)) == step
            and row.get("pool") == pool
            and row.get("used_for_gradient") is False
            for row in rows
        ),
        f"{pool} provenance mismatch",
    )
    per_prompt: dict[str, set[int]] = {}
    for row in rows:
        per_prompt.setdefault(str(row["prompt_id"]), set()).add(int(row["sample_index"]))
    _require(
        all(indexes == set(range(count)) for indexes in per_prompt.values()),
        f"{pool} indexes invalid",
    )
    ids = [str(row["response_id"]) for row in rows]
    _require(len(ids) == len(set(ids)), f"{pool} response IDs are not unique")


def generate(config: Mapping[str, Any], method: str, step: int, served_model: str) -> Path:
    _require(method in METHODS and step in EXPECTED_STEPS_BY_METHOD[method], "invalid method/step")
    root = prepare(config)
    prompts_path = root / "manifests" / "validation_prompts.jsonl"
    prompts = read_jsonl(prompts_path)
    target = root / "responses" / method / f"step-{step:03d}"
    seal_path = target / "sealed_manifest.json"
    plan = next(
        row
        for row in read_jsonl(root / "manifests" / "checkpoint_work_plan.jsonl")
        if row["method"] == method and row["policy_step"] == step
    )
    identity = {
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": step,
        "served_model": served_model,
        "hf_repo_id": plan["hf_repo_id"],
        "hf_revision_requested": plan["hf_revision_requested"],
        "prompt_manifest_sha256": sha256_file(prompts_path),
        "generation": dict(config["policy_generation"]),
    }
    if seal_path.is_file():
        _validate_seal(seal_path, identity)
        return target
    _verify_served_model(config["policy_generation"]["base_url"], served_model)
    adapter = _adapter(
        config["policy_generation"], root / "cache" / method / f"step-{step:03d}", served_model
    )
    paths = []
    for pool, count in (("probe_A", POOL_A_COUNT), ("probe_B", POOL_B_COUNT)):
        path = target / f"{pool}.jsonl"
        jobs = [(prompt, pool, index) for prompt in prompts for index in range(count)]
        with ThreadPoolExecutor(
            max_workers=int(config["policy_generation"]["workers"])
        ) as executor:
            rows = list(
                executor.map(lambda job: _generate_one(adapter, config, method, step, *job), jobs)
            )
        _require(
            all(
                row["generation"]["requested_model"] == served_model
                and row["generation"]["returned_model"] == served_model
                for row in rows
            ),
            "policy runtime model identity mismatch",
        )
        _validate_pool_rows(rows, prompts, method, step, pool, count)
        write_jsonl_atomic(path, rows)
        paths.append(path)
    a, b = read_jsonl(paths[0]), read_jsonl(paths[1])
    _require(
        not ({row["response_id"] for row in a} & {row["response_id"] for row in b}),
        "Pool A/B overlap",
    )
    provenance = target / "provenance.json"
    write_json_atomic(
        provenance,
        {
            "schema_version": 1,
            "data_role": DATA_ROLE,
            "method": method,
            "policy_step": step,
            "pool_counts": {"probe_A": len(a), "probe_B": len(b)},
            "pool_a_b_disjoint": True,
            "used_for_gradient": False,
            "served_model": served_model,
            "hf_repo_id": plan["hf_repo_id"],
            "hf_revision_requested": plan["hf_revision_requested"],
        },
    )
    _seal(seal_path, identity, [*paths, provenance])
    return target


def generate_control(config: Mapping[str, Any], method: str, served_model: str) -> Path:
    """Generate the method's own immutable pi0 controls with bounded concurrency."""
    _require(method in METHODS, "invalid method")
    root = prepare(config)
    prompts_path = root / "manifests" / "validation_prompts.jsonl"
    prompts = read_jsonl(prompts_path)
    target = root / "responses" / method / "pi0_control.jsonl"
    seal_path = target.with_suffix(".seal.json")
    method_spec = config["methods"][method]
    identity = {
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": 0,
        "served_model": served_model,
        "hf_repo_id": method_spec["base_repo"],
        "hf_revision_requested": method_spec["base_revision"],
        "prompt_manifest_sha256": sha256_file(prompts_path),
        "generation": dict(config["policy_generation"]),
        "control_count": CONTROL_COUNT,
    }
    if seal_path.is_file():
        _validate_seal(seal_path, identity)
        return target
    _verify_served_model(config["policy_generation"]["base_url"], served_model)
    adapter = _adapter(config["policy_generation"], root / "cache" / method / "pi0", served_model)
    generation = config["policy_generation"]

    def one(job: tuple[Mapping[str, Any], int]) -> dict[str, Any]:
        prompt, index = job
        seed = int(config["seed"]) * 100_000 + (0 if method == "static" else 1) * 10_000 + index
        result = adapter.generate(
            GenerationRequest(
                prompt_id=str(prompt["prompt_id"]),
                messages=tuple(prompt["messages"]),
                family="heldout_validation_pi0_control",
                seed=seed,
                temperature=float(generation["temperature"]),
                top_p=float(generation["top_p"]),
                max_output_tokens=int(generation["max_output_tokens"]),
                metadata={
                    "data_role": DATA_ROLE,
                    "method": method,
                    "policy_step": 0,
                    "pool": "pi0_control",
                    "sample_index": index,
                    "no_gradient": True,
                },
            )
        )
        return {
            "schema_version": 1,
            "data_role": DATA_ROLE,
            "domain": "medicine",
            "method": method,
            "global_step": 0,
            "checkpoint_id": "0",
            "policy_checkpoint": "0",
            "evaluator_checkpoint": None,
            "fresh_or_stale": None,
            "prompt_id": str(prompt["prompt_id"]),
            "response_id": _response_id(method, 0, str(prompt["prompt_id"]), "pi0_control", index),
            "pool": "pi0_control",
            "sample_index": index,
            "text": result.text,
            "seed": seed,
            "used_for_gradient": False,
            "generation": {
                "requested_model": result.requested_model,
                "returned_model": result.returned_model,
                "request_id": result.request_id,
                "raw_response_hash": result.raw_response_hash,
            },
        }

    jobs = [(prompt, index) for prompt in prompts for index in range(CONTROL_COUNT)]
    with ThreadPoolExecutor(max_workers=int(generation["workers"])) as executor:
        rows = list(executor.map(one, jobs))
    _require(
        all(
            row["generation"]["requested_model"] == served_model
            and row["generation"]["returned_model"] == served_model
            for row in rows
        ),
        "control-policy runtime model identity mismatch",
    )
    _validate_pool_rows(rows, prompts, method, 0, "pi0_control", CONTROL_COUNT)
    write_jsonl_atomic(target, rows)
    _seal(seal_path, identity, [target])
    return target


def validate_pools(config: Mapping[str, Any]) -> dict[str, Any]:
    root = prepare(config)
    prompts = read_jsonl(root / "manifests" / "validation_prompts.jsonl")
    completed = []
    for method in METHODS:
        control_path = root / "responses" / method / "pi0_control.jsonl"
        if control_path.is_file():
            _validate_pool_rows(
                read_jsonl(control_path), prompts, method, 0, "pi0_control", CONTROL_COUNT
            )
        for step in EXPECTED_STEPS_BY_METHOD[method]:
            target = root / "responses" / method / f"step-{step:03d}"
            if not (target / "probe_A.jsonl").is_file() or not (target / "probe_B.jsonl").is_file():
                continue
            a, b = read_jsonl(target / "probe_A.jsonl"), read_jsonl(target / "probe_B.jsonl")
            _validate_pool_rows(a, prompts, method, step, "probe_A", POOL_A_COUNT)
            _validate_pool_rows(b, prompts, method, step, "probe_B", POOL_B_COUNT)
            _require(
                not ({row["response_id"] for row in a} & {row["response_id"] for row in b}),
                "Pool A/B overlap",
            )
            completed.append({"method": method, "policy_step": step})
    report = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "completed_policy_pools": completed,
        "complete": len(completed) == sum(map(len, EXPECTED_STEPS_BY_METHOD.values())),
        "gradient_updates": 0,
        "optimizer_artifacts": 0,
    }
    write_json_atomic(root / "pool_validation.json", report, immutable=False)
    return report


def _parse_object(text: str, label: str) -> dict[str, Any]:
    candidate = text.strip()
    if not candidate.startswith("{") and "\n" in candidate:
        candidate = candidate.split("\n", 1)[1].rsplit("\n", 1)[0].strip()
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError as exc:
        raise HeldoutAuditError(f"{label} returned invalid JSON") from exc
    _require(isinstance(value, dict), f"{label} must return a JSON object")
    return value


def _r0_criteria(prompt: Mapping[str, Any]) -> list[dict[str, Any]]:
    criteria = []
    for index, item in enumerate(prompt["r0"]["criteria"]):
        weight = item.get("weight_units")
        _require(isinstance(weight, int) and weight > 0, "R0 criterion weight must be positive")
        criteria.append(
            {
                "criterion_id": str(
                    item.get("criterion_id") or f"{prompt['prompt_id']}:r0:{index}"
                ),
                "criterion": str(item["criterion"]),
                "weight": weight,
                "weight_units": weight,
                "source": "r0",
            }
        )
    _require(bool(criteria), "R0 criteria cannot be empty")
    return criteria


def _verify_served_model(base_url: str, expected_model: str) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + "/models", timeout=20) as response:
        payload = json.load(response)
    records = payload.get("data", [])
    ids = {str(item.get("id", "")) for item in records if isinstance(item, Mapping)}
    _require(
        expected_model in ids,
        f"served-model mismatch: expected {expected_model}, got {sorted(ids)}",
    )
    return {"base_url": base_url, "served_model": expected_model, "observed_models": sorted(ids)}


def _verify_judge_endpoints(spec: Mapping[str, Any], expected_model: str) -> list[dict[str, Any]]:
    configured_urls = tuple(str(url) for url in spec["base_urls"])
    _verify_runtime_judge_inventory(configured_urls)
    observations = []
    for base_url in configured_urls:
        observation = _verify_served_model(str(base_url), expected_model)
        observations.append(
            {
                **observation,
                "base_url": str(base_url),
                "configured_model": spec["model"],
                "revision": spec["revision"],
            }
        )
    _require(
        tuple(item["base_url"] for item in observations) == configured_urls,
        "judge endpoint verification inventory mismatch",
    )
    return observations


def _verify_runtime_judge_inventory(base_urls: Sequence[str]) -> tuple[str, ...]:
    urls = tuple(str(url).rstrip("/") for url in base_urls)
    _require(urls, "at least one judge endpoint is required")
    _require(len(urls) == len(set(urls)), "judge endpoints must be unique")
    _require(set(urls) <= APPROVED_RUNTIME_JUDGE_BASE_URLS, "unapproved runtime judge endpoint")
    return urls


def _runtime_judge_spec(
    config: Mapping[str, Any],
    runtime_base_urls: Sequence[str] | None = None,
    runtime_workers: int | None = None,
) -> dict[str, Any]:
    spec = dict(config["grading"])
    if runtime_base_urls:
        spec["base_urls"] = list(_verify_runtime_judge_inventory(runtime_base_urls))
    else:
        _verify_runtime_judge_inventory(spec["base_urls"])
    if runtime_workers is not None:
        _require(runtime_workers > 0, "runtime judge workers must be positive")
        spec["workers"] = int(runtime_workers)
        spec["max_in_flight"] = int(runtime_workers)
    return spec


def _seal(path: Path, identity: Mapping[str, Any], artifacts: Sequence[Path]) -> None:
    payload = {
        "schema_version": 1,
        "identity": dict(identity),
        "identity_sha256": _digest(identity),
        "artifacts": {
            item.name: {
                "path": str(item),
                "sha256": sha256_file(item),
                "bytes": item.stat().st_size,
            }
            for item in artifacts
        },
        "state": "sealed",
    }
    write_json_atomic(path, payload)


def _validate_seal(path: Path, identity: Mapping[str, Any]) -> None:
    _require(path.is_file(), f"missing sealed manifest: {path}")
    seal = json.loads(path.read_text(encoding="utf-8"))
    _require(seal.get("state") == "sealed", "artifact stage is not sealed")
    _require(seal.get("identity_sha256") == _digest(identity), "sealed identity mismatch")
    _require(seal.get("identity") == dict(identity), "sealed identity payload mismatch")
    for record in seal.get("artifacts", {}).values():
        artifact = Path(record["path"])
        _require(artifact.is_file(), f"sealed artifact missing: {artifact}")
        _require(
            artifact.stat().st_size == record["bytes"], f"sealed artifact size mismatch: {artifact}"
        )
        _require(
            sha256_file(artifact) == record["sha256"], f"sealed artifact hash mismatch: {artifact}"
        )


def _validate_existing_seal(path: Path) -> None:
    _require(path.is_file(), f"missing sealed manifest: {path}")
    seal = json.loads(path.read_text(encoding="utf-8"))
    identity = seal.get("identity")
    _require(isinstance(identity, Mapping), f"sealed identity missing: {path}")
    _validate_seal(path, identity)


def build_rubrics(
    config: Mapping[str, Any],
    method: str,
    step: int,
    served_model: str,
    *,
    runtime_base_url: str | None = None,
) -> Path:
    """Run 8 blind pair extractions and a separate semantic dedup request per prompt."""
    _require(method in METHODS and step in EXPECTED_STEPS_BY_METHOD[method], "invalid method/step")
    _require(
        served_model == config["rubric_generation"]["model"],
        "extractor served-model must exactly match configured model",
    )
    root = prepare(config)
    prompts = read_jsonl(root / "manifests" / "validation_prompts.jsonl")
    by_prompt = {str(row["prompt_id"]): row for row in prompts}
    stage = root / "rubrics" / method / f"step-{step:03d}"
    output = stage / "rubric_unions.jsonl"
    pool_a_path = root / "responses" / method / f"step-{step:03d}" / "probe_A.jsonl"
    control_path = root / "responses" / method / "pi0_control.jsonl"
    source_identity = {}
    if step > 0:
        _require(
            pool_a_path.is_file() and control_path.is_file(),
            "Pool A and method-specific pi0 controls are required",
        )
        pool_seal = pool_a_path.parent / "sealed_manifest.json"
        control_seal = control_path.with_suffix(".seal.json")
        _validate_existing_seal(pool_seal)
        _validate_existing_seal(control_seal)
        source_identity = {
            "pool_a_sha256": sha256_file(pool_a_path),
            "pool_a_seal_sha256": sha256_file(pool_seal),
            "control_sha256": sha256_file(control_path),
            "control_seal_sha256": sha256_file(control_seal),
        }
    identity = {
        "data_role": DATA_ROLE,
        "method": method,
        "evaluator_step": step,
        "prompt_manifest_sha256": sha256_file(root / "manifests" / "validation_prompts.jsonl"),
        "extractor_model": served_model,
        "configured_extractor_model": config["rubric_generation"]["model"],
        "extractor_revision": config["rubric_generation"]["revision"],
        "construction": "8_blind_pair_extractions_then_semantic_dedup_step_local",
        "endpoint_verification_required": step > 0,
        **source_identity,
    }
    seal_path = stage / "sealed_manifest.json"
    if seal_path.is_file():
        _validate_seal(seal_path, identity)
        return output
    if step == 0:
        rows = [
            {
                "schema_version": 1,
                "data_role": DATA_ROLE,
                "domain": "medicine",
                "method": method,
                "prompt_id": prompt_id,
                "evaluator_step": 0,
                "evaluator_checkpoint": "0",
                "fresh_or_stale": "fresh",
                "construction": "initial_rubric_only_not_ground_truth",
                "criteria": _r0_criteria(prompt),
                "pool": "probe_A",
                "used_for_gradient": False,
            }
            for prompt_id, prompt in sorted(by_prompt.items())
        ]
        write_jsonl_atomic(output, rows)
        _seal(seal_path, identity, [output])
        return output

    pool_a, controls = read_jsonl(pool_a_path), read_jsonl(control_path)
    _validate_pool_rows(pool_a, prompts, method, step, "probe_A", POOL_A_COUNT)
    _validate_pool_rows(controls, prompts, method, 0, "pi0_control", CONTROL_COUNT)
    a_by_prompt = {
        pid: sorted(
            (row for row in pool_a if row["prompt_id"] == pid), key=lambda row: row["sample_index"]
        )
        for pid in by_prompt
    }
    c_by_prompt = {
        pid: sorted(
            (row for row in controls if row["prompt_id"] == pid),
            key=lambda row: row["sample_index"],
        )
        for pid in by_prompt
    }
    spec = config["rubric_generation"]
    effective_base_url = runtime_base_url or str(spec["base_url"])
    endpoint_observation = _verify_served_model(effective_base_url, served_model)
    runtime_spec = {**spec, "base_url": effective_base_url}
    adapter = _adapter(
        runtime_spec,
        root / "cache" / "rubrics" / method / f"step-{step:03d}",
        served_model,
    )

    requests: list[GenerationRequest] = []
    pairing_rows = []
    prompt_request_ranges: dict[str, tuple[int, int]] = {}
    for prompt_id in sorted(by_prompt):
        prompt = by_prompt[prompt_id]
        occurrence_id = f"{DATA_ROLE}:{method}:{step}:{prompt_id}"
        pairing = make_blind_pairing(
            [row["text"] for row in a_by_prompt[prompt_id]],
            [row["text"] for row in c_by_prompt[prompt_id]],
            seed=int(config["seed"]),
            prompt_id=occurrence_id,
            step=step,
        )
        begin = len(requests)
        for pair_index, (blind, assignment) in enumerate(
            zip(pairing.generator_payload(), pairing.assignments)
        ):
            request = GenerationRequest(
                prompt_id=prompt_id,
                messages=build_onlinerubric_extractor_messages(
                    prompt=prompt["messages"],
                    existing_rubric=_r0_criteria(prompt),
                    response_a=blind.response_a,
                    response_b=blind.response_b,
                ),
                family="online_rubric_extraction",
                seed=int(config["seed"]) + pair_index,
                max_output_tokens=int(spec.get("extractor_max_output_tokens", 8192)),
                json_schema=EXTRACTION_SCHEMA,
                schema_name="onlinerubric_extraction_v1",
                metadata={
                    "data_role": DATA_ROLE,
                    "method": method,
                    "evaluator_step": step,
                    "prompt_occurrence_id": occurrence_id,
                    "pair_id": blind.pair_id,
                },
            )
            requests.append(request)
            pairing_rows.append(
                {
                    "data_role": DATA_ROLE,
                    "method": method,
                    "evaluator_step": step,
                    "prompt_id": prompt_id,
                    "prompt_occurrence_id": occurrence_id,
                    "prompt_messages": prompt["messages"],
                    "r0": _r0_criteria(prompt),
                    "blind_pair": {
                        "pair_id": blind.pair_id,
                        "response_a": blind.response_a,
                        "response_b": blind.response_b,
                    },
                    "assignment": {
                        "pair_id": assignment.pair_id,
                        "current_label": assignment.current_label,
                        "control_label": assignment.control_label,
                        "current_index": assignment.current_index,
                        "control_index": assignment.control_index,
                    },
                }
            )
        prompt_request_ranges[prompt_id] = (begin, len(requests))
    with ThreadPoolExecutor(max_workers=int(spec.get("workers", 16))) as executor:
        extraction_results = list(executor.map(adapter.generate, requests))
    _require(
        all(
            result.requested_model == served_model and result.returned_model == served_model
            for result in extraction_results
        ),
        "extractor runtime model identity mismatch",
    )
    parsed = [_parse_extraction(result) for result in extraction_results]
    extraction_receipts = [
        _receipt(result, request) for request, result in zip(requests, extraction_results)
    ]

    candidate_rows = []
    dedup_requests = []
    offline_by_prompt: dict[str, tuple[WeightedCriterion, ...]] = {}
    candidates_by_prompt: dict[str, tuple[Mapping[str, Any], ...]] = {}
    for prompt_id in sorted(by_prompt):
        begin, finish = prompt_request_ranges[prompt_id]
        candidates = tuple(candidate for result in parsed[begin:finish] for candidate in result)
        candidates_by_prompt[prompt_id] = candidates
        occurrence_id = f"{DATA_ROLE}:{method}:{step}:{prompt_id}"
        offline = tuple(
            WeightedCriterion(item["criterion_id"], item["criterion"], item["weight"], "r0")
            for item in _r0_criteria(by_prompt[prompt_id])
        )
        offline_by_prompt[prompt_id] = offline
        candidate_rows.append(
            {
                "data_role": DATA_ROLE,
                "method": method,
                "evaluator_step": step,
                "prompt_id": prompt_id,
                "prompt_messages": by_prompt[prompt_id]["messages"],
                "r0": _r0_criteria(by_prompt[prompt_id]),
                "candidates": list(candidates),
                "candidate_sha256": _digest(candidates),
            }
        )
        dedup_requests.append(
            GenerationRequest(
                prompt_id=prompt_id,
                messages=build_onlinerubric_dedup_messages(
                    prompt=by_prompt[prompt_id]["messages"],
                    existing_rubric=_r0_criteria(by_prompt[prompt_id]),
                    candidate_criteria=candidates,
                    exclude_existing=True,
                ),
                family="online_rubric_dedup",
                seed=int(config["seed"]),
                max_output_tokens=int(spec.get("dedup_max_output_tokens", 8192)),
                json_schema=DEDUP_SCHEMA,
                schema_name="onlinerubric_dedup_v1",
                metadata={
                    "data_role": DATA_ROLE,
                    "method": method,
                    "evaluator_step": step,
                    "prompt_occurrence_id": occurrence_id,
                    "semantic_dedup": True,
                },
            )
        )
    with ThreadPoolExecutor(max_workers=int(spec.get("workers", 16))) as executor:
        dedup_results = list(executor.map(adapter.generate, dedup_requests))
    _require(
        all(
            result.requested_model == served_model and result.returned_model == served_model
            for result in dedup_results
        ),
        "dedup runtime model identity mismatch",
    )
    rows = []
    for prompt_id, result in zip(sorted(by_prompt), dedup_results):
        occurrence_id = f"{DATA_ROLE}:{method}:{step}:{prompt_id}"
        online = _parse_dedup(
            result,
            occurrence_id=occurrence_id,
            offline=offline_by_prompt[prompt_id],
            candidates=candidates_by_prompt[prompt_id],
        )
        new = [
            {
                "criterion_id": item.criterion_id,
                "criterion": item.text,
                "weight": item.weight,
                "weight_units": item.weight,
                "source": item.source,
            }
            for item in online
        ]
        rows.append(
            {
                "schema_version": 1,
                "data_role": DATA_ROLE,
                "domain": "medicine",
                "method": method,
                "prompt_id": prompt_id,
                "evaluator_step": step,
                "evaluator_checkpoint": str(step),
                "fresh_or_stale": "fresh",
                "construction": "step_local_r0_union_semantically_deduplicated_new_criteria",
                "criteria_before_deduplication": list(candidates_by_prompt[prompt_id]),
                "criteria_after_deduplication": new,
                "criteria": _r0_criteria(by_prompt[prompt_id]) + new,
                "pool": "probe_A",
                "used_for_gradient": False,
            }
        )
    blind_path = stage / "blind_pairs.jsonl"
    extraction_path = stage / "extraction_receipts.jsonl"
    candidates_path = stage / "candidate_receipts.jsonl"
    dedup_path = stage / "dedup_receipts.jsonl"
    endpoint_path = stage / "extractor_endpoint_receipt.json"
    write_jsonl_atomic(blind_path, pairing_rows)
    write_jsonl_atomic(extraction_path, extraction_receipts)
    write_jsonl_atomic(candidates_path, candidate_rows)
    write_jsonl_atomic(
        dedup_path,
        [_receipt(result, request) for request, result in zip(dedup_requests, dedup_results)],
    )
    write_jsonl_atomic(output, rows)
    write_json_atomic(
        endpoint_path,
        {
            **endpoint_observation,
            "configured_model": spec["model"],
            "revision": spec["revision"],
            "configured_base_url": spec["base_url"],
            "runtime_base_url": effective_base_url,
        },
    )
    _seal(
        seal_path,
        identity,
        [blind_path, extraction_path, candidates_path, dedup_path, output, endpoint_path],
    )
    return output


def score_cell(
    config: Mapping[str, Any],
    method: str,
    policy_step: int,
    evaluator_step: int,
    served_model: str,
    *,
    runtime_base_urls: Sequence[str] | None = None,
    runtime_workers: int | None = None,
) -> Path:
    """Score one lower-triangle cell using the canonical full-rubric Qwen grader."""
    steps = EXPECTED_STEPS_BY_METHOD.get(method, ())
    _require(
        served_model == config["grading"]["model"],
        "judge served-model must exactly match configured model",
    )
    _require(
        policy_step in steps and evaluator_step in steps and evaluator_step <= policy_step,
        "invalid triangle cell",
    )
    root = prepare(config)
    pool_path = root / "responses" / method / f"step-{policy_step:03d}" / "probe_B.jsonl"
    rubric_path = root / "rubrics" / method / f"step-{evaluator_step:03d}" / "rubric_unions.jsonl"
    _require(
        pool_path.is_file() and rubric_path.is_file(),
        "Pool B and sealed evaluator rubric are required",
    )
    rubric_seal = rubric_path.parent / "sealed_manifest.json"
    _validate_existing_seal(pool_path.parent / "sealed_manifest.json")
    _validate_existing_seal(rubric_seal)
    pool_b, rubric_rows = read_jsonl(pool_path), read_jsonl(rubric_path)
    prompts = read_jsonl(root / "manifests" / "validation_prompts.jsonl")
    _validate_pool_rows(pool_b, prompts, method, policy_step, "probe_B", POOL_B_COUNT)
    prompt_by_id = {str(row["prompt_id"]): row for row in prompts}
    rubric_by_prompt = {str(row["prompt_id"]): row["criteria"] for row in rubric_rows}
    _require(set(rubric_by_prompt) == set(prompt_by_id), "rubric prompt inventory mismatch")
    enriched = [
        {**row, "prompt_messages": prompt_by_id[str(row["prompt_id"])]["messages"]}
        for row in pool_b
    ]
    target = (
        root
        / "grades"
        / "cells"
        / method
        / f"policy-{policy_step:03d}"
        / f"evaluator-{evaluator_step:03d}.jsonl"
    )
    seal_path = target.with_suffix(".seal.json")
    spec = _runtime_judge_spec(config, runtime_base_urls, runtime_workers)
    identity = {
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": policy_step,
        "evaluator_step": evaluator_step,
        "pool_b_sha256": sha256_file(pool_path),
        "rubric_sha256": sha256_file(rubric_path),
        "judge_model": served_model,
        "configured_judge_model": config["grading"]["model"],
        "judge_revision": config["grading"]["revision"],
        "judge_base_urls": list(spec["base_urls"]),
        "endpoint_verification_required": True,
    }
    if seal_path.is_file():
        _validate_existing_seal(seal_path)
        sealed_identity = json.loads(seal_path.read_text(encoding="utf-8"))["identity"]
        for key, value in identity.items():
            if key != "judge_base_urls":
                _require(sealed_identity.get(key) == value, f"sealed score identity mismatch: {key}")
        return target
    endpoint_observations = _verify_judge_endpoints(spec, served_model)
    adapter = _adapter(
        {**spec, "base_url": spec["base_urls"]}, root / "cache" / "grading" / method, served_model
    )
    records = score_pool(
        enriched,
        rubric_by_prompt,
        evaluator_checkpoint=str(evaluator_step),
        policy_checkpoint=str(policy_step),
        pool="probe_B",
        config=AuditScoreConfig(
            domain="medicine",
            method=method,
            seed=int(config["seed"]),
            judge_model=served_model,
            judge_revision=spec["revision"],
            max_output_tokens=int(spec.get("max_output_tokens", 4096)),
            concurrency=int(spec.get("workers", 32)),
        ),
        grader=adapter,
        cache_dir=root / "cache" / "score_receipts" / method,
    )
    _require(
        all(
            row["judge"]["requested_model"] == served_model
            and row["judge"]["returned_model"] == served_model
            for row in records
        ),
        "judge runtime model identity mismatch",
    )
    rows = [{**row, "data_role": DATA_ROLE, "used_for_gradient": False} for row in records]
    endpoint_path = target.with_suffix(".endpoint.json")
    write_json_atomic(
        endpoint_path,
        {
            "base_urls": list(spec["base_urls"]),
            "endpoint_model_identity": endpoint_observations,
        },
    )
    write_jsonl_atomic(target, rows)
    _seal(seal_path, identity, [target, endpoint_path])
    return target


def score_all_cells(
    config: Mapping[str, Any],
    served_model: str,
    *,
    runtime_base_urls: Sequence[str] | None = None,
    runtime_workers: int | None = None,
) -> dict[str, Any]:
    """Resume every configured lower-triangle cell against one verified judge."""
    _require(
        served_model == config["grading"]["model"],
        "judge served-model must exactly match configured model",
    )
    runtime_spec = _runtime_judge_spec(config, runtime_base_urls, runtime_workers)
    _verify_judge_endpoints(runtime_spec, served_model)
    score_kwargs = {}
    if runtime_base_urls is not None:
        score_kwargs["runtime_base_urls"] = runtime_base_urls
    if runtime_workers is not None:
        score_kwargs["runtime_workers"] = runtime_workers

    completed = []
    for method in METHODS:
        steps = tuple(int(step) for step in config["checkpoint_steps"][method])
        for policy_index, policy_step in enumerate(steps):
            for evaluator_step in steps[: policy_index + 1]:
                path = score_cell(
                    config,
                    method,
                    policy_step,
                    evaluator_step,
                    served_model,
                    **score_kwargs,
                )
                completed.append(
                    {
                        "method": method,
                        "policy_step": policy_step,
                        "evaluator_step": evaluator_step,
                        "artifact": str(path),
                    }
                )
    expected = {
        method: len(config["checkpoint_steps"][method])
        * (len(config["checkpoint_steps"][method]) + 1)
        // 2
        for method in METHODS
    }
    _require(len(completed) == sum(expected.values()), "all-cell schedule count drift")
    report = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "judge_model": served_model,
        "matrix_cells_per_method": expected,
        "completed_cells": len(completed),
        "training_updates": 0,
        "optimizer_loaded": False,
    }
    write_json_atomic(
        output_root(config) / "grades" / "all_cells_complete.json",
        report,
        immutable=False,
    )
    return report


def metric_values(rewards: Sequence[float], tie_epsilon: float) -> dict[str, float]:
    _require(bool(rewards), "reward group is empty")
    mean = sum(rewards) / len(rewards)
    pairs = list(itertools.combinations(rewards, 2))
    ptr = sum(abs(a - b) <= tie_epsilon for a, b in pairs) / len(pairs)
    return {
        "mad": sum(abs(value - mean) for value in rewards) / len(rewards),
        "zar": float(max(rewards) == min(rewards)),
        "ptr": ptr,
        "separation_rate": 1.0 - ptr,
        "reward_std": statistics.pstdev(rewards),
    }


def gain_values(stale: Mapping[str, float], current: Mapping[str, float]) -> dict[str, float]:
    """Positive always means the current rubric discriminates the group better."""
    return {
        "mad_gain": current["mad"] - stale["mad"],
        "zar_gain": stale["zar"] - current["zar"],
        "ptr_gain": stale["ptr"] - current["ptr"],
    }


def integration_smoke(
    config: Mapping[str, Any],
    method: str,
    step: int,
    component: str,
    served_model: str,
) -> dict[str, Any]:
    """Run one resumable real-endpoint smoke component in an isolated namespace."""
    _require(
        step in EXPECTED_STEPS_BY_METHOD[method]
        and EXPECTED_STEPS_BY_METHOD[method].index(step) >= 2,
        "smoke requires a checkpoint with distinct R0, previous, and current evaluators",
    )
    _require(
        component in {"policy-base", "policy-previous", "policy-current", "rubric", "score"},
        "invalid integration smoke component",
    )
    configured_runtime: dict[str, Any] = {}
    if component == "rubric":
        configured_runtime = {
            "configured_model": str(config["rubric_generation"]["model"]),
            "configured_revision": str(config["rubric_generation"]["revision"]),
        }
        _require(
            served_model == configured_runtime["configured_model"],
            "extractor served-model must exactly match configured model",
        )
    elif component == "score":
        configured_runtime = {
            "configured_model": str(config["grading"]["model"]),
            "configured_revision": str(config["grading"]["revision"]),
            "configured_base_urls": list(config["grading"]["base_urls"]),
        }
        _require(
            served_model == configured_runtime["configured_model"],
            "judge served-model must exactly match configured model",
        )
    steps = EXPECTED_STEPS_BY_METHOD[method]
    previous_step = steps[steps.index(step) - 1]
    smoke = json.loads(json.dumps(config))
    smoke["validation"]["count"] = 1
    smoke["output_root"] = str(output_root(config) / "integration_smoke")
    smoke_root = prepare(smoke)
    receipt_root = smoke_root / "_smoke_receipts" / method / f"step-{step:03d}"
    receipt_path = receipt_root / f"{component}.json"
    receipt_seal = receipt_path.with_suffix(".seal.json")
    prerequisite_components = {
        "policy-base": (),
        "policy-previous": ("policy-base",),
        "policy-current": ("policy-base", "policy-previous"),
        "rubric": ("policy-base", "policy-previous", "policy-current"),
        "score": ("policy-base", "policy-previous", "policy-current", "rubric"),
    }[component]
    prerequisite_seal_sha256 = {}
    for prerequisite in prerequisite_components:
        prior = receipt_root / f"{prerequisite}.json"
        prior_seal = prior.with_suffix(".seal.json")
        _validate_existing_seal(prior_seal)
        prior_report = json.loads(prior.read_text(encoding="utf-8"))
        _require(
            prior_report.get("method") == method
            and prior_report.get("policy_step") == step
            and prior_report.get("component") == prerequisite,
            f"integration smoke prerequisite identity mismatch: {prerequisite}",
        )
        prerequisite_seal_sha256[prerequisite] = sha256_file(prior_seal)
    receipt_identity = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": step,
        "component": component,
        "served_model": served_model,
        "smoke_manifest_sha256": sha256_file(smoke_root / "manifest.json"),
        "prerequisite_seal_sha256": prerequisite_seal_sha256,
        **configured_runtime,
    }
    if receipt_seal.is_file():
        _validate_seal(receipt_seal, receipt_identity)
        return json.loads(receipt_path.read_text(encoding="utf-8"))
    endpoint_model_identity: list[dict[str, Any]] = []
    signed_gains: dict[str, dict[str, float]] | None = None
    if component == "policy-base":
        artifact = generate_control(smoke, method, served_model)
    elif component == "policy-previous":
        artifact = generate(smoke, method, previous_step, served_model)
    elif component == "policy-current":
        artifact = generate(smoke, method, step, served_model)
    elif component == "rubric":
        build_rubrics(smoke, method, 0, served_model)
        build_rubrics(smoke, method, previous_step, served_model)
        artifact = build_rubrics(smoke, method, step, served_model)
        for evaluator_step in (previous_step, step):
            endpoint_path = (
                smoke_root
                / "rubrics"
                / method
                / f"step-{evaluator_step:03d}"
                / "extractor_endpoint_receipt.json"
            )
            endpoint = json.loads(endpoint_path.read_text(encoding="utf-8"))
            _require(
                endpoint.get("served_model") == served_model
                and endpoint.get("configured_model") == configured_runtime["configured_model"]
                and endpoint.get("revision") == configured_runtime["configured_revision"],
                "smoke extractor configured/observed identity mismatch",
            )
            endpoint_model_identity.append({"evaluator_step": evaluator_step, **endpoint})
    else:
        score_paths = {
            evaluator_step: score_cell(smoke, method, step, evaluator_step, served_model)
            for evaluator_step in (0, previous_step, step)
        }
        identities = []
        response_ids = []
        evaluator_metrics = {}
        for evaluator_step, path in score_paths.items():
            seal = json.loads(path.with_suffix(".seal.json").read_text(encoding="utf-8"))
            identities.append(seal["identity"])
            score_rows = read_jsonl(path)
            response_ids.append({str(row["response_id"]) for row in score_rows})
            evaluator_metrics[evaluator_step] = metric_values(
                [float(row["reward"]) for row in score_rows],
                float(config["metrics"]["pairwise_tie_epsilon"]),
            )
            _require(
                int(seal["identity"]["evaluator_step"]) == evaluator_step,
                "smoke score evaluator identity mismatch",
            )
            endpoint = json.loads(path.with_suffix(".endpoint.json").read_text(encoding="utf-8"))
            replica_identities = endpoint.get("endpoint_model_identity")
            _require(
                endpoint.get("base_urls") == configured_runtime["configured_base_urls"]
                and isinstance(replica_identities, list)
                and len(replica_identities) == len(APPROVED_JUDGE_BASE_URLS)
                and all(
                    replica.get("served_model") == served_model
                    and replica.get("configured_model") == configured_runtime["configured_model"]
                    and replica.get("revision") == configured_runtime["configured_revision"]
                    for replica in replica_identities
                ),
                "smoke judge configured/observed identity mismatch",
            )
            endpoint_model_identity.extend(
                {"evaluator_step": evaluator_step, **replica} for replica in replica_identities
            )
        _require(
            response_ids[0] == response_ids[1] == response_ids[2],
            "smoke evaluators did not score identical response IDs",
        )
        pool_hashes = {str(identity["pool_b_sha256"]) for identity in identities}
        _require(len(pool_hashes) == 1, "smoke evaluators did not score identical response bytes")
        signed_gains = {
            "R0_to_Rt": gain_values(evaluator_metrics[0], evaluator_metrics[step]),
            "Rprev_to_Rt": gain_values(evaluator_metrics[previous_step], evaluator_metrics[step]),
        }
        artifact = score_paths[step]
    report = {
        "schema_version": 1,
        "state": "passed" if component == "score" else "component_sealed",
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": step,
        "component": component,
        "served_model": served_model,
        "prompt_count": 1,
        "artifact": str(artifact),
        "previous_evaluator_step": previous_step,
        "evaluator_steps_compared": [0, previous_step, step] if component == "score" else None,
        "identical_response_ids_and_pool_hash": component == "score",
        "signed_gains_positive_means_current_better": signed_gains,
        "endpoint_model_identity": endpoint_model_identity,
        "prerequisite_seal_sha256": prerequisite_seal_sha256,
        **configured_runtime,
        "prerequisite_components": list(prerequisite_components),
        "full_chain_verified": component == "score",
        "training_updates": 0,
        "optimizer_loaded": False,
    }
    write_json_atomic(receipt_path, report)
    _seal(receipt_seal, receipt_identity, [receipt_path])
    return report


def _lifecycle_download_root(config: Mapping[str, Any]) -> Path:
    """Validate the pre-created dedicated download root and its ownership sentinel."""
    runtime = config["policy_lifecycle"]
    download_root = Path(config["policy_lifecycle"]["temporary_download_root"]).resolve()
    _require(
        download_root not in {Path("/"), Path.home().resolve(), output_root(config).resolve()},
        "unsafe lifecycle temporary_download_root",
    )
    _require(
        download_root.is_dir() and not download_root.is_symlink(),
        "lifecycle temporary root must be a preexisting real directory",
    )
    sentinel_name = str(runtime["temporary_root_sentinel"])
    _require(Path(sentinel_name).name == sentinel_name, "invalid lifecycle root sentinel name")
    sentinel = download_root / sentinel_name
    _require(sentinel.is_file() and not sentinel.is_symlink(), "lifecycle root sentinel missing")
    payload = json.loads(sentinel.read_text(encoding="utf-8"))
    _require(
        payload
        == {
            "schema_version": 1,
            "data_role": DATA_ROLE,
            "root_identity": runtime["temporary_root_identity"],
        },
        "lifecycle root sentinel identity mismatch",
    )
    return download_root


def _validate_owned_checkpoint_dir(
    download_root: Path, local_dir: Path, identity_sha256: str
) -> Path:
    """Accept only a unique directory already sealed as owned by this lifecycle."""
    _require(not local_dir.is_symlink(), "temporary checkpoint path must not be a symlink")
    resolved = local_dir.resolve()
    _require(
        resolved.is_dir() and resolved.parent == download_root,
        "temporary checkpoint path escaped its configured root",
    )
    marker = resolved / ".heldout_validation_checkpoint_owner.json"
    _require(marker.is_file() and not marker.is_symlink(), "checkpoint ownership sentinel missing")
    _require(
        json.loads(marker.read_text(encoding="utf-8"))
        == {
            "schema_version": 1,
            "data_role": DATA_ROLE,
            "identity_sha256": identity_sha256,
        },
        "checkpoint ownership sentinel mismatch",
    )
    return resolved


def _acquire_checkpoint_dir(
    config: Mapping[str, Any],
    method: str,
    step: int,
    identity_sha256: str,
    resumed_path: str | None = None,
) -> Path:
    download_root = _lifecycle_download_root(config)
    if resumed_path is not None:
        return _validate_owned_checkpoint_dir(download_root, Path(resumed_path), identity_sha256)
    local_dir = Path(tempfile.mkdtemp(prefix=f"{method}-step-{step:03d}-", dir=download_root))
    marker = local_dir / ".heldout_validation_checkpoint_owner.json"
    write_json_atomic(
        marker,
        {
            "schema_version": 1,
            "data_role": DATA_ROLE,
            "identity_sha256": identity_sha256,
        },
    )
    return _validate_owned_checkpoint_dir(download_root, local_dir, identity_sha256)


def policy_lifecycle(config: Mapping[str, Any], method: str, step: int) -> dict[str, Any]:
    """Download/start/health/generate/seal/stop/evict one pinned HF checkpoint."""
    _require(step in EXPECTED_STEPS_BY_METHOD[method], "invalid lifecycle checkpoint")
    root = prepare(config)
    plan = next(
        row
        for row in read_jsonl(root / "manifests" / "checkpoint_work_plan.jsonl")
        if row["method"] == method and row["policy_step"] == step
    )
    runtime = config["policy_lifecycle"]
    state_path = root / "lifecycle" / method / f"step-{step:03d}.json"
    served_model = f"heldout-{method}-{step:03d}"
    lifecycle_identity = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": step,
        "hf_repo_id": plan["hf_repo_id"],
        "hf_revision": plan["hf_revision_requested"],
        "served_model": served_model,
        "policy_endpoint": config["policy_generation"]["base_url"],
        "validation_manifest_sha256": sha256_file(root / "manifests" / "validation_prompts.jsonl"),
        "runtime": {
            key: runtime[key]
            for key in (
                "gpu",
                "vllm",
                "port",
                "gpu_memory_utilization",
                "max_model_len",
            )
        },
    }
    identity_sha256 = _digest(lifecycle_identity)
    resume_state: Mapping[str, Any] | None = None
    if state_path.is_file():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        _require(
            state.get("identity_sha256") == identity_sha256,
            f"lifecycle resume identity drift: {state_path}",
        )
        if state.get("state") == "complete":
            response_seal = (
                root / "responses" / method / f"step-{step:03d}" / "sealed_manifest.json"
            )
            _validate_existing_seal(response_seal)
            _require(
                state.get("response_seal_sha256") == sha256_file(response_seal),
                "complete lifecycle response seal hash drift",
            )
            if step == 0:
                control_seal = root / "responses" / method / "pi0_control.seal.json"
                _validate_existing_seal(control_seal)
                _require(
                    state.get("control_seal_sha256") == sha256_file(control_seal),
                    "complete lifecycle control seal hash drift",
                )
            return state
        resume_state = state
    from huggingface_hub import snapshot_download

    download_root = _lifecycle_download_root(config)
    local_dir = _acquire_checkpoint_dir(
        config,
        method,
        step,
        identity_sha256,
        str(resume_state["temporary_model_dir"])
        if resume_state and resume_state.get("temporary_model_dir")
        else None,
    )
    state = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "method": method,
        "policy_step": step,
        "hf_repo_id": plan["hf_repo_id"],
        "hf_revision": plan["hf_revision_requested"],
        "served_model": served_model,
        "identity": lifecycle_identity,
        "identity_sha256": identity_sha256,
        "temporary_model_dir": str(local_dir),
        "state": "downloading",
    }
    write_json_atomic(state_path, state, immutable=False)
    snapshot_download(
        repo_id=plan["hf_repo_id"], revision=plan["hf_revision_requested"], local_dir=local_dir
    )
    _require(local_dir.is_dir() and not local_dir.is_symlink(), "HF download directory invalid")
    state["state"] = "starting"
    write_json_atomic(state_path, state, immutable=False)
    command = [
        runtime["vllm"],
        "serve",
        str(local_dir),
        "--served-model-name",
        served_model,
        "--host",
        "127.0.0.1",
        "--port",
        str(runtime["port"]),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        str(runtime["gpu_memory_utilization"]),
        "--max-model-len",
        str(runtime["max_model_len"]),
    ]
    environment = dict(__import__("os").environ)
    environment["CUDA_VISIBLE_DEVICES"] = str(runtime["gpu"])
    process = subprocess.Popen(command, env=environment, start_new_session=True)
    try:
        deadline = time.monotonic() + float(runtime["startup_timeout_seconds"])
        while True:
            if process.poll() is not None:
                raise HeldoutAuditError(f"policy vLLM exited with code {process.returncode}")
            try:
                observation = _verify_served_model(
                    config["policy_generation"]["base_url"], served_model
                )
                break
            except Exception:
                if time.monotonic() >= deadline:
                    raise HeldoutAuditError("policy vLLM health timeout")
                time.sleep(2)
        state["state"] = "generating"
        state["endpoint_observation"] = observation
        write_json_atomic(state_path, state, immutable=False)
        generate(config, method, step, served_model)
        if step == 0:
            generate_control(config, method, served_model)
        state["state"] = "sealed"
        write_json_atomic(state_path, state, immutable=False)
    finally:
        process.terminate()
        try:
            process.wait(timeout=60)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=30)
    response_seal = root / "responses" / method / f"step-{step:03d}" / "sealed_manifest.json"
    _validate_existing_seal(response_seal)
    control_seal = root / "responses" / method / "pi0_control.seal.json"
    if step == 0:
        _validate_existing_seal(control_seal)
    _validate_owned_checkpoint_dir(download_root, local_dir, identity_sha256)
    shutil.rmtree(local_dir)
    _require(not local_dir.exists(), "verified temporary model eviction failed")
    state["state"] = "complete"
    state["temporary_model_evicted"] = True
    state["response_seal_sha256"] = sha256_file(response_seal)
    if step == 0:
        state["control_seal_sha256"] = sha256_file(control_seal)
    write_json_atomic(state_path, state, immutable=False)
    return state


def _phase_receipt(
    config: Mapping[str, Any],
    phase: str,
    details: Mapping[str, Any],
) -> dict[str, Any]:
    root = prepare(config)
    path = root / "run_all" / f"{phase}.json"
    seal_path = path.with_suffix(".seal.json")
    identity = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "phase": phase,
        "manifest_sha256": sha256_file(root / "manifest.json"),
        "details": dict(details),
    }
    if seal_path.is_file():
        _validate_seal(seal_path, identity)
        return json.loads(path.read_text(encoding="utf-8"))
    payload = {**identity, "state": "sealed"}
    write_json_atomic(path, payload)
    _seal(seal_path, identity, [path])
    return payload


def run_all(
    config: Mapping[str, Any],
    extractor_model: str,
    judge_model: str,
    stop_after_phase: str | None = None,
) -> dict[str, Any]:
    """Resume the smoke-gated policy, rubric, score, and analysis phase sequence."""
    phases = ("smoke-gate", "policies", "rubrics", "scores", "analyze")
    _require(stop_after_phase is None or stop_after_phase in phases, "invalid run-all stop phase")
    gates = config["execution"]["smoke_gates"]
    gate_records = []
    smoke_root = output_root(config) / "integration_smoke"
    for gate in gates:
        method, step = str(gate["method"]), int(gate["step"])
        receipt_root = smoke_root / "_smoke_receipts" / method / f"step-{step:03d}"
        component_reports = {}
        for component in ("policy-base", "policy-previous", "policy-current", "rubric", "score"):
            component_path = receipt_root / f"{component}.json"
            _validate_existing_seal(component_path.with_suffix(".seal.json"))
            component_reports[component] = json.loads(component_path.read_text(encoding="utf-8"))
        receipt = receipt_root / "score.json"
        _validate_existing_seal(receipt.with_suffix(".seal.json"))
        report = component_reports["score"]
        expected_prerequisite_hashes = {
            component: sha256_file((receipt_root / f"{component}.json").with_suffix(".seal.json"))
            for component in ("policy-base", "policy-previous", "policy-current", "rubric")
        }
        _require(
            report.get("state") == "passed"
            and report.get("full_chain_verified") is True
            and report.get("evaluator_steps_compared")
            == [
                0,
                EXPECTED_STEPS_BY_METHOD[method][EXPECTED_STEPS_BY_METHOD[method].index(step) - 1],
                step,
            ]
            and report.get("identical_response_ids_and_pool_hash") is True,
            f"run-all smoke gate failed: {method}/{step}",
        )
        _require(
            report.get("prerequisite_seal_sha256") == expected_prerequisite_hashes,
            f"run-all smoke prerequisite seal hash drift: {method}/{step}",
        )
        rubric_identity = component_reports["rubric"].get("endpoint_model_identity")
        _require(
            isinstance(rubric_identity, list)
            and len(rubric_identity) == 2
            and all(
                endpoint.get("served_model") == config["rubric_generation"]["model"]
                and endpoint.get("configured_model") == config["rubric_generation"]["model"]
                and endpoint.get("revision") == config["rubric_generation"]["revision"]
                for endpoint in rubric_identity
            ),
            f"run-all smoke extractor identity evidence missing: {method}/{step}",
        )
        signed_gains = report.get("signed_gains_positive_means_current_better")
        _require(
            isinstance(signed_gains, Mapping)
            and set(signed_gains) == {"R0_to_Rt", "Rprev_to_Rt"}
            and all(
                set(gains) == {"mad_gain", "zar_gain", "ptr_gain"}
                for gains in signed_gains.values()
            ),
            f"run-all smoke signed gains missing: {method}/{step}",
        )
        endpoint_identity = report.get("endpoint_model_identity")
        _require(
            isinstance(endpoint_identity, list)
            and len(endpoint_identity) == 3 * len(APPROVED_JUDGE_BASE_URLS)
            and {endpoint.get("base_url") for endpoint in endpoint_identity}
            == set(APPROVED_JUDGE_BASE_URLS)
            and all(
                endpoint.get("served_model") == config["grading"]["model"]
                and endpoint.get("configured_model") == config["grading"]["model"]
                and endpoint.get("revision") == config["grading"]["revision"]
                for endpoint in endpoint_identity
            ),
            f"run-all smoke judge identity evidence missing: {method}/{step}",
        )
        gate_records.append(
            {"method": method, "step": step, "receipt_sha256": sha256_file(receipt)}
        )
    _phase_receipt(config, "smoke-gate", {"gates": gate_records})
    if stop_after_phase == "smoke-gate":
        return {"state": "stopped_after_phase", "phase": "smoke-gate"}

    lifecycle_records = []
    for method in METHODS:
        for step in config["checkpoint_steps"][method]:
            state = policy_lifecycle(config, method, int(step))
            _require(state.get("state") == "complete", "policy lifecycle did not complete")
            lifecycle_records.append(
                {
                    "method": method,
                    "step": int(step),
                    "identity_sha256": state["identity_sha256"],
                    "response_seal_sha256": state["response_seal_sha256"],
                }
            )
    pool_report = validate_pools(config)
    _require(pool_report["complete"] is True, "all 32 policy response pools are required")
    _phase_receipt(
        config,
        "policies",
        {"checkpoint_count": len(lifecycle_records), "records": lifecycle_records},
    )
    if stop_after_phase == "policies":
        return {"state": "stopped_after_phase", "phase": "policies"}

    rubric_records = []
    for method in METHODS:
        for step in config["checkpoint_steps"][method]:
            path = build_rubrics(config, method, int(step), extractor_model)
            rubric_records.append(
                {
                    "method": method,
                    "step": int(step),
                    "artifact_sha256": sha256_file(path),
                    "seal_sha256": sha256_file(path.parent / "sealed_manifest.json"),
                }
            )
    _phase_receipt(
        config,
        "rubrics",
        {"rubric_count": len(rubric_records), "records": rubric_records},
    )
    if stop_after_phase == "rubrics":
        return {"state": "stopped_after_phase", "phase": "rubrics"}

    score_report = score_all_cells(config, judge_model)
    _phase_receipt(config, "scores", score_report)
    if stop_after_phase == "scores":
        return {"state": "stopped_after_phase", "phase": "scores"}

    analysis_report = analyze(config)
    final = _phase_receipt(config, "analyze", analysis_report)
    return {"state": "complete", "phase": "analyze", "receipt": final}


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    _require(bool(rows), f"cannot write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _verify_sealed_cell(path: Path) -> None:
    seal_path = path.with_suffix(".seal.json")
    _require(seal_path.is_file(), f"unsealed score cell: {path}")
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    _require(seal.get("state") == "sealed", f"score cell seal state invalid: {path}")
    records = seal.get("artifacts", {})
    match = next((record for record in records.values() if Path(record["path"]) == path), None)
    _require(match is not None, f"score cell is not bound by its seal: {path}")
    _require(
        path.stat().st_size == match["bytes"] and sha256_file(path) == match["sha256"],
        f"score cell differs from sealed bytes: {path}",
    )


def _write_matrix_csvs(root: Path, cells: Sequence[Mapping[str, Any]]) -> list[str]:
    paths = []
    for method in METHODS:
        steps = EXPECTED_STEPS_BY_METHOD[method]
        indexed = {
            (int(row["policy_step"]), int(row["evaluator_step"])): row
            for row in cells
            if row["method"] == method
        }
        for metric in ("mad_gain", "ptr_gain", "zar_gain"):
            rows = []
            for policy_step in steps:
                rows.append(
                    {
                        "policy_checkpoint": policy_step,
                        **{
                            f"evaluator_{evaluator_step}": (
                                indexed[(policy_step, evaluator_step)][metric]
                                if (policy_step, evaluator_step) in indexed
                                else ""
                            )
                            for evaluator_step in steps
                        },
                    }
                )
            path = root / "metrics" / f"{method}_{metric}_lower_triangle.csv"
            _write_csv(path, rows)
            paths.append(str(path))
    return paths


def _plot_outputs(
    root: Path, cells: Sequence[Mapping[str, Any]], trajectories: Sequence[Mapping[str, Any]]
) -> list[str]:
    import matplotlib.pyplot as plt

    figure_paths = []
    for method in METHODS:
        steps = EXPECTED_STEPS_BY_METHOD[method]
        subset = [row for row in cells if row["method"] == method]
        figure, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
        for axis, metric, label in zip(
            axes, ("mad_gain", "ptr_gain", "zar_gain"), ("MAD gain", "PTR gain", "ZAR gain")
        ):
            matrix = [[float("nan") for _ in steps] for _ in steps]
            values = []
            for row in subset:
                y, x = steps.index(row["policy_step"]), steps.index(row["evaluator_step"])
                matrix[y][x] = row[metric]
                values.append(abs(row[metric]))
            limit = max(values) or 1e-12
            image = axis.imshow(
                matrix, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit
            )
            axis.set_title(f"{label}\nPositive = current rubric distinguishes responses better")
            axis.set_xlabel("Reused evaluator checkpoint")
            axis.set_ylabel("Current policy checkpoint")
            axis.set_xticks(range(len(steps)), steps, rotation=90)
            axis.set_yticks(range(len(steps)), steps)
            figure.colorbar(image, ax=axis, shrink=0.75)
        path = root / "metrics" / f"{method}_positive_current_gain_matrices.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        figure_paths.append(str(path))

    figure, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)
    row_specs = (
        ("R0_vs_Rt", "Initial rubric → current rubric"),
        ("R_previous_vs_Rt", "Previous rubric → current rubric"),
    )
    for row_index, (comparison, title) in enumerate(row_specs):
        for column_index, (metric, label) in enumerate(
            (("mad_gain", "MAD gain"), ("ptr_gain", "PTR gain"), ("zar_gain", "ZAR gain"))
        ):
            axis = axes[row_index][column_index]
            for method in METHODS:
                points = [
                    row
                    for row in trajectories
                    if row["method"] == method and row["comparison"] == comparison
                ]
                axis.plot(
                    [row["policy_step"] for row in points],
                    [row[metric] for row in points],
                    marker="o",
                    label=method,
                )
            axis.axhline(0, color="black", linewidth=0.8)
            axis.set_title(f"{title}: {label}")
            axis.set_xlabel("Policy checkpoint")
            axis.set_ylabel("Positive = current rubric better")
            axis.legend()
    path = root / "metrics" / "static_online_r0_previous_to_current_2x3.png"
    figure.savefig(path, dpi=180)
    plt.close(figure)
    figure_paths.append(str(path))
    return figure_paths


def analyze(config: Mapping[str, Any]) -> dict[str, Any]:
    """Aggregate only sealed score cells into prompt metrics, matrices and figures."""
    root = prepare(config)
    cell_paths = sorted((root / "grades" / "cells").glob("*/*/*.jsonl"))
    expected_by_method = {
        method: len(steps) * (len(steps) + 1) // 2
        for method, steps in EXPECTED_STEPS_BY_METHOD.items()
    }
    expected_cells = sum(expected_by_method.values())
    _require(
        len(cell_paths) == expected_cells,
        f"all sealed score cells required: {len(cell_paths)} != {expected_cells}",
    )
    for path in cell_paths:
        _verify_sealed_cell(path)
    rows = [row for path in cell_paths for row in read_jsonl(path)]
    groups: dict[tuple[str, int, int, str], list[dict]] = {}
    for row in rows:
        _require(
            row.get("data_role") == DATA_ROLE and row.get("pool") == "probe_B",
            "score provenance mismatch",
        )
        key = (
            str(row["method"]),
            int(row["policy_step"]),
            int(row["evaluator_step"]),
            str(row["prompt_id"]),
        )
        groups.setdefault(key, []).append(row)
    epsilon = float(config["metrics"]["pairwise_tie_epsilon"])
    prompt_metrics: dict[tuple[str, int, int, str], dict[str, float]] = {}
    response_ids: dict[tuple[str, int, str], set[str]] = {}
    prompt_rows = []
    for key, group in groups.items():
        _require(
            len(group) == POOL_B_COUNT, f"score group must contain {POOL_B_COUNT} Pool-B responses"
        )
        ids = {str(row["response_id"]) for row in group}
        _require(len(ids) == POOL_B_COUNT, "duplicate score response ID")
        policy_key = (key[0], key[1], key[3])
        if policy_key in response_ids:
            _require(
                response_ids[policy_key] == ids,
                "evaluators were not scored on identical Pool-B responses",
            )
        response_ids[policy_key] = ids
        values = metric_values([float(row["reward"]) for row in group], epsilon)
        prompt_metrics[key] = values
        prompt_rows.append(
            {
                "data_role": DATA_ROLE,
                "method": key[0],
                "policy_step": key[1],
                "evaluator_step": key[2],
                "prompt_id": key[3],
                **values,
            }
        )
    _require(len(prompt_rows) == expected_cells * 100, "prompt-level metric inventory mismatch")
    write_jsonl_atomic(root / "metrics" / "prompt_level_metrics.jsonl", prompt_rows)
    _write_csv(root / "metrics" / "prompt_level_metrics.csv", prompt_rows)

    cells = []
    trajectories = []
    for method in METHODS:
        steps = EXPECTED_STEPS_BY_METHOD[method]
        for policy_index, policy_step in enumerate(steps):
            prompt_ids = sorted(
                key[3] for key in prompt_metrics if key[:3] == (method, policy_step, policy_step)
            )
            _require(
                len(prompt_ids) == 100, f"missing current evaluator groups: {method}/{policy_step}"
            )
            current_mean = {
                name: sum(
                    prompt_metrics[(method, policy_step, policy_step, pid)][name]
                    for pid in prompt_ids
                )
                / len(prompt_ids)
                for name in ("mad", "zar", "ptr", "separation_rate", "reward_std")
            }
            for evaluator_step in steps[: policy_index + 1]:
                stale_mean = {
                    name: sum(
                        prompt_metrics[(method, policy_step, evaluator_step, pid)][name]
                        for pid in prompt_ids
                    )
                    / len(prompt_ids)
                    for name in current_mean
                }
                gains = gain_values(stale_mean, current_mean)
                cells.append(
                    {
                        "data_role": DATA_ROLE,
                        "method": method,
                        "policy_step": policy_step,
                        "evaluator_step": evaluator_step,
                        **{f"stale_{name}": value for name, value in stale_mean.items()},
                        **{f"current_{name}": value for name, value in current_mean.items()},
                        **gains,
                        "positive_means_current_rubric_better": True,
                        "pairwise_tie_epsilon": epsilon,
                    }
                )
            previous = steps[policy_index - 1] if policy_index else None
            for comparison, stale_step in (("R0_vs_Rt", 0), ("R_previous_vs_Rt", previous)):
                if stale_step is None:
                    continue
                stale = next(
                    row
                    for row in cells
                    if row["method"] == method
                    and row["policy_step"] == policy_step
                    and row["evaluator_step"] == stale_step
                )
                trajectories.append(
                    {
                        "data_role": DATA_ROLE,
                        "method": method,
                        "policy_step": policy_step,
                        "comparison": comparison,
                        "stale_evaluator_step": stale_step,
                        "current_evaluator_step": policy_step,
                        "mad_gain": stale["mad_gain"],
                        "zar_gain": stale["zar_gain"],
                        "ptr_gain": stale["ptr_gain"],
                        "positive_means_current_rubric_better": True,
                    }
                )
    for method, expected in expected_by_method.items():
        _require(
            sum(row["method"] == method for row in cells) == expected,
            f"{method} matrix is not {expected} cells",
        )
    write_jsonl_atomic(root / "metrics" / "reuse_matrix_cells.jsonl", cells)
    write_jsonl_atomic(root / "metrics" / "trajectory_2x3_inputs.jsonl", trajectories)
    _write_csv(root / "metrics" / "reuse_matrix_cells.csv", cells)
    _write_csv(root / "metrics" / "trajectory_2x3_inputs.csv", trajectories)
    matrix_csvs = _write_matrix_csvs(root, cells)
    figures = _plot_outputs(root, cells, trajectories)
    summary = {
        "schema_version": 1,
        "data_role": DATA_ROLE,
        "matrix_cells": len(cells),
        "matrix_cells_per_method": expected_by_method,
        "prompt_metric_rows": len(prompt_rows),
        "trajectory_rows": len(trajectories),
        "metrics": ["MAD", "ZAR", "PTR"],
        "positive_means_current_rubric_better": True,
        "pairwise_tie_epsilon": epsilon,
        "ranking_similarity_is_not_correctness": True,
        "initial_rubric_is_ground_truth": False,
        "lower_triangle_matrix_csvs": matrix_csvs,
        "figures": figures,
    }
    write_json_atomic(root / "metrics" / "summary.json", summary)
    return summary


def _resolve_smoke_model(
    config: Mapping[str, Any],
    component: str,
    served_model: str | None,
    extractor_model: str | None,
    judge_model: str | None,
) -> str:
    if component == "rubric":
        return extractor_model or served_model or str(config["rubric_generation"]["model"])
    if component == "score":
        return judge_model or served_model or str(config["grading"]["model"])
    _require(served_model, f"integration-smoke {component} requires --served-model")
    return served_model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=(
            "preflight",
            "prepare",
            "generate-control",
            "generate",
            "validate-pools",
            "build-rubrics",
            "score-cell",
            "score-all",
            "integration-smoke",
            "policy-lifecycle",
            "run-all",
            "analyze",
        ),
        required=True,
    )
    parser.add_argument("--method", choices=METHODS)
    parser.add_argument("--step", type=int)
    parser.add_argument("--evaluator-step", type=int)
    parser.add_argument("--served-model")
    parser.add_argument("--extractor-model")
    parser.add_argument("--judge-model")
    parser.add_argument(
        "--runtime-base-url",
        help="Runtime-only extractor endpoint for build-rubrics; canonical config remains unchanged",
    )
    parser.add_argument(
        "--runtime-judge-base-url",
        action="append",
        help="Runtime-only verified judge replica; repeat for each active endpoint",
    )
    parser.add_argument(
        "--runtime-judge-workers",
        type=int,
        help="Runtime-only total grading concurrency across judge replicas",
    )
    parser.add_argument(
        "--stop-after-phase",
        choices=("smoke-gate", "policies", "rubrics", "scores", "analyze"),
    )
    parser.add_argument(
        "--smoke-component",
        choices=("policy-base", "policy-previous", "policy-current", "rubric", "score"),
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args()
    config = load_config(args.config, args.output_root)
    if args.dry_run:
        resolved_smoke_model = (
            _resolve_smoke_model(
                config,
                args.smoke_component,
                args.served_model,
                args.extractor_model,
                args.judge_model,
            )
            if args.stage == "integration-smoke" and args.smoke_component
            else None
        )
        print(
            json.dumps(
                {
                    "stage": args.stage,
                    "dry_run": True,
                    "data_role": DATA_ROLE,
                    "network_called": False,
                    "gpu_called": False,
                    **(
                        {"resolved_smoke_model": resolved_smoke_model}
                        if resolved_smoke_model
                        else {}
                    ),
                },
                sort_keys=True,
            )
        )
        return
    if args.stage == "preflight":
        result = preflight(config)
    elif args.stage == "prepare":
        result = {"output_root": str(prepare(config))}
    elif args.stage == "generate-control":
        _require(
            args.method is not None and args.served_model,
            "generate-control requires --method and --served-model",
        )
        result = {"path": str(generate_control(config, args.method, args.served_model))}
    elif args.stage == "generate":
        _require(
            args.method is not None and args.step is not None and args.served_model,
            "generate requires --method, --step and --served-model",
        )
        result = {"path": str(generate(config, args.method, args.step, args.served_model))}
    elif args.stage == "validate-pools":
        result = validate_pools(config)
    elif args.stage == "build-rubrics":
        _require(
            args.method is not None and args.step is not None and args.served_model,
            "build-rubrics requires --method, --step and --served-model",
        )
        result = {
            "path": str(
                build_rubrics(
                    config,
                    args.method,
                    args.step,
                    args.served_model,
                    runtime_base_url=args.runtime_base_url,
                )
            )
        }
    elif args.stage == "integration-smoke":
        _require(
            args.method is not None and args.step is not None and args.smoke_component,
            "integration-smoke requires method, step and smoke-component",
        )
        smoke_model = _resolve_smoke_model(
            config,
            args.smoke_component,
            args.served_model,
            args.extractor_model,
            args.judge_model,
        )
        result = integration_smoke(
            config,
            args.method,
            args.step,
            args.smoke_component,
            smoke_model,
        )
    elif args.stage == "policy-lifecycle":
        _require(
            args.method is not None and args.step is not None,
            "policy-lifecycle requires --method and --step",
        )
        result = policy_lifecycle(config, args.method, args.step)
    elif args.stage == "score-cell":
        _require(
            args.method is not None
            and args.step is not None
            and args.evaluator_step is not None
            and args.served_model,
            "score-cell requires --method, --step, --evaluator-step and --served-model",
        )
        result = {
            "path": str(
                score_cell(
                    config,
                    args.method,
                    args.step,
                    args.evaluator_step,
                    args.served_model,
                    runtime_base_urls=args.runtime_judge_base_url,
                    runtime_workers=args.runtime_judge_workers,
                )
            )
        }
    elif args.stage == "score-all":
        _require(args.served_model, "score-all requires --served-model")
        result = score_all_cells(
            config,
            args.served_model,
            runtime_base_urls=args.runtime_judge_base_url,
            runtime_workers=args.runtime_judge_workers,
        )
    elif args.stage == "run-all":
        _require(
            args.extractor_model and args.judge_model,
            "run-all requires --extractor-model and --judge-model",
        )
        result = run_all(
            config,
            args.extractor_model,
            args.judge_model,
            args.stop_after_phase,
        )
    else:
        result = analyze(config)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
