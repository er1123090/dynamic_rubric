"""Live minimum staleness experiment over the completed static-RL artifacts.

This module implements the causal comparisons requested for the minimum
experiment: static R0 versus current-aligned dynamic_fixed_budgeted or
dynamic_prev_budgeted Rt on one shared BoN pool.  Every expensive operation is
published as an immutable prompt or prompt/policy shard so a stopped run resumes
without recomputing completed work.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import os
import tempfile
import time
import urllib.request
from collections import defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence, Set
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from .artifacts import read_jsonl, write_json_atomic
from .batch_dynamic import dynamic_batch_stage
from .evaluation.bon import BON_SIZES, fixed_candidate_permutations, select_best_of_n
from .evaluation.proxy_score import normalized_yes_probability
from .hashing import canonical_json_bytes, sha256_file
from .judge_prompts import PAPER_JUDGE_PROMPT_VERSION, qwen_criterion_prompt
from .providers.local_embedding import LocalBGEEmbeddingProvider
from .rubrics.admission import AdmissionEvidence
from .rubrics.replay import (
    ReplayCandidate,
    ReplayMode,
    initial_replay_snapshot,
    replay_step,
)
from .rubrics.static import (
    Criterion,
    CriterionValidationError,
    StaticRubric,
    validate_criterion_text,
)


FOCAL_STEPS = (3, 10, 30, 50)
N_GRID = BON_SIZES
PERMUTATIONS = 5
MODE = ReplayMode.DYNAMIC_FIXED_BUDGETED.value
QWEN_MODEL = "Qwen/Qwen3-32B"
QWEN_REVISION = "9216db5781bf21249d130ec9da846c4624c16137"
BGE_MODEL = "BAAI/bge-m3"
BGE_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"
LEGACY_JUDGE_PROMPT_VERSION = "criterion-response-answer-v1"


class MinimumExperimentError(RuntimeError):
    """Raised when a minimum-experiment invariant is not satisfied."""


def _minimum_replay_stage(mode: ReplayMode | str) -> str:
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return "replay-dynamic-minimum"
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return "replay-dynamic-prev-minimum"
    raise ValueError(f"unsupported minimum replay mode: {parsed.value}")


def _minimum_score_stage(mode: ReplayMode | str) -> str:
    parsed = ReplayMode(mode)
    if parsed is ReplayMode.DYNAMIC_FIXED_BUDGETED:
        return "score-proxy-minimum"
    if parsed is ReplayMode.DYNAMIC_PREV_BUDGETED:
        return "score-proxy-prev-minimum"
    raise ValueError(f"unsupported minimum score mode: {parsed.value}")


def _jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise MinimumExperimentError(f"{path}:{line_number} is not an object")
            yield value


def _publish_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> bool:
    """Stream an immutable JSONL artifact without buffering it in memory."""

    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            for row in rows:
                stream.write(canonical_json_bytes(row) + b"\n")
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _publish_gzip_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> bool:
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as raw:
            with gzip.GzipFile(fileobj=raw, mode="wb", mtime=0) as stream:
                for row in rows:
                    stream.write(canonical_json_bytes(row) + b"\n")
            raw.flush()
            os.fsync(raw.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            return False
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def _write_mutable_json(path: Path, value: Mapping[str, Any]) -> None:
    write_json_atomic(path, value, immutable=False)


def _shard_name(*parts: object) -> str:
    label = "-".join(str(part) for part in parts)
    return hashlib.sha256(label.encode()).hexdigest()


def _score_prompt(
    criterion_text: str,
    response_text: str,
    *,
    conversation: Sequence[Mapping[str, Any]] | None = None,
    prompt_version: str = LEGACY_JUDGE_PROMPT_VERSION,
) -> str:
    if prompt_version == LEGACY_JUDGE_PROMPT_VERSION:
        if conversation is not None:
            raise MinimumExperimentError("legacy judge prompt does not accept a conversation")
        return f"Criterion: {criterion_text}\nResponse: {response_text}\nAnswer:"
    if prompt_version == PAPER_JUDGE_PROMPT_VERSION:
        if conversation is None:
            raise MinimumExperimentError("paper judge prompt requires the user conversation")
        return qwen_criterion_prompt(conversation, response_text, criterion_text)
    raise MinimumExperimentError(f"unsupported judge prompt version: {prompt_version}")


def _audit_conversations(run_root: Path) -> dict[str, tuple[dict[str, Any], ...]]:
    root = run_root.parents[2]
    path = root / "data" / "public" / "pilot_audit.jsonl"
    conversations: dict[str, tuple[dict[str, Any], ...]] = {}
    for row in _jsonl(path):
        prompt_id = str(row["prompt_id"])
        messages = row.get("messages")
        if not isinstance(messages, list) or not messages:
            raise MinimumExperimentError(f"audit conversation is malformed: {prompt_id}")
        normalized = tuple(dict(message) for message in messages if isinstance(message, Mapping))
        if len(normalized) != len(messages):
            raise MinimumExperimentError(f"audit conversation contains a non-object: {prompt_id}")
        if prompt_id in conversations:
            raise MinimumExperimentError(f"duplicate audit conversation: {prompt_id}")
        conversations[prompt_id] = normalized
    return conversations


class TargetScoreClient:
    """Concurrent client for the pinned multi-upstream target-logprob proxy."""

    def __init__(
        self,
        endpoint: str,
        *,
        batch_size: int = 32,
        workers: int = 12,
        timeout: float = 900.0,
        retries: int = 5,
    ) -> None:
        if batch_size < 1 or workers < 1:
            raise ValueError("batch_size and workers must be positive")
        self.endpoint = endpoint.rstrip("/")
        self.batch_size = batch_size
        self.workers = workers
        self.timeout = timeout
        self.retries = retries
        self._routing_weights: tuple[int, ...] | None = None

    def identity(self) -> dict[str, Any]:
        request = urllib.request.Request(f"{self.endpoint}/dynamic-rubric/identity", method="GET")
        with urllib.request.urlopen(request, timeout=30) as response:
            value = json.loads(response.read())
        if (
            value.get("served_model") != QWEN_MODEL
            or value.get("model_revision") != QWEN_REVISION
            or value.get("tokenizer_revision") != QWEN_REVISION
        ):
            raise MinimumExperimentError(f"Qwen proxy identity drift: {value}")
        return value

    def routing_weights(self) -> tuple[int, ...]:
        if self._routing_weights is None:
            request = urllib.request.Request(
                f"{self.endpoint}/dynamic-rubric/routing", method="GET"
            )
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.loads(response.read())
            weights = value.get("upstream_weights")
            if (
                value.get("strategy") != "deterministic-length-balanced"
                or not isinstance(weights, list)
                or not weights
                or len(weights) != int(value.get("upstream_count", 0))
                or any(
                    isinstance(weight, bool) or not isinstance(weight, int) or weight < 1
                    for weight in weights
                )
            ):
                raise MinimumExperimentError(f"invalid Qwen routing contract: {value}")
            self._routing_weights = tuple(weights)
        return self._routing_weights

    def _post(self, prompts: Sequence[str], upstream_index: int) -> list[dict[str, float]]:
        payload = {
            "rendered_prompts": list(prompts),
            "targets": ["YES", "NO"],
            "temperature": 0,
            "thinking": False,
            "routing_upstream_index": upstream_index,
        }
        request = urllib.request.Request(
            f"{self.endpoint}/dynamic-rubric/score-targets",
            data=json.dumps(payload, ensure_ascii=False).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        last_error: BaseException | None = None
        for attempt in range(self.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    value = json.loads(response.read())
                rows = value.get("target_logprobs")
                if isinstance(rows, Mapping):
                    rows = [rows]
                if not isinstance(rows, list) or len(rows) != len(prompts):
                    raise MinimumExperimentError("target-logprob response count mismatch")
                normalized = []
                for row in rows:
                    if not isinstance(row, Mapping) or set(row) != {"YES", "NO"}:
                        raise MinimumExperimentError("malformed YES/NO target logprobs")
                    yes = float(row["YES"])
                    no = float(row["NO"])
                    normalized.append(
                        {
                            "yes_logprob": yes,
                            "no_logprob": no,
                            "probability_yes": normalized_yes_probability(yes, no),
                        }
                    )
                return normalized
            except BaseException as error:
                last_error = error
                if attempt + 1 < self.retries:
                    time.sleep(min(30.0, 2.0**attempt))
        assert last_error is not None
        raise MinimumExperimentError(f"Qwen score request failed: {last_error}") from last_error

    def score(
        self, tasks: Sequence[tuple[tuple[str, str], str]]
    ) -> dict[tuple[str, str], dict[str, float]]:
        keys = [key for key, _ in tasks]
        if len(keys) != len(set(keys)):
            raise MinimumExperimentError("duplicate criterion/response score task")
        chunks = [
            tasks[start : start + self.batch_size]
            for start in range(0, len(tasks), self.batch_size)
        ]
        routes = _balanced_routes(chunks, self.routing_weights())
        routed_chunks: dict[int, list[Sequence[tuple[tuple[str, str], str]]]] = defaultdict(list)
        for index, chunk in enumerate(chunks):
            routed_chunks[routes[index]].append(chunk)

        def score_route(
            upstream_index: int,
        ) -> list[
            tuple[
                Sequence[tuple[tuple[str, str], str]],
                list[dict[str, float]],
            ]
        ]:
            completed = []
            for chunk in routed_chunks[upstream_index]:
                rows = self._post(
                    [prompt for _, prompt in chunk],
                    upstream_index,
                )
                completed.append((chunk, rows))
            return completed

        result: dict[tuple[str, str], dict[str, float]] = {}
        with ThreadPoolExecutor(max_workers=min(self.workers, len(routed_chunks))) as executor:
            futures = {
                executor.submit(score_route, upstream_index): upstream_index
                for upstream_index in sorted(routed_chunks)
            }
            for future in as_completed(futures):
                for chunk, rows in future.result():
                    for (key, _), row in zip(chunk, rows):
                        result[key] = row
        if len(result) != len(tasks):
            raise MinimumExperimentError("incomplete criterion/response score result")
        return result


def _balanced_routes(
    chunks: Sequence[Sequence[tuple[tuple[str, str], str]]],
    weights: Sequence[int],
) -> tuple[int, ...]:
    """Assign larger prompt batches first to the least normalized replica load."""

    if not weights or any(
        isinstance(weight, bool) or not isinstance(weight, int) or weight < 1 for weight in weights
    ):
        raise ValueError("routing weights must be positive integers")
    costs = [sum(len(prompt.encode("utf-8")) for _, prompt in chunk) for chunk in chunks]
    loads = [0] * len(weights)
    routes = [0] * len(chunks)
    for chunk_index in sorted(range(len(chunks)), key=lambda index: (-costs[index], index)):
        route = min(
            range(len(weights)),
            key=lambda index: (loads[index] / weights[index], index),
        )
        routes[chunk_index] = route
        loads[route] += costs[chunk_index]
    return tuple(routes)


def _static_rubrics(path: Path) -> tuple[dict[str, StaticRubric], dict[str, dict[str, Any]]]:
    rubrics: dict[str, StaticRubric] = {}
    source: dict[str, dict[str, Any]] = {}
    for row in _jsonl(path):
        prompt_id = str(row["prompt_id"])
        criteria = tuple(Criterion(**dict(item)) for item in row["criteria"])
        rubric = StaticRubric(prompt_id, criteria)
        if rubric.content_hash != row["content_hash"]:
            raise MinimumExperimentError(f"static rubric hash mismatch: {prompt_id}")
        rubrics[prompt_id] = rubric
        source[prompt_id] = row
    return rubrics, source


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(left, right))


def _candidate_id(prompt_id: str, step: int, index: int, text: str, mode: ReplayMode) -> str:
    prompt_hash = hashlib.sha256(prompt_id.encode()).hexdigest()[:8]
    text_hash = hashlib.sha256(text.encode()).hexdigest()[:10]
    label = "dyn-fixed" if mode is ReplayMode.DYNAMIC_FIXED_BUDGETED else "dyn-prev"
    return f"{label}-{prompt_hash}-{step:03d}-{index}-{text_hash}"


def _candidate_rows(path: Path) -> dict[str, dict[int, dict[str, Any]]]:
    rows: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for row in _jsonl(path):
        if row["replicate_id"] != "A":
            continue
        prompt_id = str(row["prompt_id"])
        step = int(row["policy_step"])
        if step in rows[prompt_id]:
            raise MinimumExperimentError(
                f"duplicate primary dynamic candidate: {prompt_id=} {step=}"
            )
        rows[prompt_id][step] = row
    return rows


def _validation_rows(
    run_root: Path, prompt_ids: set[str]
) -> tuple[
    dict[str, tuple[dict[str, Any], ...]], dict[tuple[str, int], tuple[dict[str, Any], ...]]
]:
    reference: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in _jsonl(run_root / "train-static" / "reference_responses.jsonl"):
        prompt_id = str(row["prompt_id"])
        if prompt_id in prompt_ids and row["family"] == "reference_validation":
            reference[prompt_id].append(row)
    current: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for step in range(1, 51):
        path = run_root / "train-static" / "verl-run" / "probes" / f"{step}.jsonl"
        if not path.is_file():
            raise MinimumExperimentError(f"missing trajectory probe: {path}")
        for source in _jsonl(path):
            prompt_id = str(source["prompt_id"])
            if prompt_id not in prompt_ids or source["family"] != "trajectory_validation":
                continue
            row = dict(source)
            row["response_text"] = str(row["output"])
            current[(prompt_id, step)].append(row)
    normalized_reference = {
        prompt_id: tuple(sorted(values, key=lambda row: int(row["sample_index"])))
        for prompt_id, values in reference.items()
    }
    normalized_current = {
        key: tuple(sorted(values, key=lambda row: int(row["sample_index"])))
        for key, values in current.items()
    }
    for prompt_id in prompt_ids:
        if len(normalized_reference.get(prompt_id, ())) != 4:
            raise MinimumExperimentError(f"reference validation inventory mismatch: {prompt_id}")
        for step in range(1, 51):
            if len(normalized_current.get((prompt_id, step), ())) != 4:
                raise MinimumExperimentError(
                    f"current validation inventory mismatch: {prompt_id=} {step=}"
                )
    return normalized_reference, normalized_current


def replay_current_aligned(
    run_root: Path,
    score_endpoint: str,
    embedding_model_path: Path,
    *,
    embedding_device: str = "cuda:0",
    workers: int = 12,
    mode: ReplayMode | str = ReplayMode.DYNAMIC_FIXED_BUDGETED,
) -> dict[str, Any]:
    """Admit fixed- or previous-control candidates with held-out Qwen validation."""

    parsed_mode = ReplayMode(mode)
    mode_value = parsed_mode.value
    stage_name = _minimum_replay_stage(parsed_mode)
    stage_root = run_root / stage_name
    static_path = run_root / "generate-static" / "static_rubrics.jsonl"
    candidates_path = run_root / dynamic_batch_stage(mode_value) / "dynamic_candidates.jsonl"
    references_path = run_root / "train-static" / "reference_responses.jsonl"
    input_paths = [static_path, candidates_path, references_path]
    input_paths.extend(
        run_root / "train-static" / "verl-run" / "probes" / f"{step}.jsonl" for step in range(1, 51)
    )
    missing = [str(path) for path in input_paths if not path.is_file()]
    if missing:
        raise MinimumExperimentError(f"replay inputs are incomplete: {missing[:3]}")
    client = TargetScoreClient(score_endpoint, workers=workers)
    proxy_identity = client.identity()
    routing_contract = run_root / "train-static" / "judge-routing-minimum-v2.json"
    if not routing_contract.is_file():
        raise MinimumExperimentError(f"missing judge routing contract: {routing_contract}")
    manifest = {
        "schema_version": 1,
        "comparison": ["static", mode_value],
        "focal_steps": list(FOCAL_STEPS),
        "replay_steps": [1, 50],
        "validation_responses_per_side": 4,
        "admission_score": proxy_identity,
        "score_routing": {
            "strategy": "deterministic-length-balanced-largest-first",
            "upstream_weights": list(client.routing_weights()),
            "contract_path": str(routing_contract),
            "contract_sha256": sha256_file(routing_contract),
        },
        "semantic_embedding": {
            "model": BGE_MODEL,
            "revision": BGE_REVISION,
            "device": embedding_device,
            "pooling": "normalized_cls",
        },
        "inputs": {str(path): sha256_file(path) for path in input_paths},
    }
    write_json_atomic(stage_root / "manifest.json", manifest)
    rubrics, _ = _static_rubrics(static_path)
    candidates = _candidate_rows(candidates_path)
    prompt_ids = set(candidates)
    if len(prompt_ids) != 144 or any(
        set(rows) != set(range(1, 51)) for rows in candidates.values()
    ):
        raise MinimumExperimentError(
            "primary dynamic candidate inventory must be 144 prompts x 50 steps"
        )
    references, current = _validation_rows(run_root, prompt_ids)
    audit_prompt_ids = _audit_prompt_ids(run_root)
    embedder = LocalBGEEmbeddingProvider(
        embedding_model_path,
        BGE_MODEL,
        BGE_REVISION,
        device=embedding_device,
        batch_size=64,
    )
    completed = 0
    for prompt_index, prompt_id in enumerate(sorted(prompt_ids), 1):
        shard = stage_root / "shards" / f"prompt-{_shard_name(prompt_id)}.jsonl"
        if shard.is_file():
            completed += 1
            continue
        rubric = rubrics.get(prompt_id)
        if rubric is None:
            raise MinimumExperimentError(f"missing static rubric: {prompt_id}")
        definitions: dict[int, list[dict[str, Any]]] = defaultdict(list)
        valid_texts = [criterion.text for criterion in rubric.criteria]
        for step in range(1, 51):
            row = candidates[prompt_id][step]
            for index, item in enumerate(row["criteria"]):
                text = str(item["text"])
                candidate = {
                    "criterion_id": _candidate_id(prompt_id, step, index, text, parsed_mode),
                    "text": text,
                    "rationale": str(item["rationale"]),
                    "candidate_index": index,
                }
                try:
                    validate_criterion_text(text)
                except CriterionValidationError as error:
                    candidate["structural_error"] = str(error)
                else:
                    valid_texts.append(text)
                definitions[step].append(candidate)
        unique_texts = list(dict.fromkeys(valid_texts))
        vectors = embedder.embed(unique_texts)
        vector_by_text: dict[str, Sequence[float]] = {}
        for text, vector in zip(unique_texts, vectors):
            vector_by_text[text] = vector
            # replay_step stores the structurally validated (whitespace-normalized)
            # form. Keep that form bound to the exact same embedding so criteria
            # admitted on an earlier step remain available for later similarity
            # checks without an extra provider call.
            vector_by_text[" ".join(text.split())] = vector
        score_tasks: list[tuple[tuple[str, str], str]] = []
        for step in range(1, 51):
            control_rows = (
                references[prompt_id]
                if parsed_mode is ReplayMode.DYNAMIC_FIXED_BUDGETED or step == 1
                else current[(prompt_id, step - 1)]
            )
            validation_rows = (*current[(prompt_id, step)], *control_rows)
            for candidate in definitions[step]:
                if "structural_error" in candidate:
                    continue
                for response in validation_rows:
                    key = candidate["criterion_id"], str(response["response_id"])
                    score_tasks.append(
                        (key, _score_prompt(candidate["text"], str(response["response_text"])))
                    )
        score_rows = client.score(score_tasks)
        snapshot = initial_replay_snapshot(rubric, parsed_mode)
        output: list[dict[str, Any]] = []
        for step in range(1, 51):
            current_rows = current[(prompt_id, step)]
            control_rows = (
                references[prompt_id]
                if parsed_mode is ReplayMode.DYNAMIC_FIXED_BUDGETED or step == 1
                else current[(prompt_id, step - 1)]
            )
            replay_candidates: list[ReplayCandidate] = []
            evidence_by_id: dict[str, AdmissionEvidence] = {}
            for candidate in definitions[step]:
                if "structural_error" in candidate:
                    continue
                criterion_id = str(candidate["criterion_id"])
                current_scores = [
                    score_rows[(criterion_id, str(row["response_id"]))]["probability_yes"]
                    for row in current_rows
                ]
                control_scores = [
                    score_rows[(criterion_id, str(row["response_id"]))]["probability_yes"]
                    for row in control_rows
                ]
                values = current_scores + control_scores
                mean = math.fsum(values) / len(values)
                variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
                vector = vector_by_text[str(candidate["text"])]
                max_similarity = max(
                    _cosine(vector, vector_by_text[item.text]) for item in snapshot.criteria
                )
                evidence = AdmissionEvidence(
                    satisfaction_rate=sum(value >= 0.5 for value in values) / len(values),
                    current_mean=math.fsum(current_scores) / len(current_scores),
                    reference_mean=math.fsum(control_scores) / len(control_scores),
                    max_active_similarity=max_similarity,
                    parse_success=1.0,
                    independent_validation=True,
                    score_variance=variance,
                    recent_validation_failure_rate=0.0,
                )
                evidence_by_id[criterion_id] = evidence
                replay_candidates.append(
                    ReplayCandidate(criterion_id, str(candidate["text"]), evidence)
                )
            snapshot = replay_step(snapshot, step=step, candidates=replay_candidates)
            decisions = {
                item.criterion_id: dataclasses.asdict(item.decision)
                for item in snapshot.candidate_results
            }
            evidence_rows = []
            for candidate in definitions[step]:
                criterion_id = str(candidate["criterion_id"])
                row = dict(candidate)
                evidence = evidence_by_id.get(criterion_id)
                if evidence is not None:
                    row["evidence"] = dataclasses.asdict(evidence)
                    row["decision"] = decisions[criterion_id]
                else:
                    row["decision"] = {
                        "admitted": False,
                        "utility": 0.0,
                        "failed_gates": ["structural_validation"],
                    }
                evidence_rows.append(row)
            output.append(
                {
                    "prompt_id": prompt_id,
                    "split": "final" if prompt_id in audit_prompt_ids else "development",
                    "policy_step": step,
                    "mode": mode_value,
                    "rubric_step": step,
                    "criteria": [dataclasses.asdict(item) for item in snapshot.criteria],
                    "criterion_count": len(snapshot.criteria),
                    "content_hash": snapshot.content_hash,
                    "admitted_id": snapshot.admitted_id,
                    "evicted_id": snapshot.evicted_id,
                    "candidate_evidence": evidence_rows,
                    "validation_current_response_ids": [row["response_id"] for row in current_rows],
                    "validation_control_policy": (
                        "pi_0"
                        if parsed_mode is ReplayMode.DYNAMIC_FIXED_BUDGETED or step == 1
                        else f"pi_{step - 1}"
                    ),
                    "validation_control_response_ids": [row["response_id"] for row in control_rows],
                    "generator_pairing_hash": candidates[prompt_id][step]["pairing_hash"],
                    "generator_source_blind": True,
                    "gold_access": False,
                }
            )
        _publish_jsonl(shard, output)
        completed += 1
        _write_mutable_json(
            stage_root / "progress.json",
            {
                "completed_prompt_shards": completed,
                "expected_prompt_shards": len(prompt_ids),
                "last_prompt_id": prompt_id,
                "prompt_index": prompt_index,
            },
        )
    shard_paths = sorted((stage_root / "shards").glob("prompt-*.jsonl"))
    output_path = stage_root / "replay_snapshots.jsonl"
    _publish_jsonl(output_path, (row for path in shard_paths for row in _jsonl(path)))
    snapshots = sum(1 for _ in _jsonl(output_path))
    admissions = sum(row["admitted_id"] is not None for row in _jsonl(output_path))
    result = {
        "prompt_shards": len(shard_paths),
        "snapshots": snapshots,
        "admissions": admissions,
        "focal_steps": list(FOCAL_STEPS),
        "mode": mode_value,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(stage_root / "result.json", result)
    return result


def _audit_prompt_ids(run_root: Path) -> set[str]:
    root = run_root.parents[2]
    path = root / "data" / "public" / "pilot_audit.jsonl"
    return {str(row["prompt_id"]) for row in _jsonl(path)}


def _focal_rubrics(
    run_root: Path,
    mode: ReplayMode | str = ReplayMode.DYNAMIC_FIXED_BUDGETED,
) -> tuple[dict[str, dict[str, Any]], dict[tuple[str, int], dict[str, Any]]]:
    parsed_mode = ReplayMode(mode)
    mode_value = parsed_mode.value
    _, static = _static_rubrics(run_root / "generate-static" / "static_rubrics.jsonl")
    audit = _audit_prompt_ids(run_root)
    dynamic: dict[tuple[str, int], dict[str, Any]] = {}
    replay_path = run_root / _minimum_replay_stage(parsed_mode) / "replay_snapshots.jsonl"
    for row in _jsonl(replay_path):
        key = str(row["prompt_id"]), int(row["policy_step"])
        if key[0] in audit and key[1] in FOCAL_STEPS:
            dynamic[key] = row
    expected = {(prompt_id, step) for prompt_id in audit for step in FOCAL_STEPS}
    if set(dynamic) != expected:
        raise MinimumExperimentError(
            f"current-aligned focal rubric inventory mismatch: {mode_value}"
        )
    return {key: value for key, value in static.items() if key in audit}, dynamic


def _bon_groups(path: Path) -> Iterator[tuple[tuple[str, str], list[dict[str, Any]]]]:
    previous: tuple[str, str] | None = None
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in _jsonl(path):
        key = str(row["policy_id"]), str(row["prompt_id"])
        if previous is not None and key != previous:
            if previous in seen:
                raise MinimumExperimentError(f"non-contiguous BoN group: {previous}")
            seen.add(previous)
            yield previous, rows
            rows = []
        previous = key
        rows.append(row)
    if previous is not None:
        if previous in seen:
            raise MinimumExperimentError(f"non-contiguous BoN group: {previous}")
        yield previous, rows


def score_bon(
    run_root: Path,
    score_endpoint: str,
    *,
    workers: int = 12,
    on_shard: Callable[..., None] | None = None,
    target_groups: Set[tuple[str, str]] | None = None,
    progress_filename: str = "progress.json",
    mode: ReplayMode | str = ReplayMode.DYNAMIC_FIXED_BUDGETED,
    judge_prompt_version: str = LEGACY_JUDGE_PROMPT_VERSION,
    routing_contract_path: Path | None = None,
) -> dict[str, Any]:
    """Score the shared BoN pool once over the union of R0 and current Rt criteria."""

    if Path(progress_filename).name != progress_filename:
        raise MinimumExperimentError("score progress filename must be a basename")
    targets = None if target_groups is None else frozenset(target_groups)
    if targets is not None and not targets:
        raise MinimumExperimentError("target score group set cannot be empty")
    parsed_mode = ReplayMode(mode)
    mode_value = parsed_mode.value
    stage_root = run_root / _minimum_score_stage(parsed_mode)
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    replay_path = run_root / _minimum_replay_stage(parsed_mode) / "replay_snapshots.jsonl"
    static_path = run_root / "generate-static" / "static_rubrics.jsonl"
    client = TargetScoreClient(score_endpoint, workers=workers)
    if judge_prompt_version not in {
        LEGACY_JUDGE_PROMPT_VERSION,
        PAPER_JUDGE_PROMPT_VERSION,
    }:
        raise MinimumExperimentError(f"unsupported judge prompt version: {judge_prompt_version}")
    conversations = (
        _audit_conversations(run_root) if judge_prompt_version == PAPER_JUDGE_PROMPT_VERSION else {}
    )
    routing_contract = routing_contract_path
    if routing_contract is None:
        routing_contract = run_root / "train-static" / "judge-routing-minimum-v3.json"
        if not routing_contract.is_file():
            routing_contract = run_root / "train-static" / "judge-routing-minimum-v2.json"
    if not routing_contract.is_file():
        raise MinimumExperimentError(f"missing judge routing contract: {routing_contract}")
    routing_contract = routing_contract.resolve()
    manifest = {
        "schema_version": 1,
        "comparison": ["static", mode_value],
        "focal_steps": list(FOCAL_STEPS),
        "pool_size": 1024,
        "shared_criterion_cache": True,
        "judge_prompt_version": judge_prompt_version,
        "score_identity": client.identity(),
        "score_routing": {
            "strategy": "deterministic-length-balanced-largest-first",
            "upstream_weights": list(client.routing_weights()),
            "contract_path": str(routing_contract),
            "contract_sha256": sha256_file(routing_contract),
        },
        "inputs": {str(path): sha256_file(path) for path in (bon_path, replay_path, static_path)},
    }
    write_json_atomic(stage_root / "manifest.json", manifest)
    static, dynamic = _focal_rubrics(run_root, parsed_mode)
    completed = 0
    total_pairs = 0
    expected_groups = len(static) * len(FOCAL_STEPS) if targets is None else len(targets)
    seen_targets: set[tuple[str, str]] = set()
    for group_index, ((policy_id, prompt_id), candidates) in enumerate(_bon_groups(bon_path), 1):
        group_key = policy_id, prompt_id
        if targets is not None and group_key not in targets:
            continue
        seen_targets.add(group_key)
        step = int(candidates[0]["policy_step"])
        if policy_id != f"pi_{step}" or step not in FOCAL_STEPS or len(candidates) != 1024:
            raise MinimumExperimentError(f"BoN group inventory mismatch: {(policy_id, prompt_id)}")
        shard_id = _shard_name(policy_id, prompt_id)
        score_shard = stage_root / "shards" / f"{policy_id}-{shard_id}.jsonl"
        criterion_shard = stage_root / "criterion-shards" / f"{policy_id}-{shard_id}.jsonl.gz"
        if score_shard.is_file() and criterion_shard.is_file():
            completed += 1
            if on_shard is not None:
                on_shard(policy_id, prompt_id, candidates)
            continue
        static_row = static[prompt_id]
        conversation = conversations.get(prompt_id)
        if judge_prompt_version == PAPER_JUDGE_PROMPT_VERSION and conversation is None:
            raise MinimumExperimentError(f"paper judge conversation is absent: {prompt_id}")
        dynamic_row = dynamic[(prompt_id, step)]
        criteria_by_id: dict[str, dict[str, Any]] = {}
        for criterion in (*static_row["criteria"], *dynamic_row["criteria"]):
            criterion_id = str(criterion["criterion_id"])
            previous = criteria_by_id.setdefault(criterion_id, dict(criterion))
            if previous["text"] != criterion["text"]:
                raise MinimumExperimentError(f"criterion ID collision: {prompt_id} {criterion_id}")
        tasks: list[tuple[tuple[str, str], str]] = []
        for candidate in candidates:
            response_id = str(candidate["response_id"])
            for criterion_id, criterion in criteria_by_id.items():
                tasks.append(
                    (
                        (criterion_id, response_id),
                        _score_prompt(
                            str(criterion["text"]),
                            str(candidate["response_text"]),
                            conversation=conversation,
                            prompt_version=judge_prompt_version,
                        ),
                    )
                )
        scores = client.score(tasks)
        total_pairs += len(scores)
        candidate_by_response = {str(row["response_id"]): row for row in candidates}

        def criterion_output() -> Iterator[dict[str, Any]]:
            for (criterion_id, response_id), score in sorted(scores.items()):
                yield {
                    "policy_id": policy_id,
                    "policy_step": step,
                    "prompt_id": prompt_id,
                    "response_id": response_id,
                    "global_candidate_id": candidate_by_response[response_id][
                        "global_candidate_id"
                    ],
                    "criterion_id": criterion_id,
                    **score,
                    "parse_success": True,
                }

        _publish_gzip_jsonl(criterion_shard, criterion_output())
        rubric_rows = []
        rubric_specs = (
            (
                str(static_row["rubric_id"]),
                "static",
                0,
                [str(item["criterion_id"]) for item in static_row["criteria"]],
            ),
            (
                f"{prompt_id}:{mode_value}:R_{step}",
                mode_value,
                step,
                [str(item["criterion_id"]) for item in dynamic_row["criteria"]],
            ),
        )
        for candidate in candidates:
            response_id = str(candidate["response_id"])
            for rubric_id, mode, rubric_step, criterion_ids in rubric_specs:
                values = [
                    scores[(criterion_id, response_id)]["probability_yes"]
                    for criterion_id in criterion_ids
                ]
                score = math.fsum(values) / len(values)
                rubric_rows.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": step,
                        "prompt_id": prompt_id,
                        "global_candidate_id": candidate["global_candidate_id"],
                        "response_id": response_id,
                        "rubric_id": rubric_id,
                        "mode": mode,
                        "rubric_step": rubric_step,
                        "criterion_count": len(criterion_ids),
                        "score": score,
                        "judge_repeat_score": score,
                        "judge_repeat_method": "deterministic_temperature_zero_cache_identity",
                    }
                )
        _publish_jsonl(score_shard, rubric_rows)
        completed += 1
        _write_mutable_json(
            stage_root / progress_filename,
            {
                "completed_prompt_policy_shards": completed,
                "expected_prompt_policy_shards": expected_groups,
                "last_policy_id": policy_id,
                "last_prompt_id": prompt_id,
                "source_group_index": group_index,
                "criterion_pairs_scored_this_process": total_pairs,
            },
        )
        if on_shard is not None:
            on_shard(policy_id, prompt_id, candidates)
    if targets is not None:
        if seen_targets != targets:
            missing = sorted(targets - seen_targets)
            raise MinimumExperimentError(f"target BoN groups are absent: {missing}")
        return {
            "targeted": True,
            "prompt_policy_shards": completed,
            "expected_prompt_policy_shards": expected_groups,
            "criterion_pairs_scored_this_process": total_pairs,
        }
    score_shards = sorted((stage_root / "shards").glob("*.jsonl"))
    criterion_shards = sorted((stage_root / "criterion-shards").glob("*.jsonl.gz"))
    if len(score_shards) != expected_groups or len(criterion_shards) != expected_groups:
        raise MinimumExperimentError("proxy score shard inventory is incomplete")
    output_path = stage_root / "rubric_scores.jsonl"
    _publish_jsonl(output_path, (row for path in score_shards for row in _jsonl(path)))
    rows = sum(1 for _ in _jsonl(output_path))
    result = {
        "mode": mode_value,
        "prompt_policy_shards": len(score_shards),
        "criterion_trace_shards": len(criterion_shards),
        "rubric_scores": rows,
        "expected_rubric_scores": 4 * 96 * 1024 * 2,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    if rows != result["expected_rubric_scores"]:
        raise MinimumExperimentError(f"rubric score inventory mismatch: {result}")
    write_json_atomic(stage_root / "result.json", result)
    return result


def select_bon_shard(
    run_root: Path,
    policy_id: str,
    prompt_id: str,
    candidates: Sequence[dict[str, Any]],
) -> Path:
    """Publish one selection shard as soon as its proxy score shard exists."""

    stage_root = run_root / "select-bon-minimum"
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    write_json_atomic(
        stage_root / "manifest.json",
        {
            "schema_version": 1,
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "tie_break": "lowest_global_candidate_id",
            "bon_sha256": sha256_file(bon_path),
            "score_manifest_sha256": sha256_file(
                run_root / "score-proxy-minimum" / "manifest.json"
            ),
        },
    )
    shard_id = _shard_name(policy_id, prompt_id)
    output_shard = stage_root / "shards" / f"{policy_id}-{shard_id}.jsonl"
    if output_shard.is_file():
        return output_shard
    score_path = run_root / "score-proxy-minimum" / "shards" / output_shard.name
    if not score_path.is_file():
        raise MinimumExperimentError(f"missing proxy score shard: {score_path}")
    scores = read_jsonl(score_path)
    candidate_by_id = {int(row["global_candidate_id"]): row for row in candidates}
    by_rubric: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        by_rubric[str(row["rubric_id"])].append(row)
    if len(by_rubric) != 2:
        raise MinimumExperimentError(f"expected static/current score pair: {policy_id} {prompt_id}")
    permutations = fixed_candidate_permutations(
        list(candidate_by_id),
        seed=f"pilot-static-r0-100step-20260821:{policy_id}:{prompt_id}",
        count=PERMUTATIONS,
    )
    pool_hash = hashlib.sha256(
        canonical_json_bytes(
            [
                [candidate_id, candidate_by_id[candidate_id]["response_text"]]
                for candidate_id in sorted(candidate_by_id)
            ]
        )
    ).hexdigest()
    selections = []
    for rubric_id, rubric_scores in sorted(by_rubric.items()):
        score_by_id = {
            int(row["global_candidate_id"]): float(row["score"]) for row in rubric_scores
        }
        sample = rubric_scores[0]
        if set(score_by_id) != set(candidate_by_id):
            raise MinimumExperimentError(f"candidate score inventory mismatch: {rubric_id}")
        for permutation_index, permutation in enumerate(permutations):
            for n in N_GRID:
                selected_id = int(select_best_of_n(score_by_id, permutation, n))
                selected = candidate_by_id[selected_id]
                selections.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": sample["policy_step"],
                        "prompt_id": prompt_id,
                        "rubric_id": rubric_id,
                        "mode": sample["mode"],
                        "rubric_step": sample["rubric_step"],
                        "n": n,
                        "permutation": permutation_index,
                        "pool_hash": pool_hash,
                        "global_candidate_id": selected_id,
                        "response_id": selected["response_id"],
                        "response_text": selected["response_text"],
                    }
                )
    _publish_jsonl(output_shard, selections)
    completed = len(list((stage_root / "shards").glob("*.jsonl")))
    _write_mutable_json(
        stage_root / "progress.json",
        {"completed_prompt_policy_shards": completed, "expected": 384},
    )
    return output_shard


def select_bon(run_root: Path) -> dict[str, Any]:
    """Select the full N-curve from identical candidate permutations for both rubrics."""

    stage_root = run_root / "select-bon-minimum"
    bon_path = run_root / "generate-bon" / "bon_pool.jsonl"
    score_root = run_root / "score-proxy-minimum" / "shards"
    write_json_atomic(
        stage_root / "manifest.json",
        {
            "schema_version": 1,
            "n_grid": list(N_GRID),
            "permutations": PERMUTATIONS,
            "tie_break": "lowest_global_candidate_id",
            "bon_sha256": sha256_file(bon_path),
            "score_manifest_sha256": sha256_file(
                run_root / "score-proxy-minimum" / "manifest.json"
            ),
        },
    )
    completed = 0
    for (policy_id, prompt_id), candidates in _bon_groups(bon_path):
        shard_id = _shard_name(policy_id, prompt_id)
        output_shard = stage_root / "shards" / f"{policy_id}-{shard_id}.jsonl"
        if output_shard.is_file():
            completed += 1
            continue
        score_path = score_root / f"{policy_id}-{shard_id}.jsonl"
        scores = read_jsonl(score_path)
        candidate_by_id = {int(row["global_candidate_id"]): row for row in candidates}
        by_rubric: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in scores:
            by_rubric[str(row["rubric_id"])].append(row)
        if len(by_rubric) != 2:
            raise MinimumExperimentError(
                f"expected static/current score pair: {policy_id} {prompt_id}"
            )
        permutations = fixed_candidate_permutations(
            list(candidate_by_id),
            seed=f"pilot-static-r0-100step-20260821:{policy_id}:{prompt_id}",
            count=PERMUTATIONS,
        )
        pool_hash = hashlib.sha256(
            canonical_json_bytes(
                [
                    [candidate_id, candidate_by_id[candidate_id]["response_text"]]
                    for candidate_id in sorted(candidate_by_id)
                ]
            )
        ).hexdigest()
        selections = []
        for rubric_id, rubric_scores in sorted(by_rubric.items()):
            score_by_id = {
                int(row["global_candidate_id"]): float(row["score"]) for row in rubric_scores
            }
            sample = rubric_scores[0]
            if set(score_by_id) != set(candidate_by_id):
                raise MinimumExperimentError(f"candidate score inventory mismatch: {rubric_id}")
            for permutation_index, permutation in enumerate(permutations):
                for n in N_GRID:
                    selected_id = int(select_best_of_n(score_by_id, permutation, n))
                    selected = candidate_by_id[selected_id]
                    selections.append(
                        {
                            "policy_id": policy_id,
                            "policy_step": sample["policy_step"],
                            "prompt_id": prompt_id,
                            "rubric_id": rubric_id,
                            "mode": sample["mode"],
                            "rubric_step": sample["rubric_step"],
                            "n": n,
                            "permutation": permutation_index,
                            "pool_hash": pool_hash,
                            "global_candidate_id": selected_id,
                            "response_id": selected["response_id"],
                            "response_text": selected["response_text"],
                        }
                    )
        _publish_jsonl(output_shard, selections)
        completed += 1
        _write_mutable_json(
            stage_root / "progress.json",
            {"completed_prompt_policy_shards": completed, "expected": 384},
        )
    shards = sorted((stage_root / "shards").glob("*.jsonl"))
    if len(shards) != 384:
        raise MinimumExperimentError("selection shard inventory is incomplete")
    output_path = stage_root / "selections.jsonl"
    _publish_jsonl(output_path, (row for path in shards for row in _jsonl(path)))
    count = sum(1 for _ in _jsonl(output_path))
    expected = 4 * 96 * 2 * len(N_GRID) * PERMUTATIONS
    if count != expected:
        raise MinimumExperimentError(f"selection inventory mismatch: {count} != {expected}")
    result = {
        "selections": count,
        "n_grid": list(N_GRID),
        "permutations": PERMUTATIONS,
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(stage_root / "result.json", result)
    return result


def export_audit_package(run_root: Path) -> dict[str, Any]:
    """Deduplicate selected responses before the private GPT-5 Batch audit."""

    stage_root = run_root / "export-audit-package-minimum"
    selections_path = run_root / "select-bon-minimum" / "selections.jsonl"
    unique: dict[tuple[str, str], dict[str, Any]] = {}
    selection_count = 0
    for row in _jsonl(selections_path):
        selection_count += 1
        key = str(row["prompt_id"]), str(row["response_id"])
        if key not in unique:
            text = str(row["response_text"])
            unique[key] = {
                "prompt_id": key[0],
                "response_id": key[1],
                "response_text": text,
                "response_text_hash": hashlib.sha256(text.encode()).hexdigest(),
                "selection_references": [],
            }
        unique[key]["selection_references"].append(
            {
                "policy_id": row["policy_id"],
                "mode": row["mode"],
                "rubric_step": row["rubric_step"],
                "n": row["n"],
                "permutation": row["permutation"],
            }
        )
    output_path = stage_root / "audit_package.jsonl"
    _publish_jsonl(output_path, (unique[key] for key in sorted(unique)))
    result = {
        "selection_rows": selection_count,
        "selected_unique_responses": len(unique),
        "output": str(output_path),
        "output_sha256": sha256_file(output_path),
    }
    write_json_atomic(stage_root / "result.json", result)
    return result
