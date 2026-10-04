"""Synchronous, fail-closed OnlineRubrics extraction/dedup/grading barrier."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from dynamic_rubric.artifacts import read_json, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.prompt_versions.onlinerubric_grader_prompt import (
    build_onlinerubric_grader_messages,
    onlinerubric_grader_schema,
)
from dynamic_rubric.prompt_versions.onlinerubric_prompt import (
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)
from dynamic_rubric.providers.base import GenerationRequest, GenerationResult, RubricGenerator
from dynamic_rubric.rubrics.extractor import make_blind_pairing
from dynamic_rubric.training.online_contracts import (
    ELICITATION_PAIR_COUNT,
    OnlineStepInput,
    OnlineStepManifest,
    RewardReceipt,
    RubricUnion,
    StepState,
    WeightedCriterion,
)
from dynamic_rubric.training.paper_reward import grade_and_compute


EXTRACTION_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis", "new_criteria"],
    "properties": {
        "analysis": {"type": "string", "maxLength": 4096},
        "new_criteria": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["quote", "criterion", "weight"],
                "properties": {
                    "quote": {"type": "string", "minLength": 1, "maxLength": 1024},
                    "criterion": {"type": "string", "minLength": 1, "maxLength": 512},
                    "weight": {"type": "integer", "minimum": 1},
                },
            },
        },
    },
}

DEDUP_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["analysis", "final_criteria"],
    "properties": {
        "analysis": {"type": "string", "maxLength": 4096},
        "final_criteria": {
            "type": "array",
            "maxItems": 16,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["criterion", "weight"],
                "properties": {
                    "criterion": {"type": "string", "minLength": 1, "maxLength": 512},
                    "weight": {"type": "integer", "minimum": 1},
                },
            },
        },
    },
}


def _dedup_repair_schema(
    candidates: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any]:
    """Constrain a provenance repair to exact candidate criterion/weight pairs."""

    canonical: dict[str, dict[str, Any]] = {}
    for item in candidates:
        text = str(item["criterion"]).strip()
        weight = item["weight"]
        normalized = " ".join(text.casefold().split())
        current = canonical.get(normalized)
        if current is None or weight > current["weight"]:
            canonical[normalized] = {"criterion": text, "weight": weight}
    allowed = list(canonical.values())
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["analysis", "final_criteria"],
        "properties": {
            "analysis": {"type": "string", "maxLength": 4096},
            "final_criteria": {
                "type": "array",
                "maxItems": min(16, len(canonical)),
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["criterion", "weight"],
                    "properties": {
                        "criterion": {
                            "type": "string",
                            "enum": [item["criterion"] for item in allowed],
                        },
                        "weight": {
                            "type": "integer",
                            "enum": sorted({item["weight"] for item in allowed}),
                        },
                    },
                },
            },
        },
    }


class OnlineStepError(RuntimeError):
    """A hard barrier failure; callers must not update the actor."""


@dataclass(frozen=True, slots=True)
class OnlineStepResult:
    manifest: OnlineStepManifest
    rubric_unions: tuple[RubricUnion, ...]
    rewards: tuple[RewardReceipt, ...]
    trace_refs: tuple[str, ...]
    sealed: bool = True

    @property
    def reward_scalars(self) -> tuple[float, ...]:
        return tuple(receipt.reward for receipt in self.rewards)


@dataclass(frozen=True, slots=True)
class OnlineHookResult:
    optimizer_update_index: int
    rm_scores: Any
    trace_refs: tuple[str, ...]
    manifest_hash: str
    sealed: bool


class OnlineRewardRuntime(Protocol):
    def prepare_online_step(self, batch: Any, *, step: int) -> OnlineHookResult: ...

    def apply_hook_result(self, batch: Any, result: OnlineHookResult) -> Any: ...

    def commit_step(self, *, global_step: int, checkpoint_dir: str, trainer: Any) -> None: ...


def _criterion_payload(criteria: Sequence[WeightedCriterion]) -> list[dict[str, Any]]:
    return [
        {
            "criterion_id": item.criterion_id,
            "criterion": item.text,
            "weight": item.weight,
            "source": item.source,
        }
        for item in criteria
    ]


def _strict_json(text: str, *, fields: set[str], label: str) -> Mapping[str, Any]:
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise OnlineStepError(f"{label} response is not valid JSON") from error
    if not isinstance(value, Mapping) or set(value) != fields:
        raise OnlineStepError(f"{label} response has an invalid top-level inventory")
    return value


def _parse_extraction(result: GenerationResult) -> tuple[Mapping[str, Any], ...]:
    value = _strict_json(result.text, fields={"analysis", "new_criteria"}, label="extractor")
    if not isinstance(value["analysis"], str) or not isinstance(value["new_criteria"], list):
        raise OnlineStepError("extractor response has invalid analysis or criteria")
    parsed: list[Mapping[str, Any]] = []
    for item in value["new_criteria"]:
        if not isinstance(item, Mapping) or set(item) != {"quote", "criterion", "weight"}:
            raise OnlineStepError("extractor criterion has an invalid field inventory")
        quote, criterion, weight = item["quote"], item["criterion"], item["weight"]
        if not isinstance(quote, str) or not quote.strip():
            raise OnlineStepError("extractor criterion must contain a response-grounded quote")
        if not isinstance(criterion, str) or not criterion.strip():
            raise OnlineStepError("extractor criterion text must be non-empty")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise OnlineStepError("extractor weight must be a positive integer")
        parsed.append(dict(item))
    return tuple(parsed)


_TOKEN_RE = re.compile(r"[\w]+", re.UNICODE)


def _normalized(text: str) -> str:
    return " ".join(text.casefold().split())


def _grounded_in_candidates(text: str, candidates: Sequence[Mapping[str, Any]]) -> bool:
    normalized = _normalized(text)
    final_tokens = set(_TOKEN_RE.findall(normalized))
    candidate_token_union: set[str] = set()
    for item in candidates:
        candidate = _normalized(str(item["criterion"]))
        if normalized == candidate or normalized in candidate or candidate in normalized:
            return True
        candidate_tokens = set(_TOKEN_RE.findall(candidate))
        candidate_token_union.update(candidate_tokens)
        union = final_tokens | candidate_tokens
        if union and len(final_tokens & candidate_tokens) / len(union) >= 0.5:
            return True
    return (
        bool(final_tokens) and len(final_tokens & candidate_token_union) / len(final_tokens) >= 0.5
    )


def _parse_dedup(
    result: GenerationResult,
    *,
    occurrence_id: str,
    offline: Sequence[WeightedCriterion],
    candidates: Sequence[Mapping[str, Any]],
) -> tuple[WeightedCriterion, ...]:
    value = _strict_json(result.text, fields={"analysis", "final_criteria"}, label="dedup")
    if not isinstance(value["analysis"], str) or not isinstance(value["final_criteria"], list):
        raise OnlineStepError("dedup response has invalid analysis or criteria")
    if value["final_criteria"] and not candidates:
        raise OnlineStepError("dedup introduced criteria when extraction returned none")
    offline_texts = {_normalized(item.text) for item in offline}
    seen: set[str] = set()
    criteria: list[WeightedCriterion] = []
    for index, item in enumerate(value["final_criteria"]):
        if not isinstance(item, Mapping) or set(item) != {"criterion", "weight"}:
            raise OnlineStepError("dedup criterion has an invalid field inventory")
        text, weight = item["criterion"], item["weight"]
        if not isinstance(text, str) or not text.strip():
            raise OnlineStepError("dedup criterion text must be non-empty")
        if isinstance(weight, bool) or not isinstance(weight, int) or weight <= 0:
            raise OnlineStepError("dedup weight must be a positive integer")
        normalized = _normalized(text)
        if normalized in offline_texts:
            raise OnlineStepError("dedup left a criterion that collides with the offline rubric")
        if normalized in seen:
            raise OnlineStepError("dedup output still contains duplicate criteria")
        if not _grounded_in_candidates(text, candidates):
            raise OnlineStepError("dedup introduced a criterion not grounded in candidates")
        seen.add(normalized)
        criterion_id = (
            "online-"
            + hashlib.sha256(f"{occurrence_id}\x1f{index}\x1f{normalized}".encode()).hexdigest()[
                :20
            ]
        )
        criteria.append(WeightedCriterion(criterion_id, text.strip(), weight, "online_pairwise"))
    return tuple(criteria)


def _parallel_generate(
    provider: RubricGenerator,
    requests: Sequence[GenerationRequest],
    *,
    max_concurrency: int,
) -> tuple[GenerationResult, ...]:
    if max_concurrency < 1:
        raise OnlineStepError("max_concurrency must be positive")
    try:
        with ThreadPoolExecutor(max_workers=min(max_concurrency, len(requests) or 1)) as pool:
            results = tuple(pool.map(provider.generate, requests))
    except BaseException as error:
        raise OnlineStepError("provider barrier failed; online step remains unsealed") from error
    if len(results) != len(requests):
        raise OnlineStepError("provider returned an incomplete result inventory")
    return results


def _validate_model_identity(
    results: Sequence[GenerationResult],
    *,
    expected_requested_model: str,
    expected_returned_model: str,
    family: str,
) -> None:
    for result in results:
        if result.requested_model != expected_requested_model:
            raise OnlineStepError(f"{family} requested-model identity mismatch")
        if result.returned_model != expected_returned_model:
            raise OnlineStepError(f"{family} returned-model identity mismatch")


def _receipt(result: GenerationResult, request: GenerationRequest) -> dict[str, Any]:
    request_identity = {
        "prompt_id": request.prompt_id,
        "family": request.family,
        "seed": request.seed,
        "temperature": request.temperature,
        "top_p": request.top_p,
        "max_output_tokens": request.max_output_tokens,
        "schema_name": request.schema_name,
        "reasoning_effort": request.reasoning_effort,
        "metadata": dict(request.metadata),
        "messages": [dict(message) for message in request.messages],
        "json_schema": request.json_schema,
    }
    return {
        "messages_hash": sha256_json(request_identity["messages"]),
        "schema_hash": sha256_json(request.json_schema),
        "request_hash": sha256_json(request_identity),
        "prompt_id": request.prompt_id,
        "family": request.family,
        "seed": request.seed,
        "metadata": dict(request.metadata),
        "requested_model": result.requested_model,
        "returned_model": result.returned_model,
        "request_id": result.request_id,
        "raw_response_hash": result.raw_response_hash or sha256_json(result.text),
        "result_text": result.text,
    }


class OnlineStepCoordinator:
    """Complete Algorithm 1's external-model work before exposing any reward."""

    def __init__(
        self,
        *,
        extractor: RubricGenerator,
        deduper: RubricGenerator,
        grader: RubricGenerator,
        extractor_model: str,
        grader_model: str,
        max_concurrency: int = 32,
        extractor_concurrency: int | None = None,
        grader_concurrency: int | None = None,
        extractor_reasoning_effort: str | None = None,
        extractor_max_output_tokens: int = 8192,
        dedup_max_output_tokens: int = 8192,
        grader_max_output_tokens: int = 4096,
        extractor_returned_model: str | None = None,
        grader_returned_model: str | None = None,
    ) -> None:
        if (
            min(
                extractor_max_output_tokens,
                dedup_max_output_tokens,
                grader_max_output_tokens,
            )
            <= 0
        ):
            raise ValueError("external-model output-token budgets must be positive")
        self.extractor = extractor
        self.deduper = deduper
        self.grader = grader
        self.extractor_model = extractor_model
        self.grader_model = grader_model
        self.extractor_returned_model = extractor_returned_model or extractor_model
        self.grader_returned_model = grader_returned_model or grader_model
        self.extractor_concurrency = extractor_concurrency or max_concurrency
        self.grader_concurrency = grader_concurrency or max_concurrency
        self.extractor_reasoning_effort = extractor_reasoning_effort
        self.extractor_max_output_tokens = extractor_max_output_tokens
        self.dedup_max_output_tokens = dedup_max_output_tokens
        self.grader_max_output_tokens = grader_max_output_tokens

    def run(
        self, step_input: OnlineStepInput, *, artifact_dir: Path | None = None
    ) -> OnlineStepResult:
        groups = step_input.prompt_groups
        extraction_requests: list[GenerationRequest] = []
        group_pairs: dict[str, Any] = {}
        for group in groups:
            occurrence = group.occurrence
            pairing = make_blind_pairing(
                [item.text for item in group.current_responses[:ELICITATION_PAIR_COUNT]],
                [item.text for item in group.control_responses],
                seed=step_input.seed,
                prompt_id=occurrence.prompt_occurrence_id,
                step=step_input.optimizer_update_index,
            )
            group_pairs[occurrence.prompt_occurrence_id] = pairing
            for pair_index, pair in enumerate(pairing.generator_payload()):
                extraction_requests.append(
                    GenerationRequest(
                        prompt_id=occurrence.prompt_id,
                        messages=build_onlinerubric_extractor_messages(
                            prompt=occurrence.prompt,
                            existing_rubric=_criterion_payload(group.offline_criteria),
                            response_a=pair.response_a,
                            response_b=pair.response_b,
                        ),
                        family="online_rubric_extraction",
                        seed=step_input.seed + pair_index,
                        max_output_tokens=self.extractor_max_output_tokens,
                        json_schema=EXTRACTION_SCHEMA,
                        schema_name="onlinerubric_extraction_v1",
                        reasoning_effort=self.extractor_reasoning_effort,
                        metadata={
                            "optimizer_update_index": step_input.optimizer_update_index,
                            "prompt_occurrence_id": occurrence.prompt_occurrence_id,
                            "pair_id": pair.pair_id,
                        },
                    )
                )
        if artifact_dir is not None:
            batch_path = artifact_dir / "batch.json"
            expected_batch_identity = {
                "run_id": step_input.run_id,
                "optimizer_update_index": step_input.optimizer_update_index,
                "batch_uid": step_input.batch_uid,
                "prompt_occurrence_ids": [
                    group.occurrence.prompt_occurrence_id for group in groups
                ],
            }
            if batch_path.is_file():
                batch_record = read_json(batch_path)
                if any(
                    batch_record.get(key) != value for key, value in expected_batch_identity.items()
                ):
                    raise OnlineStepError("existing batch artifact identity mismatch")
            else:
                write_json_atomic(
                    batch_path,
                    {
                        "schema_version": 1,
                        **expected_batch_identity,
                        "step_input": dataclasses.asdict(step_input),
                    },
                )
            write_jsonl_atomic(
                artifact_dir / "current_responses.jsonl",
                (
                    dataclasses.asdict(response)
                    for group in groups
                    for response in group.current_responses
                ),
            )
            write_jsonl_atomic(
                artifact_dir / "control_responses.jsonl",
                (
                    dataclasses.asdict(response)
                    for group in groups
                    for response in group.control_responses
                ),
            )
            write_jsonl_atomic(
                artifact_dir / "blind_pairs.jsonl",
                (
                    {
                        "prompt_occurrence_id": occurrence_id,
                        "blind_pair": dataclasses.asdict(pair),
                        "assignment": dataclasses.asdict(assignment),
                    }
                    for occurrence_id, plan in group_pairs.items()
                    for pair, assignment in zip(plan.blind_pairs, plan.assignments)
                ),
            )
        extraction_results = _parallel_generate(
            self.extractor, extraction_requests, max_concurrency=self.extractor_concurrency
        )
        _validate_model_identity(
            extraction_results,
            expected_requested_model=self.extractor_model,
            expected_returned_model=self.extractor_returned_model,
            family="extractor",
        )
        parsed_extractions = [_parse_extraction(result) for result in extraction_results]
        if artifact_dir is not None:
            write_jsonl_atomic(
                artifact_dir / "extraction_receipts.jsonl",
                [
                    _receipt(result, request)
                    for request, result in zip(extraction_requests, extraction_results)
                ],
            )

        dedup_requests: list[GenerationRequest] = []
        candidates_by_occurrence: dict[str, tuple[Mapping[str, Any], ...]] = {}
        offset = 0
        for group in groups:
            occurrence_id = group.occurrence.prompt_occurrence_id
            candidates = tuple(
                candidate
                for parsed in parsed_extractions[offset : offset + ELICITATION_PAIR_COUNT]
                for candidate in parsed
            )
            offset += ELICITATION_PAIR_COUNT
            candidates_by_occurrence[occurrence_id] = candidates
            dedup_requests.append(
                GenerationRequest(
                    prompt_id=group.occurrence.prompt_id,
                    messages=build_onlinerubric_dedup_messages(
                        prompt=group.occurrence.prompt,
                        existing_rubric=_criterion_payload(group.offline_criteria),
                        candidate_criteria=candidates,
                        exclude_existing=True,
                    ),
                    family="online_rubric_dedup",
                    seed=step_input.seed,
                    max_output_tokens=self.dedup_max_output_tokens,
                    json_schema=DEDUP_SCHEMA,
                    schema_name="onlinerubric_dedup_v1",
                    reasoning_effort=self.extractor_reasoning_effort,
                    metadata={
                        "optimizer_update_index": step_input.optimizer_update_index,
                        "prompt_occurrence_id": occurrence_id,
                    },
                )
            )
        initial_dedup_results = _parallel_generate(
            self.deduper, dedup_requests, max_concurrency=self.extractor_concurrency
        )
        _validate_model_identity(
            initial_dedup_results,
            expected_requested_model=self.extractor_model,
            expected_returned_model=self.extractor_returned_model,
            family="dedup",
        )
        dedup_results = list(initial_dedup_results)
        final_dedup_requests = list(dedup_requests)
        parsed_online: list[tuple[WeightedCriterion, ...] | None] = []
        repair_indexes: list[int] = []
        repair_errors: dict[int, str] = {}
        for index, (group, result) in enumerate(zip(groups, dedup_results)):
            occurrence_id = group.occurrence.prompt_occurrence_id
            try:
                online = _parse_dedup(
                    result,
                    occurrence_id=occurrence_id,
                    offline=group.offline_criteria,
                    candidates=candidates_by_occurrence[occurrence_id],
                )
            except OnlineStepError as error:
                if str(error) != "dedup introduced a criterion not grounded in candidates":
                    raise
                parsed_online.append(None)
                repair_indexes.append(index)
                repair_errors[index] = str(error)
            else:
                parsed_online.append(online)

        repair_receipts: list[dict[str, Any]] = []
        if repair_indexes:
            repair_requests: list[GenerationRequest] = []
            for index in repair_indexes:
                group = groups[index]
                occurrence_id = group.occurrence.prompt_occurrence_id
                candidates = candidates_by_occurrence[occurrence_id]
                repair_requests.append(
                    GenerationRequest(
                        prompt_id=group.occurrence.prompt_id,
                        messages=build_onlinerubric_dedup_messages(
                            prompt=group.occurrence.prompt,
                            existing_rubric=_criterion_payload(group.offline_criteria),
                            candidate_criteria=candidates,
                            exclude_existing=True,
                            deterministic_provenance_repair=True,
                        ),
                        family="online_rubric_dedup",
                        seed=step_input.seed,
                        max_output_tokens=self.dedup_max_output_tokens,
                        json_schema=_dedup_repair_schema(candidates),
                        schema_name="onlinerubric_dedup_repair_v1",
                        reasoning_effort=self.extractor_reasoning_effort,
                        metadata={
                            "optimizer_update_index": step_input.optimizer_update_index,
                            "prompt_occurrence_id": occurrence_id,
                            "dedup_attempt": 1,
                            "repair_reason": "candidate_grounding",
                        },
                    )
                )
            repaired_results = _parallel_generate(
                self.deduper,
                repair_requests,
                max_concurrency=self.extractor_concurrency,
            )
            _validate_model_identity(
                repaired_results,
                expected_requested_model=self.extractor_model,
                expected_returned_model=self.extractor_returned_model,
                family="dedup repair",
            )
            for index, request, repaired in zip(repair_indexes, repair_requests, repaired_results):
                group = groups[index]
                occurrence_id = group.occurrence.prompt_occurrence_id
                parsed_online[index] = _parse_dedup(
                    repaired,
                    occurrence_id=occurrence_id,
                    offline=group.offline_criteria,
                    candidates=candidates_by_occurrence[occurrence_id],
                )
                repair_receipts.append(
                    {
                        "prompt_occurrence_id": occurrence_id,
                        "validation_error": repair_errors[index],
                        "rejected": _receipt(initial_dedup_results[index], dedup_requests[index]),
                        "accepted": _receipt(repaired, request),
                    }
                )
                dedup_results[index] = repaired
                final_dedup_requests[index] = request

        unions = [
            RubricUnion(
                group.occurrence.prompt_occurrence_id,
                group.offline_criteria,
                online,
            )
            for group, online in zip(groups, parsed_online)
            if online is not None
        ]
        if len(unions) != len(groups):
            raise OnlineStepError("dedup repair left an incomplete rubric inventory")
        if artifact_dir is not None:
            write_jsonl_atomic(
                artifact_dir / "dedup_receipts.jsonl",
                [
                    _receipt(result, request)
                    for request, result in zip(final_dedup_requests, dedup_results)
                ],
            )
            if repair_receipts:
                write_jsonl_atomic(artifact_dir / "dedup_repair_receipts.jsonl", repair_receipts)
            write_jsonl_atomic(
                artifact_dir / "rubric_unions.jsonl",
                [dataclasses.asdict(item) | {"content_hash": item.content_hash} for item in unions],
            )

        grader_requests: list[GenerationRequest] = []
        grader_bindings: list[tuple[RubricUnion, Any]] = []
        for group, union in zip(groups, unions):
            payload = _criterion_payload(union.criteria)
            schema = onlinerubric_grader_schema(len(union.criteria))
            for response in group.current_responses:
                grader_requests.append(
                    GenerationRequest(
                        prompt_id=group.occurrence.prompt_id,
                        messages=build_onlinerubric_grader_messages(
                            prompt=group.occurrence.prompt,
                            response=response.text,
                            criteria=payload,
                        ),
                        family="online_rubric_grading",
                        seed=step_input.seed + response.rollout_index,
                        max_output_tokens=self.grader_max_output_tokens,
                        json_schema=schema,
                        schema_name="onlinerubric_grader_v1",
                        metadata={
                            "optimizer_update_index": step_input.optimizer_update_index,
                            "prompt_occurrence_id": group.occurrence.prompt_occurrence_id,
                            "response_id": response.response_id,
                            "rollout_index": response.rollout_index,
                            "rubric_hash": union.content_hash,
                        },
                    )
                )
                grader_bindings.append((union, response))
        grader_results = _parallel_generate(
            self.grader, grader_requests, max_concurrency=self.grader_concurrency
        )
        _validate_model_identity(
            grader_results,
            expected_requested_model=self.grader_model,
            expected_returned_model=self.grader_returned_model,
            family="grader",
        )
        rewards: list[RewardReceipt] = []
        for (union, response), result in zip(grader_bindings, grader_results):
            calculation = grade_and_compute(result.text, union.criteria)
            numerator = (
                calculation.numerator.numerator
                if calculation.numerator.denominator == 1
                else float(calculation.numerator)
            )
            denominator = (
                calculation.denominator.numerator
                if calculation.denominator.denominator == 1
                else float(calculation.denominator)
            )
            rewards.append(
                RewardReceipt(
                    prompt_occurrence_id=response.prompt_occurrence_id,
                    response_id=response.response_id,
                    rollout_index=response.rollout_index,
                    grades=calculation.grades,
                    numerator=numerator,
                    denominator=denominator,
                    reward=calculation.scalar,
                    rubric_hash=union.content_hash,
                )
            )
        if artifact_dir is not None:
            write_jsonl_atomic(
                artifact_dir / "grader_receipts.jsonl",
                [
                    _receipt(result, request)
                    for request, result in zip(grader_requests, grader_results)
                ],
            )
            write_jsonl_atomic(artifact_dir / "rewards.jsonl", map(dataclasses.asdict, rewards))

        artifact_hashes: dict[str, str] = {}
        if artifact_dir is not None:
            artifact_names = [
                "batch.json",
                "current_responses.jsonl",
                "control_responses.jsonl",
                "blind_pairs.jsonl",
                "extraction_receipts.jsonl",
                "dedup_receipts.jsonl",
                "rubric_unions.jsonl",
                "grader_receipts.jsonl",
                "rewards.jsonl",
            ]
            if (artifact_dir / "control_generation_receipts.jsonl").is_file():
                artifact_names.append("control_generation_receipts.jsonl")
            if (artifact_dir / "dedup_repair_receipts.jsonl").is_file():
                artifact_names.append("dedup_repair_receipts.jsonl")
            for name in artifact_names:
                artifact_hashes[name] = sha256_file(artifact_dir / name)
        manifest = OnlineStepManifest(
            schema_version=1,
            run_id=step_input.run_id,
            optimizer_update_index=step_input.optimizer_update_index,
            batch_uid=step_input.batch_uid,
            state=StepState.PRE_UPDATE_SEALED,
            prompt_occurrence_ids=tuple(group.occurrence.prompt_occurrence_id for group in groups),
            current_response_count=len(groups) * 16,
            control_response_count=len(groups) * 8,
            extraction_count=len(extraction_results),
            dedup_count=len(dedup_results),
            grader_count=len(grader_results),
            reward_count=len(rewards),
            artifacts=artifact_hashes,
        )
        if artifact_dir is not None:
            write_json_atomic(artifact_dir / "pre_update_seal.json", dataclasses.asdict(manifest))
        trace_refs = tuple(
            f"{receipt.prompt_occurrence_id}:{receipt.response_id}:{receipt.rubric_hash}"
            for receipt in rewards
        )
        return OnlineStepResult(manifest, tuple(unions), tuple(rewards), trace_refs)


def prepare_online_rewards(
    batch: Any, *, step: int, runtime: OnlineRewardRuntime
) -> OnlineHookResult:
    """Narrow veRL-independent adapter; returns rewards only after a complete seal."""

    result = runtime.prepare_online_step(batch, step=step)
    if not isinstance(result, OnlineHookResult):
        raise OnlineStepError("online runtime returned an invalid hook result")
    if not result.sealed or result.optimizer_update_index != step:
        raise OnlineStepError("online rewards are not sealed for the requested optimizer step")
    if not result.trace_refs or result.rm_scores is None or not result.manifest_hash:
        raise OnlineStepError("sealed hook result has an incomplete reward/trace inventory")
    return result


def _runtime_from_trainer(trainer: Any) -> OnlineRewardRuntime:
    runtime = getattr(trainer, "online_reward_runtime", None)
    if runtime is None:
        runtime = getattr(trainer, "_online_reward_runtime", None)
    if runtime is None:
        raise OnlineStepError("trainer has no configured online reward runtime")
    return runtime


def prepare_rewards(*, batch: Any, global_step: int, trainer: Any) -> Any:
    """veRL project-hook facade expected by the pinned trainer patch."""

    runtime = _runtime_from_trainer(trainer)
    result = prepare_online_rewards(batch, step=global_step, runtime=runtime)
    updated = runtime.apply_hook_result(batch, result)
    if updated is None:
        raise OnlineStepError("online runtime did not return a reward-bearing batch")
    return updated


def commit_step(*, global_step: int, checkpoint_dir: str, trainer: Any) -> None:
    """Delegate post-checkpoint commit; absence/failure is intentionally fatal."""

    runtime = _runtime_from_trainer(trainer)
    runtime.commit_step(global_step=global_step, checkpoint_dir=checkpoint_dir, trainer=trainer)


def build_runtime(*, trainer: Any, hook_config: Mapping[str, Any]) -> OnlineRewardRuntime:
    """Construct the concrete runtime named by the trainer hook config."""

    import importlib

    target = str(hook_config.get("runtime_factory", ""))
    if ":" not in target:
        raise OnlineStepError("hook_config.runtime_factory must be module.path:symbol")
    module_name, symbol_name = target.split(":", 1)
    if module_name == __name__ and symbol_name == "build_runtime":
        raise OnlineStepError("runtime factory must resolve to a concrete runtime builder")
    factory = getattr(importlib.import_module(module_name), symbol_name, None)
    if not callable(factory):
        raise OnlineStepError(f"runtime factory is not callable: {target}")
    runtime = factory(trainer=trainer, hook_config=hook_config)
    required = ("prepare_online_step", "apply_hook_result", "commit_step")
    if any(not callable(getattr(runtime, name, None)) for name in required):
        raise OnlineStepError("runtime factory returned an incomplete runtime")
    return runtime
