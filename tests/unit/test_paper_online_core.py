from __future__ import annotations

import json
import threading
import urllib.request
from fractions import Fraction
from pathlib import Path

import pytest

from dynamic_rubric.prompt_versions.onlinerubric_grader_prompt import (
    build_onlinerubric_grader_messages,
    onlinerubric_grader_schema,
)
from dynamic_rubric.providers.base import GenerationRequest, GenerationResult
from dynamic_rubric.providers.openai_responses import OpenAIResponsesAdapter
from dynamic_rubric.training.online_contracts import (
    OnlineContractError,
    OnlineStepInput,
    PolicySnapshot,
    PromptGroupInput,
    PromptOccurrence,
    ResponseRecord,
    StepState,
    WeightedCriterion,
    validate_online_step_manifest,
)
from dynamic_rubric.training.online_step import (
    DEDUP_SCHEMA,
    EXTRACTION_SCHEMA,
    OnlineStepCoordinator,
    OnlineStepError,
    _dedup_repair_schema,
    _grounded_in_candidates,
)
from dynamic_rubric.training.paper_reward import (
    PaperRewardError,
    compute_paper_reward,
    parse_binary_grades,
)


def test_online_structured_schemas_bound_free_form_generation() -> None:
    extraction = EXTRACTION_SCHEMA["properties"]
    extraction_item = extraction["new_criteria"]["items"]["properties"]
    dedup = DEDUP_SCHEMA["properties"]
    dedup_item = dedup["final_criteria"]["items"]["properties"]

    assert extraction["analysis"]["maxLength"] == 4096
    assert extraction["new_criteria"]["maxItems"] == 16
    assert extraction_item["quote"]["maxLength"] == 1024
    assert extraction_item["criterion"]["maxLength"] == 512
    assert dedup["analysis"]["maxLength"] == 4096
    assert dedup["final_criteria"]["maxItems"] == 16
    assert dedup_item["criterion"]["maxLength"] == 512


def test_grounding_accepts_criterion_merged_from_multiple_candidates() -> None:
    candidates = (
        {"criterion": "Recommend serum beta hCG measurement."},
        {"criterion": "Recommend chest X-ray or CT imaging."},
    )

    assert _grounded_in_candidates(
        "Recommend serum beta hCG measurement and chest X-ray or CT imaging.",
        candidates,
    )


def test_grounding_rejects_unrelated_invented_criterion() -> None:
    candidates = (
        {"criterion": "Recommend serum beta hCG measurement."},
        {"criterion": "Recommend chest X-ray or CT imaging."},
    )

    assert not _grounded_in_candidates(
        "Recommend emergency splenectomy for refractory hemorrhage.",
        candidates,
    )


def test_dedup_repair_schema_allows_only_exact_candidate_pairs() -> None:
    candidates = (
        {"criterion": "Criterion A", "weight": 2},
        {"criterion": "Criterion A", "weight": 4},
        {"criterion": "Criterion B", "weight": 3},
    )

    schema = _dedup_repair_schema(candidates)
    final = schema["properties"]["final_criteria"]

    assert final["maxItems"] == 2
    properties = final["items"]["properties"]
    assert properties["criterion"]["enum"] == ["Criterion A", "Criterion B"]
    assert properties["weight"]["enum"] == [3, 4]


def test_figure10_prompt_schema_and_strict_labels() -> None:
    criteria = (
        WeightedCriterion("positive", "Be correct", 2, "offline"),
        WeightedCriterion("negative", "Contain a hack", -1, "offline"),
    )
    messages = build_onlinerubric_grader_messages(
        prompt=({"role": "user", "content": "question"},),
        response="answer",
        criteria=[{"criterion_id": item.criterion_id, "criterion": item.text} for item in criteria],
    )
    assert '"1": "Be correct"' in messages[1]["content"]
    schema = onlinerubric_grader_schema(2)
    assert schema["required"] == ["1", "2"]
    grades = parse_binary_grades({"1": "PRESENT", "2": "NOT_PRESENT"}, criteria)
    assert grades == (("positive", 1), ("negative", 0))
    with pytest.raises(PaperRewardError, match="exactly PRESENT"):
        parse_binary_grades({"1": "PRESENT", "2": "NOT PRESENT"}, criteria)
    with pytest.raises(PaperRewardError, match="inventory mismatch"):
        parse_binary_grades({"1": "PRESENT", "2": "NOT_PRESENT", "3": "PRESENT"}, criteria)


def test_eq4_signed_and_zero_weights_are_exact() -> None:
    criteria = (
        WeightedCriterion("p", "positive", 3, "offline"),
        WeightedCriterion("n", "negative", -2, "offline"),
        WeightedCriterion("z", "zero", 0, "offline"),
    )
    calculation = compute_paper_reward(criteria, {"p": 1, "n": 1, "z": 1})
    assert calculation.numerator == Fraction(1)
    assert calculation.denominator == Fraction(3)
    assert calculation.reward == Fraction(1, 3)
    with pytest.raises(PaperRewardError, match="denominator"):
        compute_paper_reward((WeightedCriterion("n", "negative", -1, "offline"),), {"n": 1})


def _responses(
    occurrence_id: str, family: str, count: int, policy: PolicySnapshot
) -> tuple[ResponseRecord, ...]:
    return tuple(
        ResponseRecord(
            occurrence_id, f"{family}-{index}", index, f"{family} text {index}", policy, family
        )
        for index in range(count)
    )


def _step_input() -> OnlineStepInput:
    current = PolicySnapshot(0, "actor-hash", "actor", "rev")
    control = PolicySnapshot(0, "control-hash", "actor", "rev")
    occurrence = PromptOccurrence(
        "run",
        1,
        "batch",
        "row",
        "prompt",
        "occurrence",
        ({"role": "user", "content": "question"},),
    )
    group = PromptGroupInput(
        occurrence,
        (
            WeightedCriterion("base-positive", "Be correct", 2, "offline"),
            WeightedCriterion("base-negative", "Use reward hacking", -1, "offline"),
            WeightedCriterion("base-zero", "Neutral audit marker", 0, "offline"),
        ),
        _responses("occurrence", "current", 16, current),
        _responses("occurrence", "control", 8, control),
    )
    return OnlineStepInput("run", 1, "batch", 9, (group,))


class _ScriptedProvider:
    def __init__(
        self, model: str, *, fail_family: str | None = None, returned_model: str | None = None
    ) -> None:
        self.model = model
        self.returned_model = returned_model or model
        self.fail_family = fail_family
        self.calls: list[str] = []
        self.requests: list[GenerationRequest] = []
        self._lock = threading.Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            self.calls.append(request.family)
            self.requests.append(request)
        if request.family == self.fail_family:
            raise RuntimeError("injected provider failure")
        if request.family == "online_rubric_extraction":
            payload = {
                "analysis": "difference",
                "new_criteria": [
                    {"quote": "grounded quote", "criterion": "Explain the reasoning", "weight": 2}
                ],
            }
        elif request.family == "online_rubric_dedup":
            payload = {
                "analysis": "duplicates merged",
                "final_criteria": [{"criterion": "Explain the reasoning", "weight": 2}],
            }
        else:
            required = request.json_schema["required"]
            payload = {key: "PRESENT" for key in required}
        text = json.dumps(payload)
        return GenerationResult(text, self.model, self.returned_model, "request", 1, 0, {}, None)


class _RepairingProvider(_ScriptedProvider):
    def generate(self, request: GenerationRequest) -> GenerationResult:
        if (
            request.family == "online_rubric_dedup"
            and request.schema_name == "onlinerubric_dedup_v1"
        ):
            with self._lock:
                self.calls.append(request.family)
                self.requests.append(request)
            payload = {
                "analysis": "untraceable paraphrase",
                "final_criteria": [{"criterion": "Invented unrelated criterion", "weight": 2}],
            }
            return GenerationResult(
                json.dumps(payload), self.model, self.returned_model, "request", 1, 0, {}, None
            )
        return super().generate(request)


def test_online_coordinator_seals_exact_call_inventory_and_artifacts(tmp_path: Path) -> None:
    extractor = _ScriptedProvider("o3-mini")
    grader = _ScriptedProvider("gpt-4.1-mini")
    result = OnlineStepCoordinator(
        extractor=extractor,
        deduper=extractor,
        grader=grader,
        extractor_model="o3-mini",
        grader_model="gpt-4.1-mini",
        max_concurrency=4,
        extractor_concurrency=2,
        grader_concurrency=3,
        extractor_reasoning_effort="medium",
    ).run(_step_input(), artifact_dir=tmp_path)
    assert extractor.calls.count("online_rubric_extraction") == 8
    assert extractor.calls.count("online_rubric_dedup") == 1
    assert grader.calls == ["online_rubric_grading"] * 16
    assert all(request.reasoning_effort == "medium" for request in extractor.requests)
    assert all(request.reasoning_effort is None for request in grader.requests)
    assert {
        request.max_output_tokens
        for request in extractor.requests
        if request.family == "online_rubric_extraction"
    } == {8192}
    assert {
        request.max_output_tokens
        for request in extractor.requests
        if request.family == "online_rubric_dedup"
    } == {8192}
    assert {request.max_output_tokens for request in grader.requests} == {4096}
    assert len(result.rewards) == 16
    assert result.manifest.state is StepState.PRE_UPDATE_SEALED
    manifest = validate_online_step_manifest(tmp_path / "pre_update_seal.json")
    assert manifest.content_hash == result.manifest.content_hash
    assert set(manifest.artifacts) == {
        "batch.json",
        "current_responses.jsonl",
        "control_responses.jsonl",
        "blind_pairs.jsonl",
        "extraction_receipts.jsonl",
        "dedup_receipts.jsonl",
        "rubric_unions.jsonl",
        "grader_receipts.jsonl",
        "rewards.jsonl",
    }
    current_rows = [
        json.loads(line) for line in (tmp_path / "current_responses.jsonl").read_text().splitlines()
    ]
    control_rows = [
        json.loads(line) for line in (tmp_path / "control_responses.jsonl").read_text().splitlines()
    ]
    blind_rows = [
        json.loads(line) for line in (tmp_path / "blind_pairs.jsonl").read_text().splitlines()
    ]
    assert len(current_rows) == 16 and len(control_rows) == 8 and len(blind_rows) == 8
    assert all(
        set(row["blind_pair"]) == {"pair_id", "response_a", "response_b"} for row in blind_rows
    )
    assert all("current_label" not in row["blind_pair"] for row in blind_rows)
    extraction_receipt = json.loads(
        (tmp_path / "extraction_receipts.jsonl").read_text().splitlines()[0]
    )
    for hash_name in ("messages_hash", "schema_hash", "request_hash"):
        assert len(extraction_receipt[hash_name]) == 64


def test_online_coordinator_repairs_ungrounded_dedup_with_audited_constraint(
    tmp_path: Path,
) -> None:
    extractor = _RepairingProvider("o3-mini")
    grader = _ScriptedProvider("gpt-4.1-mini")

    result = OnlineStepCoordinator(
        extractor=extractor,
        deduper=extractor,
        grader=grader,
        extractor_model="o3-mini",
        grader_model="gpt-4.1-mini",
    ).run(_step_input(), artifact_dir=tmp_path)

    assert result.sealed is True
    assert extractor.calls.count("online_rubric_dedup") == 2
    repairs = [
        json.loads(line)
        for line in (tmp_path / "dedup_repair_receipts.jsonl").read_text().splitlines()
    ]
    assert len(repairs) == 1
    assert repairs[0]["validation_error"] == (
        "dedup introduced a criterion not grounded in candidates"
    )
    assert repairs[0]["accepted"]["metadata"]["dedup_attempt"] == 1
    assert "dedup_repair_receipts.jsonl" in result.manifest.artifacts
    final_receipt = json.loads((tmp_path / "dedup_receipts.jsonl").read_text().splitlines()[0])
    assert final_receipt["metadata"]["dedup_attempt"] == 1
    repair_request = next(
        request
        for request in extractor.requests
        if request.schema_name == "onlinerubric_dedup_repair_v1"
    )
    assert "Do not paraphrase" in repair_request.messages[0]["content"]
    repair_properties = repair_request.json_schema["properties"]["final_criteria"]["items"][
        "properties"
    ]
    assert repair_properties["criterion"]["enum"] == ["Explain the reasoning"]
    assert repair_properties["weight"]["enum"] == [2]


def test_online_coordinator_fails_closed_before_seal(tmp_path: Path) -> None:
    extractor = _ScriptedProvider("o3-mini", fail_family="online_rubric_dedup")
    grader = _ScriptedProvider("gpt-4.1-mini")
    coordinator = OnlineStepCoordinator(
        extractor=extractor,
        deduper=extractor,
        grader=grader,
        extractor_model="o3-mini",
        grader_model="gpt-4.1-mini",
    )
    with pytest.raises(OnlineStepError, match="barrier failed"):
        coordinator.run(_step_input(), artifact_dir=tmp_path)
    assert not (tmp_path / "pre_update_seal.json").exists()
    assert not grader.calls


def test_inventory_rejects_noncanonical_rollout_indexes() -> None:
    step = _step_input()
    group = step.prompt_groups[0]
    malformed = list(group.current_responses)
    malformed[-1] = ResponseRecord(
        "occurrence", "current-x", 17, "text", malformed[-1].policy, "current"
    )
    with pytest.raises(OnlineContractError, match="canonical"):
        PromptGroupInput(
            group.occurrence,
            group.offline_criteria,
            tuple(malformed),
            group.control_responses,
        )


class _HTTPResponse:
    headers = {"x-request-id": "request"}

    def __init__(self, output: str) -> None:
        self.output = output

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {
                "id": "request",
                "model": "snapshot",
                "status": "completed",
                "output_text": self.output,
                "created": 1,
            }
        ).encode()


def test_openai_generate_many_preserves_input_order(tmp_path: Path, monkeypatch) -> None:
    def urlopen(request, **kwargs):
        del kwargs
        payload = json.loads(request.data)
        return _HTTPResponse(payload["input"][0]["content"])

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    adapter = OpenAIResponsesAdapter("secret", "model", tmp_path, max_retries=0)
    requests = tuple(
        GenerationRequest(str(index), ({"role": "user", "content": str(index)},), "family", index)
        for index in range(8)
    )
    assert [result.text for result in adapter.generate_many(requests, max_concurrency=3)] == [
        str(index) for index in range(8)
    ]
