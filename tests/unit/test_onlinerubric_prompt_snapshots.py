from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

from dynamic_rubric.prompt_versions.onlinerubric_grader_prompt import (
    ONLINERUBRIC_GRADER_SYSTEM_PROMPT,
    build_onlinerubric_grader_messages,
    onlinerubric_grader_schema,
)
from dynamic_rubric.prompt_versions.onlinerubric_prompt import (
    ONLINERUBRIC_DEDUP_SYSTEM_PROMPT,
    ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT,
    build_onlinerubric_dedup_messages,
    build_onlinerubric_extractor_messages,
)
from dynamic_rubric.training.online_contracts import WeightedCriterion
from dynamic_rubric.training.paper_reward import PaperRewardError, grade_and_compute


FIXTURES = Path(__file__).parents[1] / "fixtures" / "paper_online_rubrics"
EXPECTED_PROMPT_HASHES = {
    "figure8": "7258589ce47230e35ab2dcc02f808fe9157d8c5458c1876bce67d9a0b8098d4b",
    "figure9": "d258436acd99efe749faad233cbd8fff7a0ffb60b510653e74921c740c455b20",
    "figure10": "70d79be861b32e4d9d1e4f6eec76d4eb722488b3d2341010a66876eaaf6a482d",
}


def _load(name: str):
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_figure_system_prompt_hashes_are_version_locked() -> None:
    assert _sha256(ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT) == EXPECTED_PROMPT_HASHES["figure8"]
    assert _sha256(ONLINERUBRIC_DEDUP_SYSTEM_PROMPT) == EXPECTED_PROMPT_HASHES["figure9"]
    assert _sha256(ONLINERUBRIC_GRADER_SYSTEM_PROMPT) == EXPECTED_PROMPT_HASHES["figure10"]


def test_figure8_request_contains_only_blinded_pair_contract() -> None:
    fixture = _load("prompt_input.json")
    messages = build_onlinerubric_extractor_messages(
        prompt=fixture["prompt"],
        existing_rubric=fixture["existing_rubric"],
        response_a=fixture["response_a"],
        response_b=fixture["response_b"],
    )
    assert messages[0]["content"] == ONLINERUBRIC_EXTRACTOR_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Prompt:" in user and "Existing Rubric:" in user
    assert "Response A:" in user and "Response B:" in user
    assert fixture["response_a"] in user and fixture["response_b"] in user
    assert "current_actor" not in user and "control_actor" not in user


def test_figure9_request_aggregates_supplied_candidates_without_response_pair() -> None:
    fixture = _load("prompt_input.json")
    extraction = _load("extraction_valid.json")
    messages = build_onlinerubric_dedup_messages(
        prompt=fixture["prompt"],
        existing_rubric=fixture["existing_rubric"],
        candidate_criteria=extraction["new_criteria"],
    )
    assert messages[0]["content"] == ONLINERUBRIC_DEDUP_SYSTEM_PROMPT
    user = messages[1]["content"]
    assert "Candidate Criteria From Pairwise Comparisons:" in user
    assert extraction["new_criteria"][0]["criterion"] in user
    assert "Response A:" not in user and "Response B:" not in user


def test_figure10_request_and_schema_use_direct_numbered_labels() -> None:
    fixture = _load("prompt_input.json")
    criteria = fixture["existing_rubric"] + [
        {
            "criterion_id": "online-1",
            "criterion": "Explain the role of atmospheric scattering",
            "weight": 2,
        }
    ]
    messages = build_onlinerubric_grader_messages(
        prompt=fixture["prompt"], response=fixture["response_a"], criteria=criteria
    )
    user = messages[1]["content"]
    assert '"1": "Answer the question directly"' in user
    assert '"3": "Explain the role of atmospheric scattering"' in user
    schema = onlinerubric_grader_schema(3)
    assert schema["required"] == ["1", "2", "3"]
    assert schema["additionalProperties"] is False
    assert all(
        schema["properties"][key]["enum"] == ["PRESENT", "NOT_PRESENT"]
        for key in schema["required"]
    )


def test_signed_eq4_golden_fixture_and_malformed_label() -> None:
    fixture = _load("signed_eq4.json")
    criteria = tuple(
        WeightedCriterion(item["criterion_id"], item["criterion"], item["weight"], item["source"])
        for item in fixture["criteria"]
    )
    calculation = grade_and_compute(fixture["grader_output"], criteria)
    assert calculation.numerator == fixture["expected"]["numerator"]
    assert calculation.denominator == fixture["expected"]["denominator"]
    assert calculation.reward == Fraction(fixture["expected"]["reward"])
    malformed = _load("grader_malformed_label.json")
    with pytest.raises(PaperRewardError, match="exactly PRESENT"):
        grade_and_compute(malformed, criteria)
