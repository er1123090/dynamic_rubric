"""Strict Figure-10 grade parsing and signed-weight OnlineRubrics Eq. 4."""

from __future__ import annotations

import json
from dataclasses import dataclass
from fractions import Fraction
from typing import Any, Mapping, Sequence

from .online_contracts import OnlineContractError, WeightedCriterion


class PaperRewardError(OnlineContractError):
    pass


@dataclass(frozen=True, slots=True)
class PaperRewardCalculation:
    grades: tuple[tuple[str, int], ...]
    numerator: Fraction
    denominator: Fraction
    reward: Fraction

    @property
    def scalar(self) -> float:
        return float(self.reward)


def parse_binary_grades(
    payload: str | Mapping[str, Any], criteria: Sequence[WeightedCriterion]
) -> tuple[tuple[str, int], ...]:
    """Parse Figure 10's direct numbered PRESENT/NOT_PRESENT object."""

    try:
        value = json.loads(payload) if isinstance(payload, str) else dict(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise PaperRewardError("grader output is not a JSON object") from error
    expected_keys = [str(index) for index in range(1, len(criteria) + 1)]
    if set(value) != set(expected_keys):
        missing = sorted(set(expected_keys) - set(value))
        extra = sorted(set(value) - set(expected_keys))
        raise PaperRewardError(
            f"grader rubric inventory mismatch: missing={missing}, extra={extra}"
        )
    label_to_grade = {"PRESENT": 1, "NOT_PRESENT": 0}
    grades: list[tuple[str, int]] = []
    for key, criterion in zip(expected_keys, criteria):
        label = value[key]
        if not isinstance(label, str) or label not in label_to_grade:
            raise PaperRewardError("grader labels must be exactly PRESENT or NOT_PRESENT")
        grades.append((criterion.criterion_id, label_to_grade[label]))
    return tuple(grades)


def compute_paper_reward(
    criteria: Sequence[WeightedCriterion],
    grades: Sequence[tuple[str, int]] | Mapping[str, int],
) -> PaperRewardCalculation:
    """Compute Eq. 4 exactly: signed numerator, positive-weight denominator."""

    if not criteria:
        raise PaperRewardError("rubric must not be empty")
    ordered = tuple(grades.items()) if isinstance(grades, Mapping) else tuple(grades)
    grade_map: dict[str, int] = {}
    for criterion_id, grade in ordered:
        if criterion_id in grade_map:
            raise PaperRewardError(f"duplicate grade for criterion {criterion_id}")
        if isinstance(grade, bool) or not isinstance(grade, int) or grade not in {0, 1}:
            raise PaperRewardError("grades must be integer 0 or 1")
        grade_map[criterion_id] = grade
    expected = {criterion.criterion_id for criterion in criteria}
    if set(grade_map) != expected:
        raise PaperRewardError("grade inventory must exactly match the rubric")
    weights = {item.criterion_id: Fraction(str(item.weight)) for item in criteria}
    denominator = sum((weight for weight in weights.values() if weight > 0), Fraction())
    if denominator <= 0:
        raise PaperRewardError("Eq. 4 denominator must contain positive rubric weight")
    numerator = sum(
        (weights[criterion_id] * grade_map[criterion_id] for criterion_id in grade_map),
        Fraction(),
    )
    canonical_grades = tuple((item.criterion_id, grade_map[item.criterion_id]) for item in criteria)
    return PaperRewardCalculation(canonical_grades, numerator, denominator, numerator / denominator)


def grade_and_compute(
    payload: str | Mapping[str, Any], criteria: Sequence[WeightedCriterion]
) -> PaperRewardCalculation:
    grades = parse_binary_grades(payload, criteria)
    return compute_paper_reward(criteria, grades)
