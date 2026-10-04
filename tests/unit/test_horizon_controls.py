from __future__ import annotations

from dataclasses import asdict

from dynamic_rubric.horizon.contracts import (
    CriterionType,
    ImportanceClass,
    WEIGHT_UNITS,
    WeightedCriterion,
    criterion_content_hash,
)
from dynamic_rubric.horizon.controls import match_control_extension
from dynamic_rubric.horizon.control_artifacts import attach_control_extensions


def criterion(identity: str, importance: ImportanceClass, candidate: str) -> WeightedCriterion:
    text = f"Criterion {identity}"
    return WeightedCriterion(
        criterion_instance_id=identity,
        canonical_criterion_hash=criterion_content_hash(text),
        text=text,
        importance_class=importance,
        criterion_type=(
            CriterionType.PITFALL
            if importance is ImportanceClass.PITFALL
            else CriterionType.QUALITY
        ),
        weight_units=WEIGHT_UNITS[importance],
        source_candidate_ids=(candidate,),
        source_checkpoint="step2",
    )


def test_stale_match_prefers_exact_weight_bins_and_reports_quality() -> None:
    current = (
        criterion("current-e", ImportanceClass.ESSENTIAL, "now0"),
        criterion("current-o", ImportanceClass.OPTIONAL, "now1"),
    )
    available = (
        criterion("old-i", ImportanceClass.IMPORTANT, "pair0"),
        criterion("old-o", ImportanceClass.OPTIONAL, "pair1"),
        criterion("old-e", ImportanceClass.ESSENTIAL, "pair2"),
    )
    result = match_control_extension(
        current, available, source_pair_order={"pair0": 0, "pair1": 1, "pair2": 2}
    )
    assert result.eligible
    assert result.exact_histogram_match
    assert {item.weight_units for item in result.selected} == {3, 10}
    assert result.total_weight_difference == 0


def test_stale_match_never_duplicates_and_marks_insufficient_coverage_na() -> None:
    current = (
        criterion("current-e", ImportanceClass.ESSENTIAL, "now0"),
        criterion("current-i", ImportanceClass.IMPORTANT, "now1"),
    )
    only_one = (criterion("old-e", ImportanceClass.ESSENTIAL, "pair0"),)
    result = match_control_extension(current, only_one)
    assert not result.eligible
    assert result.selected == ()
    assert result.total_weight_difference is None
    assert result.reason == "insufficient_control_criteria"


def test_missing_bin_uses_nearest_weight_then_oldest_pair() -> None:
    current = (criterion("current-i", ImportanceClass.IMPORTANT, "now"),)
    older = criterion("old-o", ImportanceClass.OPTIONAL, "pair0")
    newer = criterion("old-e", ImportanceClass.ESSENTIAL, "pair1")
    result = match_control_extension(
        current, (newer, older), source_pair_order={"pair0": 0, "pair1": 1}
    )
    assert result.selected == (newer,)
    assert not result.exact_histogram_match
    assert result.total_weight_difference == 3


def test_stale_controls_can_be_attached_after_independent_generation() -> None:
    current = criterion("current-i", ImportanceClass.IMPORTANT, "now")
    stale = criterion("stale-i", ImportanceClass.IMPORTANT, "old")
    rows = attach_control_extensions(
        [
            {
                "prompt_id": "p1",
                "checkpoint_id": "step6",
                "extension": [asdict(current)],
                "control_extension": None,
                "control_match": None,
            }
        ],
        [{"prompt_id": "p1", "checkpoint_id": "step3", "extension": [asdict(stale)]}],
    )
    assert rows[0]["extension"] == [asdict(current)]
    assert rows[0]["control_extension"] == [asdict(stale)]
    assert rows[0]["control_match"]["eligible"] is True
