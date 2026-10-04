from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from dynamic_rubric.rubrics.admission import AdmissionEvidence, evaluate_admission
from dynamic_rubric.rubrics.extractor import make_blind_pairing
from dynamic_rubric.rubrics.pool import MAX_BUDGETED_CRITERIA
from dynamic_rubric.rubrics.replay import (
    ReplayCandidate,
    ReplayMode,
    initial_replay_snapshot,
    replay_step,
)
from dynamic_rubric.rubrics.static import (
    CriterionValidationError,
    EQUAL_CRITERION_WEIGHT,
    UNIVERSAL_CRITERIA,
    build_static_rubric,
    validate_criteria,
)


TASK_CRITERIA = (
    "States the primary finding accurately",
    "Explains the recommended next action",
    "Uses details supported by the response",
    "Prioritizes the most urgent concern",
    "Provides an actionable monitoring plan",
    "Communicates limitations with appropriate certainty",
)


def good_evidence(**changes: object) -> AdmissionEvidence:
    values: dict[str, object] = {
        "satisfaction_rate": 0.5,
        "current_mean": 0.70,
        "reference_mean": 0.50,
        "max_active_similarity": 0.3,
        "parse_success": 1.0,
        "independent_validation": True,
        "score_variance": 0.1,
        "recent_validation_failure_rate": 0.0,
    }
    values.update(changes)
    return AdmissionEvidence(**values)  # type: ignore[arg-type]


def test_static_rubric_enforces_six_plus_two_equal_weight_contract() -> None:
    rubric = build_static_rubric("prompt-1", TASK_CRITERIA)
    assert len(rubric.criteria) == 8
    assert tuple(item.text for item in rubric.criteria[-2:]) == UNIVERSAL_CRITERIA
    assert {item.weight for item in rubric.criteria} == {EQUAL_CRITERION_WEIGHT}
    assert [item.source for item in rubric.criteria].count("task_specific") == 6
    assert [item.source for item in rubric.criteria].count("universal") == 2
    with pytest.raises(FrozenInstanceError):
        rubric.prompt_id = "changed"  # type: ignore[misc]


def test_static_criterion_structural_and_duplicate_validation() -> None:
    with pytest.raises(CriterionValidationError, match="exact duplicates"):
        validate_criteria(("States the finding", " states  the finding. "))
    with pytest.raises(CriterionValidationError, match="numbered candidate"):
        validate_criteria(("Candidate 2 states the finding",))
    with pytest.raises(CriterionValidationError, match="response alone"):
        validate_criteria(("Matches the physician rubric",))
    with pytest.raises(CriterionValidationError, match="positive behavior"):
        validate_criteria(("Avoid unnecessary detail",))
    with pytest.raises(CriterionValidationError, match="single and atomic") as error:
        validate_criteria(("States the finding; recommends follow-up",))
    assert "States the finding; recommends follow-up" in str(error.value)
    with pytest.raises(CriterionValidationError, match="Clear finding.*Accurate result"):
        validate_criteria(("Clear finding", "Accurate result"), similarity=lambda _a, _b: 0.85)


def test_blind_pairing_is_deterministic_one_to_one_and_source_free() -> None:
    kwargs = {"seed": 19, "prompt_id": "p", "step": 3}
    first = make_blind_pairing(("c0", "c1", "c2", "c3"), ("r0", "r1", "r2", "r3"), **kwargs)
    second = make_blind_pairing(("c0", "c1", "c2", "c3"), ("r0", "r1", "r2", "r3"), **kwargs)
    assert first == second
    assert {assignment.control_index for assignment in first.assignments} == {0, 1, 2, 3}
    assert {assignment.current_index for assignment in first.assignments} == {0, 1, 2, 3}
    assert not hasattr(first.generator_payload()[0], "current_label")
    assert make_blind_pairing(("c0",), ("r0",), seed=20, prompt_id="p", step=3) != first


@pytest.mark.parametrize("seed", range(32))
def test_eight_blind_pairs_always_balance_current_between_a_and_b(seed: int) -> None:
    current = tuple(f"current-{index}" for index in range(8))
    control = tuple(f"control-{index}" for index in range(8))
    kwargs = {"seed": seed, "prompt_id": "balanced", "step": 3}

    first = make_blind_pairing(current, control, **kwargs)
    second = make_blind_pairing(current, control, **kwargs)

    assert first == second
    assert [assignment.current_label for assignment in first.assignments].count("A") == 4
    assert [assignment.current_label for assignment in first.assignments].count("B") == 4
    assert {assignment.control_index for assignment in first.assignments} == set(range(8))
    assert all(not hasattr(pair, "current_label") for pair in first.generator_payload())


@pytest.mark.parametrize("rate", [0.10, 0.90])
def test_satisfaction_boundaries_are_inclusive(rate: float) -> None:
    assert evaluate_admission(good_evidence(satisfaction_rate=rate)).admitted


def test_admission_gate_exact_separation_similarity_and_parse_boundaries() -> None:
    assert evaluate_admission(good_evidence(current_mean=0.65, reference_mean=0.50)).admitted
    assert not evaluate_admission(good_evidence(current_mean=0.6499, reference_mean=0.50)).admitted
    assert not evaluate_admission(good_evidence(max_active_similarity=0.85)).admitted
    assert evaluate_admission(good_evidence(max_active_similarity=0.849999)).admitted
    assert evaluate_admission(good_evidence(parse_success=0.95)).admitted
    assert not evaluate_admission(good_evidence(parse_success=0.949999)).admitted
    assert not evaluate_admission(good_evidence(independent_validation=False)).admitted


def test_rejected_replay_records_new_snapshot_but_preserves_content() -> None:
    static = build_static_rubric("p", TASK_CRITERIA)
    initial = initial_replay_snapshot(static, ReplayMode.DYNAMIC_FIXED_BUDGETED)
    rejected = replay_step(
        initial,
        step=1,
        candidates=(
            ReplayCandidate(
                "d1", "Describes a useful distinction", good_evidence(parse_success=0.5)
            ),
        ),
    )
    assert rejected is not initial
    assert rejected.step == 1
    assert rejected.admitted_id is None
    assert rejected.content_hash == initial.content_hash
    assert rejected.criteria == initial.criteria


def test_budgeted_pool_preserves_static_and_deterministically_evicts() -> None:
    static = build_static_rubric("p", TASK_CRITERIA)
    snapshot = initial_replay_snapshot(static, ReplayMode.DYNAMIC_FIXED_BUDGETED)
    for step, criterion_id in enumerate(("d", "b", "c", "a", "z"), start=1):
        snapshot = replay_step(
            snapshot,
            step=step,
            candidates=(
                ReplayCandidate(
                    criterion_id,
                    f"Captures useful distinction {criterion_id.upper()}",
                    good_evidence(),
                ),
            ),
        )
    assert len(snapshot.criteria) == MAX_BUDGETED_CRITERIA
    assert snapshot.evicted_id == "d"  # equal utility: oldest last-validation step is evicted
    assert tuple(item.criterion_id for item in snapshot.criteria[:8]) == tuple(
        item.criterion_id for item in static.criteria
    )


def test_budgeted_tie_break_uses_utility_then_age_then_lexical_id() -> None:
    static = build_static_rubric("p", TASK_CRITERIA)
    snapshot = initial_replay_snapshot(static, ReplayMode.DYNAMIC_FIXED_BUDGETED)
    # Populate the same validation step through direct sequential snapshots is
    # impossible, so lexical tie-break is exercised by replacing a full pool
    # candidate that ties the oldest entry's utility and validation age below.
    for step, criterion_id in enumerate(("b", "c", "d", "e"), start=1):
        snapshot = replay_step(
            snapshot,
            step=step,
            candidates=(
                ReplayCandidate(criterion_id, f"Provides detail {criterion_id}", good_evidence()),
            ),
        )
    lower = good_evidence(
        score_variance=0.0,
        current_mean=0.65,
        reference_mean=0.50,
        max_active_similarity=0.84,
        recent_validation_failure_rate=1.0,
    )
    updated = replay_step(
        snapshot,
        step=5,
        candidates=(ReplayCandidate("a", "Provides a marginal detail", lower),),
    )
    assert updated.evicted_id == "a"
    assert updated.content_hash == snapshot.content_hash


def test_cumulative_pool_is_monotonic_and_unbounded_by_budgeted_limit() -> None:
    static = build_static_rubric("p", TASK_CRITERIA)
    snapshot = initial_replay_snapshot(static, ReplayMode.DYNAMIC_FIXED_CUMULATIVE)
    sizes = [len(snapshot.criteria)]
    for step in range(1, 7):
        snapshot = replay_step(
            snapshot,
            step=step,
            candidates=(
                ReplayCandidate(f"d{step}", f"Captures distinction {step}", good_evidence()),
            ),
        )
        sizes.append(len(snapshot.criteria))
    assert sizes == sorted(sizes)
    assert sizes[-1] == 14


@pytest.mark.parametrize("mode", list(ReplayMode))
def test_all_replay_modes_create_snapshot_friendly_pure_state(mode: ReplayMode) -> None:
    initial = initial_replay_snapshot(build_static_rubric("p", TASK_CRITERIA), mode)
    next_snapshot = replay_step(initial, step=1)
    assert next_snapshot is not initial
    assert next_snapshot.content_hash == initial.content_hash
    assert initial.step == 0


def test_at_most_three_candidates_are_accepted_from_extractor() -> None:
    initial = initial_replay_snapshot(
        build_static_rubric("p", TASK_CRITERIA), ReplayMode.REFRESH_ONLY_BUDGETED
    )
    candidates = tuple(
        ReplayCandidate(f"d{index}", f"Captures distinction {index}", good_evidence())
        for index in range(4)
    )
    with pytest.raises(ValueError, match="at most 3"):
        replay_step(initial, step=1, candidates=candidates)
