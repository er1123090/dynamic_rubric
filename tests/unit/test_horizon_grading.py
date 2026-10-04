from __future__ import annotations

import pytest

from dynamic_rubric.horizon.contracts import (
    CriterionType,
    ImportanceClass,
    WeightedCriterion,
    WeightedRubric,
    criterion_content_hash,
)
from dynamic_rubric.horizon.grading import (
    NO_TARGET,
    YES_TARGET,
    GradingContractError,
    HardGradeCache,
    assemble_variant_scores,
    grade_cache_key,
    hard_grade_from_target_logprobs,
)


def criterion(identity: str, text: str, weight: int) -> WeightedCriterion:
    importance = ImportanceClass.ESSENTIAL if weight == 10 else ImportanceClass.IMPORTANT
    return WeightedCriterion(
        criterion_instance_id=identity,
        canonical_criterion_hash=criterion_content_hash(text),
        text=text,
        importance_class=importance,
        criterion_type=CriterionType.QUALITY,
        weight_units=weight,
    )


def test_full_target_sequences_are_summed_and_mapped_to_public_labels() -> None:
    present = hard_grade_from_target_logprobs({YES_TARGET: (-0.2, -0.3), NO_TARGET: (-0.4, -0.5)})
    assert present.valid and present.grade == 1 and present.public_label == "PRESENT"
    assert present.yes_logprob == pytest.approx(-0.5)
    absent = hard_grade_from_target_logprobs({YES_TARGET: (-2.0,), NO_TARGET: (-0.2, -0.1)})
    assert absent.grade == 0 and absent.public_label == "NOT_PRESENT"


def test_exact_target_tie_is_auditable_and_conservatively_maps_to_absent() -> None:
    tied = hard_grade_from_target_logprobs({YES_TARGET: (-0.2, -0.3), NO_TARGET: (-0.5,)})
    assert tied.valid
    assert tied.parse_status == "ambiguous_target_tie"
    assert tied.grade == 0 and tied.public_label == "NOT_PRESENT"
    assert tied.probability_present == 0.5
    with pytest.raises(GradingContractError, match="exact targets"):
        hard_grade_from_target_logprobs({"YES": (-0.1,), "NO": (-0.2,)})


def test_cache_identity_is_blind_to_rubric_variant_and_reuses_byte_identical_grade() -> None:
    key = grade_cache_key(
        grader_model_revision="grader-rev",
        tokenizer_revision="tokenizer-rev",
        grader_prompt_hash="prompt-hash",
        prompt_hash="input-prompt-hash",
        response_hash="response-hash",
        canonical_criterion_hash="criterion-hash",
    )
    cache = HardGradeCache()
    calls = 0

    def grade():
        nonlocal calls
        calls += 1
        return hard_grade_from_target_logprobs({YES_TARGET: (-0.1,), NO_TARGET: (-1.0,)})

    first = cache.get_or_grade(key=key, grader=grade)
    second = cache.get_or_grade(key=key, grader=grade)
    assert first is second
    assert calls == 1 and cache.misses == 1 and cache.hits == 1
    assert cache.artifact(key)["public_label"] == "PRESENT"


def test_variant_score_assembly_reuses_content_grade_handles_ties_and_fails_closed() -> None:
    shared_text = "States the primary finding"
    shared_r0 = criterion("r0-shared", shared_text, 10)
    shared_current = criterion("rt-shared", shared_text, 10)
    added = criterion("rt-added", "Provides the next action", 7)
    rubrics = {
        "r0": WeightedRubric("p", (shared_r0,), rubric_id="r0"),
        "rt": WeightedRubric("p", (shared_current, added), rubric_id="rt"),
    }
    grades = {
        shared_r0.canonical_criterion_hash: hard_grade_from_target_logprobs(
            {YES_TARGET: (-0.1,), NO_TARGET: (-1.0,)}
        ),
        added.canonical_criterion_hash: hard_grade_from_target_logprobs(
            {YES_TARGET: (-1.0,), NO_TARGET: (-0.1,)}
        ),
    }
    scores = assemble_variant_scores(rubrics, grades)
    assert (scores["r0"].numerator, scores["r0"].denominator) == (10, 10)
    assert (scores["rt"].numerator, scores["rt"].denominator) == (10, 17)

    grades[added.canonical_criterion_hash] = hard_grade_from_target_logprobs(
        {YES_TARGET: (-0.5,), NO_TARGET: (-0.5,)}
    )
    tied_scores = assemble_variant_scores(rubrics, grades)
    assert (tied_scores["rt"].numerator, tied_scores["rt"].denominator) == (10, 17)

    grades.pop(added.canonical_criterion_hash)
    with pytest.raises(GradingContractError, match="missing grade"):
        assemble_variant_scores(rubrics, grades)
