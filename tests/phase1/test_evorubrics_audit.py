from __future__ import annotations

from dataclasses import replace

import pytest

from dynamic_rubric.phase1.evorubrics_audit import (
    CriterionJudgment,
    EvoAuditContractError,
    RubricGradeRecord,
    aggregate_evo_score_group,
    analyze_evo_grade_matrix,
    compare_evo_prompt,
    deterministic_subset_indices,
    normalized_rubric_reward,
)
from dynamic_rubric.phase1.metrics import ScoreGroup

RESPONSES = tuple(f"r{index:02d}" for index in range(16))
RUBRICS = tuple(f"rubric-{index}" for index in range(4))


def _records(*, evaluator_step: int = 0, policy_step: int = 2):
    rows = []
    for response_index, response_id in enumerate(RESPONSES):
        for rubric_id in RUBRICS:
            rows.append(
                RubricGradeRecord(
                    policy_step=policy_step,
                    evaluator_step=evaluator_step,
                    policy_checkpoint=f"theta-{policy_step}",
                    evaluator_checkpoint=f"psi-{evaluator_step}",
                    prompt_id="p1",
                    response_id=response_id,
                    rubric_id=rubric_id,
                    judgments=(
                        CriterionJudgment("positive", 2, response_index > 0),
                        CriterionJudgment("penalty", -1, response_index == 0),
                    ),
                )
            )
    return rows


def test_signed_reward_and_complete_four_rubric_average():
    assert (
        normalized_rubric_reward((CriterionJudgment("good", 2, 0), CriterionJudgment("bad", -1, 1)))
        == 0
    )
    assert (
        normalized_rubric_reward((CriterionJudgment("good", 2, 1), CriterionJudgment("bad", -1, 0)))
        == 1
    )

    group = aggregate_evo_score_group(
        _records(), expected_response_ids=RESPONSES, expected_rubric_ids=RUBRICS
    )
    assert group.response_ids == RESPONSES
    assert group.rewards == (0.0, *([1.0] * 15))
    assert len(group.criterion_grades) == 8


@pytest.mark.parametrize("failure", ["missing", "parse", "response_id", "rubric_id"])
def test_grade_grid_never_silently_changes_denominator(failure):
    rows = _records()
    expected_responses = RESPONSES
    expected_rubrics = RUBRICS
    match = ""
    if failure == "missing":
        rows.pop()
        match = "incomplete 16x4"
    elif failure == "parse":
        rows[0] = replace(rows[0], parse_ok=False)
        match = "parse failed"
    elif failure == "response_id":
        rows[0] = replace(rows[0], response_id="unknown")
        match = "unexpected response_id"
    else:
        expected_rubrics = ("same",) * 4
        match = "distinct rubric IDs"
    with pytest.raises(EvoAuditContractError, match=match):
        aggregate_evo_score_group(
            rows,
            expected_response_ids=expected_responses,
            expected_rubric_ids=expected_rubrics,
        )


def test_zar4_draws_are_deterministic_and_paired_between_fresh_and_stale():
    first = deterministic_subset_indices(repetitions=50, seed=17)
    second = deterministic_subset_indices(repetitions=50, seed=17)
    assert first == second
    assert all(len(indices) == len(set(indices)) == 4 for indices in first)

    stale = ScoreGroup("p1", "psi-0", "theta-2", RESPONSES, (0.5,) * 16)
    fresh = ScoreGroup("p1", "psi-2", "theta-2", RESPONSES, tuple(range(16)))
    result = compare_evo_prompt(
        stale,
        fresh,
        epsilon_z=0.01,
        epsilon_t=0.01,
        subset_indices=first,
    )
    assert result["stale_zar_at_4"] == 1
    assert result["fresh_zar_at_4"] == 0
    assert result["v_adj_zar_at_4"] == 1
    assert result["stale_tie_rate_16"] == 1
    assert result["fresh_separation_rate_16"] == 1
    assert result["delta_top_median_margin_16"] == 7.5
    assert result["kendall_tau_b_16"] is None


def _matrix_records():
    rows = []
    for prompt_id in ("p1", "p2"):
        for evaluator_step, policy_step in ((0, 0), (0, 2), (2, 2)):
            for row in _records(evaluator_step=evaluator_step, policy_step=policy_step):
                rows.append(replace(row, prompt_id=prompt_id))
    return rows


def _manifests():
    response_ids = {
        (policy_step, prompt_id): RESPONSES for policy_step in (0, 2) for prompt_id in ("p1", "p2")
    }
    rubric_ids = {
        (evaluator_step, prompt_id): RUBRICS
        for evaluator_step in (0, 2)
        for prompt_id in ("p1", "p2")
    }
    return response_ids, rubric_ids


def test_matrix_bootstraps_prompts_and_preserves_pending_expected_cells():
    response_ids, rubric_ids = _manifests()
    rows = [
        row
        for row in _matrix_records()
        if not (row.evaluator_step == 0 and row.policy_step == 2 and row.prompt_id == "p2")
    ]
    report = analyze_evo_grade_matrix(
        rows,
        expected_pairs=((0, 0), (0, 2), (2, 2)),
        expected_prompt_ids=("p1", "p2"),
        response_ids_by_policy_prompt=response_ids,
        rubric_ids_by_evaluator_prompt=rubric_ids,
        epsilon_z=0.01,
        epsilon_t=0.01,
        zar4_repetitions=25,
        bootstrap_iterations=20,
    )
    assert report["expected_cell_count"] == 3
    assert report["complete_cell_count"] == 2
    assert report["pending_cell_count"] == 1
    pending = next(cell for cell in report["cells"] if cell["status"] == "pending")
    assert pending["missing_prompt_ids"] == ["p2"]
    complete = next(
        cell
        for cell in report["cells"]
        if cell["status"] == "complete" and cell["policy_step"] == 2
    )
    assert complete["bootstrap_95ci"]["v_adj_zar_at_4"]["n_prompt_clusters"] == 2
    assert complete["bootstrap_95ci"]["v_adj_zar_at_4"]["n_prompt_clusters"] != 50
    assert report["bootstrap_unit"] == "prompt"


def test_partially_present_grid_is_an_error_not_pending():
    response_ids, rubric_ids = _manifests()
    rows = _matrix_records()
    rows.pop()
    with pytest.raises(EvoAuditContractError, match="incomplete 16x4"):
        analyze_evo_grade_matrix(
            rows,
            expected_pairs=((0, 0), (0, 2), (2, 2)),
            expected_prompt_ids=("p1", "p2"),
            response_ids_by_policy_prompt=response_ids,
            rubric_ids_by_evaluator_prompt=rubric_ids,
            epsilon_z=0.01,
            epsilon_t=0.01,
            zar4_repetitions=10,
            bootstrap_iterations=10,
        )
