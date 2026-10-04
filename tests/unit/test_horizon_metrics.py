from fractions import Fraction

import pytest

from dynamic_rubric.horizon.metrics import (
    advantage_degenerate,
    count_normalized_gain,
    count_normalized_summary,
    criterion_effectiveness,
    criterion_effectiveness_summary,
    criterion_vector_resolution,
    exact_zero_advantage,
    exact_zar_rate,
    low_reward_spread,
    low_reward_spread_rate,
    pairwise_metrics,
    ranking_agreement,
    score_spread,
)


def test_exact_zar_uses_reduced_rational_equality() -> None:
    assert exact_zero_advantage([(1, 2), (2, 4), Fraction(3, 6)])
    assert not exact_zero_advantage([(1, 2), (2, 4), (1, 3)])
    assert exact_zar_rate([[(1, 2), (2, 4)], [(0, 1), (1, 1)]]) == 0.5


def test_low_spread_and_supplied_advantage_degeneracy_are_separate() -> None:
    scores = [(500000, 1000000), (500001, 1000000)]
    assert low_reward_spread(scores, epsilon=1e-5)
    assert low_reward_spread_rate([scores, [(0, 1), (1, 1)]], epsilon=1e-5) == 0.5
    assert not advantage_degenerate([-1.0, 1.0], delta=1e-6)
    assert advantage_degenerate([-1e-8, 1e-8], delta=1e-6)


def test_sixteen_responses_produce_exactly_120_pairs_with_one_outlier() -> None:
    scores = [(0, 1)] * 15 + [(1, 1)]
    result = pairwise_metrics(scores)
    assert result["pair_count"] == 120
    assert result["tie_rate"] == 105 / 120
    assert result["separation_rate"] == 15 / 120


def test_pairwise_tie_resolution_new_tie_and_reversal() -> None:
    baseline = [0, 0, 1, 2]
    current = [0, 1, 1, -1]
    result = pairwise_metrics(current, baseline_scores=baseline)
    assert result["pair_count"] == 6
    assert result["incremental_tie_resolution"] == pytest.approx(1 / 6)
    assert result["conditional_tie_resolution"] == 1.0
    assert result["new_tie_rate"] == pytest.approx(1 / 6)
    assert result["ordering_reversal_rate"] == pytest.approx(3 / 6)
    assert sum(result["ordering_confusion"].values()) == 6


def test_criterion_effectiveness_and_empty_online_rubric() -> None:
    assert criterion_effectiveness([1] * 16) == "saturated"
    assert criterion_effectiveness([0] * 16) == "dead"
    assert criterion_effectiveness([0, 1] * 8) == "effective"
    empty = criterion_effectiveness_summary({})
    assert empty["criterion_count"] == 0
    assert empty["ratios"]["effective"] is None


def test_same_weighted_total_can_hide_different_grade_vectors() -> None:
    criteria = {"a": [1, 0], "b": [0, 1]}
    assert criterion_effectiveness_summary(criteria)["counts"]["effective"] == 2
    assert exact_zero_advantage([(1, 2), (1, 2)])
    resolution = criterion_vector_resolution([(1, 2), (1, 2)], [[1, 0], [0, 1]])
    assert resolution["conditional_vector_resolution"] == 1.0


def test_score_spread_uses_linear_iqr_and_exact_unique_ratio() -> None:
    spread = score_spread([0, 1, 2, 3])
    assert spread["iqr"] == 1.5
    assert spread["unique_score_ratio"] == 1.0


def test_constant_ranking_has_kendall_na_and_top_tie_set_semantics() -> None:
    result = ranking_agreement([1, 1, 1], [0, 1, 1])
    assert result["kendall_tau_b"] is None
    assert result["kendall_defined"] is False
    assert result["top_set_jaccard"] == pytest.approx(2 / 3)
    assert result["top_set_exact_match"] is False


def test_count_normalized_gain_reports_na_for_empty_added_rubric() -> None:
    assert count_normalized_gain(0.1, 2) == 0.05
    assert count_normalized_gain(0.1, 0) is None
    assert count_normalized_summary(
        delta_zar=0.1, delta_separation=0.2, mean_added_criteria=2
    ) == {
        "mean_added_criteria": 2.0,
        "zar_reduction_per_added_criterion": 0.05,
        "separation_gain_per_added_criterion": 0.1,
    }


def test_invalid_inputs_fail_closed() -> None:
    with pytest.raises((TypeError, ValueError)):
        exact_zero_advantage([(1, 0)])
    with pytest.raises(ValueError):
        criterion_effectiveness([0, 2])
