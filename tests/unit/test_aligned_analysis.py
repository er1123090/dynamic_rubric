from __future__ import annotations

import pytest

from dynamic_rubric.evaluation.bootstrap import BootstrapResult
from dynamic_rubric.reporting.aligned_analysis import AlignmentError, analyze_aligned
from dynamic_rubric.reporting.interpretation import classify_interpretation


def _bootstrap(point: float, low: float, high: float, classification: str) -> BootstrapResult:
    return BootstrapResult(point, low, high, classification, 2, 100, 7)


def test_all_eight_interpretations_are_reachable_and_ci_gated() -> None:
    inconclusive = _bootstrap(0.01, -0.02, 0.04, "inconclusive")
    equivalent = _bootstrap(0.0, -0.01, 0.01, "local_equivalence")
    meaningful = _bootstrap(0.04, 0.01, 0.07, "meaningful_difference")
    cases = [
        ("weak_policy_drift", 0.01, 1, meaningful, meaningful, meaningful),
        ("refresh_noise", 0.10, 1, inconclusive, inconclusive, meaningful),
        ("frequent_update_need", 0.10, 1, inconclusive, meaningful, inconclusive),
        ("general_rubric_improvement", 0.10, 1, meaningful, meaningful, inconclusive),
        ("policy_adaptive_gain", 0.10, 1, meaningful, inconclusive, inconclusive),
        ("textual_only_churn", 0.10, 1, equivalent, inconclusive, inconclusive),
        ("updater_miss", 0.10, 0, inconclusive, inconclusive, inconclusive),
        ("local_redundancy", 0.10, 1, inconclusive, inconclusive, inconclusive),
    ]
    for expected, max_kl, admissions, primary, previous, refresh in cases:
        decision = classify_interpretation(
            max_policy_kl=max_kl,
            admissions=admissions,
            primary=primary,
            previous=previous,
            refresh=refresh,
        )
        assert decision.label == expected
        assert decision.evidence


def _selection(
    *, mode: str, rubric_step: int, n: int, response_id: str, pool_hash: str = "shared"
) -> dict[str, object]:
    return {
        "policy_id": "pi_1",
        "policy_step": 1,
        "prompt_id": "p1",
        "mode": mode,
        "rubric_step": rubric_step,
        "n": n,
        "permutation": 0,
        "pool_hash": pool_hash,
        "response_id": response_id,
    }


def _score(mode: str, rubric_step: int, candidate_id: int, score: float) -> dict[str, object]:
    return {
        "policy_id": "pi_1",
        "policy_step": 1,
        "prompt_id": "p1",
        "mode": mode,
        "rubric_step": rubric_step,
        "global_candidate_id": candidate_id,
        "score": score,
        "judge_repeat_score": score,
    }


def test_aligned_analysis_uses_current_rubric_and_exact_candidate_joins() -> None:
    selections = [
        _selection(mode="static", rubric_step=0, n=1, response_id="low"),
        _selection(mode="static", rubric_step=0, n=2, response_id="low"),
        _selection(mode="dynamic_fixed_budgeted", rubric_step=1, n=1, response_id="high"),
        _selection(mode="dynamic_fixed_budgeted", rubric_step=1, n=2, response_id="high"),
        # Cross-matrix stale rubric: retained in cross-matrix, excluded from current comparison.
        _selection(mode="dynamic_fixed_budgeted", rubric_step=0, n=1, response_id="low"),
        _selection(mode="dynamic_fixed_budgeted", rubric_step=0, n=2, response_id="low"),
    ]
    gold = {("p1", "low"): 0.2, ("p1", "high"): 0.8}
    scores = [
        _score("static", 0, 1, 0.1),
        _score("static", 0, 2, 0.9),
        _score("dynamic_fixed_budgeted", 1, 1, 0.9),
        _score("dynamic_fixed_budgeted", 1, 2, 0.1),
        _score("dynamic_fixed_budgeted", 0, 1, 0.1),
        _score("dynamic_fixed_budgeted", 0, 2, 0.9),
    ]
    result = analyze_aligned(
        selections, gold, scores, iterations=100, seed=7, n_grid=(1, 2), permutations=1
    )
    assert result["gt_auc"]["static"] == pytest.approx(0.2)
    assert result["gt_auc"]["dynamic_fixed_budgeted"] == pytest.approx(0.8)
    assert result["stale_rubric_regret"]["dynamic_fixed_budgeted"] == pytest.approx(0.6)
    assert result["top1_agreement"]["dynamic_fixed_budgeted"] == 0.0
    assert result["kendall_tau_b"]["dynamic_fixed_budgeted"] == -1.0
    assert "dynamic_fixed_budgeted:R_0:pi_1" in result["gt_auc_cross_matrix"]


def test_aligned_analysis_rejects_missing_dynamic_cell_and_partial_grid() -> None:
    gold = {("p1", "low"): 0.2, ("p1", "high"): 0.8}
    missing_dynamic = [
        _selection(mode="static", rubric_step=0, n=1, response_id="low"),
        _selection(mode="static", rubric_step=0, n=2, response_id="low"),
        _selection(mode="dynamic_fixed_budgeted", rubric_step=1, n=1, response_id="high"),
    ]
    with pytest.raises(AlignmentError, match="configured N/permutation grid"):
        analyze_aligned(
            missing_dynamic,
            gold,
            [],
            iterations=10,
            seed=7,
            n_grid=(1, 2),
            permutations=1,
        )

    partial_everywhere = [
        _selection(mode="static", rubric_step=0, n=1, response_id="low"),
        _selection(mode="dynamic_fixed_budgeted", rubric_step=1, n=1, response_id="high"),
    ]
    with pytest.raises(AlignmentError, match="configured N/permutation grid"):
        analyze_aligned(
            partial_everywhere,
            gold,
            [],
            iterations=10,
            seed=7,
            n_grid=(1, 2),
            permutations=1,
        )


def test_aligned_analysis_rejects_different_pool_or_candidate_sets() -> None:
    selections = [
        _selection(mode="static", rubric_step=0, n=1, response_id="low"),
        _selection(
            mode="dynamic_fixed_budgeted",
            rubric_step=1,
            n=1,
            response_id="high",
            pool_hash="different",
        ),
    ]
    gold = {("p1", "low"): 0.2, ("p1", "high"): 0.8}
    with pytest.raises(AlignmentError, match="different candidate pools"):
        analyze_aligned(selections, gold, [], iterations=10, seed=7, n_grid=(1,), permutations=1)

    selections[1] = _selection(
        mode="dynamic_fixed_budgeted", rubric_step=1, n=1, response_id="high"
    )
    scores = [_score("static", 0, 1, 0.1), _score("dynamic_fixed_budgeted", 1, 2, 0.9)]
    with pytest.raises(AlignmentError, match="candidate IDs do not align"):
        analyze_aligned(
            selections, gold, scores, iterations=10, seed=7, n_grid=(1,), permutations=1
        )
