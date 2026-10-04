from __future__ import annotations

import pytest

from scripts.phase1 import reanalyze_through34 as analysis


def _receipt(response_id: str, numerator: int, denominator: int, grades: list[list]) -> dict:
    return {
        "response_id": response_id,
        "prompt_id": "p",
        "numerator": numerator,
        "denominator": denominator,
        "reward": numerator / denominator,
        "grades": grades,
    }


def test_exact_rational_metrics_are_response_order_invariant() -> None:
    rows = [
        _receipt("b", 1, 3, [["c", 1], ["d", 0]]),
        _receipt("a", 2, 6, [["c", 0], ["d", 0]]),
        _receipt("c", 2, 3, [["c", 1], ["d", 0]]),
    ]
    first = analysis.evaluator_metrics(rows, epsilon_z=0.01, epsilon_t=0.01)
    second = analysis.evaluator_metrics(list(reversed(rows)), epsilon_z=0.01, epsilon_t=0.01)

    assert first == second
    assert first["zar"] == 0  # 1/3 == 2/6 exactly, but the third reward differs
    assert first["tie_count"] == 1
    assert first["effective_count"] == 1
    assert first["dead_count"] == 1


def test_group_contingency_counts_rescue_and_ties_resolved() -> None:
    group = {
        "global_step": 3,
        "prompt_id": "p",
        "fresh_creation_update": 3,
        "stale_creation_update": 1,
        "evaluator_age_steps": 2,
        "clock": {"cumulative_prompt_exposures": 192},
        "stale": [
            _receipt("a", 0, 1, [["s", 0]]),
            _receipt("b", 0, 1, [["s", 0]]),
        ],
        "fresh": [
            _receipt("a", 0, 1, [["f", 0]]),
            _receipt("b", 1, 1, [["f", 1]]),
        ],
    }
    row = analysis.group_row(group, "inference_b", epsilon_z=0.01, epsilon_t=0.01)

    assert row["contingency"] == "rescue"
    assert row["v_adj_zar"] == 1
    assert row["stale_ties"] == 1
    assert row["ties_resolved"] == 1
    assert row["ties_created"] == 0
    assert row["kendall_tau_b"] is None


def test_cluster_bootstrap_is_deterministic_and_visit_weighted() -> None:
    rows = [
        {"prompt_id": "a", "value": 0.0},
        {"prompt_id": "a", "value": 1.0},
        {"prompt_id": "b", "value": 1.0},
    ]
    specs = {"value": lambda row: (row["value"], 1.0)}
    first = analysis.cluster_bootstrap(rows, 2_000, 11, specs)
    second = analysis.cluster_bootstrap(rows, 2_000, 11, specs)

    assert first == second
    assert first["value"]["estimate"] == pytest.approx(2 / 3)
    assert first["value"]["cluster_count"] == 2
    assert first["value"]["visit_count"] == 3
    assert first["value"]["bootstrap_defined"] == 2_000


def test_cluster_bootstrap_pooled_ratio_difference_uses_criterion_denominators() -> None:
    rows = [
        {"prompt_id": "a", "fn": 1, "fd": 1, "sn": 0, "sd": 9},
        {"prompt_id": "b", "fn": 0, "fd": 9, "sn": 1, "sd": 1},
    ]
    result = analysis.cluster_bootstrap_ratio_differences(
        rows,
        iterations=500,
        seed=11,
        specs={"delta": lambda row: (row["fn"], row["fd"], row["sn"], row["sd"])},
    )

    assert result["delta"]["estimate"] == pytest.approx(0.0)
    assert result["delta"]["cluster_count"] == 2


def test_tie_transition_boundary_uses_same_float_rule_as_evaluator_metrics() -> None:
    stale = [
        _receipt("a", 1, 10, [["s", 0]]),
        _receipt("b", 11, 100, [["s", 1]]),
    ]
    fresh = [
        _receipt("a", 0, 1, [["f", 0]]),
        _receipt("b", 1, 1, [["f", 1]]),
    ]
    group = {
        "global_step": 3,
        "prompt_id": "p",
        "fresh_creation_update": 3,
        "stale_creation_update": 1,
        "evaluator_age_steps": 2,
        "stale": stale,
        "fresh": fresh,
    }
    row = analysis.group_row(group, "inference_b", epsilon_z=0.01, epsilon_t=0.01)

    assert row["stale_ties"] - row["fresh_ties"] == row["ties_resolved"] - row["ties_created"]
