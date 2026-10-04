from __future__ import annotations

import pytest

from dynamic_rubric.fake_gold_audit import evaluate_fake_gold


def test_fake_audit_uses_signed_healthbench_points_per_criterion() -> None:
    score, evidence = evaluate_fake_gold(
        "prompt",
        "candidate answer",
        (
            {"criterion": "states the required fact", "points": 2},
            {"criterion": "contains a harmful claim", "points": -1},
        ),
    )
    achieved = sum(row["probability_met"] * row["points"] for row in evidence)
    assert score == pytest.approx(min(1.0, max(0.0, achieved / 2.0)))
    assert {row["points"] for row in evidence} == {2.0, -1.0}
