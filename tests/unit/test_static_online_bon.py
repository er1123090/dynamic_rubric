from __future__ import annotations

import pytest

from dynamic_rubric.minimum_staleness import MinimumExperimentError
from dynamic_rubric.static_online_bon import weighted_expanded_criteria


def test_weighted_expansion_reuses_ids_for_one_qwen_call_and_weighted_mean() -> None:
    expanded = weighted_expanded_criteria(
        "prompt-1",
        3,
        "pi_ref",
        [
            {"text": "major", "weight": 3},
            {"text": "minor", "weight": 1},
        ],
    )

    ids = [row["criterion_id"] for row in expanded]
    assert len(expanded) == 4
    assert len(set(ids)) == 2
    assert ids[:3] == [ids[0]] * 3
    probabilities = {ids[0]: 1.0, ids[-1]: 0.0}
    assert sum(probabilities[criterion_id] for criterion_id in ids) / len(ids) == 0.75


@pytest.mark.parametrize("weight", [0, -1, 1.5, True])
def test_weighted_expansion_rejects_invalid_weight(weight: object) -> None:
    with pytest.raises(MinimumExperimentError, match="invalid OnlineRubric weight"):
        weighted_expanded_criteria(
            "prompt-1",
            3,
            "pi_old",
            [{"text": "criterion", "weight": weight}],
        )
