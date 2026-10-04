from __future__ import annotations

import pytest

from dynamic_rubric.minimum_interim import target_groups
from dynamic_rubric.minimum_staleness import MinimumExperimentError


def test_target_groups_supports_policy_step_subset() -> None:
    assert target_groups(("prompt-a", "prompt-b"), (3, 10)) == frozenset(
        {
            ("pi_3", "prompt-a"),
            ("pi_3", "prompt-b"),
            ("pi_10", "prompt-a"),
            ("pi_10", "prompt-b"),
        }
    )


@pytest.mark.parametrize("steps", [(), (3, 3), (7,)])
def test_target_groups_rejects_invalid_policy_steps(steps: tuple[int, ...]) -> None:
    with pytest.raises(MinimumExperimentError, match="invalid interim policy steps"):
        target_groups(("prompt-a",), steps)
