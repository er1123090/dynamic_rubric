"""RaR static-R0 hard-binary weighted rational training reward."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from dynamic_rubric.horizon.contracts import RationalScore, WeightedCriterion, WeightedRubric


TRAINING_REWARD_MODE = "hard_binary_weighted_rational_v1"
TRAINING_REWARD_SOURCE = "rar_static_r0_only"


class RaRRewardContractError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class RaRRewardConfig:
    rubric_path: Path
    reward_source: str = TRAINING_REWARD_SOURCE
    reward_mode: str = TRAINING_REWARD_MODE

    def validate(self) -> None:
        normalized = str(self.rubric_path).replace("\\", "/").casefold()
        if self.reward_source != TRAINING_REWARD_SOURCE:
            raise RaRRewardContractError("training reward source must be rar_static_r0_only")
        if self.reward_mode != TRAINING_REWARD_MODE:
            raise RaRRewardContractError(
                "training reward mode must be hard_binary_weighted_rational_v1"
            )
        if any(fragment in normalized for fragment in ("dynamic", "horizon", "replay", "gold")):
            raise RaRRewardContractError(
                "training cannot read dynamic, horizon, replay, or gold artifacts"
            )


def weighted_rational_score(
    criteria: Sequence[WeightedCriterion], grades: Mapping[str, int]
) -> RationalScore:
    try:
        return RationalScore.from_grades(criteria, grades)
    except ValueError as exc:
        raise RaRRewardContractError(str(exc)) from exc


def score_static_r0(rubric: WeightedRubric, grades: Mapping[str, int]) -> RationalScore:
    if any(
        criterion.source_checkpoint not in (None, "step0", "initial_policy")
        for criterion in rubric.criteria
    ):
        raise RaRRewardContractError("training rubric contains non-R0 criterion provenance")
    return weighted_rational_score(rubric.criteria, grades)


def reward_value(rubric: WeightedRubric, grades: Mapping[str, int]) -> float:
    """Return veRL-facing float while keeping exact numerator/denominator available upstream."""

    return score_static_r0(rubric, grades).value
