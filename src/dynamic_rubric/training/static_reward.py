from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from dynamic_rubric.providers.base import CriterionGrader


class StaticRewardContractError(ValueError):
    pass


@dataclass(frozen=True)
class StaticRewardConfig:
    rubric_path: Path
    reward_source: str = "static_r0_only"

    def validate(self) -> None:
        normalized = str(self.rubric_path).replace("\\", "/").lower()
        if self.reward_source != "static_r0_only":
            raise StaticRewardContractError("pilot training reward source must be static_r0_only")
        if any(fragment in normalized for fragment in ("dynamic", "replay", "private_gt", "gold")):
            raise StaticRewardContractError("training cannot read dynamic or hidden-GT artifacts")


def static_rubric_score(probabilities_yes: Sequence[float]) -> float:
    if not probabilities_yes:
        raise StaticRewardContractError("a static rubric must contain criteria")
    if any(not 0.0 <= value <= 1.0 for value in probabilities_yes):
        raise StaticRewardContractError("criterion probability is outside [0, 1]")
    return sum(probabilities_yes) / len(probabilities_yes)


def score_response(
    grader: CriterionGrader,
    prompt_id: str,
    response_id: str,
    response_text: str,
    rubric: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]]]:
    criteria = rubric.get("criteria")
    if not isinstance(criteria, Sequence) or len(criteria) != 8:
        raise StaticRewardContractError("R_0 must contain exactly 8 criteria")
    values = []
    for criterion in criteria:
        if not isinstance(criterion, Mapping):
            raise StaticRewardContractError("criterion must be an object")
        values.append(
            (
                prompt_id,
                response_id,
                response_text,
                str(criterion["criterion_id"]),
                str(criterion["text"]),
            )
        )
    score_many = getattr(grader, "score_many", None)
    if callable(score_many):
        scores = cast(Any, score_many)(values)
    else:
        scores = tuple(
            grader.score(prompt, response, text, criterion_id, criterion_text)
            for prompt, response, text, criterion_id, criterion_text in values
        )
    criterion_scores: list[dict[str, Any]] = []
    for score in scores:
        if not score.parse_success:
            raise StaticRewardContractError("criterion grader parse failure")
        criterion_scores.append(
            {
                "criterion_id": score.criterion_id,
                "probability_yes": score.probability_yes,
                "parse_success": score.parse_success,
            }
        )
    return static_rubric_score(
        [value["probability_yes"] for value in criterion_scores]
    ), criterion_scores


def make_verl_reward_function(
    grader: CriterionGrader,
    rubrics_by_prompt: Mapping[str, Mapping[str, Any]],
):
    """Return the narrow callable expected by veRL custom reward loading."""

    def reward_fn(
        data_source: str, solution_str: str, ground_truth: Any, extra_info: Mapping[str, Any]
    ):
        del data_source, ground_truth
        prompt_id = str(extra_info["prompt_id"])
        response_id = str(extra_info["response_id"])
        reward, _ = score_response(
            grader, prompt_id, response_id, solution_str, rubrics_by_prompt[prompt_id]
        )
        return reward

    return reward_fn
