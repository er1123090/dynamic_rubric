"""Fresh/stale evaluator identity contracts for Phase-1 shadow scoring."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence


class ShadowEvaluatorError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class OnlineRubricSnapshot:
    prompt_id: str
    evaluator_checkpoint: str
    global_step: int
    visit_index: int
    rubric_id: str
    criteria: tuple[Mapping[str, Any], ...]
    created_from_pool_a_response_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.prompt_id or not self.evaluator_checkpoint or not self.rubric_id:
            raise ShadowEvaluatorError("rubric snapshot identities must be non-empty")
        if self.global_step < 0 or self.visit_index < 0:
            raise ShadowEvaluatorError("snapshot step and visit must be non-negative")
        if len(self.created_from_pool_a_response_ids) != len(
            set(self.created_from_pool_a_response_ids)
        ):
            raise ShadowEvaluatorError("Pool-A provenance IDs must be unique")

    def record(self) -> dict[str, Any]:
        return asdict(self)


class OnlineRubricShadowCache:
    """Prompt-indexed rubric history; cross-prompt stale lookup is impossible."""

    def __init__(self) -> None:
        self._by_prompt: dict[str, list[OnlineRubricSnapshot]] = {}

    def add(self, snapshot: OnlineRubricSnapshot) -> None:
        history = self._by_prompt.setdefault(snapshot.prompt_id, [])
        if history and snapshot.global_step <= history[-1].global_step:
            raise ShadowEvaluatorError(
                "rubric snapshots must be appended in strictly increasing step order"
            )
        if any(item.rubric_id == snapshot.rubric_id for item in history):
            raise ShadowEvaluatorError("rubric_id already exists for this prompt")
        history.append(snapshot)

    def latest_before(
        self, prompt_id: str, *, global_step: int
    ) -> OnlineRubricSnapshot | None:
        candidates = [
            item
            for item in self._by_prompt.get(prompt_id, ())
            if item.global_step < global_step
        ]
        return candidates[-1] if candidates else None

    def history(self, prompt_id: str) -> tuple[OnlineRubricSnapshot, ...]:
        return tuple(self._by_prompt.get(prompt_id, ()))

    def records(self) -> list[dict[str, Any]]:
        return [
            snapshot.record()
            for prompt_id in sorted(self._by_prompt)
            for snapshot in self._by_prompt[prompt_id]
        ]


@dataclass(frozen=True, slots=True)
class EvoScoreContext:
    evaluator_checkpoint: str
    evaluator_step: int
    policy_checkpoint: str
    response_ids: tuple[str, ...]
    judge_model: str
    rubric_sets_n: int
    rubric_generation_seeds: tuple[int, ...]


def validate_evo_fresh_stale_pair(
    stale: EvoScoreContext,
    fresh: EvoScoreContext,
) -> None:
    if stale.evaluator_step >= fresh.evaluator_step:
        raise ShadowEvaluatorError("Evo stale evaluator must precede fresh evaluator")
    invariants = (
        "policy_checkpoint",
        "response_ids",
        "judge_model",
        "rubric_sets_n",
        "rubric_generation_seeds",
    )
    for name in invariants:
        if getattr(stale, name) != getattr(fresh, name):
            raise ShadowEvaluatorError(
                f"Evo fresh/stale comparison changed invariant {name}"
            )
    if len(fresh.rubric_generation_seeds) != fresh.rubric_sets_n:
        raise ShadowEvaluatorError("Evo N must match the fixed generation seed count")


def validate_same_pool_b(
    stale_records: Sequence[Mapping[str, Any]],
    fresh_records: Sequence[Mapping[str, Any]],
) -> None:
    stale_ids = tuple(str(row["response_id"]) for row in stale_records)
    fresh_ids = tuple(str(row["response_id"]) for row in fresh_records)
    if stale_ids != fresh_ids:
        raise ShadowEvaluatorError(
            "fresh and stale evaluators must score identical ordered Pool-B responses"
        )
    if {str(row.get("pool")) for row in (*stale_records, *fresh_records)} != {
        "probe_B"
    }:
        raise ShadowEvaluatorError("fresh/stale comparison is restricted to probe_B")
