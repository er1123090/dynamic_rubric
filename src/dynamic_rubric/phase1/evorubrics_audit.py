"""Strict score aggregation for EvoRubrics fixed-probe RQ2 audits.

The live scorer can emit one :class:`RubricGradeRecord` for every
response/rubric pair.  This module validates the complete 16 x 4 cross-score
grid, reproduces the public EvoRubrics signed-weight normalization, and then
computes paired fresh/stale metrics without treating repeated ZAR@4 subsets as
independent observations.
"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .audit_analysis import prompt_bootstrap
from .metrics import MetricContractError, ScoreGroup, compare_fresh_stale

DEFAULT_POOL_B_RESPONSES = 16
DEFAULT_RUBRIC_SETS = 4
DEFAULT_ZAR4_REPETITIONS = 1_000
DEFAULT_ZAR4_SEED = 20_260_610


class EvoAuditContractError(MetricContractError):
    """Raised when EvoRubrics grades cannot support an aligned audit."""


@dataclass(frozen=True, slots=True)
class CriterionJudgment:
    criterion_id: str
    weight: float
    grade: int | bool


@dataclass(frozen=True, slots=True)
class RubricGradeRecord:
    """One judge receipt for one response scored by one generated rubric set."""

    policy_step: int
    evaluator_step: int
    policy_checkpoint: str
    evaluator_checkpoint: str
    prompt_id: str
    response_id: str
    rubric_id: str
    judgments: tuple[CriterionJudgment, ...]
    parse_ok: bool = True


def _required_text(value: object, name: str) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise EvoAuditContractError(f"{name} must be non-empty")
    return normalized


def normalized_rubric_reward(judgments: Sequence[CriterionJudgment]) -> float:
    """Apply the released EvoRubrics signed-range normalization.

    The minimum attainable score is the sum of negative weights and the
    maximum is the sum of positive weights.  This differs from dividing only
    by positive weight and deliberately supports negative criteria.
    """

    if not judgments:
        raise EvoAuditContractError("a rubric grade must contain criteria")
    criterion_ids: set[str] = set()
    total = positive_sum = negative_sum = 0.0
    for judgment in judgments:
        criterion_id = _required_text(judgment.criterion_id, "criterion_id")
        if criterion_id in criterion_ids:
            raise EvoAuditContractError(f"duplicate criterion_id: {criterion_id}")
        criterion_ids.add(criterion_id)
        weight = float(judgment.weight)
        if not math.isfinite(weight):
            raise EvoAuditContractError("criterion weights must be finite")
        if judgment.grade not in (0, 1, False, True):
            raise EvoAuditContractError("criterion grades must be binary")
        grade = int(judgment.grade)
        total += weight * grade
        positive_sum += max(0.0, weight)
        negative_sum += min(0.0, weight)
    denominator = positive_sum - negative_sum
    if denominator <= 0.0:
        raise EvoAuditContractError("rubric weights have no non-zero scoring range")
    return min(1.0, max(0.0, (total - negative_sum) / denominator))


def aggregate_evo_score_group(
    records: Sequence[RubricGradeRecord],
    *,
    expected_response_ids: Sequence[str],
    expected_rubric_ids: Sequence[str],
    expected_response_count: int = DEFAULT_POOL_B_RESPONSES,
    expected_rubric_count: int = DEFAULT_RUBRIC_SETS,
) -> ScoreGroup:
    """Validate and average exactly N rubric rewards for every Pool-B response."""

    response_ids = tuple(_required_text(value, "response_id") for value in expected_response_ids)
    rubric_ids = tuple(_required_text(value, "rubric_id") for value in expected_rubric_ids)
    if len(response_ids) != expected_response_count or len(set(response_ids)) != len(response_ids):
        raise EvoAuditContractError(
            f"expected {expected_response_count} unique ordered Pool-B response IDs"
        )
    if len(rubric_ids) != expected_rubric_count or len(set(rubric_ids)) != len(rubric_ids):
        raise EvoAuditContractError(f"expected {expected_rubric_count} distinct rubric IDs")
    if not records:
        raise EvoAuditContractError("no rubric grade records supplied")

    first = records[0]
    identity = (
        int(first.policy_step),
        int(first.evaluator_step),
        _required_text(first.policy_checkpoint, "policy_checkpoint"),
        _required_text(first.evaluator_checkpoint, "evaluator_checkpoint"),
        _required_text(first.prompt_id, "prompt_id"),
    )
    expected_responses = set(response_ids)
    expected_rubrics = set(rubric_ids)
    by_pair: dict[tuple[str, str], RubricGradeRecord] = {}
    inventories: dict[str, tuple[tuple[str, float], ...]] = {}
    for record in records:
        current_identity = (
            int(record.policy_step),
            int(record.evaluator_step),
            _required_text(record.policy_checkpoint, "policy_checkpoint"),
            _required_text(record.evaluator_checkpoint, "evaluator_checkpoint"),
            _required_text(record.prompt_id, "prompt_id"),
        )
        if current_identity != identity:
            raise EvoAuditContractError("grade records must share checkpoint and prompt identity")
        response_id = _required_text(record.response_id, "response_id")
        rubric_id = _required_text(record.rubric_id, "rubric_id")
        if response_id not in expected_responses:
            raise EvoAuditContractError(f"unexpected response_id: {response_id}")
        if rubric_id not in expected_rubrics:
            raise EvoAuditContractError(f"unexpected rubric_id: {rubric_id}")
        if not record.parse_ok:
            raise EvoAuditContractError(
                f"judge parse failed for response/rubric pair {(response_id, rubric_id)}"
            )
        pair = response_id, rubric_id
        if pair in by_pair:
            raise EvoAuditContractError(f"duplicate response/rubric grade: {pair}")
        by_pair[pair] = record
        inventory = tuple(
            (_required_text(item.criterion_id, "criterion_id"), float(item.weight))
            for item in record.judgments
        )
        if rubric_id in inventories and inventories[rubric_id] != inventory:
            raise EvoAuditContractError(f"criterion inventory drift for rubric_id: {rubric_id}")
        inventories[rubric_id] = inventory

    expected_pairs = {
        (response_id, rubric_id) for response_id in response_ids for rubric_id in rubric_ids
    }
    missing = sorted(expected_pairs - set(by_pair))
    if missing:
        raise EvoAuditContractError(
            f"incomplete {expected_response_count}x{expected_rubric_count} grade grid; "
            f"missing {len(missing)} pair(s), first={missing[0]}"
        )

    rewards: list[float] = []
    criterion_grades: dict[str, list[int]] = defaultdict(list)
    for response_id in response_ids:
        rubric_rewards = []
        for rubric_id in rubric_ids:
            record = by_pair[(response_id, rubric_id)]
            rubric_rewards.append(normalized_rubric_reward(record.judgments))
            for judgment in record.judgments:
                criterion_grades[f"{rubric_id}/{judgment.criterion_id}"].append(int(judgment.grade))
        rewards.append(math.fsum(rubric_rewards) / expected_rubric_count)

    return ScoreGroup(
        prompt_id=identity[4],
        evaluator_checkpoint=identity[3],
        policy_checkpoint=identity[2],
        response_ids=response_ids,
        rewards=tuple(rewards),
        criterion_grades={key: tuple(values) for key, values in criterion_grades.items()},
    )


def deterministic_subset_indices(
    *,
    population_size: int = DEFAULT_POOL_B_RESPONSES,
    subset_size: int = 4,
    repetitions: int = DEFAULT_ZAR4_REPETITIONS,
    seed: int = DEFAULT_ZAR4_SEED,
) -> tuple[tuple[int, ...], ...]:
    """Draw deterministic repeated subsets, without replacement within each subset."""

    if population_size < 2:
        raise EvoAuditContractError("population_size must be at least two")
    if subset_size < 2 or subset_size > population_size:
        raise EvoAuditContractError("subset_size must be between two and population_size")
    if repetitions <= 0:
        raise EvoAuditContractError("repetitions must be positive")
    rng = random.Random(int(seed))
    return tuple(
        tuple(sorted(rng.sample(range(population_size), subset_size))) for _ in range(repetitions)
    )


def _subset_group(group: ScoreGroup, indices: Sequence[int]) -> ScoreGroup:
    criterion_grades = None
    if group.criterion_grades is not None:
        criterion_grades = {
            criterion_id: tuple(grades[index] for index in indices)
            for criterion_id, grades in group.criterion_grades.items()
        }
    return ScoreGroup(
        prompt_id=group.prompt_id,
        evaluator_checkpoint=group.evaluator_checkpoint,
        policy_checkpoint=group.policy_checkpoint,
        response_ids=tuple(group.response_ids[index] for index in indices),
        rewards=tuple(group.rewards[index] for index in indices),
        criterion_grades=criterion_grades,
    )


def compare_evo_prompt(
    stale: ScoreGroup,
    fresh: ScoreGroup,
    *,
    epsilon_z: float,
    epsilon_t: float,
    subset_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    """Compare one prompt using paired ZAR@4 draws and full Pool-B diagnostics."""

    if len(stale.response_ids) != DEFAULT_POOL_B_RESPONSES:
        raise EvoAuditContractError(
            f"Evo fixed-probe comparisons require {DEFAULT_POOL_B_RESPONSES} Pool-B responses"
        )
    full = compare_fresh_stale(stale, fresh, epsilon_z=epsilon_z, epsilon_t=epsilon_t)
    if not subset_indices:
        raise EvoAuditContractError("at least one ZAR@4 subset is required")
    subset_comparisons = []
    for indices in subset_indices:
        ordered = tuple(int(index) for index in indices)
        if len(ordered) != 4 or len(set(ordered)) != 4:
            raise EvoAuditContractError("every ZAR@4 draw must contain four distinct indices")
        if min(ordered) < 0 or max(ordered) >= len(stale.response_ids):
            raise EvoAuditContractError("ZAR@4 subset index is outside Pool-B")
        subset_comparisons.append(
            compare_fresh_stale(
                _subset_group(stale, ordered),
                _subset_group(fresh, ordered),
                epsilon_z=epsilon_z,
                epsilon_t=epsilon_t,
            )
        )

    def mean(path: Sequence[str]) -> float | None:
        values: list[float] = []
        for row in subset_comparisons:
            value: Any = row
            for key in path:
                value = value[key]
            if value is not None:
                values.append(float(value))
        return math.fsum(values) / len(values) if values else None

    return {
        "prompt_id": fresh.prompt_id,
        "policy_checkpoint": fresh.policy_checkpoint,
        "stale_evaluator_checkpoint": stale.evaluator_checkpoint,
        "fresh_evaluator_checkpoint": fresh.evaluator_checkpoint,
        "response_ids": list(fresh.response_ids),
        "same_pool_b": True,
        "zar_at_4_repetitions": len(subset_comparisons),
        "stale_zar_at_4": mean(("stale", "exact_zero_advantage")),
        "fresh_zar_at_4": mean(("fresh", "exact_zero_advantage")),
        "v_adj_zar_at_4": mean(("v_adj_zar",)),
        "stale_near_zero_at_4": mean(("stale", "near_zero_advantage")),
        "fresh_near_zero_at_4": mean(("fresh", "near_zero_advantage")),
        "delta_near_zero_at_4": mean(("delta_near_zero_advantage",)),
        "stale_tie_rate_16": full["stale"]["pairwise_tie_rate"],
        "fresh_tie_rate_16": full["fresh"]["pairwise_tie_rate"],
        "delta_tie_rate_16": full["delta_tie_rate"],
        "stale_separation_rate_16": full["stale"]["pairwise_separation_rate"],
        "fresh_separation_rate_16": full["fresh"]["pairwise_separation_rate"],
        "delta_separation_rate_16": full["delta_separation_rate"],
        "stale_top_median_margin_16": full["stale"]["top_median_margin"],
        "fresh_top_median_margin_16": full["fresh"]["top_median_margin"],
        "delta_top_median_margin_16": full["delta_top_median_margin"],
        "kendall_tau_b_16": full["kendall_tau_b"],
        "ranking_similarity_is_correctness": False,
    }


def _cell_seed(seed: int, evaluator_step: int, policy_step: int) -> int:
    digest = hashlib.sha256(f"{evaluator_step}\0{policy_step}".encode()).digest()[:8]
    return int(seed) ^ int.from_bytes(digest, "big")


def analyze_evo_grade_matrix(
    records: Sequence[RubricGradeRecord],
    *,
    expected_pairs: Sequence[tuple[int, int]],
    expected_prompt_ids: Sequence[str],
    response_ids_by_policy_prompt: Mapping[tuple[int, str], Sequence[str]],
    rubric_ids_by_evaluator_prompt: Mapping[tuple[int, str], Sequence[str]],
    epsilon_z: float,
    epsilon_t: float,
    zar4_repetitions: int = DEFAULT_ZAR4_REPETITIONS,
    zar4_seed: int = DEFAULT_ZAR4_SEED,
    bootstrap_iterations: int = 10_000,
    bootstrap_seed: int = 20_250_807,
) -> dict[str, Any]:
    """Build an expected-cell-preserving matrix from live judge receipts.

    A wholly absent prompt grid is reported as pending.  A partially observed
    16 x 4 grid is a contract failure, preventing missing judge or parse results
    from changing the reward denominator silently.
    """

    pairs = tuple((int(evaluator), int(policy)) for evaluator, policy in expected_pairs)
    if not pairs or len(set(pairs)) != len(pairs):
        raise EvoAuditContractError("expected checkpoint pairs must be non-empty and unique")
    prompts = tuple(_required_text(value, "prompt_id") for value in expected_prompt_ids)
    if not prompts or len(set(prompts)) != len(prompts):
        raise EvoAuditContractError("expected prompt IDs must be non-empty and unique")
    policies = {policy for _, policy in pairs}
    if any((policy, policy) not in pairs for policy in policies):
        raise EvoAuditContractError("every policy checkpoint requires its fresh evaluator pair")

    indexed: dict[tuple[int, int, str], list[RubricGradeRecord]] = defaultdict(list)
    unexpected_pairs: set[tuple[int, int]] = set()
    unexpected_prompts: set[str] = set()
    for record in records:
        pair = int(record.evaluator_step), int(record.policy_step)
        prompt_id = _required_text(record.prompt_id, "prompt_id")
        if pair not in set(pairs):
            unexpected_pairs.add(pair)
        if prompt_id not in set(prompts):
            unexpected_prompts.add(prompt_id)
        indexed[(pair[0], pair[1], prompt_id)].append(record)
    if unexpected_pairs:
        raise EvoAuditContractError(f"unexpected checkpoint pairs: {sorted(unexpected_pairs)}")
    if unexpected_prompts:
        raise EvoAuditContractError(f"unexpected prompt IDs: {sorted(unexpected_prompts)}")

    groups: dict[tuple[int, int, str], ScoreGroup] = {}
    missing_by_pair: dict[tuple[int, int], list[str]] = defaultdict(list)
    for evaluator_step, policy_step in pairs:
        for prompt_id in prompts:
            key = evaluator_step, policy_step, prompt_id
            grade_rows = indexed.get(key, ())
            if not grade_rows:
                missing_by_pair[(evaluator_step, policy_step)].append(prompt_id)
                continue
            try:
                response_ids = response_ids_by_policy_prompt[(policy_step, prompt_id)]
                rubric_ids = rubric_ids_by_evaluator_prompt[(evaluator_step, prompt_id)]
            except KeyError as error:
                raise EvoAuditContractError(
                    f"missing ID manifest entry: {error.args[0]}"
                ) from error
            groups[key] = aggregate_evo_score_group(
                grade_rows,
                expected_response_ids=response_ids,
                expected_rubric_ids=rubric_ids,
            )

    cells: list[dict[str, Any]] = []
    prompt_comparisons: list[dict[str, Any]] = []
    complete_count = 0
    for evaluator_step, policy_step in pairs:
        missing = set(missing_by_pair[(evaluator_step, policy_step)])
        missing.update(missing_by_pair[(policy_step, policy_step)])
        if missing:
            cells.append(
                {
                    "evaluator_step": evaluator_step,
                    "policy_step": policy_step,
                    "status": "pending",
                    "expected_prompt_count": len(prompts),
                    "missing_prompt_ids": sorted(missing),
                }
            )
            continue
        subsets = deterministic_subset_indices(
            repetitions=zar4_repetitions,
            seed=_cell_seed(zar4_seed, evaluator_step, policy_step),
        )
        comparisons = [
            compare_evo_prompt(
                groups[(evaluator_step, policy_step, prompt_id)],
                groups[(policy_step, policy_step, prompt_id)],
                epsilon_z=epsilon_z,
                epsilon_t=epsilon_t,
                subset_indices=subsets,
            )
            for prompt_id in prompts
        ]

        metric_names = (
            "stale_zar_at_4",
            "fresh_zar_at_4",
            "v_adj_zar_at_4",
            "delta_near_zero_at_4",
            "delta_tie_rate_16",
            "delta_separation_rate_16",
            "delta_top_median_margin_16",
            "kendall_tau_b_16",
        )

        def prompt_mean(rows: Sequence[Mapping[str, Any]], name: str) -> float | None:
            values = [float(row[name]) for row in rows if row[name] is not None]
            return math.fsum(values) / len(values) if values else None

        metrics = {name: prompt_mean(comparisons, name) for name in metric_names}
        bootstrap = {
            name: prompt_bootstrap(
                comparisons,
                lambda rows, metric=name: prompt_mean(rows, metric),
                iterations=bootstrap_iterations,
                seed=_cell_seed(bootstrap_seed, evaluator_step, policy_step)
                ^ int.from_bytes(hashlib.sha256(name.encode()).digest()[:8], "big"),
            )
            for name in metric_names
        }
        cells.append(
            {
                "evaluator_step": evaluator_step,
                "policy_step": policy_step,
                "status": "complete",
                "prompt_count": len(comparisons),
                "pool_b_response_count": DEFAULT_POOL_B_RESPONSES,
                "rubric_set_count": DEFAULT_RUBRIC_SETS,
                "zar_at_4_repetitions": zar4_repetitions,
                "metrics": metrics,
                "bootstrap_95ci": bootstrap,
            }
        )
        prompt_comparisons.extend(
            {
                "evaluator_step": evaluator_step,
                "policy_step": policy_step,
                **comparison,
            }
            for comparison in comparisons
        )
        complete_count += 1

    return {
        "schema_version": 1,
        "method": "evorubrics",
        "expected_cell_count": len(pairs),
        "complete_cell_count": complete_count,
        "pending_cell_count": len(pairs) - complete_count,
        "expected_prompt_comparison_count": len(pairs) * len(prompts),
        "cells": cells,
        "prompt_comparisons": prompt_comparisons,
        "bootstrap_unit": "prompt",
        "zar_at_4_subset_draws_are_not_independent_observations": True,
        "same_pool_b_fresh_stale": True,
        "absolute_metric_scope": "within_method_only",
    }
