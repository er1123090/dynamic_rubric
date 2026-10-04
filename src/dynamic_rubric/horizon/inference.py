"""Crossed seed-by-prompt inference and pre-registered horizon decisions."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random


DEFAULT_BOOTSTRAP_SEED = 20260829


@dataclass(frozen=True)
class ProcessObservation:
    seed_id: str
    prompt_id: str
    checkpoint: float
    d_r0: float
    g_refresh: float
    g_count: float


@dataclass(frozen=True)
class ConfidenceInterval:
    point: float
    low: float
    high: float


@dataclass(frozen=True)
class CrossedBootstrapResult:
    checkpoints: tuple[float, ...]
    bands: Mapping[str, Mapping[float, ConfidenceInterval]]
    pointwise: Mapping[str, Mapping[float, ConfidenceInterval]]
    simultaneous_quantiles: Mapping[str, float]
    seed_count: int
    prompt_count: int
    iterations: int
    bootstrap_seed: int
    small_seed_cluster_warning: bool


@dataclass(frozen=True)
class HorizonDecision:
    status: str
    t_star: float | None
    t_refresh: float | None
    refresh_status: str
    early_equivalence_checkpoint: float | None
    candidate_checkpoint: float | None
    core_qualifying_checkpoints: tuple[float, ...]
    reason: str


def make_process_observation(
    *,
    seed_id: str,
    prompt_id: str,
    checkpoint: float,
    r0_zar: float,
    r0_baseline_zar: float,
    current_zar: float,
    control_zar: float,
) -> ProcessObservation:
    """Build the three paired processes from prompt-level exact-ZAR indicators."""

    values = (r0_zar, r0_baseline_zar, current_zar, control_zar)
    if not seed_id or not prompt_id or any(value not in (0, 1, 0.0, 1.0) for value in values):
        raise ValueError("IDs must be non-empty and prompt-level ZAR values must be binary")
    return ProcessObservation(
        seed_id=str(seed_id),
        prompt_id=str(prompt_id),
        checkpoint=float(checkpoint),
        d_r0=float(r0_zar) - float(r0_baseline_zar),
        g_refresh=float(r0_zar) - float(current_zar),
        g_count=float(control_zar) - float(current_zar),
    )


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def crossed_bootstrap(
    observations: Sequence[ProcessObservation],
    *,
    iterations: int = 10_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> CrossedBootstrapResult:
    """Independently resample seeds and prompts, retaining all paired trajectories."""

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    if not observations:
        raise ValueError("observations must not be empty")

    seeds = tuple(sorted({row.seed_id for row in observations}))
    prompts = tuple(sorted({row.prompt_id for row in observations}))
    checkpoints = tuple(sorted({float(row.checkpoint) for row in observations}))
    cells: dict[tuple[str, str, float], ProcessObservation] = {}
    for row in observations:
        key = (row.seed_id, row.prompt_id, float(row.checkpoint))
        if key in cells:
            raise ValueError(f"duplicate seed/prompt/checkpoint cell: {key}")
        if any(not math.isfinite(value) for value in (row.d_r0, row.g_refresh, row.g_count)):
            raise ValueError("process values must be finite")
        cells[key] = row
    expected = len(seeds) * len(prompts) * len(checkpoints)
    if len(cells) != expected:
        raise ValueError("observations must form a complete seed x prompt x checkpoint grid")

    process_names = ("d_r0", "g_refresh", "g_count")

    def estimate(sampled_seeds: Sequence[str], sampled_prompts: Sequence[str]) -> dict[str, dict[float, float]]:
        denominator = len(sampled_seeds) * len(sampled_prompts)
        return {
            process: {
                checkpoint: math.fsum(
                    getattr(cells[(seed_id, prompt_id, checkpoint)], process)
                    for seed_id in sampled_seeds
                    for prompt_id in sampled_prompts
                )
                / denominator
                for checkpoint in checkpoints
            }
            for process in process_names
        }

    point = estimate(seeds, prompts)
    rng = random.Random(seed)
    replicates: dict[str, dict[float, list[float]]] = {
        process: {checkpoint: [] for checkpoint in checkpoints} for process in process_names
    }
    max_errors: dict[str, list[float]] = {process: [] for process in process_names}
    for _ in range(iterations):
        sampled_seeds = tuple(rng.choice(seeds) for _ in seeds)
        sampled_prompts = tuple(rng.choice(prompts) for _ in prompts)
        replicate = estimate(sampled_seeds, sampled_prompts)
        for process in process_names:
            errors = []
            for checkpoint in checkpoints:
                value = replicate[process][checkpoint]
                replicates[process][checkpoint].append(value)
                errors.append(abs(value - point[process][checkpoint]))
            max_errors[process].append(max(errors))

    alpha = (1.0 - confidence) / 2.0
    quantiles = {
        process: _quantile(errors, confidence) for process, errors in max_errors.items()
    }
    bands = {
        process: {
            checkpoint: ConfidenceInterval(
                point=point[process][checkpoint],
                low=point[process][checkpoint] - quantiles[process],
                high=point[process][checkpoint] + quantiles[process],
            )
            for checkpoint in checkpoints
        }
        for process in process_names
    }
    pointwise = {
        process: {
            checkpoint: ConfidenceInterval(
                point=point[process][checkpoint],
                low=_quantile(replicates[process][checkpoint], alpha),
                high=_quantile(replicates[process][checkpoint], 1.0 - alpha),
            )
            for checkpoint in checkpoints
        }
        for process in process_names
    }
    return CrossedBootstrapResult(
        checkpoints=checkpoints,
        bands=bands,
        pointwise=pointwise,
        simultaneous_quantiles=quantiles,
        seed_count=len(seeds),
        prompt_count=len(prompts),
        iterations=iterations,
        bootstrap_seed=seed,
        small_seed_cluster_warning=len(seeds) <= 3,
    )


def decide_horizon(
    result: CrossedBootstrapResult,
    *,
    deterioration_margin: float = 0.03,
    refresh_margin: float = 0.03,
    equivalence_margin: float = 0.015,
    gate_passed: Mapping[float, bool] | None = None,
) -> HorizonDecision:
    """Apply the pre-registered onset, equivalence, persistence, and censoring rules."""

    if min(deterioration_margin, refresh_margin, equivalence_margin) < 0.0:
        raise ValueError("decision margins must be non-negative")
    checkpoints = result.checkpoints
    if len(checkpoints) < 2:
        raise ValueError("at least two target checkpoints are required")
    required = {"d_r0", "g_refresh", "g_count"}
    if set(result.bands) != required or set(result.pointwise) != required:
        raise ValueError("bootstrap result is missing a required process")
    gates = {checkpoint: True for checkpoint in checkpoints}
    if gate_passed is not None:
        unknown = set(gate_passed) - set(checkpoints)
        if unknown:
            raise ValueError(f"gate results contain unknown checkpoints: {sorted(unknown)}")
        gates.update({float(key): bool(value) for key, value in gate_passed.items()})

    core = {
        checkpoint: (
            result.bands["d_r0"][checkpoint].low > deterioration_margin
            and result.bands["g_refresh"][checkpoint].low > refresh_margin
            and result.bands["g_count"][checkpoint].low > 0.0
            and gates[checkpoint]
        )
        for checkpoint in checkpoints
    }
    refresh_core = {
        checkpoint: (
            result.bands["g_refresh"][checkpoint].low > refresh_margin
            and result.bands["g_count"][checkpoint].low > 0.0
            and gates[checkpoint]
        )
        for checkpoint in checkpoints
    }
    equivalent = {
        checkpoint: (
            result.pointwise["g_refresh"][checkpoint].low >= -equivalence_margin
            and result.pointwise["g_refresh"][checkpoint].high <= equivalence_margin
        )
        for checkpoint in checkpoints
    }

    refresh_candidates = [
        checkpoint
        for index, checkpoint in enumerate(checkpoints[:-1])
        if refresh_core[checkpoint] and refresh_core[checkpoints[index + 1]]
    ]
    t_refresh = refresh_candidates[0] if refresh_candidates else None
    refresh_status = "observed" if t_refresh is not None else "not_observed"
    if t_refresh is None and refresh_core[checkpoints[-1]]:
        refresh_status = "candidate_onset_right_censored_at_final"

    horizon_candidates = []
    for index, checkpoint in enumerate(checkpoints[:-1]):
        has_earlier_equivalence = any(
            earlier > 0.0 and equivalent[earlier] for earlier in checkpoints[:index]
        )
        if core[checkpoint] and core[checkpoints[index + 1]] and has_earlier_equivalence:
            horizon_candidates.append(checkpoint)
    if horizon_candidates:
        t_star = horizon_candidates[0]
        earlier_equivalence = next(
            checkpoint
            for checkpoint in checkpoints
            if 0.0 < checkpoint < t_star and equivalent[checkpoint]
        )
        return HorizonDecision(
            status="observed",
            t_star=t_star,
            t_refresh=t_refresh,
            refresh_status=refresh_status,
            early_equivalence_checkpoint=earlier_equivalence,
            candidate_checkpoint=t_star,
            core_qualifying_checkpoints=tuple(cp for cp in checkpoints if core[cp]),
            reason="all deterioration, refresh, count-control, equivalence, and persistence gates passed",
        )

    first = checkpoints[0]
    if core[first] and core[checkpoints[1]]:
        return HorizonDecision(
            status="left_censored_before_first_checkpoint",
            t_star=None,
            t_refresh=t_refresh,
            refresh_status=refresh_status,
            early_equivalence_checkpoint=None,
            candidate_checkpoint=first,
            core_qualifying_checkpoints=tuple(cp for cp in checkpoints if core[cp]),
            reason="onset is already present at the first checkpoint, so early equivalence is unobserved",
        )
    final = checkpoints[-1]
    if core[final] and not core[checkpoints[-2]]:
        return HorizonDecision(
            status="candidate_onset_right_censored_at_final",
            t_star=None,
            t_refresh=t_refresh,
            refresh_status=refresh_status,
            early_equivalence_checkpoint=next(
                (
                    checkpoint
                    for checkpoint in checkpoints[:-1]
                    if checkpoint > 0.0 and equivalent[checkpoint]
                ),
                None,
            ),
            candidate_checkpoint=final,
            core_qualifying_checkpoints=tuple(cp for cp in checkpoints if core[cp]),
            reason="first qualifying onset is final and cannot satisfy next-checkpoint persistence",
        )
    return HorizonDecision(
        status="not_observed",
        t_star=None,
        t_refresh=t_refresh,
        refresh_status=refresh_status,
        early_equivalence_checkpoint=next(
            (checkpoint for checkpoint in checkpoints if checkpoint > 0.0 and equivalent[checkpoint]),
            None,
        ),
        candidate_checkpoint=None,
        core_qualifying_checkpoints=tuple(cp for cp in checkpoints if core[cp]),
        reason="no checkpoint satisfies every pre-registered horizon condition",
    )
