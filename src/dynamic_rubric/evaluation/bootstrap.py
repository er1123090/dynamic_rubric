"""Prompt-clustered paired bootstrap and decision classification."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import random


DEFAULT_BOOTSTRAP_SEED = 20250807
EQUIVALENCE_MARGIN = 0.015
MEANINGFUL_DIFFERENCE = 0.03


@dataclass(frozen=True)
class BootstrapResult:
    point_estimate: float
    ci_low: float
    ci_high: float
    classification: str
    n_prompt_clusters: int
    iterations: int
    seed: int

    @property
    def excludes_zero(self) -> bool:
        return self.ci_low > 0.0 or self.ci_high < 0.0


def classify_difference(point_estimate: float, ci_low: float, ci_high: float) -> str:
    if ci_low >= -EQUIVALENCE_MARGIN and ci_high <= EQUIVALENCE_MARGIN:
        return "local_equivalence"
    if abs(point_estimate) >= MEANINGFUL_DIFFERENCE and (ci_low > 0.0 or ci_high < 0.0):
        return "meaningful_difference"
    return "inconclusive"


def paired_prompt_bootstrap(
    pairs: Mapping[str, tuple[float, float]] | Iterable[tuple[str, float, float]],
    *,
    iterations: int = 10_000,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> BootstrapResult:
    """Bootstrap paired differences by prompt ID, preserving prompt clusters.

    Multiple observations for a prompt are reduced to their within-prompt mean
    difference before clusters are sampled with replacement.
    """

    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    grouped: dict[str, list[float]] = defaultdict(list)
    items = (
        ((key, value[0], value[1]) for key, value in pairs.items())
        if isinstance(pairs, Mapping)
        else pairs
    )
    for prompt_id, current, baseline in items:
        difference = float(current) - float(baseline)
        if not prompt_id or not math.isfinite(difference):
            raise ValueError("prompt IDs and paired values must be valid")
        grouped[str(prompt_id)].append(difference)
    if not grouped:
        raise ValueError("at least one prompt cluster is required")
    cluster_differences = [math.fsum(values) / len(values) for _, values in sorted(grouped.items())]
    point = math.fsum(cluster_differences) / len(cluster_differences)
    rng = random.Random(seed)
    count = len(cluster_differences)
    estimates = []
    for _ in range(iterations):
        estimates.append(math.fsum(rng.choice(cluster_differences) for _ in range(count)) / count)
    estimates.sort()
    alpha = (1.0 - confidence) / 2.0
    low = _quantile(estimates, alpha)
    high = _quantile(estimates, 1.0 - alpha)
    return BootstrapResult(
        point_estimate=point,
        ci_low=low,
        ci_high=high,
        classification=classify_difference(point, low, high),
        n_prompt_clusters=count,
        iterations=iterations,
        seed=seed,
    )


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    fraction = position - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction
