"""Pure-Python reproduction of pinned veRL GRPO scalar advantages."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Sequence


VERL_COMMIT = "890dfc3ebdd5647f7ea9730375414b1e3fb4e9a6"
VERL_CORE_ALGOS_SHA256 = "0ec87b90cba24fa0aa959adff573efb2cc49434d705621cf546be231e56a8932"
VERL_GROUPWISE_SHA256 = "409403d0002ff0d1fd1c41812163b1cfa52a52248677aab00e6ffcf3b937b442"
DEFAULT_EPSILON = 1e-6


def sample_std(values: Sequence[float]) -> float:
    if len(values) <= 1:
        return 1.0
    mean = sum(float(value) for value in values) / len(values)
    variance = sum((float(value) - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(max(variance, 0.0))


def grpo_scalar_advantages(
    rewards: Sequence[float],
    *,
    epsilon: float = DEFAULT_EPSILON,
    normalize_by_std: bool = True,
) -> tuple[float, ...]:
    if not rewards:
        raise ValueError("rewards must be non-empty")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")
    values = tuple(float(value) for value in rewards)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("rewards must be finite")
    if len(values) == 1:
        mean, std = 0.0, 1.0
    else:
        mean, std = sum(values) / len(values), sample_std(values)
    if normalize_by_std:
        return tuple((value - mean) / (std + epsilon) for value in values)
    return tuple(value - mean for value in values)


def advantage_is_degenerate(
    rewards: Sequence[float], *, delta: float = 1e-8, epsilon: float = DEFAULT_EPSILON
) -> bool:
    if delta < 0:
        raise ValueError("delta must be non-negative")
    return max(abs(value) for value in grpo_scalar_advantages(rewards, epsilon=epsilon)) <= delta


def verify_pinned_verl_sources(root: Path) -> dict[str, str]:
    files = {
        "core_algos": (
            root / "verl/trainer/ppo/core_algos.py",
            VERL_CORE_ALGOS_SHA256,
        ),
        "groupwise": (root / "verl/utils/groupwise.py", VERL_GROUPWISE_SHA256),
    }
    observed: dict[str, str] = {}
    for name, (path, expected) in files.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError(f"pinned veRL {name} source hash differs")
        observed[name] = digest
    return observed
