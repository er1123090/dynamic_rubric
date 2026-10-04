"""Deterministic shared-pool best-of-N selection."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
import math
import random
from typing import Any


BON_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)


def _candidate_order(candidate_id: Any) -> tuple[int, Any]:
    if isinstance(candidate_id, bool):
        return (2, str(candidate_id))
    if isinstance(candidate_id, int):
        return (0, candidate_id)
    text = str(candidate_id)
    try:
        return (0, int(text))
    except ValueError:
        return (1, text)


def _seed_value(seed: int | str, permutation_index: int) -> int:
    digest = hashlib.sha256(f"bon-permutation\0{seed}\0{permutation_index}".encode()).digest()
    return int.from_bytes(digest[:8], "big")


def fixed_candidate_permutations(
    candidate_ids: Sequence[Any], *, seed: int | str, count: int = 5
) -> tuple[tuple[Any, ...], ...]:
    """Create stable permutations independent of process hash randomization."""

    if count <= 0:
        raise ValueError("count must be positive")
    canonical = sorted(candidate_ids, key=_candidate_order)
    if len(set(map(str, canonical))) != len(canonical):
        raise ValueError("candidate IDs must be unique")
    permutations = []
    for index in range(count):
        current = list(canonical)
        random.Random(_seed_value(seed, index)).shuffle(current)
        permutations.append(tuple(current))
    return tuple(permutations)


def select_best_of_n(scores: Mapping[Any, float], permutation: Sequence[Any], n: int) -> Any:
    """Select within the first N candidates, breaking score ties by lowest ID."""

    if n not in BON_SIZES:
        raise ValueError(f"N must be one of {BON_SIZES}")
    if n > len(permutation):
        raise ValueError("N exceeds candidate pool size")
    eligible = permutation[:n]
    if len(set(eligible)) != len(eligible):
        raise ValueError("permutation contains duplicate candidate IDs")
    try:
        pairs = [(candidate_id, float(scores[candidate_id])) for candidate_id in eligible]
    except KeyError as exc:
        raise KeyError(f"missing score for candidate {exc.args[0]!r}") from exc
    if any(not math.isfinite(score) for _, score in pairs):
        raise ValueError("candidate scores must be finite")
    best_score = max(score for _, score in pairs)
    return min(
        (candidate_id for candidate_id, score in pairs if score == best_score), key=_candidate_order
    )


def selections_for_grid(scores: Mapping[Any, float], permutation: Sequence[Any]) -> dict[int, Any]:
    return {n: select_best_of_n(scores, permutation, n) for n in BON_SIZES if n <= len(permutation)}


def shared_pool_hash(candidates: Sequence[Mapping[str, Any]]) -> str:
    """Hash IDs and response bytes in ID order to prove rubric pool identity."""

    normalized = []
    for candidate in candidates:
        if "candidate_id" not in candidate or "response_text" not in candidate:
            raise ValueError("each candidate needs candidate_id and response_text")
        normalized.append((candidate["candidate_id"], candidate["response_text"]))
    normalized.sort(key=lambda item: _candidate_order(item[0]))
    encoded = json.dumps(normalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
