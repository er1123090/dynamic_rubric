"""Private hidden-gold weighted scoring and strict cache identity."""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math


def weighted_gold_score(
    criterion_scores: Mapping[str, float], criterion_weights: Mapping[str, float]
) -> float:
    """Compute the HealthBench-style normalized weighted score in [0, 1]."""

    if not criterion_weights:
        raise ValueError("gold rubric must contain at least one weighted criterion")
    if set(criterion_scores) != set(criterion_weights):
        raise ValueError("criterion score and weight IDs must match exactly")
    weights = {key: float(value) for key, value in criterion_weights.items()}
    scores = {key: float(value) for key, value in criterion_scores.items()}
    if any(not math.isfinite(value) for value in weights.values()):
        raise ValueError("criterion points must be finite")
    if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in scores.values()):
        raise ValueError("criterion scores must be finite values in [0, 1]")
    # HealthBench criteria may carry negative points. The official evaluator
    # applies achieved points with their sign, normalizes by the total possible
    # positive points, and clips the aggregate rather than individual criteria.
    denominator = math.fsum(max(value, 0.0) for value in weights.values())
    if denominator <= 0.0:
        raise ValueError("gold rubric must contain at least one positive-point criterion")
    numerator = math.fsum(scores[key] * weights[key] for key in sorted(weights))
    return min(1.0, max(0.0, numerator / denominator))


def gold_cache_key(
    *,
    prompt_id: str,
    response_text_hash: str,
    gold_rubric_hash: str,
    requested_model: str,
    returned_model: str,
    grader_prompt_hash: str,
    schema_hash: str,
    reasoning_effort: str,
) -> str:
    """Hash every setting whose change must invalidate a hidden-GT result."""

    fields = (
        prompt_id,
        response_text_hash,
        gold_rubric_hash,
        requested_model,
        returned_model,
        grader_prompt_hash,
        schema_hash,
        reasoning_effort,
    )
    if any(not isinstance(field, str) or not field for field in fields):
        raise ValueError("all gold cache identity fields must be non-empty strings")
    encoded = json.dumps(fields, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
