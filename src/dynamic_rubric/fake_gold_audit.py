"""Credential-free hidden audit using the production HealthBench aggregation."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping, Sequence

from .evaluation.gold_score import weighted_gold_score
from .hashing import sha256_json


def _probability(*parts: object) -> float:
    digest = hashlib.sha256(sha256_json(parts).encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def evaluate_fake_gold(
    prompt_id: str,
    response_text: str,
    rubric: Sequence[Mapping[str, Any]],
) -> tuple[float, list[dict[str, Any]]]:
    """Grade each private criterion, then apply signed HealthBench weights."""

    scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    evidence: list[dict[str, Any]] = []
    for index, criterion in enumerate(rubric):
        text = str(criterion.get("criterion", criterion.get("text", "")))
        if not text:
            raise ValueError("gold criterion text is missing")
        criterion_id = hashlib.sha256(f"{prompt_id}\x1f{index}\x1f{text}".encode()).hexdigest()
        points = float(criterion.get("points", criterion.get("weight", 0.0)))
        probability = _probability(prompt_id, response_text, criterion_id, "fake/gpt-5-v1")
        scores[criterion_id] = probability
        weights[criterion_id] = points
        evidence.append(
            {
                "criterion_id_hash": criterion_id,
                "probability_met": probability,
                "points": points,
            }
        )
    return weighted_gold_score(scores, weights), evidence
