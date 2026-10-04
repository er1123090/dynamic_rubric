"""Deterministic policy-simulator metrics for the credential-free lane."""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

from .providers.fake import FakeCriterionGrader, FakeEmbeddingProvider
from .rubrics.static import StaticRubric
from .training.static_reward import static_rubric_score


def _distribution(prompt_id: str, step: int) -> tuple[float, ...]:
    digest = hashlib.sha256(f"{prompt_id}\x1f{step}\x1fpolicy-simulator-v1".encode()).digest()
    values = tuple(1.0 + byte / 255.0 for byte in digest[:16])
    total = sum(values)
    return tuple(value / total for value in values)


def simulated_kl_from_pi0(prompt_id: str, step: int) -> float:
    """KL for deterministic fake token distributions, never a fitted constant."""

    current = _distribution(prompt_id, step)
    baseline = _distribution(prompt_id, 0)
    return sum(value * math.log(value / reference) for value, reference in zip(current, baseline))


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def ground_probe_record(
    record: Mapping[str, Any],
    rubric: StaticRubric,
    reference_text: str,
) -> dict[str, Any]:
    """Attach reward and distance values computed through the fake adapters."""

    value = dict(record)
    grader = FakeCriterionGrader()
    criterion_scores = [
        grader.score(
            rubric.prompt_id,
            str(value["response_id"]),
            str(value["response_text"]),
            criterion.criterion_id,
            criterion.text,
        )
        for criterion in rubric.criteria
    ]
    if not all(score.parse_success for score in criterion_scores):
        raise RuntimeError("fake static-rubric grader parse failure")
    reward = static_rubric_score([score.probability_yes for score in criterion_scores])
    embedder = FakeEmbeddingProvider()
    current_vector, reference_vector = embedder.embed([str(value["response_text"]), reference_text])
    value.update(
        {
            "kl_from_pi0": simulated_kl_from_pi0(rubric.prompt_id, int(value["policy_step"])),
            "static_proxy_reward": reward,
            "static_proxy_criterion_scores": [
                {
                    "criterion_id": score.criterion_id,
                    "probability_yes": score.probability_yes,
                    "parse_success": score.parse_success,
                }
                for score in criterion_scores
            ],
            "response_embedding_distance": 1.0 - _cosine(current_vector, reference_vector),
            "metric_provenance": {
                "policy_distance": "fake_distribution_kl_v1",
                "reward": "static_r0_fake_criterion_grader_v1",
                "embedding": dict(embedder.identity),
            },
        }
    )
    return value
