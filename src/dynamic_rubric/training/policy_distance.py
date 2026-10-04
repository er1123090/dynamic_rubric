"""Ground post-update probe movement against immutable pi0 references."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from dynamic_rubric.providers.base import EmbeddingProvider


REFERENCE_COUNTS = {"reference_discovery": 8, "reference_validation": 4}
TRAJECTORY_TO_REFERENCE = {
    "trajectory_discovery": "reference_discovery",
    "trajectory_validation": "reference_validation",
}
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)


def _unit_centroid(vectors: Sequence[Sequence[float]]) -> tuple[float, ...]:
    if not vectors or any(len(vector) != len(vectors[0]) for vector in vectors):
        raise ValueError("reference embedding inventory is empty or ragged")
    centroid = tuple(sum(column) / len(vectors) for column in zip(*vectors))
    norm = math.sqrt(sum(value * value for value in centroid))
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("reference embedding centroid is invalid")
    return tuple(value / norm for value in centroid)


@dataclass(frozen=True)
class PolicyDistanceEnricher:
    embedding: EmbeddingProvider
    centroids: Mapping[tuple[str, str], tuple[float, ...]]

    @classmethod
    def from_references(
        cls,
        embedding: EmbeddingProvider,
        references: Sequence[Mapping[str, Any]],
    ) -> "PolicyDistanceEnricher":
        grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
        seen_ids: set[str] = set()
        for row in references:
            prompt_id = str(row["prompt_id"])
            family = str(row["family"])
            response_id = str(row["response_id"])
            if family not in REFERENCE_COUNTS:
                raise ValueError(f"invalid pi0 reference family: {family}")
            if response_id in seen_ids:
                raise ValueError(f"duplicate pi0 reference response_id: {response_id}")
            seen_ids.add(response_id)
            grouped[(prompt_id, family)].append(str(row["response_text"]))
        if not grouped:
            raise ValueError("pi0 reference responses are absent")
        if any(len(texts) != REFERENCE_COUNTS[key[1]] for key, texts in grouped.items()):
            raise ValueError("pi0 reference response inventory is incomplete")
        centroids: dict[tuple[str, str], tuple[float, ...]] = {}
        for key, texts in sorted(grouped.items()):
            centroids[key] = _unit_centroid(embedding.embed(texts))
        return cls(embedding=embedding, centroids=centroids)

    def enrich(self, rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        vectors = self.embedding.embed([str(row["response_text"]) for row in rows])
        if len(vectors) != len(rows):
            raise ValueError("probe embedding count mismatch")
        enriched: list[dict[str, Any]] = []
        for raw, vector in zip(rows, vectors):
            row = dict(raw)
            reference_family = TRAJECTORY_TO_REFERENCE.get(str(row.get("family")))
            key = (str(row.get("prompt_id")), str(reference_family))
            centroid = self.centroids.get(key)
            if centroid is None or len(centroid) != len(vector):
                raise ValueError(f"pi0 reference centroid is absent for {key}")
            kl = float(row["kl_from_pi0"])
            if not math.isfinite(kl):
                raise ValueError("probe KL-from-pi0 is non-finite")
            cosine = sum(left * float(right) for left, right in zip(centroid, vector))
            if not math.isfinite(cosine):
                raise ValueError("probe response cosine is non-finite")
            text = str(row["response_text"])
            tokens = _TOKEN.findall(text.casefold())
            row.update(
                {
                    "response_embedding_distance": 1.0 - max(-1.0, min(1.0, cosine)),
                    "response_length_chars": len(text),
                    "response_length_words": len(tokens),
                    "style_markdown_heading": any(
                        line.lstrip().startswith("#") for line in text.splitlines()
                    ),
                    "style_bullet_list": any(
                        line.lstrip().startswith(("- ", "* ", "• "))
                        for line in text.splitlines()
                    ),
                }
            )
            enriched.append(row)
        return enriched


def summarize_policy_distance(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["split"]), int(row["policy_step"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (split, step), values in sorted(grouped.items()):
        tokens = [
            token
            for row in values
            for token in _TOKEN.findall(str(row["response_text"]).casefold())
        ]
        count = len(values)
        summaries.append(
            {
                "split": split,
                "policy_step": step,
                "records": count,
                "mean_kl_from_pi0": sum(float(row["kl_from_pi0"]) for row in values)
                / count,
                "mean_static_proxy_reward": sum(
                    float(row["static_proxy_reward"]) for row in values
                )
                / count,
                "mean_response_embedding_distance": sum(
                    float(row["response_embedding_distance"]) for row in values
                )
                / count,
                "mean_response_length_chars": sum(
                    int(row["response_length_chars"]) for row in values
                )
                / count,
                "mean_response_length_words": sum(
                    int(row["response_length_words"]) for row in values
                )
                / count,
                "vocabulary_type_token_ratio": len(set(tokens)) / len(tokens) if tokens else 0.0,
                "markdown_heading_rate": sum(
                    bool(row["style_markdown_heading"]) for row in values
                )
                / count,
                "bullet_list_rate": sum(bool(row["style_bullet_list"]) for row in values)
                / count,
            }
        )
    return summaries
