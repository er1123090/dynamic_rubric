from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from typing import Mapping, Sequence

from .base import CriterionScore, GenerationRequest, GenerationResult


def _fraction(*parts: object) -> float:
    digest = hashlib.sha256(": ".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


class FakeGenerator:
    """Deterministic provider used by credential-free integration tests."""

    def __init__(self, model: str = "fake/model-v1", drift_after: int | None = None) -> None:
        self.model = model
        self.drift_after = drift_after
        self.calls = 0

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        returned = (
            self.model
            if self.drift_after is None or self.calls <= self.drift_after
            else f"{self.model}-drift"
        )
        if request.json_schema:
            criteria_schema = request.json_schema.get("properties", {}).get("criteria")
            if not isinstance(criteria_schema, Mapping):
                raise ValueError("fake generator supports only schemas with a criteria array")
            minimum = int(criteria_schema.get("minItems", 0))
            maximum = int(criteria_schema.get("maxItems", minimum))
            if minimum < 0 or maximum < minimum:
                raise ValueError("invalid criteria array bounds in JSON schema")
            criteria = [
                {
                    "text": f"The response demonstrates synthetic behavior {index + 1} for {request.prompt_id}.",
                    "rationale": "synthetic fixture",
                }
                for index in range(maximum)
            ]
            text = json.dumps({"criteria": criteria}, sort_keys=True)
        else:
            # Keep the provider output source-blind: replay receives only this
            # text, so family labels must not reveal current/control origin.
            digest = hashlib.sha256(f"{request.prompt_id}\x1f{request.seed}".encode()).hexdigest()
            text = f"synthetic response {request.prompt_id} {digest[:16]}"
        return GenerationResult(
            text=text,
            requested_model=self.model,
            returned_model=returned,
            request_id=f"fake-{self.calls:08d}",
            created_at=self.calls,
            retry_count=0,
            usage={"input_tokens": 1, "output_tokens": 1},
            raw_response_hash=hashlib.sha256(text.encode()).hexdigest(),
        )

    def generate_many(self, request: GenerationRequest, count: int) -> Sequence[GenerationResult]:
        return [
            self.generate(replace(request, seed=request.seed + index)) for index in range(count)
        ]


class FakeCriterionGrader:
    model = "fake/qwen3-32b-v1"

    def score(
        self,
        prompt_id: str,
        response_id: str,
        response_text: str,
        criterion_id: str,
        criterion_text: str,
    ) -> CriterionScore:
        probability = 0.05 + 0.9 * _fraction(
            prompt_id, response_id, response_text, criterion_id, criterion_text
        )
        return CriterionScore(prompt_id, response_id, criterion_id, probability, True)


class FakeEmbeddingProvider:
    identity: Mapping[str, str] = {"model": "fake-embedding", "revision": "v1"}

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            vector = [(byte - 127.5) / 127.5 for byte in digest[:16]]
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors
