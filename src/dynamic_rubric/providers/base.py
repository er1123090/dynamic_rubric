from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence


@dataclass(frozen=True)
class GenerationRequest:
    prompt_id: str
    messages: tuple[Mapping[str, str], ...]
    family: str
    seed: int
    temperature: float = 0.0
    top_p: float = 1.0
    max_output_tokens: int = 1024
    json_schema: Mapping[str, Any] | None = None
    schema_name: str | None = None
    reasoning_effort: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    text: str
    requested_model: str
    returned_model: str
    request_id: str
    created_at: int | str | None
    retry_count: int
    usage: Mapping[str, Any] = field(default_factory=dict)
    raw_response_hash: str | None = None


class RubricGenerator(Protocol):
    def generate(self, request: GenerationRequest) -> GenerationResult: ...


class PolicyGenerator(Protocol):
    def generate_many(
        self, request: GenerationRequest, count: int
    ) -> Sequence[GenerationResult]: ...


@dataclass(frozen=True)
class CriterionScore:
    prompt_id: str
    response_id: str
    criterion_id: str
    probability_yes: float
    parse_success: bool = True


class CriterionGrader(Protocol):
    def score(
        self,
        prompt_id: str,
        response_id: str,
        response_text: str,
        criterion_id: str,
        criterion_text: str,
    ) -> CriterionScore: ...


class EmbeddingProvider(Protocol):
    @property
    def identity(self) -> Mapping[str, str]: ...

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]: ...
