from __future__ import annotations

import json
import math
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .base import CriterionScore

YES_TARGET = " YES"
NO_TARGET = " NO"

class VLLMPreflightError(RuntimeError):
    pass


def normalized_yes_probability(yes_logprob: float, no_logprob: float) -> float:
    if not (math.isfinite(yes_logprob) and math.isfinite(no_logprob)):
        raise ValueError("YES/NO log probabilities must be finite")
    maximum = max(yes_logprob, no_logprob)
    yes = math.exp(yes_logprob - maximum)
    no = math.exp(no_logprob - maximum)
    return yes / (yes + no)


@dataclass(frozen=True)
class VLLMIdentity:
    served_model: str
    model_revision: str
    tokenizer_revision: str
    thinking: bool


@dataclass(frozen=True)
class FullCriterionScore:
    prompt_id: str
    response_id: str
    criterion_id: str
    yes_logprob: float
    no_logprob: float
    probability_present: float
    parse_status: str
    retry_count: int = 0


class VLLMCriterionGrader:
    """Adapter for a version-pinned vLLM scoring sidecar.

    The sidecar contract intentionally requires explicit sequence log-likelihoods
    for both target continuations. It never falls back to sampled Boolean text.
    """

    def __init__(
        self,
        base_url: str,
        expected: VLLMIdentity,
        timeout_seconds: float = 900.0,
        max_items_per_request: int = 32,
        max_retries: int = 6,
        retry_initial_seconds: float = 1.0,
        retry_max_seconds: float = 15.0,
    ):
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_items_per_request <= 0:
            raise ValueError("max_items_per_request must be positive")
        if max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if retry_initial_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("retry delays must be non-negative")
        self.base_url = base_url.rstrip("/")
        self.expected = expected
        self.timeout_seconds = timeout_seconds
        self.max_items_per_request = max_items_per_request
        self.max_retries = max_retries
        self.retry_initial_seconds = retry_initial_seconds
        self.retry_max_seconds = retry_max_seconds

    def _post(self, endpoint: str, payload: Mapping[str, Any]) -> Mapping[str, Any]:
        value: Any = None
        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}{endpoint}",
                data=json.dumps(payload).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    value = json.loads(response.read())
                break
            except Exception as error:
                if not self._is_transient_transport_error(error) or attempt >= self.max_retries:
                    raise VLLMPreflightError(
                        f"vLLM sidecar request failed after {attempt + 1} attempt(s): "
                        f"{type(error).__name__}: {error}"
                    ) from error
                delay = min(
                    self.retry_initial_seconds * (2**attempt),
                    self.retry_max_seconds,
                )
                time.sleep(delay)
        if not isinstance(value, Mapping):
            raise VLLMPreflightError("vLLM sidecar returned a non-object")
        return value

    @staticmethod
    def _is_transient_transport_error(error: Exception) -> bool:
        if isinstance(error, urllib.error.HTTPError):
            return error.code in {408, 429} or error.code >= 500
        return isinstance(
            error,
            (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
        )

    def preflight(self) -> Mapping[str, Any]:
        with urllib.request.urlopen(f"{self.base_url}/dynamic-rubric/identity") as response:
            identity = json.loads(response.read())
        expected = self.expected.__dict__
        if any(identity.get(key) != value for key, value in expected.items()):
            raise VLLMPreflightError(
                f"served identity mismatch: expected={expected}, got={identity}"
            )
        probe = self._post(
            "/dynamic-rubric/score-targets",
            {
                "rendered_prompts": [
                    f"Criterion: synthetic-{index}\nResponse: synthetic\nAnswer:"
                    for index in range(10)
                ],
                "targets": [YES_TARGET, NO_TARGET],
            },
        )
        pairs = self._extract_pairs(probe, 10)
        if len(pairs) != 10:
            raise VLLMPreflightError("target scorer did not return all ten probe pairs")
        return {**identity, "criterion_pair_probe_count": len(pairs), "parse_success": 1.0}

    @staticmethod
    def _extract_pair(value: Mapping[str, Any]) -> tuple[float, float]:
        scores = value.get("target_logprobs")
        if not isinstance(scores, Mapping) or set(scores) != {YES_TARGET, NO_TARGET}:
            raise VLLMPreflightError(
                'sidecar must return exactly " YES" and " NO" target_logprobs'
            )
        try:
            return float(scores[YES_TARGET]), float(scores[NO_TARGET])
        except (TypeError, ValueError) as error:
            raise VLLMPreflightError("target logprobs were malformed") from error

    @classmethod
    def _extract_pairs(
        cls, value: Mapping[str, Any], expected: int
    ) -> tuple[tuple[float, float], ...]:
        scores = value.get("target_logprobs")
        if expected == 1 and isinstance(scores, Mapping):
            return (cls._extract_pair(value),)
        if not isinstance(scores, list) or len(scores) != expected:
            raise VLLMPreflightError(
                f"sidecar returned {type(scores).__name__} instead of {expected} score pairs"
            )
        return tuple(cls._extract_pair({"target_logprobs": item}) for item in scores)

    def score_many_full(
        self,
        items: Sequence[tuple[str, str, str, str, str]],
        *,
        prompt_text_by_id: Mapping[str, str] | None = None,
    ) -> tuple[FullCriterionScore, ...]:
        if not items:
            return ()
        scores: list[FullCriterionScore] = []
        for start in range(0, len(items), self.max_items_per_request):
            chunk = items[start : start + self.max_items_per_request]
            result = self._post(
                "/dynamic-rubric/score-targets",
                {
                    "rendered_prompts": [
                        "Prompt: "
                        + str((prompt_text_by_id or {}).get(prompt_id, ""))
                        + f"\nResponse: {response_text}\nCriterion: {criterion_text}"
                        + "\nYES = criterion PRESENT; NO = criterion NOT_PRESENT.\nAnswer:"
                        for prompt_id, _, response_text, _, criterion_text in chunk
                    ],
                    "targets": [YES_TARGET, NO_TARGET],
                    "temperature": 0,
                    "thinking": False,
                },
            )
            pairs = self._extract_pairs(result, len(chunk))
            scores.extend(
                FullCriterionScore(
                    prompt_id,
                    response_id,
                    criterion_id,
                    yes,
                    no,
                    normalized_yes_probability(yes, no),
                    "ambiguous_target_tie" if yes == no else "ok",
                )
                for (
                    prompt_id,
                    response_id,
                    _,
                    criterion_id,
                    _,
                ), (yes, no) in zip(chunk, pairs)
            )
        return tuple(scores)

    def score_many(
        self,
        items: Sequence[tuple[str, str, str, str, str]],
    ) -> tuple[CriterionScore, ...]:
        return tuple(
            CriterionScore(
                item.prompt_id,
                item.response_id,
                item.criterion_id,
                item.probability_present,
                item.parse_status == "ok",
            )
            for item in self.score_many_full(items)
        )

    def score(
        self,
        prompt_id: str,
        response_id: str,
        response_text: str,
        criterion_id: str,
        criterion_text: str,
    ) -> CriterionScore:
        return self.score_many(
            ((prompt_id, response_id, response_text, criterion_id, criterion_text),)
        )[0]
