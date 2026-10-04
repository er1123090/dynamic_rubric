"""Hybrid horizon grader that resumes complete criterion caches, then grades whole rubrics."""

from __future__ import annotations

import hashlib
import json
import math
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from ..artifacts import ImmutableArtifactError, read_json, write_json_atomic
from ..prompt_versions.onlinerubric_grader_prompt import (
    build_onlinerubric_grader_messages,
    onlinerubric_grader_schema,
)
from ..training.online_contracts import WeightedCriterion
from ..training.paper_reward import parse_binary_grades
from .vllm import FullCriterionScore, normalized_yes_probability


YES_TARGET = " YES"
NO_TARGET = " NO"
TARGETS = [YES_TARGET, NO_TARGET]
CRITERION_CACHE_SCORING_MODE = "full_vocab_yes_no_prompt_logprob_v4"
HYBRID_TARGET_ENCODING_VERSION = (
    "hybrid_cached_yes_no_then_onlinerubric_full_rubric_json_v1"
)
HOLISTIC_EXECUTION_MODE = "onlinerubric_full_rubric_one_call_per_response_v1"

GradeItem = tuple[str, str, str, str, str]


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


class HybridVLLMFullRubricGrader:
    """Reuse only fully cached responses; grade every incomplete response in one call.

    The legacy score-proxy cache is criterion-addressed. A response is reused only
    when every criterion in its current rubric is present and validates against the
    exact legacy cache envelope. If even one criterion is absent, all legacy entries
    for that response are ignored and the complete rubric is sent in one structured
    OnlineRubric-style chat-completions request.
    """

    def __init__(
        self,
        *,
        base_urls: Sequence[str],
        served_model: str,
        model_revision: str,
        tokenizer_revision: str,
        criterion_cache_dir: Path,
        holistic_cache_dir: Path,
        max_workers: int = 16,
        timeout_seconds: float = 900.0,
        max_attempts: int = 3,
    ) -> None:
        normalized = tuple(url.rstrip("/") for url in base_urls if url.rstrip("/"))
        if not normalized:
            raise ValueError("at least one vLLM base URL is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("vLLM base URLs must be unique")
        if max_workers < 1:
            raise ValueError("max_workers must be positive")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        self.base_urls = normalized
        self.served_model = served_model
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.criterion_cache_dir = criterion_cache_dir
        self.holistic_cache_dir = holistic_cache_dir
        self.max_workers = max_workers
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self._stats_lock = Lock()
        self._stats: dict[str, Any] = {
            "response_groups_seen": 0,
            "reused_complete_responses": 0,
            "reused_criterion_items": 0,
            "holistic_responses": 0,
            "holistic_api_calls": 0,
            "holistic_cache_hits": 0,
            "ignored_partial_criterion_items": 0,
            "endpoint_api_calls": {url: 0 for url in normalized},
        }

    @property
    def criterion_cache_identity(self) -> dict[str, Any]:
        return {
            "served_model": self.served_model,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "thinking": False,
            "scoring_mode": CRITERION_CACHE_SCORING_MODE,
        }

    @property
    def holistic_identity(self) -> dict[str, Any]:
        return {
            "served_model": self.served_model,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "thinking": False,
            "temperature": 0,
            "prompt_version": "onlinerubric-figure10-grader-v1",
            "execution_mode": HOLISTIC_EXECUTION_MODE,
        }

    @property
    def artifact_metadata(self) -> dict[str, Any]:
        with self._stats_lock:
            stats = {
                **self._stats,
                "endpoint_api_calls": dict(self._stats["endpoint_api_calls"]),
            }
        return {
            "schema_version": 1,
            "execution_mode": HOLISTIC_EXECUTION_MODE,
            "resume_policy": "reuse_legacy_cache_only_for_complete_response_rubrics",
            "vllm_structured_output_transport": "exact_key_regex_no_whitespace_v1",
            "base_urls": list(self.base_urls),
            "criterion_cache_dir": str(self.criterion_cache_dir),
            "holistic_cache_dir": str(self.holistic_cache_dir),
            "stats": stats,
        }

    def _increment(self, field: str, count: int = 1) -> None:
        with self._stats_lock:
            self._stats[field] += count

    def _record_endpoint_call(self, base_url: str) -> None:
        with self._stats_lock:
            self._stats["holistic_api_calls"] += 1
            self._stats["endpoint_api_calls"][base_url] += 1

    def preflight(self) -> dict[str, Any]:
        checked: list[str] = []
        for base_url in self.base_urls:
            request = urllib.request.Request(f"{base_url}/v1/models", method="GET")
            with urllib.request.urlopen(request, timeout=30) as response:
                value = json.loads(response.read())
            models = value.get("data", [])
            model_ids = {
                model_id
                for item in models
                if isinstance(item, Mapping)
                and isinstance(model_id := item.get("id"), str)
            }
            if self.served_model not in model_ids:
                raise ValueError(
                    "vLLM model drift: "
                    f"base_url={base_url}, expected={self.served_model}, "
                    f"actual={sorted(model_ids)}"
                )
            checked.append(base_url)
        return {
            **self.holistic_identity,
            "validated_base_urls": checked,
        }

    @staticmethod
    def _render_criterion_prompt(
        prompt_text: str,
        response_text: str,
        criterion_text: str,
    ) -> str:
        return (
            "Prompt: "
            + prompt_text
            + f"\nResponse: {response_text}\nCriterion: {criterion_text}"
            + "\nYES = criterion PRESENT; NO = criterion NOT_PRESENT.\nAnswer:"
        )

    def _criterion_cache_key(self, rendered_prompt: str) -> str:
        return hashlib.sha256(
            _canonical_bytes(
                {
                    "identity": self.criterion_cache_identity,
                    "rendered_prompt": rendered_prompt,
                    "targets": TARGETS,
                }
            )
        ).hexdigest()

    def _read_criterion_cache(
        self,
        *,
        prompt_text: str,
        response_text: str,
        criterion_text: str,
    ) -> tuple[float, float] | None:
        rendered = self._render_criterion_prompt(
            prompt_text,
            response_text,
            criterion_text,
        )
        key = self._criterion_cache_key(rendered)
        path = self.criterion_cache_dir / key[:2] / f"{key}.json"
        try:
            envelope = read_json(path)
        except FileNotFoundError:
            return None
        scores = envelope.get("target_logprobs")
        if not isinstance(scores, Mapping) or set(scores) != set(TARGETS):
            raise ValueError(f"legacy criterion cache has malformed scores: {path}")
        try:
            normalized = {target: float(scores[target]) for target in TARGETS}
        except (TypeError, ValueError) as error:
            raise ValueError(f"legacy criterion cache has non-numeric scores: {path}") from error
        if not all(math.isfinite(value) for value in normalized.values()):
            raise ValueError(f"legacy criterion cache has non-finite scores: {path}")
        expected = {
            "schema_version": 1,
            "key": key,
            "identity": self.criterion_cache_identity,
            "rendered_prompt_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
            "targets": TARGETS,
            "target_logprobs": normalized,
        }
        if envelope != expected:
            raise ValueError(f"legacy criterion cache identity mismatch: {path}")
        return normalized[YES_TARGET], normalized[NO_TARGET]

    def _holistic_request(
        self,
        *,
        prompt: Sequence[Mapping[str, str]],
        response_text: str,
        items: Sequence[GradeItem],
    ) -> dict[str, Any]:
        criteria = [
            {"criterion_id": criterion_id, "criterion": criterion_text}
            for _, _, _, criterion_id, criterion_text in items
        ]
        messages = build_onlinerubric_grader_messages(
            prompt=prompt,
            response=response_text,
            criteria=criteria,
        )
        schema = onlinerubric_grader_schema(len(criteria))
        return {
            "model": self.served_model,
            "messages": list(messages),
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "onlinerubric_grader_v1",
                    "schema": schema,
                    "strict": True,
                },
            },
            "temperature": 0,
            "seed": 0,
            "max_tokens": 512,
            "chat_template_kwargs": {"enable_thinking": False},
        }

    @staticmethod
    def _transport_request(request_payload: Mapping[str, Any]) -> dict[str, Any]:
        """Replace whitespace-flexible JSON schema with an equivalent exact regex.

        The logical request remains unchanged for cache compatibility. vLLM's
        JSON-schema grammar permits arbitrary whitespace, which can make greedy
        decoding emit spaces until ``max_tokens`` before a required final key.
        The transport regex preserves the exact key/label inventory while making
        that whitespace loop impossible.
        """

        response_format = request_payload.get("response_format")
        if not isinstance(response_format, Mapping):
            raise ValueError("logical holistic request has no response_format")
        json_schema = response_format.get("json_schema")
        if not isinstance(json_schema, Mapping):
            raise ValueError("logical holistic request has no JSON schema")
        schema = json_schema.get("schema")
        if not isinstance(schema, Mapping):
            raise ValueError("logical holistic request schema is malformed")
        required = schema.get("required")
        properties = schema.get("properties")
        if (
            not isinstance(required, list)
            or not all(isinstance(key, str) for key in required)
            or not isinstance(properties, Mapping)
            or set(required) != set(properties)
        ):
            raise ValueError("logical holistic request rubric inventory is malformed")
        keys = sorted(required)
        fields = [f'"{key}":"(PRESENT|NOT_PRESENT)"' for key in keys]
        transport = dict(request_payload)
        transport.pop("response_format")
        transport["structured_outputs"] = {
            "regex": r"\{" + ",".join(fields) + r"\}"
        }
        return transport

    def _holistic_cache_key(self, request_payload: Mapping[str, Any]) -> str:
        return hashlib.sha256(
            _canonical_bytes(
                {
                    "identity": self.holistic_identity,
                    "request": request_payload,
                }
            )
        ).hexdigest()

    def _holistic_cache_path(self, key: str) -> Path:
        return self.holistic_cache_dir / key[:2] / f"{key}.json"

    def _read_holistic_cache(
        self,
        *,
        key: str,
        request_payload: Mapping[str, Any],
        criterion_ids: Sequence[str],
    ) -> tuple[dict[str, str], int] | None:
        path = self._holistic_cache_path(key)
        try:
            envelope = read_json(path)
        except FileNotFoundError:
            return None
        labels = envelope.get("labels")
        if not isinstance(labels, Mapping):
            raise ValueError(f"holistic cache labels are malformed: {path}")
        normalized = {str(item): str(value) for item, value in labels.items()}
        if set(normalized) != set(criterion_ids) or any(
            value not in {"PRESENT", "NOT_PRESENT"} for value in normalized.values()
        ):
            raise ValueError(f"holistic cache rubric inventory mismatch: {path}")
        retry_count = int(envelope.get("retry_count", 0))
        expected = {
            "schema_version": 1,
            "key": key,
            "identity": self.holistic_identity,
            "request_sha256": hashlib.sha256(_canonical_bytes(request_payload)).hexdigest(),
            "labels": normalized,
            "retry_count": retry_count,
        }
        if envelope != expected:
            raise ValueError(f"holistic cache identity mismatch: {path}")
        return normalized, retry_count

    def _publish_holistic_cache(
        self,
        *,
        key: str,
        request_payload: Mapping[str, Any],
        labels: Mapping[str, str],
        retry_count: int,
    ) -> tuple[dict[str, str], int]:
        envelope = {
            "schema_version": 1,
            "key": key,
            "identity": self.holistic_identity,
            "request_sha256": hashlib.sha256(_canonical_bytes(request_payload)).hexdigest(),
            "labels": dict(labels),
            "retry_count": retry_count,
        }
        try:
            write_json_atomic(self._holistic_cache_path(key), envelope)
            return dict(labels), retry_count
        except ImmutableArtifactError:
            # Separate checkpoint graders can legitimately issue the same
            # request at the same time. vLLM can return a different greedy
            # label across endpoints, so the first immutable cache writer is
            # canonical and every losing writer must use that same result.
            cached = self._read_holistic_cache(
                key=key,
                request_payload=request_payload,
                criterion_ids=tuple(labels),
            )
            if cached is None:
                raise
            return cached

    def _request_holistic(
        self,
        *,
        key: str,
        request_payload: Mapping[str, Any],
        criteria: Sequence[WeightedCriterion],
    ) -> tuple[dict[str, str], int]:
        transport_payload = self._transport_request(request_payload)
        start_index = int(key[:16], 16) % len(self.base_urls)
        errors: list[str] = []
        for retry_count in range(self.max_attempts):
            base_url = self.base_urls[(start_index + retry_count) % len(self.base_urls)]
            request = urllib.request.Request(
                f"{base_url}/v1/chat/completions",
                data=_canonical_bytes(transport_payload),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            self._record_endpoint_call(base_url)
            try:
                with urllib.request.urlopen(
                    request,
                    timeout=self.timeout_seconds,
                ) as response:
                    value = json.loads(response.read())
                choices = value.get("choices")
                if not isinstance(choices, list) or len(choices) != 1:
                    raise ValueError("vLLM returned the wrong number of choices")
                message = choices[0].get("message")
                if not isinstance(message, Mapping) or not isinstance(
                    message.get("content"), str
                ):
                    raise ValueError("vLLM returned no structured message content")
                parsed = parse_binary_grades(str(message["content"]), criteria)
                return {
                    criterion_id: "PRESENT" if grade else "NOT_PRESENT"
                    for criterion_id, grade in parsed
                }, retry_count
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
                errors.append(f"{base_url}: {type(error).__name__}: {error}")
        raise RuntimeError(
            "full-rubric grading failed after endpoint rotation: " + " | ".join(errors)
        )

    @staticmethod
    def _score_from_pair(item: GradeItem, yes: float, no: float, retry_count: int = 0) -> FullCriterionScore:
        prompt_id, response_id, _, criterion_id, _ = item
        return FullCriterionScore(
            prompt_id=prompt_id,
            response_id=response_id,
            criterion_id=criterion_id,
            yes_logprob=yes,
            no_logprob=no,
            probability_present=normalized_yes_probability(yes, no),
            parse_status="ambiguous_target_tie" if yes == no else "ok",
            retry_count=retry_count,
        )

    def _grade_response_group(
        self,
        items: Sequence[GradeItem],
        *,
        prompt_text: str,
    ) -> tuple[FullCriterionScore, ...]:
        if not items:
            return ()
        prompt_ids = {item[0] for item in items}
        response_ids = {item[1] for item in items}
        response_texts = {item[2] for item in items}
        criterion_ids = [item[3] for item in items]
        if len(prompt_ids) != 1 or len(response_ids) != 1 or len(response_texts) != 1:
            raise ValueError("one holistic request must bind one prompt and one response")
        if len(criterion_ids) != len(set(criterion_ids)):
            raise ValueError("one response rubric contains duplicate criterion identities")

        cached_pairs = [
            self._read_criterion_cache(
                prompt_text=prompt_text,
                response_text=item[2],
                criterion_text=item[4],
            )
            for item in items
        ]
        present_count = sum(pair is not None for pair in cached_pairs)
        if present_count == len(items):
            self._increment("reused_complete_responses")
            self._increment("reused_criterion_items", len(items))
            return tuple(
                self._score_from_pair(item, pair[0], pair[1])
                for item, pair in zip(items, cached_pairs)
                if pair is not None
            )
        if present_count:
            self._increment("ignored_partial_criterion_items", present_count)

        try:
            raw_prompt = json.loads(prompt_text)
        except json.JSONDecodeError as error:
            raise ValueError("prompt_text_by_id must contain JSON messages") from error
        if not isinstance(raw_prompt, list) or not all(
            isinstance(message, Mapping) for message in raw_prompt
        ):
            raise ValueError("prompt_text_by_id must contain a JSON message list")
        prompt = [
            {
                "role": str(message.get("role", "")),
                "content": str(message.get("content", "")),
            }
            for message in raw_prompt
        ]
        request_payload = self._holistic_request(
            prompt=prompt,
            response_text=items[0][2],
            items=items,
        )
        key = self._holistic_cache_key(request_payload)
        cached = self._read_holistic_cache(
            key=key,
            request_payload=request_payload,
            criterion_ids=criterion_ids,
        )
        if cached is None:
            criteria = tuple(
                WeightedCriterion(criterion_id, criterion_text, 1, "horizon_eval")
                for _, _, _, criterion_id, criterion_text in items
            )
            labels, retry_count = self._request_holistic(
                key=key,
                request_payload=request_payload,
                criteria=criteria,
            )
            labels, retry_count = self._publish_holistic_cache(
                key=key,
                request_payload=request_payload,
                labels=labels,
                retry_count=retry_count,
            )
        else:
            labels, retry_count = cached
            self._increment("holistic_cache_hits")
        self._increment("holistic_responses")
        return tuple(
            self._score_from_pair(
                item,
                0.0 if labels[item[3]] == "PRESENT" else -1.0,
                -1.0 if labels[item[3]] == "PRESENT" else 0.0,
                retry_count,
            )
            for item in items
        )

    def score_many_full(
        self,
        items: Sequence[GradeItem],
        *,
        prompt_text_by_id: Mapping[str, str] | None = None,
    ) -> tuple[FullCriterionScore, ...]:
        if not items:
            return ()
        prompt_texts = prompt_text_by_id or {}
        groups: dict[tuple[str, str], list[GradeItem]] = {}
        for item in items:
            groups.setdefault((item[0], item[1]), []).append(item)
        self._increment("response_groups_seen", len(groups))
        ordered_groups = list(groups.items())
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(ordered_groups))
        ) as executor:
            futures = [
                executor.submit(
                    self._grade_response_group,
                    group,
                    prompt_text=prompt_texts.get(prompt_id, ""),
                )
                for (prompt_id, _), group in ordered_groups
            ]
            results = [future.result() for future in futures]
        return tuple(score for group in results for score in group)
