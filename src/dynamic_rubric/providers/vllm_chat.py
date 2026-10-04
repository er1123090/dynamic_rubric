from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_json, write_json_atomic
from .base import GenerationRequest, GenerationResult


class VLLMChatError(RuntimeError):
    pass


class _RetryableResponseError(VLLMChatError):
    """A successful HTTP response whose completion body is not yet usable."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


class VLLMChatAdapter:
    """OpenAI-compatible vLLM Chat Completions adapter with immutable caching."""

    def __init__(
        self,
        base_url: str | Sequence[str],
        model: str,
        cache_dir: Path,
        *,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
        max_in_flight: int | None = None,
        bounded_grading_whitespace: bool = False,
    ) -> None:
        raw_urls = (base_url,) if isinstance(base_url, str) else tuple(base_url)
        if not raw_urls or any(not str(url).strip() for url in raw_urls):
            raise ValueError("at least one non-empty vLLM base URL is required")
        normalized_urls = tuple(
            root if root.endswith("/v1") else f"{root}/v1"
            for root in (str(url).strip().rstrip("/") for url in raw_urls)
        )
        if len(set(normalized_urls)) != len(normalized_urls):
            raise ValueError("vLLM base URLs must be unique")
        if not model.strip():
            raise ValueError("a served model name is required")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ValueError("max_retries cannot be negative")
        if max_in_flight is not None and max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        self._in_flight = threading.BoundedSemaphore(max_in_flight) if max_in_flight is not None else None
        self.bounded_grading_whitespace = bounded_grading_whitespace
        self.base_urls = normalized_urls
        # Backward-compatible inspection surface for one-endpoint callers.
        self.base_url = normalized_urls[0]
        self.model = model
        self.cache_dir = cache_dir
        self.api_key = api_key or None
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._returned_model: str | None = None
        self._identity_lock = threading.Lock()
        self.calls = 0

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in request.messages],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
            "top_p": request.top_p,
            "seed": request.seed,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if request.reasoning_effort:
            payload["reasoning_effort"] = request.reasoning_effort
        if request.json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": request.schema_name or "structured_response",
                    "schema": dict(request.json_schema),
                    "strict": True,
                },
            }
        if self.bounded_grading_whitespace:
            if request.family != 'phase1_audit_grading' or request.schema_name != 'onlinerubric_grader_v1':
                raise ValueError('bounded whitespace is scoped to the binary audit grader')
            payload.pop('response_format', None)
            payload['structured_outputs'] = {'grammar': self._bounded_grading_grammar(request.json_schema)}
        return payload

    @staticmethod
    def _bounded_grading_grammar(schema: Mapping[str, Any] | None) -> str:
        from ..prompt_versions.onlinerubric_grader_prompt import onlinerubric_grader_schema

        if not isinstance(schema, Mapping) or not isinstance(schema.get('properties'), Mapping):
            raise ValueError('bounded whitespace requires the exact binary grading schema')
        count = len(schema['properties'])
        if count < 1 or schema != onlinerubric_grader_schema(count):
            raise ValueError('bounded whitespace requires the exact binary grading schema')
        # Match the lexicographic property order in the canonical JSON HTTP body.
        # Constrain serialization whitespace only; do not infer any missing grade.
        fields = [json.dumps(json.dumps(key)) + ' ws ":" ws grade' for key in sorted(schema['properties'])]
        return ('root ::= "{" ws ' + ' ws "," ws '.join(fields) + ' ws "}"\n'
                + 'ws ::= [ \\t\\r\\n]?\n'
                + 'grade ::= "\\"PRESENT\\"" | "\\"NOT_PRESENT\\""\n')

    def _routing_identity(
        self, request: GenerationRequest, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "requested_model": self.model,
            "prompt_id": request.prompt_id,
            "family": request.family,
            "logical_seed": request.seed,
            "metadata": dict(request.metadata),
            "payload": dict(payload),
        }

    def _select_base_url(self, request: GenerationRequest, payload: Mapping[str, Any]) -> str:
        digest = hashlib.sha256(_canonical(self._routing_identity(request, payload))).digest()
        index = int.from_bytes(digest[:8], "big") % len(self.base_urls)
        return self.base_urls[index]

    def _identity(
        self,
        request: GenerationRequest,
        payload: Mapping[str, Any],
        selected_base_url: str,
    ) -> dict[str, Any]:
        return {
            "adapter": "vllm_chat_v2",
            "base_urls": list(self.base_urls),
            "selected_base_url": selected_base_url,
            **self._routing_identity(request, payload),
        }

    def _cache_path(self, identity: Mapping[str, Any]) -> Path:
        key = hashlib.sha256(_canonical(identity)).hexdigest()
        return self.cache_dir / f"{key}.json"

    def _record_response_failure(
        self,
        *,
        identity: Mapping[str, Any],
        cache_path: Path,
        retry_index: int,
        raw_body: bytes,
        parsed_response: Any,
        request_id: str,
        error: json.JSONDecodeError | _RetryableResponseError,
    ) -> Path:
        """Preserve rejected HTTP responses outside the immutable success cache."""

        failure_id = f"{time.time_ns()}-{threading.get_ident()}-{uuid.uuid4().hex}"
        failure_path = self.cache_dir / "failures" / f"{cache_path.stem}-{failure_id}.json"
        write_json_atomic(
            failure_path,
            {
                "schema_version": 1,
                "cache_key": cache_path.stem,
                "request_identity": dict(identity),
                "request_id": request_id,
                "retry_index": retry_index,
                "raw_body": raw_body.decode("utf-8", errors="replace"),
                "raw_body_hash": hashlib.sha256(raw_body).hexdigest(),
                "parsed_response": parsed_response,
                "parse_error": str(error),
                "parse_error_type": type(error).__name__,
            },
            immutable=True,
        )
        return failure_path.resolve()

    def request_provenance(self, request: GenerationRequest) -> dict[str, str]:
        """Expose the deterministic transport/cache join without changing the request."""
        payload = self._payload(request)
        selected = self._select_base_url(request, payload)
        identity = self._identity(request, payload, selected)
        return {
            "selected_base_url": selected,
            "provider_cache_path": str(self._cache_path(identity).resolve()),
        }

    def _validate_model(self, returned_model: str) -> None:
        if not returned_model:
            raise VLLMChatError("chat completion did not report a model identifier")
        with self._identity_lock:
            if self._returned_model is not None and returned_model != self._returned_model:
                raise VLLMChatError(
                    f"returned model drifted from {self._returned_model} to {returned_model}"
                )
            self._returned_model = returned_model

    @staticmethod
    def _text(response: Mapping[str, Any]) -> str:
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise _RetryableResponseError("chat completion must contain exactly one choice")
        choice = choices[0]
        if not isinstance(choice, Mapping) or not isinstance(choice.get("message"), Mapping):
            raise _RetryableResponseError("chat completion choice has no message")
        text = choice["message"].get("content")
        if not isinstance(text, str) or not text.strip():
            usage = response.get("usage")
            completion_tokens = (
                usage.get("completion_tokens") if isinstance(usage, Mapping) else None
            )
            finish_reason = choice.get("finish_reason")
            raise _RetryableResponseError(
                "chat completion message has no usable text content "
                f"(finish_reason={finish_reason!r}, "
                f"completion_tokens={completion_tokens!r}, "
                f"content_type={type(text).__name__})"
            )
        return text

    @staticmethod
    def _escape_raw_c0_in_json_strings(text: str) -> tuple[str, int]:
        escapes = {
            "\b": "\\b",
            "\t": "\\t",
            "\n": "\\n",
            "\f": "\\f",
            "\r": "\\r",
        }
        output: list[str] = []
        in_string = False
        escaped = False
        count = 0
        for character in text:
            if not in_string:
                output.append(character)
                if character == '"':
                    in_string = True
                continue
            if escaped:
                output.append(character)
                escaped = False
            elif character == "\\":
                output.append(character)
                escaped = True
            elif character == '"':
                output.append(character)
                in_string = False
            elif ord(character) < 0x20:
                output.append(escapes.get(character, f"\\u{ord(character):04x}"))
                count += 1
            else:
                output.append(character)
        return "".join(output), count

    @staticmethod
    def _validate_structured_value(value: Any, request: GenerationRequest) -> None:
        expected_type = request.json_schema.get("type") if request.json_schema else None
        if expected_type == "object" and not isinstance(value, Mapping):
            raise _RetryableResponseError("structured chat completion content is not a JSON object")
        required = request.json_schema.get("required") if request.json_schema else None
        if isinstance(required, list) and isinstance(value, Mapping):
            missing = [key for key in required if key not in value]
            if missing:
                raise _RetryableResponseError(
                    f"structured chat completion is missing required keys: {missing}"
                )

    @classmethod
    def _validate_structured_text(
        cls, text: str, request: GenerationRequest
    ) -> tuple[str, tuple[int, json.JSONDecodeError] | None]:
        """Reject truncated structured completions before they enter immutable cache."""

        if request.json_schema is None:
            return text, None
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            try:
                permissive_value = json.loads(text, strict=False)
            except json.JSONDecodeError:
                raise _RetryableResponseError(
                    "structured chat completion content is not valid JSON "
                    f"(reason={error.msg!r}, line={error.lineno}, "
                    f"column={error.colno}, position={error.pos})"
                ) from error
            normalized, count = cls._escape_raw_c0_in_json_strings(text)
            if count == 0:
                raise _RetryableResponseError(
                    "structured chat completion content is not valid JSON "
                    f"(reason={error.msg!r}, line={error.lineno}, "
                    f"column={error.colno}, position={error.pos})"
                ) from error
            try:
                value = json.loads(normalized)
            except json.JSONDecodeError as normalized_error:
                raise _RetryableResponseError(
                    "structured chat completion content is not valid JSON after raw-control "
                    f"escaping (reason={normalized_error.msg!r}, line={normalized_error.lineno}, "
                    f"column={normalized_error.colno}, position={normalized_error.pos})"
                ) from normalized_error
            if value != permissive_value:
                raise _RetryableResponseError(
                    "raw-control escaping changed the structured response value"
                )
            cls._validate_structured_value(value, request)
            return normalized, (count, error)
        cls._validate_structured_value(value, request)
        return text, None

    def generate(self, request: GenerationRequest) -> GenerationResult:
        # Scheduling only: payloads, seeds, routing and cache keys remain unchanged.
        if self._in_flight is None:
            return self._generate(request)
        with self._in_flight:
            return self._generate(request)

    def _generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        payload = self._payload(request)
        selected_base_url = self._select_base_url(request, payload)
        identity = self._identity(request, payload, selected_base_url)
        cache_path = self._cache_path(identity)
        if cache_path.is_file():
            cached = read_json(cache_path)
            if cached.get("identity") != identity:
                raise VLLMChatError("cache identity does not match the immutable call")
            result = GenerationResult(**cached["result"])
            if cached.get("model_identity") != {
                "requested_model": result.requested_model,
                "returned_model": result.returned_model,
            }:
                raise VLLMChatError("cache model identity is missing or inconsistent")
            if result.requested_model != self.model:
                raise VLLMChatError("cached requested model does not match adapter model")
            try:
                validated_text, normalization = self._validate_structured_text(
                    result.text, request
                )
            except _RetryableResponseError:
                cache_path.unlink()
            else:
                if normalization is not None or validated_text != result.text:
                    cache_path.unlink()
                else:
                    self._validate_model(result.returned_model)
                    return result

        body = _canonical(payload)
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        last_error: Exception | None = None
        for retry in range(self.max_retries + 1):
            raw_body: bytes | None = None
            response: Any = None
            header_request_id = ""
            http_request = urllib.request.Request(
                f"{selected_base_url}/chat/completions",
                data=body,
                method="POST",
                headers=headers,
            )
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as raw:
                    raw_body = raw.read()
                    header_request_id = str(raw.headers.get("x-request-id") or "")
                    response = json.loads(raw_body)
                if not isinstance(response, Mapping):
                    raise _RetryableResponseError("chat completion response must be a JSON object")
                returned_model = str(response.get("model", ""))
                self._validate_model(returned_model)
                text = self._text(response)
                original_text = text
                text, normalization = self._validate_structured_text(text, request)
                choice = response["choices"][0]
                usage = dict(response.get("usage", {}))
                usage["finish_reason"] = choice.get("finish_reason")
                if self.bounded_grading_whitespace:
                    usage['structured_output_format'] = 'binary_grader_bounded_whitespace_v1'
                if normalization is not None:
                    count, parse_error = normalization
                    diagnostic_path = self._record_response_failure(
                        identity=identity,
                        cache_path=cache_path,
                        retry_index=retry,
                        raw_body=raw_body,
                        parsed_response=response,
                        request_id=str(response.get("id") or header_request_id or ""),
                        error=parse_error,
                    )
                    usage.update(
                        {
                            "structured_normalization_kind": "escape_raw_c0_in_json_strings",
                            "structured_normalization_count": count,
                            "structured_normalization_original_text_sha256": hashlib.sha256(
                                original_text.encode("utf-8")
                            ).hexdigest(),
                            "structured_normalization_diagnostic_path": str(diagnostic_path),
                        }
                    )
                result = GenerationResult(
                    text=text,
                    requested_model=self.model,
                    returned_model=returned_model,
                    request_id=str(response.get("id") or header_request_id or ""),
                    created_at=response.get("created"),
                    retry_count=retry,
                    usage=usage,
                    raw_response_hash=hashlib.sha256(_canonical(response)).hexdigest(),
                )
                write_json_atomic(
                    cache_path,
                    {
                        "schema_version": 1,
                        "identity": identity,
                        "model_identity": {
                            "requested_model": result.requested_model,
                            "returned_model": result.returned_model,
                        },
                        "result": result.__dict__,
                    },
                    immutable=True,
                )
                return result
            except (
                urllib.error.URLError,
                TimeoutError,
                json.JSONDecodeError,
                _RetryableResponseError,
            ) as error:
                last_error = error
                if raw_body is not None and isinstance(
                    error, (json.JSONDecodeError, _RetryableResponseError)
                ):
                    response_request_id = (
                        str(response.get("id") or "") if isinstance(response, Mapping) else ""
                    )
                    self._record_response_failure(
                        identity=identity,
                        cache_path=cache_path,
                        retry_index=retry,
                        raw_body=raw_body,
                        parsed_response=response,
                        request_id=response_request_id or header_request_id,
                        error=error,
                    )
                if retry >= self.max_retries:
                    break
                delay = min(30.0, 0.5 * (2**retry)) + random.Random(retry).random() * 0.1
                time.sleep(delay)
        raise VLLMChatError(f"request failed after retries: {last_error}")
