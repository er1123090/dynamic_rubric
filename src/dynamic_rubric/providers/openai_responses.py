from __future__ import annotations

import hashlib
import json
import random
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_json, write_json_atomic
from .base import GenerationRequest, GenerationResult


class OpenAIResponsesError(RuntimeError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _output_text(response: Mapping[str, Any]) -> str:
    if isinstance(response.get("output_text"), str):
        return str(response["output_text"])
    parts: list[str] = []
    for item in response.get("output", []):
        if not isinstance(item, Mapping):
            continue
        for content in item.get("content", []):
            if isinstance(content, Mapping) and content.get("type") in {"output_text", "text"}:
                text = content.get("text")
                if isinstance(text, str):
                    parts.append(text)
    if not parts:
        raise OpenAIResponsesError("Responses API result contained no output text")
    return "".join(parts)


def _require_completed_response(response: Mapping[str, Any]) -> None:
    """Reject non-terminal API results before trusting any response content."""

    status = response.get("status")
    if status != "completed":
        raise OpenAIResponsesError(f"Responses API result was not completed: {status!r}")
    if response.get("incomplete_details") is not None:
        raise OpenAIResponsesError("completed Responses API result had incomplete_details")
    if response.get("error") is not None:
        raise OpenAIResponsesError("completed Responses API result reported an error")


class OpenAIResponsesAdapter:
    """Small Responses API adapter with schema/model identity hard-fails.

    No request is sent at import or construction time. The cache is written only
    after a fully parsed response has passed the returned-model drift check.
    """

    def __init__(
        self,
        api_key: str,
        model: str,
        cache_dir: Path,
        base_url: str = "https://api.openai.com/v1",
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        if not api_key:
            raise ValueError("an API key is required for the live OpenAI adapter")
        self.api_key = api_key
        self.model = model
        self.cache_dir = cache_dir
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self._returned_model: str | None = None
        self._identity_lock = threading.Lock()
        self.calls = 0

    def _payload(self, request: GenerationRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model,
            "input": [dict(message) for message in request.messages],
            "max_output_tokens": request.max_output_tokens,
        }
        if request.reasoning_effort:
            payload["reasoning"] = {"effort": request.reasoning_effort}
        if request.json_schema:
            payload["text"] = {
                "format": {
                    "type": "json_schema",
                    "name": request.schema_name or "structured_response",
                    "schema": dict(request.json_schema),
                    "strict": True,
                }
            }
        # GPT-5-family support varies by parameter. Keep requested generation
        # settings in the manifest metadata, but send only explicitly non-default
        # values so alias preflight fails visibly rather than silently substituting.
        if request.temperature != 0.0:
            payload["temperature"] = request.temperature
        if request.top_p != 1.0:
            payload["top_p"] = request.top_p
        return payload

    def _call_identity(
        self, request: GenerationRequest, payload: Mapping[str, Any]
    ) -> dict[str, Any]:
        return {
            "adapter": 2,
            "requested_model": self.model,
            "prompt_id": request.prompt_id,
            "family": request.family,
            "logical_seed": request.seed,
            "metadata": dict(request.metadata),
            "payload": payload,
        }

    def _cache_path(self, identity: Mapping[str, Any]) -> Path:
        key = hashlib.sha256(_canonical(identity)).hexdigest()
        return self.cache_dir / f"{key}.json"

    def generate(self, request: GenerationRequest) -> GenerationResult:
        self.calls += 1
        payload = self._payload(request)
        identity = self._call_identity(request, payload)
        cache_path = self._cache_path(identity)
        if cache_path.is_file():
            cached = read_json(cache_path)
            if cached.get("identity") != identity:
                raise OpenAIResponsesError("cache identity does not match the immutable call")
            result = GenerationResult(**cached["result"])
            model_identity = cached.get("model_identity")
            expected_model_identity = {
                "requested_model": result.requested_model,
                "returned_model": result.returned_model,
            }
            if model_identity != expected_model_identity:
                raise OpenAIResponsesError("cache model identity is missing or inconsistent")
            if result.requested_model != self.model:
                raise OpenAIResponsesError("cached requested model does not match adapter model")
            with self._identity_lock:
                if (
                    self._returned_model is not None
                    and result.returned_model != self._returned_model
                ):
                    raise OpenAIResponsesError(
                        f"cached returned model drifted from {self._returned_model} "
                        f"to {result.returned_model}"
                    )
                self._returned_model = result.returned_model
            return result
        body = _canonical(payload)
        idempotency_key = hashlib.sha256(_canonical(identity)).hexdigest()
        last_error: Exception | None = None
        for retry in range(self.max_retries + 1):
            http_request = urllib.request.Request(
                f"{self.base_url}/responses",
                data=body,
                method="POST",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Idempotency-Key": idempotency_key,
                },
            )
            try:
                with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as raw:
                    response = json.loads(raw.read())
                    header_request_id = raw.headers.get("x-request-id")
                _require_completed_response(response)
                returned_model = str(response.get("model", ""))
                if not returned_model:
                    raise OpenAIResponsesError("response did not report a model identifier")
                with self._identity_lock:
                    if self._returned_model is not None and returned_model != self._returned_model:
                        raise OpenAIResponsesError(
                            f"returned model drifted from {self._returned_model} "
                            f"to {returned_model}"
                        )
                    self._returned_model = returned_model
                text = _output_text(response)
                result = GenerationResult(
                    text=text,
                    requested_model=self.model,
                    returned_model=returned_model,
                    request_id=str(response.get("id") or header_request_id or ""),
                    created_at=response.get("created_at", response.get("created")),
                    retry_count=retry,
                    usage=dict(response.get("usage", {})),
                    raw_response_hash=hashlib.sha256(_canonical(response)).hexdigest(),
                )
                write_json_atomic(
                    cache_path,
                    {
                        "schema_version": 2,
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
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
                last_error = error
                if retry >= self.max_retries:
                    break
                delay = min(30.0, 0.5 * (2**retry)) + random.Random(retry).random() * 0.1
                time.sleep(delay)
        raise OpenAIResponsesError(f"request failed after retries: {last_error}")

    def generate_many(
        self,
        requests: Sequence[GenerationRequest],
        *,
        max_concurrency: int = 32,
    ) -> tuple[GenerationResult, ...]:
        """Generate a stable ordered batch with bounded request concurrency."""

        from concurrent.futures import ThreadPoolExecutor

        pending = tuple(requests)
        if max_concurrency < 1:
            raise ValueError("max_concurrency must be positive")
        if not pending:
            return ()
        try:
            with ThreadPoolExecutor(max_workers=min(max_concurrency, len(pending))) as pool:
                results = tuple(pool.map(self.generate, pending))
        except BaseException as error:
            raise OpenAIResponsesError("concurrent Responses API batch failed") from error
        if len(results) != len(pending):
            raise OpenAIResponsesError("concurrent Responses API batch was incomplete")
        return results
