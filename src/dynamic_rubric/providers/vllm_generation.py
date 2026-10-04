from __future__ import annotations

import hashlib
import json
import math
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..seeds import vllm_seed
from .base import GenerationRequest, GenerationResult


class VLLMGenerationError(RuntimeError):
    pass


class VLLMPolicyGenerator:
    """OpenAI-compatible vLLM chat adapter with explicit non-thinking mode."""

    def __init__(
        self,
        base_url: str,
        model: str,
        revision: str,
        tokenizer_revision: str,
        timeout_seconds: float = 180.0,
        launch_spec_path: Path | None = None,
        expected_checkpoint_hash: str | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.timeout_seconds = timeout_seconds
        self.launch_spec_path = launch_spec_path
        self.expected_checkpoint_hash = expected_checkpoint_hash
        self.calls = 0

    def preflight(self) -> Mapping[str, Any]:
        expected = {
            "served_model": self.model,
            "model_revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "thinking": False,
        }
        if self.expected_checkpoint_hash:
            expected["checkpoint_hash"] = self.expected_checkpoint_hash
        try:
            with urllib.request.urlopen(
                f"{self.base_url}/dynamic-rubric/identity"
            ) as response:
                identity = json.loads(response.read())
        except Exception as identity_error:
            if self.expected_checkpoint_hash not in (None, self.revision):
                raise VLLMGenerationError(
                    "fine-tuned checkpoint identity requires /dynamic-rubric/identity"
                ) from identity_error
            if self.launch_spec_path is None or not self.launch_spec_path.is_file():
                raise VLLMGenerationError(
                    "standard vLLM policy identity requires an immutable launch spec"
                ) from identity_error
            try:
                launch_spec = json.loads(self.launch_spec_path.read_text())
            except (OSError, json.JSONDecodeError) as error:
                raise VLLMGenerationError("policy launch spec is unreadable") from error
            if any(launch_spec.get(key) != value for key, value in expected.items()):
                raise VLLMGenerationError(
                    f"policy launch spec mismatch: expected={expected}, got={launch_spec}"
                )
            model_path = Path(str(launch_spec.get("model_path", "")))
            if not model_path.is_dir() or model_path.name != self.revision:
                raise VLLMGenerationError("policy launch spec snapshot is absent or unpinned")
            with urllib.request.urlopen(f"{self.base_url}/v1/models") as response:
                models = json.loads(response.read())
            served = {
                str(item.get("id"))
                for item in models.get("data", [])
                if isinstance(item, Mapping)
            }
            if self.model not in served:
                raise VLLMGenerationError(
                    f"served policy model is absent: expected={self.model}, got={sorted(served)}"
                )
            identity = {
                **expected,
                "model_path": str(model_path),
                "launch_spec_sha256": hashlib.sha256(
                    self.launch_spec_path.read_bytes()
                ).hexdigest(),
                "identity_source": "v1/models+immutable_launch_spec",
            }
        if any(identity.get(key) != value for key, value in expected.items()):
            raise VLLMGenerationError(
                f"policy identity mismatch: expected={expected}, got={identity}"
            )
        return identity

    def generate(self, request: GenerationRequest) -> GenerationResult:
        return self.generate_many(request, 1)[0]

    def generate_many(self, request: GenerationRequest, count: int) -> Sequence[GenerationResult]:
        if count <= 0:
            raise ValueError("count must be positive")
        results: list[GenerationResult] = []
        for sample_index in range(count):
            self.calls += 1
            current = replace(request, seed=request.seed + sample_index)
            provider_seed = vllm_seed(current.seed)
            payload = {
                "model": self.model,
                "messages": [dict(message) for message in current.messages],
                "temperature": current.temperature,
                "top_p": current.top_p,
                "max_tokens": current.max_output_tokens,
                "n": 1,
                "seed": provider_seed,
                "chat_template_kwargs": {"enable_thinking": False},
            }
            http_request = urllib.request.Request(
                f"{self.base_url}/v1/chat/completions",
                data=json.dumps(payload).encode(),
                method="POST",
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                value = json.loads(response.read())
                request_id = response.headers.get("x-request-id") or value.get("id", "")
            returned_model = str(value.get("model", ""))
            if returned_model != self.model:
                raise VLLMGenerationError(
                    f"served policy model changed: expected={self.model}, got={returned_model}"
                )
            try:
                text = str(value["choices"][0]["message"]["content"])
            except (KeyError, IndexError, TypeError) as error:
                raise VLLMGenerationError("chat completion response was malformed") from error
            results.append(
                GenerationResult(
                    text=text,
                    requested_model=self.model,
                    returned_model=returned_model,
                    request_id=str(request_id),
                    created_at=value.get("created"),
                    retry_count=0,
                    usage={
                        **dict(value.get("usage", {})),
                        "dynamic_rubric_logical_seed": current.seed,
                        "dynamic_rubric_vllm_seed": provider_seed,
                    },
                    raw_response_hash=hashlib.sha256(
                        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
                    ).hexdigest(),
                )
            )
        return results


class VLLMEmbeddingProvider:
    def __init__(self, base_url: str, model: str, revision: str, timeout_seconds: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.revision = revision
        self.timeout_seconds = timeout_seconds

    @property
    def identity(self) -> Mapping[str, str]:
        return {"model": self.model, "revision": self.revision}

    def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        request = urllib.request.Request(
            f"{self.base_url}/v1/embeddings",
            data=json.dumps({"model": self.model, "input": list(texts)}).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read())
        if value.get("model") not in {None, self.model}:
            raise VLLMGenerationError("embedding model identity drifted")
        try:
            rows = sorted(value["data"], key=lambda row: row["index"])
            vectors = [[float(number) for number in row["embedding"]] for row in rows]
        except (KeyError, TypeError, ValueError) as error:
            raise VLLMGenerationError("embedding response was malformed") from error
        if len(vectors) != len(texts) or any(not vector for vector in vectors):
            raise VLLMGenerationError("embedding response count/dimension mismatch")
        normalized = []
        for vector in vectors:
            norm = math.sqrt(sum(value * value for value in vector))
            if not math.isfinite(norm) or norm == 0:
                raise VLLMGenerationError("embedding vector norm is invalid")
            normalized.append([value / norm for value in vector])
        return normalized


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("cosine vectors must have the same non-zero length")
    return sum(float(a) * float(b) for a, b in zip(left, right))
