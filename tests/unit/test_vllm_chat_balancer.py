from __future__ import annotations

import io
import json
import urllib.error
from typing import Any

import pytest

from dynamic_rubric.services.vllm_chat_balancer import ChatBalancerState


class _Response:
    status = 200

    def __init__(self, payload: object, request_id: str = "request-1") -> None:
        self._body = json.dumps(payload).encode()
        self.headers = {
            "content-type": "application/json",
            "x-request-id": request_id,
        }

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _models(root: str = "/models/revision") -> dict[str, Any]:
    return {
        "object": "list",
        "data": [
            {
                "id": "openai/gpt-oss-120b",
                "root": root,
                "max_model_len": 32768,
                "owned_by": "vllm",
            }
        ],
    }


def _state() -> ChatBalancerState:
    return ChatBalancerState(
        upstreams=("http://trainer/", "http://inference_b"),
        served_model="openai/gpt-oss-120b",
    )


def test_balancer_requires_identical_model_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def urlopen(request: Any, *, timeout: float) -> _Response:
        url = request if isinstance(request, str) else request.full_url
        return _Response(_models("/models/trainer" if "trainer" in url else "/models/inference_b"))

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(ValueError, match="semantic identity mismatch"):
        _state().validate_upstreams()


def test_balancer_exposes_canonical_identity_and_round_robins(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def urlopen(request: Any, *, timeout: float) -> _Response:
        url = request if isinstance(request, str) else request.full_url
        calls.append(url)
        if url.endswith("/v1/models"):
            return _Response(_models())
        return _Response({"model": "openai/gpt-oss-120b", "choices": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    state = _state()
    state.validate_upstreams()
    assert json.loads(state.models_response().body)["data"][0]["root"] == "/models/revision"
    assert state.forward_chat(b"{}").upstream == "http://trainer"
    assert state.forward_chat(b"{}").upstream == "http://inference_b"
    assert state.stats()["requests"] == {"http://trainer": 1, "http://inference_b": 1}


def test_balancer_retries_a_second_upstream_on_server_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def urlopen(request: Any, *, timeout: float) -> _Response:
        url = request.full_url
        calls.append(url)
        if "trainer" in url:
            raise urllib.error.HTTPError(
                url, 503, "busy", {"content-type": "application/json"}, io.BytesIO(b"{}")
            )
        return _Response({"choices": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    result = _state().forward_chat(b"{}")
    assert result.upstream == "http://inference_b"
    assert calls == [
        "http://trainer/v1/chat/completions",
        "http://inference_b/v1/chat/completions",
    ]
