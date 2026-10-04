from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dynamic_rubric.services.vllm_policy_proxy import PolicyProxyState


class _Headers(dict[str, str]):
    pass


class _Response:
    status = 200

    def __init__(self, payload: object) -> None:
        self.payload = payload
        self.headers = _Headers(
            {"content-type": "application/json", "x-request-id": "request-1"}
        )

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        return json.dumps(self.payload).encode()


def _state(tmp_path: Path) -> PolicyProxyState:
    model_path = tmp_path / "checkpoint"
    model_path.mkdir()
    return PolicyProxyState(
        upstream="http://policy/",
        model_path=model_path,
        served_model="policy",
        model_revision="base-revision",
        tokenizer_revision="tokenizer-revision",
        checkpoint_hash="checkpoint-sha256",
    )


def test_policy_proxy_binds_checkpoint_identity_and_validates_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(
            {"data": [{"id": "policy", "root": str(state.model_path.resolve())}]}
        ),
    )

    state.validate_upstream()
    assert state.identity == {
        "served_model": "policy",
        "model_revision": "base-revision",
        "tokenizer_revision": "tokenizer-revision",
        "thinking": False,
        "checkpoint_hash": "checkpoint-sha256",
        "model_path": str(state.model_path.resolve()),
        "identity_source": "validated-vllm-policy-proxy",
    }


def test_policy_proxy_rejects_wrong_upstream_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(
            {"data": [{"id": "policy", "root": str(tmp_path / "wrong")}]}
        ),
    )

    with pytest.raises(ValueError, match="path mismatch"):
        state.validate_upstream()


def test_policy_proxy_forwards_only_required_routes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    captured: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _Response:
        captured.append((request, timeout))
        return _Response({"model": "policy", "choices": []})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    forwarded = state.forward(
        "/v1/chat/completions", method="POST", body=b'{"model":"policy"}'
    )

    assert forwarded.status_code == 200
    assert forwarded.request_id == "request-1"
    assert captured[0][0].full_url == "http://policy/v1/chat/completions"
    assert captured[0][0].method == "POST"
    with pytest.raises(ValueError, match="unsupported"):
        state.forward("/metrics")
