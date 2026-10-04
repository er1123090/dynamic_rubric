from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.openai_responses import OpenAIResponsesAdapter
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter, VLLMChatError
from dynamic_rubric.prompt_versions.onlinerubric_grader_prompt import onlinerubric_grader_schema
from dynamic_rubric.training.verl_online_runtime import (
    OnlineRuntimeFactoryError,
    _online_generation_providers,
)


class _Response:
    def __init__(self, body: dict[str, Any] | bytes) -> None:
        self.body = body
        self.headers = {"x-request-id": "header-request"}

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def read(self) -> bytes:
        if isinstance(self.body, bytes):
            return self.body
        return json.dumps(self.body).encode()


def test_bounded_grading_whitespace_preserves_request_and_versions_transport(tmp_path):
    request = GenerationRequest(prompt_id='p1', messages=({'role': 'user', 'content': 'grade'},),
                                family='phase1_audit_grading', seed=22, max_output_tokens=4096,
                                json_schema=onlinerubric_grader_schema(11), schema_name='onlinerubric_grader_v1')
    normal = VLLMChatAdapter('http://judge', 'Qwen/Qwen3-32B', tmp_path)
    fixed = VLLMChatAdapter('http://judge', 'Qwen/Qwen3-32B', tmp_path, bounded_grading_whitespace=True)
    before, after = normal._payload(request), fixed._payload(request)
    grammar = after.pop('structured_outputs')['grammar']
    before.pop('response_format')
    assert before == after
    assert '[ \\t\\r\\n]?' in grammar
    assert '*' not in grammar and '+' not in grammar
    assert grammar.count(' ws grade') == 11
    assert 'PRESENT' in grammar and 'NOT_PRESENT' in grammar
    assert normal.request_provenance(request) != fixed.request_provenance(request)
    with pytest.raises(ValueError, match='scoped'):
        fixed._payload(_request())
    with pytest.raises(ValueError, match='exact binary'):
        fixed._bounded_grading_grammar({'type': 'object', 'properties': {}})


def test_global_in_flight_bound_preserves_payload_cache_and_releases_on_error(tmp_path, monkeypatch):
    limited = VLLMChatAdapter('http://judge', 'model', tmp_path, max_in_flight=2)
    normal = VLLMChatAdapter('http://judge', 'model', tmp_path)
    request = _request()
    assert limited._payload(request) == normal._payload(request)
    assert limited._identity(request, limited._payload(request), limited.base_url) == normal._identity(
        request, normal._payload(request), normal.base_url
    )
    lock = threading.Lock()
    active = peak = 0

    def generate(item):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.005)
            return item
        finally:
            with lock:
                active -= 1

    monkeypatch.setattr(limited, '_generate', generate)
    with ThreadPoolExecutor(max_workers=12) as executor:
        assert list(executor.map(limited.generate, range(24))) == list(range(24))
    assert peak == 2
    for _ in range(4):
        monkeypatch.setattr(limited, '_generate', lambda item: (_ for _ in ()).throw(RuntimeError('failure')))
        with pytest.raises(RuntimeError, match='failure'):
            limited.generate(request)
    monkeypatch.setattr(limited, '_generate', generate)
    assert limited.generate(request) is request
    with pytest.raises(ValueError, match='max_in_flight'):
        VLLMChatAdapter('http://judge', 'model', tmp_path, max_in_flight=0)


def _request(
    *, reasoning_effort: str | None = "medium", prompt_id: str = "p1"
) -> GenerationRequest:
    return GenerationRequest(
        prompt_id=prompt_id,
        messages=({"role": "user", "content": "return JSON"},),
        family="extractor",
        seed=11,
        temperature=0.25,
        top_p=0.9,
        max_output_tokens=77,
        json_schema={"type": "object", "properties": {"ok": {"type": "boolean"}}},
        schema_name="result_v1",
        reasoning_effort=reasoning_effort,
        metadata={"checkpoint": 3},
    )


def test_vllm_chat_structured_payload_no_auth_and_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    requests: list[Any] = []

    def urlopen(request: Any, *, timeout: float) -> _Response:
        requests.append(request)
        assert timeout == 9
        return _Response(
            {
                "id": "chatcmpl-1",
                "created": 123,
                "model": "gpt-oss-120b",
                "choices": [{"message": {"role": "assistant", "content": '{"ok":true}'}}],
                "usage": {"completion_tokens": 4},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    adapter = VLLMChatAdapter(
        "http://inference_a:8000",
        "gpt-oss-120b",
        tmp_path / "cache",
        timeout_seconds=9,
        max_retries=0,
    )
    result = adapter.generate(_request())
    cached = adapter.generate(_request())

    assert result == cached
    assert len(requests) == 1
    sent = requests[0]
    assert sent.full_url == "http://inference_a:8000/v1/chat/completions"
    assert sent.get_method() == "POST"
    assert sent.get_header("Authorization") is None
    payload = json.loads(sent.data)
    assert payload["seed"] == 11
    assert payload["reasoning_effort"] == "medium"
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert payload["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "result_v1",
            "schema": _request().json_schema,
            "strict": True,
        },
    }


def test_vllm_chat_omits_reasoning_and_retries_transient_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def urlopen(request: Any, *, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise urllib.error.URLError("not ready")
        assert "reasoning_effort" not in json.loads(request.data)
        assert request.get_header("Authorization") == "Bearer dummy"
        return _Response(
            {
                "id": "chatcmpl-2",
                "model": "Qwen3-32B",
                "choices": [{"message": {"content": '{"ok":true}'}}],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    result = VLLMChatAdapter(
        "http://inference_b:8000/v1/",
        "Qwen3-32B",
        tmp_path,
        api_key="dummy",
        max_retries=1,
    ).generate(_request(reasoning_effort=None))
    assert result.retry_count == 1
    assert calls == 2


def test_vllm_chat_retries_empty_success_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def urlopen(request: Any, *, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        content = None if calls == 1 else json.dumps({"ok": True})
        return _Response(
            {
                "id": f"chatcmpl-{calls}",
                "model": "gpt-oss-120b",
                "choices": [
                    {
                        "finish_reason": "length" if calls == 1 else "stop",
                        "message": {"content": content},
                    }
                ],
                "usage": {"completion_tokens": 77 if calls == 1 else 4},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    result = VLLMChatAdapter(
        "http://inference_a:8000",
        "gpt-oss-120b",
        tmp_path,
        max_retries=1,
    ).generate(_request())

    assert calls == 2
    assert result.retry_count == 1
    assert result.text == json.dumps({"ok": True})


def test_vllm_chat_retries_truncated_structured_content_before_caching(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def urlopen(request: Any, *, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        content = '{"ok": "unterminated' if calls == 1 else '{"ok": true}'
        return _Response(
            {
                "id": f"chatcmpl-{calls}",
                "model": "gpt-oss-120b",
                "choices": [
                    {
                        "finish_reason": "length" if calls == 1 else "stop",
                        "message": {"content": content},
                    }
                ],
                "usage": {"completion_tokens": 77 if calls == 1 else 4},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    adapter = VLLMChatAdapter("http://inference_a:8000", "gpt-oss-120b", tmp_path, max_retries=1)

    result = adapter.generate(_request())

    assert calls == 2
    assert result.retry_count == 1
    assert result.usage["finish_reason"] == "stop"
    assert json.loads(result.text) == {"ok": True}
    assert len(list(tmp_path.glob("*.json"))) == 1
    failures = list((tmp_path / "failures").glob("*.json"))
    assert len(failures) == 1
    failure = json.loads(failures[0].read_text())
    assert failure["retry_index"] == 0
    assert failure["request_id"] == "chatcmpl-1"
    assert failure["parsed_response"]["choices"][0]["message"]["content"] == (
        '{"ok": "unterminated'
    )
    assert failure["raw_body"]
    assert failure["cache_key"] == adapter._cache_path(
        adapter._identity(
            _request(),
            adapter._payload(_request()),
            adapter._select_base_url(_request(), adapter._payload(_request())),
        )
    ).stem
    assert "Unterminated string" in failure["parse_error"]


def test_vllm_chat_preserves_each_malformed_http_response_without_secrets_or_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    malformed = b'{"id":"bad-json","choices":['
    calls = 0

    def urlopen(*args: Any, **kwargs: Any) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(malformed)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    adapter = VLLMChatAdapter(
        "http://inference_a:8000",
        "gpt-oss-120b",
        tmp_path,
        api_key="dummy-api-key",
        max_retries=1,
    )

    with pytest.raises(VLLMChatError, match="Expecting value"):
        adapter.generate(_request())

    assert calls == 2
    assert list(tmp_path.glob("*.json")) == []
    failures = sorted((tmp_path / "failures").glob("*.json"))
    assert len(failures) == 2
    records = [json.loads(path.read_text()) for path in failures]
    assert {record["retry_index"] for record in records} == {0, 1}
    for record in records:
        assert record["raw_body"] == malformed.decode()
        assert record["parsed_response"] is None
        assert record["request_id"] == "header-request"
        assert record["cache_key"]
        assert record["raw_body_hash"]
        assert record["request_identity"]["prompt_id"] == "p1"
        serialized = json.dumps(record)
        assert "Authorization" not in serialized
        assert "dummy-api-key" not in serialized


@pytest.mark.parametrize("control", ["\t", "\n", "\r", "\x00", "\x08", "\x0c", "\x1f"])
def test_vllm_chat_losslessly_escapes_raw_c0_inside_structured_json_strings(
    control: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_text = '{"analysis":"ok","new_criteria":[{"quote":"a' + control + 'b"}]}'
    response_body = {
        "id": "chatcmpl-control",
        "model": "gpt-oss-120b",
        "choices": [{"finish_reason": "stop", "message": {"content": original_text}}],
        "usage": {"completion_tokens": 12},
    }
    calls = 0

    def urlopen(*args: Any, **kwargs: Any) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(response_body)

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    adapter = VLLMChatAdapter("http://inference_a:8000", "gpt-oss-120b", tmp_path)
    request = _request()

    result = adapter.generate(request)
    cached = adapter.generate(request)

    assert calls == 1
    assert result == cached
    assert json.loads(result.text)["new_criteria"][0]["quote"] == f"a{control}b"
    assert control not in result.text
    assert result.usage["structured_normalization_kind"] == "escape_raw_c0_in_json_strings"
    assert result.usage["structured_normalization_count"] == 1
    assert result.usage["structured_normalization_original_text_sha256"] == hashlib.sha256(
        original_text.encode()
    ).hexdigest()
    diagnostic_path = Path(result.usage["structured_normalization_diagnostic_path"])
    assert diagnostic_path.is_file()
    diagnostic = json.loads(diagnostic_path.read_text())
    assert diagnostic["parsed_response"]["choices"][0]["message"]["content"] == original_text
    assert result.raw_response_hash == hashlib.sha256(
        json.dumps(response_body, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_vllm_chat_leaves_valid_structured_json_byte_for_byte_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    text = '{ "ok" : true, "value": "backslash \\\\ quote \\\" escaped-tab \\t" }'
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(
            {
                "id": "chatcmpl-valid",
                "model": "gpt-oss-120b",
                "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            }
        ),
    )
    result = VLLMChatAdapter(
        "http://inference_a:8000", "gpt-oss-120b", tmp_path, max_retries=0
    ).generate(_request())

    assert result.text == text
    assert "structured_normalization_kind" not in result.usage
    assert not (tmp_path / "failures").exists()


@pytest.mark.parametrize(
    "text",
    [
        '{"ok":"invalid\\q"}',
        '{"ok":"unclosed}',
        '{"ok": true',
        '{"ok":\x00true}',
    ],
)
def test_vllm_chat_does_not_normalize_other_json_lexical_failures(
    text: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(
            {
                "id": "chatcmpl-still-invalid",
                "model": "gpt-oss-120b",
                "choices": [{"finish_reason": "stop", "message": {"content": text}}],
            }
        ),
    )
    adapter = VLLMChatAdapter(
        "http://inference_a:8000", "gpt-oss-120b", tmp_path, max_retries=0
    )

    with pytest.raises(VLLMChatError):
        adapter.generate(_request())
    assert list(tmp_path.glob("*.json")) == []


def test_vllm_chat_discards_invalid_structured_cache_and_regenerates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    request = _request()
    adapter = VLLMChatAdapter("http://inference_a:8000", "gpt-oss-120b", tmp_path, max_retries=0)
    payload = adapter._payload(request)
    selected = adapter._select_base_url(request, payload)
    identity = adapter._identity(request, payload, selected)
    cache_path = adapter._cache_path(identity)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "identity": identity,
                "model_identity": {
                    "requested_model": "gpt-oss-120b",
                    "returned_model": "gpt-oss-120b",
                },
                "result": {
                    "text": '{"ok": "unterminated',
                    "requested_model": "gpt-oss-120b",
                    "returned_model": "gpt-oss-120b",
                    "request_id": "bad-cache",
                    "created_at": 1,
                    "retry_count": 0,
                    "usage": {"completion_tokens": 77},
                    "raw_response_hash": None,
                },
            }
        )
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda *args, **kwargs: _Response(
            {
                "id": "chatcmpl-recovered",
                "model": "gpt-oss-120b",
                "choices": [{"finish_reason": "stop", "message": {"content": '{"ok": true}'}}],
                "usage": {"completion_tokens": 4},
            }
        ),
    )

    result = adapter.generate(request)

    assert result.request_id == "chatcmpl-recovered"
    assert json.loads(result.text) == {"ok": True}
    assert json.loads(cache_path.read_text())["result"]["request_id"] == "chatcmpl-recovered"


def test_vllm_chat_fails_closed_after_repeated_empty_success_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls = 0

    def urlopen(*args: Any, **kwargs: Any) -> _Response:
        nonlocal calls
        calls += 1
        return _Response(
            {
                "model": "gpt-oss-120b",
                "choices": [{"finish_reason": "length", "message": {"content": None}}],
                "usage": {"completion_tokens": 77},
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    adapter = VLLMChatAdapter(
        "http://inference_a:8000",
        "gpt-oss-120b",
        tmp_path,
        max_retries=1,
    )

    with pytest.raises(VLLMChatError, match="finish_reason=\x27length\x27"):
        adapter.generate(_request())
    assert calls == 2


def test_vllm_chat_rejects_returned_model_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    bodies = iter(
        [
            {"model": "served-a", "choices": [{"message": {"content": "{}"}}]},
            {"model": "served-b", "choices": [{"message": {"content": "{}"}}]},
        ]
    )
    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: _Response(next(bodies)))
    adapter = VLLMChatAdapter("http://localhost:8000", "requested", tmp_path)
    adapter.generate(_request())
    with pytest.raises(VLLMChatError, match="model drifted"):
        adapter.generate(_request(prompt_id="p2"))


def test_runtime_provider_routing_and_legacy_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URL", "http://inference_a:8000/v1")
    monkeypatch.setenv("PHASE1_QWEN32B_BASE_URL", "http://inference_b:8000/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    extractor, grader = _online_generation_providers(
        extractor_model="gpt-oss-120b",
        grader_model="Qwen3-32B",
        cache_root=tmp_path,
    )
    assert isinstance(extractor, VLLMChatAdapter)
    assert isinstance(grader, VLLMChatAdapter)
    assert extractor.base_url == "http://inference_a:8000/v1"
    assert grader.base_url == "http://inference_b:8000/v1"

    monkeypatch.delenv("PHASE1_GPT_OSS_BASE_URL")
    monkeypatch.delenv("PHASE1_QWEN32B_BASE_URL")
    monkeypatch.setenv("OPENAI_API_KEY", "secret")
    extractor, grader = _online_generation_providers(
        extractor_model="extractor", grader_model="grader", cache_root=tmp_path
    )
    assert isinstance(extractor, OpenAIResponsesAdapter)
    assert isinstance(grader, OpenAIResponsesAdapter)


def test_runtime_rejects_partial_vllm_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URL", "http://inference_a:8000")
    monkeypatch.delenv("PHASE1_QWEN32B_BASE_URL", raising=False)
    with pytest.raises(OnlineRuntimeFactoryError, match="must be set together"):
        _online_generation_providers(
            extractor_model="extractor", grader_model="grader", cache_root=tmp_path
        )


def test_vllm_chat_two_endpoint_routing_is_deterministic_and_retry_sticky(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    attempted_urls: list[str] = []
    calls = 0

    def urlopen(request: Any, *, timeout: float) -> _Response:
        nonlocal calls
        calls += 1
        attempted_urls.append(request.full_url)
        if calls == 1:
            raise urllib.error.URLError("warming")
        return _Response(
            {
                "model": "gpt-oss-120b",
                "choices": [{"message": {"content": '{"ok":true}'}}],
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    monkeypatch.setattr("time.sleep", lambda _: None)
    endpoints = ("http://inference_a:8101", "http://inference_a:8102/v1")
    adapter = VLLMChatAdapter(
        endpoints,
        "gpt-oss-120b",
        tmp_path / "first",
        max_retries=1,
    )
    adapter.generate(_request(prompt_id="stable"))

    assert len(attempted_urls) == 2
    assert attempted_urls[0] == attempted_urls[1]
    assert attempted_urls[0] in {
        "http://inference_a:8101/v1/chat/completions",
        "http://inference_a:8102/v1/chat/completions",
    }

    other = VLLMChatAdapter(endpoints, "gpt-oss-120b", tmp_path / "second")
    request = _request(prompt_id="stable")
    assert adapter._select_base_url(request, adapter._payload(request)) == other._select_base_url(
        request, other._payload(request)
    )

    selected = {
        adapter._select_base_url(candidate, adapter._payload(candidate))
        for candidate in (_request(prompt_id=f"p-{index}") for index in range(64))
    }
    assert selected == {"http://inference_a:8101/v1", "http://inference_a:8102/v1"}


def test_runtime_accepts_two_endpoints_per_evaluator_family(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("PHASE1_GPT_OSS_BASE_URL", raising=False)
    monkeypatch.delenv("PHASE1_QWEN32B_BASE_URL", raising=False)
    monkeypatch.setenv(
        "PHASE1_GPT_OSS_BASE_URLS",
        "http://inference_a:8101/v1, http://inference_a:8102/v1",
    )
    monkeypatch.setenv(
        "PHASE1_QWEN32B_BASE_URLS",
        "http://inference_b:8201/v1,http://inference_b:8202/v1",
    )

    extractor, grader = _online_generation_providers(
        extractor_model="gpt-oss-120b",
        grader_model="Qwen3-32B",
        cache_root=tmp_path,
    )

    assert extractor.base_urls == (
        "http://inference_a:8101/v1",
        "http://inference_a:8102/v1",
    )
    assert grader.base_urls == ("http://inference_b:8201/v1", "http://inference_b:8202/v1")


def test_runtime_rejects_ambiguous_or_oversized_endpoint_configuration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URL", "http://inference_a:8101")
    monkeypatch.setenv(
        "PHASE1_GPT_OSS_BASE_URLS",
        "http://inference_a:8101,http://inference_a:8102",
    )
    with pytest.raises(OnlineRuntimeFactoryError, match="cannot both be set"):
        _online_generation_providers(
            extractor_model="extractor", grader_model="grader", cache_root=tmp_path
        )

    monkeypatch.delenv("PHASE1_GPT_OSS_BASE_URL")
    monkeypatch.setenv(
        "PHASE1_GPT_OSS_BASE_URLS",
        "http://inference_a:1,http://inference_a:2,http://inference_a:3",
    )
    with pytest.raises(OnlineRuntimeFactoryError, match="one or two"):
        _online_generation_providers(
            extractor_model="extractor", grader_model="grader", cache_root=tmp_path
        )
