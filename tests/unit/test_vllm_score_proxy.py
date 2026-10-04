from __future__ import annotations

import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock
from typing import Any

import pytest

from dynamic_rubric.services.vllm_score_proxy import ProxyState


def _state(cache_dir: Path, *, model_revision: str = "revision-a") -> ProxyState:
    return ProxyState(
        upstream="http://unused",
        served_model="grader",
        model_revision=model_revision,
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=cache_dir,
    )


def test_score_cache_is_durable_and_deduplicates_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = _state(tmp_path)
    calls: list[list[str]] = []

    def score_uncached(
        prompts: list[str], targets: list[str], *, upstream_index: int | None = None
    ) -> list[dict[str, float]]:
        del upstream_index
        calls.append(prompts)
        return [{"YES": -0.25, "NO": -1.5} for _ in prompts]

    monkeypatch.setattr(state, "_score_uncached", score_uncached)
    assert state.score(["same", "same"], ["YES", "NO"]) == [
        {"YES": -0.25, "NO": -1.5},
        {"YES": -0.25, "NO": -1.5},
    ]
    assert calls == [["same"]]

    restarted = _state(tmp_path)

    def must_not_score(*args: Any) -> list[dict[str, float]]:
        raise AssertionError("durable cache was not used")

    monkeypatch.setattr(restarted, "_score_uncached", must_not_score)
    assert restarted.score(["same"], ["YES", "NO"]) == [{"YES": -0.25, "NO": -1.5}]


def test_score_cache_key_binds_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    first = _state(tmp_path, model_revision="revision-a")
    second = _state(tmp_path, model_revision="revision-b")
    calls = 0

    def score_uncached(
        prompts: list[str], targets: list[str], *, upstream_index: int | None = None
    ) -> list[dict[str, float]]:
        del upstream_index
        nonlocal calls
        calls += 1
        return [{"YES": -float(calls), "NO": -2.0} for _ in prompts]

    monkeypatch.setattr(first, "_score_uncached", score_uncached)
    monkeypatch.setattr(second, "_score_uncached", score_uncached)
    assert first.score(["prompt"], ["YES", "NO"])[0]["YES"] == -1.0
    assert second.score(["prompt"], ["YES", "NO"])[0]["YES"] == -2.0
    assert calls == 2


def test_multi_upstream_routing_is_deterministic_and_content_addressed(
    tmp_path: Path,
) -> None:
    state = ProxyState(
        upstream=["http://replica-a/", "http://replica-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
    )

    assignments = {
        prompt: state._select_upstream([prompt], ["YES", "NO"])
        for prompt in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    }
    assert set(assignments.values()) == {"http://replica-a", "http://replica-b"}
    assert assignments == {
        prompt: state._select_upstream([prompt], ["YES", "NO"]) for prompt in assignments
    }


def test_multi_upstream_routing_honors_positive_integer_weights(tmp_path: Path) -> None:
    state = ProxyState(
        upstream=["http://fast", "http://slow-a", "http://slow-b"],
        upstream_weights=[5, 2, 2],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
    )
    assignments = [
        state._select_upstream([f"prompt-{index}"], ["YES", "NO"]) for index in range(900)
    ]
    assert assignments.count("http://fast") > assignments.count("http://slow-a") * 2
    assert assignments.count("http://fast") > assignments.count("http://slow-b") * 2


def test_multi_upstream_batch_is_length_balanced_parallel_and_order_preserving(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://replica-a", "http://replica-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=None,
    )
    prompts = ["a" * 100, "b" * 80, "c" * 20, "d" * 10]
    barrier = Barrier(2)
    visited: list[int] = []

    def score_on_upstream(
        shard: list[str], targets: list[str], *, upstream_index: int | None = None
    ) -> list[dict[str, float]]:
        assert targets == ["YES", "NO"]
        assert upstream_index is not None
        visited.append(upstream_index)
        barrier.wait(timeout=2)
        return [
            {"YES": float(prompts.index(prompt)), "NO": float(upstream_index)} for prompt in shard
        ]

    monkeypatch.setattr(state, "_score_on_upstream", score_on_upstream)
    rows = state.score(prompts, ["YES", "NO"])
    assert sorted(visited) == [0, 1]
    assert [row["YES"] for row in rows] == [0.0, 1.0, 2.0, 3.0]
    assert set(state._balanced_assignments(prompts)) == {0, 1}


def test_request_affinity_keeps_a_multi_prompt_batch_on_one_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://replica-a", "http://replica-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        request_affinity=True,
        cache_dir=None,
    )
    prompts = ["shared-prefix criterion-a", "shared-prefix criterion-b"]
    visited: list[tuple[int, tuple[str, ...]]] = []
    monkeypatch.setattr(
        state,
        "_select_upstream_index",
        lambda rendered_prompts, targets: 1,
    )

    def score_on_upstream(
        shard: list[str], targets: list[str], *, upstream_index: int | None = None
    ) -> list[dict[str, float]]:
        assert targets == ["YES", "NO"]
        assert upstream_index is not None
        visited.append((upstream_index, tuple(shard)))
        return [{"YES": float(index), "NO": -1.0} for index, _ in enumerate(shard)]

    monkeypatch.setattr(state, "_score_on_upstream", score_on_upstream)
    rows = state.score(prompts, ["YES", "NO"])

    assert visited == [(1, tuple(prompts))]
    assert [row["YES"] for row in rows] == [0.0, 1.0]


def test_multi_upstream_transient_failure_falls_back_without_reordering(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://replica-a", "http://replica-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=None,
    )
    prompts = ["first", "second"]
    attempts: list[tuple[int, tuple[str, ...]]] = []
    monkeypatch.setattr(state, "_balanced_assignments", lambda _: (0, 1))

    def score_on_upstream(
        shard: list[str], targets: list[str], *, upstream_index: int | None = None
    ) -> list[dict[str, float]]:
        assert targets == ["YES", "NO"]
        assert upstream_index is not None
        attempts.append((upstream_index, tuple(shard)))
        if upstream_index == 1:
            raise urllib.error.URLError("replica unavailable")
        return [
            {"YES": float(prompts.index(prompt)), "NO": float(upstream_index)} for prompt in shard
        ]

    monkeypatch.setattr(state, "_score_on_upstream", score_on_upstream)
    rows = state.score(prompts, ["YES", "NO"])

    assert [row["YES"] for row in rows] == [0.0, 1.0]
    assert (1, ("second",)) in attempts
    assert (0, ("second",)) in attempts


def test_multi_upstream_non_transient_failure_is_not_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://replica-a", "http://replica-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=None,
    )
    monkeypatch.setattr(state, "_balanced_assignments", lambda _: (0, 1))

    def score_on_upstream(*args: Any, **kwargs: Any) -> list[dict[str, float]]:
        del args, kwargs
        raise ValueError("score identity mismatch")

    monkeypatch.setattr(state, "_score_on_upstream", score_on_upstream)
    with pytest.raises(ValueError, match="identity mismatch"):
        state.score(["first", "second"], ["YES", "NO"])


def test_adaptive_weights_boost_local_replica_only_below_threshold(tmp_path: Path) -> None:
    utilization = [15.0]
    state = ProxyState(
        upstream=["http://local", "http://remote-a", "http://remote-b"],
        upstream_weights=[1, 2, 2],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
        adaptive_upstream_index=0,
        adaptive_gpu_index=0,
        adaptive_low_utilization_weight=2,
        adaptive_utilization_threshold=70,
        utilization_cache_seconds=0,
        utilization_reader=lambda _: utilization[0],
    )

    assert state._effective_weights() == (2, 2, 2)
    utilization[0] = 95.0
    assert state._effective_weights() == (1, 2, 2)


def test_managed_sleep_upstream_wakes_once_and_sleeps_after_parallel_requests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://local", "http://remote-a", "http://remote-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=None,
        managed_sleep_upstream_index=0,
        managed_sleep_idle_seconds=0.01,
    )
    transitions: list[bool] = []
    transitions_lock = Lock()

    def transition(awake: bool) -> None:
        with transitions_lock:
            transitions.append(awake)

    monkeypatch.setattr(state, "_set_managed_upstream_awake", transition)
    state.sleep_managed_upstream()
    transitions.clear()

    barrier = Barrier(3)

    def score_uncached(
        prompts: list[str], targets: list[str], *, upstream_index: int | None = None
    ) -> list[dict[str, float]]:
        del targets, upstream_index
        barrier.wait(timeout=2)
        return [{"YES": -0.25, "NO": -1.5} for _ in prompts]

    monkeypatch.setattr(state, "_score_uncached", score_uncached)
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(state.score, [prompt], ["YES", "NO"]) for prompt in ("first", "second")
        ]
        barrier.wait(timeout=2)
        assert transitions == [True]
        assert [future.result(timeout=2)[0]["YES"] for future in futures] == [-0.25, -0.25]

    assert transitions == [True, False]
    assert state.runtime_metrics["managed_sleep_state"] == "asleep"
    assert state.runtime_metrics["managed_wake_count"] == 1
    assert state.runtime_metrics["managed_sleep_count"] == 2


def test_managed_sleep_upstream_does_not_wake_for_cache_hits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://local", "http://remote"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
        managed_sleep_upstream_index=0,
        managed_sleep_idle_seconds=0,
    )
    targets = ["YES", "NO"]
    prompt = "already cached"
    key = state._cache_key(prompt, targets)
    state._publish_or_read_cache(key, prompt, targets, {"YES": -0.25, "NO": -1.5})
    state._managed_awake = False

    def unexpected_transition(awake: bool) -> None:
        raise AssertionError(f"cache hit attempted managed transition: awake={awake}")

    monkeypatch.setattr(state, "_set_managed_upstream_awake", unexpected_transition)
    assert state.score([prompt], targets) == [{"YES": -0.25, "NO": -1.5}]


def test_managed_sleep_initialization_reuses_an_already_sleeping_upstream(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://local", "http://remote"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
        managed_sleep_upstream_index=0,
    )
    monkeypatch.setattr(state, "_managed_upstream_is_sleeping", lambda: True)

    def unexpected_transition(awake: bool) -> None:
        raise AssertionError(f"already-sleeping upstream transitioned: awake={awake}")

    monkeypatch.setattr(state, "_set_managed_upstream_awake", unexpected_transition)
    state.initialize_managed_upstream()

    assert state.runtime_metrics["managed_sleep_state"] == "asleep"
    assert state.runtime_metrics["managed_sleep_count"] == 0


def test_managed_wake_waits_for_gpu_capacity(tmp_path: Path) -> None:
    memory_used = iter([90_000.0, 70_000.0, 50_000.0])
    readings: list[float] = []

    def read_memory(_: int) -> float:
        reading = next(memory_used)
        readings.append(reading)
        return reading

    state = ProxyState(
        upstream=["http://local", "http://remote"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
        managed_sleep_upstream_index=0,
        managed_wake_gpu_index=1,
        managed_wake_max_memory_used_mib=60_000,
        managed_wake_poll_seconds=0.001,
        memory_used_reader=read_memory,
    )

    state._wait_for_managed_wake_capacity()
    assert readings == [90_000.0, 70_000.0, 50_000.0]


@pytest.mark.parametrize(
    ("index", "idle_seconds", "match"),
    [(3, 1.0, "managed sleep upstream index"), (0, -0.1, "managed sleep idle")],
)
def test_managed_sleep_upstream_rejects_invalid_settings(
    tmp_path: Path, index: int, idle_seconds: float, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        ProxyState(
            upstream=["http://local", "http://remote"],
            served_model="grader",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            tokenizer=object(),
            cache_dir=tmp_path,
            managed_sleep_upstream_index=index,
            managed_sleep_idle_seconds=idle_seconds,
        )


@pytest.mark.parametrize("weights", ([1], [1, 0], [1, 1.5], [1, True]))
def test_multi_upstream_rejects_invalid_weights(tmp_path: Path, weights: list[object]) -> None:
    with pytest.raises(ValueError, match="weights"):
        ProxyState(
            upstream=["http://replica-a", "http://replica-b"],
            upstream_weights=weights,  # type: ignore[arg-type]
            served_model="grader",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            tokenizer=object(),
            cache_dir=tmp_path,
        )


def test_rejects_non_positive_upstream_timeout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="upstream timeout"):
        ProxyState(
            upstream="http://replica-a",
            served_model="grader",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            tokenizer=object(),
            cache_dir=tmp_path,
            upstream_timeout_seconds=0,
        )


def test_explicit_routing_index_is_deterministic_and_bounded(tmp_path: Path) -> None:
    state = ProxyState(
        upstream=["http://replica-a", "http://replica-b"],
        upstream_weights=[3, 1],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
    )
    assert state._select_upstream(["same"], ["YES", "NO"], upstream_index=1) == "http://replica-b"
    with pytest.raises(ValueError, match="out of range"):
        state._select_upstream(["same"], ["YES", "NO"], upstream_index=2)


def test_multi_upstream_validation_checks_every_replica(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    state = ProxyState(
        upstream=["http://replica-a", "http://replica-b"],
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=object(),
        cache_dir=tmp_path,
    )
    visited: list[str] = []

    class Response:
        def __init__(self, body: dict[str, Any]) -> None:
            self.body = body

        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return __import__("json").dumps(self.body).encode()

    def urlopen(request: Any, *, timeout: float) -> Response:
        del timeout
        visited.append(request.full_url)
        if request.full_url.startswith("http://replica-b"):
            return Response({"data": [{"id": "wrong-model"}]})
        return Response({"data": [{"id": "grader"}]})

    monkeypatch.setattr("urllib.request.urlopen", urlopen)
    with pytest.raises(ValueError, match="replica-b.*expected=grader"):
        state.validate_upstreams()
    assert visited == [
        "http://replica-a/v1/models",
        "http://replica-b/v1/models",
    ]


def test_multi_upstream_rejects_duplicates_and_empty_values(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unique"):
        ProxyState(
            upstream=["http://same", "http://same/"],
            served_model="grader",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            tokenizer=object(),
            cache_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="at least one"):
        ProxyState(
            upstream=[],
            served_model="grader",
            model_revision="revision-a",
            tokenizer_revision="tokenizer-a",
            tokenizer=object(),
            cache_dir=tmp_path,
        )


def test_upstream_scores_each_single_token_target_against_full_vocabulary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if add_special_tokens:
                return [1]
            return {" YES": [2], " NO": [3]}[text]

    captured_payloads: list[dict[str, object]] = []

    class Response:
        def __enter__(self) -> Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                __import__("json")
                .dumps(
                    {
                        "model": "grader",
                        "choices": [
                            {
                                "index": index,
                                "prompt_token_ids": prompt_ids,
                                "token_ids": [],
                                "logprobs": {
                                    "token_logprobs": [
                                        None,
                                        {2: -0.25, 3: -1.5}[prompt_ids[-1]],
                                    ]
                                },
                            }
                            for index, prompt_ids in enumerate(captured_payloads[-1]["prompt"])
                        ],
                    }
                )
                .encode()
            )

    state = ProxyState(
        upstream="http://replica",
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=Tokenizer(),
        cache_dir=None,
    )

    def fake_urlopen(request: object, timeout: float) -> Response:
        del timeout
        payload = __import__("json").loads(request.data)
        captured_payloads.append(payload)
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert state.score(["prompt"], [" YES", " NO"]) == [{" YES": -0.25, " NO": -1.5}]
    assert len(captured_payloads) == 1
    payload = captured_payloads[0]
    assert payload["prompt"] == [[1, 2], [1, 3]]
    assert payload["logprobs"] == 1
    assert payload["max_tokens"] == 0
    assert payload["echo"] is True
    assert payload["return_tokens_as_token_ids"] is True
    assert "allowed_token_ids" not in payload
    assert "prompt_logprobs" not in payload


def test_upstream_rejects_multitoken_target(tmp_path: Path) -> None:
    class Tokenizer:
        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            if add_special_tokens:
                return [1]
            return {" YES": [2, 4], " NO": [3]}[text]

    state = ProxyState(
        upstream="http://replica",
        served_model="grader",
        model_revision="revision-a",
        tokenizer_revision="tokenizer-a",
        tokenizer=Tokenizer(),
        cache_dir=None,
    )

    with pytest.raises(ValueError, match="exactly one token"):
        state.score(["prompt"], [" YES", " NO"])
