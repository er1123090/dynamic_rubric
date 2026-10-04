from __future__ import annotations

from threading import Lock
import time

import pytest
from dynamic_rubric.rubrics.replay import ReplayMode
from dynamic_rubric.minimum_staleness import (
    MinimumExperimentError,
    TargetScoreClient,
    _balanced_routes,
    _candidate_id,
    _minimum_replay_stage,
    _minimum_score_stage,
    score_bon,
)


def _chunk(size: int, label: str) -> list[tuple[tuple[str, str], str]]:
    return [((label, str(index)), label * size) for index in range(2)]


def test_balanced_routes_are_deterministic_and_capacity_weighted() -> None:
    chunks = [
        _chunk(100, "a"),
        _chunk(90, "b"),
        _chunk(80, "c"),
        _chunk(70, "d"),
        _chunk(60, "e"),
        _chunk(50, "f"),
        _chunk(40, "g"),
        _chunk(30, "h"),
        _chunk(20, "i"),
    ]
    routes = _balanced_routes(chunks, (5, 2, 2))
    assert routes == _balanced_routes(chunks, (5, 2, 2))
    assert set(routes) == {0, 1, 2}


def test_balanced_routes_reject_invalid_weights() -> None:
    with pytest.raises(ValueError, match="positive integers"):
        _balanced_routes([], (1, 0))


def test_target_score_rejects_empty_targets_before_live_setup(tmp_path) -> None:
    with pytest.raises(MinimumExperimentError, match="cannot be empty"):
        score_bon(tmp_path, "http://unused", target_groups=set())


def test_target_score_rejects_progress_path_before_live_setup(tmp_path) -> None:
    with pytest.raises(MinimumExperimentError, match="basename"):
        score_bon(tmp_path, "http://unused", progress_filename="../progress.json")


def test_minimum_prev_mode_uses_isolated_replay_and_score_stages() -> None:
    assert _minimum_replay_stage(ReplayMode.DYNAMIC_PREV_BUDGETED) == (
        "replay-dynamic-prev-minimum"
    )
    assert _minimum_score_stage(ReplayMode.DYNAMIC_PREV_BUDGETED) == ("score-proxy-prev-minimum")
    assert _minimum_replay_stage(ReplayMode.DYNAMIC_FIXED_BUDGETED) == ("replay-dynamic-minimum")
    assert _minimum_score_stage(ReplayMode.DYNAMIC_FIXED_BUDGETED) == ("score-proxy-minimum")


def test_score_keeps_only_one_in_flight_request_per_upstream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = TargetScoreClient("http://unused", batch_size=1, workers=12)
    client._routing_weights = (1, 1)
    lock = Lock()
    active = {0: 0, 1: 0}
    maximum = {0: 0, 1: 0}

    def fake_post(prompts: list[str], upstream_index: int) -> list[dict[str, float]]:
        with lock:
            active[upstream_index] += 1
            maximum[upstream_index] = max(maximum[upstream_index], active[upstream_index])
        time.sleep(0.005)
        with lock:
            active[upstream_index] -= 1
        return [
            {
                "yes_logprob": -0.1,
                "no_logprob": -1.0,
                "probability_yes": 0.7,
            }
            for _ in prompts
        ]

    monkeypatch.setattr(client, "_post", fake_post)
    tasks = [((f"criterion-{index}", f"response-{index}"), "x") for index in range(20)]

    assert len(client.score(tasks)) == len(tasks)
    assert maximum == {0: 1, 1: 1}


def test_candidate_ids_are_namespaced_by_replay_mode() -> None:
    fixed = _candidate_id("p", 3, 0, "criterion text", ReplayMode.DYNAMIC_FIXED_BUDGETED)
    previous = _candidate_id("p", 3, 0, "criterion text", ReplayMode.DYNAMIC_PREV_BUDGETED)
    assert fixed.startswith("dyn-fixed-")
    assert previous.startswith("dyn-prev-")
    assert fixed != previous
