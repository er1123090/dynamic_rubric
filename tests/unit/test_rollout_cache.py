from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.training.rollout_cache import (
    ImmutableRolloutCache,
    RolloutCacheError,
)


def test_rollout_cache_is_durable_and_first_writer_wins(tmp_path: Path) -> None:
    cache = ImmutableRolloutCache(tmp_path)
    identity = {"response_id": "resp_training_a", "seed": 7}
    first = {"response_ids": [1, 2], "response_logprobs": [-0.1, -0.2]}
    second = {"response_ids": [9], "response_logprobs": [-9.0]}

    assert cache.publish(identity, first) == first
    assert ImmutableRolloutCache(tmp_path).read(identity) == first
    assert cache.publish(identity, second) == first


def test_rollout_cache_key_binds_sampling_and_policy_identity(tmp_path: Path) -> None:
    cache = ImmutableRolloutCache(tmp_path)
    first = {"response_id": "r", "seed": 1, "policy_model_path": "/model/a"}
    second = {"response_id": "r", "seed": 2, "policy_model_path": "/model/a"}
    third = {"response_id": "r", "seed": 1, "policy_model_path": "/model/b"}

    assert len({cache.key(first), cache.key(second), cache.key(third)}) == 3


def test_rollout_cache_rejects_corruption(tmp_path: Path) -> None:
    cache = ImmutableRolloutCache(tmp_path)
    identity = {"response_id": "r", "seed": 1}
    cache.publish(identity, {"response_ids": [1], "response_logprobs": [-0.1]})
    path = next(tmp_path.rglob("*.json"))
    envelope = json.loads(path.read_text())
    envelope["output"]["response_ids"] = [2]
    path.write_text(json.dumps(envelope))

    with pytest.raises(RolloutCacheError):
        cache.read(identity)


@pytest.mark.parametrize(
    "output",
    [
        {"response_ids": [], "response_logprobs": None},
        {"response_ids": [1], "response_logprobs": None},
        {"response_ids": [1, 2], "response_logprobs": [-0.1]},
    ],
)
def test_rollout_cache_rejects_incomplete_generation(
    tmp_path: Path, output: dict[str, object]
) -> None:
    cache = ImmutableRolloutCache(tmp_path)

    with pytest.raises(RolloutCacheError):
        cache.publish({"response_id": "interrupted", "seed": 3}, output)

    assert not list(tmp_path.rglob("*.json"))
