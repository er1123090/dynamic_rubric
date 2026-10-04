from __future__ import annotations

from dynamic_rubric.live_bon import (
    _prompt_is_assigned,
    generate_prompt_rows,
    generate_sample_rows,
)
from dynamic_rubric.providers.fake import FakeGenerator
from dynamic_rubric.seeds import SeedFamily, seed_namespace


def test_live_bon_rows_have_stable_global_ids_and_seed_namespace() -> None:
    rows = generate_prompt_rows(
        "run",
        30,
        {"prompt_id": "p", "messages": [{"role": "user", "content": "x"}]},
        2,
        3,
        FakeGenerator("pi_30"),
        2,
        64,
    )
    assert [row["sample_index"] for row in rows] == [0, 1, 2]
    assert [row["global_candidate_id"] for row in rows] == [
        30_000_000_006,
        30_000_000_007,
        30_000_000_008,
    ]
    assert len({row["response_id"] for row in rows}) == 3
    assert all(seed_namespace(row["seed"]) is SeedFamily.AUDIT_BON for row in rows)


def test_live_bon_extension_uses_absolute_sample_indices_and_target_stride() -> None:
    rows = generate_sample_rows(
        "run",
        30,
        {"prompt_id": "p", "messages": [{"role": "user", "content": "x"}]},
        2,
        64,
        67,
        1024,
        FakeGenerator("pi_30"),
        2,
        64,
    )
    assert [row["sample_index"] for row in rows] == [64, 65, 66]
    assert [row["global_candidate_id"] for row in rows] == [
        30_000_002_112,
        30_000_002_113,
        30_000_002_114,
    ]
    assert len({row["response_id"] for row in rows}) == 3
    assert all(seed_namespace(row["seed"]) is SeedFamily.AUDIT_BON for row in rows)


def test_prompt_shards_are_disjoint_and_complete() -> None:
    first = {index for index in range(96) if _prompt_is_assigned(index, 0, 2)}
    second = {index for index in range(96) if _prompt_is_assigned(index, 1, 2)}
    assert not first & second
    assert first | second == set(range(96))
    assert len(first) == len(second) == 48
