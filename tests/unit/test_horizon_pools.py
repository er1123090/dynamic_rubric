from __future__ import annotations

import pytest

from dynamic_rubric.horizon.advantage import advantage_is_degenerate, grpo_scalar_advantages
from dynamic_rubric.horizon.pools import (
    PoolSpec,
    combine_pool_a_with_fixed_control,
    generate_pool_rows,
    pool_identity,
    validate_horizon_pool_inventory,
)
from dynamic_rubric.providers.fake import FakeGenerator


def test_pool_families_are_disjoint_and_reproducible() -> None:
    common = dict(
        suite_id="suite",
        domain="medicine",
        prompt_id="p1",
        sample_index=0,
        checkpoint_hash="abc",
        model_revision="rev",
        tokenizer_revision="rev",
        generation_config={"temperature": 1.0},
    )
    fixed = pool_identity(spec=PoolSpec("fixed_control", 8, 0, None), **common)
    sham = pool_identity(spec=PoolSpec("sham_control", 8, 0, None), **common)
    current = pool_identity(spec=PoolSpec("pool_b", 16, 0, 11), **common)
    assert len({fixed["response_id"], sham["response_id"], current["response_id"]}) == 3
    assert fixed == pool_identity(spec=PoolSpec("fixed_control", 8, 0, None), **common)


def test_generate_pool_rows_exact_inventory() -> None:
    rows = generate_pool_rows(
        FakeGenerator("fake/policy"),
        [{"prompt_id": "p1", "messages": [{"role": "user", "content": "hi"}]}],
        suite_id="suite",
        domain="science",
        spec=PoolSpec("pool_a", 8, 3, 29),
        checkpoint_hash="checkpoint",
        model="fake/policy",
        model_revision="rev",
        tokenizer_revision="rev",
    )
    assert len(rows) == 8
    assert {row["sample_index"] for row in rows} == set(range(8))


def test_combine_pool_a_with_fixed_control_preserves_response_provenance() -> None:
    provider = FakeGenerator("fake/policy")
    prompt = {"prompt_id": "p1", "messages": [{"role": "user", "content": "hi"}]}
    common = dict(
        prompts=[prompt],
        suite_id="suite",
        domain="medicine",
        model="fake/policy",
        model_revision="rev",
        tokenizer_revision="rev",
    )
    current = generate_pool_rows(
        provider,
        spec=PoolSpec("pool_a", 8, 3, 11),
        checkpoint_hash="current",
        **common,
    )
    fixed = generate_pool_rows(
        provider,
        spec=PoolSpec("fixed_control", 8, 0, None),
        checkpoint_hash="initial",
        **common,
    )
    combined = combine_pool_a_with_fixed_control(current, fixed)

    assert len(combined) == 16
    assert {row["sample_index"] for row in combined} == set(range(16))
    assert {row["pool_family"] for row in combined} == {"pool_a_combined"}
    assert {row["training_seed"] for row in combined} == {11}
    assert {row["policy_step"] for row in combined} == {3}
    assert [row["source_pool_family"] for row in combined[:8]] == ["pool_a"] * 8
    assert [row["source_pool_family"] for row in combined[8:]] == ["fixed_control"] * 8
    assert {row["response_id"] for row in combined} == {
        row["response_id"] for row in current + fixed
    }
    assert {row["checkpoint_hash"] for row in combined} == {"current", "initial"}


def test_pool_generation_concurrency_preserves_deterministic_row_order() -> None:
    prompts = [
        {"prompt_id": "p0", "messages": [{"role": "user", "content": "zero"}]},
        {"prompt_id": "p1", "messages": [{"role": "user", "content": "one"}]},
    ]
    common = dict(
        prompts=prompts,
        suite_id="suite",
        domain="medicine",
        spec=PoolSpec("fixed_control", 3, 0, None),
        checkpoint_hash="base",
        model="fake/policy",
        model_revision="revision",
        tokenizer_revision="tokenizer",
    )
    sequential = generate_pool_rows(FakeGenerator("fake/policy"), concurrency=1, **common)
    parallel = generate_pool_rows(FakeGenerator("fake/policy"), concurrency=3, **common)

    def stable(rows: list[dict]) -> list[dict]:
        return [
            {key: value for key, value in row.items() if key != "request_id"} for row in rows
        ]

    assert stable(parallel) == stable(sequential)


def test_pinned_grpo_advantage_semantics() -> None:
    assert advantage_is_degenerate([0.5] * 16)
    values = grpo_scalar_advantages([0.0, 1.0])
    assert values[0] < 0 < values[1]
    assert max(abs(value) for value in values) > 0.7


def test_full_inventory_rejects_a_missing_preregistered_group() -> None:
    provider = FakeGenerator("fake/policy")
    prompt = {"prompt_id": "p1", "messages": [{"role": "user", "content": "hi"}]}
    rows = []
    for family, count, step, seed in (
        ("fixed_control", 8, 0, None),
        ("sham_control", 8, 0, None),
        ("pool_b", 16, 0, 11),
        ("pool_b", 16, 1, 11),
    ):
        rows.extend(
            generate_pool_rows(
                provider,
                [prompt],
                suite_id="suite",
                domain="medicine",
                spec=PoolSpec(family, count, step, seed),
                checkpoint_hash=f"checkpoint-{step}",
                model="fake/policy",
                model_revision="rev",
                tokenizer_revision="rev",
            )
        )
    with pytest.raises(ValueError, match="complete preregistered grid"):
        validate_horizon_pool_inventory(
            rows,
            expected_counts={
                "fixed_control": 8,
                "sham_control": 8,
                "pool_a": 8,
                "pool_b": 16,
            },
            expected_prompt_ids=["p1"],
            training_seeds=[11],
            checkpoint_steps=[0, 1],
        )


def test_full_inventory_accepts_disabled_sham_family() -> None:
    provider = FakeGenerator("fake/policy")
    prompt = {"prompt_id": "p1", "messages": [{"role": "user", "content": "hi"}]}
    rows = []
    for family, count, step, seed in (
        ("fixed_control", 8, 0, None),
        ("pool_b", 16, 0, 11),
        ("pool_a", 8, 1, 11),
        ("pool_b", 16, 1, 11),
    ):
        rows.extend(
            generate_pool_rows(
                provider,
                [prompt],
                suite_id="suite",
                domain="medicine",
                spec=PoolSpec(family, count, step, seed),
                checkpoint_hash=f"checkpoint-{step}",
                model="fake/policy",
                model_revision="rev",
                tokenizer_revision="rev",
            )
        )

    result = validate_horizon_pool_inventory(
        rows,
        expected_counts={
            "fixed_control": 8,
            "sham_control": 0,
            "pool_a": 8,
            "pool_b": 16,
        },
        expected_prompt_ids=["p1"],
        training_seeds=[11],
        checkpoint_steps=[0, 1],
    )

    assert result["valid"] is True
    assert result["family_rows"]["sham_control"] == 0
