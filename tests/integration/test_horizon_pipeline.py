from __future__ import annotations

import json
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import write_json_atomic, write_jsonl_atomic
from dynamic_rubric.config import load_config
from dynamic_rubric.hashing import sha256_file
from dynamic_rubric.horizon.observations import build_horizon_observations
from dynamic_rubric.horizon.orchestration import analyze_horizon_observations
from dynamic_rubric.horizon.pools import (
    PoolSpec,
    generate_pool_rows,
    validate_horizon_pool_inventory,
)
from dynamic_rubric.providers.fake import FakeGenerator


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    ("domain", "config_name"),
    (("medicine", "horizon_medicine.yaml"), ("science", "horizon_science.yaml")),
)
def test_two_seed_four_checkpoint_offline_horizon_pipeline(
    tmp_path: Path, domain: str, config_name: str
) -> None:
    config = load_config(PROJECT_ROOT / "configs" / config_name)
    model = "Qwen/Qwen3-1.7B"
    prompt = {
        "prompt_id": f"{domain}:fixture",
        "messages": [{"role": "user", "content": "Synthetic question"}],
    }
    rows = []
    provider = FakeGenerator(model)
    for family in ("fixed_control", "sham_control"):
        rows.extend(
            generate_pool_rows(
                provider,
                [prompt],
                suite_id="rar-horizon-offline-v1",
                domain=domain,
                spec=PoolSpec(family, 8, 0, None),
                checkpoint_hash="initial",
                model=model,
                model_revision="policy-revision",
                tokenizer_revision="tokenizer-revision",
            )
        )
    for seed in (11, 29):
        for checkpoint in (1, 2, 3):
            rows.extend(
                generate_pool_rows(
                    provider,
                    [prompt],
                    suite_id="rar-horizon-offline-v1",
                    domain=domain,
                    spec=PoolSpec("pool_a", 8, checkpoint, seed),
                    checkpoint_hash=f"seed-{seed}-step-{checkpoint}",
                    model=model,
                    model_revision="policy-revision",
                    tokenizer_revision="tokenizer-revision",
                )
            )
        for checkpoint in (0, 1, 2, 3):
            rows.extend(
                generate_pool_rows(
                    provider,
                    [prompt],
                    suite_id="rar-horizon-offline-v1",
                    domain=domain,
                    spec=PoolSpec("pool_b", 16, checkpoint, seed),
                    checkpoint_hash=f"seed-{seed}-step-{checkpoint}",
                    model=model,
                    model_revision="policy-revision",
                    tokenizer_revision="tokenizer-revision",
                )
            )
    inventory = validate_horizon_pool_inventory(
        rows,
        expected_counts={
            "fixed_control": 8,
            "sham_control": 8,
            "pool_a": 8,
            "pool_b": 16,
        },
        expected_prompt_ids=[prompt["prompt_id"]],
        training_seeds=[11, 29],
        checkpoint_steps=[0, 1, 2, 3],
    )
    assert inventory["valid"]
    assert inventory["full_grid_validated"]
    assert inventory["unique_response_ids"] == len(rows)

    summaries = []
    for seed in (11, 29):
        for checkpoint in (0.0, 0.2, 0.4, 0.6):
            late = checkpoint >= 0.4
            summary_dir = tmp_path / domain / f"seed-{seed}" / f"checkpoint-{checkpoint}"
            summary_dir.mkdir(parents=True)
            summary_path = summary_dir / "prompt_summary.jsonl"
            summary_rows = []
            for prompt_index in range(2):
                summary_rows.append(
                    {
                        "schema_version": 1,
                        "seed_id": str(seed),
                        "prompt_id": f"{domain}:p{prompt_index}",
                        "checkpoint": checkpoint,
                        "response_count": 16,
                        "variants": {
                            "r0": {"exact_zar": late},
                            "current": {"exact_zar": False},
                            "control": {"exact_zar": late},
                        },
                    }
                )
            write_jsonl_atomic(summary_path, summary_rows)
            write_json_atomic(
                summary_dir / "score_seal.json",
                {
                    "schema_version": 1,
                    "artifact_type": "horizon_score_seal",
                    "config_hash": config.config_hash,
                    "seed_id": str(seed),
                    "checkpoint": checkpoint,
                    "prompt_count": 2,
                    "outputs": {"prompt_summary": sha256_file(summary_path)},
                },
            )
            summaries.append(summary_path)
    observation_path = tmp_path / f"{domain}-observations.jsonl"
    build_horizon_observations(
        summaries,
        output_path=observation_path,
        expected_seed_ids=["11", "29"],
        expected_prompt_ids=[f"{domain}:p0", f"{domain}:p1"],
        expected_checkpoints=[0.0, 0.2, 0.4, 0.6],
        expected_config_hash=config.config_hash,
    )
    output_path = tmp_path / domain / "horizon_report.json"
    report = analyze_horizon_observations(
        config, observation_path, output_path, iterations=100
    )
    assert report["horizon_decision"]["status"] == "observed"
    assert report["horizon_decision"]["t_star"] == 0.4
    assert report["horizon_decision"]["early_equivalence_checkpoint"] == 0.2
    assert report["claim_scope"] == "empirical_discriminability_only"
    assert json.loads(output_path.read_text()) == report
    assert (output_path.parent / "horizon_decision.json").is_file()
