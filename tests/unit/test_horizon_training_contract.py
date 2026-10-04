from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from dynamic_rubric.config import ConfigError, config_from_mapping, load_config
from dynamic_rubric.horizon.checkpoints import ObservedCheckpoint, map_checkpoint_schedule
from dynamic_rubric.horizon.orchestration import (
    HorizonOrchestrationError,
    estimate_horizon_cost,
    validate_horizon_launch_environment,
)
from dynamic_rubric.training.verl_dataset import build_rar_verl_rows


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "config_name",
    ("horizon_medicine.yaml", "horizon_science.yaml"),
)
def test_horizon_configs_use_final_only_audit(config_name: str) -> None:
    config = load_config(ROOT / "configs" / config_name)
    assert config.horizon is not None
    assert (config.horizon.development_count, config.horizon.final_count) == (0, 100)
    estimate = estimate_horizon_cost(config)
    assert estimate["prompts"] == {"development": 0, "final": 100, "audited_total": 100}


def _prompt(prompt_id: str) -> dict:
    return {
        "prompt_id": prompt_id,
        "domain": "medicine",
        "messages": [{"role": "user", "content": "question"}],
    }


def test_rar_verl_rows_use_static_weighted_reward_only() -> None:
    train, development = build_rar_verl_rows("run", [_prompt("train")], [_prompt("dev")])
    assert train[0]["data_source"] == "rar_static_r0"
    assert train[0]["reward_model"] == {
        "style": "hard_binary_weighted_rational_v1",
        "ground_truth": "rar_static_r0_only",
    }
    assert len(development) == 16
    assert {row["seed_sample_index"] for row in development} == set(range(8))


def test_checkpoint_mapping_uses_actual_unique_global_steps() -> None:
    observed = [
        ObservedCheckpoint(step, step / 10, step * 96, f"hash-{step}")
        for step in range(31)
    ]
    result = map_checkpoint_schedule((0.0, 0.2, 0.4, 1.0, 2.0, 3.0), observed)
    assert [item.global_step for item in result] == [0, 2, 4, 10, 20, 30]


def test_horizon_config_rejects_training_hyperparameter_drift() -> None:
    data = yaml.safe_load((ROOT / "configs" / "horizon_medicine.yaml").read_text())
    data["training"]["learning_rate"] = 1e-5
    with pytest.raises(ConfigError, match="drifted"):
        config_from_mapping(data)


def test_launcher_environment_is_checked_against_typed_config() -> None:
    config = load_config(ROOT / "configs" / "horizon_medicine.yaml")
    environment = {
        "DOMAIN": "medicine",
        "TRAINING_SEED": "11",
        "TOTAL_STEPS": "48",
        "TRAIN_BATCH_SIZE": "96",
        "ROLLOUT_N": "16",
        "MAX_RESPONSE_LENGTH": "3584",
        "LEARNING_RATE": "5e-6",
        "WARMUP_RATIO": "0.1",
        "KL_COEFFICIENT": "0.01",
        "ROLLOUT_TEMPERATURE": "1.0",
        "CHECKPOINT_STEPS": "[0,3,6,9,13,16,24,32,40,48]",
    }
    assert validate_horizon_launch_environment(config, environment)["valid"]
    environment["ROLLOUT_N"] = "8"
    with pytest.raises(HorizonOrchestrationError, match="drifted"):
        validate_horizon_launch_environment(config, environment)


def test_medicine_eval300_config_disables_sham() -> None:
    config = load_config(ROOT / "configs" / "horizon_medicine_eval300_no_sham.yaml")
    assert config.horizon is not None
    assert (config.horizon.development_count, config.horizon.final_count) == (0, 300)
    assert config.horizon.sham_control_count == 0
    estimate = estimate_horizon_cost(config)
    assert estimate["extractor_calls"]["sham_pairs"] == 0
