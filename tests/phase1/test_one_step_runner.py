from __future__ import annotations

import json
import sys
from types import SimpleNamespace
from dataclasses import replace
from pathlib import Path

import pytest

from dynamic_rubric.phase1.config import load_phase1_config
from dynamic_rubric.phase1.one_step import (
    CANARY_RUN_PREFIX,
    OneStepCanaryError,
    build_one_step_environment,
    select_canary_train_rows,
    validate_one_step_config,
    write_canary_parquets,
)


ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "configs/phase1"


def _medicine():
    return load_phase1_config(CONFIGS / "medicine_online_rubrics.yaml")


def test_one_step_accepts_only_medicine_online() -> None:
    validate_one_step_config(_medicine())
    science = load_phase1_config(CONFIGS / "science_online_rubrics.yaml")
    with pytest.raises(OneStepCanaryError, match="medicine"):
        validate_one_step_config(science)


def test_seed_11_selection_is_exact_deterministic_and_unique() -> None:
    rows = [{"prompt_id": f"p-{index}"} for index in range(1500)]
    first = select_canary_train_rows(rows)
    second = select_canary_train_rows(rows)
    assert len(first) == 96
    assert [index for index, _ in first] == [index for index, _ in second]
    assert len({row["prompt_id"] for _, row in first}) == 96


def test_canary_parquet_manifest_retains_source_provenance(
    tmp_path: Path, monkeypatch
) -> None:
    class FakeDataset:
        def __init__(self, rows):
            self.rows = rows

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def to_parquet(self, path):
            Path(path).write_text(json.dumps(self.rows))

    monkeypatch.setitem(sys.modules, "datasets", SimpleNamespace(Dataset=FakeDataset))
    config = _medicine()
    run_root = tmp_path / f"{CANARY_RUN_PREFIX}-pytest"
    run_root.mkdir()
    train_path, validation_path, manifest_path = write_canary_parquets(
        config, repo_root=ROOT, run_root=run_root
    )
    manifest = json.loads(manifest_path.read_text())
    assert train_path.is_file()
    assert validation_path.is_file()
    assert manifest["pool"] == "train_batch"
    assert manifest["selection"]["selected_row_count"] == 96
    assert len(manifest["selected_rows"]) == 96
    assert all("source_index" in row and "prompt_hash" in row for row in manifest["selected_rows"])
    assert manifest["validation"]["enabled"] is False
    assert manifest["validation"]["row_count"] == 1


def test_environment_pins_one_step_gpu_and_endpoints(tmp_path: Path, monkeypatch) -> None:
    model = tmp_path / "model"
    model.mkdir()
    launch_spec = tmp_path / "control.json"
    launch_spec.write_text("{}")
    config = _medicine()
    raw = dict(config.raw)
    raw["models"] = {**config.models, "policy": {**config.models["policy"], "local_snapshot": str(model)}}
    config = replace(config, raw=raw)
    monkeypatch.setenv("PHASE1_GPT_OSS_BASE_URL", "http://gpt-oss")
    monkeypatch.setenv("PHASE1_QWEN32B_BASE_URL", "http://qwen32b")
    monkeypatch.setenv("ONLINE_CONTROL_URL", "http://control")
    monkeypatch.setenv("ONLINE_CONTROL_CHECKPOINT_HASH", "a" * 64)
    monkeypatch.setenv("ONLINE_CONTROL_LAUNCH_SPEC", str(launch_spec))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("dynamic_rubric.phase1.one_step._directory_tree_hash", lambda _: "a" * 64)

    class Control:
        def __init__(self, *args, **kwargs):
            pass

        def preflight(self):
            return {"checkpoint_hash": "a" * 64}

    monkeypatch.setattr("dynamic_rubric.phase1.one_step.VLLMPolicyGenerator", Control)
    environment = build_one_step_environment(
        config,
        repo_root=ROOT,
        run_root=tmp_path / f"{CANARY_RUN_PREFIX}-pytest",
        train_path=tmp_path / "train.parquet",
        validation_path=tmp_path / "validation.parquet",
    )
    assert "OPENAI_API_KEY" not in environment
    assert environment["EXPECTED_UPDATES"] == "1"
    assert environment["TRAIN_BATCH_SIZE"] == "96"
    assert environment["ROLLOUT_N"] == "16"
    assert environment["ELICITATION_PAIRS"] == "8"
    assert environment["CUDA_VISIBLE_DEVICES"] == "0"
    assert environment["N_GPUS_PER_NODE"] == "1"
    assert environment["ONLINE_EXTRACTOR_BASE_URL"] == "http://gpt-oss"
    assert environment["ONLINE_GRADER_BASE_URL"] == "http://qwen32b"
    assert environment["ONLINE_EXTRACTOR_CONCURRENCY"] == "16"
    assert environment["ONLINE_GRADER_CONCURRENCY"] == "32"
    assert environment["PHASE1_VLLM_TIMEOUT_SECONDS"] == "600"
    assert environment["PHASE1_VLLM_MAX_RETRIES"] == "4"


def test_shell_launcher_hard_codes_one_step_contract() -> None:
    launcher = (ROOT / "scripts/phase1/run_online_one_step.sh").read_text()
    assert "-m verl.trainer.main_ppo" in launcher
    assert "trainer.total_training_steps=1" in launcher
    assert "trainer.save_freq=1" in launcher
    assert "actor_rollout_ref.rollout.n=16" in launcher
    assert "+actor_rollout_ref.model.override_config.attn_implementation=\"${ATTN_IMPLEMENTATION:-sdpa}\"" in launcher
    assert '[[ "${CUDA_VISIBLE_DEVICES:-}" == "0" ]]' in launcher
    assert "OPENAI_API_KEY" not in launcher
    assert "scripts/run_online_grpo.sh" not in launcher
