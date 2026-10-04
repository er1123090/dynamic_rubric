from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from dynamic_rubric.artifacts import read_json, read_jsonl, write_jsonl_atomic
from dynamic_rubric.config import config_from_mapping
from dynamic_rubric.seeds import SeedFamily, derive_seed, response_id
from dynamic_rubric.training.live_static import (
    build_training_environment,
    checkpoint_inventory,
    finalize_probe_exports,
    publish_reference_export,
)
from dynamic_rubric.training.verl_dataset import build_verl_rows


def _context(root: Path) -> SimpleNamespace:
    config = config_from_mapping(
        {
            "experiment": "live-static-test",
            "paths": {"public_data": "data/public", "artifacts": "artifacts"},
            "models": {
                "policy": {"model": "Qwen/policy"},
                "proxy_grader": {
                    "model": "Qwen/grader",
                    "revision": "grader-revision",
                    "tokenizer_revision": "tokenizer-revision",
                },
            },
            "training": {
                "max_steps": 1,
                "train_batch_size": 48,
                "rollout_n": 8,
                "max_prompt_length": 4096,
                "max_response_length": 1536,
                "checkpoint_steps": [0, 1],
                "reward_source": "static_r0_only",
            },
        },
        stage="train-static",
    )
    run_root = root / "artifacts" / "runs" / "pilot"
    stage = run_root / "train-static"
    return SimpleNamespace(
        root=root,
        public_root=root / "data" / "public",
        run_root=run_root,
        run_id="pilot",
        raw=config.raw,
        config=config,
        stage_root=lambda: stage,
    )


def _probe_rows(run_id: str, prompt_id: str, step: int) -> list[dict[str, object]]:
    rows = []
    for family in (
        SeedFamily.TRAJECTORY_DISCOVERY,
        SeedFamily.TRAJECTORY_VALIDATION,
    ):
        for sample_index in range(4):
            seed = derive_seed(run_id, family, prompt_id, step, sample_index)
            rows.append(
                {
                    "step": step,
                    "policy_step": step,
                    "prompt_id": prompt_id,
                    "family": family.value,
                    "sample_index": sample_index,
                    "logical_seed": seed,
                    "response_id": response_id(
                        run_id, family, prompt_id, step, sample_index
                    ),
                    "output": f"response-{family.value}-{sample_index}",
                    "static_reward": 0.5,
                    "criterion_probabilities": [0.5] * 8,
                }
            )
    return rows


def test_live_training_environment_pins_two_gpu_topology(
    tmp_path: Path, monkeypatch
) -> None:
    context = _context(tmp_path)
    monkeypatch.setenv("DYNAMIC_RUBRIC_VLLM_URL", "http://127.0.0.1:8102")
    environment = build_training_environment(
        context,
        tmp_path / "train.parquet",
        tmp_path / "probe.parquet",
        tmp_path / "static_rubrics.jsonl",
        tmp_path / "run",
    )
    assert environment["POLICY_GPU"] == "0"
    assert environment["ROLLOUT_GPU_MEMORY"] == "0.42"
    assert environment["DYNAMIC_RUBRIC_GRADER_MODEL"] == "Qwen/grader"
    assert environment["TOTAL_STEPS"] == "1"
    assert environment["ACTOR_MAX_TOKEN_LEN"] == "8192"
    assert environment["VAL_BEFORE_TRAIN"] == "False"
    assert environment["TEST_FREQ"] == "1"


def test_verl_rows_delegate_training_replica_index_to_agent_loop() -> None:
    prompt = {"prompt_id": "train-1", "messages": [{"role": "user", "content": "x"}]}
    probe = {**prompt, "prompt_id": "probe-1", "split": "pilot_probe"}

    training, validation = build_verl_rows("pilot", [prompt], [probe])

    assert [row["seed_sample_index"] for row in training] == [-1]
    assert sorted(row["seed_sample_index"] for row in validation) == [
        0,
        0,
        1,
        1,
        2,
        2,
        3,
        3,
    ]
    assert {row["seed_family"] for row in validation} == {
        SeedFamily.TRAJECTORY_DISCOVERY.value,
        SeedFamily.TRAJECTORY_VALIDATION.value,
    }


def test_finalize_probe_exports_splits_and_seals(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.public_root.mkdir(parents=True)
    write_jsonl_atomic(
        context.public_root / "pilot_probe.jsonl", [{"prompt_id": "dev", "messages": []}]
    )
    write_jsonl_atomic(
        context.public_root / "pilot_audit.jsonl", [{"prompt_id": "final", "messages": []}]
    )
    run_dir = context.stage_root() / "verl-run"
    write_jsonl_atomic(
        run_dir / "probes" / "1.jsonl",
        [*_probe_rows(context.run_id, "dev", 1), *_probe_rows(context.run_id, "final", 1)],
    )

    result = finalize_probe_exports(context, run_dir)

    assert result["development_records"] == 8
    assert result["sealed_final_records"] == 8
    development = read_jsonl(context.stage_root() / "trajectory_development.jsonl")
    sealed = read_jsonl(
        context.run_root / "trajectory" / "final_sealed" / "responses.jsonl"
    )
    assert {row["split"] for row in development} == {"pilot_probe"}
    assert {row["split"] for row in sealed} == {"pilot_audit"}
    assert all(row["timing"] == "after_optimizer_update" for row in development + sealed)
    assert all(row["checkpoint_hash"] for row in development + sealed)


def test_checkpoint_inventory_requires_exact_focal_steps(tmp_path: Path) -> None:
    context = _context(tmp_path)
    run_dir = context.stage_root() / "verl-run"
    for step in (0, 1):
        checkpoint = run_dir / "checkpoints" / f"global_step_{step}" / "actor"
        checkpoint.mkdir(parents=True)
        (checkpoint / "state.pt").write_bytes(f"state-{step}".encode())

    result = checkpoint_inventory(context, run_dir)

    assert [row["step"] for row in result] == [0, 1]
    assert all(row["content_hash"] for row in result)
    assert read_json(context.stage_root() / "checkpoints.json") == result


def test_publish_reference_export_requires_exact_pi0_families(tmp_path: Path) -> None:
    context = _context(tmp_path)
    context.public_root.mkdir(parents=True)
    write_jsonl_atomic(
        context.public_root / "pilot_probe.jsonl", [{"prompt_id": "dev", "messages": []}]
    )
    write_jsonl_atomic(
        context.public_root / "pilot_audit.jsonl", [{"prompt_id": "final", "messages": []}]
    )
    rows = []
    for prompt_id, split in (("dev", "pilot_probe"), ("final", "pilot_audit")):
        for family, count in (
            (SeedFamily.REFERENCE_DISCOVERY, 8),
            (SeedFamily.REFERENCE_VALIDATION, 4),
        ):
            for sample_index in range(count):
                seed = derive_seed(
                    context.run_id, family, prompt_id, 0, sample_index
                )
                rows.append(
                    {
                        "prompt_id": prompt_id,
                        "split": split,
                        "policy_step": 0,
                        "family": family.value,
                        "sample_index": sample_index,
                        "seed": seed,
                        "response_id": response_id(
                            context.run_id, family, prompt_id, 0, sample_index
                        ),
                        "response_text": "reference",
                    }
                )
    source = tmp_path / "references.jsonl"
    write_jsonl_atomic(source, rows)

    exported = publish_reference_export(context, source)

    assert len(exported) == 24
    assert read_jsonl(context.stage_root() / "reference_responses.jsonl") == rows
