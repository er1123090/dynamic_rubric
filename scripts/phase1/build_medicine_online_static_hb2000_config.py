#!/usr/bin/env python3
"""Build the shared 2,000-prompt HealthBench config for Online and Static policies."""

from __future__ import annotations

import json
from pathlib import Path

import yaml


REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "configs/evaluation"
OUTPUT = CONFIG_DIR / "medicine_online60_static14_healthbench2000_20261001.yaml"

ONLINE_1_48 = CONFIG_DIR / "medicine_dense_all48_healthbench500_20260926.yaml"
ONLINE_49_60 = CONFIG_DIR / "medicine_dense_steps49_60_healthbench500_20260930.yaml"
STATIC_3_42 = CONFIG_DIR / "medicine_static_matched_checkpoint_trajectory_hb500_20260926.yaml"

ONLINE_1_48_RUN = (
    REPO
    / "outputs/policy_eval/medicine_dense_all48_healthbench500_20260926"
    / "full-6b52dfbc30980b62"
)
ONLINE_49_60_RUN = (
    REPO
    / "outputs/policy_eval/medicine_dense_steps49_60_healthbench500_20260930"
    / "full-366ea6b189c27e23"
)
STATIC_3_42_RUN = (
    REPO
    / "outputs/policy_eval/medicine_static_matched_checkpoint_trajectory_hb500_20260926"
    / "full-3f9ab542f8a7f87d"
)
STATIC_TRAIN_RUN = (
    REPO
    / "outputs/medicine/static_r0_matched/seed-11"
    / "phase1-static-r0-medicine-qwen3-4b-matched-20260914"
)


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def relative_to_config(path: Path) -> str:
    return str(path.relative_to(REPO)).replace("configs/evaluation/", "../../")


def archive_identity(step: int) -> tuple[str, str]:
    path = (
        STATIC_TRAIN_RUN
        / "verl-run/checkpoint_archives"
        / f"static_global_step_{step}.json"
    )
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("state") != "verified" or not value.get("repo_id") or not value.get("revision"):
        raise RuntimeError(f"unusable Static archive manifest: {path}")
    return str(value["repo_id"]), str(value["revision"])


def main() -> None:
    online_1_48 = load_yaml(ONLINE_1_48)
    online_49_60 = load_yaml(ONLINE_49_60)
    static_3_42 = load_yaml(STATIC_3_42)

    static_models = {}
    for name, source in static_3_42["models"].items():
        spec = dict(source)
        step = int(spec["step"])
        local_path = Path(str(spec.get("local_path", "")))
        if not local_path.is_dir():
            spec.pop("local_path", None)
            repo_id, revision = archive_identity(step)
            spec["repo_id"] = repo_id
            spec["revision"] = revision
        static_models[name] = spec

    online_models = {
        **{name: dict(spec) for name, spec in online_1_48["models"].items()},
        **{name: dict(spec) for name, spec in online_49_60["models"].items()},
    }
    for spec in online_models.values():
        step = int(spec["step"])
        spec["role"] = "base" if step == 1 else "final" if step == 60 else "checkpoint"

    models = {**static_models, **online_models}
    if len(static_models) != 14 or len(online_models) != 60 or len(models) != 74:
        raise RuntimeError("unexpected checkpoint inventory")
    for method in ("static", "online"):
        selected = [spec for spec in models.values() if spec["method"] == method]
        if sum(spec["role"] == "base" for spec in selected) != 1:
            raise RuntimeError(f"{method} must have exactly one base")
        if sum(spec["role"] == "final" for spec in selected) != 1:
            raise RuntimeError(f"{method} must have exactly one final")

    dataset = dict(online_1_48["datasets"]["healthbench"])
    dataset["sample_count"] = 2000
    dataset["sample_seed"] = 11
    dataset["include_prompt_ids_from"] = relative_to_config(
        ONLINE_1_48_RUN / "prepared/prompts.jsonl"
    )

    reuse_roots = [ONLINE_1_48_RUN, ONLINE_49_60_RUN, STATIC_3_42_RUN]
    config = {
        "schema_version": 1,
        "output_root": "../../outputs/policy_eval/medicine_online60_static14_healthbench2000_20261001",
        "seed": 11,
        "bootstrap_replicates": 10_000,
        "healthbench_upstream": online_1_48["healthbench_upstream"],
        "datasets": {"healthbench": dataset},
        "models": models,
        "reuse_responses_from": [relative_to_config(path) for path in reuse_roots],
        "reuse_grades_from": [relative_to_config(path) for path in reuse_roots],
        "generation": {
            "base_url": "http://127.0.0.1:28141/v1",
            "timeout_seconds": 600,
            "max_retries": 4,
            "max_output_tokens": 4096,
            "workers": 16,
            "max_in_flight": 16,
            "seed": 11,
            "temperature": 0.0,
            "top_p": 1.0,
        },
        "grading": {
            "base_url": ["http://127.0.0.1:28145/v1", "http://127.0.0.1:28146/v1"],
            "served_model": "Qwen/Qwen3-32B",
            "revision": "9216db5781bf21249d130ec9da846c4624c16137",
            "timeout_seconds": 600,
            "max_retries": 4,
            "max_output_tokens": 1024,
            "workers": 128,
            "max_in_flight": 128,
            "parse_retries": 2,
        },
        "runtime": {
            "policy_gpu": 1,
            "policy_port": 28141,
            "vllm": str(REPO / ".venvs/judge/bin/vllm"),
            "max_model_len": 32768,
            "gpu_memory_utilization": 0.10,
            "startup_timeout_seconds": 900,
            "temporary_download_root": str(REPO / "artifacts/tmp/medicine_online60_static14_healthbench2000_20261001"),
            "delete_download_after_generation": True,
        },
    }
    OUTPUT.write_text(
        yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()
