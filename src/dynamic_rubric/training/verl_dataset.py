"""Build immutable public veRL parquet inputs for static-only GRPO."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamic_rubric.artifacts import read_jsonl
from dynamic_rubric.seeds import SeedFamily


def _base_row(
    prompt: Mapping[str, Any],
    split: str,
    run_id: str,
    family: SeedFamily,
    *,
    data_source: str = "dynamic_rubric_static_r0",
    ability: str = "healthbench_consensus",
    reward_style: str = "static_r0_proxy",
    ground_truth: str = "static_r0_only",
) -> dict[str, Any]:
    prompt_id = str(prompt["prompt_id"])
    return {
        "data_source": data_source,
        "prompt": [dict(message) for message in prompt["messages"]],
        "ability": ability,
        "reward_model": {"style": reward_style, "ground_truth": ground_truth},
        "agent_name": "seeded_single_turn_agent",
        "run_id": run_id,
        "seed_family": family.value,
        "prompt_id": prompt_id,
        "seed_sample_index": -1,
        "extra_info": {"prompt_id": prompt_id, "split": split},
    }


def build_verl_rows(
    run_id: str,
    train_prompts: Sequence[Mapping[str, Any]],
    probe_prompts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train_rows = [
        _base_row(prompt, "pilot_train", run_id, SeedFamily.TRAINING)
        for prompt in train_prompts
    ]
    validation_rows: list[dict[str, Any]] = []
    for prompt in probe_prompts:
        split = str(prompt["split"])
        for family in (
            SeedFamily.TRAJECTORY_DISCOVERY,
            SeedFamily.TRAJECTORY_VALIDATION,
        ):
            for sample_index in range(4):
                row = _base_row(prompt, split, run_id, family)
                row["seed_sample_index"] = sample_index
                row["extra_info"].update(
                    {
                        "family": family.value,
                        "sample_index": sample_index,
                    }
                )
                validation_rows.append(row)
    return train_rows, validation_rows


def write_verl_parquets(
    public_root: Path,
    output_root: Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Write train and probe parquets and return their absolute paths."""

    from datasets import Dataset  # pyright: ignore[reportMissingImports]

    train_prompts = read_jsonl(public_root / "pilot_train.jsonl")
    probes = []
    for split in ("pilot_probe", "pilot_audit"):
        for row in read_jsonl(public_root / f"{split}.jsonl"):
            probes.append({**row, "split": split})
    train_rows, validation_rows = build_verl_rows(run_id, train_prompts, probes)
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.parquet"
    validation_path = output_root / "probes.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(validation_rows).to_parquet(str(validation_path))
    return train_path.resolve(), validation_path.resolve()


def build_rar_verl_rows(
    run_id: str,
    train_prompts: Sequence[Mapping[str, Any]],
    development_prompts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    def row(prompt: Mapping[str, Any], split: str, family: SeedFamily) -> dict[str, Any]:
        return _base_row(
            prompt,
            split,
            run_id,
            family,
            data_source="rar_static_r0",
            ability=f"rar_{prompt['domain']}",
            reward_style="hard_binary_weighted_rational_v1",
            ground_truth="rar_static_r0_only",
        )

    train_rows = [row(prompt, "train", SeedFamily.TRAINING) for prompt in train_prompts]
    validation_rows: list[dict[str, Any]] = []
    for prompt in development_prompts:
        for family in (SeedFamily.TRAJECTORY_DISCOVERY, SeedFamily.TRAJECTORY_VALIDATION):
            for sample_index in range(8):
                value = row(prompt, "development", family)
                value["seed_sample_index"] = sample_index
                value["extra_info"].update(
                    {"family": family.value, "sample_index": sample_index}
                )
                validation_rows.append(value)
    return train_rows, validation_rows


def write_rar_verl_parquets(
    public_root: Path, output_root: Path, run_id: str
) -> tuple[Path, Path]:
    from datasets import Dataset  # pyright: ignore[reportMissingImports]

    train_rows, validation_rows = build_rar_verl_rows(
        run_id,
        read_jsonl(public_root / "train.jsonl"),
        read_jsonl(public_root / "development.jsonl"),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train.parquet"
    validation_path = output_root / "development.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(validation_rows).to_parquet(str(validation_path))
    return train_path.resolve(), validation_path.resolve()


def _online_row_identity(
    prompt: Mapping[str, Any],
    *,
    split: str,
    source_index: int,
) -> tuple[str, str]:
    """Return stable source and occurrence IDs without changing static datasets."""

    prompt_id = str(prompt["prompt_id"])
    source_row_id = str(prompt.get("source_row_id", prompt_id))
    payload = json.dumps(
        {
            "prompt_id": prompt_id,
            "source_row_id": source_row_id,
            "source_index": source_index,
            "split": split,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    occurrence_hash = hashlib.sha256(payload).hexdigest()[:20]
    return source_row_id, f"{split}:{source_index}:{occurrence_hash}"


def build_online_rar_verl_rows(
    run_id: str,
    train_prompts: Sequence[Mapping[str, Any]],
    development_prompts: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build online-only rows with deterministic occurrence identity.

    The existing static builders are deliberately untouched so their parquet
    bytes and reward source semantics remain stable.
    """

    def online_row(
        prompt: Mapping[str, Any],
        *,
        split: str,
        family: SeedFamily,
        source_index: int,
        sample_index: int,
    ) -> dict[str, Any]:
        value = _base_row(
            prompt,
            split,
            run_id,
            family,
            data_source="rar_online_rubrics_v1",
            ability=f"rar_{prompt['domain']}",
            reward_style="paper_binary_weighted_rational_v1",
            ground_truth="online_step_local",
        )
        source_row_id, prompt_occurrence_id = _online_row_identity(
            prompt, split=split, source_index=source_index
        )
        value["seed_sample_index"] = sample_index
        value["source_row_id"] = source_row_id
        offline_criteria = [
            {
                "criterion_id": str(item["criterion_id"]),
                "text": str(item["criterion"]),
                "weight": int(item["weight_units"]),
            }
            for item in prompt.get("r0", {}).get("criteria", [])
        ]
        if not offline_criteria:
            raise ValueError(f"online prompt has no offline R0 criteria: {prompt['prompt_id']}")
        value["prompt_occurrence_id"] = prompt_occurrence_id
        value["offline_criteria"] = offline_criteria
        value["extra_info"].update(
            {
                "family": family.value,
                "sample_index": sample_index,
                "source_row_id": source_row_id,
                "prompt_occurrence_id": prompt_occurrence_id,
                "prompt_messages": [dict(message) for message in prompt["messages"]],
                "offline_criteria": offline_criteria,
            }
        )
        return value

    train_rows = [
        online_row(
            prompt,
            split="train",
            family=SeedFamily.TRAINING,
            source_index=index,
            sample_index=-1,
        )
        for index, prompt in enumerate(train_prompts)
    ]
    validation_rows: list[dict[str, Any]] = []
    for source_index, prompt in enumerate(development_prompts):
        for family in (SeedFamily.TRAJECTORY_DISCOVERY, SeedFamily.TRAJECTORY_VALIDATION):
            for sample_index in range(8):
                validation_rows.append(
                    online_row(
                        prompt,
                        split="development",
                        family=family,
                        source_index=source_index,
                        sample_index=sample_index,
                    )
                )
    return train_rows, validation_rows


def write_online_rar_verl_parquets(
    public_root: Path, output_root: Path, run_id: str
) -> tuple[Path, Path]:
    from datasets import Dataset  # pyright: ignore[reportMissingImports]

    train_rows, validation_rows = build_online_rar_verl_rows(
        run_id,
        read_jsonl(public_root / "train.jsonl"),
        read_jsonl(public_root / "development.jsonl"),
    )
    output_root.mkdir(parents=True, exist_ok=True)
    train_path = output_root / "train-online.parquet"
    validation_path = output_root / "development-online.parquet"
    Dataset.from_list(train_rows).to_parquet(str(train_path))
    Dataset.from_list(validation_rows).to_parquet(str(validation_path))
    return train_path.resolve(), validation_path.resolve()
