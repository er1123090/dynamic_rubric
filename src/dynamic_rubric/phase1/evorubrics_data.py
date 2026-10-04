"""Immutable RaR-to-EvoRubrics dataset conversion with source provenance."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import artifact_record, read_json, write_json_atomic
from ..hashing import sha256_file, sha256_json
from .config import Phase1Config


class EvoRubricsDataError(ValueError):
    """Raised when a source split cannot satisfy the EvoRubrics data contract."""


@dataclass(frozen=True, slots=True)
class PreparedEvoRubricsData:
    train_path: Path
    heldout_path: Path
    manifest_path: Path
    manifest: Mapping[str, Any]


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise EvoRubricsDataError(f"missing RaR source split: {path}")
    records: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise EvoRubricsDataError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(value, Mapping):
                raise EvoRubricsDataError(
                    f"source row must be an object at {path}:{line_number}"
                )
            records.append(value)
    return records


def _required_text(record: Mapping[str, Any], key: str, *, row_number: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value.strip():
        raise EvoRubricsDataError(f"row {row_number} has invalid {key}")
    return value


def _validated_messages(record: Mapping[str, Any], *, row_number: int) -> list[dict[str, str]]:
    raw_messages = record.get("messages")
    if not isinstance(raw_messages, Sequence) or isinstance(raw_messages, (str, bytes)):
        raise EvoRubricsDataError(f"row {row_number} messages must be a sequence")
    messages: list[dict[str, str]] = []
    for message_index, message in enumerate(raw_messages):
        if not isinstance(message, Mapping):
            raise EvoRubricsDataError(
                f"row {row_number} message {message_index} must be an object"
            )
        role = message.get("role")
        content = message.get("content")
        if not isinstance(role, str) or not role.strip():
            raise EvoRubricsDataError(
                f"row {row_number} message {message_index} has invalid role"
            )
        if not isinstance(content, str) or not content.strip():
            raise EvoRubricsDataError(
                f"row {row_number} message {message_index} has invalid content"
            )
        messages.append({"role": role, "content": content})
    if not messages:
        raise EvoRubricsDataError(f"row {row_number} messages must not be empty")
    if not any(message["role"] == "user" for message in messages):
        raise EvoRubricsDataError(f"row {row_number} messages have no user question")
    return messages


def _question(messages: Sequence[Mapping[str, str]], *, row_number: int) -> str:
    for message in reversed(messages):
        if message["role"] == "user":
            return message["content"].strip()
    raise EvoRubricsDataError(f"row {row_number} messages have no user question")


def _rubrics(record: Mapping[str, Any], *, row_number: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    r0 = record.get("r0")
    if not isinstance(r0, Mapping):
        raise EvoRubricsDataError(f"row {row_number} has invalid r0")
    source_criteria = r0.get("criteria")
    if not isinstance(source_criteria, Sequence) or isinstance(source_criteria, (str, bytes)):
        raise EvoRubricsDataError(f"row {row_number} r0.criteria must be a sequence")
    rubrics: list[dict[str, Any]] = []
    metadata: list[dict[str, Any]] = []
    for criterion_index, source in enumerate(source_criteria):
        if not isinstance(source, Mapping):
            raise EvoRubricsDataError(
                f"row {row_number} criterion {criterion_index} must be an object"
            )
        text = source.get("criterion")
        points = source.get("weight_units")
        if not isinstance(text, str) or not text.strip():
            raise EvoRubricsDataError(
                f"row {row_number} criterion {criterion_index} has invalid text"
            )
        if isinstance(points, bool) or not isinstance(points, (int, float)):
            raise EvoRubricsDataError(
                f"row {row_number} criterion {criterion_index} has invalid weight_units"
            )
        if not math.isfinite(float(points)):
            raise EvoRubricsDataError(
                f"row {row_number} criterion {criterion_index} has non-finite weight_units"
            )
        rubrics.append({"criterion": text, "points": points, "tags": []})
        metadata.append(
            {
                key: source[key]
                for key in (
                    "criterion_id",
                    "canonical_criterion_hash",
                    "criterion_type",
                    "importance_class",
                    "positive_form",
                )
                if key in source
            }
        )
    if not rubrics:
        raise EvoRubricsDataError(f"row {row_number} r0.criteria must not be empty")
    return rubrics, metadata


def convert_rar_record(record: Mapping[str, Any], *, row_number: int) -> dict[str, Any]:
    """Convert one RaR record without copying its reference answer."""

    prompt_id = _required_text(record, "prompt_id", row_number=row_number)
    prompt_hash = _required_text(record, "prompt_hash", row_number=row_number)
    messages = _validated_messages(record, row_number=row_number)
    calculated_prompt_hash = sha256_json(messages)
    if calculated_prompt_hash != prompt_hash:
        raise EvoRubricsDataError(
            f"row {row_number} prompt_hash does not match canonical messages"
        )
    rubrics, criterion_metadata = _rubrics(record, row_number=row_number)
    return {
        "question": _question(messages, row_number=row_number),
        "prompt": messages,
        "rubrics": rubrics,
        "prompt_id": prompt_id,
        "prompt_hash": prompt_hash,
        "source_index": row_number - 1,
        "data_source": f"rar_{record.get('domain', 'unknown')}",
        "dataset_mode": "open_rubrics",
        "metadata": {
            "prompt_id": prompt_id,
            "prompt_hash": prompt_hash,
            "source_prompt_id": record.get("source_prompt_id"),
            "source": record.get("source"),
            "domain": record.get("domain"),
            "source_schema_version": record.get("schema_version"),
            "source_row_sha256": sha256_json(record),
            "criteria": criterion_metadata,
        },
    }


def _convert_split(
    path: Path, *, expected_count: int, split_name: str
) -> tuple[list[dict[str, Any]], list[Mapping[str, Any]]]:
    source = _read_jsonl(path)
    if len(source) != expected_count:
        raise EvoRubricsDataError(
            f"{split_name} must contain {expected_count} rows, got {len(source)}"
        )
    converted = [
        convert_rar_record(record, row_number=index)
        for index, record in enumerate(source, 1)
    ]
    ids = [row["prompt_id"] for row in converted]
    if len(ids) != len(set(ids)):
        raise EvoRubricsDataError(f"{split_name} contains duplicate prompt_id values")
    return converted, source


def _resolve_source(repo_root: Path, value: object, *, name: str) -> Path:
    if not isinstance(value, str) or not value:
        raise EvoRubricsDataError(f"{name} must be a path")
    path = Path(value)
    return path if path.is_absolute() else repo_root / path


def _validate_probe_manifest(
    path: Path,
    *,
    config: Phase1Config,
    train_source_path: Path,
    train_prompt_ids: set[str],
) -> Mapping[str, Any]:
    if not path.is_file():
        raise EvoRubricsDataError(f"missing fixed-train-probe manifest: {path}")
    value = read_json(path)
    if not isinstance(value, Mapping):
        raise EvoRubricsDataError("fixed-train-probe manifest must be an object")
    expected_count = int(config.data["fixed_train_probe"]["count"])
    prompt_ids = value.get("prompt_ids")
    if not isinstance(prompt_ids, list) or len(prompt_ids) != expected_count:
        raise EvoRubricsDataError(
            f"fixed-train-probe manifest must contain {expected_count} prompt IDs"
        )
    if len(set(prompt_ids)) != expected_count or not set(prompt_ids) <= train_prompt_ids:
        raise EvoRubricsDataError(
            "fixed-train-probe prompt IDs must be unique members of the training split"
        )
    if value.get("source_sha256") != sha256_file(train_source_path):
        raise EvoRubricsDataError("fixed-train-probe manifest source hash differs from train")
    if value.get("prompt_ids_sha256") != sha256_json(prompt_ids):
        raise EvoRubricsDataError("fixed-train-probe prompt ID hash is invalid")
    if value.get("domain") != config.domain or value.get("seed") != config.seed:
        raise EvoRubricsDataError("fixed-train-probe manifest domain or seed differs")
    return value


def prepare_evorubrics_data(
    config: Phase1Config,
    *,
    repo_root: str | Path,
    output_dir: str | Path,
    fixed_probe_manifest_path: str | Path,
) -> PreparedEvoRubricsData:
    """Prepare immutable EvoRubrics JSON arrays and a provenance manifest."""

    if config.method != "evorubrics":
        raise EvoRubricsDataError("EvoRubrics data preparation requires method=evorubrics")
    root = Path(repo_root).resolve()
    destination = Path(output_dir).resolve()
    train_source_path = _resolve_source(root, config.data.get("train_path"), name="data.train_path")
    heldout_config = config.data.get("in_domain_policy_eval")
    if not isinstance(heldout_config, Mapping):
        raise EvoRubricsDataError("data.in_domain_policy_eval must be an object")
    heldout_source_path = _resolve_source(
        root, heldout_config.get("path"), name="data.in_domain_policy_eval.path"
    )

    train, _ = _convert_split(
        train_source_path,
        expected_count=int(config.data["train_prompt_count"]),
        split_name="train",
    )
    heldout, _ = _convert_split(
        heldout_source_path,
        expected_count=int(heldout_config["count"]),
        split_name="heldout",
    )
    train_ids = {row["prompt_id"] for row in train}
    heldout_ids = {row["prompt_id"] for row in heldout}
    if train_ids & heldout_ids:
        raise EvoRubricsDataError("train and heldout prompt IDs must be disjoint")

    probe_path = Path(fixed_probe_manifest_path).resolve()
    probe_manifest = _validate_probe_manifest(
        probe_path,
        config=config,
        train_source_path=train_source_path,
        train_prompt_ids=train_ids,
    )

    train_path = destination / "train.open_rubrics.json"
    heldout_path = destination / "heldout.open_rubrics.json"
    manifest_path = destination / "manifest.json"
    write_json_atomic(train_path, train)
    write_json_atomic(heldout_path, heldout)

    manifest: dict[str, Any] = {
        "schema_version": 1,
        "artifact_type": "evorubrics_open_rubrics_dataset",
        "experiment": config.experiment,
        "domain": config.domain,
        "method": config.method,
        "seed": config.seed,
        "config_sha256": config.config_hash,
        "answer_fields_exported": False,
        "splits": {
            "train": {
                "source": artifact_record(train_source_path),
                "source_format": "jsonl",
                "output": artifact_record(train_path),
                "output_format": "json_array",
                "row_count": len(train),
                "ordered_prompt_ids_sha256": sha256_json(
                    [row["prompt_id"] for row in train]
                ),
            },
            "heldout": {
                "source": artifact_record(heldout_source_path),
                "source_format": "jsonl",
                "output": artifact_record(heldout_path),
                "output_format": "json_array",
                "row_count": len(heldout),
                "ordered_prompt_ids_sha256": sha256_json(
                    [row["prompt_id"] for row in heldout]
                ),
                "policy_only": bool(heldout_config.get("policy_only")),
            },
        },
        "fixed_train_probe": {
            "manifest": artifact_record(probe_path),
            "prompt_count": len(probe_manifest["prompt_ids"]),
            "prompt_ids_sha256": probe_manifest["prompt_ids_sha256"],
            "remains_in_training": bool(probe_manifest.get("remains_in_training")),
            "extra_responses_used_for_gradient": bool(
                probe_manifest.get("probe_extra_responses_used_for_gradient")
            ),
        },
    }
    write_json_atomic(manifest_path, manifest)
    return PreparedEvoRubricsData(
        train_path=train_path,
        heldout_path=heldout_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
