"""Build and verify inference observations exclusively from sealed score summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from ..hashing import sha256_file


class ObservationSealError(RuntimeError):
    pass


def observation_seal_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.seal.json")


def _verified_summary(path: Path) -> tuple[list[Mapping[str, Any]], Mapping[str, Any]]:
    seal_path = path.parent / "score_seal.json"
    if not seal_path.is_file():
        raise ObservationSealError(f"score seal is missing for summary: {path}")
    seal = read_json(seal_path)
    if seal.get("artifact_type") != "horizon_score_seal":
        raise ObservationSealError(f"wrong score seal artifact type: {seal_path}")
    outputs = seal.get("outputs")
    if not isinstance(outputs, Mapping) or outputs.get("prompt_summary") != sha256_file(path):
        raise ObservationSealError(f"score summary digest mismatch: {path}")
    rows = read_jsonl(path)
    if len(rows) != int(seal.get("prompt_count", -1)):
        raise ObservationSealError(f"score summary prompt count mismatch: {path}")
    return rows, seal


def build_horizon_observations(
    summary_paths: Sequence[Path],
    *,
    output_path: Path,
    expected_seed_ids: Sequence[str] | None = None,
    expected_prompt_ids: Sequence[str] | None = None,
    expected_checkpoints: Sequence[float] | None = None,
    expected_config_hash: str | None = None,
) -> dict[str, Any]:
    if not summary_paths:
        raise ValueError("at least one sealed score summary is required")
    source_records: list[dict[str, Any]] = []
    cells: dict[tuple[str, str, float], Mapping[str, Any]] = {}
    for path in summary_paths:
        rows, seal = _verified_summary(path)
        if expected_config_hash is not None and seal.get("config_hash") != expected_config_hash:
            raise ObservationSealError("score seal config hash differs from the requested run")
        source_records.append(
            {
                "summary_path": str(path),
                "summary_sha256": sha256_file(path),
                "score_seal_path": str(path.parent / "score_seal.json"),
                "score_seal_sha256": sha256_file(path.parent / "score_seal.json"),
                "seed_id": str(seal["seed_id"]),
                "checkpoint": float(seal["checkpoint"]),
            }
        )
        for row in rows:
            key = (str(row["seed_id"]), str(row["prompt_id"]), float(row["checkpoint"]))
            if key in cells:
                raise ObservationSealError(f"duplicate score-summary cell: {key}")
            if int(row.get("response_count", 0)) != 16:
                raise ObservationSealError(f"score-summary cell is not a 16-response group: {key}")
            variants = row.get("variants")
            status = str(row.get("analysis_status", "valid"))
            if status not in {"valid", "na"}:
                raise ObservationSealError(f"unknown score-summary analysis status: {key}")
            required_variants = (
                {"r0", "current", "control"} if status == "valid" else {"r0", "current"}
            )
            if not isinstance(variants, Mapping) or not required_variants <= set(variants):
                raise ObservationSealError(f"score-summary variants are incomplete: {key}")
            cells[key] = row

    observed_seeds = {key[0] for key in cells}
    observed_prompts = {key[1] for key in cells}
    observed_checkpoints = {key[2] for key in cells}
    seeds = tuple(
        sorted({str(value) for value in expected_seed_ids})
        if expected_seed_ids is not None
        else sorted(observed_seeds)
    )
    expected_prompts = tuple(
        sorted({str(value) for value in expected_prompt_ids})
        if expected_prompt_ids is not None
        else sorted(observed_prompts)
    )
    checkpoints = tuple(
        sorted({float(value) for value in expected_checkpoints})
        if expected_checkpoints is not None
        else sorted(observed_checkpoints)
    )
    if not observed_seeds <= set(seeds):
        raise ObservationSealError("score summaries do not match the expected training seeds")
    if not observed_prompts <= set(expected_prompts):
        raise ObservationSealError("score summaries do not match the expected prompt IDs")
    if not observed_checkpoints <= set(checkpoints):
        raise ObservationSealError("score summaries do not match the expected checkpoints")
    if 0.0 not in checkpoints:
        raise ObservationSealError("score summaries must include a checkpoint-zero R0 baseline")

    expected_keys = {
        (seed_id, prompt_id, checkpoint)
        for seed_id in seeds
        for prompt_id in expected_prompts
        for checkpoint in checkpoints
    }
    missing_keys = expected_keys - set(cells)
    na_keys = {
        key for key, row in cells.items() if str(row.get("analysis_status", "valid")) == "na"
    }
    eligible_prompts = tuple(
        prompt_id
        for prompt_id in expected_prompts
        if all(
            (seed_id, prompt_id, checkpoint) in cells
            and (seed_id, prompt_id, checkpoint) not in na_keys
            for seed_id in seeds
            for checkpoint in checkpoints
        )
    )
    if not eligible_prompts:
        raise ObservationSealError("no prompt has complete valid paired coverage")

    baseline: dict[tuple[str, str], int] = {}
    for seed_id in seeds:
        for prompt_id in eligible_prompts:
            row = cells[(seed_id, prompt_id, 0.0)]
            baseline[(seed_id, prompt_id)] = int(bool(row["variants"]["r0"]["exact_zar"]))
    observations = []
    for seed_id in seeds:
        for prompt_id in eligible_prompts:
            for checkpoint in checkpoints:
                row = cells[(seed_id, prompt_id, checkpoint)]
                observations.append(
                    {
                        "schema_version": 1,
                        "seed_id": seed_id,
                        "prompt_id": prompt_id,
                        "checkpoint": checkpoint,
                        "r0_zar": int(bool(row["variants"]["r0"]["exact_zar"])),
                        "r0_baseline_zar": baseline[(seed_id, prompt_id)],
                        "current_zar": int(bool(row["variants"]["current"]["exact_zar"])),
                        "control_zar": int(bool(row["variants"]["control"]["exact_zar"])),
                    }
                )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl_atomic(output_path, observations)
    expected_cell_count = len(expected_keys)
    valid_count = len(observations)
    paired_excluded = expected_cell_count - valid_count - len(na_keys) - len(missing_keys)
    seal = {
        "schema_version": 1,
        "artifact_type": "horizon_observation_seal",
        "config_hash": expected_config_hash,
        "observation_sha256": sha256_file(output_path),
        "observation_count": len(observations),
        "seed_ids": list(seeds),
        "prompt_ids": list(eligible_prompts),
        "expected_prompt_ids": list(expected_prompts),
        "checkpoints": list(checkpoints),
        "coverage": {
            "expected": expected_cell_count,
            "valid": valid_count,
            "invalid": paired_excluded,
            "missing": len(missing_keys),
            "na": len(na_keys),
            "paired_excluded": paired_excluded,
        },
        "score_summary_sources": source_records,
    }
    seal_path = observation_seal_path(output_path)
    write_json_atomic(seal_path, seal)
    return {"observations": str(output_path), "seal": str(seal_path), **seal}


def verify_horizon_observation_seal(path: Path) -> Mapping[str, Any]:
    seal_path = observation_seal_path(path)
    if not seal_path.is_file():
        raise ObservationSealError(f"observation seal is missing: {seal_path}")
    seal = read_json(seal_path)
    if seal.get("artifact_type") != "horizon_observation_seal":
        raise ObservationSealError("observation seal has the wrong artifact type")
    if seal.get("observation_sha256") != sha256_file(path):
        raise ObservationSealError("observation digest does not match its seal")
    if int(seal.get("observation_count", -1)) != len(read_jsonl(path)):
        raise ObservationSealError("observation count does not match its seal")
    sources = seal.get("score_summary_sources")
    if not isinstance(sources, list) or not sources:
        raise ObservationSealError("observation seal has no score-summary lineage")
    for source in sources:
        summary_path = Path(str(source["summary_path"]))
        score_seal_path = Path(str(source["score_seal_path"]))
        if (
            not summary_path.is_file()
            or sha256_file(summary_path) != source["summary_sha256"]
            or not score_seal_path.is_file()
            or sha256_file(score_seal_path) != source["score_seal_sha256"]
        ):
            raise ObservationSealError("a sealed observation source is absent or has drifted")
    return seal
