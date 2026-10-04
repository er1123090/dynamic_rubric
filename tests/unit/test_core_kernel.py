from __future__ import annotations

import dataclasses

import pytest

from dynamic_rubric.artifacts import (
    ImmutableArtifactError,
    InventoryError,
    ResumeIncompatibleError,
    assert_resume_compatible,
    read_jsonl,
    validate_inventory,
    write_json_atomic,
    write_jsonl_atomic,
    write_manifest,
)
from dynamic_rubric.config import ConfigError, config_from_mapping, validate_stage_paths
from dynamic_rubric.hashing import canonical_json_bytes, sha256_json
from dynamic_rubric.schemas import Criterion, RubricSnapshot, RunManifest
from dynamic_rubric.seeds import RESPONSE_FAMILIES, derive_seed, response_id, seed_namespace


def test_canonical_json_and_hash_are_order_independent() -> None:
    left = {"b": [1, 2], "a": "한글"}
    right = {"a": "한글", "b": [1, 2]}
    assert canonical_json_bytes(left) == canonical_json_bytes(right)
    assert sha256_json(left) == sha256_json(right)
    with pytest.raises(ValueError):
        canonical_json_bytes({"bad": float("nan")})


def test_seed_namespaces_and_response_ids_are_disjoint_and_deterministic() -> None:
    seeds: set[int] = set()
    ids: set[str] = set()
    for family in RESPONSE_FAMILIES:
        for prompt in ("p0", "p1", "p2"):
            for step in range(4):
                for sample in range(12):
                    seed = derive_seed("run", family, prompt, step, sample)
                    identifier = response_id("run", family, prompt, step, sample)
                    assert seed_namespace(seed) is family
                    assert seed not in seeds
                    assert identifier not in ids
                    assert seed == derive_seed("run", family, prompt, step, sample)
                    assert identifier == response_id("run", family, prompt, step, sample)
                    seeds.add(seed)
                    ids.add(identifier)


def test_frozen_schema_and_rubric_content_hash() -> None:
    criterion = Criterion("c1", "States the relevant action.", "static")
    rubric = RubricSnapshot("r0", "p0", "static", 0, (criterion,))
    assert rubric.content_hash == RubricSnapshot("r1", "p0", "static", 1, (criterion,)).content_hash
    with pytest.raises(dataclasses.FrozenInstanceError):
        criterion.text = "changed"  # type: ignore[misc]


def test_config_rejects_private_public_and_dynamic_training_paths() -> None:
    with pytest.raises(ConfigError):
        validate_stage_paths({"input": "data/private_gt/pilot.jsonl"}, stage="replay-dynamic")
    with pytest.raises(ConfigError):
        config_from_mapping(
            {"training": {"artifact_inputs": ["artifacts/dynamic_fixed/rubric.json"]}},
            stage="train-static",
        )
    audit = config_from_mapping({"paths": {"private_gt": "data/private_gt"}}, stage="audit-gold")
    assert audit.paths.private_gt == "data/private_gt"


def _manifest(config_hash: str = "a" * 64) -> RunManifest:
    return RunManifest("pilot", "prepare-data", config_hash, {"source": "b" * 64})


def test_immutable_manifest_resume_compatibility(tmp_path) -> None:
    path = tmp_path / "manifest.json"
    manifest = _manifest()
    assert write_manifest(path, manifest)
    original = path.read_bytes()
    assert not write_manifest(path, manifest)
    assert path.read_bytes() == original
    assert_resume_compatible(path, manifest)
    with pytest.raises((ImmutableArtifactError, ResumeIncompatibleError)):
        write_manifest(path, _manifest("c" * 64))
    with pytest.raises(ResumeIncompatibleError):
        assert_resume_compatible(path, _manifest("c" * 64))


def test_atomic_json_and_jsonl_resume_are_byte_identical(tmp_path) -> None:
    json_path = tmp_path / "x.json"
    jsonl_path = tmp_path / "x.jsonl"
    records = [{"z": 2, "a": 1}, {"text": "ok"}]
    assert write_json_atomic(json_path, records[0])
    assert write_jsonl_atomic(jsonl_path, records)
    first_json, first_jsonl = json_path.read_bytes(), jsonl_path.read_bytes()
    assert not write_json_atomic(json_path, {"a": 1, "z": 2})
    assert not write_jsonl_atomic(jsonl_path, records)
    assert (json_path.read_bytes(), jsonl_path.read_bytes()) == (first_json, first_jsonl)
    assert read_jsonl(jsonl_path) == records
    with pytest.raises(ImmutableArtifactError):
        write_jsonl_atomic(jsonl_path, records[:-1])
    assert not list(tmp_path.glob("*.tmp"))


def test_inventory_counts_and_uniqueness() -> None:
    records = [
        {"prompt_id": prompt, "sample_index": sample, "response_id": f"{prompt}-{sample}"}
        for prompt in ("p0", "p1")
        for sample in range(4)
    ]
    assert validate_inventory(
        records, group_by=("prompt_id",), expected_per_group=4, unique_by=("response_id",)
    ) == {("p0",): 4, ("p1",): 4}
    with pytest.raises(InventoryError):
        validate_inventory(records[:-1], group_by=("prompt_id",), expected_per_group=4)
    with pytest.raises(InventoryError):
        validate_inventory(records + [records[0]], group_by=("prompt_id",), expected_per_group=4)
