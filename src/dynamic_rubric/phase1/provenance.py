"""Phase-1 response provenance and fixed-train-probe contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..artifacts import read_jsonl, write_json_atomic
from ..hashing import canonical_json_bytes, sha256_file
from .config import Phase1Config


REQUIRED_RECORD_FIELDS = (
    "domain",
    "method",
    "seed",
    "global_step",
    "checkpoint_id",
    "prompt_id",
    "response_id",
    "pool",
    "policy_checkpoint",
    "evaluator_checkpoint",
    "fresh_or_stale",
)
VALID_POOLS = {"train_batch", "probe_A", "probe_B"}
VALID_EVALUATOR_ROLES = {"fresh", "stale", "not_applicable"}


class ProvenanceError(ValueError):
    pass


def response_id(
    *,
    domain: str,
    method: str,
    seed: int,
    prompt_id: str,
    pool: str,
    policy_checkpoint: str,
    sample_index: int,
    source: str = "current",
) -> str:
    if pool not in VALID_POOLS:
        raise ProvenanceError(f"unknown pool: {pool}")
    if sample_index < 0:
        raise ProvenanceError("sample_index must be non-negative")
    payload = [
        "phase1-response-v1",
        domain,
        method,
        seed,
        prompt_id,
        pool,
        policy_checkpoint,
        source,
        sample_index,
    ]
    return "p1r_" + hashlib.sha256(canonical_json_bytes(payload)).hexdigest()[:32]


def validate_record(record: Mapping[str, Any]) -> None:
    missing = [key for key in REQUIRED_RECORD_FIELDS if key not in record]
    if missing:
        raise ProvenanceError(f"record is missing required fields: {missing}")
    if str(record["pool"]) not in VALID_POOLS:
        raise ProvenanceError(f"invalid pool: {record['pool']}")
    if str(record["fresh_or_stale"]) not in VALID_EVALUATOR_ROLES:
        raise ProvenanceError(
            f"invalid fresh_or_stale value: {record['fresh_or_stale']}"
        )
    if not str(record["prompt_id"]) or not str(record["response_id"]):
        raise ProvenanceError("prompt_id and response_id must be non-empty")


def validate_records(records: Iterable[Mapping[str, Any]]) -> None:
    for record in records:
        validate_record(record)


def validate_pool_ab_disjoint(
    pool_a: Sequence[Mapping[str, Any]],
    pool_b: Sequence[Mapping[str, Any]],
) -> None:
    if not pool_a or not pool_b:
        raise ProvenanceError("Pool A and Pool B must both be non-empty")
    if {str(row.get("pool")) for row in pool_a} != {"probe_A"}:
        raise ProvenanceError("Pool A rows must be labelled probe_A")
    if {str(row.get("pool")) for row in pool_b} != {"probe_B"}:
        raise ProvenanceError("Pool B rows must be labelled probe_B")
    a_ids = {str(row["response_id"]) for row in pool_a}
    b_ids = {str(row["response_id"]) for row in pool_b}
    if len(a_ids) != len(pool_a) or len(b_ids) != len(pool_b):
        raise ProvenanceError("response IDs must be unique within each pool")
    overlap = a_ids & b_ids
    if overlap:
        raise ProvenanceError(f"Pool A/B response overlap: {sorted(overlap)[:3]}")
    if {str(row["prompt_id"]) for row in pool_a} != {
        str(row["prompt_id"]) for row in pool_b
    }:
        raise ProvenanceError("Pool A and Pool B must cover identical prompt IDs")


def _manifest_path(config: Phase1Config, repo_root: Path) -> Path:
    template = str(config.data["fixed_train_probe"]["manifest"])
    return repo_root / template.format(domain=config.domain)


def prepare_fixed_train_probe_manifest(
    config: Phase1Config,
    *,
    repo_root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    root = Path(repo_root)
    train_path = root / str(config.data["train_path"])
    rows = read_jsonl(train_path)
    expected = int(config.data["train_prompt_count"])
    if len(rows) != expected:
        raise ProvenanceError(
            f"expected {expected} train prompts, found {len(rows)} in {train_path}"
        )
    prompt_ids = [str(row.get("prompt_id", "")) for row in rows]
    if any(not prompt_id for prompt_id in prompt_ids):
        raise ProvenanceError("every training row must have prompt_id")
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ProvenanceError("training prompt IDs must be unique")

    probe = config.data["fixed_train_probe"]
    seed = int(probe["sample_seed"])
    count = int(probe["count"])
    ranked = sorted(
        prompt_ids,
        key=lambda prompt_id: (
            hashlib.sha256(f"phase1-probe-v1:{seed}:{prompt_id}".encode()).digest(),
            prompt_id,
        ),
    )
    selected = ranked[:count]
    selected_set = set(selected)
    if not selected_set.issubset(set(prompt_ids)):
        raise ProvenanceError("fixed probe must be a subset of training prompts")

    try:
        source_label = str(train_path.relative_to(root))
    except ValueError:
        source_label = str(train_path)

    manifest = {
        "schema_version": 1,
        "experiment": config.experiment,
        "domain": config.domain,
        "seed": seed,
        "source_split": "train",
        "source_path": source_label,
        "source_sha256": sha256_file(train_path),
        "train_prompt_count": expected,
        "probe_prompt_count": count,
        "sample_algorithm": "sha256(phase1-probe-v1:seed:prompt_id)",
        "prompt_ids": selected,
        "prompt_ids_sha256": hashlib.sha256(canonical_json_bytes(selected)).hexdigest(),
        "remains_in_training": True,
        "probe_extra_responses_used_for_gradient": False,
        "shared_across_methods": True,
    }
    destination = _manifest_path(config, root)
    write_json_atomic(destination, manifest, immutable=True)
    return destination, manifest


def load_probe_rows(
    config: Phase1Config,
    *,
    repo_root: str | Path,
    limit: int | None = None,
) -> list[Mapping[str, Any]]:
    root = Path(repo_root)
    _, manifest = prepare_fixed_train_probe_manifest(config, repo_root=root)
    selected = list(manifest["prompt_ids"])
    if limit is not None:
        if limit <= 0 or limit > len(selected):
            raise ProvenanceError("probe row limit is outside the manifest")
        selected = selected[:limit]
    wanted = set(selected)
    rows = read_jsonl(root / str(config.data["train_path"]))
    by_id = {str(row["prompt_id"]): row for row in rows if str(row["prompt_id"]) in wanted}
    if set(by_id) != wanted:
        raise ProvenanceError("probe manifest references missing train prompts")
    return [by_id[prompt_id] for prompt_id in selected]
