from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from ..artifacts import ImmutableArtifactError, write_bytes_atomic, write_jsonl_atomic


class SealedTrajectoryError(PermissionError):
    pass


def _canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def operator_hash(operator: Mapping[str, Any]) -> str:
    return hashlib.sha256(_canonical(operator)).hexdigest()


def _atomic_create(path: Path, payload: bytes) -> None:
    try:
        write_bytes_atomic(path, payload)
    except ImmutableArtifactError as error:
        raise FileExistsError(
            f"immutable file already exists with different bytes: {path}"
        ) from error


def freeze_updater(lock_path: Path, operator: Mapping[str, Any], run_id: str) -> dict[str, Any]:
    lock = {
        "schema_version": 1,
        "run_id": run_id,
        "operator": dict(operator),
        "operator_hash": operator_hash(operator),
        "scope": "development_only_freeze",
    }
    _atomic_create(lock_path, _canonical(lock) + b"\n")
    return lock


def load_updater_lock(lock_path: Path, expected_operator: Mapping[str, Any]) -> dict[str, Any]:
    if not lock_path.is_file():
        raise SealedTrajectoryError("final trajectory is sealed until updater_lock.json exists")
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    expected_hash = operator_hash(expected_operator)
    if lock.get("operator_hash") != expected_hash:
        raise SealedTrajectoryError("updater lock does not match configured operator")
    return lock


def write_sealed_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    write_jsonl_atomic(path, rows)


def read_final_sealed(
    path: Path,
    lock_path: Path,
    expected_operator: Mapping[str, Any],
    unseal_receipt: Path,
) -> list[dict[str, Any]]:
    lock = load_updater_lock(lock_path, expected_operator)
    receipt = {
        "schema_version": 1,
        "sealed_input_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "operator_hash": lock["operator_hash"],
        "one_time_unseal": True,
    }
    _atomic_create(unseal_receipt, _canonical(receipt) + b"\n")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
