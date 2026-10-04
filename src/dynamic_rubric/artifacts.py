"""Crash-safe, immutable artifact and inventory helpers."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .hashing import canonical_json_bytes, sha256_bytes, sha256_file
from .schemas import RunManifest


class ArtifactError(RuntimeError):
    pass


class ImmutableArtifactError(ArtifactError):
    pass


class ResumeIncompatibleError(ArtifactError):
    pass


class InventoryError(ArtifactError):
    pass


def _atomic_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _immutable_write(path: str | Path, payload: bytes) -> bool:
    destination = Path(path)
    if destination.exists():
        existing = destination.read_bytes()
        if existing == payload:
            return False
        raise ImmutableArtifactError(f"refusing to overwrite immutable artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            # link(2) atomically creates without overwriting a concurrent winner.
            os.link(temporary, destination)
        except FileExistsError:
            existing = destination.read_bytes()
            if existing == payload:
                return False
            raise ImmutableArtifactError(
                f"refusing to overwrite immutable artifact: {destination}"
            ) from None
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        return True
    finally:
        temporary.unlink(missing_ok=True)


def write_json_atomic(path: str | Path, value: Any, *, immutable: bool = True) -> bool:
    payload = canonical_json_bytes(value) + b"\n"
    if immutable:
        return _immutable_write(path, payload)
    _atomic_bytes(Path(path), payload)
    return True


def write_bytes_atomic(path: str | Path, payload: bytes, *, immutable: bool = True) -> bool:
    if immutable:
        return _immutable_write(path, payload)
    _atomic_bytes(Path(path), payload)
    return True


def write_text_atomic(
    path: str | Path, text: str, *, immutable: bool = True, encoding: str = "utf-8"
) -> bool:
    payload = text.encode(encoding)
    if immutable:
        return _immutable_write(path, payload)
    _atomic_bytes(Path(path), payload)
    return True


def jsonl_bytes(records: Iterable[Any]) -> bytes:
    return b"".join(canonical_json_bytes(record) + b"\n" for record in records)


def write_jsonl_atomic(path: str | Path, records: Iterable[Any], *, immutable: bool = True) -> bool:
    payload = jsonl_bytes(records)
    if immutable:
        return _immutable_write(path, payload)
    _atomic_bytes(Path(path), payload)
    return True


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path: str | Path) -> list[Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_manifest(path: str | Path, manifest: RunManifest | Mapping[str, Any]) -> bool:
    value = dataclasses.asdict(manifest) if dataclasses.is_dataclass(manifest) else dict(manifest)
    return write_json_atomic(path, value, immutable=True)


def assert_resume_compatible(path: str | Path, expected: RunManifest | Mapping[str, Any]) -> None:
    destination = Path(path)
    if not destination.exists():
        raise ResumeIncompatibleError(f"manifest does not exist: {destination}")
    actual = read_json(destination)
    expected_value = (
        dataclasses.asdict(expected) if dataclasses.is_dataclass(expected) else dict(expected)
    )
    if canonical_json_bytes(actual) != canonical_json_bytes(expected_value):
        raise ResumeIncompatibleError("manifest differs from the requested run")


def artifact_record(path: str | Path) -> dict[str, Any]:
    artifact = Path(path)
    return {
        "path": str(artifact),
        "sha256": sha256_file(artifact),
        "bytes": artifact.stat().st_size,
    }


def validate_artifact_record(record: Mapping[str, Any]) -> None:
    path = Path(str(record["path"]))
    if not path.is_file():
        raise InventoryError(f"missing artifact: {path}")
    if path.stat().st_size != int(record["bytes"]) or sha256_file(path) != record["sha256"]:
        raise InventoryError(f"artifact digest/size mismatch: {path}")


def inventory(records: Sequence[Mapping[str, Any]], *keys: str) -> dict[tuple[Any, ...], int]:
    if not keys:
        raise ValueError("at least one inventory key is required")
    counts: Counter[tuple[Any, ...]] = Counter()
    for record in records:
        try:
            counts[tuple(record[key] for key in keys)] += 1
        except KeyError as exc:
            raise InventoryError(f"inventory record is missing key {exc.args[0]!r}") from exc
    return dict(sorted(counts.items(), key=lambda item: repr(item[0])))


def validate_inventory(
    records: Sequence[Mapping[str, Any]],
    *,
    group_by: Sequence[str],
    expected_per_group: int,
    unique_by: Sequence[str] = (),
) -> dict[tuple[Any, ...], int]:
    if expected_per_group < 0:
        raise ValueError("expected_per_group must be non-negative")
    counts = inventory(records, *group_by)
    wrong = {key: count for key, count in counts.items() if count != expected_per_group}
    if wrong:
        raise InventoryError(f"unexpected group counts: {wrong}")
    if unique_by:
        values = [tuple(record[key] for key in unique_by) for record in records]
        if len(values) != len(set(values)):
            raise InventoryError(f"duplicate inventory identity for keys {tuple(unique_by)!r}")
    return counts


atomic_write_json = write_json_atomic
atomic_write_jsonl = write_jsonl_atomic


def manifest_hash(manifest: RunManifest | Mapping[str, Any]) -> str:
    value = dataclasses.asdict(manifest) if dataclasses.is_dataclass(manifest) else manifest
    return sha256_bytes(canonical_json_bytes(value))
