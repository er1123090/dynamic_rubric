"""Immutable, process-safe cache for canonical policy rollout payloads."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from dynamic_rubric.hashing import canonical_json_bytes, sha256_json


class RolloutCacheError(RuntimeError):
    """Raised when an immutable rollout cache entry is malformed or mismatched."""


def _validate_output(output: Mapping[str, Any]) -> None:
    response_ids = output.get("response_ids")
    response_logprobs = output.get("response_logprobs")
    if not isinstance(response_ids, list) or not response_ids:
        raise RolloutCacheError("rollout cache output has no response tokens")
    if not isinstance(response_logprobs, list) or len(response_logprobs) != len(
        response_ids
    ):
        raise RolloutCacheError("rollout cache output has incomplete response logprobs")


@dataclass(frozen=True)
class ImmutableRolloutCache:
    root: Path

    @staticmethod
    def key(identity: Mapping[str, Any]) -> str:
        return sha256_json({"schema_version": 1, "identity": identity})

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def read(self, identity: Mapping[str, Any]) -> dict[str, Any] | None:
        key = self.key(identity)
        path = self._path(key)
        try:
            envelope = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        if (
            envelope.get("schema_version") != 1
            or envelope.get("key") != key
            or envelope.get("identity") != identity
            or not isinstance(envelope.get("output"), dict)
            or envelope.get("output_sha256") != sha256_json(envelope["output"])
        ):
            raise RolloutCacheError(f"rollout cache identity mismatch: {path}")
        _validate_output(envelope["output"])
        return dict(envelope["output"])

    def publish(
        self, identity: Mapping[str, Any], output: Mapping[str, Any]
    ) -> dict[str, Any]:
        key = self.key(identity)
        path = self._path(key)
        normalized_output = dict(output)
        _validate_output(normalized_output)
        envelope = {
            "schema_version": 1,
            "key": key,
            "identity": dict(identity),
            "output_sha256": sha256_json(normalized_output),
            "output": normalized_output,
        }
        encoded = canonical_json_bytes(envelope) + b"\n"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                temporary = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                cached = self.read(identity)
                if cached is None:
                    raise RolloutCacheError("rollout cache publication race")
                return cached
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
        return normalized_output
