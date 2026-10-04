"""Machine-readable report assembly for discriminability horizon results."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import json
import math
from pathlib import Path
from typing import Any

from .inference import CrossedBootstrapResult, HorizonDecision


REPORT_SCHEMA_VERSION = "horizon_report_v1"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("reports must not contain non-finite floats")
    return value


def coverage_summary(*, valid: int, invalid: int, missing: int, na: int = 0) -> dict[str, Any]:
    counts = (valid, invalid, missing, na)
    if any(not isinstance(value, int) or value < 0 for value in counts):
        raise ValueError("coverage counts must be non-negative integers")
    total = sum(counts)
    return {
        "total": total,
        "valid": valid,
        "invalid": invalid,
        "missing": missing,
        "na": na,
        "valid_rate": valid / total if total else None,
    }


def build_horizon_report(
    *,
    domain: str,
    bootstrap: CrossedBootstrapResult,
    decision: HorizonDecision,
    coverage: dict[str, Any],
    revisions: dict[str, str],
    directional_validity: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not domain:
        raise ValueError("domain must not be empty")
    claim_scope = (
        "directionally_validated_discriminability"
        if directional_validity is not None
        else "empirical_discriminability_only"
    )
    return _jsonable(
        {
            "schema_version": REPORT_SCHEMA_VERSION,
            "domain": domain,
            "bootstrap": bootstrap,
            "horizon_decision": decision,
            "coverage": coverage,
            "revisions": revisions,
            "directional_validity": directional_validity,
            "claim_scope": claim_scope,
            "limitations": [
                "seed clusters are few; seed-specific trajectories must accompany aggregate bands"
            ]
            if bootstrap.small_seed_cluster_warning
            else [],
        }
    )


def canonical_report_json(report: dict[str, Any]) -> str:
    return json.dumps(_jsonable(report), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def write_horizon_report(path: str | Path, report: dict[str, Any]) -> None:
    """Write a deterministic JSON artifact; parent creation is intentionally explicit upstream."""

    target = Path(path)
    target.write_text(canonical_report_json(report) + "\n", encoding="utf-8")

