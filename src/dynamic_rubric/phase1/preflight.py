"""Credential-free readiness checks for the Phase-1 multi-host topology."""

from __future__ import annotations

import json
import os
import socket
import urllib.request
from pathlib import Path
from typing import Any

from .config import Phase1Config


class PreflightError(RuntimeError):
    pass


def _served_models(base_url: str, *, timeout_seconds: float) -> list[str]:
    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/v1/models",
        headers={"Authorization": "Bearer EMPTY"},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        payload = json.loads(response.read())
    return [str(item["id"]) for item in payload.get("data", ())]


def _endpoint_urls(environment: str, *, expected_count: int) -> tuple[str, ...]:
    singular = environment.removesuffix("S")
    plural = os.getenv(environment, "").strip()
    legacy = os.getenv(singular, "").strip()
    if plural and legacy:
        raise PreflightError(f"{singular} and {environment} cannot both be set")
    raw = plural or legacy
    urls = tuple(value.strip() for value in raw.split(",") if value.strip())
    if len(urls) != expected_count or len(set(urls)) != expected_count:
        raise PreflightError(
            f"{environment} must contain {expected_count} unique comma-separated URLs"
        )
    return urls


def _qwen_endpoint_count() -> int:
    raw = os.getenv("PHASE1_QWEN32B_EXPECTED_COUNT", "2").strip()
    try:
        count = int(raw)
    except ValueError as error:
        raise PreflightError("PHASE1_QWEN32B_EXPECTED_COUNT must be 1 or 2") from error
    if count not in (1, 2):
        raise PreflightError("PHASE1_QWEN32B_EXPECTED_COUNT must be 1 or 2")
    return count


def topology_preflight(
    config: Phase1Config,
    *,
    repo_root: str | Path,
    require_endpoints: bool,
    timeout_seconds: float = 5.0,
) -> dict[str, Any]:
    root = Path(repo_root)
    expected_host = str(config.raw["infrastructure"]["code_host"])
    actual_host = socket.gethostname().split(".", 1)[0]
    snapshot_value = str(config.models["policy"].get("local_snapshot") or "").strip()
    if not snapshot_value:
        raise PreflightError("Set models.policy.local_snapshot to your local model directory")
    local_snapshot = Path(snapshot_value).expanduser()
    if not local_snapshot.is_absolute():
        local_snapshot = root / local_snapshot
    checks: dict[str, Any] = {
        "code_host": {
            "configured": expected_host,
            "actual": actual_host,
            "matched": actual_host == expected_host if expected_host else None,
            "required": False,
            "passed": True,
        },
        "policy_snapshot": {
            "path": str(local_snapshot),
            "passed": local_snapshot.is_dir(),
        },
        "train_data": {
            "path": str(config.data["train_path"]),
            "passed": (root / str(config.data["train_path"])).is_file(),
        },
        "heldout_policy_eval": {
            "path": str(config.data["in_domain_policy_eval"]["path"]),
            "passed": (root / str(config.data["in_domain_policy_eval"]["path"])).is_file(),
            "used_for_update_timing": False,
        },
        "external_policy_eval": {
            "path": str(config.data["external_policy_eval"]["path"]),
            "passed": (root / str(config.data["external_policy_eval"]["path"])).is_file(),
            "required_for_live_smoke": False,
            "policy_only": True,
        },
    }
    try:
        qwen_endpoint_count = _qwen_endpoint_count()
    except PreflightError as error:
        qwen_endpoint_count = 0
        qwen_count_error = str(error)
    else:
        qwen_count_error = None
    services = config.raw["infrastructure"]["services"]
    gpt_count = len(services["gpt_oss_120b"]["instances"])
    if (
        config.raw.get("launch") is not None
        and not os.getenv("PHASE1_QWEN32B_EXPECTED_COUNT", "").strip()
    ):
        configured_urls = os.getenv("PHASE1_QWEN32B_BASE_URLS", "")
        inferred = len([url for url in configured_urls.split(",") if url.strip()])
        qwen_endpoint_count = inferred or len(services["qwen3_32b"]["instances"])
        qwen_count_error = None
    endpoint_specs = {
        "gpt_oss_120b": (
            "PHASE1_GPT_OSS_BASE_URLS",
            "openai/gpt-oss-120b",
            gpt_count,
        ),
        "qwen3_32b": (
            "PHASE1_QWEN32B_BASE_URLS",
            "Qwen/Qwen3-32B",
            qwen_endpoint_count,
        ),
    }
    endpoint_checks: dict[str, Any] = {}
    if require_endpoints:
        for name, (environment, expected_model, expected_count) in endpoint_specs.items():
            if name == "qwen3_32b" and qwen_count_error is not None:
                endpoint_checks[name] = {
                    "environment": environment,
                    "passed": False,
                    "error": qwen_count_error,
                }
                continue
            try:
                base_urls = _endpoint_urls(environment, expected_count=expected_count)
            except PreflightError as error:
                endpoint_checks[name] = {
                    "environment": environment,
                    "passed": False,
                    "error": str(error),
                }
                continue
            instances: list[dict[str, Any]] = []
            for base_url in base_urls:
                try:
                    models = _served_models(base_url, timeout_seconds=timeout_seconds)
                except (OSError, ValueError, json.JSONDecodeError) as error:
                    instances.append(
                        {
                            "base_url": base_url,
                            "passed": False,
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
                else:
                    instances.append(
                        {
                            "base_url": base_url,
                            "served_models": models,
                            "expected_model": expected_model,
                            "passed": expected_model in models,
                        }
                    )
            if not all(value["passed"] for value in instances):
                endpoint_checks[name] = {
                    "environment": environment,
                    "passed": False,
                    "instances": instances,
                }
            else:
                endpoint_checks[name] = {
                    "environment": environment,
                    "passed": True,
                    "instances": instances,
                }
    checks["remote_endpoints"] = endpoint_checks
    required = [
        checks["policy_snapshot"]["passed"],
        checks["train_data"]["passed"],
        checks["heldout_policy_eval"]["passed"],
    ]
    if require_endpoints:
        required.extend(item["passed"] for item in endpoint_checks.values())
        if len(endpoint_checks) != len(endpoint_specs):
            required.append(False)
    result = {
        "schema_version": 1,
        "status": "passed" if all(required) else "failed",
        "require_endpoints": require_endpoints,
        "checks": checks,
        "secrets_logged": False,
        "full_training_authorized": False,
    }
    if result["status"] != "passed":
        raise PreflightError(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return result
