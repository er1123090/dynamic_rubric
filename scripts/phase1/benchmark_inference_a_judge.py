#!/usr/bin/env python3
"""Deterministic concurrent benchmark for the OpenAI-compatible judge endpoint."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


SERVED_MODEL = "openai/gpt-oss-120b"


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def strip_markdown_fence(text: str) -> str:
    value = text.strip()
    if value.startswith("```json"):
        value = value[7:].strip()
    elif value.startswith("```"):
        value = value[3:].strip()
    if value.endswith("```"):
        value = value[:-3].strip()
    return value


def validate_content(content: str, expected_items: int) -> None:
    parsed = json.loads(strip_markdown_fence(content))
    if not isinstance(parsed, list) or len(parsed) != expected_items:
        raise ValueError(f"expected JSON array of {expected_items} items")
    for index, row in enumerate(parsed):
        if not isinstance(row, dict):
            raise ValueError(f"item {index} is not an object")
        if row.get("index") != index:
            raise ValueError(f"item {index} has wrong index {row.get('index')!r}")
        if not isinstance(row.get("criteria_met"), bool):
            raise ValueError(f"item {index} has non-boolean criteria_met")
        if not isinstance(row.get("explanation"), str) or not row["explanation"].strip():
            raise ValueError(f"item {index} has empty explanation")


def request_json(url: str, payload: dict[str, Any], timeout: float) -> tuple[dict[str, Any], dict[str, str]]:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        method="POST",
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
        headers = {key.lower(): value for key, value in response.headers.items()}
    return body, headers


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def run_one(base_url: str, case: dict[str, Any], timeout: float) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        body, headers = request_json(
            f"{base_url.rstrip('/')}/v1/chat/completions",
            case["payload"],
            timeout,
        )
        choice = body["choices"][0]
        content = choice["message"].get("content") or ""
        validate_content(content, int(case["expected_items"]))
        usage = body.get("usage") or {}
        return {
            "ok": True,
            "latency_seconds": time.perf_counter() - started,
            "completion_tokens": int(usage.get("completion_tokens") or 0),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "finish_reason": choice.get("finish_reason"),
            "upstream": headers.get("x-judge-upstream"),
        }
    except Exception as error:  # benchmark must retain every failed sample
        return {
            "ok": False,
            "latency_seconds": time.perf_counter() - started,
            "error": f"{type(error).__name__}: {error}",
        }


def load_cases(path: Path) -> list[dict[str, Any]]:
    cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not cases:
        raise ValueError(f"fixture is empty: {path}")
    for case in cases:
        if "payload" not in case or "expected_items" not in case:
            raise ValueError("each fixture row requires payload and expected_items")
        case["payload"]["model"] = SERVED_MODEL
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--write-baseline", action="store_true")
    parser.add_argument("--require-speedup", type=float, default=1.0)
    parser.add_argument("--require-success-rate", type=float, default=1.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.requests <= 0 or args.concurrency <= 0:
        raise ValueError("requests and concurrency must be positive")
    base_url = args.base_url.rstrip("/")
    models = get_json(f"{base_url}/v1/models", min(args.timeout, 30.0))
    model_ids = [row.get("id") for row in models.get("data", [])]
    if model_ids.count(SERVED_MODEL) != 1:
        raise RuntimeError(f"expected exactly one {SERVED_MODEL!r}, got {model_ids!r}")

    fixture_bytes = args.fixture.read_bytes()
    cases = load_cases(args.fixture)
    workload = [cases[index % len(cases)] for index in range(args.requests)]
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        results = list(executor.map(lambda case: run_one(base_url, case, args.timeout), workload))
    wall_seconds = time.perf_counter() - started

    ok_results = [row for row in results if row["ok"]]
    latencies = [float(row["latency_seconds"]) for row in results]
    success_rate = len(ok_results) / len(results)
    summary: dict[str, Any] = {
        "schema_version": 1,
        "served_model": SERVED_MODEL,
        "fixture_sha256": hashlib.sha256(fixture_bytes).hexdigest(),
        "requests": args.requests,
        "concurrency": args.concurrency,
        "successes": len(ok_results),
        "success_rate": success_rate,
        "wall_seconds": wall_seconds,
        "requests_per_minute": args.requests * 60.0 / wall_seconds,
        "latency_seconds": {
            "median": statistics.median(latencies),
            "p95": percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "prompt_tokens": sum(int(row.get("prompt_tokens", 0)) for row in ok_results),
        "completion_tokens": sum(int(row.get("completion_tokens", 0)) for row in ok_results),
        "finish_reasons": {},
        "upstreams": {},
        "failures": [row.get("error") for row in results if not row["ok"]],
    }
    for row in ok_results:
        for key, field in (("finish_reasons", "finish_reason"), ("upstreams", "upstream")):
            value = str(row.get(field))
            summary[key][value] = summary[key].get(value, 0) + 1

    passed = success_rate >= args.require_success_rate
    if args.baseline and args.baseline.exists() and not args.write_baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        if baseline.get("fixture_sha256") != summary["fixture_sha256"]:
            raise RuntimeError("baseline fixture hash does not match candidate fixture")
        baseline_rpm = float(baseline["requests_per_minute"])
        summary["baseline_requests_per_minute"] = baseline_rpm
        summary["speedup"] = summary["requests_per_minute"] / baseline_rpm
        passed = passed and summary["speedup"] >= args.require_speedup
    elif not args.write_baseline:
        raise FileNotFoundError("baseline is required unless --write-baseline is set")

    summary["passed"] = passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (urllib.error.URLError, TimeoutError, ValueError, RuntimeError, FileNotFoundError) as error:
        print(f"benchmark error: {error}", file=sys.stderr)
        raise SystemExit(2)
