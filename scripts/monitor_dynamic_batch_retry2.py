#!/usr/bin/env python3
"""Monitor the second dynamic-rubric retry and assemble the complete inventory."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI  # type: ignore[import-not-found]

from dynamic_rubric.artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    write_bytes_atomic,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.batch_dynamic import _as_dict, _download_file, _normalize_batch_row
from dynamic_rubric.pipeline import StageError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_ROOT = PROJECT_ROOT / "artifacts/runs/pilot-static-r0-100step-20260821/dynamic-batch"
RETRY1_ROOT = STAGE_ROOT / "retry-001"
RETRY2_ROOT = STAGE_ROOT / "retry-002"
POLL_SECONDS = 300


def _body(row: Mapping[str, Any]) -> Mapping[str, Any]:
    response = row.get("response")
    if not isinstance(response, Mapping):
        raise StageError(f"Batch response is missing: {row.get('custom_id')}")
    body = response.get("body")
    if not isinstance(body, Mapping):
        raise StageError(f"Batch response body is missing: {row.get('custom_id')}")
    return body


def _restored_normalized(
    row: Mapping[str, Any],
    original_id: str,
    identities: Mapping[str, Mapping[str, Any]],
    requested_model: str,
) -> dict[str, Any]:
    restored = dict(row)
    restored["custom_id"] = original_id
    return _normalize_batch_row(restored, identities[original_id], requested_model)


def _collect(client: OpenAI, batch: Mapping[str, Any]) -> dict[str, Any]:
    output_file_id = str(batch.get("output_file_id") or "")
    if not output_file_id:
        raise StageError("second retry Batch completed without an output file")
    payload = _download_file(client, output_file_id)
    raw_path = RETRY2_ROOT / "outputs/retry-incomplete.raw.jsonl"
    write_bytes_atomic(raw_path, payload)
    retry2_rows = [json.loads(line) for line in payload.splitlines() if line.strip()]

    retry2_to_original = {
        str(row["retry_custom_id"]): str(row["original_custom_id"])
        for row in read_jsonl(RETRY2_ROOT / "id_map.jsonl")
    }
    actual_retry2_ids = {str(row.get("custom_id", "")) for row in retry2_rows}
    if actual_retry2_ids != set(retry2_to_original):
        raise StageError("second retry Batch output identity inventory mismatch")

    original_manifest = read_json(STAGE_ROOT / "manifest.json")
    requested_model = str(original_manifest["model"])
    identities = {
        str(row["custom_id"]): row for row in read_jsonl(original_manifest["request_map"]["path"])
    }

    retry2_normalized = []
    for row in retry2_rows:
        retry2_id = str(row["custom_id"])
        body = _body(row)
        if body.get("status") != "completed":
            reason = (body.get("incomplete_details") or {}).get("reason")
            raise StageError(
                f"second retry response is not completed: {retry2_id}="
                f"{body.get('status')}:{reason}"
            )
        retry2_normalized.append(
            _restored_normalized(
                row,
                retry2_to_original[retry2_id],
                identities,
                requested_model,
            )
        )

    retry1_to_original = {
        str(row["retry_custom_id"]): str(row["original_custom_id"])
        for row in read_jsonl(RETRY1_ROOT / "id_map.jsonl")
    }
    retry1_normalized = []
    retry1_incomplete: set[str] = set()
    for row in read_jsonl(RETRY1_ROOT / "outputs/retry-incomplete.raw.jsonl"):
        retry1_id = str(row["custom_id"])
        original_id = retry1_to_original[retry1_id]
        status = _body(row).get("status")
        if status == "completed":
            retry1_normalized.append(
                _restored_normalized(row, original_id, identities, requested_model)
            )
        elif status == "incomplete":
            retry1_incomplete.add(original_id)
        else:
            raise StageError(f"unexpected first retry response status: {retry1_id}={status}")
    if retry1_incomplete != set(retry2_to_original.values()):
        raise StageError("second retry source inventory does not match first retry incompletes")

    original_normalized = []
    original_incomplete: set[str] = set()
    for path in sorted((STAGE_ROOT / "outputs").glob("dynamic-fixed-*.raw.jsonl")):
        for row in read_jsonl(path):
            custom_id = str(row["custom_id"])
            status = _body(row).get("status")
            if status == "completed":
                original_normalized.append(
                    _normalize_batch_row(row, identities[custom_id], requested_model)
                )
            elif status == "incomplete":
                original_incomplete.add(custom_id)
            else:
                raise StageError(f"unexpected original response status: {custom_id}={status}")
    if original_incomplete != set(retry1_to_original.values()):
        raise StageError("first retry source inventory does not match original incompletes")

    normalized = original_normalized + retry1_normalized + retry2_normalized
    identity_keys = [
        (str(row["prompt_id"]), int(row["policy_step"]), str(row["replicate_id"]))
        for row in normalized
    ]
    if len(normalized) != len(identities) or len(identity_keys) != len(set(identity_keys)):
        raise StageError("combined dynamic-rubric identity inventory is incomplete or duplicated")
    returned_models = {str(row["provider_call"]["returned_model"]) for row in normalized}
    if len(returned_models) != 1:
        raise StageError(f"returned-model drift detected: {sorted(returned_models)}")

    normalized.sort(
        key=lambda row: (
            int(row["policy_step"]),
            str(row["prompt_id"]),
            str(row["replicate_id"]),
        )
    )
    retry2_normalized_path = RETRY2_ROOT / "outputs/retry-incomplete.jsonl"
    write_jsonl_atomic(retry2_normalized_path, retry2_normalized)
    output_path = STAGE_ROOT / "dynamic_candidates.jsonl"
    write_jsonl_atomic(output_path, normalized)
    result = {
        "run_id": original_manifest["run_id"],
        "requests": len(normalized),
        "completed_from_original": len(original_normalized),
        "completed_from_retry1": len(retry1_normalized),
        "completed_from_retry2": len(retry2_normalized),
        "returned_model": next(iter(returned_models)),
        "retry2_batch_id": batch["id"],
        "retry2_raw": artifact_record(raw_path),
        "retry2_normalized": artifact_record(retry2_normalized_path),
        "output": artifact_record(output_path),
    }
    write_json_atomic(STAGE_ROOT / "collection.json", result)
    return result


def _status(client: OpenAI) -> dict[str, Any]:
    submission = read_json(RETRY2_ROOT / "submission.json")
    batch = _as_dict(client.batches.retrieve(str(submission["batch_id"])))
    status = {
        "batch_id": batch["id"],
        "status": batch["status"],
        "request_counts": batch.get("request_counts"),
        "output_file_id": batch.get("output_file_id"),
        "error_file_id": batch.get("error_file_id"),
        "expires_at": batch.get("expires_at"),
    }
    write_json_atomic(RETRY2_ROOT / "status.json", status, immutable=False)
    print(json.dumps(status, sort_keys=True), flush=True)
    return batch


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    api_key = os.environ.get("OPENAI_API_KEY", "")
    if not api_key:
        raise StageError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=api_key)
    while True:
        batch = _status(client)
        status = str(batch["status"])
        if status == "completed":
            result = _collect(client, batch)
            print(json.dumps(result, sort_keys=True), flush=True)
            return 0
        if status in {"failed", "expired", "cancelled"}:
            raise StageError(f"second retry Batch terminated with status={status}")
        if args.once:
            return 0
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    raise SystemExit(main())
