#!/usr/bin/env python3
"""Reproduce the step-41 extractor request that emitted invalid JSON.

This diagnostic bypasses ``VLLMChatAdapter.generate`` so it cannot populate or
alter the immutable training provider cache. It writes the exact request and
raw HTTP response bodies to a separate repair directory for regression tests.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

from dynamic_rubric.artifacts import write_bytes_atomic, write_json_atomic
from dynamic_rubric.prompt_versions.onlinerubric_prompt import (
    build_onlinerubric_extractor_messages,
)
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter, _canonical
from dynamic_rubric.training.online_step import EXTRACTION_SCHEMA


OCCURRENCE_ID = "train:112:8703bd2cc10ac9d974f7"
PAIR_INDEX = 1
PAIR_ID = "eb2ccedd02ab613d8c52"
REQUEST_BODY_SHA256 = "9578a15066cc050bf9fb8a1ad82d57472aa1af5eafc340669058f92b62ec5ab7"
CACHE_FILENAME = "b302645db33bf7652e448f67857d6467fc2eb932990ec9e684a556f49a55aa71.json"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:28001")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _args()
    run_root = args.run_root.resolve()
    step_root = run_root / "verl-run/online_steps/step-000041"
    batch = json.loads((step_root / "batch.json").read_text(encoding="utf-8"))
    if int(batch["optimizer_update_index"]) != 41:
        raise RuntimeError("diagnostic input is not step 41")

    source_row = next(
        row
        for row in pq.read_table(run_root / "verl-data/train-online-full.parquet").to_pylist()
        if row["prompt_occurrence_id"] == OCCURRENCE_ID
    )
    with (step_root / "blind_pairs.jsonl").open(encoding="utf-8") as stream:
        occurrence_pairs = [
            row
            for line in stream
            if (row := json.loads(line))["prompt_occurrence_id"] == OCCURRENCE_ID
        ]
    blind_pair = occurrence_pairs[PAIR_INDEX]["blind_pair"]
    if blind_pair["pair_id"] != PAIR_ID:
        raise RuntimeError("persisted pair identity drifted")

    existing_rubric = [
        {
            "criterion_id": item["criterion_id"],
            "criterion": item["text"],
            "weight": item["weight"],
            "source": "offline_r0",
        }
        for item in source_row["extra_info"]["offline_criteria"]
    ]
    request = GenerationRequest(
        prompt_id=source_row["prompt_id"],
        messages=build_onlinerubric_extractor_messages(
            prompt=source_row["extra_info"]["prompt_messages"],
            existing_rubric=existing_rubric,
            response_a=blind_pair["response_a"],
            response_b=blind_pair["response_b"],
        ),
        family="online_rubric_extraction",
        seed=12,
        max_output_tokens=8192,
        json_schema=EXTRACTION_SCHEMA,
        schema_name="onlinerubric_extraction_v1",
        reasoning_effort="medium",
        metadata={
            "optimizer_update_index": 41,
            "prompt_occurrence_id": OCCURRENCE_ID,
            "pair_id": PAIR_ID,
        },
    )
    adapter = VLLMChatAdapter(
        args.base_url,
        "openai/gpt-oss-120b",
        run_root / "verl-run/provider_cache/extractor",
    )
    request_body = _canonical(adapter._payload(request))
    observed_request_hash = hashlib.sha256(request_body).hexdigest()
    if observed_request_hash != REQUEST_BODY_SHA256:
        raise RuntimeError(f"request body drifted: {observed_request_hash}")
    provenance = adapter.request_provenance(request)
    cache_path = Path(provenance["provider_cache_path"])
    if cache_path.name != CACHE_FILENAME or cache_path.exists():
        raise RuntimeError("diagnostic requires the original uncached request identity")

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    request_path = output_dir / "request433.request-body.json"
    response_path = output_dir / "request433.reproduced-raw-http-body.json"
    write_bytes_atomic(request_path, request_body, immutable=True)

    http_request = urllib.request.Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=request_body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(http_request, timeout=180) as response:
        raw_response_body = response.read()
    write_bytes_atomic(response_path, raw_response_body, immutable=True)
    if cache_path.exists():
        raise RuntimeError("diagnostic unexpectedly changed the training provider cache")

    outer = json.loads(raw_response_body)
    text = outer["choices"][0]["message"]["content"]
    try:
        json.loads(text)
        strict_result: dict[str, object] = {"valid": True}
    except json.JSONDecodeError as error:
        strict_result = {
            "valid": False,
            "message": error.msg,
            "line": error.lineno,
            "column": error.colno,
            "position": error.pos,
        }
    controls = [
        {"position": index, "codepoint": ord(character)}
        for index, character in enumerate(text)
        if ord(character) < 32 and character not in "\n\r"
    ]
    write_json_atomic(
        output_dir / "request433.reproduction-manifest.json",
        {
            "schema_version": 1,
            "kind": "step41_extractor_request433_reproduction",
            "historical_failure_raw_response_available": False,
            "artifact_is_new_reproduction": True,
            "request": {
                "optimizer_update_index": 41,
                "global_request_index": 433,
                "prompt_occurrence_id": OCCURRENCE_ID,
                "pair_index": PAIR_INDEX,
                "pair_id": PAIR_ID,
                "seed": 12,
                "body_sha256": observed_request_hash,
                "provider_cache_path": str(cache_path),
            },
            "response": {
                "body_sha256": hashlib.sha256(raw_response_body).hexdigest(),
                "finish_reason": outer["choices"][0].get("finish_reason"),
                "completion_tokens": outer.get("usage", {}).get("completion_tokens"),
                "content_length_characters": len(text),
                "strict_json": strict_result,
                "raw_c0_characters": controls,
            },
            "training_provider_cache_unchanged": True,
            "request_body_artifact": str(request_path),
            "raw_response_body_artifact": str(response_path),
        },
        immutable=True,
    )
    print(output_dir / "request433.reproduction-manifest.json")


if __name__ == "__main__":
    main()
