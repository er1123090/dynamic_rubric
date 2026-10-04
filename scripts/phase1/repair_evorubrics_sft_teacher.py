#!/usr/bin/env python3
"""Repair rejected GPT-OSS teacher rows and assemble a mixed-profile SFT dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_ROOT = SCRIPT_DIR.parents[1] / "src"
for import_root in (SRC_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from generate_evorubrics_sft_teacher import (
    EXPECTED_HELDOUT,
    EXPECTED_TRAIN,
    MAX_STUDENT_TOKENS,
    STUDENT_MODEL,
    STUDENT_MODEL_ID,
    STUDENT_REVISION,
    TEACHER_MODEL,
    TEACHER_REVISION,
    OpenAIBackend,
    TeacherGenerationError,
    TeacherSettings,
    deterministic_seed,
    load_and_validate_splits,
    receipt_name,
    validate_teacher_response,
)

from dynamic_rubric.artifacts import read_json, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json

WORD_LIMITS = (250, 180, 120, 100, 80)
MAX_TOKENS = 4096


class RepairError(RuntimeError):
    """Raised when repair or mixed-profile assembly cannot be proven valid."""


def repair_instruction(word_limit: int) -> str:
    return (
        "Answer the medical question accurately and completely using no more than "
        f"{word_limit} words. This is a hard limit. Give only the final answer, with no "
        "chain of thought, hidden reasoning, grading rubric, or reference-answer discussion. "
        "Prioritize the clinically essential facts and do not say that you shortened the answer."
    )


def repair_request_messages(
    messages: Sequence[Mapping[str, str]], word_limit: int
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": repair_instruction(word_limit)},
        *[dict(message) for message in messages],
    ]


def _load_original_identity(run_root: Path) -> dict[str, Any]:
    progress_path = run_root / "teacher" / "progress.json"
    if not progress_path.is_file():
        raise RepairError(f"original generator has no completed progress artifact: {progress_path}")
    progress = read_json(progress_path)
    identity = progress.get("cache_identity")
    if not isinstance(identity, dict) or identity.get("sha256") != sha256_json(
        {key: value for key, value in identity.items() if key != "sha256"}
    ):
        raise RepairError("original generator cache identity is missing or invalid")
    teacher = identity.get("teacher", {})
    if teacher.get("model") != TEACHER_MODEL or teacher.get("revision") != TEACHER_REVISION:
        raise RepairError("original receipts do not use the pinned GPT-OSS teacher")
    return identity


def _validated_receipt(
    path: Path,
    *,
    row: Mapping[str, Any],
    expected_identity_sha256: str,
    tokenizer: Any,
) -> dict[str, Any]:
    receipt = read_json(path)
    if receipt.get("status") != "accepted":
        raise RepairError(f"receipt is not accepted: {path}")
    if receipt.get("cache_identity_sha256") != expected_identity_sha256:
        raise RepairError(f"receipt cache identity mismatch: {path}")
    if (
        receipt.get("prompt_id") != row["prompt_id"]
        or receipt.get("prompt_hash") != row["prompt_hash"]
        or receipt.get("source_row_sha256") != row["source_row_sha256"]
        or receipt.get("original_messages") != row["messages"]
    ):
        raise RepairError(f"receipt source identity mismatch: {path}")
    attempt_number = receipt.get("accepted_attempt")
    attempts = receipt.get("attempts")
    if not isinstance(attempt_number, int) or not isinstance(attempts, list):
        raise RepairError(f"receipt has invalid attempt provenance: {path}")
    accepted = next(
        (
            attempt
            for attempt in attempts
            if attempt.get("attempt") == attempt_number and attempt.get("status") == "accepted"
        ),
        None,
    )
    if not isinstance(accepted, dict) or not isinstance(accepted.get("raw_api_response"), dict):
        raise RepairError(f"receipt has no accepted raw API response: {path}")
    try:
        validated = validate_teacher_response(
            accepted["raw_api_response"],
            tokenizer=tokenizer,
            prompt_messages=row["messages"],
            max_student_tokens=MAX_STUDENT_TOKENS,
        )
    except TeacherGenerationError as exc:
        raise RepairError(f"receipt response no longer validates: {path}: {exc}") from exc
    if validated["final_answer"] != receipt.get("final_answer") or validated[
        "student_target_tokens_including_eos"
    ] != receipt.get("student_target_tokens_including_eos"):
        raise RepairError(f"receipt derived fields mismatch its raw response: {path}")
    return receipt


def _repair_identity(
    *,
    train_path: Path,
    heldout_path: Path,
    original_identity: Mapping[str, Any],
    model_identity: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "teacher": {
            "model": TEACHER_MODEL,
            "revision": TEACHER_REVISION,
            "identity": dict(model_identity),
        },
        "student": {
            "model": STUDENT_MODEL_ID,
            "revision": STUDENT_REVISION,
            "local_snapshot": str(Path(STUDENT_MODEL).resolve()),
        },
        "decoder": {
            "reasoning_effort": "medium",
            "temperature": 0.7,
            "top_p": 0.95,
            "max_tokens": MAX_TOKENS,
            "max_student_tokens_including_eos": MAX_STUDENT_TOKENS,
        },
        "word_limit_profiles": list(WORD_LIMITS),
        "instructions": [repair_instruction(limit) for limit in WORD_LIMITS],
        "sources": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256_file(train_path)},
            "heldout": {
                "path": str(heldout_path.resolve()),
                "sha256": sha256_file(heldout_path),
            },
        },
        "original_cache_identity_sha256": original_identity["sha256"],
        "repair_code_sha256": sha256_file(__file__),
    }
    identity["sha256"] = sha256_json(identity)
    return identity


async def repair_and_assemble(
    *,
    train_path: Path,
    heldout_path: Path,
    run_root: Path,
    backend: Any,
    tokenizer: Any,
    concurrency: int,
) -> dict[str, Any]:
    if concurrency < 1:
        raise RepairError("concurrency must be positive")
    train, heldout = load_and_validate_splits(train_path, heldout_path)
    if len(train) != EXPECTED_TRAIN or len(heldout) != EXPECTED_HELDOUT:
        raise RepairError("unexpected source split sizes")
    original_identity = _load_original_identity(run_root)
    original_dir = run_root / "teacher" / "accepted"
    final_root = run_root / "teacher_final"
    repair_dir = final_root / "repair" / "accepted"
    expected_names = {receipt_name(row["prompt_id"], row["prompt_hash"]) for row in train}
    unmatched_original = {path.name for path in original_dir.glob("*.json")} - expected_names
    unmatched_repair = {path.name for path in repair_dir.glob("*.json")} - expected_names
    if unmatched_original or unmatched_repair:
        raise RepairError(
            "receipt directories contain prompt IDs outside the training source: "
            f"original={sorted(unmatched_original)}, repair={sorted(unmatched_repair)}"
        )
    original_receipts: dict[str, tuple[Path, dict[str, Any]]] = {}
    missing: list[dict[str, Any]] = []
    for row in train:
        path = original_dir / receipt_name(row["prompt_id"], row["prompt_hash"])
        if path.is_file():
            receipt = _validated_receipt(
                path,
                row=row,
                expected_identity_sha256=original_identity["sha256"],
                tokenizer=tokenizer,
            )
            original_receipts[row["prompt_id"]] = (path, receipt)
        else:
            missing.append(row)

    model_identity = await backend.model_identity()
    repair_identity = _repair_identity(
        train_path=train_path,
        heldout_path=heldout_path,
        original_identity=original_identity,
        model_identity=model_identity,
    )
    semaphore = asyncio.Semaphore(concurrency)

    async def repair_one(row: Mapping[str, Any]) -> tuple[Path, dict[str, Any]] | None:
        path = repair_dir / receipt_name(row["prompt_id"], row["prompt_hash"])
        if path.is_file():
            return path, _validated_receipt(
                path,
                row=row,
                expected_identity_sha256=repair_identity["sha256"],
                tokenizer=tokenizer,
            )
        attempts: list[dict[str, Any]] = []
        async with semaphore:
            for attempt, word_limit in enumerate(WORD_LIMITS, 1):
                seed = deterministic_seed("repair:" + row["prompt_id"], attempt)
                request_messages = repair_request_messages(row["messages"], word_limit)
                try:
                    raw = await backend.complete(
                        messages=request_messages, seed=seed, max_tokens=MAX_TOKENS
                    )
                    validated = validate_teacher_response(
                        raw,
                        tokenizer=tokenizer,
                        prompt_messages=row["messages"],
                        max_student_tokens=MAX_STUDENT_TOKENS,
                    )
                    attempts.append(
                        {
                            "attempt": attempt,
                            "profile": f"repair_{word_limit}_words",
                            "word_limit": word_limit,
                            "seed": seed,
                            "max_tokens": MAX_TOKENS,
                            "request_messages": request_messages,
                            "status": "accepted",
                            "raw_api_response": dict(raw),
                        }
                    )
                    receipt = {
                        "schema_version": 1,
                        "status": "accepted",
                        "cache_identity_sha256": repair_identity["sha256"],
                        "prompt_id": row["prompt_id"],
                        "prompt_hash": row["prompt_hash"],
                        "source_row_sha256": row["source_row_sha256"],
                        "original_messages": row["messages"],
                        "final_answer": validated["final_answer"],
                        "reasoning": validated["reasoning"],
                        "finish_reason": validated["finish_reason"],
                        "student_target_tokens_including_eos": validated[
                            "student_target_tokens_including_eos"
                        ],
                        "accepted_attempt": attempt,
                        "generation_profile": f"repair_{word_limit}_words",
                        "attempts": attempts,
                    }
                    write_json_atomic(path, receipt)
                    return path, receipt
                except Exception as exc:  # noqa: BLE001 - retain API and validation failures
                    attempts.append(
                        {
                            "attempt": attempt,
                            "profile": f"repair_{word_limit}_words",
                            "word_limit": word_limit,
                            "seed": seed,
                            "max_tokens": MAX_TOKENS,
                            "request_messages": request_messages,
                            "status": "rejected",
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_api_response": dict(raw) if "raw" in locals() else None,
                        }
                    )
                    if "raw" in locals():
                        del raw
        write_json_atomic(
            final_root / "repair" / "failed" / receipt_name(row["prompt_id"], row["prompt_hash"]),
            {
                "schema_version": 1,
                "status": "failed",
                "cache_identity_sha256": repair_identity["sha256"],
                "prompt_id": row["prompt_id"],
                "prompt_hash": row["prompt_hash"],
                "source_row_sha256": row["source_row_sha256"],
                "original_messages": row["messages"],
                "attempts": attempts,
            },
            immutable=False,
        )
        return None

    repaired_results = await asyncio.gather(*(repair_one(row) for row in missing))
    repaired = {
        row["prompt_id"]: result
        for row, result in zip(missing, repaired_results)
        if result is not None
    }
    if len(original_receipts) + len(repaired) != EXPECTED_TRAIN:
        write_json_atomic(
            final_root / "progress.json",
            {
                "status": "partial",
                "original_accepted": len(original_receipts),
                "repair_accepted": len(repaired),
                "repair_failed": len(missing) - len(repaired),
                "required": EXPECTED_TRAIN,
                "repair_identity": repair_identity,
            },
            immutable=False,
        )
        raise RepairError(f"{len(missing) - len(repaired)} rows remain unrepaired")

    output_rows: list[dict[str, Any]] = []
    profile_counts: dict[str, int] = {}
    for row in train:
        if row["prompt_id"] in original_receipts:
            path, receipt = original_receipts[row["prompt_id"]]
            profile = "original_250_words"
        else:
            path, receipt = repaired[row["prompt_id"]]
            profile = receipt["generation_profile"]
        profile_counts[profile] = profile_counts.get(profile, 0) + 1
        output_rows.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt_hash": row["prompt_hash"],
                "messages": [
                    *row["messages"],
                    {"role": "assistant", "content": receipt["final_answer"]},
                ],
                "teacher": {
                    "model": TEACHER_MODEL,
                    "revision": TEACHER_REVISION,
                    "reasoning_effort": "medium",
                    "generation_profile": profile,
                    "student_target_tokens_including_eos": receipt[
                        "student_target_tokens_including_eos"
                    ],
                    "receipt": {
                        "path": str(path.resolve()),
                        "sha256": sha256_file(path),
                    },
                },
            }
        )
    output_path = final_root / "train.jsonl"
    write_jsonl_atomic(output_path, output_rows)
    combined_identity = {
        "teacher": {"model": TEACHER_MODEL, "revision": TEACHER_REVISION},
        "original_cache_identity_sha256": original_identity["sha256"],
        "repair_cache_identity_sha256": repair_identity["sha256"],
    }
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "count": EXPECTED_TRAIN,
        "cache_identity": combined_identity,
        "train_jsonl": {
            "path": str(output_path.resolve()),
            "sha256": sha256_file(output_path),
            "bytes": output_path.stat().st_size,
        },
        "sources": repair_identity["sources"],
        "student": repair_identity["student"],
        "generation_profiles": {
            "original_250_words": {
                "instruction": original_identity["request_instruction"],
                "cache_identity_sha256": original_identity["sha256"],
            },
            **{
                f"repair_{limit}_words": {
                    "instruction": repair_instruction(limit),
                    "cache_identity_sha256": repair_identity["sha256"],
                }
                for limit in WORD_LIMITS
            },
        },
        "generation_profile_counts": profile_counts,
        "reference_answers_or_rubrics_sent_to_teacher": False,
        "answer_reasoning_exported": False,
    }
    write_json_atomic(final_root / "manifest.json", manifest)
    write_json_atomic(
        final_root / "progress.json",
        {
            "status": "complete",
            "original_accepted": len(original_receipts),
            "repair_accepted": len(repaired),
            "repair_failed": 0,
            "required": EXPECTED_TRAIN,
            "repair_identity": repair_identity,
        },
        immutable=False,
    )
    return manifest


def _load_tokenizer() -> Any:
    snapshot = Path(STUDENT_MODEL)
    if not snapshot.is_dir() or snapshot.name != STUDENT_REVISION:
        raise RepairError("pinned local student tokenizer snapshot is unavailable")
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--expected-train", type=int, default=EXPECTED_TRAIN)
    parser.add_argument("--expected-heldout", type=int, default=EXPECTED_HELDOUT)
    return parser.parse_args()


async def _main() -> None:
    args = parse_args()
    settings = TeacherSettings(reasoning_effort="medium")
    global EXPECTED_TRAIN, EXPECTED_HELDOUT
    if args.expected_train < 1 or args.expected_heldout < 1:
        raise RepairError("expected split sizes must be positive")
    EXPECTED_TRAIN = args.expected_train
    EXPECTED_HELDOUT = args.expected_heldout
    load_and_validate_splits.__globals__["EXPECTED_TRAIN"] = EXPECTED_TRAIN
    load_and_validate_splits.__globals__["EXPECTED_HELDOUT"] = EXPECTED_HELDOUT

    result = await repair_and_assemble(
        train_path=args.train,
        heldout_path=args.heldout,
        run_root=args.run_root.resolve(),
        backend=OpenAIBackend(base_url=args.base_url, settings=settings),
        tokenizer=_load_tokenizer(),
        concurrency=args.concurrency,
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
