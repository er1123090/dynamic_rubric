#!/usr/bin/env python3
"""Generate a resumable, provenance-bound GPT-OSS SFT teacher dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from dynamic_rubric.artifacts import read_json, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_bytes, sha256_file, sha256_json

TEACHER_MODEL = "openai/gpt-oss-120b"
TEACHER_REVISION = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
STUDENT_MODEL_ID = "Qwen/Qwen3-4B-Instruct-2507"
STUDENT_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"
STUDENT_MODEL = str(Path(__file__).resolve().parents[2] / "models/Qwen3-4B-Instruct-2507")
EXPECTED_TRAIN = 1500
EXPECTED_HELDOUT = 300
MAX_STUDENT_TOKENS = 1024
MAX_ATTEMPTS = 3
MAX_TOKEN_SCHEDULE = (4096, 6144, 8192)
TEMPERATURE = 0.7
TOP_P = 0.95
TEACHER_INSTRUCTION = (
    "Answer the medical question accurately and completely. Give only the final answer, "
    "with no chain of thought, hidden reasoning, grading rubric, or reference-answer discussion. "
    "Keep the answer concise and use no more than 250 words."
)
_LEAK = re.compile(r"<\/?think>|<\|[^>]+\|>", re.IGNORECASE)


class TeacherGenerationError(RuntimeError):
    """Raised when source, cache, model, or response validation fails."""


class TeacherBackend(Protocol):
    async def model_identity(self) -> Mapping[str, Any]: ...

    async def complete(
        self, *, messages: Sequence[Mapping[str, str]], seed: int, max_tokens: int
    ) -> Mapping[str, Any]: ...


@dataclass(frozen=True, slots=True)
class TeacherSettings:
    teacher_model: str = TEACHER_MODEL
    teacher_revision: str = TEACHER_REVISION
    student_model: str = STUDENT_MODEL
    student_revision: str = STUDENT_REVISION
    reasoning_effort: str = "medium"
    temperature: float = TEMPERATURE
    top_p: float = TOP_P
    max_student_tokens: int = MAX_STUDENT_TOKENS


class OpenAIBackend:
    def __init__(self, *, base_url: str, settings: TeacherSettings) -> None:
        from openai import AsyncOpenAI

        endpoint = base_url.rstrip("/")
        if not endpoint.endswith("/v1"):
            endpoint += "/v1"
        self._client = AsyncOpenAI(base_url=endpoint, api_key="not-needed", timeout=600.0)
        self._settings = settings
        self._endpoint = endpoint

    async def model_identity(self) -> Mapping[str, Any]:
        response = await self._client.models.list()
        raw = response.model_dump(mode="json")
        models = raw.get("data", [])
        selected = next(
            (item for item in models if item.get("id") == self._settings.teacher_model), None
        )
        if selected is None:
            found = [item.get("id") for item in models]
            raise TeacherGenerationError(
                f"teacher model {self._settings.teacher_model!r} is not served; found {found!r}"
            )
        root = selected.get("root")
        if (
            isinstance(root, str)
            and root.startswith("/")
            and self._settings.teacher_revision not in root
        ):
            raise TeacherGenerationError(
                "served teacher root does not contain the pinned revision: " + root
            )
        return {
            "endpoint": self._endpoint,
            "requested_model": self._settings.teacher_model,
            "requested_revision": self._settings.teacher_revision,
            "served_model": {
                key: selected[key]
                for key in ("id", "root", "parent", "max_model_len")
                if key in selected
            },
        }

    async def complete(
        self, *, messages: Sequence[Mapping[str, str]], seed: int, max_tokens: int
    ) -> Mapping[str, Any]:
        response = await self._client.chat.completions.create(
            model=self._settings.teacher_model,
            messages=list(messages),
            temperature=self._settings.temperature,
            top_p=self._settings.top_p,
            max_tokens=max_tokens,
            seed=seed,
            extra_body={"reasoning_effort": self._settings.reasoning_effort},
        )
        return response.model_dump(mode="json")


def _read_source(path: Path, *, expected: int, split: str) -> list[dict[str, Any]]:
    if not path.is_file():
        raise TeacherGenerationError(f"missing {split} split: {path}")
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TeacherGenerationError(
                    f"invalid JSON at {path}:{line_number}: {exc.msg}"
                ) from exc
            if not isinstance(row, dict):
                raise TeacherGenerationError(f"{split} row {line_number} is not an object")
            prompt_id = row.get("prompt_id")
            prompt_hash = row.get("prompt_hash")
            messages = row.get("messages")
            if not isinstance(prompt_id, str) or not prompt_id:
                raise TeacherGenerationError(f"{split} row {line_number} has invalid prompt_id")
            if not isinstance(prompt_hash, str) or not prompt_hash:
                raise TeacherGenerationError(f"{split} row {line_number} has invalid prompt_hash")
            if not isinstance(messages, list) or not messages:
                raise TeacherGenerationError(f"{split} row {line_number} has invalid messages")
            clean_messages: list[dict[str, str]] = []
            for message in messages:
                if not isinstance(message, dict):
                    raise TeacherGenerationError(
                        f"{split} row {line_number} has a non-object message"
                    )
                role, content = message.get("role"), message.get("content")
                if role not in {"system", "user", "assistant"}:
                    raise TeacherGenerationError(
                        f"{split} row {line_number} has invalid message role"
                    )
                if not isinstance(content, str) or not content.strip():
                    raise TeacherGenerationError(
                        f"{split} row {line_number} has invalid message content"
                    )
                clean_messages.append({"role": role, "content": content})
            if sha256_json(clean_messages) != prompt_hash:
                raise TeacherGenerationError(
                    f"{split} row {line_number} prompt_hash does not match messages"
                )
            rows.append(
                {
                    "prompt_id": prompt_id,
                    "prompt_hash": prompt_hash,
                    "messages": clean_messages,
                    "source_row_sha256": sha256_json(row),
                }
            )
    if len(rows) != expected:
        raise TeacherGenerationError(f"{split} must have {expected} rows, got {len(rows)}")
    ids = [row["prompt_id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise TeacherGenerationError(f"{split} contains duplicate prompt_id values")
    return rows


def load_and_validate_splits(
    train_path: Path, heldout_path: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = _read_source(train_path, expected=EXPECTED_TRAIN, split="train")
    heldout = _read_source(heldout_path, expected=EXPECTED_HELDOUT, split="heldout")
    overlap = {row["prompt_id"] for row in train} & {row["prompt_id"] for row in heldout}
    if overlap:
        raise TeacherGenerationError(
            f"train and heldout prompt IDs overlap ({len(overlap)} values)"
        )
    return train, heldout


def build_request_messages(messages: Sequence[Mapping[str, str]]) -> list[dict[str, str]]:
    return [{"role": "system", "content": TEACHER_INSTRUCTION}, *[dict(m) for m in messages]]


def _choice(raw: Mapping[str, Any]) -> tuple[str, str | None, str]:
    choices = raw.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        raise TeacherGenerationError("response must contain exactly one choice")
    choice = choices[0]
    if not isinstance(choice, Mapping):
        raise TeacherGenerationError("response choice is not an object")
    message = choice.get("message")
    if not isinstance(message, Mapping):
        raise TeacherGenerationError("response choice has no message")
    content = message.get("content")
    if not isinstance(content, str):
        content = ""
    reasoning = message.get("reasoning_content", message.get("reasoning"))
    if reasoning is not None and not isinstance(reasoning, str):
        reasoning = json.dumps(reasoning, ensure_ascii=False, sort_keys=True)
    finish_reason = choice.get("finish_reason")
    return content.strip(), reasoning, str(finish_reason)


def validate_teacher_response(
    raw: Mapping[str, Any],
    *,
    tokenizer: Any,
    prompt_messages: Sequence[Mapping[str, str]],
    max_student_tokens: int = MAX_STUDENT_TOKENS,
) -> dict[str, Any]:
    content, reasoning, finish_reason = _choice(raw)
    if finish_reason != "stop":
        raise TeacherGenerationError(f"finish_reason is {finish_reason!r}, expected 'stop'")
    if not content:
        raise TeacherGenerationError("final answer is empty")
    if _LEAK.search(content):
        raise TeacherGenerationError("final answer contains thinking/Harmony control markup")
    template_options = {"tokenize": False, "enable_thinking": False}
    prompt_text = tokenizer.apply_chat_template(
        list(prompt_messages), add_generation_prompt=True, **template_options
    )
    complete_text = tokenizer.apply_chat_template(
        [*prompt_messages, {"role": "assistant", "content": content}],
        add_generation_prompt=False,
        **template_options,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    complete_ids = tokenizer(complete_text, add_special_tokens=False)["input_ids"]
    if complete_ids[: len(prompt_ids)] != prompt_ids:
        raise TeacherGenerationError(
            "student chat template does not preserve the generation-prompt prefix"
        )
    target_ids = complete_ids[len(prompt_ids) :]
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not target_ids or eos_token_id is None or eos_token_id not in target_ids:
        raise TeacherGenerationError("student assistant target has no EOS token")
    student_tokens = len(target_ids)
    if student_tokens > max_student_tokens:
        raise TeacherGenerationError(
            f"final answer has {student_tokens} student tokens, limit is {max_student_tokens}"
        )
    return {
        "final_answer": content,
        "reasoning": reasoning,
        "finish_reason": finish_reason,
        "student_target_tokens_including_eos": student_tokens,
    }


def deterministic_seed(prompt_id: str, attempt: int) -> int:
    digest = sha256_bytes(f"{prompt_id}:{attempt}".encode())
    return int(digest[:8], 16) & 0x7FFFFFFF


def receipt_name(prompt_id: str, prompt_hash: str) -> str:
    safe_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", prompt_id)[:96]
    return f"{safe_id}-{prompt_hash[:16]}.json"


def build_cache_identity(
    *,
    settings: TeacherSettings,
    train_path: Path,
    heldout_path: Path,
    model_identity: Mapping[str, Any],
    script_path: Path,
) -> dict[str, Any]:
    identity = {
        "schema_version": 1,
        "teacher": {
            "model": settings.teacher_model,
            "revision": settings.teacher_revision,
            "identity": dict(model_identity),
        },
        "student": {
            "model": STUDENT_MODEL_ID,
            "revision": settings.student_revision,
            "local_snapshot": str(Path(settings.student_model).resolve()),
        },
        "decoder": {
            "reasoning_effort": settings.reasoning_effort,
            "temperature": settings.temperature,
            "top_p": settings.top_p,
            "max_token_schedule": list(MAX_TOKEN_SCHEDULE),
            "max_attempts": MAX_ATTEMPTS,
            "max_student_tokens_including_eos": settings.max_student_tokens,
        },
        "request_instruction": TEACHER_INSTRUCTION,
        "sources": {
            "train": {"path": str(train_path.resolve()), "sha256": sha256_file(train_path)},
            "heldout": {
                "path": str(heldout_path.resolve()),
                "sha256": sha256_file(heldout_path),
            },
        },
        "generator_code_sha256": sha256_file(script_path),
    }
    identity["sha256"] = sha256_json(identity)
    return identity


async def generate_dataset(
    *,
    train_path: Path,
    heldout_path: Path,
    run_root: Path,
    settings: TeacherSettings,
    backend: TeacherBackend,
    tokenizer: Any,
    concurrency: int,
    limit: int | None,
    script_path: Path,
) -> dict[str, Any]:
    if concurrency < 1:
        raise TeacherGenerationError("concurrency must be positive")
    train, _heldout = load_and_validate_splits(train_path, heldout_path)
    if limit is not None and not 1 <= limit <= len(train):
        raise TeacherGenerationError(f"limit must be between 1 and {len(train)}")
    selected = train[:limit] if limit is not None else train
    model_identity = await backend.model_identity()
    cache_identity = build_cache_identity(
        settings=settings,
        train_path=train_path,
        heldout_path=heldout_path,
        model_identity=model_identity,
        script_path=script_path,
    )
    teacher_root = run_root / "teacher"
    accepted_dir = teacher_root / "accepted"
    failed_dir = teacher_root / "failed"
    semaphore = asyncio.Semaphore(concurrency)
    counts = {"selected": len(selected), "reused": 0, "generated": 0, "failed": 0, "retries": 0}

    async def generate_one(row: Mapping[str, Any]) -> dict[str, Any] | None:
        receipt_path = accepted_dir / receipt_name(row["prompt_id"], row["prompt_hash"])
        if receipt_path.exists():
            receipt = read_json(receipt_path)
            if receipt.get("cache_identity_sha256") != cache_identity["sha256"]:
                raise TeacherGenerationError(
                    "accepted cache identity mismatch for " + str(row["prompt_id"])
                )
            if (
                receipt.get("prompt_hash") != row["prompt_hash"]
                or receipt.get("source_row_sha256") != row["source_row_sha256"]
            ):
                raise TeacherGenerationError(f"accepted source mismatch for {row['prompt_id']}")
            counts["reused"] += 1
            return receipt

        request_messages = build_request_messages(row["messages"])
        attempts: list[dict[str, Any]] = []
        async with semaphore:
            for attempt, max_tokens in enumerate(MAX_TOKEN_SCHEDULE, 1):
                seed = deterministic_seed(row["prompt_id"], attempt)
                try:
                    raw = await backend.complete(
                        messages=request_messages, seed=seed, max_tokens=max_tokens
                    )
                    validated = validate_teacher_response(
                        raw,
                        tokenizer=tokenizer,
                        prompt_messages=row["messages"],
                        max_student_tokens=settings.max_student_tokens,
                    )
                    attempts.append(
                        {
                            "attempt": attempt,
                            "seed": seed,
                            "max_tokens": max_tokens,
                            "status": "accepted",
                            "raw_api_response": dict(raw),
                        }
                    )
                    receipt = {
                        "schema_version": 1,
                        "status": "accepted",
                        "cache_identity_sha256": cache_identity["sha256"],
                        "prompt_id": row["prompt_id"],
                        "prompt_hash": row["prompt_hash"],
                        "source_row_sha256": row["source_row_sha256"],
                        "original_messages": row["messages"],
                        "request_messages": request_messages,
                        "final_answer": validated["final_answer"],
                        "reasoning": validated["reasoning"],
                        "finish_reason": validated["finish_reason"],
                        "student_target_tokens_including_eos": validated[
                            "student_target_tokens_including_eos"
                        ],
                        "accepted_attempt": attempt,
                        "attempts": attempts,
                    }
                    write_json_atomic(receipt_path, receipt)
                    (failed_dir / receipt_path.name).unlink(missing_ok=True)
                    counts["generated"] += 1
                    counts["retries"] += attempt - 1
                    return receipt
                except Exception as exc:  # noqa: BLE001 - persist API and validation failures
                    attempts.append(
                        {
                            "attempt": attempt,
                            "seed": seed,
                            "max_tokens": max_tokens,
                            "status": "rejected",
                            "error": f"{type(exc).__name__}: {exc}",
                            "raw_api_response": dict(raw) if "raw" in locals() else None,
                        }
                    )
                    if "raw" in locals():
                        del raw
            counts["failed"] += 1
            counts["retries"] += MAX_ATTEMPTS - 1
            write_json_atomic(
                failed_dir / receipt_path.name,
                {
                    "schema_version": 1,
                    "status": "failed",
                    "cache_identity_sha256": cache_identity["sha256"],
                    "prompt_id": row["prompt_id"],
                    "prompt_hash": row["prompt_hash"],
                    "source_row_sha256": row["source_row_sha256"],
                    "original_messages": row["messages"],
                    "request_messages": request_messages,
                    "attempts": attempts,
                },
                immutable=False,
            )
            return None

    receipts = await asyncio.gather(*(generate_one(row) for row in selected))
    accepted = [receipt for receipt in receipts if receipt is not None]
    progress = {
        "schema_version": 1,
        "status": "complete" if len(accepted) == EXPECTED_TRAIN and limit is None else "partial",
        "cache_identity": cache_identity,
        "counts": counts,
        "accepted_count": len(accepted),
        "required_count": EXPECTED_TRAIN,
        "limit": limit,
    }
    write_json_atomic(teacher_root / "progress.json", progress, immutable=False)
    if counts["failed"]:
        raise TeacherGenerationError(
            f"{counts['failed']} teacher responses failed after {MAX_ATTEMPTS} attempts"
        )
    if limit is not None:
        return progress
    if len(accepted) != EXPECTED_TRAIN:
        raise TeacherGenerationError(
            f"full dataset requires {EXPECTED_TRAIN} accepted responses, got {len(accepted)}"
        )

    by_id = {receipt["prompt_id"]: receipt for receipt in accepted}
    output_rows = []
    for source in train:
        receipt = by_id[source["prompt_id"]]
        output_rows.append(
            {
                "prompt_id": source["prompt_id"],
                "prompt_hash": source["prompt_hash"],
                "messages": [
                    *source["messages"],
                    {"role": "assistant", "content": receipt["final_answer"]},
                ],
                "teacher": {
                    "model": settings.teacher_model,
                    "revision": settings.teacher_revision,
                    "reasoning_effort": settings.reasoning_effort,
                    "accepted_attempt": receipt["accepted_attempt"],
                    "student_target_tokens_including_eos": receipt[
                        "student_target_tokens_including_eos"
                    ],
                    "receipt": str(
                        accepted_dir / receipt_name(source["prompt_id"], source["prompt_hash"])
                    ),
                },
            }
        )
    train_output = teacher_root / "train.jsonl"
    write_jsonl_atomic(train_output, output_rows)
    manifest = {
        "schema_version": 1,
        "status": "complete",
        "count": EXPECTED_TRAIN,
        "cache_identity": cache_identity,
        "train_jsonl": {
            "path": str(train_output),
            "sha256": sha256_file(train_output),
            "bytes": train_output.stat().st_size,
        },
        "prompt_ids_sha256": sha256_json([row["prompt_id"] for row in output_rows]),
        "prompt_hashes_sha256": sha256_json([row["prompt_hash"] for row in output_rows]),
        "answer_reasoning_exported": False,
        "reference_answers_or_rubrics_sent_to_teacher": False,
    }
    write_json_atomic(teacher_root / "manifest.json", manifest)
    return manifest


def _load_tokenizer(settings: TeacherSettings) -> Any:
    snapshot = Path(settings.student_model).resolve()
    if not snapshot.is_dir() or snapshot.name != settings.student_revision:
        raise TeacherGenerationError(
            "student model must be the local snapshot at the pinned revision"
        )
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--heldout", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    parser.add_argument("--teacher-revision", default=TEACHER_REVISION)
    parser.add_argument("--student-model", default=STUDENT_MODEL)
    parser.add_argument("--student-revision", default=STUDENT_REVISION)
    parser.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="medium")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--expected-train", type=int, default=EXPECTED_TRAIN)
    parser.add_argument("--expected-heldout", type=int, default=EXPECTED_HELDOUT)
    return parser.parse_args()


async def _main() -> None:
    global EXPECTED_TRAIN, EXPECTED_HELDOUT
    args = parse_args()
    if args.expected_train < 1 or args.expected_heldout < 1:
        raise TeacherGenerationError("expected split sizes must be positive")
    EXPECTED_TRAIN = args.expected_train
    EXPECTED_HELDOUT = args.expected_heldout
    settings = TeacherSettings(
        teacher_model=args.teacher_model,
        teacher_revision=args.teacher_revision,
        student_model=args.student_model,
        student_revision=args.student_revision,
        reasoning_effort=args.reasoning_effort,
    )
    tokenizer = _load_tokenizer(settings)
    backend = OpenAIBackend(base_url=args.base_url, settings=settings)
    result = await generate_dataset(
        train_path=args.train,
        heldout_path=args.heldout,
        run_root=args.run_root,
        settings=settings,
        backend=backend,
        tokenizer=tokenizer,
        concurrency=args.concurrency,
        limit=args.limit,
        script_path=Path(__file__).resolve(),
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    asyncio.run(_main())
