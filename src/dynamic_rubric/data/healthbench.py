from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..artifacts import write_jsonl_atomic


GOLD_KEYS = ("rubric", "rubrics", "criteria", "gold_rubric", "physician_rubric")


class HealthBenchFormatError(ValueError):
    """Raised when a source row cannot be converted without guessing."""


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_conversation(record: Mapping[str, Any]) -> list[dict[str, str]]:
    raw = record.get("messages", record.get("conversation", record.get("prompt")))
    if isinstance(raw, str):
        raw = [{"role": "user", "content": raw}]
    if not isinstance(raw, Sequence) or isinstance(raw, (bytes, bytearray, str)):
        raise HealthBenchFormatError("row has no supported prompt/messages/conversation field")
    messages: list[dict[str, str]] = []
    for index, message in enumerate(raw):
        if isinstance(message, str):
            role, content = ("user" if index == 0 else "assistant"), message
        elif isinstance(message, Mapping):
            role = str(message.get("role", "user"))
            content = message.get("content", message.get("text"))
            if not isinstance(content, str):
                raise HealthBenchFormatError(f"message {index} has non-text content")
        else:
            raise HealthBenchFormatError(f"message {index} has unsupported shape")
        messages.append({"role": role.strip().lower(), "content": content.strip()})
    if not messages:
        raise HealthBenchFormatError("conversation is empty")
    return messages


def stable_prompt_id(messages: Sequence[Mapping[str, str]]) -> str:
    return "hb_" + _sha256(_canonical_bytes(list(messages)))[:24]


def extract_gold(record: Mapping[str, Any]) -> Any:
    for key in GOLD_KEYS:
        if key in record:
            return record[key]
    raise HealthBenchFormatError("row has no physician-rubric field")


def _atomic_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    write_jsonl_atomic(path, rows)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise HealthBenchFormatError(f"line {line_number} is not an object")
            rows.append(value)
    return rows


def prepare_healthbench_source(
    source: Path,
    public_output: Path,
    private_output: Path,
) -> dict[str, Any]:
    """Physically separate public conversations from physician annotations.

    The returned manifest contains source and public hashes only. Private hashes
    remain private and are intentionally absent from public artifacts.
    """

    source_bytes = source.read_bytes()
    public_rows: list[dict[str, Any]] = []
    private_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in load_jsonl(source):
        messages = canonical_conversation(record)
        prompt_id = stable_prompt_id(messages)
        if prompt_id in seen:
            raise HealthBenchFormatError(f"duplicate canonical prompt: {prompt_id}")
        seen.add(prompt_id)
        public_rows.append(
            {
                "prompt_id": prompt_id,
                "messages": messages,
                "source": "healthbench_consensus",
            }
        )
        private_rows.append({"prompt_id": prompt_id, "gold_rubric": extract_gold(record)})
    _atomic_jsonl(public_output, public_rows)
    _atomic_jsonl(private_output, private_rows)
    return {
        "source_sha256": _sha256(source_bytes),
        "public_sha256": _sha256(public_output.read_bytes()),
        "row_count": len(public_rows),
    }


def _gold_fragments(private_rows: Iterable[Mapping[str, Any]]) -> set[str]:
    fragments: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, str) and len(value.strip()) >= 12:
            text = value.strip()
            fragments.add(text)
            fragments.add(_sha256(text.encode()))
        elif isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for nested in value:
                visit(nested)

    for row in private_rows:
        visit(row.get("gold_rubric"))
    return fragments


def scan_public_outputs_for_gold(public_paths: Iterable[Path], private_gt: Path) -> None:
    """Fail closed if a public artifact contains physician criterion text/hash."""

    fragments = _gold_fragments(load_jsonl(private_gt))
    for path in public_paths:
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for fragment in fragments:
            if fragment and fragment in text:
                raise PermissionError(f"gold-rubric leakage detected in public file: {path}")
