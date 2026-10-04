from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "phase1" / "generate_evorubrics_sft_teacher.py"
SPEC = importlib.util.spec_from_file_location("generate_evorubrics_sft_teacher", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
teacher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = teacher
SPEC.loader.exec_module(teacher)


class FakeTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, **_kwargs) -> str:
        if messages[-1]["role"] == "assistant" and not add_generation_prompt:
            prefix = " ".join(message["content"] for message in messages[:-1])
            return prefix + " assistant " + messages[-1]["content"] + " EOS NEWLINE"
        return " ".join(message["content"] for message in messages) + " assistant"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        ids = [1 if token == "EOS" else index + 2 for index, token in enumerate(text.split())]
        return {"input_ids": ids}


def response(content: str, *, finish_reason: str = "stop", reasoning: str = "private") -> dict:
    return {
        "id": "fake",
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "reasoning_content": reasoning},
            }
        ],
        "usage": {"completion_tokens": 10},
    }


def test_response_boundary_counts_exact_sft_target_and_keeps_reasoning_separate() -> None:
    prompt_messages = [{"role": "user", "content": "question"}]
    accepted = teacher.validate_teacher_response(
        response("one two"),
        tokenizer=FakeTokenizer(),
        prompt_messages=prompt_messages,
        max_student_tokens=4,
    )
    assert accepted["final_answer"] == "one two"
    assert accepted["reasoning"] == "private"
    assert accepted["student_target_tokens_including_eos"] == 4
    with pytest.raises(teacher.TeacherGenerationError, match="5 student tokens"):
        teacher.validate_teacher_response(
            response("one two three"),
            tokenizer=FakeTokenizer(),
            prompt_messages=prompt_messages,
            max_student_tokens=4,
        )


@pytest.mark.parametrize(
    ("raw", "message"),
    [
        (response(""), "empty"),
        (response("answer <think>secret</think>"), "thinking/Harmony"),
        (response("answer <|channel|>analysis"), "thinking/Harmony"),
        (response("answer", finish_reason="length"), "finish_reason"),
    ],
)
def test_response_rejects_missing_truncated_or_reasoning_leak(raw: dict, message: str) -> None:
    with pytest.raises(teacher.TeacherGenerationError, match=message):
        teacher.validate_teacher_response(
            raw,
            tokenizer=FakeTokenizer(),
            prompt_messages=[{"role": "user", "content": "question"}],
        )


def _source_row(prompt_id: str, text: str) -> dict:
    messages = [{"role": "user", "content": text}]
    return {
        "prompt_id": prompt_id,
        "prompt_hash": teacher.sha256_json(messages),
        "messages": messages,
        "reference_answer": "must never be sent",
        "r0": {"criteria": [{"criterion": "must never be sent"}]},
    }


def _write_split(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_splits_require_disjoint_ids_and_valid_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(teacher, "EXPECTED_TRAIN", 1)
    monkeypatch.setattr(teacher, "EXPECTED_HELDOUT", 1)
    train_path, heldout_path = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    _write_split(train_path, [_source_row("same", "question")])
    _write_split(heldout_path, [_source_row("same", "heldout")])
    with pytest.raises(teacher.TeacherGenerationError, match="overlap"):
        teacher.load_and_validate_splits(train_path, heldout_path)
    changed = _source_row("heldout", "heldout")
    changed["messages"][0]["content"] = "changed"
    _write_split(heldout_path, [changed])
    with pytest.raises(teacher.TeacherGenerationError, match="prompt_hash"):
        teacher.load_and_validate_splits(train_path, heldout_path)


class FakeBackend:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    async def model_identity(self) -> dict:
        return {
            "requested_model": teacher.TEACHER_MODEL,
            "served_model": {"id": teacher.TEACHER_MODEL},
        }

    async def complete(self, *, messages, seed: int, max_tokens: int) -> dict:
        self.calls.append({"messages": messages, "seed": seed, "max_tokens": max_tokens})
        return self.outputs.pop(0)


def test_limit_run_retries_and_resumes_without_exporting_reference(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(teacher, "EXPECTED_TRAIN", 2)
    monkeypatch.setattr(teacher, "EXPECTED_HELDOUT", 1)
    train_path, heldout_path = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    _write_split(train_path, [_source_row("p1", "q1"), _source_row("p2", "q2")])
    _write_split(heldout_path, [_source_row("h1", "heldout")])
    run_root = tmp_path / "run"
    backend = FakeBackend([response("too long", finish_reason="length"), response("final answer")])
    settings = teacher.TeacherSettings(max_student_tokens=10)
    first = asyncio.run(
        teacher.generate_dataset(
            train_path=train_path,
            heldout_path=heldout_path,
            run_root=run_root,
            settings=settings,
            backend=backend,
            tokenizer=FakeTokenizer(),
            concurrency=1,
            limit=1,
            script_path=SCRIPT_PATH,
        )
    )
    assert first["status"] == "partial"
    assert first["counts"]["retries"] == 1
    assert len(backend.calls) == 2
    assert "reference_answer" not in json.dumps(backend.calls)
    assert "must never be sent" not in json.dumps(backend.calls)
    receipt = next((run_root / "teacher" / "accepted").glob("*.json"))
    saved = json.loads(receipt.read_text())
    assert saved["reasoning"] == "private"
    assert len(saved["attempts"]) == 2

    resumed = FakeBackend([])
    second = asyncio.run(
        teacher.generate_dataset(
            train_path=train_path,
            heldout_path=heldout_path,
            run_root=run_root,
            settings=settings,
            backend=resumed,
            tokenizer=FakeTokenizer(),
            concurrency=1,
            limit=1,
            script_path=SCRIPT_PATH,
        )
    )
    assert second["counts"]["reused"] == 1
    assert resumed.calls == []


def test_seed_and_receipt_identity_are_stable() -> None:
    assert teacher.deterministic_seed("prompt", 1) == teacher.deterministic_seed("prompt", 1)
    assert teacher.deterministic_seed("prompt", 1) != teacher.deterministic_seed("prompt", 2)
    assert teacher.receipt_name("a/b", "f" * 64) == "a_b-ffffffffffffffff.json"
