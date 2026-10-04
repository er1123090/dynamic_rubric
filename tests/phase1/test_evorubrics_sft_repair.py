from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = REPO_ROOT / "scripts" / "phase1"
sys.path.insert(0, str(SCRIPT_DIR))
SCRIPT_PATH = SCRIPT_DIR / "repair_evorubrics_sft_teacher.py"
SPEC = importlib.util.spec_from_file_location("repair_evorubrics_sft_teacher", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
repair = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = repair
SPEC.loader.exec_module(repair)


class FakeTokenizer:
    eos_token_id = 1

    def apply_chat_template(self, messages, *, add_generation_prompt: bool, **_kwargs) -> str:
        if messages[-1]["role"] == "assistant" and not add_generation_prompt:
            prefix = " ".join(message["content"] for message in messages[:-1])
            return prefix + " assistant " + messages[-1]["content"] + " EOS NEWLINE"
        return " ".join(message["content"] for message in messages) + " assistant"

    def __call__(self, text: str, *, add_special_tokens: bool) -> dict[str, list[int]]:
        assert add_special_tokens is False
        return {
            "input_ids": [
                1 if token == "EOS" else index + 2 for index, token in enumerate(text.split())
            ]
        }


def _response(content: str, finish_reason: str = "stop") -> dict:
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": {"content": content, "reasoning_content": "private"},
            }
        ]
    }


class FakeBackend:
    def __init__(self, outputs: list[dict]) -> None:
        self.outputs = outputs
        self.calls: list[dict] = []

    async def model_identity(self) -> dict:
        return {"served_model": {"id": repair.TEACHER_MODEL}}

    async def complete(self, *, messages, seed: int, max_tokens: int) -> dict:
        self.calls.append({"messages": messages, "seed": seed, "max_tokens": max_tokens})
        return self.outputs.pop(0)


def _row(prompt_id: str) -> dict:
    messages = [{"role": "user", "content": "question " + prompt_id}]
    return {
        "prompt_id": prompt_id,
        "prompt_hash": repair.sha256_json(messages),
        "messages": messages,
        "reference_answer": "never sent",
        "r0": {"criteria": ["never sent"]},
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _original_receipt(run_root: Path, row: dict, identity: dict) -> None:
    source = {
        "prompt_id": row["prompt_id"],
        "prompt_hash": row["prompt_hash"],
        "messages": row["messages"],
        "source_row_sha256": repair.sha256_json(row),
    }
    raw = _response("original")
    validated = repair.validate_teacher_response(
        raw, tokenizer=FakeTokenizer(), prompt_messages=row["messages"]
    )
    receipt = {
        "status": "accepted",
        "cache_identity_sha256": identity["sha256"],
        **source,
        "original_messages": row["messages"],
        "final_answer": "original",
        "student_target_tokens_including_eos": validated["student_target_tokens_including_eos"],
        "accepted_attempt": 1,
        "attempts": [{"attempt": 1, "status": "accepted", "raw_api_response": raw}],
    }
    path = (
        run_root
        / "teacher"
        / "accepted"
        / repair.receipt_name(row["prompt_id"], row["prompt_hash"])
    )
    repair.write_json_atomic(path, receipt)


def test_repair_rejects_truncated_then_assembles_distinct_profiles(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(repair, "EXPECTED_TRAIN", 2)
    monkeypatch.setattr(repair, "EXPECTED_HELDOUT", 1)
    monkeypatch.setitem(
        repair.load_and_validate_splits.__globals__, "EXPECTED_TRAIN", repair.EXPECTED_TRAIN
    )
    monkeypatch.setitem(
        repair.load_and_validate_splits.__globals__, "EXPECTED_HELDOUT", repair.EXPECTED_HELDOUT
    )
    rows, heldout = [_row("p1"), _row("p2")], [_row("h1")]
    train_path, heldout_path = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    _write_jsonl(train_path, rows)
    _write_jsonl(heldout_path, heldout)
    run_root = tmp_path / "run"
    original_identity = {
        "teacher": {"model": repair.TEACHER_MODEL, "revision": repair.TEACHER_REVISION},
        "request_instruction": "original instruction",
    }
    original_identity["sha256"] = repair.sha256_json(original_identity)
    repair.write_json_atomic(
        run_root / "teacher" / "progress.json",
        {"cache_identity": original_identity},
    )
    _original_receipt(run_root, rows[0], original_identity)
    backend = FakeBackend([_response("bad", "length"), _response("repaired")])
    manifest = asyncio.run(
        repair.repair_and_assemble(
            train_path=train_path,
            heldout_path=heldout_path,
            run_root=run_root,
            backend=backend,
            tokenizer=FakeTokenizer(),
            concurrency=1,
        )
    )
    assert manifest["status"] == "complete"
    assert manifest["generation_profile_counts"] == {
        "original_250_words": 1,
        "repair_180_words": 1,
    }
    assert "250 words" in backend.calls[0]["messages"][0]["content"]
    assert "180 words" in backend.calls[1]["messages"][0]["content"]
    assert "never sent" not in json.dumps(backend.calls)
    final_rows = [
        json.loads(line)
        for line in (run_root / "teacher_final" / "train.jsonl").read_text().splitlines()
    ]
    assert [row["prompt_id"] for row in final_rows] == ["p1", "p2"]
    assert len({row["prompt_id"] for row in final_rows}) == 2
    assert all("sha256" in row["teacher"]["receipt"] for row in final_rows)


def test_repair_refuses_unmatched_or_tampered_original_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(repair, "EXPECTED_TRAIN", 1)
    monkeypatch.setattr(repair, "EXPECTED_HELDOUT", 1)
    monkeypatch.setitem(
        repair.load_and_validate_splits.__globals__, "EXPECTED_TRAIN", repair.EXPECTED_TRAIN
    )
    monkeypatch.setitem(
        repair.load_and_validate_splits.__globals__, "EXPECTED_HELDOUT", repair.EXPECTED_HELDOUT
    )
    row, heldout = _row("p1"), _row("h1")
    train_path, heldout_path = tmp_path / "train.jsonl", tmp_path / "heldout.jsonl"
    _write_jsonl(train_path, [row])
    _write_jsonl(heldout_path, [heldout])
    run_root = tmp_path / "run"
    identity = {
        "teacher": {"model": repair.TEACHER_MODEL, "revision": repair.TEACHER_REVISION},
        "request_instruction": "original",
    }
    identity["sha256"] = repair.sha256_json(identity)
    repair.write_json_atomic(run_root / "teacher" / "progress.json", {"cache_identity": identity})
    _original_receipt(run_root, row, identity)
    extra = run_root / "teacher" / "accepted" / "not-in-source.json"
    repair.write_json_atomic(extra, {})
    with pytest.raises(repair.RepairError, match="outside the training source"):
        asyncio.run(
            repair.repair_and_assemble(
                train_path=train_path,
                heldout_path=heldout_path,
                run_root=run_root,
                backend=FakeBackend([]),
                tokenizer=FakeTokenizer(),
                concurrency=1,
            )
        )
    extra.unlink()
    receipt_path = next((run_root / "teacher" / "accepted").glob("*.json"))
    receipt = repair.read_json(receipt_path)
    receipt["prompt_hash"] = "tampered"
    repair.write_json_atomic(receipt_path, receipt, immutable=False)
    with pytest.raises(repair.RepairError, match="source identity mismatch"):
        asyncio.run(
            repair.repair_and_assemble(
                train_path=train_path,
                heldout_path=heldout_path,
                run_root=run_root,
                backend=FakeBackend([]),
                tokenizer=FakeTokenizer(),
                concurrency=1,
            )
        )
