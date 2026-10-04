from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "phase1" / "train_evorubrics_sft.py"
SPEC = importlib.util.spec_from_file_location("train_evorubrics_sft", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class FakeTokenizer:
    eos_token_id = 99

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False
        assert enable_thinking is False
        if add_generation_prompt:
            return "PROMPT|"
        return "PROMPT|ANSWER|EOS"

    def __call__(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        values = {
            "PROMPT|": [1, 2, 3],
            "PROMPT|ANSWER|EOS": [1, 2, 3, 7, 8, 99],
        }
        return {"input_ids": values[text]}


def teacher_record():
    messages = [{"role": "user", "content": "question"}]
    return {
        "prompt_id": "p1",
        "prompt_hash": MODULE.sha256_json(messages),
        "messages": messages + [{"role": "assistant", "content": "answer"}],
    }


def test_assistant_only_mask_uses_strict_chat_template_boundary_and_eos() -> None:
    example = MODULE.tokenize_teacher_record(
        teacher_record(), FakeTokenizer(), row_number=1, max_length=10
    )
    assert example.input_ids == (1, 2, 3, 7, 8, 99)
    assert example.prompt_tokens == 3
    assert example.target_tokens == 3
    assert example.labels == (-100, -100, -100, 7, 8, 99)


def test_tokenizer_prefix_mismatch_fails_instead_of_guessing_boundary() -> None:
    class MismatchedTokenizer(FakeTokenizer):
        def __call__(self, text, *, add_special_tokens):
            result = super().__call__(text, add_special_tokens=add_special_tokens)
            if text != "PROMPT|":
                return {"input_ids": [1, 42, 3, 7, 8, 99]}
            return result

    tokenizer = MismatchedTokenizer()
    with pytest.raises(MODULE.SFTError, match="generation-prompt prefix"):
        MODULE.tokenize_teacher_record(teacher_record(), tokenizer, row_number=1, max_length=10)


def test_overlength_example_is_rejected_without_truncation() -> None:
    with pytest.raises(MODULE.SFTError, match="refusing silent truncation"):
        MODULE.tokenize_teacher_record(
            teacher_record(), FakeTokenizer(), row_number=1, max_length=5
        )


def test_token_normalization_is_exact_for_partial_effective_batch() -> None:
    batches = list(MODULE.effective_batch_indices(5, batch_size=3, epochs=1, seed=11))
    assert [len(indices) for _, indices in batches] == [3, 2]
    counts = [2, 6]
    weights = MODULE.token_normalization_weights(counts)
    assert weights == (0.25, 0.75)
    losses = [4.0, 8.0]
    assert sum(loss * weight for loss, weight in zip(losses, weights)) == pytest.approx(7.0)


def test_each_epoch_has_every_example_once_and_includes_final_partial() -> None:
    batches = list(MODULE.effective_batch_indices(7, batch_size=3, epochs=2, seed=11))
    assert [len(indices) for _, indices in batches] == [3, 3, 1, 3, 3, 1]
    for epoch in (0, 1):
        flattened = [
            index for batch_epoch, indices in batches if batch_epoch == epoch for index in indices
        ]
        assert sorted(flattened) == list(range(7))


def test_teacher_manifest_binds_complete_gptoss_dataset(tmp_path: Path) -> None:
    data_path = tmp_path / "train.jsonl"
    data_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.json"
    manifest = {
        "status": "complete",
        "count": 1,
        "cache_identity": {
            "teacher": {
                "model": MODULE.TEACHER_MODEL,
                "revision": MODULE.TEACHER_REVISION,
            }
        },
        "train_jsonl": {
            "sha256": MODULE.sha256_file(data_path),
            "bytes": data_path.stat().st_size,
        },
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    proof = MODULE.validate_teacher_manifest(data_path, expected_examples=1)
    assert proof["teacher_model"] == MODULE.TEACHER_MODEL
    assert proof["train_jsonl_sha256"] == MODULE.sha256_file(data_path)

    manifest["cache_identity"]["teacher"]["revision"] = "wrong"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(MODULE.SFTError, match="pinned GPT-OSS-120B revision"):
        MODULE.validate_teacher_manifest(data_path, expected_examples=1)

    manifest["cache_identity"]["teacher"]["revision"] = MODULE.TEACHER_REVISION
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    data_path.write_text("changed\n", encoding="utf-8")
    with pytest.raises(MODULE.SFTError, match="train_jsonl hash"):
        MODULE.validate_teacher_manifest(data_path, expected_examples=1)


def test_output_file_manifests_persist_each_exact_hash(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "checkpoint" / "adapter"
    merged_dir = tmp_path / "merged_model"
    adapter_dir.mkdir(parents=True)
    merged_dir.mkdir()
    adapter_file = adapter_dir / "adapter_model.safetensors"
    merged_file = merged_dir / "model.safetensors"
    adapter_file.write_bytes(b"adapter")
    merged_file.write_bytes(b"merged")

    adapter_files, merged_files = MODULE.persist_output_file_manifests(
        tmp_path, adapter_dir, merged_dir
    )
    assert json.loads((tmp_path / "adapter_files.json").read_text()) == adapter_files
    assert json.loads((tmp_path / "merged_model_files.json").read_text()) == merged_files
    assert adapter_files[0]["sha256"] == MODULE.sha256_file(adapter_file)
    assert merged_files[0]["sha256"] == MODULE.sha256_file(merged_file)


def test_default_base_model_is_the_pinned_local_snapshot(tmp_path: Path) -> None:
    args = MODULE.build_parser().parse_args(
        ["--data", str(tmp_path / "data.jsonl"), "--run-root", str(tmp_path / "run")]
    )
    assert args.base_model == MODULE.BASE_MODEL_SNAPSHOT


def test_cli_help_does_not_import_gpu_stack() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--help"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--data" in result.stdout
    assert "--run-root" in result.stdout
    assert "--gpu" in result.stdout
