from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.horizon.checkpoint_kl import _encode_float32
from dynamic_rubric.phase1.audit_policy import (
    AuditPolicyError,
    export_checkpoint,
    generate_probe_pools,
    inspect_checkpoint,
    load_run_contract,
    publish_probe_prompts,
    summarize_sampled_policy_distance,
)
from dynamic_rubric.providers.base import GenerationResult


def _production_fixture(tmp_path: Path) -> Path:
    run_dir = tmp_path / "phase1-run"
    run_dir.mkdir()
    train = tmp_path / "train.jsonl"
    prompts = [
        {
            "prompt_id": f"prompt-{index:03d}",
            "source_row_id": f"source-{index:03d}",
            "messages": [{"role": "user", "content": f"question {index}"}],
        }
        for index in range(100)
    ]
    write_jsonl_atomic(train, prompts)
    probe = tmp_path / "fixed_train_probe.json"
    prompt_ids = [row["prompt_id"] for row in prompts]
    write_json_atomic(
        probe,
        {
            "prompt_ids": prompt_ids,
            "prompt_ids_sha256": sha256_json(prompt_ids),
            "source_sha256": sha256_file(train),
        },
    )
    config = {
        "domain": "medicine",
        "method": "online_rubrics",
        "seed": 11,
        "models": {
            "policy": {
                "model": "Qwen/Qwen3-4B-Instruct-2507",
                "revision": "base-revision",
                "tokenizer_revision": "base-revision",
            }
        },
        "data": {
            "train_path": str(train),
            "fixed_train_probe": {"count": 100},
        },
    }
    write_json_atomic(run_dir / "config.resolved.json", config)
    launch = {
        "domain": "medicine",
        "method": "online_rubrics",
        "primary_seed": 11,
        "run_id": run_dir.name,
        "models": {"policy": "Qwen/Qwen3-4B-Instruct-2507"},
        "fixed_probe_manifest": str(probe),
        "fixed_probe_manifest_sha256": sha256_file(probe),
    }
    write_json_atomic(run_dir / "launch_spec.json", launch)
    actor = run_dir / "verl-run/checkpoints/global_step_0/actor"
    (actor / "huggingface").mkdir(parents=True)
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"checkpoint-zero")
    return run_dir


class _FakePolicyGenerator:
    calls: list[int] = []

    def __init__(
        self,
        base_url: str,
        model: str,
        revision: str,
        tokenizer_revision: str,
        **kwargs: Any,
    ) -> None:
        self.model = model
        self.revision = revision
        self.tokenizer_revision = tokenizer_revision
        self.checkpoint_hash = kwargs["expected_checkpoint_hash"]

    def preflight(self) -> dict[str, Any]:
        return {
            "served_model": self.model,
            "model_revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "checkpoint_hash": self.checkpoint_hash,
            "thinking": False,
        }

    def generate(self, request: Any) -> GenerationResult:
        self.calls.append(request.seed)
        return GenerationResult(
            text=f"answer-{request.seed}",
            requested_model=self.model,
            returned_model=self.model,
            request_id=f"request-{request.seed}",
            created_at=1,
            retry_count=0,
            usage={"dynamic_rubric_logical_seed": request.seed},
            raw_response_hash=sha256_json(request.seed),
        )


def test_step_zero_probe_a_b_are_disjoint_seeded_and_resumable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = _production_fixture(tmp_path)
    contract = load_run_contract(run_dir)
    _FakePolicyGenerator.calls = []
    monkeypatch.setattr(
        "dynamic_rubric.phase1.audit_policy.VLLMPolicyGenerator", _FakePolicyGenerator
    )

    first = generate_probe_pools(
        contract,
        step=0,
        output_root=tmp_path / "audit",
        base_url="http://policy",
        concurrency=8,
    )
    assert first["reused"] is False
    a = read_jsonl(tmp_path / "audit/responses/checkpoint-000000/probe_A.jsonl")
    b = read_jsonl(tmp_path / "audit/responses/checkpoint-000000/probe_B.jsonl")
    assert len(a) == 800
    assert len(b) == 1600
    assert {row["response_id"] for row in a}.isdisjoint({row["response_id"] for row in b})
    assert {row["logical_seed"] for row in a + b} == set(_FakePolicyGenerator.calls)
    assert all(row["usage"]["dynamic_rubric_logical_seed"] == row["logical_seed"] for row in a + b)
    assert {row["policy_checkpoint"] for row in a + b} == {0}
    assert {row["pool"] for row in a} == {"probe_A"}
    assert {row["pool_family"] for row in b} == {"pool_b"}
    prompt_path = publish_probe_prompts(contract, tmp_path / "audit")
    assert len(read_jsonl(prompt_path)) == 100

    calls = len(_FakePolicyGenerator.calls)
    second = generate_probe_pools(
        contract,
        step=0,
        output_root=tmp_path / "audit",
        base_url="http://unreachable",
    )
    assert second["reused"] is True
    assert len(_FakePolicyGenerator.calls) == calls


def test_pool_a_first_then_b_preserves_responses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = load_run_contract(_production_fixture(tmp_path))
    _FakePolicyGenerator.calls = []
    monkeypatch.setattr("dynamic_rubric.phase1.audit_policy.VLLMPolicyGenerator", _FakePolicyGenerator)
    kwargs = dict(step=0, output_root=tmp_path / "audit", base_url="http://policy")
    generate_probe_pools(contract, **kwargs, pools=("probe_A",))
    root = tmp_path / "audit/responses/checkpoint-000000"
    original_a = (root / "probe_A.jsonl").read_bytes()
    assert len(_FakePolicyGenerator.calls) == 800
    assert not (root / "probe_B.jsonl").exists()
    assert read_json(root / "provenance.json")["pool_a_b_disjoint"] is None
    generate_probe_pools(contract, **kwargs, pools=("probe_B",))
    assert len(_FakePolicyGenerator.calls) == 2400
    assert (root / "probe_A.jsonl").read_bytes() == original_a
    assert read_json(root / "provenance.json")["pool_a_b_disjoint"] is True
    result = generate_probe_pools(contract, **kwargs)
    assert result["reused"] is True
    assert len(_FakePolicyGenerator.calls) == 2400


def test_initial_policy_pool_b_does_not_generate_pool_a(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    contract = load_run_contract(_production_fixture(tmp_path))
    _FakePolicyGenerator.calls = []
    monkeypatch.setattr("dynamic_rubric.phase1.audit_policy.VLLMPolicyGenerator", _FakePolicyGenerator)
    generate_probe_pools(contract, step=0, output_root=tmp_path / "audit", base_url="http://policy", pools=("probe_B",))
    root = tmp_path / "audit/responses/checkpoint-000000"
    assert len(_FakePolicyGenerator.calls) == 1600
    assert not (root / "probe_A.jsonl").exists()
    assert read_json(root / "provenance.json")["selected_pools"] == ["probe_B"]


def test_export_is_source_hash_bound_and_atomic(tmp_path: Path) -> None:
    contract = load_run_contract(_production_fixture(tmp_path))

    def runner(command: list[str], **kwargs: Any) -> None:
        target = Path(command[command.index("--target_dir") + 1])
        (target / "config.json").write_text("{}", encoding="utf-8")
        (target / "model-00001-of-00001.safetensors").write_bytes(b"weights")

    exported = export_checkpoint(
        contract,
        step=0,
        export_root=tmp_path / "exports",
        merger_python="runtime-python",
        runner=runner,
    )
    manifest = read_json(exported / "audit_export_manifest.json")
    assert manifest["source_model_sha256"] == inspect_checkpoint(contract, 0).source_model_sha256
    assert (
        export_checkpoint(
            contract,
            step=0,
            export_root=tmp_path / "exports",
            runner=lambda *_args, **_kwargs: pytest.fail("must reuse export"),
        )
        == exported
    )

    checkpoint = contract.run_dir / (
        "verl-run/checkpoints/global_step_0/actor/model_world_size_1_rank_0.pt"
    )
    checkpoint.write_bytes(b"changed")
    with pytest.raises(AuditPolicyError, match="provenance mismatch"):
        export_checkpoint(contract, step=0, export_root=tmp_path / "exports")


def _score_row(
    *, policy_step: int, prompt_id: str, response_id: str, values: list[float]
) -> dict[str, Any]:
    return {
        "policy_step": policy_step,
        "pool_policy_step": 3,
        "prompt_id": prompt_id,
        "response_id": response_id,
        "sample_index": int(response_id.rsplit("-", 1)[1]),
        "response_token_count": len(values),
        "response_token_hash": f"tokens-{response_id}",
        "response_token_logprobs_f32le_b64": _encode_float32(values),
    }


def test_sampled_policy_distance_uses_same_current_pool_b_tokens(tmp_path: Path) -> None:
    score_dir = tmp_path / "scores"
    score_dir.mkdir()
    stale = []
    current = []
    for prompt_id in ("p1", "p2"):
        for sample in range(2):
            response_id = f"{prompt_id}-{sample}"
            stale.append(
                _score_row(
                    policy_step=0,
                    prompt_id=prompt_id,
                    response_id=response_id,
                    values=[-2.0, -3.0],
                )
            )
            current.append(
                _score_row(
                    policy_step=3,
                    prompt_id=prompt_id,
                    response_id=response_id,
                    values=[-1.75, -2.75],
                )
            )
    write_jsonl_atomic(score_dir / "policy-step-0_pool-step-3.jsonl", stale)
    write_jsonl_atomic(score_dir / "policy-step-3_pool-step-3.jsonl", current)

    summary = summarize_sampled_policy_distance(
        score_dir=score_dir,
        stale_policy_step=0,
        current_policy_step=3,
        response_policy_step=3,
        output_dir=tmp_path / "distance",
    )
    assert summary["prompt_balanced_sampled_kl_mean"] == pytest.approx(0.25)
    assert summary["token_weighted_sampled_kl_mean"] == pytest.approx(0.25)
    assert summary["prompt_count"] == 2
    assert summary["response_count"] == 4
    rows = read_jsonl(tmp_path / "distance/policy_distance_response.jsonl")
    assert {row["response_id"] for row in rows} == {row["response_id"] for row in current}
    assert all(row["response_policy_checkpoint"] == 3 for row in rows)
