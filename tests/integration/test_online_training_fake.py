from __future__ import annotations

import json
import threading
from pathlib import Path
from types import SimpleNamespace

from dynamic_rubric.artifacts import read_json, read_jsonl
from dynamic_rubric.providers.base import GenerationRequest, GenerationResult
from dynamic_rubric.training.online_contracts import StepState, validate_online_step_manifest
from dynamic_rubric.training.online_step import OnlineStepCoordinator, prepare_online_rewards
from dynamic_rubric.training.verl_online_runtime import VerlOnlineRewardRuntime


class _Tokenizer:
    def batch_decode(self, values, skip_special_tokens=True):
        del skip_special_tokens
        return [f"current answer {index}" for index in range(len(values))]


class _PiRefControl:
    model = "actor"
    revision = "revision"

    def __init__(self) -> None:
        self.request_count = 0
        self.response_count = 0

    def preflight(self):
        return {
            "served_model": self.model,
            "revision": self.revision,
            "checkpoint_hash": "a0",
        }

    def generate_many(self, request: GenerationRequest, count: int):
        self.request_count += 1
        self.response_count += count
        return tuple(
            GenerationResult(
                text=f"control answer {index}",
                requested_model=self.model,
                returned_model=self.model,
                request_id=f"control-{index}",
                created_at=0,
                retry_count=0,
                raw_response_hash=f"{index:064x}",
            )
            for index in range(count)
        )


class _ScriptedRubricProvider:
    def __init__(self, model: str) -> None:
        self.model = model
        self.calls: list[str] = []
        self._lock = threading.Lock()

    def generate(self, request: GenerationRequest) -> GenerationResult:
        with self._lock:
            self.calls.append(request.family)
        if request.family == "online_rubric_extraction":
            payload = {
                "analysis": "The current answer explains the mechanism more clearly.",
                "new_criteria": [
                    {
                        "quote": "current answer",
                        "criterion": "Explain the causal mechanism",
                        "weight": 2,
                    }
                ],
            }
        elif request.family == "online_rubric_dedup":
            payload = {
                "analysis": "All candidates express the same criterion.",
                "final_criteria": [{"criterion": "Explain the causal mechanism", "weight": 2}],
            }
        else:
            payload = {key: "PRESENT" for key in request.json_schema["required"]}
        text = json.dumps(payload)
        return GenerationResult(
            text=text,
            requested_model=self.model,
            returned_model=self.model,
            request_id=f"{request.family}-{len(self.calls)}",
            created_at=0,
            retry_count=0,
            raw_response_hash=None,
        )


def _batch():
    extra = {
        "prompt_id": "prompt",
        "source_row_id": "source-row",
        "prompt_occurrence_id": "occurrence",
        "prompt_messages": [{"role": "user", "content": "Explain the observation."}],
        "offline_criteria": [
            {"criterion_id": "offline-positive", "text": "Answer directly", "weight": 3},
            {"criterion_id": "offline-negative", "text": "Use unsupported claims", "weight": -1},
        ],
    }
    return SimpleNamespace(
        batch={

            "responses": [[1, 1, 0]] * 16,
            "attention_mask": [[1, 1, 0]] * 16,
        },
        non_tensor_batch={"uid": ["uid"] * 16, "extra_info": [extra] * 16},
        meta_info={},
    )


def _two_prompt_batch():
    rows = []
    uids = []
    for prompt_index in range(2):
        extra = {
            "prompt_id": f"prompt-{prompt_index}",
            "source_row_id": f"source-{prompt_index}",
            "prompt_occurrence_id": f"occurrence-{prompt_index}",
            "prompt_messages": [
                {"role": "user", "content": f"Explain observation {prompt_index}."}
            ],
            "offline_criteria": [
                {"criterion_id": f"positive-{prompt_index}", "text": "Answer directly", "weight": 3},
                {"criterion_id": f"negative-{prompt_index}", "text": "Use unsupported claims", "weight": -1},
            ],
        }
        rows.extend([extra] * 16)
        uids.extend([f"uid-{prompt_index}"] * 16)
    return SimpleNamespace(
        batch={"responses": [[1, 1, 0]] * 32, "attention_mask": [[1, 1, 0]] * 32},
        non_tensor_batch={"uid": uids, "extra_info": rows},
        meta_info={},
    )



def test_fake_pi_ref_online_step_seals_rewards_and_checkpoint(tmp_path: Path) -> None:
    extractor = _ScriptedRubricProvider("o3-mini")
    grader = _ScriptedRubricProvider("gpt-4.1-mini")
    coordinator = OnlineStepCoordinator(
        extractor=extractor,
        deduper=extractor,
        grader=grader,
        extractor_model="o3-mini",
        grader_model="gpt-4.1-mini",
        extractor_concurrency=4,
        grader_concurrency=8,
        extractor_reasoning_effort="medium",
    )
    control = _PiRefControl()
    checkpoint_root = tmp_path / "checkpoints"
    runtime = VerlOnlineRewardRuntime(
        trainer=SimpleNamespace(
            tokenizer=_Tokenizer(),
            config=SimpleNamespace(trainer=SimpleNamespace(default_local_dir=str(checkpoint_root))),
        ),
        coordinator=coordinator,
        control_generator=control,
        artifact_root=tmp_path / "online_steps",
        run_id="fake-run",
        actor_model="actor",
        actor_revision="revision",
        control_hash="a0",
        control_concurrency=2,
        seed=17,
    )

    hook = prepare_online_rewards(_batch(), step=1, runtime=runtime)

    assert hook.sealed is True
    assert hook.optimizer_update_index == 1
    assert len(hook.rm_scores) == 16
    assert len(hook.trace_refs) == 16
    assert control.request_count == 1 and control.response_count == 8
    assert extractor.calls.count("online_rubric_extraction") == 8
    assert extractor.calls.count("online_rubric_dedup") == 1
    assert grader.calls.count("online_rubric_grading") == 16

    step_root = tmp_path / "online_steps" / "step-000001"
    sealed = validate_online_step_manifest(step_root / "pre_update_seal.json")
    assert sealed.state is StepState.PRE_UPDATE_SEALED
    assert sealed.content_hash == hook.manifest_hash
    assert len(read_jsonl(step_root / "control_generation_receipts.jsonl")) == 8
    assert len(read_jsonl(step_root / "extraction_receipts.jsonl")) == 8
    assert len(read_jsonl(step_root / "dedup_receipts.jsonl")) == 1
    assert len(read_jsonl(step_root / "grader_receipts.jsonl")) == 16
    assert len(read_jsonl(step_root / "rewards.jsonl")) == 16
    assert "control_generation_receipts.jsonl" in sealed.artifacts

    checkpoint = checkpoint_root / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"optimizer+dataloader")
    actor = checkpoint / "actor"
    actor.mkdir()
    (actor / "model.bin").write_bytes(b"updated actor")

    runtime.commit_step(global_step=1, checkpoint_dir=str(checkpoint), trainer=object())

    committed = validate_online_step_manifest(step_root / "commit.json")
    latest = read_json(tmp_path / "latest_commit.json")
    assert committed.state is StepState.COMMITTED
    assert len(committed.artifacts["logical_policy_token"]) == 64
    assert len(committed.artifacts["resume_checkpoint_hash"]) == 64
    assert len(committed.artifacts["actor_parameter_hash"]) == 64
    assert latest["optimizer_update_index"] == 1
    assert latest["checkpoint_saved"] is True
    assert latest["checkpoint"] == str(checkpoint)
    assert latest["resume_checkpoint_hash"] == committed.artifacts["resume_checkpoint_hash"]
    assert latest["manifest_hash"] == committed.content_hash


def test_fake_pi_ref_two_prompts_two_steps_preserve_exact_inventories(tmp_path: Path) -> None:
    extractor = _ScriptedRubricProvider("o3-mini")
    grader = _ScriptedRubricProvider("gpt-4.1-mini")
    coordinator = OnlineStepCoordinator(
        extractor=extractor,
        deduper=extractor,
        grader=grader,
        extractor_model="o3-mini",
        grader_model="gpt-4.1-mini",
        extractor_concurrency=4,
        grader_concurrency=8,
        extractor_reasoning_effort="medium",
    )
    control = _PiRefControl()
    checkpoint_root = tmp_path / "checkpoints"
    runtime = VerlOnlineRewardRuntime(
        trainer=SimpleNamespace(
            tokenizer=_Tokenizer(),
            config=SimpleNamespace(
                trainer=SimpleNamespace(default_local_dir=str(checkpoint_root))
            ),
        ),
        coordinator=coordinator,
        control_generator=control,
        artifact_root=tmp_path / "online_steps",
        run_id="fake-run-2x2",
        actor_model="actor",
        actor_revision="revision",
        control_hash="a0",
        control_concurrency=2,
        seed=17,
    )

    first = prepare_online_rewards(_two_prompt_batch(), step=1, runtime=runtime)
    assert len(first.rm_scores) == 32
    checkpoint_one = checkpoint_root / "global_step_1"
    checkpoint_one.mkdir(parents=True)
    (checkpoint_one / "data.pt").write_bytes(b"step-one-state")
    actor_one = checkpoint_one / "actor"
    actor_one.mkdir()
    (actor_one / "model.bin").write_bytes(b"actor-one")
    runtime.commit_step(global_step=1, checkpoint_dir=str(checkpoint_one), trainer=object())
    first_latest = read_json(tmp_path / "latest_commit.json")

    second = prepare_online_rewards(_two_prompt_batch(), step=2, runtime=runtime)
    assert len(second.rm_scores) == 32
    second_batch = read_json(tmp_path / "online_steps" / "step-000002" / "batch.json")
    assert (
        second_batch["current_policy"]["content_hash"]
        == first_latest["logical_policy_token"]
    )
    checkpoint_two = checkpoint_root / "global_step_2"
    checkpoint_two.mkdir(parents=True)
    (checkpoint_two / "data.pt").write_bytes(b"step-two-state")
    actor_two = checkpoint_two / "actor"
    actor_two.mkdir()
    (actor_two / "model.bin").write_bytes(b"actor-two")
    runtime.commit_step(global_step=2, checkpoint_dir=str(checkpoint_two), trainer=object())

    assert control.request_count == 4
    assert control.response_count == 32
    assert extractor.calls.count("online_rubric_extraction") == 32
    assert extractor.calls.count("online_rubric_dedup") == 4
    assert grader.calls.count("online_rubric_grading") == 64
    for step in (1, 2):
        root = tmp_path / "online_steps" / f"step-{step:06d}"
        manifest = validate_online_step_manifest(root / "commit.json")
        assert manifest.current_response_count == 32
        assert manifest.control_response_count == 16
        assert manifest.extraction_count == 16
        assert manifest.dedup_count == 2
        assert manifest.grader_count == 32
        assert manifest.reward_count == 32
    assert read_json(tmp_path / "latest_commit.json")["optimizer_update_index"] == 2

