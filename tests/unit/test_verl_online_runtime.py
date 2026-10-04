from __future__ import annotations

from types import SimpleNamespace

import pytest

from dynamic_rubric.artifacts import read_json
from dynamic_rubric.providers.base import GenerationResult
from dynamic_rubric.training.online_contracts import OnlineStepManifest, StepState
from dynamic_rubric.training.online_step import OnlineHookResult
from dynamic_rubric.training.verl_online_runtime import (
    OnlineRuntimeFactoryError,
    VerlOnlineRewardRuntime,
    create_online_reward_runtime,
)


class FakeTokenizer:
    def batch_decode(self, values, skip_special_tokens=True):
        return [f"current-{index}" for index in range(len(values))]


class FakeControl:
    model = "actor"
    revision = "rev"

    def preflight(self):
        return {"served_model": "actor", "revision": "rev", "checkpoint_hash": "a0"}

    def generate_many(self, request, count):
        return [
            GenerationResult(
                text=f"control-{index}",
                requested_model="actor",
                returned_model="actor",
                request_id=str(index),
                created_at=0,
                retry_count=0,
            )
            for index in range(count)
        ]


def runtime(tmp_path):
    return VerlOnlineRewardRuntime(
        trainer=SimpleNamespace(
            tokenizer=FakeTokenizer(),
            config=SimpleNamespace(
                trainer=SimpleNamespace(default_local_dir=str(tmp_path / "checkpoints"))
            ),
        ),
        coordinator=object(),
        control_generator=FakeControl(),
        artifact_root=tmp_path / "online_steps",
        run_id="run",
        actor_model="actor",
        actor_revision="rev",
        control_hash="a0",
        control_concurrency=2,
        seed=7,
    )


def batch(*, tensor: bool = False):
    extra = {
        "prompt_id": "p",
        "source_row_id": "source",
        "prompt_occurrence_id": "occ",
        "prompt_messages": [{"role": "user", "content": "question"}],
        "offline_criteria": [
            {"criterion_id": "c-positive", "text": "be correct", "weight": 10},
            {"criterion_id": "c-negative", "text": "avoid harm", "weight": -3},
        ],
    }
    if tensor:
        torch = pytest.importorskip("torch")
        batch_values = {
            "responses": torch.ones((16, 4), dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]] * 16),
        }
    else:
        batch_values = {"responses": [[1, 1, 1, 0]] * 16, "attention_mask": [[1, 1, 1, 0]] * 16}
    return SimpleNamespace(
        batch=batch_values,
        non_tensor_batch={
            "uid": ["uid"] * 16,
            "extra_info": [extra] * 16,
        },
        meta_info={},
    )


def test_dataproto_adapter_builds_exact_current_and_pi_ref_inventories(tmp_path) -> None:
    step_input = runtime(tmp_path)._step_input(batch(), step=1)

    assert len(step_input.prompt_groups) == 1
    group = step_input.prompt_groups[0]
    assert [item.rollout_index for item in group.current_responses] == list(range(16))
    assert [item.rollout_index for item in group.control_responses] == list(range(8))
    assert group.control_responses[0].policy.policy_version == 0
    assert [item.weight for item in group.offline_criteria] == [10, -3]


def test_batch_identity_does_not_depend_on_ephemeral_verl_uids(tmp_path) -> None:
    first = batch()
    second = batch()
    second.non_tensor_batch["uid"] = ["new-process-uid"] * 16

    first_input = runtime(tmp_path)._step_input(first, step=1)
    second_input = runtime(tmp_path)._step_input(second, step=1)

    assert first_input.batch_uid == second_input.batch_uid


def test_dataproto_adapter_rejects_mixed_verl_group_uids(tmp_path) -> None:
    value = batch()
    value.non_tensor_batch["uid"][-1] = "different-group"

    with pytest.raises(Exception, match="mixes veRL group uids"):
        runtime(tmp_path)._step_input(value, step=1)



def test_apply_hook_result_places_scalar_only_on_last_valid_token(tmp_path) -> None:
    torch = pytest.importorskip("torch")
    value = batch(tensor=True)
    hook = OnlineHookResult(
        optimizer_update_index=1,
        rm_scores=tuple(float(index) for index in range(16)),
        trace_refs=tuple(f"trace-{index}" for index in range(16)),
        manifest_hash="manifest",
        sealed=True,
    )

    updated = runtime(tmp_path).apply_hook_result(value, hook)

    scores = updated.batch["rm_scores"]
    assert torch.all(scores[:, :2] == 0)
    assert torch.all(scores[:, 3] == 0)
    assert scores[:, 2].tolist() == [float(index) for index in range(16)]
    assert updated.meta_info["online_step_sealed"] is True
    assert len(updated.non_tensor_batch["online_trace_ref"]) == 16


def test_commit_requires_and_hashes_complete_per_step_checkpoint(tmp_path) -> None:
    value = runtime(tmp_path)
    manifest = OnlineStepManifest(
        schema_version=1,
        run_id="run",
        optimizer_update_index=1,
        batch_uid="batch",
        state=StepState.PRE_UPDATE_SEALED,
        prompt_occurrence_ids=("occ",),
        current_response_count=16,
        control_response_count=8,
        extraction_count=8,
        dedup_count=1,
        grader_count=16,
        reward_count=16,
        artifacts={},
    )
    value._prepared[1] = SimpleNamespace(manifest=manifest, sealed=True)
    checkpoint = tmp_path / "checkpoints" / "global_step_1"
    checkpoint.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"dataloader")
    actor = checkpoint / "actor"
    actor.mkdir()
    (actor / "model.bin").write_bytes(b"actor")

    value.commit_step(global_step=1, checkpoint_dir=str(checkpoint), trainer=object())

    commit = read_json(tmp_path / "online_steps" / "step-000001" / "commit.json")
    latest = read_json(tmp_path / "latest_commit.json")
    assert commit["state"] == "committed"
    assert len(commit["artifacts"]["logical_policy_token"]) == 64
    assert len(commit["artifacts"]["actor_parameter_hash"]) == 64
    assert len(commit["artifacts"]["resume_checkpoint_hash"]) == 64
    assert latest["checkpoint_saved"] is True
    assert latest["logical_policy_token"] == commit["artifacts"]["logical_policy_token"]
    assert latest["optimizer_update_index"] == 1


def test_factory_requires_explicit_frozen_pi_ref() -> None:
    config = SimpleNamespace(
        reward={"online_step_runtime": {"control_policy": "pi_ref", "artifact_root": "/tmp/x"}}
    )
    with pytest.raises(OnlineRuntimeFactoryError, match="frozen_control"):
        create_online_reward_runtime(config=config, trainer=object())


def test_factory_rehashes_local_a0_before_control_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"different-a0")
    for name, value in {
        "OPENAI_API_KEY": "test-key",
        "ONLINE_EXTRACTOR_MODEL": "o3-mini",
        "ONLINE_GRADER_MODEL": "gpt-4.1-mini",
        "ONLINE_ACTOR_MODEL": "actor",
        "ONLINE_ACTOR_REVISION": "revision",
        "ONLINE_CONTROL_CHECKPOINT_HASH": "0" * 64,
        "MODEL_PATH": str(snapshot),
    }.items():
        monkeypatch.setenv(name, value)
    config = SimpleNamespace(
        reward={
            "online_step_runtime": {
                "control_policy": "pi_ref",
                "frozen_control": True,
                "artifact_root": str(tmp_path / "online_steps"),
            }
        }
    )

    with pytest.raises(OnlineRuntimeFactoryError, match="local A0 actor snapshot differs"):
        create_online_reward_runtime(config=config, trainer=object())



def test_factory_fails_pi_old_closed_before_provider_construction() -> None:
    config = SimpleNamespace(
        reward={"online_step_runtime": {"control_policy": "pi_old", "artifact_root": "/tmp/x"}}
    )
    with pytest.raises(OnlineRuntimeFactoryError, match="bounded one-batch"):
        create_online_reward_runtime(config=config, trainer=object())


def _manifest(step: int) -> OnlineStepManifest:
    return OnlineStepManifest(
        schema_version=1,
        run_id="run",
        optimizer_update_index=step,
        batch_uid=f"batch-{step}",
        state=StepState.PRE_UPDATE_SEALED,
        prompt_occurrence_ids=("occ",),
        current_response_count=16,
        control_response_count=8,
        extraction_count=8,
        dedup_count=1,
        grader_count=16,
        reward_count=16,
        artifacts={},
    )


def test_step_two_rejects_missing_or_malformed_prior_logical_token(tmp_path) -> None:
    value = runtime(tmp_path)
    with pytest.raises(Exception, match="latest_commit"):
        value._current_policy_hash(2)
    write_json = __import__(
        "dynamic_rubric.artifacts", fromlist=["write_json_atomic"]
    ).write_json_atomic
    write_json(
        tmp_path / "latest_commit.json",
        {
            "optimizer_update_index": 1,
            "logical_policy_token": "not-a-hash",
        },
    )
    with pytest.raises(Exception, match="immutable prior commit manifest"):
        value._current_policy_hash(2)


def test_unsaved_online_commit_carries_only_logical_policy_token(tmp_path) -> None:
    value = runtime(tmp_path)
    value._prepared[1] = SimpleNamespace(manifest=_manifest(1), sealed=True)

    value.commit_step(global_step=1, checkpoint_dir=None, trainer=object())

    commit = read_json(tmp_path / "online_steps/step-000001/commit.json")
    latest = read_json(tmp_path / "latest_commit.json")
    assert latest["checkpoint_saved"] is False
    assert "checkpoint" not in latest
    assert "actor_parameter_hash" not in commit["artifacts"]
    assert len(commit["artifacts"]["logical_policy_token"]) == 64
    assert value._current_policy_hash(2) == latest["logical_policy_token"]


def test_nonsequential_commit_fails_before_publishing(tmp_path) -> None:
    value = runtime(tmp_path)
    value._prepared[3] = SimpleNamespace(manifest=_manifest(3), sealed=True)
    value._current_hashes[3] = "a" * 64
    write_json = __import__(
        "dynamic_rubric.artifacts", fromlist=["write_json_atomic"]
    ).write_json_atomic
    write_json(
        tmp_path / "latest_commit.json",
        {
            "optimizer_update_index": 1,
            "logical_policy_token": "a" * 64,
        },
    )
    with pytest.raises(Exception, match="sequential"):
        value.commit_step(
            global_step=3,
            checkpoint_dir=str(tmp_path / "checkpoints" / "global_step_3"),
            trainer=object(),
        )
    assert not (tmp_path / "online_steps" / "step-000003" / "commit.json").exists()


def test_control_identity_drift_fails_before_generation(tmp_path) -> None:
    value = runtime(tmp_path)
    value.control_generator.preflight = lambda: {
        "served_model": "actor",
        "revision": "rev",
        "checkpoint_hash": "changed",
    }
    occurrence = SimpleNamespace(
        prompt_id="p",
        prompt=({"role": "user", "content": "q"},),
        prompt_occurrence_id="occ",
    )
    with pytest.raises(Exception, match="identity drifted"):
        value._control_responses((occurrence,), step=1)


def test_checkpoint_tree_hash_streams_files_without_path_read_bytes(tmp_path, monkeypatch) -> None:
    from dynamic_rubric.training.verl_online_runtime import _directory_hash

    root = tmp_path / "tree"
    root.mkdir()
    (root / "shard.bin").write_bytes(b"x" * 1024)
    monkeypatch.setattr(
        type(root),
        "read_bytes",
        lambda self: (_ for _ in ()).throw(AssertionError("must stream")),
    )
    assert len(_directory_hash(root)) == 64


def test_between_step_checkpoint_tampering_is_detected(tmp_path) -> None:
    value = runtime(tmp_path)
    value._prepared[1] = SimpleNamespace(manifest=_manifest(1), sealed=True)
    checkpoint = tmp_path / "checkpoints" / "global_step_1"
    actor = checkpoint / "actor"
    actor.mkdir(parents=True)
    (checkpoint / "data.pt").write_bytes(b"dataloader")
    model = actor / "model.bin"
    model.write_bytes(b"actor")
    value.commit_step(global_step=1, checkpoint_dir=str(checkpoint), trainer=object())

    model.write_bytes(b"tampered")
    with pytest.raises(Exception, match="no longer matches disk"):
        value._current_policy_hash(2)
