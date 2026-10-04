from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from dynamic_rubric.artifacts import read_json, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.cli import build_parser
from dynamic_rubric.config import ConfigError, config_from_mapping, load_config
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.training import live_online
from dynamic_rubric.training.live_online import (
    LiveOnlineTrainingError,
    build_online_training_environment,
    _actor_parameter_tree_hash,
    _checkpoint_tree_hash,
    _directory_tree_hash,
    online_cost_estimate,
    preflight_online_training,
    resolve_committed_resume,
    validate_online_launch_environment,
    validate_online_step_artifact,
)
from dynamic_rubric.training.verl_adapter import VerlCapabilities


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("domain", ("medicine", "science"))
@pytest.mark.parametrize("control", ("pi_ref", "pi_old"))
def test_online_configs_pin_paper_contract(domain: str, control: str) -> None:
    config = load_config(ROOT / "configs" / f"online_{domain}_{control}.yaml", stage="train-online")
    online = config.online_training
    assert online is not None
    assert online.runtime_claim == "paper_algorithm_faithful"
    assert online.control_policy == control
    assert online.expected_updates == 45
    assert online.effective_prompt_batch_size == 96
    assert online.rollouts_per_prompt == 16
    assert online.elicitation_pairs_per_prompt == 8
    assert online_cost_estimate(online) == {
        "optimizer_updates": 45,
        "prompts": 4320,
        "extractor_calls": 34560,
        "dedup_calls": 4320,
        "grader_calls": 69120,
        "external_calls": 108000,
        "control_generations": 34560,
    }


def _online_mapping() -> dict:
    return yaml.safe_load((ROOT / "configs" / "online_medicine_pi_ref.yaml").read_text())


@pytest.mark.parametrize(
    ("field", "value", "match"),
    (
        ("rollouts_per_prompt", 8, "drifted"),
        ("criteria_scope", "cumulative", "prompt_step_ephemeral"),
        ("failure_policy", "best_effort", "fail_closed"),
        ("checkpoint_interval_steps", 1, "checkpoint_interval_steps"),
    ),
)
def test_online_config_fails_closed_on_contract_drift(field: str, value: object, match: str) -> None:
    data = _online_mapping()
    data["online_training"][field] = value
    with pytest.raises(ConfigError, match=match):
        config_from_mapping(data, stage="train-online")


def test_online_config_rejects_deduplicator_model_drift() -> None:
    data = _online_mapping()
    data["models"]["rubric_deduplicator"]["requested_model"] = "gpt-5-mini"
    with pytest.raises(ConfigError, match="model identities disagree"):
        config_from_mapping(data, stage="train-online")


def test_online_config_rejects_cap_and_evaluation_artifact() -> None:
    data = _online_mapping()
    data["online_training"]["max_online_criteria"] = 8
    with pytest.raises(ConfigError, match="forbidden|forbids"):
        config_from_mapping(data, stage="train-online")
    data = _online_mapping()
    data["online_training"]["artifact_inputs"] = ["data/rar/medicine/public/final.jsonl"]
    with pytest.raises(ConfigError, match="evaluation/replay"):
        config_from_mapping(data, stage="train-online")


def test_full_reproduction_claim_requires_exact_dataset_and_hardware() -> None:
    data = _online_mapping()
    data["online_training"]["reproduction_profile"] = "paper_full_reproduction"
    with pytest.raises(ConfigError, match="paper dataset and exact 8xH100"):
        config_from_mapping(data, stage="train-online")


def _context(tmp_path: Path) -> SimpleNamespace:
    config = load_config(ROOT / "configs" / "online_medicine_pi_ref.yaml")
    stage = tmp_path / "artifacts" / "runs" / "run" / "train-online"
    return SimpleNamespace(
        root=ROOT,
        config_path=ROOT / "configs" / "online_medicine_pi_ref.yaml",
        config=config,
        run_id="run",
        run_root=stage.parent,
        stage_root=lambda: stage,
    )


def test_online_launcher_environment_is_exact_and_has_separate_runtime(tmp_path: Path) -> None:
    context = _context(tmp_path)
    environment = build_online_training_environment(
        context,
        tmp_path / "train.parquet",
        tmp_path / "development.parquet",
        tmp_path / "verl-run",
    )
    online = context.config.online_training
    assert online is not None
    assert validate_online_launch_environment(online, environment)["valid"]
    assert environment["TOTAL_EPOCHS"] == "3"
    assert environment["EXPECTED_UPDATES"] == "45"
    assert environment["ONLINE_STEP_HOOK_NAME"] == "prepare_rewards"
    assert environment["ONLINE_STEP_RUNTIME_NAME"] == "create_online_reward_runtime"
    assert environment["ONLINE_STEP_HOOK_PATH"] == "pkg://dynamic_rubric.training.online_step"
    assert environment["ONLINE_STEP_RUNTIME_PATH"] == "pkg://dynamic_rubric.training.verl_online_runtime"
    assert environment["ONLINE_EXTRACTOR_REASONING_EFFORT"] == "medium"
    assert environment["ONLINE_EXTRACTOR_RETURNED_MODEL"] == "o3-mini"
    assert environment["ONLINE_GRADER_RETURNED_MODEL"] == "gpt-4.1-mini"
    assert environment["ONLINE_ACTOR_MODEL"] == "Qwen/Qwen3-4B-Instruct-2507"
    assert environment["ONLINE_ACTOR_REVISION"] == context.config.online_training.actor_revision
    assert environment["CHECKPOINT_INTERVAL_STEPS"] == "3"
    assert set(yaml.safe_load(environment["CHECKPOINT_STEPS"])) >= {3, 23, 38, 45}
    assert "TOTAL_STEPS" not in environment

    environment["ROLLOUT_N"] = "8"
    with pytest.raises(LiveOnlineTrainingError, match="drifted"):
        validate_online_launch_environment(online, environment)


def test_online_hook_modules_are_package_loader_compatible() -> None:
    online_step = (ROOT / "src/dynamic_rubric/training/online_step.py").read_text()
    runtime = (ROOT / "src/dynamic_rubric/training/verl_online_runtime.py").read_text()

    assert "from .." not in online_step
    assert "from .online_contracts" not in online_step
    assert "from .paper_reward" not in online_step
    assert "from .." not in runtime


def test_seeded_agent_does_not_mutate_batch_extra_info() -> None:
    source = (ROOT / "src/dynamic_rubric/training/seeded_agent.py").read_text()

    assert "metadata = dict(extra_info) if isinstance(extra_info, dict) else {}" in source
    assert "metadata = extra_info if isinstance(extra_info, dict) else {}" not in source


def test_online_launcher_uses_epochs_and_full_batch_hook() -> None:
    launcher = (ROOT / "scripts" / "run_online_grpo.sh").read_text()
    assert 'trainer.total_epochs="${TOTAL_EPOCHS}"' in launcher
    assert "trainer.total_training_steps" not in launcher
    assert "reward.online_step_hook.path" in launcher
    assert "reward.online_step_runtime.path" in launcher
    assert "reward.online_step_runtime.frozen_control=true" in launcher
    assert "reward.custom_reward_function" not in launcher
    assert 'trainer.save_freq="${CHECKPOINT_INTERVAL_STEPS}"' in launcher
    assert '+trainer.checkpoint_steps="${CHECKPOINT_STEPS}"' in launcher
    assert "prune_online_checkpoint_state.py" in launcher


def test_online_cli_commands_are_distinct() -> None:
    parser = build_parser()
    train = parser.parse_args(
        ["train-online", "--config", "configs/online_medicine_pi_ref.yaml", "--run-id", "x"]
    )
    resume = parser.parse_args(
        ["resume-online", "--config", "configs/online_medicine_pi_ref.yaml", "--run-id", "x"]
    )
    validate = parser.parse_args(
        ["validate-online-step", "--run-dir", "artifacts/run", "--step", "1"]
    )
    assert (train.command, resume.command, validate.command) == (
        "train-online",
        "resume-online",
        "validate-online-step",
    )


def test_validate_online_step_checks_provider_receipt_hashes(tmp_path: Path) -> None:
    step_dir = tmp_path / "online_steps" / "step-000001"
    hashes = {}
    for name in (
        "control_generation_receipts.jsonl",
        "extraction_receipts.jsonl",
        "dedup_receipts.jsonl",
        "grader_receipts.jsonl",
    ):
        path = step_dir / name
        write_jsonl_atomic(path, [{"receipt": name}])
        hashes[name] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "run_id": "run",
        "optimizer_update_index": 1,
        "batch_uid": "batch-1",
        "state": "committed",
        "prompt_occurrence_ids": ["occ-1"],
        "current_response_count": 16,
        "control_response_count": 8,
        "extraction_count": 8,
        "dedup_count": 1,
        "grader_count": 16,
        "reward_count": 16,
        "artifacts": hashes,
    }
    write_json_atomic(step_dir / "commit.json", manifest)

    result = validate_online_step_artifact(
        tmp_path, 1, require_provider_receipts=True
    )
    assert result["valid"]
    assert result["manifest"]["state"] == "committed"

    (step_dir / "grader_receipts.jsonl").write_text('{"changed":true}\n')
    with pytest.raises(LiveOnlineTrainingError, match="provider receipt hash mismatch"):
        validate_online_step_artifact(tmp_path, 1, require_provider_receipts=True)



def _write_committed_step(
    run_dir: Path,
    step: int,
    *,
    include_file_artifacts: bool = False,
    checkpoint_saved: bool = True,
) -> dict:
    step_dir = run_dir / "online_steps" / f"step-{step:06d}"
    logical_token = sha256_json(["logical-policy-version", step])
    artifacts = {"logical_policy_token": logical_token}
    pointer = {
        "schema_version": 1,
        "optimizer_update_index": step,
        "logical_policy_token": logical_token,
        "checkpoint_saved": checkpoint_saved,
    }
    if checkpoint_saved:
        checkpoint = run_dir / "checkpoints" / f"global_step_{step}"
        checkpoint.mkdir(parents=True)
        (checkpoint / "data.pt").write_bytes(f"step-{step}".encode())
        actor = checkpoint / "actor"
        actor.mkdir()
        (actor / "weights.safetensors").write_bytes(f"actor-{step}".encode())
        artifacts.update(
            {
                "resume_checkpoint_hash": _checkpoint_tree_hash(checkpoint),
                "actor_parameter_hash": _actor_parameter_tree_hash(actor),
            }
        )
        pointer.update(
            {
                "checkpoint": str(checkpoint.resolve()),
                "resume_checkpoint_hash": artifacts["resume_checkpoint_hash"],
                "actor_parameter_hash": artifacts["actor_parameter_hash"],
            }
        )
        tracker = run_dir / "checkpoints/latest_checkpointed_iteration.txt"
        tracker.write_text(str(step), encoding="utf-8")
    if include_file_artifacts:
        for name in ("current_responses.jsonl", "rubric_unions.jsonl", "rewards.jsonl"):
            path = step_dir / name
            write_jsonl_atomic(path, [{"step": step, "artifact": name}])
            artifacts[name] = sha256_file(path)
    manifest = {
        "schema_version": 1,
        "run_id": "run",
        "optimizer_update_index": step,
        "batch_uid": f"batch-{step}",
        "state": "committed",
        "prompt_occurrence_ids": [f"occ-{step}"],
        "current_response_count": 16,
        "control_response_count": 8,
        "extraction_count": 8,
        "dedup_count": 1,
        "grader_count": 16,
        "reward_count": 16,
        "artifacts": artifacts,
    }
    write_json_atomic(step_dir / "commit.json", manifest)
    pointer["manifest_hash"] = sha256_json(manifest)
    return pointer


def test_resume_recovers_missing_step_one_pointer(tmp_path: Path) -> None:
    run_dir = tmp_path / "verl-run"
    pointer = _write_committed_step(run_dir, 1)

    assert resolve_committed_resume(run_dir) == (
        Path(pointer["checkpoint"]),
        1,
    )
    assert read_json(run_dir / "latest_commit.json") == pointer


def test_resume_ignores_checkpoint_created_before_commit(tmp_path: Path) -> None:
    run_dir = tmp_path / "verl-run"
    pointer = _write_committed_step(run_dir, 1)
    write_json_atomic(run_dir / "latest_commit.json", pointer, immutable=False)
    uncommitted = run_dir / "checkpoints" / "global_step_2"
    uncommitted.mkdir(parents=True)
    (uncommitted / "data.pt").write_bytes(b"uncommitted")

    assert resolve_committed_resume(run_dir) == (Path(pointer["checkpoint"]), 1)
    assert read_json(run_dir / "latest_commit.json") == pointer


def test_resume_recovers_commit_created_before_pointer_update(tmp_path: Path) -> None:
    run_dir = tmp_path / "verl-run"
    old_pointer = _write_committed_step(run_dir, 1)
    write_json_atomic(run_dir / "latest_commit.json", old_pointer, immutable=False)
    new_pointer = _write_committed_step(run_dir, 2)

    assert resolve_committed_resume(run_dir) == (Path(new_pointer["checkpoint"]), 2)
    assert read_json(run_dir / "latest_commit.json") == new_pointer
    assert resolve_committed_resume(run_dir) == (Path(new_pointer["checkpoint"]), 2)


def _write_archive_receipt(run_dir: Path, step: int, actor_hash: str) -> None:
    write_json_atomic(
        run_dir / "checkpoint_archives" / f"global_step_{step}.json",
        {
            "checkpoint_step": step,
            "run_id": run_dir.parent.name,
            "actor_parameter_hash": actor_hash,
        },
    )


def test_resume_accepts_verified_remote_actor_for_historical_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "phase1-run" / "verl-run"
    historical = _write_committed_step(run_dir, 1)
    latest = _write_committed_step(run_dir, 2)
    actor_hash = historical["actor_parameter_hash"]
    _write_archive_receipt(run_dir, 1, actor_hash)
    actor = run_dir / "checkpoints/global_step_1/actor"
    for path in sorted(actor.rglob("*"), reverse=True):
        path.unlink() if path.is_file() else path.rmdir()
    actor.rmdir()
    monkeypatch.setattr(live_online, "verify_public_archive", lambda receipt: actor_hash)

    assert resolve_committed_resume(run_dir) == (Path(latest["checkpoint"]), 2)


def test_resume_rejects_missing_historical_actor_without_archive(tmp_path: Path) -> None:
    run_dir = tmp_path / "phase1-run" / "verl-run"
    _write_committed_step(run_dir, 1)
    _write_committed_step(run_dir, 2)
    actor = run_dir / "checkpoints/global_step_1/actor"
    for path in actor.iterdir():
        path.unlink()
    actor.rmdir()

    with pytest.raises(LiveOnlineTrainingError, match="has no archive receipt at step 1"):
        resolve_committed_resume(run_dir)


def test_resume_does_not_fallback_when_local_actor_is_corrupt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "phase1-run" / "verl-run"
    historical = _write_committed_step(run_dir, 1)
    _write_committed_step(run_dir, 2)
    _write_archive_receipt(run_dir, 1, historical["actor_parameter_hash"])
    (run_dir / "checkpoints/global_step_1/actor/weights.safetensors").write_bytes(b"corrupt")
    monkeypatch.setattr(
        live_online,
        "verify_public_archive",
        lambda receipt: historical["actor_parameter_hash"],
    )

    with pytest.raises(LiveOnlineTrainingError, match="parameter hash mismatch at step 1"):
        resolve_committed_resume(run_dir)


def test_resume_never_uses_remote_only_latest_full_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_dir = tmp_path / "phase1-run" / "verl-run"
    latest = _write_committed_step(run_dir, 1)
    actor_hash = latest["actor_parameter_hash"]
    _write_archive_receipt(run_dir, 1, actor_hash)
    actor = run_dir / "checkpoints/global_step_1/actor"
    for path in actor.iterdir():
        path.unlink()
    actor.rmdir()
    monkeypatch.setattr(live_online, "verify_public_archive", lambda receipt: actor_hash)

    with pytest.raises(LiveOnlineTrainingError, match="latest full resume checkpoint hash mismatch"):
        resolve_committed_resume(run_dir)


@pytest.mark.parametrize(
    "artifact_name",
    ("current_responses.jsonl", "rubric_unions.jsonl", "rewards.jsonl"),
)
def test_resume_rejects_mutated_file_backed_artifact(
    tmp_path: Path, artifact_name: str
) -> None:
    run_dir = tmp_path / "verl-run"
    pointer = _write_committed_step(run_dir, 1, include_file_artifacts=True)
    write_json_atomic(run_dir / "latest_commit.json", pointer, immutable=False)
    (run_dir / "online_steps" / "step-000001" / artifact_name).write_text(
        '{"mutated":true}\n'
    )

    with pytest.raises(LiveOnlineTrainingError, match=f"artifact hash mismatch: {artifact_name}"):
        resolve_committed_resume(run_dir)


def test_resume_rejects_noncontiguous_commit_chain(tmp_path: Path) -> None:
    run_dir = tmp_path / "verl-run"
    _write_committed_step(run_dir, 1)
    _write_committed_step(run_dir, 3)

    with pytest.raises(LiveOnlineTrainingError, match="noncontiguous"):
        resolve_committed_resume(run_dir)


def test_resume_rejects_malformed_commit_path(tmp_path: Path) -> None:
    run_dir = tmp_path / "verl-run"
    pointer = _write_committed_step(run_dir, 1)
    malformed = run_dir / "online_steps" / "step-1"
    malformed.mkdir()
    write_json_atomic(malformed / "commit.json", read_json(
        run_dir / "online_steps" / "step-000001" / "commit.json"
    ))
    write_json_atomic(run_dir / "latest_commit.json", pointer, immutable=False)

    with pytest.raises(LiveOnlineTrainingError, match="path is malformed"):
        resolve_committed_resume(run_dir)


def test_online_environment_preserves_control_checkpoint_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ONLINE_CONTROL_CHECKPOINT_HASH", "a" * 64)
    context = _context(tmp_path)
    environment = build_online_training_environment(
        context,
        tmp_path / "train.parquet",
        tmp_path / "development.parquet",
        tmp_path / "verl-run",
    )
    assert environment["ONLINE_CONTROL_CHECKPOINT_HASH"] == "a" * 64


def test_resume_archives_logical_tail_and_selects_latest_full_checkpoint(tmp_path: Path) -> None:
    run_dir = tmp_path / "verl-run"
    full = _write_committed_step(run_dir, 1)
    logical_tail = _write_committed_step(run_dir, 2, checkpoint_saved=False)
    write_json_atomic(run_dir / "latest_commit.json", logical_tail, immutable=False)

    selected, step = resolve_committed_resume(run_dir)

    assert (selected, step) == (Path(full["checkpoint"]), 1)
    assert read_json(run_dir / "latest_commit.json") == full
    archived = list(
        (run_dir / "replay_archive/resume-from-000001").glob(
            "attempt-*/online_steps/step-000002/commit.json"
        )
    )
    assert len(archived) == 1
    assert not (run_dir / "online_steps/step-000002").exists()

    context = _context(tmp_path)
    environment = build_online_training_environment(
        context,
        tmp_path / "train.parquet",
        tmp_path / "development.parquet",
        run_dir,
        resume=True,
    )
    assert environment["RESUME_MODE"] == "resume_path"
    assert environment["RESUME_FROM_PATH"] == str(Path(full["checkpoint"]))


def test_pi_old_preflight_is_explicitly_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    from dynamic_rubric.pipeline import PipelineContext

    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    context = PipelineContext.create(
        ROOT,
        ROOT / "configs" / "online_medicine_pi_old.yaml",
        "train-online",
        "pi-old-preflight-test",
    )
    with pytest.raises(LiveOnlineTrainingError, match="not production-ready.*bounded"):
        preflight_online_training(context)


def _online_ready_capabilities() -> VerlCapabilities:
    return VerlCapabilities(
        checkout="checkout",
        revision="revision",
        custom_reward_hook=True,
        raw_validation_export=True,
        focal_checkpoint_retention=True,
        probe_patch_required=False,
        stage5_patch_applied=True,
        patch_sha256="static-patch",
        online_patch_applied=True,
        online_patch_sha256="online-patch",
    )


@pytest.mark.parametrize("supplied_hash", (None, "0" * 64))
def test_pi_ref_preflight_requires_exact_local_a0_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    supplied_hash: str | None,
) -> None:
    context = _context(tmp_path)
    snapshot = tmp_path / "a0"
    snapshot.mkdir()
    (snapshot / "weights.safetensors").write_bytes(b"a0")
    context.config.models["policy"]["local_snapshot"] = str(snapshot)
    monkeypatch.setattr(live_online, "dependency_gate", lambda *_: _online_ready_capabilities())
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("ONLINE_CONTROL_URL", "http://control.invalid")
    if supplied_hash is None:
        monkeypatch.delenv("ONLINE_CONTROL_CHECKPOINT_HASH", raising=False)
        match = "requires ONLINE_CONTROL_CHECKPOINT_HASH"
    else:
        monkeypatch.setenv("ONLINE_CONTROL_CHECKPOINT_HASH", supplied_hash)
        match = "does not match models.policy.local_snapshot"

    with pytest.raises(LiveOnlineTrainingError, match=match):
        preflight_online_training(context)


def test_pi_ref_preflight_binds_local_a0_hash_to_control_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    context = _context(tmp_path)
    context.public_root = ROOT / "data" / "rar" / "medicine" / "public"
    snapshot = tmp_path / "a0"
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"model":"a0"}\n')
    (snapshot / "weights.safetensors").write_bytes(b"a0-weights")
    expected_hash = _directory_tree_hash(snapshot)
    context.config.models["policy"]["local_snapshot"] = str(snapshot)
    captured: dict[str, str] = {}

    class _ControlIdentity:
        def __init__(
            self,
            _url: str,
            _model: str,
            _revision: str,
            _tokenizer_revision: str,
            *,
            launch_spec_path: Path | None,
            expected_checkpoint_hash: str,
        ) -> None:
            del launch_spec_path
            captured["checkpoint_hash"] = expected_checkpoint_hash

        def preflight(self) -> dict[str, str]:
            return {"checkpoint_hash": captured["checkpoint_hash"], "identity_source": "test"}

    monkeypatch.setattr(live_online, "dependency_gate", lambda *_: _online_ready_capabilities())
    monkeypatch.setattr(live_online, "VLLMPolicyGenerator", _ControlIdentity)
    monkeypatch.setenv("OPENAI_API_KEY", "test-only")
    monkeypatch.setenv("ONLINE_CONTROL_URL", "http://control.invalid")
    monkeypatch.setenv("ONLINE_CONTROL_CHECKPOINT_HASH", expected_hash)

    result = preflight_online_training(context)

    assert captured["checkpoint_hash"] == expected_hash
    assert result["control_identity"]["checkpoint_hash"] == expected_hash
    assert read_json(context.stage_root() / "online_preflight.json")["control_identity"][
        "checkpoint_hash"
    ] == expected_hash
