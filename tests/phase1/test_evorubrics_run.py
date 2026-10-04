import copy
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynamic_rubric.phase1 import evorubrics_observer as observer
from dynamic_rubric.phase1.config import (
    Phase1ConfigError,
    load_phase1_config,
    validate_phase1_mapping,
)
from dynamic_rubric.phase1.evorubrics_run import build_training_config, execute

ROOT = Path(__file__).resolve().parents[2]


def test_upstream_evaluator_bounds_response_and_uses_runtime_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module_path = ROOT / "environment/upstream/EvoRubrics/evorubric-main/llm_evaluator.py"
    spec = importlib.util.spec_from_file_location("phase1_evorubrics_llm_evaluator", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    captured = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured["request"] = kwargs
            return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="[]"))])

    class FakeOpenAI:
        def __init__(self, *, timeout, **kwargs):
            captured["timeout"] = timeout
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setenv("LLM_EVALUATOR_TIMEOUT_SECONDS", "600")
    monkeypatch.setenv("LLM_EVALUATOR_MAX_COMPLETION_TOKENS", "4096")
    monkeypatch.setattr(module, "OpenAI", FakeOpenAI)
    evaluator = module.LLMEvaluator(
        api_key="EMPTY",
        base_url="http://judge/v1",
        model="openai/gpt-oss-120b",
    )

    assert evaluator._call_api([{"role": "user", "content": "grade"}]) == "[]"
    assert captured["timeout"] == 600.0
    assert captured["request"]["max_tokens"] == 4096


def test_combines_rq2_exposure_with_evo_optimization(tmp_path):
    config = load_phase1_config(ROOT / "configs/phase1/medicine_evorubrics.yaml")
    full = build_training_config(
        config, repo_root=ROOT, run_root=tmp_path, train_path=tmp_path / "train.json"
    )
    assert full["data"]["train_batch_size"] == 96
    assert full["trainer"]["total_epochs"] == 3
    assert full["trainer"]["total_training_steps"] == 48
    assert full["trainer"]["save_freq"] == 1
    assert full["trainer"]["save_lora_freq"] == 1
    assert config.checkpoint_steps == tuple(range(1, 49))
    assert full["data"]["num_answers"] == full["data"]["num_rubrics"] == 4
    actor = full["actor_rollout_ref"]["actor"]
    assert actor["optim"]["policy_llm_lr"] == 2e-5
    assert actor["optim"]["rubrics_generator_lr"] == 5e-6
    assert actor["kl_loss_coef"] == 1e-4
    assert actor["ppo_micro_batch_size_per_gpu"] == 1
    assert full["adversarial"]["reflect_use_golden_rubrics"] is True
    assert full["adversarial"]["answer_aware_num_refs"] == 0
    assert full["trainer"]["eval"]["enabled"] is False
    assert full["data"]["filter_overlong_prompts"] is False
    assert config.method_config["pool_b_count"] == 16
    smoke = build_training_config(
        config, repo_root=ROOT, run_root=tmp_path, train_path=tmp_path / "train.json", smoke=True
    )
    assert smoke["data"]["train_max_samples"] == 2
    assert smoke["trainer"]["total_epochs"] == 1
    assert smoke["adversarial"] == full["adversarial"]


def test_custom_evo_yaml_values_reach_upstream_config(tmp_path: Path) -> None:
    base = load_phase1_config(ROOT / "configs/launch/medicine_evorubric.yaml")
    raw = copy.deepcopy(base.raw)
    raw["launch"]["tuning_mode"] = "custom"
    raw["infrastructure"]["optimizer"]["gpus"] = [0]
    evo = raw["evorubrics"]
    evo["lora_rank"] = 16
    evo["lora_alpha"] = 32
    evo["policy_learning_rate"] = 1e-5
    evo["rubric_generator_learning_rate"] = 3e-6
    evo["kl_loss_coefficient"] = 0.002
    evo["generation_temperature"] = 0.8
    evo["reward_weights"] = {
        "similarity": 0.4,
        "discrimination": 0.3,
        "diversity": 0.2,
        "reflect": 0.1,
    }
    config = validate_phase1_mapping(raw, source_path=base.source_path)
    assert config.raw["infrastructure"]["optimizer"]["gpus"] == [0]
    upstream = build_training_config(
        config, repo_root=ROOT, run_root=tmp_path, train_path=tmp_path / "train.json"
    )
    assert upstream["actor_rollout_ref"]["model"]["lora_rank"] == 16
    assert upstream["actor_rollout_ref"]["actor"]["optim"]["policy_llm_lr"] == 1e-5
    assert upstream["actor_rollout_ref"]["actor"]["optim"]["rubrics_generator_lr"] == 3e-6
    assert upstream["actor_rollout_ref"]["actor"]["kl_loss_coef"] == 0.002
    assert upstream["actor_rollout_ref"]["rollout"]["temperature"] == 0.8
    assert upstream["adversarial"]["similarity_weight"] == 0.4
    raw["evorubrics"]["reward_weights"]["similarity"] = 0.5
    with pytest.raises(Phase1ConfigError, match="must sum to 1"):
        validate_phase1_mapping(raw, source_path=base.source_path)


def test_paper_mode_rejects_changed_evo_learning_rate() -> None:
    base = load_phase1_config(ROOT / "configs/launch/medicine_evorubric.yaml")
    raw = copy.deepcopy(base.raw)
    raw["evorubrics"]["policy_learning_rate"] = 1e-5
    with pytest.raises(Phase1ConfigError, match="Evo policy_learning_rate"):
        validate_phase1_mapping(raw, source_path=base.source_path)


def test_full_launch_cannot_use_synthetic_smoke_proof(tmp_path):
    (tmp_path / "launch_spec.json").write_text(
        json.dumps({"mode": "full", "phase1_config_sha256": "abc"})
    )
    with pytest.raises(ValueError, match="live smoke"):
        execute(tmp_path, ROOT, Path("missing-python"), "http://localhost:1")
    proof = tmp_path / "proof.json"
    proof.write_text(json.dumps({"status": "synthetic", "phase1_config_sha256": "abc"}))
    with pytest.raises(ValueError, match="does not validate"):
        execute(tmp_path, ROOT, Path("missing-python"), "http://localhost:1", smoke_proof=proof)


def trainer(tmp_path, step=0):
    return SimpleNamespace(
        global_steps=step,
        config=SimpleNamespace(
            rq2=SimpleNamespace(run_root=str(tmp_path), domain="medicine", seed=11)
        ),
    )


def payload(input_step=0):
    return {
        "input_step": input_step,
        "update_step": input_step + 1,
        "questions": ["question"],
        "prompt_ids": ["original-id"],
        "answers_per_query": [["a", "b", "c", "d"]],
        "rubrics_per_query": [["r1", "r2", "r3", "r4"]],
    }


def test_observer_preserves_preupdate_identity_and_exposure(tmp_path):
    t = trainer(tmp_path)
    observer.on_iteration(t, payload())
    t.global_steps = 1
    observer.on_step(t, {}, {})
    first = json.loads((tmp_path / "audit/train_batch/step_000001.json").read_text())
    assert first["responses"][0]["policy_checkpoint"] == 0
    assert first["responses"][0]["global_step"] == 1
    assert len({r["response_id"] for r in first["responses"]}) == 4
    resumed = trainer(tmp_path, step=1)
    observer.on_iteration(resumed, payload(1))
    resumed.global_steps = 2
    observer.on_step(resumed, {}, {})
    result = json.loads((tmp_path / "metrics/step_000002.json").read_text())
    assert result["cumulative_prompt_exposures"] == 2
    assert result["cumulative_policy_completions"] == 8
    assert result["adjacent_policy_kl"] is None


def test_observer_fails_instead_of_inventing_missing_provenance(tmp_path):
    t = trainer(tmp_path)
    data = payload()
    data.pop("prompt_ids")
    with pytest.raises(ValueError, match="original prompt IDs"):
        observer.on_iteration(t, data)
    observer.on_iteration(t, payload(1))
    t.global_steps = 2
    with pytest.raises(ValueError, match="Previous step metrics"):
        observer.on_step(t, {}, {})


def test_advantages_are_binary_with_compact_hash_manifest(tmp_path):
    torch = pytest.importorskip("torch")
    safetensors_torch = pytest.importorskip("safetensors.torch")

    t = trainer(tmp_path)
    aliased_scores = torch.tensor([[0.75, 0.5]])
    tensors = {
        "responses": torch.tensor([[1, 2]]),
        "response_mask": torch.tensor([[1, 1]]),
        "token_level_scores": aliased_scores,
        "token_level_rewards": aliased_scores,
        "advantages": torch.tensor([[0.25, -0.25]]),
        "old_log_probs": torch.tensor([[-1.0, -2.0]]),
        "ref_log_prob": torch.tensor([[-1.1, -2.1]]),
    }
    batch = SimpleNamespace(batch=tensors, non_tensor_batch={"uid": ["group-1"]})
    observer.on_advantages(t, "policy_llm", batch)
    manifest_path = tmp_path / "audit/advantages/step_000001_policy_llm.json"
    manifest = json.loads(manifest_path.read_text())
    artifact = tmp_path / manifest["artifact"]["path"]
    assert manifest["artifact"]["format"] == "safetensors"
    assert manifest["uid_count"] == 1
    assert manifest["tensor_keys"] == sorted(tensors)
    assert "values" not in manifest
    loaded = safetensors_torch.load(artifact.read_bytes())
    assert torch.equal(loaded["advantages"], tensors["advantages"])
    assert torch.equal(loaded["token_level_scores"], aliased_scores)
    assert torch.equal(loaded["token_level_rewards"], aliased_scores)


def test_resume_quarantines_only_uncommitted_later_artifacts(tmp_path):
    from dynamic_rubric.phase1.evorubrics_run import quarantine_uncommitted_after

    later = [
        tmp_path / "audit/train_batch/step_000003.json",
        tmp_path / "audit/advantages/step_000003_policy_llm.safetensors",
        tmp_path / "metrics/step_000003.json",
        tmp_path / "upstream-run/policy_llm/step_3/partial.bin",
    ]
    keep = tmp_path / "metrics/step_000002.json"
    for path in [*later, keep]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"evidence")
    result = quarantine_uncommitted_after(tmp_path, 2)
    assert len(result["artifacts"]) == len(later)
    assert keep.is_file()
    assert all(not path.exists() for path in later)
    quarantined = tmp_path / result["artifacts"][0]["quarantined"]
    assert quarantined.exists()


def test_semantic_provenance_ignores_smoke_full_scope(tmp_path, monkeypatch):
    from dynamic_rubric.phase1 import evorubrics_run

    data_manifest = tmp_path / "data.json"
    data_manifest.write_text(
        json.dumps(
            {
                "splits": {
                    name: {
                        "source": {"sha256": name + "-source"},
                        "row_count": count,
                        "ordered_prompt_ids_sha256": name + "-ids",
                    }
                    for name, count in (("train", 1500), ("heldout", 300))
                },
                "fixed_train_probe": {"prompt_count": 100, "prompt_ids_sha256": "probe-ids"},
            }
        )
    )
    lock = tmp_path / "environment/evorubrics-runtime-lock.txt"
    lock.parent.mkdir()
    lock.write_text("locked")
    code = tmp_path / "code.py"
    code.write_text("stable")
    monkeypatch.setattr(evorubrics_run, "_PROVENANCE_CODE_PATHS", ("code.py",))

    common = {
        "domain": "medicine",
        "method": "evorubrics",
        "seed": 11,
        "phase1_config_sha256": "phase",
        "source_archive_sha256": "zip",
        "dataset_manifest": str(data_manifest),
        "fixed_probe_manifest_sha256": "probe",
        "policy_model": {"model": "policy", "revision": "rev"},
        "training_responses_m": 4,
        "rubric_sets_n": 4,
        "probe_pool_b": 16,
    }
    identities = []
    for mode, steps, exposures in (("smoke", 1, 2), ("full", 48, 4500)):
        root = tmp_path / mode
        root.mkdir()
        spec = {
            **common,
            "mode": mode,
            "upstream_config_sha256": mode + "-upstream",
            "expected_steps": steps,
            "expected_prompt_exposures": exposures,
        }
        (root / "launch_spec.json").write_text(json.dumps(spec))
        provenance = evorubrics_run.build_run_provenance(
            root,
            tmp_path,
            {"id": "judge", "root": "/snapshot/fixed"},
            actual_training=False,
        )
        assert provenance["scope"]["mode"] == mode
        identities.append(provenance["semantic_identity_sha256"])
    assert identities[0] == identities[1]


def test_execute_launches_importable_module_not_main(tmp_path, monkeypatch):
    from dynamic_rubric.phase1 import evorubrics_run

    (tmp_path / "launch_spec.json").write_text(
        json.dumps({"mode": "smoke", "judge_model": "judge"})
    )
    monkeypatch.setattr(evorubrics_run, "resolve_judge_identity", lambda *_: {"id": "judge"})
    monkeypatch.setattr(
        evorubrics_run,
        "build_run_provenance",
        lambda *_args, **_kwargs: {"semantic_identity_sha256": "semantic"},
    )
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(evorubrics_run.subprocess, "run", fake_run)
    result = execute(tmp_path, ROOT, Path("runtime-python"), "http://judge")
    assert result == 0
    assert captured["command"][:3] == [
        "runtime-python",
        "-m",
        "dynamic_rubric.phase1.evorubrics_run",
    ]


def test_runtime_uses_short_bindable_ray_socket_path(tmp_path):
    import socket

    from dynamic_rubric.phase1.evorubrics_run import runtime_environment

    long_run_root = tmp_path / ("very-long-experiment-name-" * 5)
    env = runtime_environment(ROOT, long_run_root, "http://judge", gpu="1")
    assert env["LLM_EVALUATOR_TIMEOUT_SECONDS"] == "1200"
    assert env["LLM_EVALUATOR_MAX_COMPLETION_TOKENS"] == "8192"
    ray_root = Path(env["RAY_TMPDIR"])
    candidate = ray_root / "session_2026-09-10_12-34-56_123456_123456/sockets/raylet"
    candidate.parent.mkdir(parents=True, exist_ok=True)
    assert len(str(candidate).encode()) <= 107
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        server.bind(str(candidate))
    finally:
        server.close()
        candidate.unlink(missing_ok=True)
