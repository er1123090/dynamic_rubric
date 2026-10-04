"""Prepare and execute the public EvoRubrics trainer under the RQ2 design.

Usage: python -m dynamic_rubric.phase1.evorubrics_run --help
Training is explicit; preparation never loads model weights or calls a judge.
"""

from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import os
import re
import subprocess
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from dynamic_rubric.artifacts import write_json_atomic
from dynamic_rubric.hashing import sha256_bytes, sha256_file, sha256_json

from .config import Phase1Config, load_phase1_config
from .evorubrics_data import prepare_evorubrics_data
from .provenance import prepare_fixed_train_probe_manifest


def _set(config: dict[str, Any], key: str, value: Any) -> None:
    parts = key.split(".")
    node = config
    for part in parts[:-1]:
        node = node.setdefault(part, {})
    node[parts[-1]] = value


def build_training_config(
    config: Phase1Config,
    *,
    repo_root: Path,
    run_root: Path,
    train_path: Path,
    smoke: bool = False,
    micro_batch: int = 1,
) -> dict[str, Any]:
    if config.method != "evorubrics":
        raise ValueError("An EvoRubrics Phase-1 config is required")
    if micro_batch < 1:
        raise ValueError("micro_batch must be positive")
    upstream = repo_root / "environment/upstream/EvoRubrics"
    cfg = yaml.safe_load((upstream / "evorubric-main/config/shared_base_config.yaml").read_text())
    evo = config.method_config
    batch = 2 if smoke else int(config.training["global_prompt_batch"])
    if batch % micro_batch:
        raise ValueError("micro_batch must divide the global batch")
    policy_path = Path(str(config.models["policy"]["local_snapshot"])).expanduser()
    if not policy_path.is_absolute():
        policy_path = repo_root / policy_path
    fields = {
        "policy_model.path": str(policy_path.resolve()),
        "data.train_files": str(train_path),
        "data.val_files": None,
        "data.train_batch_size": batch,
        "data.train_max_samples": 2 if smoke else -1,
        "data.max_prompt_length": 3584,
        "data.max_response_length": 1024,
        "data.filter_overlong_prompts": False,
        "data.truncation": "error",
        "data.shuffle": False,
        "data.dataloader_num_workers": 0,
        "data.dataset_mode": "open_rubrics",
        "data.num_answers": int(evo["policy_responses_m"]),
        "data.num_rubrics": int(evo["rubric_sets_n"]),
        "actor_rollout_ref.model.lora_rank": evo["lora_rank"],
        "actor_rollout_ref.model.lora_alpha": evo["lora_alpha"],
        "actor_rollout_ref.model.override_config.attn_implementation": "flash_attention_2",
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.use_dynamic_bsz": False,
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": micro_batch,
        "actor_rollout_ref.actor.optim.lr": evo["policy_learning_rate"],
        "actor_rollout_ref.actor.optim.policy_llm_lr": evo["policy_learning_rate"],
        "actor_rollout_ref.actor.optim.rubrics_generator_lr": evo["rubric_generator_learning_rate"],
        "actor_rollout_ref.actor.optim.total_training_steps": 1
        if smoke
        else int(config.training["expected_global_steps"]),
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": evo["kl_loss_coefficient"],
        "actor_rollout_ref.actor.kl_loss_type": "low_var_kl",
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 8192,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.use_orig_params": True,
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.ref.fsdp_config.param_offload": False,
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": 8192,
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": 8192,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": 0.4,
        "actor_rollout_ref.rollout.free_cache_engine": False,
        "actor_rollout_ref.rollout.max_num_seqs": 64,
        "actor_rollout_ref.rollout.max_num_batched_tokens": 4096,
        "actor_rollout_ref.rollout.max_model_len": 8192,
        "actor_rollout_ref.rollout.prompt_length": 3584,
        "actor_rollout_ref.rollout.response_length": 1024,
        "actor_rollout_ref.rollout.n": int(evo["policy_responses_m"]),
        "actor_rollout_ref.rollout.temperature": evo["generation_temperature"],
        "actor_rollout_ref.rollout.top_p": 0.95,
        "actor_rollout_ref.rollout.enable_prefix_caching": False,
        "actor_rollout_ref.rollout.seed": config.seed,
        "algorithm.adv_estimator": "grpo",
        "algorithm.use_kl_in_reward": False,
        "trainer.project_name": "phase1_dynamic_evaluator_updates",
        "trainer.experiment_name": run_root.name,
        "trainer.n_gpus_per_node": len(config.raw["infrastructure"]["optimizer"]["gpus"]),
        "trainer.nnodes": 1,
        "trainer.total_epochs": 1 if smoke else int(config.training["epochs"]),
        "trainer.total_training_steps": 1
        if smoke
        else int(config.training["expected_global_steps"]),
        "trainer.save_freq": 1 if smoke else int(config.training["checkpoint_interval_steps"]),
        "trainer.save_lora_freq": 1 if smoke else int(config.training["checkpoint_interval_steps"]),
        "trainer.default_local_dir": str(run_root / "upstream-run"),
        "trainer.logger": ["console"],
        "trainer.eval.enabled": False,
        "trainer.eval.eval_before_train": False,
        "trainer.val_before_train": False,
        "custom_reward_function.path": str(upstream / "evorubric-main/custom_reward_fn.py"),
        "custom_reward_function.name": "adversarial_reward_fn_batch",
        "adversarial.training_flow": "unified",
        "adversarial.policy_update_freq": 1,
        "adversarial.rubrics_update_freq": 1,
        "adversarial.policy_grad_accum_steps": 1,
        "adversarial.rubrics_grad_accum_steps": 1,
        "adversarial.num_iterations_per_query": 1,
        "adversarial.max_rubrics_parse_retries": 1,
        "adversarial.generation_temperature": evo["generation_temperature"],
        "adversarial.reflect_generation_temperature": 0.0,
        "adversarial.reflect_use_golden_rubrics": True,
        "adversarial.answer_aware_num_refs": 0,
        "adversarial.enable_exemplar_trick": False,
        "adversarial.use_universal_rubrics_system_prompt": True,
        "adversarial.enable_rubrics_similarity_filter": False,
        "adversarial.similarity_penalty_weight": 0.0,
        "adversarial.similarity_penalty_threshold": 0.0,
        # Keep all three replicas batched without pushing queued 8K-token
        # requests beyond the evaluator timeout on the slower InferenceB replicas.
        "adversarial.max_concurrent_api_calls": 8,
        "rq2.enabled": True,
        "rq2.run_root": str(run_root),
        "rq2.domain": config.domain,
        "rq2.seed": config.seed,
        "rq2.observer_module": "dynamic_rubric.phase1.evorubrics_observer",
        "rq2.prune_old_optimizers": not smoke,
    }
    for name, weight in evo["reward_weights"].items():
        fields[f"adversarial.enable_{name}"] = True
        fields[f"adversarial.{name}_weight"] = weight
    for role in ("policy_llm", "rubrics_generator"):
        fields[f"dual_lora.{role}.rank"] = evo["lora_rank"]
        fields[f"dual_lora.{role}.alpha"] = evo["lora_alpha"]
    for key, value in fields.items():
        _set(cfg, key, value)
    return cfg


def prepare(
    config_path: Path, repo_root: Path, run_id: str, *, smoke: bool = False, micro_batch: int = 1
) -> Path:
    config = load_phase1_config(config_path)
    run_root = config.run_root(repo_root, run_id).resolve()
    probe_path, _ = prepare_fixed_train_probe_manifest(config, repo_root=repo_root)
    data = prepare_evorubrics_data(
        config,
        repo_root=repo_root,
        output_dir=run_root / "data",
        fixed_probe_manifest_path=probe_path,
    )
    cfg = build_training_config(
        config,
        repo_root=repo_root,
        run_root=run_root,
        train_path=data.train_path,
        smoke=smoke,
        micro_batch=micro_batch,
    )
    write_json_atomic(run_root / "config.resolved.json", dict(config.raw))
    write_json_atomic(run_root / "upstream.config.json", cfg)
    manifest = {
        "schema_version": 1,
        "run_id": run_id,
        "domain": config.domain,
        "method": "evorubrics",
        "seed": config.seed,
        "mode": "smoke" if smoke else "full",
        "phase1_config_sha256": config.config_hash,
        "upstream_config_sha256": sha256_json(cfg),
        "dataset_manifest": str(data.manifest_path),
        "dataset_manifest_sha256": sha256_file(data.manifest_path),
        "train_path": str(data.train_path),
        "fixed_probe_manifest": str(probe_path),
        "fixed_probe_manifest_sha256": sha256_file(probe_path),
        "policy_model": config.models["policy"],
        "judge_model": config.models["judge"]["model"],
        "expected_steps": 1 if smoke else int(config.training["expected_global_steps"]),
        "expected_prompt_exposures": 2
        if smoke
        else int(config.data["train_prompt_count"]) * int(config.training["epochs"]),
        "training_responses_m": int(config.method_config["policy_responses_m"]),
        "rubric_sets_n": int(config.method_config["rubric_sets_n"]),
        "probe_pool_b": int(config.method_config["pool_b_count"]),
        "main_comparison": "within_method_same_response_fresh_vs_stale",
        "source_archive": str(repo_root / "docs/EvoRubrics-2155.zip"),
        "source_archive_sha256": sha256_file(repo_root / "docs/EvoRubrics-2155.zip"),
        "training_started": False,
    }
    write_json_atomic(run_root / "launch_spec.json", manifest)
    return run_root


def runtime_environment(
    repo_root: Path, run_root: Path, judge_url: str, gpu: str = "0"
) -> dict[str, str]:
    if not judge_url.startswith(("http://", "https://")):
        raise ValueError("An HTTP(S) judge endpoint is required")
    upstream = repo_root / "environment/upstream/EvoRubrics"
    run_digest = sha256_bytes(str(run_root.resolve()).encode("utf-8"))[:12]
    ray_tmpdir = Path("/tmp") / f"evo-ray-{run_digest}"
    ray_tmpdir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONPATH": os.pathsep.join(
                [
                    str(repo_root / "src"),
                    str(upstream),
                    str(upstream / "evorubric-main"),
                    str(upstream / "third_party/verl"),
                    str(upstream / "rubric_guidance"),
                ]
                + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
            ),
            "CUDA_VISIBLE_DEVICES": gpu,
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONNOUSERSITE": "1",
            "HYDRA_FULL_ERROR": "1",
            "NCCL_DEBUG": "WARN",
            "RAY_TMPDIR": str(ray_tmpdir),
            "RAY_ADDRESS": "local",
            "DEEPSEEK_BASE_URL": judge_url.rstrip("/")
            if judge_url.rstrip("/").endswith("/v1")
            else judge_url.rstrip("/") + "/v1",
            "DEEPSEEK_MODEL": "openai/gpt-oss-120b",
            "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "EMPTY"),
            "LLM_EVALUATOR_TYPE": "deepseek",
            "LLM_EVALUATOR_MAX_CONCURRENT_FALLBACK": "4",
            "LLM_EVALUATOR_TIMEOUT_SECONDS": "1200",
            # Four answers against a full medicine rubric can exceed 4K output
            # tokens; truncation silently turns missing answers into zero reward.
            "LLM_EVALUATOR_MAX_COMPLETION_TOKENS": "8192",
            "LLM_JUDGE_DEBUG_LOG": str(run_root / "logs/judge.jsonl"),
        }
    )
    return env


_PROVENANCE_CODE_PATHS = (
    "environment/upstream/EvoRubrics/third_party/verl/verl/workers/config/rollout.py",
    "environment/upstream/EvoRubrics/evorubric-main/adversarial_dataset.py",
    "environment/upstream/EvoRubrics/evorubric-main/custom_reward_fn.py",
    "environment/upstream/EvoRubrics/evorubric-main/dual_lora_worker.py",
    "environment/upstream/EvoRubrics/evorubric-main/llm_evaluator.py",
    "environment/upstream/EvoRubrics/evorubric-main/main_shared_base.py",
    "environment/upstream/EvoRubrics/evorubric-main/reward_calculator.py",
    "environment/upstream/EvoRubrics/evorubric-main/shared_base_trainer.py",
    "src/dynamic_rubric/phase1/evorubrics_audit.py",
    "src/dynamic_rubric/phase1/evorubrics_data.py",
    "src/dynamic_rubric/phase1/evorubrics_observer.py",
    "src/dynamic_rubric/phase1/evorubrics_probe.py",
    "src/dynamic_rubric/phase1/evorubrics_run.py",
)


def resolve_judge_identity(judge_url: str, expected_model: str) -> dict[str, Any]:
    """Resolve the deployed judge identity from its OpenAI-compatible model endpoint."""
    base = judge_url.rstrip("/")
    endpoint = base + "/models" if base.endswith("/v1") else base + "/v1/models"
    with urllib.request.urlopen(endpoint, timeout=15) as response:
        payload = json.load(response)
    matches = [row for row in payload.get("data", []) if row.get("id") == expected_model]
    if len(matches) != 1:
        raise ValueError(f"Judge endpoint does not expose exactly one {expected_model!r} model")
    row = matches[0]
    return {key: row[key] for key in ("id", "root", "max_model_len", "owned_by") if key in row}


def build_run_provenance(
    run_root: Path,
    repo_root: Path,
    judge_identity: dict[str, Any],
    *,
    actual_training: bool,
) -> dict[str, Any]:
    """Build a scope-aware record with a smoke/full-compatible semantic identity."""
    spec = json.loads((run_root / "launch_spec.json").read_text())
    data = json.loads(Path(spec["dataset_manifest"]).read_text())
    splits = data["splits"]
    semantic_identity = {
        "schema_version": 1,
        "domain": spec["domain"],
        "method": spec["method"],
        "seed": spec["seed"],
        "phase1_config_sha256": spec["phase1_config_sha256"],
        "source_archive_sha256": spec["source_archive_sha256"],
        "runtime_lock_sha256": sha256_file(repo_root / "environment/evorubrics-runtime-lock.txt"),
        "code_sha256": {
            relative: sha256_file(repo_root / relative) for relative in _PROVENANCE_CODE_PATHS
        },
        "data": {
            name: {
                "source_sha256": splits[name]["source"]["sha256"],
                "row_count": splits[name]["row_count"],
                "ordered_prompt_ids_sha256": splits[name]["ordered_prompt_ids_sha256"],
            }
            for name in ("train", "heldout")
        },
        "fixed_probe": {
            "manifest_sha256": spec["fixed_probe_manifest_sha256"],
            "prompt_count": data["fixed_train_probe"]["prompt_count"],
            "prompt_ids_sha256": data["fixed_train_probe"]["prompt_ids_sha256"],
        },
        "policy_model": spec["policy_model"],
        "judge_identity": judge_identity,
        "training_responses_m": spec["training_responses_m"],
        "rubric_sets_n": spec["rubric_sets_n"],
        "probe_pool_b": spec["probe_pool_b"],
    }
    return {
        "schema_version": 1,
        "semantic_identity": semantic_identity,
        "semantic_identity_sha256": sha256_json(semantic_identity),
        "scope": {
            "mode": spec["mode"],
            "upstream_config_sha256": spec["upstream_config_sha256"],
            "expected_steps": spec["expected_steps"],
            "expected_prompt_exposures": spec["expected_prompt_exposures"],
        },
        "actual_training": actual_training,
    }


def quarantine_uncommitted_after(run_root: Path, resume_step: int) -> dict[str, Any]:
    """Move replay-conflicting later artifacts aside without deleting evidence."""
    committed_dir = run_root / "checkpoints/committed"
    committed_steps = {
        int(path.stem.removeprefix("step_")) for path in committed_dir.glob("step_*.json")
    }
    later_committed = sorted(step for step in committed_steps if step > resume_step)
    if later_committed:
        raise ValueError(f"Cannot resume behind committed checkpoint steps: {later_committed}")

    candidates: list[Path] = []
    for folder in ("audit/train_batch", "audit/advantages", "metrics"):
        for path in (run_root / folder).glob("step_*.*"):
            match = re.match(r"step_(\d+)", path.name)
            if match and int(match.group(1)) > resume_step:
                candidates.append(path)
    for role in ("policy_llm", "rubrics_generator"):
        for path in (run_root / "upstream-run" / role).glob("step_*"):
            match = re.fullmatch(r"step_(\d+)", path.name)
            if match and int(match.group(1)) > resume_step:
                candidates.append(path)

    attempt_root = run_root / "resume-quarantine" / f"from_{resume_step:06d}"
    suffix = 1
    destination_root = attempt_root / f"attempt_{suffix:03d}"
    while destination_root.exists():
        suffix += 1
        destination_root = attempt_root / f"attempt_{suffix:03d}"
    records = []
    for source in sorted(set(candidates)):
        relative = source.relative_to(run_root)
        before_hash = sha256_file(source) if source.is_file() else None
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.replace(destination)
        records.append(
            {
                "source": str(relative),
                "quarantined": str(destination.relative_to(run_root)),
                "sha256": before_hash,
            }
        )
    manifest = {
        "schema_version": 1,
        "resume_step": resume_step,
        "artifacts": records,
        "replay_guarantee": "later uncommitted artifacts moved; vLLM exact replay not claimed",
    }
    write_json_atomic(destination_root / "manifest.json", manifest)
    return manifest


def execute(
    run_root: Path,
    repo_root: Path,
    runtime_python: Path,
    judge_url: str,
    *,
    gpu: str = "1",
    resume_step: int | None = None,
    smoke_proof: Path | None = None,
) -> int:
    spec = json.loads((run_root / "launch_spec.json").read_text())
    proof = None
    if spec["mode"] == "full":
        if smoke_proof is None:
            raise ValueError("Full training requires a passed live smoke report")
        proof = json.loads(smoke_proof.read_text())
        if (
            proof.get("status") != "passed"
            or proof.get("phase1_config_sha256") != spec["phase1_config_sha256"]
            or not proof.get("actual_training")
            or not proof.get("actual_judge")
            or not proof.get("resume_verified")
        ):
            raise ValueError("Smoke report does not validate this experiment config")
    if (run_root / "training_started.json").exists() and resume_step is None:
        raise ValueError("Run already started; use an explicit committed resume step")
    env = runtime_environment(repo_root, run_root, judge_url, gpu)
    judge_identity = resolve_judge_identity(env["DEEPSEEK_BASE_URL"], spec["judge_model"])
    provenance = build_run_provenance(run_root, repo_root, judge_identity, actual_training=False)
    if (
        proof is not None
        and proof.get("semantic_identity_sha256") != provenance["semantic_identity_sha256"]
    ):
        raise ValueError("Smoke report semantic provenance differs from this full run")
    write_json_atomic(run_root / "run_provenance.json", provenance, immutable=False)
    (run_root / "logs").mkdir(parents=True, exist_ok=True)
    Path(env["RAY_TMPDIR"]).mkdir(parents=True, exist_ok=True)
    launch = {
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "runtime_python": str(runtime_python),
        "gpu": gpu,
        "judge_url": env["DEEPSEEK_BASE_URL"],
        "resume_step": resume_step,
    }
    write_json_atomic(run_root / "training_started.json", launch, immutable=False)
    command = [
        str(runtime_python),
        "-m",
        "dynamic_rubric.phase1.evorubrics_run",
        "_train",
        "--run-root",
        str(run_root),
    ]
    if resume_step is not None:
        command += ["--resume-step", str(resume_step)]
    log = (
        run_root
        / "logs"
        / (f"train-resume-{resume_step}.log" if resume_step is not None else "train.log")
    )
    with log.open("a") as handle:
        result = subprocess.run(
            command, cwd=repo_root, env=env, stdout=handle, stderr=subprocess.STDOUT, check=False
        )
    write_json_atomic(
        run_root / "process_result.json",
        {"returncode": result.returncode, "log": str(log), "resume_step": resume_step},
        immutable=False,
    )
    return result.returncode


def _train(run_root: Path, resume_step: int | None) -> None:
    import random

    import numpy as np
    import torch
    from main_shared_base import run_shared_base_adversarial_training
    from omegaconf import OmegaConf

    from .evorubrics_observer import discover_checkpoint_pairs

    raw = json.loads((run_root / "upstream.config.json").read_text())
    spec = json.loads((run_root / "launch_spec.json").read_text())
    random.seed(spec["seed"])
    np.random.seed(spec["seed"])
    torch.manual_seed(spec["seed"])
    resume_evidence = None
    if resume_step is not None:
        inventory = discover_checkpoint_pairs(run_root)
        pair = next((p for p in inventory["checkpoints"] if p["global_step"] == resume_step), None)
        if pair is None:
            raise ValueError("Resume requires a validated policy/generator pair")
        resume_evidence = {"checkpoint_step": resume_step, "roles": {}}
        for role, key in (("policy", "policy_lora_path"), ("generator", "rubrics_lora_path")):
            optimizer = Path(pair[role]["optimizer_path"])
            if not optimizer.is_file():
                raise ValueError(f"Optimizer was pruned or is missing: {role}")
            resume_evidence["roles"][role] = {
                "adapter_path": pair[role]["adapter_path"],
                "weights_sha256": pair[role]["weights_sha256"],
                "config_sha256": pair[role]["config_sha256"],
                "optimizer_path": str(optimizer),
                "optimizer_sha256": sha256_file(optimizer),
            }
            raw["actor_rollout_ref"]["model"][key] = pair[role]["adapter_path"]
        raw["trainer"]["resume_step"] = resume_step
        resume_evidence["quarantine"] = quarantine_uncommitted_after(run_root, resume_step)
    packages = {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()}
    write_json_atomic(run_root / "runtime-packages.json", packages)
    main_call_started_at = dt.datetime.now(dt.timezone.utc).isoformat()
    run_shared_base_adversarial_training(OmegaConf.create(raw))
    main_call_completed_at = dt.datetime.now(dt.timezone.utc).isoformat()
    inventory = discover_checkpoint_pairs(run_root)
    expected = int(spec["expected_steps"])
    if expected not in inventory["steps"] or inventory["excluded"]:
        raise RuntimeError("Training returned without a valid final checkpoint pair")
    required_training_artifacts = [
        run_root / f"metrics/step_{expected:06d}.json",
        run_root / f"audit/advantages/step_{expected:06d}_policy_llm.json",
        run_root / f"audit/advantages/step_{expected:06d}_rubrics_generator.json",
    ]
    if expected < 1 or any(not path.is_file() for path in required_training_artifacts):
        raise RuntimeError("Training returned without complete real-update audit evidence")
    provenance_path = run_root / "run_provenance.json"
    provenance = json.loads(provenance_path.read_text())
    provenance["actual_training"] = True
    write_json_atomic(provenance_path, provenance, immutable=False)
    report = {
        "status": "training_passed",
        "actual_training": True,
        "phase1_config_sha256": spec["phase1_config_sha256"],
        "semantic_identity_sha256": provenance["semantic_identity_sha256"],
        "checkpoints": inventory["steps"],
        "model_weights_loaded": True,
        "remote_endpoints_called": True,
        "expected_steps": expected,
        "main_call_started_at": main_call_started_at,
        "main_call_completed_at": main_call_completed_at,
        "probe_audit_status": "pending_live_probe",
        "run_root": str(run_root),
    }
    write_json_atomic(run_root / "training_complete.json", report, immutable=False)
    if resume_evidence is not None:
        resume_evidence.update(
            {
                "schema_version": 1,
                "status": "passed",
                "actual_reload": True,
                "completed_main_call": True,
                "strict_optimizer_reload": True,
                "main_call_started_at": main_call_started_at,
                "main_call_completed_at": main_call_completed_at,
                "final_checkpoint_step": expected,
                "final_checkpoint_steps": inventory["steps"],
                "semantic_identity_sha256": provenance["semantic_identity_sha256"],
            }
        )
        write_json_atomic(run_root / "resume_verified.json", resume_evidence, immutable=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument(
        "--config", type=Path, default=Path("configs/phase1/medicine_evorubrics.yaml")
    )
    prep.add_argument("--repo-root", type=Path, default=Path.cwd())
    prep.add_argument("--run-id", required=True)
    prep.add_argument("--smoke", action="store_true")
    prep.add_argument("--micro-batch", type=int, default=1)
    launch = sub.add_parser("launch")
    launch.add_argument("--repo-root", type=Path, default=Path.cwd())
    launch.add_argument("--run-root", type=Path, required=True)
    launch.add_argument(
        "--runtime-python",
        type=Path,
        default=Path(os.environ.get("EVORUBRICS_RUNTIME_PYTHON", ".venvs/evorubrics/bin/python")),
    )
    launch.add_argument("--judge-base-url", required=True)
    launch.add_argument("--gpu", default=os.environ.get("EVORUBRICS_TRAINER_GPUS", "0"))
    launch.add_argument("--resume-step", type=int)
    launch.add_argument("--smoke-proof", type=Path)
    train = sub.add_parser("_train")
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--resume-step", type=int)
    args = parser.parse_args()
    if args.command == "prepare":
        print(
            prepare(
                args.config,
                args.repo_root.resolve(),
                args.run_id,
                smoke=args.smoke,
                micro_batch=args.micro_batch,
            )
        )
    elif args.command == "launch":
        raise SystemExit(
            execute(
                args.run_root.resolve(),
                args.repo_root.resolve(),
                args.runtime_python,
                args.judge_base_url,
                gpu=args.gpu,
                resume_step=args.resume_step,
                smoke_proof=args.smoke_proof,
            )
        )
    else:
        _train(args.run_root.resolve(), args.resume_step)


if __name__ == "__main__":
    main()
