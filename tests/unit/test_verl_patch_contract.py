from __future__ import annotations

import ast
from pathlib import Path

import pytest

from dynamic_rubric.artifacts import read_json
from dynamic_rubric.training import verl_adapter
from dynamic_rubric.training.verl_adapter import DependencyGateError, dependency_gate


ROOT = Path(__file__).resolve().parents[2]


def test_online_launcher_limits_dynamic_microbatches_to_one_max_length_sequence() -> None:
    launcher = (ROOT / "scripts" / "phase1" / "run_online_full.sh").read_text()

    assert "ACTOR_MAX_TOKEN_LEN=${ACTOR_MAX_TOKEN_LEN:-8192}" in launcher
    assert "ROLLOUT_LOG_PROB_MAX_TOKEN_LEN=${ROLLOUT_LOG_PROB_MAX_TOKEN_LEN:-8192}" in launcher
    assert "REF_LOG_PROB_MAX_TOKEN_LEN=${REF_LOG_PROB_MAX_TOKEN_LEN:-8192}" in launcher


def test_stage5_launcher_enables_actor_and_reference_determinism() -> None:
    launcher = (ROOT / "scripts" / "run_static_grpo.sh").read_text()

    assert 'actor_rollout_ref.actor.fsdp_config.full_determinism="${FULL_DETERMINISM}"' in launcher
    assert 'actor_rollout_ref.ref.fsdp_config.full_determinism="${FULL_DETERMINISM}"' in launcher
    assert 'trainer.total_epochs="${TOTAL_EPOCHS}"' in launcher
    assert "trainer.total_epochs=1" not in launcher


def test_static_launcher_exposes_system_only_weight_transfer_bucket_size() -> None:
    launcher = (ROOT / "scripts" / "run_static_grpo.sh").read_text()

    assert (
        "ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES="
        "${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES:-2048}"
    ) in launcher
    assert (
        'actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes="'
        '${ROLLOUT_UPDATE_WEIGHTS_BUCKET_MEGABYTES}"'
    ) in launcher


def test_horizon_launcher_uses_h200_compute_profile_without_changing_batch_semantics() -> None:
    horizon_launcher = (ROOT / "scripts" / "run_horizon_static_grpo.sh").read_text()
    verl_launcher = (ROOT / "scripts" / "run_static_grpo.sh").read_text()
    dollar = "$"

    assert f"export TRAIN_BATCH_SIZE={dollar}{{TRAIN_BATCH_SIZE:-96}}" in horizon_launcher
    assert f"export ROLLOUT_N={dollar}{{ROLLOUT_N:-16}}" in horizon_launcher
    assert f"export MAX_RESPONSE_LENGTH={dollar}{{MAX_RESPONSE_LENGTH:-3584}}" in horizon_launcher
    assert f"export ACTOR_MAX_TOKEN_LEN={dollar}{{ACTOR_MAX_TOKEN_LEN:-24576}}" in horizon_launcher
    assert f"export ROLLOUT_LOG_PROB_MAX_TOKEN_LEN={dollar}{{ROLLOUT_LOG_PROB_MAX_TOKEN_LEN:-49152}}" in horizon_launcher
    assert f"export REF_LOG_PROB_MAX_TOKEN_LEN={dollar}{{REF_LOG_PROB_MAX_TOKEN_LEN:-65536}}" in horizon_launcher
    assert f"export USE_FUSED_KERNELS={dollar}{{USE_FUSED_KERNELS:-True}}" in horizon_launcher
    assert f"export ENABLE_GRADIENT_CHECKPOINTING={dollar}{{ENABLE_GRADIENT_CHECKPOINTING:-False}}" in horizon_launcher
    assert f"export FULL_DETERMINISM={dollar}{{FULL_DETERMINISM:-False}}" in horizon_launcher
    assert f"export ENFORCE_EAGER={dollar}{{ENFORCE_EAGER:-False}}" in horizon_launcher
    assert f"export REF_PARAM_OFFLOAD={dollar}{{REF_PARAM_OFFLOAD:-False}}" in horizon_launcher

    assert f'actor_rollout_ref.model.use_fused_kernels="{dollar}{{USE_FUSED_KERNELS}}"' in verl_launcher
    assert f'actor_rollout_ref.model.fused_kernel_options.impl_backend="{dollar}{{FUSED_KERNEL_BACKEND}}"' in verl_launcher
    assert f'actor_rollout_ref.model.enable_gradient_checkpointing="{dollar}{{ENABLE_GRADIENT_CHECKPOINTING}}"' in verl_launcher
    assert f'actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="{dollar}{{ROLLOUT_LOG_PROB_MAX_TOKEN_LEN}}"' in verl_launcher
    assert f'actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="{dollar}{{REF_LOG_PROB_MAX_TOKEN_LEN}}"' in verl_launcher


def test_stage4_stage5_launcher_uses_both_gpus_without_overlap() -> None:
    launcher = (ROOT / "scripts" / "run_stage4_stage5.sh").read_text()

    assert "CUDA_VISIBLE_DEVICES=0" in launcher
    assert "CUDA_VISIBLE_DEVICES=1" in launcher
    assert launcher.count("nohup vllm serve") == 2
    assert "-m vllm serve" not in launcher
    grader_launcher = launcher.split("start_grader_if_needed() {", 1)[1].split(
        "start_policy() {", 1
    )[0]
    assert "local proxy_ready=false" in grader_launcher
    assert grader_launcher.index("GRADER_UPSTREAM_URL") < grader_launcher.index(
        'if [[ "${proxy_ready}" == true ]]'
    )
    assert "DYNAMIC_RUBRIC_EMBEDDING_DEVICE:-cuda:1" in launcher
    assert "export VLLM_USE_DEEP_GEMM=0" in launcher
    assert "DYNAMIC_RUBRIC_GRADER_UPSTREAM_URLS" in launcher
    assert 'proxy_upstream_args+=(--upstream "${upstream}")' in launcher
    assert '[[ "${GRADER_UPSTREAM_URL}" != "http://127.0.0.1:8002" ]]' in launcher
    assert '((${#grader_upstream_urls[@]} != 1))' not in launcher
    assert 'stop_process "${policy_pid}"' in launcher
    assert "wait_gpu0_free" in launcher
    assert "unset OPENAI_API_KEY" in launcher
    assert launcher.index("generate-static") < launcher.rindex('stop_process "${policy_pid}"')
    assert launcher.rindex("wait_gpu0_free") < launcher.rindex("train-static")


def test_stage5_launcher_can_resume_without_repeating_stage4() -> None:
    launcher = (ROOT / "scripts" / "run_stage4_stage5.sh").read_text()

    resume_branch = launcher.split('if [[ "${RESUME_STATIC_ONLY}" == true ]]', 1)[1]
    resume_branch = resume_branch.split("\nfi\n", 1)[0]
    assert "validate-config" in resume_branch
    assert "train-static" in resume_branch
    assert "generate-static" not in resume_branch
    assert "preflight" not in resume_branch
    assert "OPENAI_API_KEY" not in resume_branch
    assert "exit 0" in resume_branch


def test_verl_patch_preserves_global_rollout_indices_and_math_sdpa() -> None:
    patch = (ROOT / "patches" / "verl_stage5_determinism.patch").read_text()

    assert 'prompts.non_tensor_batch["__rollout_index__"]' in patch
    assert 'internal_keys = {"__do_sample__", "__rollout_index__"}' in patch
    assert "torch.backends.cuda.enable_flash_sdp(False)" in patch
    assert "torch.backends.cuda.enable_mem_efficient_sdp(False)" in patch
    assert "torch.backends.cuda.enable_math_sdp(True)" in patch


def test_dependency_gate_verifies_exact_applied_stage5_patch() -> None:
    capabilities = dependency_gate(
        read_json(ROOT / "environment" / "upstream-lock.json"), ROOT
    )

    assert capabilities.stage5_patch_applied is True
    assert capabilities.patch_sha256


def test_dependency_gate_rejects_checkout_head_that_differs_from_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = read_json(ROOT / "environment" / "upstream-lock.json")
    lock["verl"] = {**lock["verl"], "commit": "0" * 40}

    def unexpected_patch_comparison(*_args: object, **_kwargs: object) -> bool:
        raise AssertionError("patch comparison must not run for a mismatched checkout")

    monkeypatch.setattr(
        verl_adapter, "_patch_bundle_is_applied", unexpected_patch_comparison
    )

    with pytest.raises(DependencyGateError, match="HEAD does not match lock"):
        dependency_gate(lock, ROOT)


def test_online_patch_adds_fail_closed_full_batch_lifecycle_hooks() -> None:
    patch = (ROOT / "patches" / "verl_online_rubrics.patch").read_text()

    union = "batch = batch.union(gen_batch_output)"
    prepare = "batch = self._prepare_online_rewards(batch)"
    balance = 'if "response_mask" not in batch.batch.keys():'
    assert patch.index(union) < patch.index(prepare) < patch.index(balance)
    assert "online_step_hook refuses precomputed rm_scores" in patch
    assert "online scalar rewards must be placed only at the last valid token" in patch
    assert "online_step_enabled" in patch
    assert "checkpoint_saved = scheduled_checkpoint or esi_close_to_expiration" in patch
    assert "if checkpoint_saved" in patch
    assert "self._commit_online_step(checkpoint_dir)" in patch
    assert 'drop_last=self.config.data.get("train_drop_last", True)' in patch


def test_online_patch_disables_streaming_reward_before_full_batch_hook() -> None:
    trainer = (
        ROOT
        / "environment"
        / "upstream"
        / "verl"
        / "verl"
        / "trainer"
        / "ppo"
        / "ray_trainer.py"
    ).read_text()

    assert "online_step_enabled = self._online_step_hook_config() is not None" in trainer
    assert "enable_agent_reward_loop = not online_step_enabled and (" in trainer
    assert "reward_loop_worker_handles=reward_loop_worker_handles" in trainer
    assert "online_reward_input_keys = list(" in trainer
    assert "gen_batch_output.pop(non_tensor_batch_keys=online_reward_input_keys)" in trainer


def test_online_patch_skips_unused_old_log_prob_entropy() -> None:
    trainer = (
        ROOT
        / "environment"
        / "upstream"
        / "verl"
        / "verl"
        / "trainer"
        / "ppo"
        / "ray_trainer.py"
    ).read_text()
    patch = (ROOT / "patches" / "verl_online_rubrics.patch").read_text()

    for source in (trainer, patch):
        assert "calculate_entropy = actor_config.calculate_entropy or (" in source
        assert "actor_config.entropy_coeff != 0.0" in source
        assert "batch, calculate_entropy=calculate_entropy" in source
        assert "if calculate_entropy:" in source
        assert 'old_log_prob_metrics["actor/entropy"]' in source

    unconditional_call = "old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)"
    assert unconditional_call not in trainer
    assert f"-                            {unconditional_call}" in patch


def test_online_patch_adapts_ppo_minibatch_to_final_partial_prompt_batch() -> None:
    trainer_path = (
        ROOT
        / "environment"
        / "upstream"
        / "verl"
        / "verl"
        / "trainer"
        / "ppo"
        / "ray_trainer.py"
    )
    trainer = trainer_path.read_text()
    patch = (ROOT / "patches" / "verl_online_rubrics.patch").read_text()

    function = next(
        node
        for node in ast.parse(trainer).body
        if isinstance(node, ast.FunctionDef) and node.name == "_resolve_ppo_mini_batch_size"
    )
    namespace: dict[str, object] = {}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(trainer_path), "exec"), namespace)
    resolve = namespace["_resolve_ppo_mini_batch_size"]

    assert resolve(
        configured_prompt_batch_size=96, rollout_n=16, actual_response_batch_size=1536
    ) == 1536
    assert resolve(
        configured_prompt_batch_size=96, rollout_n=16, actual_response_batch_size=960
    ) == 960
    with pytest.raises(ValueError, match="not grouped"):
        resolve(configured_prompt_batch_size=96, rollout_n=16, actual_response_batch_size=961)

    for source in (trainer, patch):
        assert "_resolve_ppo_mini_batch_size" in source
        assert "actual_response_batch_size=batch_td.shape[0]" in source
        assert "Adjusted PPO mini-batch size for partial prompt batch" in source


def test_dependency_gate_verifies_online_patch_bundle() -> None:
    capabilities = dependency_gate(
        read_json(ROOT / "environment" / "upstream-lock.json"), ROOT
    )

    assert capabilities.online_patch_applied is True
    assert capabilities.online_patch_sha256
    assert capabilities.online_ready is True


def test_seeded_agent_exports_per_response_metadata_to_async_reward_manager() -> None:
    source = (ROOT / "src" / "dynamic_rubric" / "training" / "seeded_agent.py").read_text()

    # Fresh and cache-hit rollouts must both expose the immutable identifiers
    # through AgentLoopOutput.extra_fields, which veRL merges into extra_info.
    assert source.count("result.extra_fields.update(metadata)") == 2
