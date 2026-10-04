"""Deterministic Phase-1 contract smoke; it never launches training or remote models."""

from __future__ import annotations

from pathlib import Path
import socket
import statistics
import time
from typing import Any, Mapping

from ..artifacts import (
    read_json,
    write_json_atomic,
    write_jsonl_atomic,
)
from ..hashing import sha256_json
from ..training.probe_export import prove_probe_side_effect_free
from .config import Phase1Config, load_phase1_config
from .metrics import (
    ScoreGroup,
    aggregate_comparisons,
    compare_fresh_stale,
)
from .provenance import (
    load_probe_rows,
    response_id,
    validate_pool_ab_disjoint,
    validate_records,
)
from .shadow import (
    EvoScoreContext,
    OnlineRubricShadowCache,
    OnlineRubricSnapshot,
    validate_evo_fresh_stale_pair,
    validate_same_pool_b,
)


class SmokeError(RuntimeError):
    pass


def _checkpoint(step: int) -> str:
    return f"step-{step:03d}"


def _common(
    config: Phase1Config,
    *,
    step: int,
    prompt_id: str,
    response: str,
    pool: str,
    policy_checkpoint: str,
    evaluator_checkpoint: str = "not_applicable",
    role: str = "not_applicable",
) -> dict[str, Any]:
    return {
        "domain": config.domain,
        "method": config.method,
        "seed": config.seed,
        "global_step": step,
        "checkpoint_id": _checkpoint(step),
        "prompt_id": prompt_id,
        "response_id": response,
        "pool": pool,
        "policy_checkpoint": policy_checkpoint,
        "evaluator_checkpoint": evaluator_checkpoint,
        "fresh_or_stale": role,
    }


def _synthetic_text(prompt_id: str, pool: str, source: str, index: int) -> str:
    return f"deterministic-smoke::{prompt_id}::{pool}::{source}::{index}"


def _make_probe_pools(
    config: Phase1Config,
    prompts: list[Mapping[str, Any]],
    *,
    step: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    pool_a: list[dict[str, Any]] = []
    pool_b: list[dict[str, Any]] = []
    policy_checkpoint = _checkpoint(step)
    if config.method == "online_rubrics":
        sources = (("current", 8, policy_checkpoint), ("pi0_control", 8, _checkpoint(0)))
    else:
        sources = (("psi_rubric_set", 4, policy_checkpoint),)
    pool_b_count = int(config.method_config["pool_b_count"])

    for prompt in prompts:
        prompt_id = str(prompt["prompt_id"])
        for source, count, source_checkpoint in sources:
            for index in range(count):
                rid = response_id(
                    domain=config.domain,
                    method=config.method,
                    seed=config.seed,
                    prompt_id=prompt_id,
                    pool="probe_A",
                    policy_checkpoint=source_checkpoint,
                    sample_index=index,
                    source=source,
                )
                pool_a.append(
                    {
                        **_common(
                            config,
                            step=step,
                            prompt_id=prompt_id,
                            response=rid,
                            pool="probe_A",
                            policy_checkpoint=source_checkpoint,
                        ),
                        "sample_index": index,
                        "source": source,
                        "response_text": _synthetic_text(
                            prompt_id, "probe_A", source, index
                        ),
                        "used_for_evaluator_construction": True,
                        "used_for_gradient": False,
                    }
                )
        for index in range(pool_b_count):
            rid = response_id(
                domain=config.domain,
                method=config.method,
                seed=config.seed,
                prompt_id=prompt_id,
                pool="probe_B",
                policy_checkpoint=policy_checkpoint,
                sample_index=index,
            )
            pool_b.append(
                {
                    **_common(
                        config,
                        step=step,
                        prompt_id=prompt_id,
                        response=rid,
                        pool="probe_B",
                        policy_checkpoint=policy_checkpoint,
                    ),
                    "sample_index": index,
                    "source": "current",
                    "response_text": _synthetic_text(
                        prompt_id, "probe_B", "current", index
                    ),
                    "used_for_evaluator_construction": False,
                    "used_for_gradient": False,
                }
            )
    validate_records((*pool_a, *pool_b))
    validate_pool_ab_disjoint(pool_a, pool_b)
    return pool_a, pool_b


def _criterion_records(
    config: Phase1Config,
    *,
    prompt_id: str,
    pool_a_rows: list[Mapping[str, Any]],
    stale_checkpoint: str,
    fresh_checkpoint: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], tuple[dict[str, Any], ...]]:
    source = pool_a_rows[0]
    initial = {
        "criterion_id": f"{prompt_id}:r0:smoke",
        "criterion": "Provides a relevant and internally consistent answer.",
        "origin": "initial_prompt_rubric_not_ground_truth",
    }
    candidate = {
        "criterion_id": f"{prompt_id}:candidate:smoke",
        "criterion": "Distinguishes the stronger answer with a concrete domain detail.",
        "origin": "step_local_online_candidate",
    }
    before = [
        {
            **_common(
                config,
                step=3,
                prompt_id=prompt_id,
                response=str(source["response_id"]),
                pool="probe_A",
                policy_checkpoint=_checkpoint(3),
                evaluator_checkpoint=fresh_checkpoint,
                role="fresh",
            ),
            **candidate,
            "deduplicated": False,
        }
    ]
    after = [
        {
            **before[0],
            "criterion_id": f"{prompt_id}:online:3:smoke",
            "deduplicated": True,
            "covered_by_initial": False,
        }
    ]
    return before, after, (initial, after[0])


def _score_groups(
    config: Phase1Config,
    *,
    prompt_index: int,
    prompt_id: str,
    pool_b_rows: list[Mapping[str, Any]],
    stale_checkpoint: str,
    fresh_checkpoint: str,
) -> tuple[ScoreGroup, ScoreGroup]:
    response_ids = tuple(str(row["response_id"]) for row in pool_b_rows)
    count = len(response_ids)
    if prompt_index == 0:
        stale_rewards = tuple(0.5 for _ in range(count))
    else:
        stale_rewards = tuple(0.45 + 0.1 * (index % 2) for index in range(count))
    fresh_rewards = tuple(
        min(1.0, 0.2 + 0.6 * index / max(1, count - 1))
        for index in range(count)
    )
    stale_grades = {
        "stale:saturated": tuple(1 for _ in range(count)),
        "stale:dead": tuple(0 for _ in range(count)),
    }
    fresh_grades = {
        "fresh:effective": tuple(index % 2 for index in range(count)),
        "fresh:saturated": tuple(1 for _ in range(count)),
    }
    policy_checkpoint = _checkpoint(3)
    return (
        ScoreGroup(
            prompt_id=prompt_id,
            evaluator_checkpoint=stale_checkpoint,
            policy_checkpoint=policy_checkpoint,
            response_ids=response_ids,
            rewards=stale_rewards,
            criterion_grades=stale_grades,
        ),
        ScoreGroup(
            prompt_id=prompt_id,
            evaluator_checkpoint=fresh_checkpoint,
            policy_checkpoint=policy_checkpoint,
            response_ids=response_ids,
            rewards=fresh_rewards,
            criterion_grades=fresh_grades,
        ),
    )


def _score_records(
    config: Phase1Config,
    *,
    group: ScoreGroup,
    pool_rows: list[Mapping[str, Any]],
    role: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    mean = statistics.fmean(group.rewards)
    std = statistics.pstdev(group.rewards)
    rewards: list[dict[str, Any]] = []
    grades: list[dict[str, Any]] = []
    rows_by_id = {str(row["response_id"]): row for row in pool_rows}
    for index, (rid, reward) in enumerate(zip(group.response_ids, group.rewards)):
        base = _common(
            config,
            step=3,
            prompt_id=group.prompt_id,
            response=rid,
            pool="probe_B",
            policy_checkpoint=group.policy_checkpoint,
            evaluator_checkpoint=group.evaluator_checkpoint,
            role=role,
        )
        rewards.append(
            {
                **base,
                "sample_index": index,
                "scalar_reward": reward,
                "group_advantage": 0.0 if std == 0.0 else (reward - mean) / std,
                "response_text_sha256": sha256_json(rows_by_id[rid]["response_text"]),
            }
        )
        assert group.criterion_grades is not None
        for criterion_id, values in group.criterion_grades.items():
            grades.append(
                {
                    **base,
                    "criterion_id": criterion_id,
                    "criterion_grade": int(values[index]),
                }
            )
    return rewards, grades


def _global_step_logs(config: Phase1Config) -> list[dict[str, Any]]:
    completion_multiplier = 16 if config.method == "online_rubrics" else 4
    return [
        {
            "domain": config.domain,
            "method": config.method,
            "seed": config.seed,
            "global_step": step,
            "cumulative_prompt_exposures": 96 * step,
            "cumulative_completions": 96 * completion_multiplier * step,
            "cumulative_response_tokens": 96 * completion_multiplier * 128 * step,
            "adjacent_policy_kl": 0.0 if step == 0 else 0.001 * step,
            "cumulative_policy_kl_from_pi0": 0.0015 * step,
            "response_length": {
                "mean": 128.0 + step,
                "median": 126.0 + step,
                "min": 32,
                "max": 256 + step,
            },
            "elapsed_seconds": 0.01 * step,
            "throughput_completions_per_second": 1000.0,
            "backend": "deterministic_contract_smoke",
        }
        for step in range(4)
    ]


def _checkpoint_artifacts(
    config: Phase1Config,
    *,
    run_root: Path,
) -> list[str]:
    written: list[str] = []
    for step in (0, 3):
        policy = run_root / "checkpoints" / f"policy_{_checkpoint(step)}.json"
        write_json_atomic(
            policy,
            {
                "schema_version": 1,
                "checkpoint": _checkpoint(step),
                "adapter": "theta" if config.method == "evorubrics" else None,
                "model": config.models["policy"]["model"],
                "revision": config.models["policy"]["revision"],
                "synthetic": True,
            },
        )
        written.append(str(policy.relative_to(run_root)))
        if config.method == "evorubrics":
            generator = (
                run_root / "checkpoints" / f"rubric_generator_{_checkpoint(step)}.json"
            )
            write_json_atomic(
                generator,
                {
                    "schema_version": 1,
                    "checkpoint": _checkpoint(step),
                    "adapter": "psi",
                    "shared_backbone": config.models["shared_backbone"]["model"],
                    "synthetic": True,
                },
            )
            written.append(str(generator.relative_to(run_root)))
    return written


def run_deterministic_smoke(
    config: Phase1Config,
    *,
    repo_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    root = Path(repo_root)
    run_root = config.run_root(root, run_id)
    complete_path = run_root / "smoke_complete.json"
    if complete_path.exists():
        complete = read_json(complete_path)
        if complete.get("config_hash") != config.config_hash:
            raise SmokeError("existing smoke run has a different resolved config")
        missing = [
            path
            for path in complete.get("required_artifacts", ())
            if not (run_root / str(path)).is_file()
        ]
        if missing:
            raise SmokeError(f"smoke resume is missing artifacts: {missing}")
        return {**complete, "resumed": True, "run_root": str(run_root)}

    smoke_config = config.raw.get("smoke", {})
    prompt_count = int(smoke_config.get("prompt_count", 2))
    stale_step = int(smoke_config.get("stale_checkpoint", 0))
    fresh_step = int(smoke_config.get("fresh_checkpoint", 3))
    if (stale_step, fresh_step) != (0, 3):
        raise SmokeError("deterministic smoke is fixed to checkpoints 0 and 3")

    prompts = load_probe_rows(config, repo_root=root, limit=prompt_count)
    pool_a, pool_b = _make_probe_pools(config, prompts, step=fresh_step)
    policy_checkpoint = _checkpoint(fresh_step)
    stale_checkpoint = (
        "rubric-step-000" if config.method == "online_rubrics" else "psi-step-000"
    )
    fresh_checkpoint = (
        "rubric-step-003" if config.method == "online_rubrics" else "psi-step-003"
    )

    rubric_before: list[dict[str, Any]] = []
    rubric_after: list[dict[str, Any]] = []
    rubric_sets: list[dict[str, Any]] = []
    cache = OnlineRubricShadowCache()
    comparisons: list[dict[str, Any]] = []
    reward_records: list[dict[str, Any]] = []
    grade_records: list[dict[str, Any]] = []
    state_records: list[dict[str, Any]] = []

    for prompt_index, prompt in enumerate(prompts):
        prompt_id = str(prompt["prompt_id"])
        a_rows = [row for row in pool_a if row["prompt_id"] == prompt_id]
        b_rows = [row for row in pool_b if row["prompt_id"] == prompt_id]
        if config.method == "online_rubrics":
            before, after, criteria = _criterion_records(
                config,
                prompt_id=prompt_id,
                pool_a_rows=a_rows,
                stale_checkpoint=stale_checkpoint,
                fresh_checkpoint=fresh_checkpoint,
            )
            rubric_before.extend(before)
            rubric_after.extend(after)
            r0_criteria = tuple(prompt.get("r0", {}).get("criteria", ()))
            stale_snapshot = OnlineRubricSnapshot(
                prompt_id=prompt_id,
                evaluator_checkpoint=stale_checkpoint,
                global_step=0,
                visit_index=0,
                rubric_id=f"{prompt_id}:rubric:0",
                criteria=r0_criteria or (criteria[0],),
                created_from_pool_a_response_ids=(),
            )
            fresh_snapshot = OnlineRubricSnapshot(
                prompt_id=prompt_id,
                evaluator_checkpoint=fresh_checkpoint,
                global_step=3,
                visit_index=1,
                rubric_id=f"{prompt_id}:rubric:3",
                criteria=tuple(r0_criteria) + (criteria[-1],),
                created_from_pool_a_response_ids=tuple(
                    str(row["response_id"]) for row in a_rows
                ),
            )
            cache.add(stale_snapshot)
            if cache.latest_before(prompt_id, global_step=3) != stale_snapshot:
                raise SmokeError("prompt-matched stale rubric lookup failed")
            cache.add(fresh_snapshot)
        else:
            seeds = tuple(int(value) for value in config.method_config["rubric_generation_seeds"])
            for role, evaluator in (("stale", stale_checkpoint), ("fresh", fresh_checkpoint)):
                for index, seed in enumerate(seeds):
                    source = a_rows[index]
                    rubric_sets.append(
                        {
                            **_common(
                                config,
                                step=3,
                                prompt_id=prompt_id,
                                response=str(source["response_id"]),
                                pool="probe_A",
                                policy_checkpoint=policy_checkpoint,
                                evaluator_checkpoint=evaluator,
                                role=role,
                            ),
                            "rubric_set_index": index,
                            "rubric_generation_seed": seed,
                            "rubric_set": [
                                {
                                    "criterion_id": f"{evaluator}:{prompt_id}:{index}",
                                    "criterion": f"Deterministic criterion {index}",
                                }
                            ],
                            "synthetic": True,
                        }
                    )
            stale_context = EvoScoreContext(
                evaluator_checkpoint=stale_checkpoint,
                evaluator_step=0,
                policy_checkpoint=policy_checkpoint,
                response_ids=tuple(str(row["response_id"]) for row in b_rows),
                judge_model=str(config.models["judge"]["model"]),
                rubric_sets_n=4,
                rubric_generation_seeds=seeds,
            )
            fresh_context = EvoScoreContext(
                evaluator_checkpoint=fresh_checkpoint,
                evaluator_step=3,
                policy_checkpoint=policy_checkpoint,
                response_ids=stale_context.response_ids,
                judge_model=stale_context.judge_model,
                rubric_sets_n=stale_context.rubric_sets_n,
                rubric_generation_seeds=stale_context.rubric_generation_seeds,
            )
            validate_evo_fresh_stale_pair(stale_context, fresh_context)

        stale_group, fresh_group = _score_groups(
            config,
            prompt_index=prompt_index,
            prompt_id=prompt_id,
            pool_b_rows=b_rows,
            stale_checkpoint=stale_checkpoint,
            fresh_checkpoint=fresh_checkpoint,
        )
        comparison = compare_fresh_stale(
            stale_group,
            fresh_group,
            epsilon_z=float(config.analysis["epsilon_z"]),
            epsilon_t=float(config.analysis["epsilon_t"]),
        )
        comparisons.append(comparison)
        stale_reward, stale_grades = _score_records(
            config, group=stale_group, pool_rows=b_rows, role="stale"
        )
        fresh_reward, fresh_grades = _score_records(
            config, group=fresh_group, pool_rows=b_rows, role="fresh"
        )
        validate_same_pool_b(stale_reward, fresh_reward)
        reward_records.extend((*stale_reward, *fresh_reward))
        grade_records.extend((*stale_grades, *fresh_grades))
        stale_metrics = comparison["stale"]
        state_records.append(
            {
                "domain": config.domain,
                "method": config.method,
                "seed": config.seed,
                "global_step": 3,
                "prompt_id": prompt_id,
                "evaluator_checkpoint": stale_checkpoint,
                "realized_update_value_v_adj_zar": comparison["v_adj_zar"],
                "evaluator_age_steps": 3,
                "evaluator_age_checkpoints": 1,
                "cumulative_prompts_since_evaluator": 288,
                "cumulative_completions_since_evaluator": (
                    288 * (16 if config.method == "online_rubrics" else 4)
                ),
                "cumulative_response_tokens_since_evaluator": (
                    288 * (16 if config.method == "online_rubrics" else 4) * 128
                ),
                "policy_kl_current_vs_evaluator": 0.0045,
                "cumulative_policy_kl_from_pi0": 0.0045,
                "response_length_shift": 3.0,
                "stale_zar": stale_metrics["exact_zero_advantage"],
                "stale_pairwise_tie_rate": stale_metrics["pairwise_tie_rate"],
                "stale_pairwise_separation_rate": stale_metrics[
                    "pairwise_separation_rate"
                ],
                "stale_effective_criterion_ratio": stale_metrics[
                    "effective_criterion_ratio"
                ],
                "stale_saturation_ratio": stale_metrics[
                    "saturated_criterion_ratio"
                ],
                "stale_reward_std": stale_metrics["reward_std"],
                "stale_top_median_margin": stale_metrics["top_median_margin"],
            }
        )

    validate_records((*reward_records, *grade_records))
    summary = aggregate_comparisons(
        comparisons, comparison_kind="adjacent_update_value"
    )
    state_hashes = prove_probe_side_effect_free(
        lambda enabled: {
            "policy": b"identical-policy-after-3-steps",
            "optimizer": b"identical-optimizer-after-3-steps",
            "rng": b"identical-rng-after-3-steps",
            "probe_enabled": "excluded-from-state" if False else 0,
        }
    )

    write_json_atomic(run_root / "config.resolved.json", dict(config.raw))
    write_json_atomic(
        run_root / "manifests" / "run.json",
        {
            "schema_version": 1,
            "experiment": config.experiment,
            "config_hash": config.config_hash,
            "domain": config.domain,
            "method": config.method,
            "seed": config.seed,
            "backend": "deterministic_contract_smoke",
            "hostname": socket.gethostname(),
            "static_grpo_baseline": False,
            "response_level_ground_truth": False,
            "created_unix_seconds": int(time.time()),
        },
    )
    write_json_atomic(
        run_root / "manifests" / "model_and_topology.json",
        {
            "models": dict(config.models),
            "infrastructure": dict(config.raw["infrastructure"]),
            "secrets_persisted": False,
            "model_weights_loaded": False,
            "remote_endpoints_called": False,
        },
    )
    write_jsonl_atomic(run_root / "responses" / "probe_A.jsonl", pool_a)
    write_jsonl_atomic(run_root / "responses" / "probe_B.jsonl", pool_b)
    write_jsonl_atomic(run_root / "grades" / "criterion_grades.jsonl", grade_records)
    write_jsonl_atomic(run_root / "grades" / "rewards_and_advantages.jsonl", reward_records)
    write_jsonl_atomic(run_root / "metrics" / "prompt_comparisons.jsonl", comparisons)
    write_json_atomic(run_root / "metrics" / "adjacent_update_value.json", summary)
    write_jsonl_atomic(run_root / "metrics" / "pre_update_state.jsonl", state_records)
    write_jsonl_atomic(run_root / "logs" / "global_steps.jsonl", _global_step_logs(config))
    write_json_atomic(
        run_root / "logs" / "probe_side_effect_contract.json",
        {
            "deterministic_contract_verified": True,
            "live_trainer_verified": False,
            "state_hashes": state_hashes,
        },
    )
    if config.method == "online_rubrics":
        write_jsonl_atomic(
            run_root / "rubrics" / "extracted_before_dedup.jsonl", rubric_before
        )
        write_jsonl_atomic(
            run_root / "rubrics" / "extracted_after_dedup.jsonl", rubric_after
        )
        write_jsonl_atomic(
            run_root / "rubrics" / "prompt_matched_shadow_cache.jsonl",
            cache.records(),
        )
    else:
        write_jsonl_atomic(
            run_root / "rubrics" / "generated_rubric_sets.jsonl", rubric_sets
        )
    checkpoint_paths = _checkpoint_artifacts(config, run_root=run_root)

    required = [
        "config.resolved.json",
        "manifests/run.json",
        "manifests/model_and_topology.json",
        "responses/probe_A.jsonl",
        "responses/probe_B.jsonl",
        "grades/criterion_grades.jsonl",
        "grades/rewards_and_advantages.jsonl",
        "metrics/prompt_comparisons.jsonl",
        "metrics/adjacent_update_value.json",
        "metrics/pre_update_state.jsonl",
        "logs/global_steps.jsonl",
        "logs/probe_side_effect_contract.json",
        *checkpoint_paths,
    ]
    required.extend(
        [
            "rubrics/extracted_before_dedup.jsonl",
            "rubrics/extracted_after_dedup.jsonl",
            "rubrics/prompt_matched_shadow_cache.jsonl",
        ]
        if config.method == "online_rubrics"
        else ["rubrics/generated_rubric_sets.jsonl"]
    )
    complete = {
        "schema_version": 1,
        "status": "passed",
        "backend": "deterministic_contract_smoke",
        "config_hash": config.config_hash,
        "domain": config.domain,
        "method": config.method,
        "prompt_count": len(prompts),
        "pool_a_response_count": len(pool_a),
        "pool_b_response_count": len(pool_b),
        "pool_a_b_disjoint": True,
        "same_pool_b_fresh_stale": True,
        "prompt_matched_stale_lookup": config.method == "online_rubrics",
        "evo_invariants_verified": config.method == "evorubrics",
        "checkpoint_saving_verified": True,
        "probe_side_effect_contract_verified": True,
        "live_trainer_probe_side_effect_verified": False,
        "metric_computation_verified": True,
        "model_weights_loaded": False,
        "remote_generation_and_grading_verified": False,
        "full_training_authorized": False,
        "required_next_gate": "live_model_smoke",
        "adjacent_update_value": summary,
        "required_artifacts": required,
        "run_root": str(run_root),
        "resumed": False,
    }
    write_json_atomic(complete_path, complete)
    return complete


def run_from_path(
    config_path: str | Path,
    *,
    repo_root: str | Path,
    run_id: str,
) -> dict[str, Any]:
    return run_deterministic_smoke(
        load_phase1_config(config_path),
        repo_root=repo_root,
        run_id=run_id,
    )
