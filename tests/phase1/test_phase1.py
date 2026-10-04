from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from dynamic_rubric.phase1.analysis import reuse_horizon_matrix
from dynamic_rubric.phase1.config import REUSE_ANCHORS, load_phase1_config
from dynamic_rubric.phase1.metrics import ScoreGroup, compare_fresh_stale
from dynamic_rubric.phase1.provenance import prepare_fixed_train_probe_manifest
from dynamic_rubric.phase1.shadow import (
    EvoScoreContext,
    OnlineRubricShadowCache,
    OnlineRubricSnapshot,
    ShadowEvaluatorError,
    validate_evo_fresh_stale_pair,
)
from dynamic_rubric.phase1.smoke import run_deterministic_smoke

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_ROOT = REPO_ROOT / "configs" / "phase1"


def _config(domain: str, method: str):
    return load_phase1_config(CONFIG_ROOT / f"{domain}_{method}.yaml")

@pytest.mark.parametrize("domain", ["medicine", "science"])
@pytest.mark.parametrize("method", ["online_rubrics", "evorubrics"])
def test_all_phase1_configs_resolve(domain: str, method: str) -> None:
    config = _config(domain, method)
    assert config.domain == domain
    assert config.method == method
    assert config.raw["analysis"]["enable_bon"] is False
    assert config.raw["analysis"]["use_initial_rubric_as_ground_truth"] is False
    assert config.raw["tracking"]["project"] != config.raw["tracking"]["legacy_static_project"]
    assert config.checkpoint_steps == tuple(range(1, 49))
    assert config.raw["infrastructure"]["optimizer"]["gpus"] == [1]
    services = config.raw["infrastructure"]["services"]
    assert services["gpt_oss_120b"]["base_url_env"].endswith("BASE_URLS")
    assert services["qwen3_32b"]["base_url_env"].endswith("BASE_URLS")
    assert [item["gpus"] for item in services["gpt_oss_120b"]["instances"]] == [
        [0, 1],
    ]
    assert services["qwen3_32b"]["hosts"] == [""]
    assert [item["host"] for item in services["qwen3_32b"]["instances"]] == [
        "",
    ]
    assert [item["gpus"] for item in services["qwen3_32b"]["instances"]] == [
        [0, 1],
    ]
    assert [
        item["tensor_parallel_size"]
        for item in services["qwen3_32b"]["instances"]
    ] == [2]


def _absolute_train(config):
    return replace(
        config,
        raw={
            **config.raw,
            "data": {
                **config.data,
                "train_path": str(REPO_ROOT / config.data["train_path"]),
            },
        },
    )


def test_fixed_probe_is_shared_across_methods(tmp_path: Path) -> None:
    online = _absolute_train(_config("medicine", "online_rubrics"))
    evo = _absolute_train(_config("medicine", "evorubrics"))
    online_path, online_manifest = prepare_fixed_train_probe_manifest(
        online, repo_root=tmp_path
    )
    evo_path, evo_manifest = prepare_fixed_train_probe_manifest(
        evo, repo_root=tmp_path
    )
    assert online_path == evo_path
    assert online_manifest["probe_prompt_count"] == 100
    assert online_manifest["prompt_ids_sha256"] == evo_manifest["prompt_ids_sha256"]


def test_online_stale_cache_is_prompt_matched() -> None:
    cache = OnlineRubricShadowCache()
    first = OnlineRubricSnapshot(
        prompt_id="p1",
        evaluator_checkpoint="r0",
        global_step=0,
        visit_index=0,
        rubric_id="p1-r0",
        criteria=(),
        created_from_pool_a_response_ids=(),
    )
    other = OnlineRubricSnapshot(
        prompt_id="p2",
        evaluator_checkpoint="r2",
        global_step=2,
        visit_index=0,
        rubric_id="p2-r2",
        criteria=(),
        created_from_pool_a_response_ids=(),
    )
    cache.add(first)
    cache.add(other)
    assert cache.latest_before("p1", global_step=3) is first
    assert cache.latest_before("unseen", global_step=3) is None


def test_evo_comparison_fails_when_pool_changes() -> None:
    stale = EvoScoreContext("psi0", 0, "pi3", ("a", "b"), "judge", 4, (1, 2, 3, 4))
    fresh = EvoScoreContext("psi3", 3, "pi3", ("a", "c"), "judge", 4, (1, 2, 3, 4))
    with pytest.raises(ShadowEvaluatorError, match="response_ids"):
        validate_evo_fresh_stale_pair(stale, fresh)


def test_metric_orientation_and_same_pool_contract() -> None:
    stale = ScoreGroup("p", "r0", "pi3", ("a", "b"), (0.5, 0.5))
    fresh = ScoreGroup("p", "r3", "pi3", ("a", "b"), (0.2, 0.8))
    result = compare_fresh_stale(stale, fresh, epsilon_z=0.01, epsilon_t=0.01)
    assert result["v_adj_zar"] == 1
    assert result["delta_tie_rate"] == 1
    assert result["delta_separation_rate"] == 1
    assert result["incremental_tie_resolution"] == 1
    assert result["incremental_tie_resolution_unconditional"] == 1
    assert result["conditional_tie_resolution"] == 1
    assert result["kendall_tau_b"] is None
    changed = ScoreGroup("p", "r3", "pi3", ("a", "c"), (0.2, 0.8))
    with pytest.raises(ValueError, match="identical ordered response IDs"):
        compare_fresh_stale(stale, changed, epsilon_z=0.01, epsilon_t=0.01)


def test_reuse_horizon_matrix_uses_only_fixed_probe() -> None:
    groups = {}
    prompt_ids = ("p",)
    for policy_step in REUSE_ANCHORS:
        current = ScoreGroup(
            "p",
            f"e{policy_step}",
            f"pi{policy_step}",
            ("a", "b"),
            (0.2, 0.8),
        )
        groups[(policy_step, policy_step, "p")] = current
        for evaluator_step in REUSE_ANCHORS:
            if evaluator_step > policy_step:
                continue
            groups[(evaluator_step, policy_step, "p")] = (
                current
                if evaluator_step == policy_step
                else ScoreGroup(
                    "p",
                    f"e{evaluator_step}",
                    f"pi{policy_step}",
                    ("a", "b"),
                    (0.5, 0.5) if policy_step >= 32 else (0.2, 0.8),
                )
            )
    result = reuse_horizon_matrix(
        groups,
        prompt_ids=prompt_ids,
        epsilon_z=0.01,
        epsilon_t=0.01,
        practical_margin_delta_d=0.2,
    )
    assert len(result["matrix"]) == 15
    assert result["heldout_used"] is False
    assert result["actual_training_batches_used"] is False
    assert result["empirical_reuse_horizon"]["0"] == 16


@pytest.mark.parametrize("method", ["online_rubrics", "evorubrics"])
def test_deterministic_smoke_and_resume(method: str, tmp_path: Path) -> None:
    config = _absolute_train(_config("medicine", method))
    first = run_deterministic_smoke(
        config, repo_root=tmp_path, run_id="pytest-smoke"
    )
    second = run_deterministic_smoke(
        config, repo_root=tmp_path, run_id="pytest-smoke"
    )
    assert first["status"] == "passed"
    assert first["pool_a_b_disjoint"] is True
    assert first["same_pool_b_fresh_stale"] is True
    assert first["full_training_authorized"] is False
    assert first["model_weights_loaded"] is False
    assert first["probe_side_effect_contract_verified"] is True
    assert first["live_trainer_probe_side_effect_verified"] is False
    assert second["resumed"] is True
