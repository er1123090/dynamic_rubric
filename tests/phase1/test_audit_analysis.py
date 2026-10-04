import json

import pytest

from dynamic_rubric.phase1.audit_analysis import (
    AuditAnalysisError,
    analyze_records,
    main as audit_analysis_main,
    pooled_criterion_summary,
    run_audit_analysis,
    run_training_group_analysis,
    training_comparison_row,
    score_groups,
)


def _saved_checkpoint(run_root, step):
    actor = run_root / "verl-run" / "checkpoints" / f"global_step_{step}" / "actor"
    actor.mkdir(parents=True)
    (actor / "fsdp_config.json").write_text('{"world_size": 1}\n')
    (actor / "model_world_size_1_rank_0.pt").write_bytes(b"parameters")


def _grade_inventory(prompt, evaluator, policy):
    if (policy, evaluator) == (2, 0):
        return ["c"] if prompt == "p1" else ["a", "b", "c"]
    return ["c"]


def _grades(prompt, evaluator, policy, response_index):
    criteria = _grade_inventory(prompt, evaluator, policy)
    if (policy, evaluator) == (2, 0) and prompt == "p2":
        return [[criterion, 1] for criterion in criteria]
    return [[criterion, response_index] for criterion in criteria]


def _rewards(evaluator, policy):
    if evaluator == 0 and policy in (2, 3):
        return (0.5, 0.5)
    return (0.2, 0.8)


def _score_rows(policy_steps=(0, 2, 3)):
    rows = []
    anchors = (0, 2)
    for policy in policy_steps:
        evaluators = [step for step in anchors if step <= policy]
        if policy not in evaluators:
            evaluators.append(policy)
        for evaluator in evaluators:
            for prompt in ("p1", "p2"):
                for index, reward in enumerate(_rewards(evaluator, policy)):
                    rows.append(
                        {
                            "schema_version": 1,
                            "pool": "probe_B",
                            "policy_step": policy,
                            "evaluator_step": evaluator,
                            "policy_checkpoint": str(policy),
                            "evaluator_checkpoint": str(evaluator),
                            "prompt_id": prompt,
                            "response_id": f"p{policy}-{prompt}-r{index}",
                            "reward": reward,
                            "grades": _grades(prompt, evaluator, policy, index),
                        }
                    )
    return rows


def _states():
    return [
        {
            "policy_step": 0,
            "cumulative_prompt_exposures": 0,
            "cumulative_completions": 0,
            "cumulative_response_tokens": 0,
            "response_length": 5,
        },
        {
            "policy_step": 2,
            "cumulative_prompt_exposures": 20,
            "cumulative_completions": 40,
            "cumulative_response_tokens": 100,
            "response_length": 7,
        },
        {
            "policy_step": 3,
            "cumulative_prompt_exposures": 30,
            "cumulative_completions": 60,
            "cumulative_response_tokens": 160,
            "response_length": 8,
        },
    ]


def _distances():
    return [
        {
            "current_policy_step": 2,
            "stale_policy_step": 0,
            "response_policy_step": 2,
            "prompt_balanced_sampled_kl_mean": 0.2,
        },
        {
            "current_policy_step": 3,
            "stale_policy_step": 0,
            "response_policy_step": 3,
            "prompt_balanced_sampled_kl_mean": 0.3,
        },
        {
            "current_policy_step": 3,
            "stale_policy_step": 2,
            "response_policy_step": 3,
            "prompt_balanced_sampled_kl_mean": 0.1,
        },
    ]


def test_triangle_pooled_metrics_censoring_and_preupdate_join():
    report = analyze_records(
        _score_rows(),
        _states(),
        _distances(),
        anchors=(0, 2),
        exploratory_step=3,
        epsilon_z=0.01,
        epsilon_t=0.01,
        practical_margin_delta_d=0.2,
        bootstrap_iterations=100,
        bootstrap_seed=17,
    )
    stale = next(
        row for row in report["triangle"] if row["evaluator_step"] == 0 and row["policy_step"] == 2
    )
    assert stale["stale_zar"] == 1
    assert stale["fresh_zar"] == 0
    assert stale["l_zar"] == 1
    assert stale["delta_tie_rate"] == 1
    assert stale["conditional_tie_resolution"] == 1
    assert stale["criterion_pooled"]["stale"]["effective_criterion_ratio"] == 0.25
    assert stale["criterion_pooled"]["fresh"]["effective_criterion_ratio"] == 1
    assert stale["criterion_pooled"]["delta_effective_criterion_ratio"] == 0.75
    assert stale["bootstrap_95ci"]["l_zar"]["ci_low"] == 1
    anchor_zero = report["reuse_horizon"][0]
    assert anchor_zero["last_acceptable_policy_step"] == 0
    assert anchor_zero["first_exceedance_policy_step"] == 2
    assert anchor_zero["right_censored"] is False
    assert report["reuse_horizon"][1]["right_censored"] is True
    assert len(report["adjacent"]) == 2
    joined = next(
        row
        for row in report["preupdate_stale_state"]
        if row["evaluator_step"] == 0 and row["global_step"] == 2
    )
    assert joined["cumulative_prompts_since_evaluator"] == 20
    assert joined["cumulative_response_tokens_since_evaluator"] == 100
    assert joined["policy_kl_current_vs_evaluator"] == 0.2
    assert joined["response_length_shift"] == 2
    assert all(row["predictor"] != "stale_zar" for row in report["exploratory_associations"])
    assert report["guards"]["step_34_excluded_from_confirmatory_triangle_and_horizon"]


def test_criterion_pooling_is_not_mean_of_prompt_ratios():
    groups = score_groups(_score_rows(policy_steps=(0, 2)))
    summary = pooled_criterion_summary([groups[(2, 0, "p1")], groups[(2, 0, "p2")]])
    assert summary["effective_criterion_ratio"] == 0.25
    assert summary["effective_criterion_ratio"] != 0.5


def test_same_pool_response_identity_is_enforced():
    rows = _score_rows()
    target = next(
        row
        for row in rows
        if row["policy_step"] == 2
        and row["evaluator_step"] == 0
        and row["prompt_id"] == "p1"
        and row["response_id"].endswith("r1")
    )
    target["response_id"] = "changed"
    with pytest.raises(ValueError, match="identical ordered response IDs"):
        analyze_records(
            rows,
            _states(),
            _distances(),
            anchors=(0, 2),
            exploratory_step=3,
            epsilon_z=0.01,
            epsilon_t=0.01,
            practical_margin_delta_d=0.2,
            bootstrap_iterations=10,
        )


def test_policy_distance_rejects_off_policy_pool():
    distances = _distances()
    distances[0]["response_policy_step"] = 0
    with pytest.raises(AuditAnalysisError, match="current policy's pool"):
        analyze_records(
            _score_rows(),
            _states(),
            distances,
            anchors=(0, 2),
            exploratory_step=3,
            epsilon_z=0.01,
            epsilon_t=0.01,
            practical_margin_delta_d=0.2,
            bootstrap_iterations=10,
        )


def test_restartable_file_report(tmp_path):
    score_path = tmp_path / "scores.jsonl"
    score_path.write_text(
        "\n".join(json.dumps(row) for row in _score_rows(policy_steps=(0,))) + "\n"
    )
    state_path = tmp_path / "state.jsonl"
    state_path.write_text(json.dumps(_states()[0]) + "\n")
    distance_path = tmp_path / "policy_distance_summary.json"
    distance_path.write_text(
        json.dumps(
            {
                "current_policy_step": 0,
                "stale_policy_step": 0,
                "response_policy_step": 0,
                "prompt_balanced_sampled_kl_mean": 0.0,
            }
        )
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "epsilon_z": 0.01,
                    "epsilon_t": 0.01,
                    "practical_margin_delta_d": 0.02,
                }
            }
        )
    )
    output = tmp_path / "analysis"
    first = run_audit_analysis(
        score_paths=[score_path],
        policy_state_path=state_path,
        policy_distance_paths=[distance_path],
        config_path=config_path,
        output_dir=output,
        through_step=0,
        bootstrap_iterations=10,
    )
    second = run_audit_analysis(
        score_paths=[score_path],
        policy_state_path=state_path,
        policy_distance_paths=[distance_path],
        config_path=config_path,
        output_dir=output,
        through_step=0,
        bootstrap_iterations=10,
        resume=True,
    )
    assert first["status"] == "completed"
    assert second["resumed"] is True
    assert json.loads((output / "report.json").read_text())["anchors"] == [0]


def test_all_saved_cli_defaults_to_latest_discovered_and_includes_step34(tmp_path):
    run_root = tmp_path / "run"
    for step in (0, 2, 3):
        _saved_checkpoint(run_root, step)
    tracker = run_root / "verl-run" / "checkpoints" / "latest_checkpointed_iteration.txt"
    tracker.write_text("3\n")
    score_path = tmp_path / "scores.jsonl"
    score_path.write_text("\n".join(json.dumps(row) for row in _score_rows()) + "\n")
    state_path = tmp_path / "state.jsonl"
    state_path.write_text("\n".join(json.dumps(row) for row in _states()) + "\n")
    distance_path = tmp_path / "policy_distance_summary.json"
    distance_path.write_text(json.dumps(_distances()))
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(
            {
                "analysis": {
                    "epsilon_z": 0.01,
                    "epsilon_t": 0.01,
                    "practical_margin_delta_d": 0.02,
                }
            }
        )
    )
    output = tmp_path / "all-saved-analysis"

    audit_analysis_main(
        [
            "--all-saved-checkpoints-from",
            str(run_root),
            "--scores",
            str(score_path),
            "--policy-state",
            str(state_path),
            "--policy-distance",
            str(distance_path),
            "--config",
            str(config_path),
            "--output",
            str(output),
            "--bootstrap-iterations",
            "10",
        ]
    )

    report = json.loads((output / "report.json").read_text())
    assert report["anchors"] == [0, 2, 3]
    assert len(report["triangle"]) == 6
    assert len(report["adjacent"]) == 2
    assert report["guards"]["step_34_excluded_from_confirmatory_triangle_and_horizon"] is False


def test_training_group_cli_rejects_all_saved_checkpoint_option(tmp_path):
    with pytest.raises(SystemExit):
        audit_analysis_main(
            [
                "--training-groups",
                str(tmp_path / "groups.jsonl"),
                "--all-saved-checkpoints-from",
                str(tmp_path / "run"),
                "--output",
                str(tmp_path / "output"),
            ]
        )


def _training_group(step, prompt, stale_ids=None):
    ids = [f"{prompt}-r0", f"{prompt}-r1"]
    stale_response_ids = stale_ids or ids
    return {
        "schema_version": 1,
        "domain": "medicine",
        "method": "online_rubrics",
        "seed": 11,
        "global_step": step,
        "policy_step": step - 1,
        "evaluator_step": 0,
        "prompt_id": prompt,
        "pool": "train_batch",
        "fresh_creation_update": step,
        "stale_creation_update": 1,
        "evaluator_age_steps": step - 1,
        "clock": {
            "cumulative_prompt_exposures": 1500 + step,
            "cumulative_completions": (1500 + step) * 16,
            "cumulative_prompts_since_evaluator": 1500,
            "cumulative_completions_since_evaluator": 24000,
        },
        "fresh": [
            {"response_id": ids[0], "rollout_index": 0, "reward": 0.2, "grades": [["c", 0]]},
            {"response_id": ids[1], "rollout_index": 1, "reward": 0.8, "grades": [["c", 1]]},
        ],
        "stale": [
            {
                "response_id": stale_response_ids[0],
                "rollout_index": 0,
                "reward": 0.5,
                "grades": [["old", 1]],
            },
            {
                "response_id": stale_response_ids[1],
                "rollout_index": 1,
                "reward": 0.5,
                "grades": [["old", 1]],
            },
        ],
    }


def test_training_group_report_has_weighted_metrics_and_explicit_scope(tmp_path):
    groups = tmp_path / "groups"
    for step, prompt in ((17, "p1"), (18, "p2")):
        directory = groups / f"step-{step:06d}"
        directory.mkdir(parents=True)
        (directory / f"{prompt}.json").write_text(json.dumps(_training_group(step, prompt)))
    output = tmp_path / "report"
    result = run_training_group_analysis(
        training_groups=groups,
        output_dir=output,
        expected_groups=2,
        epsilon_z=0.01,
        epsilon_t=0.01,
        bootstrap_iterations=50,
        bootstrap_seed=7,
    )
    report = json.loads((output / "report.json").read_text())
    rows = [
        json.loads(line) for line in (output / "comparison_rows.jsonl").read_text().splitlines()
    ]
    assert result["status"] == "final"
    assert report["coverage"]["observed_groups"] == 2
    assert report["overall"]["metrics"]["v_adj_zar"] == 1
    assert report["overall"]["metrics"]["incremental_tie_resolution_weighted"] == 1
    assert report["overall"]["fresh_criterion_counts"]["effective"] == 2
    assert report["overall"]["stale_criterion_counts"]["saturated"] == 2
    assert report["guards"]["fixed_train_probe_used"] is False
    assert report["guards"]["reuse_horizon_triangle_available"] is False
    assert all(row["same_response_pool"] and not row["same_pool_b"] for row in rows)
    assert "operational/in-sample" in (output / "summary.md").read_text()


def test_training_group_report_marks_partial_coverage(tmp_path):
    groups = tmp_path / "groups/step-000017"
    groups.mkdir(parents=True)
    (groups / "p1.json").write_text(json.dumps(_training_group(17, "p1")))
    result = run_training_group_analysis(
        training_groups=groups.parent,
        output_dir=tmp_path / "report",
        expected_groups=2,
        epsilon_z=0.01,
        epsilon_t=0.01,
        bootstrap_iterations=10,
    )
    assert result["status"] == "interim"
    assert result["missing_groups"] == 1


def test_training_group_rejects_fresh_stale_response_mismatch():
    group = _training_group(17, "p1", stale_ids=["p1-r0", "different"])
    with pytest.raises(ValueError, match="identical ordered response IDs"):
        training_comparison_row(group, epsilon_z=0.01, epsilon_t=0.01)
