from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from .artifacts import read_jsonl
from .batch_dynamic import (
    SUPPORTED_BATCH_MODES,
    collect_dynamic_batch,
    dynamic_batch_stage,
    dynamic_batch_status,
    prepare_dynamic_batch,
    submit_dynamic_batch,
)
from .config import load_config
from .data.rar import prepare_rar_source
from .horizon.orchestration import (
    analyze_horizon_observations,
    estimate_horizon_cost,
    validate_horizon_launch_environment,
    verify_horizon_models,
)
from .horizon.batch_rubrics import build_horizon_rubrics_batch_from_files
from .horizon.control_artifacts import attach_control_extensions_from_files
from .horizon.sync_rubrics import build_horizon_rubrics_sync_from_files
from .horizon.live_grading import (
    grade_horizon_pool_a_combined_from_files,
    grade_horizon_pool_a_from_files,
    grade_horizon_pool_b_from_files,
)
from .horizon.observations import build_horizon_observations
from .horizon.pools import (
    PoolSpec,
    combine_pool_a_with_fixed_control,
    generate_pool_rows,
    publish_pool_shard,
    validate_horizon_pool_inventory,
)
from .live_bon import (
    extend_live_bon_worker,
    finalize_live_bon,
    run_live_bon_worker,
)
from .live_preflight import run_preflight
from .pipeline import (
    PipelineContext,
    run_freeze_updater,
    run_generate_static,
    run_prepare_data,
    run_replay_dynamic,
    run_train_online,
    run_train_static,
)
from .pipeline_audit import (
    estimate_cost_counts,
    run_analyze,
    run_audit_gold,
    run_export_audit_package,
    run_validate_inventory,
)
from .pipeline_evaluation import run_generate_bon, run_score_proxy, run_select_bon
from .providers.fake import FakeGenerator
from .providers.vllm_generation import VLLMPolicyGenerator
from .providers.vllm import VLLMCriterionGrader, VLLMIdentity
from .training.live_online import validate_online_step_artifact
from .training.verl_dataset import write_rar_verl_parquets


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _common(subparser: argparse.ArgumentParser, *, run_required: bool = True) -> None:
    subparser.add_argument(
        "--config",
        default="configs/pilot.yaml",
        help="project-relative YAML/JSON config; its canonical hash is recorded in every stage manifest",
    )
    subparser.add_argument(
        "--run-id",
        required=run_required,
        help="immutable run namespace; resume requires a byte-compatible manifest",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dynamic_rubric",
        description="Replay-first Dynamic Rubric Staleness Audit. Live stages fail closed until preflight passes.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True, metavar="STAGE")

    validate = subparsers.add_parser(
        "validate-config", help="validate config and public/private stage boundaries"
    )
    _common(validate, run_required=False)

    prepare = subparsers.add_parser(
        "prepare-data", help="secure HealthBench ingestion and deterministic split"
    )
    _common(prepare, run_required=False)
    prepare.add_argument("--source", default="data/source/healthbench_consensus.jsonl")

    preflight = subparsers.add_parser(
        "preflight", help="verify revisions, aliases, API, model and reward hooks"
    )
    _common(preflight)

    static = subparsers.add_parser(
        "generate-static", help="generate pi_0 candidates and immutable 6+2 R_0"
    )
    _common(static)

    train = subparsers.add_parser(
        "train-static", help="run/export static-R_0-only GRPO and post-update probes"
    )
    _common(train)

    train_online = subparsers.add_parser(
        "train-online",
        help="run paper-faithful same-step OnlineRubrics GRPO",
    )
    _common(train_online)

    resume_online = subparsers.add_parser(
        "resume-online",
        help="resume a sealed train-online run from its latest committed step",
    )
    _common(resume_online)

    validate_online = subparsers.add_parser(
        "validate-online-step",
        help="validate one sealed online step and its optional provider receipts",
    )
    _common(validate_online, run_required=False)
    validate_online.add_argument("--run-dir", required=True)
    validate_online.add_argument("--step", required=True, type=int)
    validate_online.add_argument("--require-provider-receipts", action="store_true")

    replay = subparsers.add_parser(
        "replay-dynamic", help="post-hoc fixed/prev/refresh/cumulative replay"
    )
    _common(replay)
    replay.add_argument("--split", required=True, choices=("development", "final"))

    freeze = subparsers.add_parser(
        "freeze-updater", help="freeze development-only updater_lock.json"
    )
    _common(freeze)

    bon = subparsers.add_parser(
        "generate-bon", help="generate shared focal-policy BoN candidate pools"
    )
    _common(bon)

    live_bon = subparsers.add_parser(
        "generate-bon-live", help="generate one checkpoint BoN pool from a live vLLM endpoint"
    )
    _common(live_bon)
    live_bon.add_argument("--policy-step", type=int, required=True)
    live_bon.add_argument("--base-url", required=True)
    live_bon.add_argument("--model", required=True)
    live_bon.add_argument("--concurrency", type=int, default=64)

    extend_bon = subparsers.add_parser(
        "extend-bon-live", help="extend one completed live BoN checkpoint pool in resumable chunks"
    )
    _common(extend_bon)
    extend_bon.add_argument("--policy-step", type=int, required=True)
    extend_bon.add_argument("--base-url", required=True)
    extend_bon.add_argument("--model", required=True)
    extend_bon.add_argument("--target-pool-size", type=int, required=True)
    extend_bon.add_argument("--concurrency", type=int, default=64)
    extend_bon.add_argument("--chunk-size", type=int, default=64)
    extend_bon.add_argument("--prompt-shard-index", type=int, default=0)
    extend_bon.add_argument("--prompt-shard-count", type=int, default=1)

    finalize_bon = subparsers.add_parser(
        "finalize-bon-live", help="combine completed checkpoint BoN pools"
    )
    _common(finalize_bon)
    finalize_bon.add_argument("--policy-steps", type=int, nargs="+", required=True)
    finalize_bon.add_argument("--target-pool-size", type=int)

    prepare_batch = subparsers.add_parser(
        "prepare-dynamic-batch", help="prepare fixed-control dynamic-rubric Batch inputs"
    )
    _common(prepare_batch)
    prepare_batch.add_argument("--max-step", type=int, default=50)
    prepare_batch.add_argument(
        "--mode", choices=SUPPORTED_BATCH_MODES, default=SUPPORTED_BATCH_MODES[0]
    )

    submit_batch = subparsers.add_parser(
        "submit-dynamic-batch", help="upload and submit prepared OpenAI Batch inputs"
    )
    _common(submit_batch)
    submit_batch.add_argument(
        "--mode", choices=SUPPORTED_BATCH_MODES, default=SUPPORTED_BATCH_MODES[0]
    )

    status_batch = subparsers.add_parser(
        "status-dynamic-batch", help="refresh OpenAI Batch job status"
    )
    _common(status_batch)
    status_batch.add_argument(
        "--mode", choices=SUPPORTED_BATCH_MODES, default=SUPPORTED_BATCH_MODES[0]
    )

    collect_batch = subparsers.add_parser(
        "collect-dynamic-batch", help="download and validate completed dynamic-rubric Batch outputs"
    )
    _common(collect_batch)
    collect_batch.add_argument(
        "--mode", choices=SUPPORTED_BATCH_MODES, default=SUPPORTED_BATCH_MODES[0]
    )

    proxy = subparsers.add_parser(
        "score-proxy", help="score each response/criterion once with frozen proxy"
    )
    _common(proxy)

    select = subparsers.add_parser(
        "select-bon", help="deterministic N/permutation selection for every rubric"
    )
    _common(select)

    export = subparsers.add_parser(
        "export-audit-package", help="export only selected unique responses for the private audit"
    )
    _common(export)

    audit = subparsers.add_parser(
        "audit-gold",
        help="PRIVATE PROCESS: grade selected unique responses against physician rubrics",
    )
    _common(audit)
    audit.add_argument(
        "--private-gt",
        default="data/private_gt/healthbench_gold_rubrics.jsonl",
        help="audit-only read-only mount; this option is intentionally absent from every public stage",
    )

    analyze = subparsers.add_parser(
        "analyze", help="metrics, paired bootstrap, interpretation and report"
    )
    _common(analyze)

    inventory = subparsers.add_parser(
        "validate-inventory", help="verify counts, hashes, seeds and boundaries"
    )
    _common(inventory)

    estimate = subparsers.add_parser(
        "estimate-cost", help="deterministic request/pair/count upper bounds"
    )
    _common(estimate, run_required=False)

    prepare_rar = subparsers.add_parser(
        "prepare-rar-data", help="normalize and split RaR Medicine/Science source data"
    )
    _common(prepare_rar, run_required=False)
    prepare_rar.add_argument("--source", required=True)

    export_rar = subparsers.add_parser(
        "export-rar-verl", help="export RaR train/development parquet inputs for veRL"
    )
    _common(export_rar)
    export_rar.add_argument("--output")

    horizon_cost = subparsers.add_parser(
        "estimate-horizon", help="estimate policy/extractor request counts before execution"
    )
    _common(horizon_cost, run_required=False)

    verify_models = subparsers.add_parser(
        "verify-horizon-models", help="verify pinned local Qwen snapshots and veRL sources"
    )
    _common(verify_models, run_required=False)

    verify_launch = subparsers.add_parser(
        "validate-horizon-launch",
        help="fail closed if GRPO launcher environment differs from the horizon config",
    )
    _common(verify_launch, run_required=False)

    generate_pools = subparsers.add_parser(
        "generate-horizon-pools", help="generate one immutable horizon pool shard"
    )
    _common(generate_pools)
    generate_pools.add_argument("--prompts", required=True)
    generate_pools.add_argument(
        "--pool-family",
        required=True,
        choices=("fixed_control", "sham_control", "pool_a", "pool_b"),
    )
    generate_pools.add_argument("--count", required=True, type=int)
    generate_pools.add_argument("--policy-step", required=True, type=int)
    generate_pools.add_argument("--training-seed", type=int)
    generate_pools.add_argument("--checkpoint-hash", required=True)
    generate_pools.add_argument("--output", required=True)
    generate_pools.add_argument("--base-url")

    combine_pool_a = subparsers.add_parser(
        "combine-horizon-pool-a",
        help="combine Pool A current responses with fixed pi0 controls for 16-response analysis",
    )
    _common(combine_pool_a, run_required=False)
    combine_pool_a.add_argument("--pool-a", required=True)
    combine_pool_a.add_argument("--fixed-control", required=True)
    combine_pool_a.add_argument("--output", required=True)

    analyze_horizon = subparsers.add_parser(
        "analyze-horizon", help="run crossed inference and emit horizon decision/report"
    )
    _common(analyze_horizon, run_required=False)
    analyze_horizon.add_argument("--observations", required=True)
    analyze_horizon.add_argument("--output", required=True)
    analyze_horizon.add_argument("--iterations", type=int)

    validate_horizon = subparsers.add_parser(
        "validate-horizon-inventory",
        help="validate response-pool counts, namespaces, and collisions",
    )
    _common(validate_horizon, run_required=False)
    validate_horizon.add_argument("--pools", nargs="+", required=True)
    validate_horizon.add_argument("--prompts", required=True)

    build_rubrics = subparsers.add_parser(
        "build-horizon-rubrics",
        help="run resumable two-stage OpenAI Responses extraction/dedup for one checkpoint",
    )
    _common(build_rubrics)
    build_rubrics.add_argument("--prompts", required=True)
    build_rubrics.add_argument("--current-pool", required=True)
    build_rubrics.add_argument("--control-pool", required=True)
    build_rubrics.add_argument("--checkpoint-id", required=True)
    build_rubrics.add_argument("--output", required=True)
    build_rubrics.add_argument("--control-rubrics")
    build_rubrics.add_argument(
        "--batch-poll-interval-seconds",
        type=float,
        default=60.0,
        help="seconds between automatic OpenAI Batch status polls",
    )
    build_rubrics.add_argument(
        "--api-mode",
        choices=("batch", "sync"),
        help="OpenAI execution mode; defaults to rubric_extractor.api_mode or batch",
    )
    build_rubrics.add_argument(
        "--sync-concurrency",
        type=int,
        help="maximum concurrent synchronous Responses calls",
    )
    build_rubrics.add_argument(
        "--rubric-state-root",
        "--batch-state-root",
        dest="rubric_state_root",
        help="optional resumable rubric artifact directory; defaults inside the run stage",
    )

    attach_controls = subparsers.add_parser(
        "attach-horizon-controls",
        help="attach count/weight-matched stale controls to an independently built rubric",
    )
    _common(attach_controls, run_required=False)
    attach_controls.add_argument("--current-rubrics", required=True)
    attach_controls.add_argument("--control-rubrics", required=True)
    attach_controls.add_argument("--output", required=True)

    grade_horizon = subparsers.add_parser(
        "grade-horizon",
        help="grade one Pool-A or Pool-B seed/checkpoint shard and emit sealed metrics",
    )
    _common(grade_horizon)
    grade_horizon.add_argument("--prompts", required=True)
    grading_pool = grade_horizon.add_mutually_exclusive_group(required=True)
    grading_pool.add_argument("--pool-b", help="canonical held-out 16-response evaluation pool")
    grading_pool.add_argument("--pool-a", help="auxiliary 8-response rubric-construction pool")
    grading_pool.add_argument(
        "--pool-a-combined",
        help="16-response Pool A analysis pool: current 8 plus fixed pi0 8",
    )
    grade_horizon.add_argument(
        "--rubrics", help="required after checkpoint zero; omitted at zero to enforce R0-only"
    )
    grade_horizon.add_argument("--checkpoint", required=True, type=float)
    grade_horizon.add_argument("--base-url", required=True)
    grade_horizon.add_argument("--output-dir", required=True)
    grade_horizon.add_argument(
        "--reuse-score-dir",
        help="reuse compatible sealed criterion grades and judge only missing criteria",
    )
    grade_horizon.add_argument(
        "--r0-current-only",
        action="store_true",
        help="evaluate only R0 and Rt without requiring a stale-control extension",
    )

    build_observations = subparsers.add_parser(
        "build-horizon-observations",
        help="derive sealed bootstrap observations from sealed score summaries",
    )
    _common(build_observations, run_required=False)
    build_observations.add_argument("--summaries", nargs="+", required=True)
    build_observations.add_argument("--prompts", required=True)
    build_observations.add_argument("--output", required=True)
    return parser


def _context(args: argparse.Namespace, stage: str | None = None) -> PipelineContext:
    config_path = Path(args.config)
    run_id = getattr(args, "run_id", None) or "data-prep"
    return PipelineContext.create(PROJECT_ROOT, config_path, stage or args.command, run_id)


def _relative(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else PROJECT_ROOT / value


def dispatch(args: argparse.Namespace) -> Any:
    command = args.command
    if command == "validate-config":
        config = load_config(_relative(args.config), stage=command)
        result = {
            "valid": True,
            "config_hash": config.config_hash,
            "experiment": config.experiment,
            "reward_source": (
                "online_r0_union_elicited_same_step"
                if config.online_training is not None
                else config.training.reward_source
            ),
        }
        if config.online_training is not None:
            result.update(
                {
                    "online_runtime_claim": config.online_training.runtime_claim,
                    "online_control_policy": config.online_training.control_policy,
                    "online_expected_updates": config.online_training.expected_updates,
                }
            )
        return result
    if command == "prepare-data":
        return run_prepare_data(_context(args), _relative(args.source))
    if command == "preflight":
        return run_preflight(_context(args))
    if command == "generate-static":
        return run_generate_static(_context(args))
    if command == "train-static":
        return run_train_static(_context(args))
    if command == "train-online":
        return run_train_online(_context(args, "train-online"))
    if command == "resume-online":
        return run_train_online(_context(args, "train-online"), resume=True)
    if command == "validate-online-step":
        load_config(_relative(args.config), stage=command)
        return validate_online_step_artifact(
            _relative(args.run_dir),
            args.step,
            require_provider_receipts=args.require_provider_receipts,
        )
    if command == "replay-dynamic":
        return run_replay_dynamic(_context(args, f"replay-dynamic-{args.split}"), args.split)
    if command == "freeze-updater":
        return run_freeze_updater(_context(args))
    if command == "generate-bon":
        return run_generate_bon(_context(args))
    if command == "generate-bon-live":
        return run_live_bon_worker(
            _context(args, "generate-bon-live"),
            policy_step=args.policy_step,
            base_url=args.base_url,
            model=args.model,
            concurrency=args.concurrency,
        )
    if command == "extend-bon-live":
        return extend_live_bon_worker(
            _context(args, "generate-bon-live"),
            policy_step=args.policy_step,
            base_url=args.base_url,
            model=args.model,
            target_pool_size=args.target_pool_size,
            concurrency=args.concurrency,
            chunk_size=args.chunk_size,
            prompt_shard_index=args.prompt_shard_index,
            prompt_shard_count=args.prompt_shard_count,
        )
    if command == "finalize-bon-live":
        return finalize_live_bon(
            _context(args, "generate-bon-live"),
            args.policy_steps,
            args.target_pool_size,
        )
    if command == "prepare-dynamic-batch":
        return prepare_dynamic_batch(
            _context(args, dynamic_batch_stage(args.mode)), args.max_step, args.mode
        )
    if command == "submit-dynamic-batch":
        return submit_dynamic_batch(_context(args, dynamic_batch_stage(args.mode)))
    if command == "status-dynamic-batch":
        return dynamic_batch_status(_context(args, dynamic_batch_stage(args.mode)))
    if command == "collect-dynamic-batch":
        return collect_dynamic_batch(_context(args, dynamic_batch_stage(args.mode)))
    if command == "score-proxy":
        return run_score_proxy(_context(args))
    if command == "select-bon":
        return run_select_bon(_context(args))
    if command == "export-audit-package":
        return run_export_audit_package(_context(args))
    if command == "audit-gold":
        return run_audit_gold(_context(args), _relative(args.private_gt))
    if command == "analyze":
        return run_analyze(_context(args))
    if command == "validate-inventory":
        return run_validate_inventory(_context(args))
    if command == "estimate-cost":
        return estimate_cost_counts(_context(args))
    if command == "prepare-rar-data":
        config = load_config(_relative(args.config), stage=command)
        if config.horizon is None:
            raise ValueError("config has no horizon section")
        return prepare_rar_source(
            _relative(args.source),
            _relative(config.paths.public_data),
            domain=config.horizon.domain,
            seed=config.split_seed,
            train_count=config.horizon.train_count,
            development_count=config.horizon.development_count,
            final_count=config.horizon.final_count,
        )
    if command == "export-rar-verl":
        config = load_config(_relative(args.config), stage=command)
        output = (
            _relative(args.output)
            if args.output
            else _relative(config.paths.public_data).parent / "verl"
        )
        train, development = write_rar_verl_parquets(
            _relative(config.paths.public_data), output, args.run_id
        )
        return {"train": str(train), "development": str(development)}
    if command == "estimate-horizon":
        return estimate_horizon_cost(load_config(_relative(args.config), stage=command))
    if command == "verify-horizon-models":
        return verify_horizon_models(
            load_config(_relative(args.config), stage=command),
            verl_root=PROJECT_ROOT / "environment" / "upstream" / "verl",
        )
    if command == "validate-horizon-launch":
        return validate_horizon_launch_environment(
            load_config(_relative(args.config), stage=command), os.environ
        )
    if command == "generate-horizon-pools":
        config = load_config(_relative(args.config), stage=command)
        if config.horizon is None:
            raise ValueError("config has no horizon section")
        policy = config.models["policy"]
        if not isinstance(policy, dict):
            policy = dict(policy)
        model_name = str(policy["model"])
        mode = str(config.raw.get("execution", {}).get("mode", "live"))
        if mode == "fake":
            provider = FakeGenerator(model_name)
        else:
            if not args.base_url:
                raise ValueError("--base-url is required for live horizon pool generation")
            provider = VLLMPolicyGenerator(
                args.base_url,
                model_name,
                str(policy["revision"]),
                str(policy["tokenizer_revision"]),
                launch_spec_path=PROJECT_ROOT / "environment" / "policy-horizon-launch.json",
                expected_checkpoint_hash=args.checkpoint_hash,
            )
            provider.preflight()
        rows = generate_pool_rows(
            provider,
            read_jsonl(_relative(args.prompts)),
            suite_id=args.run_id,
            domain=config.horizon.domain,
            spec=PoolSpec(args.pool_family, args.count, args.policy_step, args.training_seed),
            checkpoint_hash=args.checkpoint_hash,
            model=model_name,
            model_revision=str(policy["revision"]),
            tokenizer_revision=str(policy["tokenizer_revision"]),
            concurrency=int(os.environ.get("DYNAMIC_RUBRIC_POLICY_CONCURRENCY", "32")),
        )
        return publish_pool_shard(_relative(args.output), rows)
    if command == "combine-horizon-pool-a":
        rows = combine_pool_a_with_fixed_control(
            read_jsonl(_relative(args.pool_a)),
            read_jsonl(_relative(args.fixed_control)),
        )
        return publish_pool_shard(_relative(args.output), rows)
    if command == "analyze-horizon":
        return analyze_horizon_observations(
            load_config(_relative(args.config), stage=command),
            _relative(args.observations),
            _relative(args.output),
            iterations=args.iterations,
        )
    if command == "validate-horizon-inventory":
        config = load_config(_relative(args.config), stage=command)
        if config.horizon is None:
            raise ValueError("config has no horizon section")
        rows = [row for path in args.pools for row in read_jsonl(_relative(path))]
        return validate_horizon_pool_inventory(
            rows,
            expected_counts={
                "fixed_control": config.horizon.fixed_control_count,
                "sham_control": config.horizon.sham_control_count,
                "pool_a": config.horizon.pool_a_count,
                "pool_b": config.horizon.pool_b_count,
            },
            expected_prompt_ids=[
                str(row["prompt_id"]) for row in read_jsonl(_relative(args.prompts))
            ],
            training_seeds=config.horizon.training_seeds,
            checkpoint_steps=config.training.checkpoint_steps,
        )
    if command == "build-horizon-rubrics":
        context = _context(args, command)
        config = context.config
        if config.horizon is None:
            raise ValueError("config has no horizon section")
        extractor = config.models["rubric_extractor"]
        api_mode = str(args.api_mode or extractor.get("api_mode", "batch"))
        if api_mode not in {"batch", "sync"}:
            raise ValueError(f"unsupported rubric extractor api_mode: {api_mode}")
        state_root = (
            _relative(args.rubric_state_root)
            if args.rubric_state_root
            else context.stage_root() / "checkpoints" / args.checkpoint_id / api_mode
        )
        common = {
            "run_id": context.run_id,
            "prompts_path": _relative(args.prompts),
            "current_pool_path": _relative(args.current_pool),
            "control_pool_path": _relative(args.control_pool),
            "checkpoint_id": args.checkpoint_id,
            "pairing_seed": config.split_seed,
            "model": str(extractor["requested_model"]),
            "extraction_schema_path": _relative(str(extractor["schema"])),
            "dedup_schema_path": PROJECT_ROOT / "configs/schemas/horizon_dedup_v1.json",
            "output_path": _relative(args.output),
            "state_root": state_root,
            "max_online_criteria": config.horizon.max_online_criteria,
            "reasoning_effort": str(extractor.get("reasoning_effort", "medium")),
            "control_rubrics_path": (
                _relative(args.control_rubrics) if args.control_rubrics else None
            ),
        }
        if api_mode == "sync":
            concurrency = (
                args.sync_concurrency
                if args.sync_concurrency is not None
                else int(extractor.get("sync_concurrency", 16))
            )
            return build_horizon_rubrics_sync_from_files(
                **common, max_workers=concurrency
            )
        return build_horizon_rubrics_batch_from_files(
            **common, poll_interval_seconds=args.batch_poll_interval_seconds
        )
    if command == "attach-horizon-controls":
        return attach_control_extensions_from_files(
            current_path=_relative(args.current_rubrics),
            control_path=_relative(args.control_rubrics),
            output_path=_relative(args.output),
        )
    if command == "grade-horizon":
        config = load_config(_relative(args.config), stage=command)
        if config.horizon is None:
            raise ValueError("config has no horizon section")
        model = config.models["proxy_grader"]
        grader = VLLMCriterionGrader(
            args.base_url,
            VLLMIdentity(
                served_model=str(model["model"]),
                model_revision=str(model["revision"]),
                tokenizer_revision=str(model["tokenizer_revision"]),
                thinking=bool(model.get("thinking", False)),
            ),
        )
        grader.preflight()
        horizon_raw = config.raw.get("horizon", {})
        if args.checkpoint not in config.horizon.target_epochs:
            raise ValueError("--checkpoint is not one of the preregistered target epochs")
        checkpoint_index = config.horizon.target_epochs.index(args.checkpoint)
        expected_policy_step = config.training.checkpoint_steps[checkpoint_index]
        if args.checkpoint != 0.0 and not args.rubrics:
            raise ValueError("--rubrics is required after checkpoint zero")
        common = {
            "prompts_path": _relative(args.prompts),
            "checkpoint": args.checkpoint,
            "output_dir": _relative(args.output_dir),
            "grader_model_revision": str(model["revision"]),
            "tokenizer_revision": str(model["tokenizer_revision"]),
            "epsilon_spread": float(horizon_raw.get("epsilon_spread", 0.01)),
            "delta_advantage": float(horizon_raw.get("delta_advantage", 1e-8)),
            "expected_policy_step": expected_policy_step,
            "config_hash": config.config_hash,
            "policy_training_regime": config.horizon.policy_training_regime,
            "reuse_score_dir": (
                _relative(args.reuse_score_dir) if args.reuse_score_dir else None
            ),
            "include_control": not args.r0_current_only,
        }
        if args.pool_a or args.pool_a_combined:
            if args.checkpoint == 0.0 or not args.rubrics:
                raise ValueError(
                    "Pool A grading is only valid for nonzero checkpoints with --rubrics"
                )
            if args.pool_a_combined:
                return grade_horizon_pool_a_combined_from_files(
                    grader,
                    pool_a_combined_path=_relative(args.pool_a_combined),
                    rubrics_path=_relative(args.rubrics),
                    **common,
                )
            return grade_horizon_pool_a_from_files(
                grader,
                pool_a_path=_relative(args.pool_a),
                rubrics_path=_relative(args.rubrics),
                **common,
            )
        return grade_horizon_pool_b_from_files(
            grader,
            pool_b_path=_relative(args.pool_b),
            rubrics_path=_relative(args.rubrics) if args.rubrics else None,
            **common,
        )
    if command == "build-horizon-observations":
        config = load_config(_relative(args.config), stage=command)
        if config.horizon is None:
            raise ValueError("config has no horizon section")
        return build_horizon_observations(
            [_relative(path) for path in args.summaries],
            output_path=_relative(args.output),
            expected_seed_ids=[str(seed) for seed in config.horizon.training_seeds],
            expected_prompt_ids=[
                str(row["prompt_id"]) for row in read_jsonl(_relative(args.prompts))
            ],
            expected_checkpoints=config.horizon.target_epochs,
            expected_config_hash=config.config_hash,
        )
    raise AssertionError(f"unhandled command: {command}")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = dispatch(args)
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "failed",
                    "stage": args.command,
                    "error": str(error),
                    "type": type(error).__name__,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(
        json.dumps(
            {"status": "ok", "stage": args.command, "result": result},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
