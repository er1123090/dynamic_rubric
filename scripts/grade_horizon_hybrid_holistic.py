#!/usr/bin/env python3
"""Grade one horizon shard with complete-response cache reuse."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dynamic_rubric.artifacts import read_jsonl
from dynamic_rubric.config import load_config
from dynamic_rubric.horizon.live_grading import grade_horizon_pool
from dynamic_rubric.providers.vllm_holistic import (
    HYBRID_TARGET_ENCODING_VERSION,
    HybridVLLMFullRubricGrader,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    pool = parser.add_mutually_exclusive_group(required=True)
    pool.add_argument("--pool-b", type=Path)
    pool.add_argument("--pool-a-combined", type=Path)
    parser.add_argument("--rubrics", type=Path)
    parser.add_argument("--include-control", action="store_true")
    parser.add_argument("--checkpoint", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", action="append", required=True)
    parser.add_argument("--criterion-cache-dir", type=Path, required=True)
    parser.add_argument("--holistic-cache-dir", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--max-attempts", type=int, default=3)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    config = load_config(args.config, stage="grade-horizon")
    if config.horizon is None:
        raise ValueError("config has no horizon section")
    if args.checkpoint not in config.horizon.target_epochs:
        raise ValueError("--checkpoint is not one of the configured target epochs")
    checkpoint_index = config.horizon.target_epochs.index(args.checkpoint)
    expected_policy_step = config.training.checkpoint_steps[checkpoint_index]
    if (args.checkpoint != 0.0 or args.pool_a_combined is not None) and args.rubrics is None:
        raise ValueError("--rubrics is required after checkpoint zero")

    model = config.models["proxy_grader"]
    grader = HybridVLLMFullRubricGrader(
        base_urls=args.base_url,
        served_model=str(model["model"]),
        model_revision=str(model["revision"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        criterion_cache_dir=args.criterion_cache_dir,
        holistic_cache_dir=args.holistic_cache_dir,
        max_workers=args.concurrency,
        max_attempts=args.max_attempts,
    )
    preflight = grader.preflight()

    prompts = read_jsonl(args.prompts)
    pool_path = args.pool_b or args.pool_a_combined
    assert pool_path is not None
    pool_family = "pool_b" if args.pool_b is not None else "pool_a_combined"
    pool_rows = read_jsonl(pool_path)
    if args.rubrics is None:
        rubric_rows = [
            {"prompt_id": row["prompt_id"], "extension": []} for row in prompts
        ]
    else:
        rubric_rows = read_jsonl(args.rubrics)
    sources = {"prompts": args.prompts, pool_family: pool_path}
    if args.rubrics is not None:
        sources["rubrics"] = args.rubrics

    horizon_raw = config.raw.get("horizon", {})
    result = grade_horizon_pool(
        grader,
        prompts=prompts,
        pool_rows=pool_rows,
        pool_family=pool_family,
        rubric_rows=rubric_rows,
        checkpoint=args.checkpoint,
        output_dir=args.output_dir,
        grader_model_revision=str(model["revision"]),
        tokenizer_revision=str(model["tokenizer_revision"]),
        epsilon_spread=float(horizon_raw.get("epsilon_spread", 0.01)),
        delta_advantage=float(horizon_raw.get("delta_advantage", 1e-8)),
        source_paths=sources,
        include_control=args.include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config.config_hash,
        target_encoding_version=HYBRID_TARGET_ENCODING_VERSION,
    )
    print(
        json.dumps(
            {"status": "ok", "preflight": preflight, "result": result},
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
