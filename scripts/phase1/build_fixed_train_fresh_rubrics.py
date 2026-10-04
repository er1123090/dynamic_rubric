#!/usr/bin/env python3
"""Build resumable fresh OnlineRubrics for one fixed-train policy checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path

from dynamic_rubric.phase1.probe_fresh_rubrics import build_probe_fresh_rubrics
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--probe-manifest", required=True, type=Path)
    parser.add_argument("--pool-a", required=True, type=Path)
    parser.add_argument("--pi0-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint-step", required=True, type=int)
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--endpoint", default="http://127.0.0.1:28001")
    parser.add_argument("--model", default="openai/gpt-oss-120b")
    parser.add_argument("--returned-model")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prompt-workers", type=int, default=4)
    parser.add_argument("--extractor-concurrency", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    provider = VLLMChatAdapter(
        args.endpoint,
        args.model,
        args.output_root / "provider_cache" / "gpt_oss_120b",
        timeout_seconds=args.timeout,
        max_retries=6,
        max_in_flight=args.max_in_flight,
    )
    result = build_probe_fresh_rubrics(
        provider,
        run_id=args.run_id,
        train_path=args.train,
        probe_manifest_path=args.probe_manifest,
        pool_a_path=args.pool_a,
        pi0_manifest_path=args.pi0_manifest,
        checkpoint_step=args.checkpoint_step,
        checkpoint_hash=args.checkpoint_hash,
        seed=args.seed,
        extractor_model=args.model,
        extractor_returned_model=args.returned_model or args.model,
        output_root=args.output_root,
        prompt_workers=args.prompt_workers,
        extractor_concurrency=args.extractor_concurrency,
        run_dir=args.run_dir,
    )
    print(result)


if __name__ == "__main__":
    main()
