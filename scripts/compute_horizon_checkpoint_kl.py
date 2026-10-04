#!/usr/bin/env python3
"""CLI bridge for post-training horizon checkpoint KL scoring and analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from dynamic_rubric.horizon.checkpoint_kl import (
    VLLMPolicyLogprobClient,
    analyze_adjacent_checkpoint_kl,
    score_policy_logprobs_from_files,
)


class DualEndpointPolicyLogprobClient(VLLMPolicyLogprobClient):
    """Use the identity proxy for preflight and the local vLLM port for scoring."""

    def __init__(self, identity_base_url: str, score_base_url: str, **kwargs: object) -> None:
        super().__init__(identity_base_url, **kwargs)
        self.score_base_url = score_base_url.rstrip("/")

    def score(
        self,
        token_sequences: Sequence[Sequence[int]],
        response_starts: Sequence[int],
    ) -> list[tuple[float, ...]]:
        identity_base_url = self.base_url
        try:
            self.base_url = self.score_base_url
            return super().score(token_sequences, response_starts)
        finally:
            self.base_url = identity_base_url


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    score = subparsers.add_parser("score")
    score.add_argument("--prompts", type=Path, required=True)
    score.add_argument("--pool-b", type=Path, nargs="+", required=True)
    score.add_argument("--policy-step", type=int, required=True)
    score.add_argument("--checkpoint-hash", required=True)
    score.add_argument("--identity-base-url", required=True)
    score.add_argument("--score-base-url", required=True)
    score.add_argument("--served-model", required=True)
    score.add_argument("--model-revision", required=True)
    score.add_argument("--tokenizer-revision", required=True)
    score.add_argument("--output-dir", type=Path, required=True)
    score.add_argument("--batch-size", type=int, default=32)

    analyze = subparsers.add_parser("analyze")
    analyze.add_argument("--score-dir", type=Path, required=True)
    analyze.add_argument("--checkpoint-steps", type=int, nargs="+", required=True)
    analyze.add_argument("--prompts", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, required=True)
    analyze.add_argument("--responses-per-prompt", type=int, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.command == "score":
        client = DualEndpointPolicyLogprobClient(
            args.identity_base_url,
            args.score_base_url,
            served_model=args.served_model,
            model_revision=args.model_revision,
            tokenizer_revision=args.tokenizer_revision,
            checkpoint_hash=args.checkpoint_hash,
        )
        result = score_policy_logprobs_from_files(
            client,
            prompts_path=args.prompts,
            pool_paths=args.pool_b,
            policy_step=args.policy_step,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    else:
        result = analyze_adjacent_checkpoint_kl(
            score_dir=args.score_dir,
            checkpoint_steps=args.checkpoint_steps,
            prompts_path=args.prompts,
            output_dir=args.output_dir,
            expected_responses_per_prompt=args.responses_per_prompt,
        )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
