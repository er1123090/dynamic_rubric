"""CLI for Phase-1 evaluator-update experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_phase1_config
from .full_run import FULL_RUN_PREFIX, run_online_full
from .one_step import CANARY_RUN_PREFIX, run_online_one_step
from .preflight import topology_preflight
from .provenance import prepare_fixed_train_probe_manifest
from .smoke import run_deterministic_smoke


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dynamic_rubric.phase1")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-probe")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--repo-root", default=".")

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--config", required=True)
    preflight.add_argument("--repo-root", default=".")
    preflight.add_argument("--require-endpoints", action="store_true")

    smoke = subparsers.add_parser("smoke")
    smoke.add_argument("--config", required=True)
    smoke.add_argument("--repo-root", default=".")
    smoke.add_argument("--run-id", default="contract-smoke")

    one_step = subparsers.add_parser("train-online-one-step")
    one_step.add_argument("--config", required=True)
    one_step.add_argument("--repo-root", default=".")
    one_step.add_argument("--run-id", default=CANARY_RUN_PREFIX)

    full = subparsers.add_parser("train-online")
    full.add_argument("--config", required=True)
    full.add_argument("--repo-root", default=".")
    full.add_argument("--run-id", default=FULL_RUN_PREFIX)

    resume = subparsers.add_parser("resume-online")
    resume.add_argument("--config", required=True)
    resume.add_argument("--repo-root", default=".")
    resume.add_argument("--run-id", required=True)

    return parser


def main() -> None:
    args = _parser().parse_args()
    config = load_phase1_config(args.config)
    if args.command == "prepare-probe":
        path, manifest = prepare_fixed_train_probe_manifest(config, repo_root=Path(args.repo_root))
        result = {
            "status": "passed",
            "manifest": str(path),
            "prompt_count": manifest["probe_prompt_count"],
            "prompt_ids_sha256": manifest["prompt_ids_sha256"],
        }
    elif args.command == "preflight":
        result = topology_preflight(
            config,
            repo_root=Path(args.repo_root),
            require_endpoints=args.require_endpoints,
        )
    elif args.command == "smoke":
        result = run_deterministic_smoke(
            config,
            repo_root=Path(args.repo_root),
            run_id=args.run_id,
        )
    elif args.command == "train-online-one-step":
        result = run_online_one_step(config, repo_root=Path(args.repo_root), run_id=args.run_id)
    else:
        result = run_online_full(
            config,
            repo_root=Path(args.repo_root),
            run_id=args.run_id,
            resume=args.command == "resume-online",
        )
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
