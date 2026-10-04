#!/usr/bin/env python3
"""Prepare checkpoint-48 fixed-probe pools and its fresh OnlineRubrics evaluator.

This wrapper intentionally does not launch or stop model servers.  It binds the
work to the completed production checkpoint, reuses the existing BF16 export,
generates Pool A first, then overlaps fresh-rubric construction with independent
Pool B generation.  The underlying builders retain their immutable artifacts
and crash-resumable prompt shards.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from dynamic_rubric.artifacts import read_json, validate_artifact_record, write_json_atomic
from dynamic_rubric.phase1.audit_policy import (
    export_checkpoint,
    generate_probe_pools,
    inspect_checkpoint,
    load_run_contract,
)


FINAL_STEP = 48
EXPECTED_POOL_COUNTS = {"probe_A": 800, "probe_B": 1600}


class FinalProbePreparationError(RuntimeError):
    """Raised when final-probe preparation is not safely bound to checkpoint 48."""


def validate_completed_final_checkpoint(run_dir: Path, output_root: Path) -> Mapping[str, Any]:
    """Require the saved optimizer checkpoint and existing export for update 48."""

    contract = load_run_contract(run_dir)
    latest_path = contract.run_dir / "verl-run" / "latest_commit.json"
    if not latest_path.is_file():
        raise FinalProbePreparationError("production run has no latest_commit.json")
    latest = read_json(latest_path)
    expected_checkpoint = (
        contract.run_dir / "verl-run" / "checkpoints" / f"global_step_{FINAL_STEP}"
    ).resolve()
    try:
        committed_checkpoint = Path(str(latest["checkpoint"])).resolve()
    except KeyError as exc:
        raise FinalProbePreparationError("latest commit does not name its checkpoint") from exc
    if (
        int(latest.get("optimizer_update_index", -1)) != FINAL_STEP
        or latest.get("checkpoint_saved") is not True
        or committed_checkpoint != expected_checkpoint
        or not str(latest.get("resume_checkpoint_hash", ""))
        or not str(latest.get("actor_parameter_hash", ""))
    ):
        raise FinalProbePreparationError(
            "latest commit is not the complete resumable checkpoint-48 state"
        )

    checkpoint = inspect_checkpoint(contract, FINAL_STEP)
    export_dir = output_root.resolve() / "exports" / f"global_step_{FINAL_STEP}"
    if not export_dir.is_dir():
        raise FinalProbePreparationError(f"existing checkpoint-48 export is absent: {export_dir}")

    def refuse_merge(*_args: Any, **_kwargs: Any) -> None:
        raise FinalProbePreparationError("refusing to create a new export in this wrapper")

    validated_export = export_checkpoint(
        contract,
        step=FINAL_STEP,
        export_root=output_root.resolve() / "exports",
        runner=refuse_merge,
    )
    manifest = read_json(validated_export / "audit_export_manifest.json")
    if manifest.get("source_model_sha256") != checkpoint.source_model_sha256:
        raise FinalProbePreparationError("checkpoint-48 export is not bound to the saved actor")
    return {
        "contract": contract,
        "checkpoint": checkpoint,
        "export_dir": validated_export,
        "export_manifest": manifest,
        "latest_commit": latest,
    }


def validate_pool_provenance(
    output_root: Path, *, checkpoint_hash: str, required_pools: Sequence[str]
) -> Mapping[str, Any]:
    path = output_root.resolve() / "responses" / "checkpoint-000048" / "provenance.json"
    if not path.is_file():
        raise FinalProbePreparationError("checkpoint-48 pool provenance is absent")
    provenance = read_json(path)
    for record in provenance.get("artifacts", []):
        validate_artifact_record(record)
    if (
        int(provenance.get("global_step", -1)) != FINAL_STEP
        or provenance.get("checkpoint_hash") != checkpoint_hash
        or not set(required_pools) <= set(provenance.get("selected_pools", ()))
        or any(
            int(provenance.get("pool_counts", {}).get(pool, -1))
            != EXPECTED_POOL_COUNTS[pool]
            for pool in required_pools
        )
    ):
        raise FinalProbePreparationError("checkpoint-48 pool provenance is incomplete or mismatched")
    if set(required_pools) == set(EXPECTED_POOL_COUNTS) and provenance.get(
        "pool_a_b_disjoint"
    ) is not True:
        raise FinalProbePreparationError("checkpoint-48 Pool A/B separation is not verified")
    return provenance


def fresh_rubric_command(
    *,
    contract: Any,
    output_root: Path,
    checkpoint_hash: str,
    endpoint: str,
    prompt_workers: int,
    extractor_concurrency: int,
    max_in_flight: int,
    timeout_seconds: float,
) -> list[str]:
    launch = read_json(contract.launch_spec_path)
    control = launch.get("control_cache", {})
    control_path = Path(str(control.get("path", "")))
    if not control_path.is_file():
        raise FinalProbePreparationError("fixed pi_0 control manifest is absent")
    if control.get("manifest_sha256"):
        from dynamic_rubric.hashing import sha256_file

        if sha256_file(control_path) != control["manifest_sha256"]:
            raise FinalProbePreparationError("fixed pi_0 control manifest changed")
    responses = output_root.resolve() / "responses" / "checkpoint-000048"
    return [
        sys.executable,
        "scripts/phase1/build_fixed_train_fresh_rubrics.py",
        "--run-dir",
        str(contract.run_dir),
        "--run-id",
        contract.run_id,
        "--train",
        str(contract.train_path),
        "--probe-manifest",
        str(contract.probe_manifest_path),
        "--pool-a",
        str(responses / "probe_A.jsonl"),
        "--pi0-manifest",
        str(control_path),
        "--checkpoint-step",
        str(FINAL_STEP),
        "--checkpoint-hash",
        checkpoint_hash,
        "--output-root",
        str(output_root.resolve() / "rubrics"),
        "--endpoint",
        endpoint,
        "--seed",
        str(contract.seed),
        "--prompt-workers",
        str(prompt_workers),
        "--extractor-concurrency",
        str(extractor_concurrency),
        "--max-in-flight",
        str(max_in_flight),
        "--timeout",
        str(timeout_seconds),
    ]


def prepare(
    *,
    run_dir: Path,
    output_root: Path,
    policy_base_url: str,
    extractor_endpoint: str,
    log_dir: Path,
    policy_concurrency: int = 32,
    prompt_workers: int = 8,
    extractor_concurrency: int = 8,
    max_in_flight: int = 32,
    timeout_seconds: float = 900.0,
    pool_generator: Callable[..., Mapping[str, Any]] = generate_probe_pools,
) -> Mapping[str, Any]:
    """Prepare final inputs while preserving each builder's independent restart state."""

    if not policy_base_url or not extractor_endpoint:
        raise ValueError("policy and extractor endpoints must be non-empty")
    if min(policy_concurrency, prompt_workers, extractor_concurrency, max_in_flight) <= 0:
        raise ValueError("all concurrency settings must be positive")
    root = output_root.resolve()
    logs = log_dir.resolve()
    logs.mkdir(parents=True, exist_ok=True)
    bound = validate_completed_final_checkpoint(run_dir, root)
    contract = bound["contract"]
    checkpoint = bound["checkpoint"]
    started = time.time()
    status_path = logs / "status.json"
    write_json_atomic(
        status_path,
        {
            "state": "generating_pool_A",
            "checkpoint": FINAL_STEP,
            "started_at": started,
            "policy_base_url": policy_base_url,
            "extractor_endpoint": extractor_endpoint,
        },
        immutable=False,
    )
    try:
        pool_a = pool_generator(
            contract,
            step=FINAL_STEP,
            output_root=root,
            base_url=policy_base_url,
            concurrency=policy_concurrency,
            timeout_seconds=timeout_seconds,
            pools=("probe_A",),
        )
        validate_pool_provenance(
            root, checkpoint_hash=checkpoint.source_model_sha256, required_pools=("probe_A",)
        )
        command = fresh_rubric_command(
            contract=contract,
            output_root=root,
            checkpoint_hash=checkpoint.source_model_sha256,
            endpoint=extractor_endpoint,
            prompt_workers=prompt_workers,
            extractor_concurrency=extractor_concurrency,
            max_in_flight=max_in_flight,
            timeout_seconds=timeout_seconds,
        )
        write_json_atomic(
            status_path,
            {
                "state": "building_fresh_rubric_and_generating_pool_B",
                "checkpoint": FINAL_STEP,
                "started_at": started,
                "pool_A_result": dict(pool_a),
            },
            immutable=False,
        )

        # The two endpoints are independent.  Starting the rubric subprocess first
        # lets extraction consume Pool A while policy inference produces Pool B.
        rubric_log_path = logs / "fresh-rubrics-48.log"
        with rubric_log_path.open("ab") as rubric_log:
            rubric_process = subprocess.Popen(
                command,
                stdout=rubric_log,
                stderr=subprocess.STDOUT,
            )
            pool_b_error: BaseException | None = None
            pool_b: Mapping[str, Any] | None = None
            try:
                pool_b = pool_generator(
                    contract,
                    step=FINAL_STEP,
                    output_root=root,
                    base_url=policy_base_url,
                    concurrency=policy_concurrency,
                    timeout_seconds=timeout_seconds,
                    pools=("probe_B",),
                )
            except BaseException as error:
                pool_b_error = error
            rubric_returncode = rubric_process.wait()
        if rubric_returncode != 0:
            raise subprocess.CalledProcessError(rubric_returncode, command)
        if pool_b_error is not None:
            raise pool_b_error

        provenance = validate_pool_provenance(
            root,
            checkpoint_hash=checkpoint.source_model_sha256,
            required_pools=("probe_A", "probe_B"),
        )
        rubric_status = read_json(root / "rubrics" / "checkpoint-000048" / "status.json")
        if rubric_status.get("state") != "complete" or int(
            rubric_status.get("prompt_count", -1)
        ) != 100:
            raise FinalProbePreparationError("checkpoint-48 fresh rubric is not complete")
        result = {
            "state": "complete",
            "checkpoint": FINAL_STEP,
            "started_at": started,
            "finished_at": time.time(),
            "pool_A_result": dict(pool_a),
            "pool_B_result": dict(pool_b or {}),
            "pool_counts": dict(provenance["pool_counts"]),
            "pool_a_b_disjoint": provenance["pool_a_b_disjoint"],
            "rubric_prompt_count": rubric_status["prompt_count"],
            "checkpoint_hash": checkpoint.source_model_sha256,
        }
        write_json_atomic(status_path, result, immutable=False)
        return result
    except BaseException as error:
        write_json_atomic(
            status_path,
            {
                "state": "failed",
                "checkpoint": FINAL_STEP,
                "started_at": started,
                "failed_at": time.time(),
                "error": repr(error),
            },
            immutable=False,
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--policy-base-url", required=True)
    parser.add_argument("--extractor-endpoint", default="http://127.0.0.1:28011")
    parser.add_argument("--log-dir", type=Path)
    parser.add_argument("--policy-concurrency", type=int, default=32)
    parser.add_argument("--prompt-workers", type=int, default=8)
    parser.add_argument("--extractor-concurrency", type=int, default=8)
    parser.add_argument("--max-in-flight", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()
    log_dir = args.log_dir or args.output_root.resolve() / "logs" / "final-probe48"
    print(
        prepare(
            run_dir=args.run_dir,
            output_root=args.output_root,
            policy_base_url=args.policy_base_url,
            extractor_endpoint=args.extractor_endpoint,
            log_dir=log_dir,
            policy_concurrency=args.policy_concurrency,
            prompt_workers=args.prompt_workers,
            extractor_concurrency=args.extractor_concurrency,
            max_in_flight=args.max_in_flight,
            timeout_seconds=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
