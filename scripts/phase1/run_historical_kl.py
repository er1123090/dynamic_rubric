"""Replay the selected Phase-1 matrix with pinned HF policies, then evict downloads.

Scores existing fixed-train Pool B only. These are response-only, raw-softmax
sampled log-ratio/K3 proxies on nucleus-sampled text, not exact/unbiased KL.
No model training, response generation, rubric generation, or judge is needed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import statistics
import subprocess
import time
import urllib.request

from dynamic_rubric.artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
    write_jsonl_atomic,
)
from dynamic_rubric.hashing import sha256_json
from dynamic_rubric.horizon.checkpoint_kl import (
    VLLMPolicyLogprobClient,
    _decode_float32,
    score_policy_logprobs_from_files,
)
from dynamic_rubric.phase1.audit_policy import load_run_contract
from scripts.phase1.run_regular_probe_pool_a import ready, stop_owned
from scripts.phase1.prefetch_historical_kl import download_model, model_lock

ESTIMATOR = "response-only retokenized raw-softmax log-ratio/K3 on nucleus-sampled current-policy Pool B; not exact or unbiased distribution KL"


def serving_options(args):
    if not 0 < args.gpu_memory_utilization <= 1:
        raise ValueError("Invalid KL GPU memory fraction")
    if args.max_num_seqs < 1 or args.max_num_batched_tokens < 1:
        raise ValueError("KL serving limits must be positive")
    if args.share_gpu and args.gpu_memory_utilization > 0.08:
        raise ValueError("Shared Trainer GPU1 KL is capped at 8% of device memory")
    kv_bytes = getattr(args, "kv_cache_memory_bytes", None)
    if args.share_gpu and (kv_bytes is None or not 0 < kv_bytes <= 1610612736):
        raise ValueError("Shared KL requires an explicit KV cache cap of at most 1.5GiB")
    if args.share_gpu and (args.max_num_seqs > 4 or args.max_num_batched_tokens > 1024):
        raise ValueError("Shared KL serving is limited to 4 sequences and 1024 tokens")
    options = [
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--max-num-batched-tokens",
        str(args.max_num_batched_tokens),
    ]
    if kv_bytes is not None:
        if kv_bytes <= 0:
            raise ValueError("KV cache bytes must be positive")
        options.extend(["--kv-cache-memory-bytes", str(kv_bytes)])
    return options


def validate_context_budget(model_path, max_model_len, kv_bytes):
    """Check BF16 Qwen3 KV capacity without changing or truncating score inputs."""
    if max_model_len < 2:
        raise ValueError("KL context length must include input and one output token")
    config = read_json(model_path / "config.json")
    if max_model_len > config["max_position_embeddings"]:
        raise ValueError("KL context exceeds model position capacity")
    if kv_bytes is not None:
        bytes_per_token = (
            2 * 2 * config["num_hidden_layers"] * config["num_key_value_heads"]
            * config["head_dim"]
        )
        # Leave a full block of slack after rounding up the requested context.
        required = (math.ceil(max_model_len / 16) + 1) * 16 * bytes_per_token
        if required > kv_bytes:
            raise ValueError(f"KL context requires at least {required} KV bytes")


def wait_for_memory(args, root, step):
    """Defer KL startup if training has not left enough memory; never stop training."""
    deadline = time.monotonic() + args.memory_wait_seconds
    while True:
        raw = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                "1",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        free, total = (int(v.strip()) for v in raw.split(","))
        required = math.ceil(total * args.gpu_memory_utilization) + 2048
        if free >= required:
            return
        write_json_atomic(
            root / "memory-wait.json",
            {
                "state": "waiting_for_shared_gpu_headroom",
                "step": step,
                "free_mib": free,
                "required_mib": required,
                "updated_at": time.time(),
            },
            immutable=False,
        )
        if time.monotonic() >= deadline:
            raise RuntimeError(f"KL GPU memory wait expired: {free} < {required} MiB")
        time.sleep(10)


def matrix_cells(config, available):
    if (
        config["primary_dataset"] != "fixed_train_probe_100"
        or config["prompt_count"] != 100
        or config["responses_per_prompt"] != 16
    ):
        raise ValueError("KL requires the fixed train100 x16 design")
    cells = [(int(c["evaluator_step"]), int(c["policy_step"])) for c in config["cells"]]
    if len(set(cells)) != len(cells) or any(a > b for a, b in cells):
        raise ValueError("Duplicate or forward evaluator matrix cells")
    active = [c for c in cells if c[0] in available and c[1] in available]
    if any((t, t) not in active for _, t in active):
        raise ValueError("Current-policy diagonal is required for every column")
    return active, [c for c in cells if c not in active]


def validate_scores(path, pool, scoring_step, checkpoint_hash):
    rows = read_jsonl(path)
    index = {r["response_id"]: r for r in pool}
    if (
        len(rows) != len(pool)
        or len({r["response_id"] for r in rows}) != len(rows)
        or {r["response_id"] for r in rows} != set(index)
    ):
        raise ValueError("Score response inventory mismatch")
    for row in rows:
        source = index[row["response_id"]]
        expected = {
            "policy_step": scoring_step,
            "pool_policy_step": source["policy_step"],
            "scoring_checkpoint_hash": checkpoint_hash,
            "source_checkpoint_hash": source["checkpoint_hash"],
            "prompt_id": source["prompt_id"],
            "sample_index": source["sample_index"],
        }
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Score identity mismatch")
        count = row["response_token_count"]
        if count <= 0 or not row["response_token_hash"]:
            raise ValueError("Invalid token inventory")
        _decode_float32(row["response_token_logprobs_f32le_b64"], count)
    return rows


class CachedClient(VLLMPolicyLogprobClient):
    def __init__(self, *args, model_path, cache, progress, max_model_len=10240, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_path, self.cache, self.progress = model_path, cache, progress
        self.scored = 0
        self.max_model_len = max_model_len

    def preflight(self):
        with urllib.request.urlopen(self.base_url + "/v1/models", timeout=10) as response:
            value = json.loads(response.read())
        if [r["id"] for r in value["data"]] != [self.served_model]:
            raise RuntimeError("Loaded server model identity mismatch")
        return {
            "model_path": str(self.model_path),
            "served_model": self.served_model,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "checkpoint_hash": self.checkpoint_hash,
            "thinking": False,
        }

    def score(self, token_sequences, response_starts):
        required = max((len(tokens) + 1 for tokens in token_sequences), default=1)
        if required > self.max_model_len:
            raise ValueError(
                f"KL requires context {required}, configured {self.max_model_len}; "
                "increase --max-model-len and verify KV capacity; inputs are never truncated"
            )
        key = sha256_json(
            {
                "model": self.checkpoint_hash,
                "tokens": token_sequences,
                "starts": response_starts,
                "model_revision": self.model_revision,
                "tokenizer_revision": self.tokenizer_revision,
            }
        )
        path = self.cache / f"{key}.json"
        if path.exists():
            cached = read_json(path)
            if cached["request_hash"] != key:
                raise RuntimeError("Batch cache identity mismatch")
            values = cached["logprobs"]
        else:
            for attempt in range(3):
                try:
                    values = super().score(token_sequences, response_starts)
                    break
                except (OSError, TimeoutError):
                    if attempt == 2:
                        raise
                    time.sleep(2**attempt)
            write_json_atomic(path, {"request_hash": key, "logprobs": values})
        if len(values) != len(token_sequences) or any(
            len(v) != len(t) - s or not all(math.isfinite(x) for x in v)
            for v, t, s in zip(values, token_sequences, response_starts)
        ):
            raise RuntimeError("Invalid batch log-probs")
        self.scored += len(values)
        write_json_atomic(
            self.progress,
            {
                "state": "scoring",
                "responses_in_loaded_model": self.scored,
                "model": self.served_model,
                "updated_at": time.time(),
            },
            immutable=False,
        )
        print(
            json.dumps({"event": "batch", "model": self.served_model, "responses": self.scored}),
            flush=True,
        )
        return values


def summarize(root, a, t):
    target = root / "pairs" / f"stale-{a:06d}_current-{t:06d}"
    seal = target / "seal.json"
    if seal.exists():
        for rec in read_json(seal)["artifacts"]:
            validate_artifact_record(rec)
        return
    old_path = root / "scores" / f"policy-step-{a}_pool-step-{t}.jsonl"
    new_path = root / "scores" / f"policy-step-{t}_pool-step-{t}.jsonl"
    old = {r["response_id"]: r for r in read_jsonl(old_path)}
    new = read_jsonl(new_path)
    grouped = defaultdict(list)
    responses = []
    token_k1 = token_k3 = 0.0
    total = clipped_count = 0
    for current in new:
        stale = old[current["response_id"]]
        for field in ("prompt_id", "sample_index", "response_token_count", "response_token_hash"):
            if current[field] != stale[field]:
                raise ValueError("Fresh/stale token identity mismatch")
        count = current["response_token_count"]
        c = _decode_float32(current["response_token_logprobs_f32le_b64"], count)
        s = _decode_float32(stale["response_token_logprobs_f32le_b64"], count)
        diff = [x - y for x, y in zip(c, s)]
        ratios = [-d for d in diff]
        clipped = [max(-20.0, min(20.0, r)) for r in ratios]
        k1 = sum(diff)
        k3 = sum(math.expm1(r) - r for r in clipped)
        nclip = sum(r != q for r, q in zip(ratios, clipped))
        row = {
            "domain": "medicine",
            "method": "online_rubrics",
            "seed": 11,
            "global_step": t,
            "checkpoint_id": f"global_step_{t}",
            "prompt_id": current["prompt_id"],
            "response_id": current["response_id"],
            "pool": "probe_B",
            "policy_checkpoint": t,
            "stale_policy_checkpoint": a,
            "evaluator_checkpoint": a,
            "fresh_or_stale": "paired_policy_logprobs",
            "sample_index": current["sample_index"],
            "k1_mean": k1 / count,
            "k3_mean": k3 / count,
            "token_count": count,
            "clipped_tokens": nclip,
            "response_token_hash": current["response_token_hash"],
        }
        responses.append(row)
        grouped[row["prompt_id"]].append(row)
        token_k1 += k1
        token_k3 += k3
        total += count
        clipped_count += nclip
    if (
        len(new) != 1600
        or len(old) != 1600
        or len(grouped) != 100
        or any(len(v) != 16 for v in grouped.values())
    ):
        raise ValueError("Pair is not the full fixed100 x16 pool")
    prompts = [
        {
            "prompt_id": pid,
            "current_policy_checkpoint": t,
            "stale_policy_checkpoint": a,
            "k1_mean": statistics.fmean(r["k1_mean"] for r in rows),
            "k3_mean": statistics.fmean(r["k3_mean"] for r in rows),
        }
        for pid, rows in sorted(grouped.items())
    ]
    summary = {
        "stale_policy_checkpoint": a,
        "current_policy_checkpoint": t,
        "direction": "pi_current || pi_stale on current Pool B",
        "estimator": ESTIMATOR,
        "prompt_count": 100,
        "response_count": 1600,
        "tokens": total,
        "clipped_tokens": clipped_count,
        "k3_log_ratio_clip": 20.0,
        "k1_prompt_mean": statistics.fmean(r["k1_mean"] for r in prompts),
        "k3_prompt_mean": statistics.fmean(r["k3_mean"] for r in prompts),
        "k1_prompt_se": statistics.stdev(r["k1_mean"] for r in prompts) / 10,
        "k3_prompt_se": statistics.stdev(r["k3_mean"] for r in prompts) / 10,
        "k1_token_mean": token_k1 / total,
        "k3_token_mean": token_k3 / total,
        "source_scores": [artifact_record(old_path), artifact_record(new_path)],
    }
    write_jsonl_atomic(target / "responses.jsonl", responses)
    write_jsonl_atomic(target / "prompts.jsonl", prompts)
    write_json_atomic(target / "summary.json", summary)
    write_json_atomic(
        seal,
        {
            "artifacts": [
                artifact_record(target / n)
                for n in ["responses.jsonl", "prompts.jsonl", "summary.json"]
            ]
        },
    )


def delete_download(target, download_root, receipt, root, cells):
    with model_lock(root, receipt["checkpoint_step"]):
        return _delete_download_locked(target, download_root, receipt, root, cells)


def _delete_download_locked(target, download_root, receipt, root, cells):
    step = receipt["checkpoint_step"]
    if (
        target.is_symlink()
        or download_root.is_symlink()
        or target.parent.resolve() != download_root.resolve()
        or target.name != f"global_step_{step}"
    ):
        raise ValueError("Unsafe download deletion target")
    if any(p.is_symlink() for p in target.rglob("*")):
        raise ValueError("Download tree contains symlinks")
    for a, t in cells:
        if a == step:
            seal = read_json(root / "seals" / f"model-{a}_pool-{t}.json")
            validate_artifact_record(seal["scores"])
    ownership = read_json(target / "kl-download-owner.json")
    if ownership != {
        "repo_id": receipt["repo_id"],
        "revision": receipt["revision"],
        "checkpoint_step": step,
        "kl_root": str(root),
    }:
        raise ValueError("Download ownership mismatch")
    size = sum(p.stat().st_size for p in target.rglob("*") if p.is_file())
    write_json_atomic(
        root / "downloads" / f"step-{step}-delete-intent.json",
        {
            "target": str(target),
            "bytes": size,
            "repo_id": receipt["repo_id"],
            "revision": receipt["revision"],
        },
    )
    shutil.rmtree(target)
    write_json_atomic(
        root / "downloads" / f"step-{step}-deleted.json",
        {
            "target": str(target),
            "bytes": size,
            "deleted_at": time.time(),
            "recoverable_from": receipt["repo_id"],
            "revision": receipt["revision"],
        },
    )


def run(args):
    options = serving_options(args)
    root = args.output_root.resolve()
    audit = args.audit_root.resolve()
    run_dir = args.run_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / "run.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        contract = load_run_contract(run_dir)
        config = read_json(args.matrix)
        prompts_path = audit / "manifests/fixed_train_probe_prompts.jsonl"
        pids = {p["prompt_id"] for p in read_jsonl(prompts_path)}
        assert len(pids) == 100
        assert pids == set(read_json(contract.probe_manifest_path)["prompt_ids"])
        pools = {}
        bindings = {}
        for step in config["steps"]:
            path = audit / f"responses/checkpoint-{step:06d}/probe_B.jsonl"
            if not path.exists():
                continue
            rows = read_jsonl(path)
            assert len(rows) == len({r["response_id"] for r in rows}) == 1600
            assert {r["prompt_id"] for r in rows} == pids
            assert {r["policy_step"] for r in rows} == {step} and {r["pool"] for r in rows} == {
                "probe_B"
            }
            assert {r["run_id"] for r in rows} == {contract.run_id}
            assert {r["model_revision"] for r in rows} == {contract.model_revision}
            assert {r["tokenizer_revision"] for r in rows} == {contract.tokenizer_revision}
            assert len({r["checkpoint_hash"] for r in rows}) == 1
            slots = defaultdict(set)
            for row in rows:
                slots[row["prompt_id"]].add(row["sample_index"])
            assert all(s == set(range(16)) for s in slots.values())
            pool_a = path.with_name("probe_A.jsonl")
            if pool_a.exists():
                assert not {r["response_id"] for r in rows} & {
                    r["response_id"] for r in read_jsonl(pool_a)
                }
            pools[step] = rows
            bindings[step] = artifact_record(path)
        cells, missing = matrix_cells(config, set(pools))
        plan = {
            "matrix": artifact_record(args.matrix),
            "prompts": artifact_record(prompts_path),
            "pool_files": bindings,
            "cells": cells,
            "pending_cells": missing,
            "estimator": ESTIMATOR,
            "batch_size": args.batch_size,
            "run_id": contract.run_id,
            "model_revision": contract.model_revision,
            "tokenizer_revision": contract.tokenizer_revision,
        }
        write_json_atomic(root / "plan.json", plan)
        if args.plan_only:
            print(
                {
                    "runnable_cells": len(cells),
                    "pending_cells": len(missing),
                    "policy_models": len(pools),
                }
            )
            return
        pause = read_json(run_dir / "logs/pause46-for-kl-20260909/sealed.json")
        assert pause["resume_at_step"] == 47
        gpu_used = int(
            subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    "1",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
            ).strip()
        )
        if gpu_used > 512 and not args.share_gpu:
            raise RuntimeError(f"TrainerGPU1 not free: {gpu_used}MiB")
        write_json_atomic(
            root / f"runtime-{time.time_ns()}.json",
            {
                "share_gpu": args.share_gpu,
                "serving_options": options,
                "max_model_len": args.max_model_len,
                "gpu": 1,
                "scientific_plan": artifact_record(root / "plan.json"),
            },
        )
        download_root = root / "temporary_models"
        download_root.mkdir(exist_ok=True)
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": "1",
            "PYTHONUNBUFFERED": "1",
            "HF_HUB_DISABLE_XET": "1",
        }
        status_path = root / "status.json"
        completed = []
        try:
            for step in sorted({a for a, t in cells}, reverse=True):
                assigned = sorted(t for a, t in cells if a == step)
                write_json_atomic(
                    status_path,
                    {
                        "state": "preparing_model",
                        "step": step,
                        "completed_models": completed,
                        "pending_cells": missing,
                        "updated_at": time.time(),
                    },
                    immutable=False,
                )
                receipt_path = run_dir / f"verl-run/checkpoint_archives/global_step_{step}.json"
                receipt = read_json(receipt_path) if receipt_path.exists() else None
                if receipt:
                    assert (
                        receipt["run_id"] == contract.run_id
                        and receipt["checkpoint_step"] == step
                        and receipt["state"] == "verified"
                    )
                    checkpoint_hash = receipt["audit_export_manifest"]["source_model_sha256"]
                else:
                    manifest = read_json(
                        audit / f"exports/global_step_{step}/audit_export_manifest.json"
                    )
                    checkpoint_hash = manifest["source_model_sha256"]
                assert {r["checkpoint_hash"] for r in pools[step]} == {checkpoint_hash}
                done = []
                for t in assigned:
                    sealpath = root / "seals" / f"model-{step}_pool-{t}.json"
                    if sealpath.exists():
                        seal = read_json(sealpath)
                        assert (
                            seal["pool"] == bindings[t]
                            and seal["checkpoint_hash"] == checkpoint_hash
                        )
                        validate_artifact_record(seal["scores"])
                        done.append(t)
                target = download_root / f"global_step_{step}"
                if len(done) == len(assigned):
                    for t in assigned:
                        summarize(root, step, t)
                    if receipt and target.exists():
                        delete_download(target, download_root, receipt, root, cells)
                    completed.append(step)
                    continue
                if receipt:
                    model_path = download_model(run_dir, root, receipt, hf_cli=args.hf_cli, env=env)
                else:
                    model_path = audit / f"exports/global_step_{step}"
                    assert (
                        manifest["run_id"] == contract.run_id
                        and manifest["checkpoint_step"] == step
                    )
                    for rec in manifest["artifacts"]:
                        validate_artifact_record(rec)
                server = None
                try:
                    validate_context_budget(
                        model_path, args.max_model_len, args.kv_cache_memory_bytes
                    )
                    if args.share_gpu:
                        wait_for_memory(args, root, step)
                    with (root / f"vllm-step-{step}.log").open("ab") as out:
                        server = subprocess.Popen(
                            [
                                args.policy_python,
                                "-m",
                                "vllm.entrypoints.openai.api_server",
                                "--model",
                                str(model_path),
                                "--served-model-name",
                                f"phase1-kl-step-{step}",
                                "--host",
                                "127.0.0.1",
                                "--port",
                                "28020",
                                "--dtype",
                                "bfloat16",
                                "--max-model-len",
                                str(args.max_model_len),
                                "--generation-config",
                                "vllm",
                                *options,
                                "--no-enable-prefix-caching",
                                "--enforce-eager",
                            ],
                            env=env,
                            stdout=out,
                            stderr=subprocess.STDOUT,
                            start_new_session=True,
                        )
                        ready(server, "http://127.0.0.1:28020/v1/models", timeout=600)
                        client = CachedClient(
                            "http://127.0.0.1:28020",
                            served_model=f"phase1-kl-step-{step}",
                            model_revision=contract.model_revision,
                            tokenizer_revision=contract.tokenizer_revision,
                            checkpoint_hash=checkpoint_hash,
                            model_path=model_path,
                            cache=root / "batch_cache" / f"model-{step}",
                            progress=root / "batch-progress.json",
                            max_model_len=args.max_model_len,
                        )
                        for t in assigned:
                            if t in done:
                                summarize(root, step, t)
                                continue
                            write_json_atomic(
                                status_path,
                                {
                                    "state": "scoring",
                                    "scoring_step": step,
                                    "pool_step": t,
                                    "completed_models": completed,
                                    "pending_cells": missing,
                                    "updated_at": time.time(),
                                },
                                immutable=False,
                            )
                            pool_path = audit / f"responses/checkpoint-{t:06d}/probe_B.jsonl"
                            score_policy_logprobs_from_files(
                                client,
                                prompts_path=prompts_path,
                                pool_paths=[pool_path],
                                policy_step=step,
                                output_dir=root / "scores",
                                batch_size=args.batch_size,
                            )
                            score_path = root / "scores" / f"policy-step-{step}_pool-step-{t}.jsonl"
                            validate_scores(score_path, pools[t], step, checkpoint_hash)
                            write_json_atomic(
                                root / "seals" / f"model-{step}_pool-{t}.json",
                                {
                                    "scores": artifact_record(score_path),
                                    "pool": bindings[t],
                                    "checkpoint_hash": checkpoint_hash,
                                    "estimator": ESTIMATOR,
                                },
                            )
                            summarize(root, step, t)
                finally:
                    stop_owned(server)
                if receipt:
                    delete_download(target, download_root, receipt, root, cells)
                completed.append(step)
            summaries = [
                read_json(root / "pairs" / f"stale-{a:06d}_current-{t:06d}" / "summary.json")
                for a, t in cells
            ]
            write_json_atomic(
                root / "summary.json",
                {"estimator": ESTIMATOR, "cells": summaries, "pending_cells": missing},
            )
            write_json_atomic(
                status_path,
                {
                    "state": "complete_available_cells",
                    "completed_cells": len(cells),
                    "pending_cells": missing,
                    "completed_models": completed,
                    "finished_at": time.time(),
                    "training_resumed": False,
                },
                immutable=False,
            )
        except BaseException as error:
            write_json_atomic(
                status_path,
                {
                    "state": "failed",
                    "error": repr(error),
                    "completed_models": completed,
                    "updated_at": time.time(),
                },
                immutable=False,
            )
            raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-root", type=Path, required=True)
    parser.add_argument("--matrix", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--policy-python", default=str(Path(__file__).resolve().parents[2] / ".venvs/judge/bin/python")
    )
    parser.add_argument("--hf-cli", default="hf")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--max-num-batched-tokens", type=int, default=16384)
    parser.add_argument("--kv-cache-memory-bytes", type=int)
    parser.add_argument("--max-model-len", type=int, default=10240)
    parser.add_argument("--share-gpu", action="store_true")
    parser.add_argument("--memory-wait-seconds", type=float, default=7200)
    parser.add_argument("--plan-only", action="store_true")
    run(parser.parse_args())
