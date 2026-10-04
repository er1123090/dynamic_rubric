"""Read-only training-artifact audit; only missing stale judge calls are executed.

No optimizer, policy generation, rubric extraction, or held-out data is used.
The output namespace is separate from the canonical training run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
import fcntl
import json
import math
from pathlib import Path
import time
import urllib.request

from dynamic_rubric.artifacts import read_json, read_jsonl, write_json_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter
from .audit_scoring import AuditScoreConfig, score_pool
from .metrics import ScoreGroup, compare_fresh_stale


@dataclass
class AuditTask:
    step: int
    prior_step: int
    prompt_id: str
    occurrence_id: str
    responses: list
    fresh: list
    stale_rubric: list
    fresh_rubric_hash: str
    stale_rubric_hash: str
    clock: dict


def build_tasks(run: Path, through: int) -> tuple[list[AuditTask], dict]:
    """Resolve SAME-prompt prior visits, verifying committed artifact hashes."""
    cfg = read_json(run / "config.resolved.json")
    spec = read_json(run / "launch_spec.json")
    source = Path(cfg["data"]["train_path"])
    if not source.is_absolute():
        source = Path(__file__).resolve().parents[3] / source
    rows = read_jsonl(source)
    manifest_path = Path(spec["train_selection_manifest"])
    if sha256_file(manifest_path) != spec["train_selection_manifest_sha256"]:
        raise ValueError("training selection manifest changed")
    manifest = read_json(manifest_path)
    selection = {int(x["source_index"]): x for x in manifest["ordered_rows"]}
    history = {}
    tasks = []
    exposure = 0
    verified = []
    fresh_all = []
    for step in range(1, through + 1):
        root = run / "verl-run" / "online_steps" / f"step-{step:06d}"
        commit = read_json(root / "commit.json")
        if commit["state"] != "committed" or commit["optimizer_update_index"] != step:
            raise ValueError(f"not a committed optimizer update: {step}")
        for name in (
            "batch.json",
            "rubric_unions.jsonl",
            "current_responses.jsonl",
            "rewards.jsonl",
            "grader_receipts.jsonl",
        ):
            digest = sha256_file(root / name)
            if digest != commit["artifacts"][name]:
                raise ValueError(f"artifact changed: {root / name}")
            verified.append({"step": step, "file": name, "sha256": digest})
        batch = read_json(root / "batch.json")
        if batch["current_policy"]["policy_version"] != step - 1:
            raise ValueError("policy checkpoint/update indexing mismatch")
        response_groups = defaultdict(list)
        for row in read_jsonl(root / "current_responses.jsonl"):
            response_groups[row["prompt_occurrence_id"]].append(row)
        reward_by_id = {x["response_id"]: x for x in read_jsonl(root / "rewards.jsonl")}
        grade_by_id = {
            x["metadata"]["response_id"]: x for x in read_jsonl(root / "grader_receipts.jsonl")
        }
        for union in read_jsonl(root / "rubric_unions.jsonl"):
            occ = union["prompt_occurrence_id"]
            index = int(occ.split(":")[1])
            entry = selection[index]
            pid = entry["prompt_id"]
            prompt = rows[index]
            if prompt["prompt_id"] != pid:
                raise ValueError("source index does not match the frozen train manifest")
            if (
                sha256_json(prompt) != entry["source_row_sha256"]
                or prompt["prompt_hash"] != entry["prompt_hash"]
            ):
                raise ValueError("frozen source prompt content changed")
            messages = prompt.get("messages", prompt.get("prompt"))
            if not isinstance(messages, list):
                raise ValueError(f"source prompt messages missing: {pid}")
            group = sorted(response_groups[occ], key=lambda x: x["rollout_index"])
            if len(group) != 16 or len({x["response_id"] for x in group}) != 16:
                raise ValueError("expected 16 unique training responses")
            fresh = []
            responses = []
            for row in group:
                rid = row["response_id"]
                reward = reward_by_id[rid]
                receipt = grade_by_id[rid]
                if receipt["prompt_id"] != pid or reward["rubric_hash"] != union["content_hash"]:
                    raise ValueError("prompt/rubric grade provenance mismatch")
                metadata = receipt["metadata"]
                if (
                    metadata["optimizer_update_index"] != step
                    or metadata["prompt_occurrence_id"] != occ
                    or metadata["rollout_index"] != row["rollout_index"]
                    or metadata["rubric_hash"] != union["content_hash"]
                    or reward["prompt_occurrence_id"] != occ
                    or reward["rollout_index"] != row["rollout_index"]
                ):
                    raise ValueError("grade metadata does not match the canonical response")
                if (
                    receipt["requested_model"] != cfg["models"]["judge"]["model"]
                    or receipt["returned_model"] != cfg["models"]["judge"]["model"]
                ):
                    raise ValueError("original judge model mismatch")
                if receipt["seed"] != cfg["seed"] + row["rollout_index"]:
                    raise ValueError("original judge seed recipe changed")
                provenance = dict(
                    domain=cfg["domain"],
                    method=cfg["method"],
                    seed=cfg["seed"],
                    global_step=step,
                    policy_step=step - 1,
                    checkpoint_id=str(step - 1),
                    policy_checkpoint=str(step - 1),
                    prompt_id=pid,
                    pool="train_batch",
                )
                fresh.append(
                    {
                        **reward,
                        **provenance,
                        "evaluator_step": step - 1,
                        "evaluator_checkpoint": f"visit-{step}",
                        "fresh_or_stale": "fresh",
                        "judge_receipt": receipt,
                    }
                )
                responses.append({**row, **provenance, "prompt_messages": messages})
            prior = history.get(pid)
            fresh_all.append(
                {
                    "global_step": step,
                    "prompt_id": pid,
                    "eligible_stale": prior is not None,
                    "fresh": fresh,
                }
            )
            if prior is not None:
                old_step, old_union, old_exposure = prior
                tasks.append(
                    AuditTask(
                        step,
                        old_step,
                        pid,
                        occ,
                        responses,
                        fresh,
                        old_union["offline_criteria"] + old_union["online_criteria"],
                        union["content_hash"],
                        old_union["content_hash"],
                        dict(
                            cumulative_prompt_exposures=exposure,
                            cumulative_completions=exposure * 16,
                            cumulative_prompts_since_evaluator=exposure - old_exposure,
                            cumulative_completions_since_evaluator=(exposure - old_exposure) * 16,
                        ),
                    )
                )
            history[pid] = (step, union, exposure)
        exposure += len(batch["prompt_occurrence_ids"])
    return tasks, {
        "through_update": through,
        "prompt_exposures": exposure,
        "completion_count": exposure * 16,
        "eligible_groups": len(tasks),
        "stale_response_count": len(tasks) * 16,
        "source_hashes": verified,
        "source_train_sha256": sha256_file(source),
        "fresh_all": fresh_all,
    }


def score_group(rows, task, evaluator):
    rows = sorted(rows, key=lambda x: x["rollout_index"])
    maps = [dict(row["grades"]) for row in rows]
    ids = set(maps[0])
    if any(set(m) != ids for m in maps):
        raise ValueError("criterion inventory differs within response group")
    return ScoreGroup(
        task.prompt_id,
        evaluator,
        str(task.step - 1),
        tuple(x["response_id"] for x in rows),
        tuple(x["reward"] for x in rows),
        {cid: tuple(m[cid] for m in maps) for cid in sorted(ids)},
    )


def equivalent_metrics(expected, actual):
    """Allow runtime rounding only in derived floats; preserve exact structure."""
    if isinstance(expected, dict):
        return (
            isinstance(actual, dict)
            and expected.keys() == actual.keys()
            and all(equivalent_metrics(value, actual[key]) for key, value in expected.items())
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(expected) == len(actual)
            and all(equivalent_metrics(a, b) for a, b in zip(expected, actual))
        )
    if isinstance(expected, float):
        return isinstance(actual, float) and math.isclose(
            expected, actual, rel_tol=1e-14, abs_tol=1e-15
        )
    return type(expected) is type(actual) and expected == actual


def run_task(task, output, cfg, grader, epsilon_z, epsilon_t):
    path = output / "groups" / f"step-{task.step:06d}" / f"{task.prompt_id}.json"
    seal_path = output / "group_hashes" / f"step-{task.step:06d}" / f"{task.prompt_id}.json"
    identity = sha256_json(
        {
            "task": asdict(task),
            "score_config": asdict(cfg),
            "epsilon_z": epsilon_z,
            "epsilon_t": epsilon_t,
        }
    )
    if path.exists():
        value = read_json(path)
        if value["identity"] != identity:
            raise ValueError(f"completed group identity changed: {path}")
        digest = sha256_file(path)
        if seal_path.exists() and read_json(seal_path)["sha256"] != digest:
            raise ValueError("completed group content hash changed")
        if value["fresh"] != task.fresh:
            raise ValueError("cached fresh grades differ from canonical training")
        expected = compare_fresh_stale(
            score_group(value["stale"], task, f"visit-{task.prior_step}"),
            score_group(task.fresh, task, f"visit-{task.step}"),
            epsilon_z=epsilon_z,
            epsilon_t=epsilon_t,
        )
        expected.update(same_pool_b=False, same_response_pool=True)
        if not equivalent_metrics(expected, value["comparison"]):
            raise ValueError("cached comparison does not match its underlying grades")
        write_json_atomic(seal_path, {"sha256": digest}, immutable=True)
        return value
    start = time.monotonic()
    stale = score_pool(
        task.responses,
        {task.prompt_id: task.stale_rubric},
        evaluator_checkpoint=str(task.prior_step),
        policy_checkpoint=str(task.step - 1),
        pool="train_batch",
        config=cfg,
        grader=grader,
        cache_dir=output / "grade_cache" / f"step-{task.step:06d}" / task.prompt_id,
    )
    for row, response in zip(stale, task.responses):
        if (
            row["judge"]["requested_model"] != cfg.judge_model
            or row["judge"]["returned_model"] != cfg.judge_model
        ):
            raise ValueError("stale judge changed model identity")
        row.update(
            rollout_index=response["rollout_index"],
            fresh_or_stale="stale",
            evaluator_step=task.prior_step - 1,
            evaluator_checkpoint=f"visit-{task.prior_step}",
            rubric_hash=task.stale_rubric_hash,
        )
    comparison = compare_fresh_stale(
        score_group(stale, task, f"visit-{task.prior_step}"),
        score_group(task.fresh, task, f"visit-{task.step}"),
        epsilon_z=epsilon_z,
        epsilon_t=epsilon_t,
    )
    comparison["same_pool_b"] = False
    comparison["same_response_pool"] = True
    value = dict(
        schema_version=1,
        identity=identity,
        domain=cfg.domain,
        method=cfg.method,
        seed=cfg.seed,
        global_step=task.step,
        policy_step=task.step - 1,
        evaluator_step=task.prior_step - 1,
        prompt_id=task.prompt_id,
        pool="train_batch",
        fresh_creation_update=task.step,
        stale_creation_update=task.prior_step,
        evaluator_age_steps=task.step - task.prior_step,
        clock=task.clock,
        comparison=comparison,
        fresh=task.fresh,
        stale=stale,
        elapsed_seconds=time.monotonic() - start,
        interpretation="operational in-sample discriminability only; not fixed-probe or correctness",
    )
    write_json_atomic(path, value, immutable=True)
    write_json_atomic(seal_path, {"sha256": sha256_file(path)}, immutable=True)
    return value


def endpoint_identity(url, model, revision):
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as r:
        data = json.load(r)
    models = [m for m in data["data"] if m["id"] == model]
    if len(models) != 1 or revision not in models[0]["root"]:
        raise ValueError(f"unpinned judge endpoint: {url}")
    with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=10) as r:
        version = json.load(r)
    return {"url": url, "model": models[0], "version": version}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--through", type=int, default=34)
    parser.add_argument(
        "--judge-urls", nargs="+", default=["http://127.0.0.1:28002", "http://127.0.0.1:28004"]
    )
    parser.add_argument("--group-workers", type=int, default=8)
    parser.add_argument("--requests-per-group", type=int, default=8)
    parser.add_argument("--limit-groups", type=int)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "audit.lock").open("a") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        tasks, inventory = build_tasks(args.run.resolve(), args.through)
        fresh_all = inventory.pop("fresh_all")
        write_json_atomic(args.output / "inventory.json", inventory, immutable=True)
        write_json_atomic(args.output / "fresh_training.json", fresh_all, immutable=True)
        print(json.dumps({k: v for k, v in inventory.items() if k != "source_hashes"}), flush=True)
        if args.inventory_only:
            return
        source_cfg = read_json(args.run / "config.resolved.json")
        cfg = AuditScoreConfig(
            domain=source_cfg["domain"],
            seed=source_cfg["seed"],
            judge_revision=source_cfg["models"]["judge"]["revision"],
            concurrency=args.requests_per_group,
        )
        endpoints = [
            endpoint_identity(url, cfg.judge_model, cfg.judge_revision) for url in args.judge_urls
        ]
        if len({str(x["version"]) for x in endpoints}) != 1:
            raise ValueError("judge endpoint vLLM versions differ")
        common = [
            {
                "model": cfg.judge_model,
                "revision": cfg.judge_revision,
                "model_root": endpoint["model"]["root"],
                "vllm_version": endpoint["version"],
            }
            for endpoint in endpoints
        ]
        # Placement may differ (disk versus RAM staging); pinned model/runtime may not.
        if (
            len(
                {
                    sha256_json({k: v for k, v in item.items() if k != "model_root"})
                    for item in common
                }
            )
            != 1
        ):
            raise ValueError("judge replicas do not share a common pinned identity")
        write_json_atomic(args.output / "judge_identity.json", common[0], immutable=True)
        write_json_atomic(
            args.output / "executions" / f"endpoints-{sha256_json(endpoints)}.json",
            endpoints,
            immutable=True,
        )
        contract = dict(
            run=str(args.run.resolve()),
            through=args.through,
            judge_model=cfg.judge_model,
            judge_revision=cfg.judge_revision,
            seed_recipe="11 + rollout_index",
            pool="train_batch",
            fresh_source="canonical saved training grades",
            stale_source="latest SAME-prompt prior visit",
            epsilon_z=source_cfg["analysis"]["epsilon_z"],
            epsilon_t=source_cfg["analysis"]["epsilon_t"],
        )
        write_json_atomic(args.output / "config.json", contract, immutable=True)
        write_json_atomic(args.output / "endpoint_identity.json", endpoints, immutable=False)
        grader = VLLMChatAdapter(
            args.judge_urls,
            cfg.judge_model,
            args.output / "provider_cache",
            timeout_seconds=600,
            max_retries=4,
        )
        selected = tasks[: args.limit_groups] if args.limit_groups else tasks
        start = time.monotonic()
        errors = []
        completed = 0
        with ThreadPoolExecutor(max_workers=args.group_workers) as pool:
            futures = {
                pool.submit(
                    run_task,
                    task,
                    args.output,
                    cfg,
                    grader,
                    contract["epsilon_z"],
                    contract["epsilon_t"],
                ): task
                for task in selected
            }
            for future in as_completed(futures):
                task = futures[future]
                try:
                    future.result()
                    completed += 1
                except Exception as error:
                    errors.append(dict(step=task.step, prompt_id=task.prompt_id, error=repr(error)))
                status = dict(
                    state="running",
                    selected_groups=len(selected),
                    total_groups=len(tasks),
                    completed_groups=completed,
                    errors=errors,
                    elapsed_seconds=time.monotonic() - start,
                    training_resumed=False,
                )
                write_json_atomic(args.output / "status.json", status, immutable=False)
                if completed % 10 == 0 or errors or completed == len(selected):
                    print(json.dumps(status), flush=True)
        status["state"] = (
            "failed" if errors else "smoke_complete" if args.limit_groups else "complete"
        )
        write_json_atomic(args.output / "status.json", status, immutable=False)
        if errors:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
