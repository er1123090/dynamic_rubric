"""Validated fixed-train-probe scoring for explicit fresh/stale rubric cells."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import time
import urllib.request
from typing import Any, Mapping, Sequence

from ..artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
)
from ..hashing import sha256_file, sha256_json
from ..providers.vllm_chat import VLLMChatAdapter
from .audit_policy import (
    _validate_pool_rows,
    load_probe_prompts,
    load_run_contract,
)
from .audit_checkpoint_identity import inspect_scoring_checkpoint as inspect_checkpoint
from .audit_scoring import AuditScoreConfig, score_pool, write_score_receipts
from .provenance import validate_pool_ab_disjoint


class ProbeAdjacentScoringError(RuntimeError):
    """Raised when fixed-probe scoring inputs are incomplete or incompatible."""


REGULAR_STEPS = tuple(range(0, 46, 3))
SCHEDULER_POLICY = "finish_earliest_ready_policy_pair_then_any_ready_cell"


def _r0_criteria(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    try:
        criteria = [
            {
                "criterion_id": str(item["criterion_id"]),
                "text": str(item["criterion"]),
                "weight": int(item["weight_units"]),
                "source": "r0",
            }
            for item in row["r0"]["criteria"]
        ]
    except (KeyError, TypeError, ValueError) as error:
        raise ProbeAdjacentScoringError("training prompt has malformed R0 criteria") from error
    if not criteria:
        raise ProbeAdjacentScoringError("training prompt has an empty R0 rubric")
    return criteria


def _probe_source_rows(contract: Any) -> dict[str, Mapping[str, Any]]:
    prompt_ids = {str(row["prompt_id"]) for row in load_probe_prompts(contract)}
    rows = read_jsonl(contract.train_path)
    by_id = {str(row.get("prompt_id", "")): row for row in rows}
    if len(by_id) != len(rows) or not prompt_ids <= set(by_id):
        raise ProbeAdjacentScoringError("fixed probe is not an exact train-source subset")
    return {prompt_id: by_id[prompt_id] for prompt_id in prompt_ids}


def load_evaluator_rubrics(
    contract: Any, artifact_root: Path, evaluator_step: int
) -> tuple[dict[str, list[dict[str, Any]]], Mapping[str, Any]]:
    """Load E_tau, treating E_0 as the original prompt-specific R0."""

    source_rows = _probe_source_rows(contract)
    if evaluator_step == 0:
        rubrics = {prompt_id: _r0_criteria(row) for prompt_id, row in source_rows.items()}
        return rubrics, {
            "kind": "initial_prompt_specific_r0",
            "train_path": str(contract.train_path),
            "train_sha256": contract.train_sha256,
            "evaluator_step": 0,
        }

    rubric_dir = artifact_root / "rubrics" / f"checkpoint-{evaluator_step:06d}"
    status_path = rubric_dir / "status.json"
    invocation_path = rubric_dir / "invocation.json"
    rubric_path = rubric_dir / "fresh_rubrics.jsonl"
    if not status_path.is_file() or not invocation_path.is_file() or not rubric_path.is_file():
        raise ProbeAdjacentScoringError(f"fresh rubric E_{evaluator_step} is not ready")
    status = read_json(status_path)
    invocation = read_json(invocation_path)
    checkpoint = inspect_checkpoint(contract, evaluator_step)
    expected_invocation = {
        "run_id": contract.run_id,
        "checkpoint_step": evaluator_step,
        "checkpoint_hash": checkpoint.source_model_sha256,
        "seed": contract.seed,
        "train_sha256": contract.train_sha256,
        "probe_manifest_sha256": contract.probe_manifest_sha256,
        "prompt_count": 100,
        "pool_a_responses_per_prompt": 8,
        "pi0_controls_per_prompt": 8,
        "domain": contract.domain,
        "method": contract.method,
        "run_contract_dir": str(contract.run_dir),
    }
    if any(invocation.get(key) != value for key, value in expected_invocation.items()):
        raise ProbeAdjacentScoringError(f"fresh rubric E_{evaluator_step} invocation mismatch")
    if (
        status.get("state") != "complete"
        or int(status.get("checkpoint_step", -1)) != evaluator_step
        or status.get("checkpoint_hash") != checkpoint.source_model_sha256
        or int(status.get("prompt_count", -1)) != 100
        or status.get("fresh_rubrics_sha256") != sha256_file(rubric_path)
    ):
        raise ProbeAdjacentScoringError(f"fresh rubric E_{evaluator_step} failed status binding")
    rows = read_jsonl(rubric_path)
    rubrics: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        prompt_id = str(row.get("prompt_id", ""))
        fresh = row.get("fresh_rubric")
        if (
            row.get("state") != "verified_complete"
            or row.get("domain") != contract.domain
            or row.get("method") != contract.method
            or int(row.get("seed", -1)) != contract.seed
            or int(row.get("global_step", -1)) != evaluator_step
            or int(row.get("evaluator_checkpoint", -1)) != evaluator_step
            or row.get("fresh_or_stale") != "fresh"
            or not isinstance(fresh, Mapping)
            or prompt_id in rubrics
        ):
            raise ProbeAdjacentScoringError(f"invalid fresh-rubric row for E_{evaluator_step}")
        criteria = list(fresh.get("offline_criteria", ())) + list(
            fresh.get("online_criteria", ())
        )
        if not criteria:
            raise ProbeAdjacentScoringError(f"empty fresh rubric for {prompt_id}")
        row_invocation = row.get("invocation")
        if (
            not isinstance(row_invocation, Mapping)
            or row.get("invocation_hash") != sha256_json(row_invocation)
            or row_invocation.get("run_id") != contract.run_id
            or int(row_invocation.get("checkpoint_step", -1)) != evaluator_step
            or row_invocation.get("checkpoint_hash") != checkpoint.source_model_sha256
            or row_invocation.get("prompt_id") != prompt_id
            or int(row_invocation.get("seed", -1)) != contract.seed
            or row_invocation.get("r0_hash") != sha256_json(_r0_criteria(source_rows[prompt_id]))
            or fresh.get("content_hash") != sha256_json(criteria)
        ):
            raise ProbeAdjacentScoringError(f"fresh rubric content binding failed for {prompt_id}")
        rubrics[prompt_id] = criteria
    if set(rubrics) != set(source_rows) or len(rubrics) != 100:
        raise ProbeAdjacentScoringError("fresh rubric prompt inventory differs from fixed train probe")
    return rubrics, artifact_record(rubric_path)


def load_pool_b(contract: Any, artifact_root: Path, policy_step: int) -> tuple[list[dict[str, Any]], Mapping[str, Any]]:
    """Validate and hydrate the exact 100x16 independent Pool-B responses."""

    checkpoint = inspect_checkpoint(contract, policy_step)
    response_dir = artifact_root / "responses" / f"checkpoint-{policy_step:06d}"
    pool_path = response_dir / "probe_B.jsonl"
    provenance_path = response_dir / "provenance.json"
    if not pool_path.is_file() or not provenance_path.is_file():
        raise ProbeAdjacentScoringError(f"Pool B for pi_{policy_step} is not ready")
    provenance = read_json(provenance_path)
    for record in provenance.get("artifacts", ()):
        validate_artifact_record(record)
    expected = {
        "artifact_kind": "phase1_fixed_train_probe_policy_pools",
        "domain": contract.domain,
        "method": contract.method,
        "seed": contract.seed,
        "run_id": contract.run_id,
        "global_step": policy_step,
        "checkpoint_id": f"global_step_{policy_step}",
        "policy_checkpoint": policy_step,
        "checkpoint_hash": checkpoint.source_model_sha256,
        "config_sha256": contract.config_sha256,
        "launch_spec_sha256": contract.launch_spec_sha256,
        "probe_manifest_sha256": contract.probe_manifest_sha256,
        "train_sha256": contract.train_sha256,
    }
    if any(provenance.get(key) != value for key, value in expected.items()):
        raise ProbeAdjacentScoringError("Pool B provenance identity mismatch")
    if (
        "probe_B" not in provenance.get("selected_pools", ())
        or provenance.get("pool_counts", {}).get("probe_B") != 1600
        or not any(
            Path(str(record.get("path", ""))).resolve() == pool_path.resolve()
            and record.get("sha256") == sha256_file(pool_path)
            for record in provenance.get("artifacts", ())
        )
    ):
        raise ProbeAdjacentScoringError("Pool B provenance artifact binding mismatch")
    prompts = _probe_source_rows(contract)
    rows = read_jsonl(pool_path)
    _validate_pool_rows(
        rows,
        contract,
        checkpoint,
        pool="probe_B",
        expected_prompt_ids=set(prompts),
    )
    if len(rows) != 1600 or len({str(row["response_id"]) for row in rows}) != 1600:
        raise ProbeAdjacentScoringError("Pool B must contain 1600 unique response IDs")
    if "probe_A" in provenance.get("selected_pools", ()):
        pool_a_path = response_dir / "probe_A.jsonl"
        if not pool_a_path.is_file():
            raise ProbeAdjacentScoringError("Pool provenance names Pool A but its artifact is missing")
        pool_a = read_jsonl(pool_a_path)
        _validate_pool_rows(
            pool_a,
            contract,
            checkpoint,
            pool="probe_A",
            expected_prompt_ids=set(prompts),
        )
        validate_pool_ab_disjoint(pool_a, rows)
        a_seeds: dict[str, set[int]] = {}
        for row in pool_a:
            a_seeds.setdefault(str(row["prompt_id"]), set()).add(int(row["vllm_seed"]))
        if any(int(row["vllm_seed"]) in a_seeds[str(row["prompt_id"])] for row in rows):
            raise ProbeAdjacentScoringError("Pool A/B provider seeds collide within a prompt")
    hydrated = [
        {
            **row,
            "text": str(row["response_text"]),
            "prompt_messages": [dict(message) for message in prompts[str(row["prompt_id"])]["messages"]],
            "rollout_index": int(row["sample_index"]),
        }
        for row in rows
    ]
    hydrated.sort(key=lambda row: (str(row["prompt_id"]), int(row["sample_index"])))
    return hydrated, artifact_record(pool_path)


def endpoint_identity(url: str, model: str, revision: str) -> Mapping[str, Any]:
    with urllib.request.urlopen(url.rstrip("/") + "/v1/models", timeout=10) as response:
        data = json.load(response)
    models = [item for item in data["data"] if item["id"] == model]
    if len(models) != 1 or revision not in str(models[0].get("root", "")):
        raise ProbeAdjacentScoringError(f"unpinned judge endpoint: {url}")
    with urllib.request.urlopen(url.rstrip("/") + "/version", timeout=10) as response:
        version = json.load(response)
    return {"url": url, "model": models[0], "version": version}


def _stable_endpoint_identity(identity: Mapping[str, Any]) -> Mapping[str, Any]:
    model = identity.get("model")
    if not isinstance(model, Mapping):
        raise ProbeAdjacentScoringError("judge endpoint identity has no model record")
    stable = {
        "url": str(identity.get("url", "")).rstrip("/"),
        "model_id": str(model.get("id", "")),
        "model_root": str(model.get("root", "")),
        "max_model_len": int(model.get("max_model_len", -1)),
        "version": identity.get("version"),
    }
    if not stable["url"] or not stable["model_id"] or not stable["model_root"] or stable["max_model_len"] <= 0:
        raise ProbeAdjacentScoringError("judge endpoint stable identity is incomplete")
    return stable


def _publish_endpoint_identities(path: Path, identities: Sequence[Mapping[str, Any]]) -> None:
    current = [_stable_endpoint_identity(item) for item in identities]
    if path.is_file():
        saved = read_json(path)
        if not isinstance(saved, list) or [_stable_endpoint_identity(item) for item in saved] != current:
            raise ProbeAdjacentScoringError("judge endpoint stable identity changed on resume")
        return
    # Preserve the first raw observation for diagnostics; subsequent volatile
    # `created` and permission IDs are intentionally excluded from comparison.
    write_json_atomic(path, list(identities), immutable=True)


def authorized_cells(steps: Sequence[int]) -> tuple[tuple[int, int], ...]:
    """Return only diagonal and immediately-adjacent cells as (evaluator, policy)."""

    ordered = tuple(int(step) for step in steps)
    if not ordered or ordered != tuple(sorted(set(ordered))):
        raise ProbeAdjacentScoringError("steps must be non-empty, unique, and increasing")
    if any(step not in REGULAR_STEPS for step in ordered):
        raise ProbeAdjacentScoringError("only regular checkpoints 0,3,...,45 are supported")
    cells: list[tuple[int, int]] = []
    for policy_step in ordered:
        cells.append((policy_step, policy_step))
        if policy_step > 0:
            cells.append((policy_step - 3, policy_step))
    return tuple(cells)


def load_cell_plan(path: Path) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], Mapping[str, Any]]:
    """Load an explicit evaluator-policy cell plan without expanding it implicitly.

    Schema version 1 contains ``steps`` and ``cells``. Each cell is an object with
    integer ``evaluator_step`` and ``policy_step`` fields. Non-regular checkpoints
    are accepted only through this explicit plan path.
    """

    plan = read_json(path)
    if not isinstance(plan, Mapping) or plan.get("schema_version") != 1:
        raise ProbeAdjacentScoringError("cell plan must use schema_version 1")
    raw_steps = plan.get("steps")
    raw_cells = plan.get("cells")
    if not isinstance(raw_steps, list) or not isinstance(raw_cells, list):
        raise ProbeAdjacentScoringError("cell plan must contain list-valued steps and cells")
    if any(isinstance(step, bool) or not isinstance(step, int) for step in raw_steps):
        raise ProbeAdjacentScoringError("cell plan steps must be integers")
    steps = tuple(raw_steps)
    if not steps or steps != tuple(sorted(set(steps))) or any(step < 0 for step in steps):
        raise ProbeAdjacentScoringError(
            "cell plan steps must be non-negative, unique, and increasing"
        )
    step_set = set(steps)
    cells: list[tuple[int, int]] = []
    for index, item in enumerate(raw_cells):
        if not isinstance(item, Mapping):
            raise ProbeAdjacentScoringError(f"cell plan cell {index} must be an object")
        evaluator_step = item.get("evaluator_step")
        policy_step = item.get("policy_step")
        if (
            isinstance(evaluator_step, bool)
            or not isinstance(evaluator_step, int)
            or isinstance(policy_step, bool)
            or not isinstance(policy_step, int)
        ):
            raise ProbeAdjacentScoringError(
                f"cell plan cell {index} must contain integer evaluator_step/policy_step"
            )
        if evaluator_step not in step_set or policy_step not in step_set:
            raise ProbeAdjacentScoringError(
                f"cell plan cell {index} references a checkpoint outside steps"
            )
        if evaluator_step > policy_step:
            raise ProbeAdjacentScoringError(
                f"cell plan cell {index} has evaluator_step after policy_step"
            )
        cells.append((evaluator_step, policy_step))
    if not cells or len(cells) != len(set(cells)):
        raise ProbeAdjacentScoringError("cell plan cells must be non-empty and unique")
    return steps, tuple(cells), artifact_record(path)


def _pool_b_readiness(artifact_root: Path, policy_step: int) -> tuple[bool, str]:
    response_dir = artifact_root / "responses" / f"checkpoint-{policy_step:06d}"
    pool_path = response_dir / "probe_B.jsonl"
    provenance_path = response_dir / "provenance.json"
    if not provenance_path.is_file():
        return False, "pool_b_provenance_missing"
    provenance = read_json(provenance_path)
    if "probe_B" not in provenance.get("selected_pools", ()):
        return False, "pool_b_not_committed"
    if provenance.get("pool_counts", {}).get("probe_B") != 1600:
        raise ProbeAdjacentScoringError(
            f"Pool B provenance has an invalid completed count at pi_{policy_step}"
        )
    if not pool_path.is_file():
        raise ProbeAdjacentScoringError(
            f"Pool B provenance is committed but its file is missing at pi_{policy_step}"
        )
    return True, "ready"


def _evaluator_readiness(artifact_root: Path, evaluator_step: int) -> tuple[bool, str]:
    if evaluator_step == 0:
        return True, "initial_r0_ready"
    rubric_dir = artifact_root / "rubrics" / f"checkpoint-{evaluator_step:06d}"
    status_path = rubric_dir / "status.json"
    if not status_path.is_file():
        return False, "fresh_rubric_status_missing"
    status = read_json(status_path)
    state = status.get("state")
    if state == "running":
        return False, "fresh_rubric_running"
    if state != "complete":
        raise ProbeAdjacentScoringError(
            f"fresh rubric status is not resumable at E_{evaluator_step}: {state!r}"
        )
    for name in ("invocation.json", "fresh_rubrics.jsonl"):
        if not (rubric_dir / name).is_file():
            raise ProbeAdjacentScoringError(
                f"fresh rubric E_{evaluator_step} is complete but {name} is missing"
            )
    return True, "ready"


def _scan_ready_cells(
    artifact_root: Path,
    pending: Sequence[tuple[int, int]],
    completed: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], list[dict[str, Any]]]:
    ready: list[tuple[int, int]] = []
    blocked: list[dict[str, Any]] = []
    for evaluator_step, policy_step in pending:
        pool_ready, pool_reason = _pool_b_readiness(artifact_root, policy_step)
        evaluator_ready, evaluator_reason = _evaluator_readiness(artifact_root, evaluator_step)
        if pool_ready and evaluator_ready:
            ready.append((evaluator_step, policy_step))
        else:
            blocked.append(
                {
                    "evaluator_step": evaluator_step,
                    "policy_step": policy_step,
                    "pool_b": pool_reason,
                    "evaluator": evaluator_reason,
                }
            )

    def priority(cell: tuple[int, int]) -> tuple[int, int, int]:
        evaluator_step, policy_step = cell
        pair_started = any(done_policy == policy_step for _, done_policy in completed)
        return (0 if pair_started else 1, policy_step, 0 if evaluator_step < policy_step else 1)

    ready.sort(key=priority)
    return ready, blocked


def _completed_cell(
    manifest_path: Path,
    *,
    policy_step: int,
    evaluator_step: int,
    response_ids_hash: str,
) -> Mapping[str, Any] | None:
    if not manifest_path.is_file():
        return None
    manifest = read_json(manifest_path)
    expected = {
        "state": "complete",
        "policy_step": policy_step,
        "evaluator_step": evaluator_step,
        "response_count": 1600,
        "response_ids_sha256": response_ids_hash,
        "same_pool_b": True,
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ProbeAdjacentScoringError(f"completed cell manifest identity mismatch: {manifest_path}")
    validate_artifact_record(manifest["scores"])
    rows = read_jsonl(Path(str(manifest["scores"]["path"])))
    if (
        len(rows) != 1600
        or sha256_json(sorted(str(row["response_id"]) for row in rows)) != response_ids_hash
        or any(
            int(row["policy_step"]) != policy_step
            or int(row["evaluator_step"]) != evaluator_step
            or row.get("pool") != "probe_B"
            for row in rows
        )
    ):
        raise ProbeAdjacentScoringError(f"completed cell score inventory is corrupt: {manifest_path}")
    return manifest


def score_regular_adjacent(
    *,
    run_dir: Path,
    artifact_root: Path,
    output_root: Path,
    steps: Sequence[int],
    judge_urls: Sequence[str],
    concurrency: int = 32,
    wait_timeout_seconds: float = 0.0,
    cell_plan_path: Path | None = None,
    bounded_grading_whitespace: bool = False,
) -> Mapping[str, Any]:
    """Score adjacent defaults or exactly the cells named by an explicit plan."""

    if cell_plan_path is None:
        ordered = tuple(int(step) for step in steps)
        cells = authorized_cells(ordered)
        plan_artifact = None
        analysis = "fixed_train_probe_fresh_adjacent"
    else:
        ordered, cells, plan_artifact = load_cell_plan(cell_plan_path)
        supplied_steps = tuple(int(step) for step in steps)
        if supplied_steps and supplied_steps != ordered:
            raise ProbeAdjacentScoringError("CLI steps differ from explicit cell plan steps")
        analysis = "fixed_train_probe_explicit_cell_plan"
    status_context: dict[str, Any] = {
        "analysis": analysis,
        "authorized_cell_count": len(cells),
    }
    if bounded_grading_whitespace:
        status_context['structured_output_format'] = 'binary_grader_bounded_whitespace_v1'
    if plan_artifact is not None:
        status_context["cell_plan"] = plan_artifact
    if concurrency <= 0 or wait_timeout_seconds < 0 or not judge_urls:
        raise ProbeAdjacentScoringError("judge URLs/concurrency/wait timeout are invalid")
    contract = load_run_contract(run_dir)
    source_config = read_json(contract.config_path)
    judge = source_config["models"]["judge"]
    identities = [endpoint_identity(url, judge["model"], judge["revision"]) for url in judge_urls]
    if len({json.dumps(item["version"], sort_keys=True) for item in identities}) != 1:
        raise ProbeAdjacentScoringError("judge replicas run different vLLM versions")
    output_root.mkdir(parents=True, exist_ok=True)
    _publish_endpoint_identities(output_root / "judge_endpoints.json", identities)
    grader = VLLMChatAdapter(
        judge_urls,
        judge["model"],
        output_root / "provider_cache",
        timeout_seconds=600,
        max_retries=4,
        bounded_grading_whitespace=bounded_grading_whitespace,
    )
    config = AuditScoreConfig(
        domain=contract.domain,
        method=contract.method,
        seed=contract.seed,
        judge_model=judge["model"],
        judge_revision=judge["revision"],
        max_output_tokens=int(judge["max_output_tokens"]),
        concurrency=concurrency,
    )
    completed: list[dict[str, Any]] = []
    completed_cells: set[tuple[int, int]] = set()
    pending = list(cells)
    pool_cache: dict[int, tuple[list[dict[str, Any]], Mapping[str, Any], str]] = {}
    evaluator_cache: dict[
        int, tuple[dict[str, list[dict[str, Any]]], Mapping[str, Any]]
    ] = {}
    run_started = time.monotonic()
    run_started_at = datetime.now(timezone.utc).isoformat()
    deadline = run_started + wait_timeout_seconds
    while pending:
        ready, blocked = _scan_ready_cells(artifact_root, pending, completed_cells)
        if not ready:
            write_json_atomic(
                output_root / "status.json",
                {
                    "schema_version": 1,
                    "state": "waiting_inputs",
                    **status_context,
                    "scheduler_policy": SCHEDULER_POLICY,
                    "ready_cell_count": 0,
                    "pending_cell_count": len(pending),
                    "completed_cell_count": len(completed),
                    "blocked_cells": blocked,
                    "started_at": run_started_at,
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "elapsed_seconds": time.monotonic() - run_started,
                },
                immutable=False,
            )
            if time.monotonic() >= deadline:
                raise ProbeAdjacentScoringError(
                    f"no pending scoring cell became ready; blocked={blocked}"
                )
            time.sleep(5)
            continue

        evaluator_step, policy_step = ready[0]
        if policy_step not in pool_cache:
            responses, response_artifact = load_pool_b(contract, artifact_root, policy_step)
            response_ids_hash = sha256_json(
                sorted(str(row["response_id"]) for row in responses)
            )
            pool_cache[policy_step] = (responses, response_artifact, response_ids_hash)
        responses, response_artifact, response_ids_hash = pool_cache[policy_step]
        if evaluator_step not in evaluator_cache:
            evaluator_cache[evaluator_step] = load_evaluator_rubrics(
                contract, artifact_root, evaluator_step
            )
        rubrics, rubric_artifact = evaluator_cache[evaluator_step]
        cell_dir = (
            output_root
            / f"policy-{policy_step:06d}"
            / f"evaluator-{evaluator_step:06d}"
        )
        receipt_path = cell_dir / "scores.jsonl"
        manifest_path = cell_dir / "manifest.json"
        existing = _completed_cell(
            manifest_path,
            policy_step=policy_step,
            evaluator_step=evaluator_step,
            response_ids_hash=response_ids_hash,
        )
        if existing is not None:
            completed.append(dict(existing))
            completed_cells.add((evaluator_step, policy_step))
            pending.remove((evaluator_step, policy_step))
            deadline = time.monotonic() + wait_timeout_seconds
            continue

        cell_started = time.monotonic()
        cell_started_at = datetime.now(timezone.utc).isoformat()
        write_json_atomic(
            output_root / "status.json",
            {
                "schema_version": 1,
                "state": "scoring",
                **status_context,
                "scheduler_policy": SCHEDULER_POLICY,
                "current_policy_step": policy_step,
                "current_evaluator_step": evaluator_step,
                "fresh_or_stale": "fresh" if evaluator_step == policy_step else "stale",
                "ready_cell_count": len(ready),
                "pending_cell_count": len(pending),
                "completed_cell_count": len(completed),
                "blocked_cells": blocked,
                "started_at": run_started_at,
                "cell_started_at": cell_started_at,
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": time.monotonic() - run_started,
            },
            immutable=False,
        )
        records = score_pool(
            responses,
            rubrics,
            evaluator_checkpoint=str(evaluator_step),
            policy_checkpoint=str(policy_step),
            pool="probe_B",
            config=config,
            grader=grader,
            cache_dir=cell_dir / "grade_cache",
        )
        if [str(row["response_id"]) for row in records] != [
            str(row["response_id"]) for row in responses
        ]:
            raise ProbeAdjacentScoringError(
                "grader changed the Pool-B response identity order"
            )
        write_score_receipts(receipt_path, records)
        manifest = {
            "schema_version": 1,
            "state": "complete",
            "analysis": analysis,
            "domain": contract.domain,
            "method": contract.method,
            "seed": contract.seed,
            "policy_step": policy_step,
            "evaluator_step": evaluator_step,
            "fresh_or_stale": "fresh" if evaluator_step == policy_step else "stale",
            "prompt_count": 100,
            "response_count": 1600,
            "response_ids_sha256": response_ids_hash,
            "pool_b": response_artifact,
            "rubric": rubric_artifact,
            "scores": artifact_record(receipt_path),
            "judge_model": judge["model"],
            "judge_revision": judge["revision"],
            "seed_recipe": "11 + sample_index",
            "same_pool_b": True,
            "initial_rubric_is_ground_truth": False,
            "started_at": cell_started_at,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": time.monotonic() - cell_started,
        }
        if plan_artifact is not None:
            manifest["cell_plan"] = plan_artifact
        write_json_atomic(manifest_path, manifest, immutable=True)
        completed.append(manifest)
        completed_cells.add((evaluator_step, policy_step))
        pending.remove((evaluator_step, policy_step))
        deadline = time.monotonic() + wait_timeout_seconds

    summary = {
        "schema_version": 1,
        "state": "complete",
        "steps": list(ordered),
        "analysis": analysis,
        "scheduler_policy": SCHEDULER_POLICY,
        "authorized_cell_count": len(cells),
        "cell_count": len(completed),
        "score_count": sum(int(item["response_count"]) for item in completed),
        "started_at": run_started_at,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": time.monotonic() - run_started,
        "initial_rubric_is_ground_truth": False,
        "cells": [
            {"evaluator_step": item["evaluator_step"], "policy_step": item["policy_step"]}
            for item in completed
        ],
    }
    if plan_artifact is not None:
        summary["cell_plan"] = plan_artifact
    write_json_atomic(output_root / "status.json", summary, immutable=False)
    return summary
