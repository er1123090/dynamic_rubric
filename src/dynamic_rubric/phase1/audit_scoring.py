"""Resumable, provenance-preserving scoring for Phase-1 audit pools.

This module deliberately scores saved responses only.  Response generation and
rubric construction belong to the pool-generation runner; keeping the grader
here makes every fresh/stale comparison use the same response IDs and judge.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from ..prompt_versions.onlinerubric_grader_prompt import (
    build_onlinerubric_grader_messages,
    onlinerubric_grader_schema,
)
from ..providers.base import GenerationRequest, GenerationResult
from ..training.paper_reward import grade_and_compute
from ..training.online_contracts import WeightedCriterion


@dataclasses.dataclass(frozen=True, slots=True)
class AuditScoreConfig:
    domain: str
    method: str = "online_rubrics"
    seed: int = 11
    judge_model: str = "Qwen/Qwen3-32B"
    judge_revision: str = ""
    max_output_tokens: int = 4096
    concurrency: int = 8


def _criteria(criteria: Sequence[Mapping[str, Any]]) -> tuple[WeightedCriterion, ...]:
    out: list[WeightedCriterion] = []
    for item in criteria:
        if isinstance(item, WeightedCriterion):
            out.append(item)
            continue
        cid = str(item.get("criterion_id", ""))
        text = str(item.get("text", item.get("criterion", ""))).strip()
        if not cid or not text:
            raise ValueError("criteria require criterion_id and text")
        out.append(
            WeightedCriterion(
                cid, text, int(item.get("weight", 1)), str(item.get("source", "audit"))
            )
        )
    if not out:
        raise ValueError("rubric must contain at least one criterion")
    return tuple(out)


def _request(
    config: AuditScoreConfig,
    response: Mapping[str, Any],
    rubric: Sequence[Mapping[str, Any]] | Sequence[WeightedCriterion],
    *,
    evaluator_checkpoint: str,
) -> GenerationRequest:
    criteria = _criteria(rubric)
    return GenerationRequest(
        prompt_id=str(response.get("prompt_id", response.get("prompt_occurrence_id", ""))),
        messages=build_onlinerubric_grader_messages(
            prompt=response.get("prompt_messages")
            or [
                {"role": "user", "content": str(response.get("prompt", response.get("input", "")))}
            ],
            response=str(response.get("text", response.get("output", ""))),
            criteria=[
                {
                    "criterion_id": c.criterion_id,
                    "criterion": c.text,
                    "weight": c.weight,
                    "source": c.source,
                }
                for c in criteria
            ],
        ),
        family="phase1_audit_grading",
        # Preserve the OnlineRubrics training recipe exactly.
        seed=config.seed + int(response.get("rollout_index", 0)),
        max_output_tokens=config.max_output_tokens,
        json_schema=onlinerubric_grader_schema(len(criteria)),
        schema_name="onlinerubric_grader_v1",
        metadata={
            "audit": True,
            "evaluator_checkpoint": evaluator_checkpoint,
            "response_id": str(response["response_id"]),
        },
    )


def score_pool(
    responses: Sequence[Mapping[str, Any]],
    rubric_by_prompt: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    evaluator_checkpoint: str,
    policy_checkpoint: str,
    pool: str = "probe_B",
    config: AuditScoreConfig,
    grader: Any,
    cache_dir: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Grade saved responses with one evaluator and return immutable receipts.

    Existing per-response cache records are reused after identity validation.
    ``grader`` is any existing GenerationProvider implementing ``generate``.
    """
    if not responses or not rubric_by_prompt:
        raise ValueError("responses and rubric_by_prompt must be non-empty")
    if pool not in {"train_batch", "probe_A", "probe_B"}:
        raise ValueError(f"invalid pool: {pool}")
    cache = Path(cache_dir) if cache_dir else None
    if cache:
        cache.mkdir(parents=True, exist_ok=True)
    pending: list[tuple[Mapping[str, Any], GenerationRequest, tuple[dict[str, Any], ...]]] = []
    records: dict[str, dict[str, Any]] = {}
    for response in responses:
        rid = str(response.get("response_id", ""))
        pid = str(response.get("prompt_id", response.get("prompt_occurrence_id", "")))
        if not rid or not pid or pid not in rubric_by_prompt:
            raise ValueError("each response needs response_id, prompt ID, and a matching rubric")
        rub = _criteria(rubric_by_prompt[pid])
        rubric_hash = hashlib.sha256(
            json.dumps([dataclasses.asdict(c) for c in rub], sort_keys=True).encode()
        ).hexdigest()
        identity = json.dumps(
            {
                "evaluator": evaluator_checkpoint,
                "policy": policy_checkpoint,
                "response": rid,
                "rubric": rubric_hash,
                "judge": config.judge_model,
                "revision": config.judge_revision,
                "max_output_tokens": config.max_output_tokens,
                "seed": config.seed,
            },
            sort_keys=True,
        )
        key = hashlib.sha256(identity.encode()).hexdigest()
        path = cache / f"{key}.json" if cache else None
        if path and path.is_file():
            value = json.loads(path.read_text())
            if (
                value.get("response_id") != rid
                or value.get("evaluator_checkpoint") != evaluator_checkpoint
                or value.get("rubric_hash") != rubric_hash
            ):
                raise ValueError(f"cache identity mismatch for {rid}")
            records[rid] = value
            continue
        pending.append(
            (
                response,
                _request(config, response, rub, evaluator_checkpoint=evaluator_checkpoint),
                rub,
            )
        )

    def run(
        item: tuple[Mapping[str, Any], GenerationRequest, tuple[dict[str, Any], ...]],
    ) -> dict[str, Any]:
        response, request, rubric = item
        result: GenerationResult = grader.generate(request)
        calc = grade_and_compute(result.text, rubric)
        rid = str(response["response_id"])
        value = {
            "schema_version": 1,
            "domain": config.domain,
            "method": config.method,
            "seed": config.seed,
            "global_step": int(response.get("global_step", response.get("step", 0))),
            "policy_step": int(
                response.get("policy_step", response.get("global_step", response.get("step", 0)))
            ),
            "evaluator_step": int(evaluator_checkpoint)
            if str(evaluator_checkpoint).isdigit()
            else evaluator_checkpoint,
            "checkpoint_id": str(policy_checkpoint),
            "prompt_id": str(response.get("prompt_id", response.get("prompt_occurrence_id", ""))),
            "response_id": rid,
            "pool": pool,
            "policy_checkpoint": policy_checkpoint,
            "evaluator_checkpoint": evaluator_checkpoint,
            "fresh_or_stale": "fresh" if evaluator_checkpoint == policy_checkpoint else "stale",
            "grades": [[str(cid), int(grade)] for cid, grade in calc.grades],
            "numerator": float(calc.numerator),
            "denominator": float(calc.denominator),
            "reward": float(calc.scalar),
            "rubric_hash": hashlib.sha256(
                json.dumps([dataclasses.asdict(c) for c in rubric], sort_keys=True).encode()
            ).hexdigest(),
            "judge": {
                "requested_model": result.requested_model,
                "returned_model": result.returned_model,
                "revision": config.judge_revision,
                "request_id": result.request_id,
                "created_at": result.created_at,
                "retry_count": result.retry_count,
                "usage": dict(result.usage),
                "raw_response_hash": result.raw_response_hash,
            },
            "same_response_pool_key": rid,
        }
        provenance = getattr(grader, "request_provenance", None)
        if provenance is not None:
            value["judge"]["transport"] = dict(provenance(request))
        if cache:
            identity = json.dumps(
                {
                    "evaluator": evaluator_checkpoint,
                    "policy": policy_checkpoint,
                    "response": rid,
                    "rubric": value["rubric_hash"],
                    "judge": config.judge_model,
                    "revision": config.judge_revision,
                    "max_output_tokens": config.max_output_tokens,
                    "seed": config.seed,
                },
                sort_keys=True,
            )
            key = hashlib.sha256(identity.encode()).hexdigest()
            write_json_atomic(cache / f"{key}.json", value, immutable=True)
        return value

    with ThreadPoolExecutor(max_workers=max(1, int(config.concurrency))) as executor:
        for value in executor.map(run, pending):
            records[str(value["response_id"])] = value
    return [records[str(row["response_id"])] for row in responses]


def write_score_receipts(path: str | Path, records: Sequence[Mapping[str, Any]]) -> None:
    """Atomically publish an ordered score receipt file; safe to rerun."""
    ordered = sorted((dict(row) for row in records), key=lambda row: str(row["response_id"]))
    write_jsonl_atomic(path, ordered, immutable=True)


def score_canonical_train_steps(
    run_root: str | Path,
    *,
    steps: Sequence[int],
    prompt_messages_by_id: Mapping[str, Sequence[Mapping[str, str]]],
    grader: Any,
    config: AuditScoreConfig,
    output_root: str | Path,
    concurrency: int | None = None,
) -> dict[str, Any]:
    """Re-grade canonical committed train steps with the prompt-matched prior rubric.

    This is the operational stale-shadow path: step ``t`` responses are scored
    with the rubric union from step ``t`` and the latest earlier rubric for the
    same prompt.  Incomplete steps are skipped; each evaluator's receipts are
    immutable and resumable.
    """
    root = Path(run_root) / "online_steps"
    out = Path(output_root)
    ordered = sorted({int(s) for s in steps if int(s) > 0})
    committed = [s for s in ordered if (root / f"step-{s:06d}" / "commit.json").is_file()]
    rubrics: dict[int, dict[str, list[dict[str, Any]]]] = {}
    for s in committed:
        path = root / f"step-{s:06d}" / "rubric_unions.jsonl"
        if not path.is_file():
            continue
        rubrics[s] = {}
        for row in read_jsonl(path):
            pid = str(row["prompt_occurrence_id"])
            rubrics[s][pid] = list(row.get("offline_criteria", [])) + list(
                row.get("online_criteria", [])
            )

    def prompt_key(value: str) -> str:
        # train:<source index>:<visit hash>; the source index is the stable
        # prompt identity across visits/checkpoints.
        parts = str(value).split(":")
        return ":".join(parts[:2]) if len(parts) >= 2 else str(value)

    results = []
    for s in committed:
        response_path = root / f"step-{s:06d}" / "current_responses.jsonl"
        if not response_path.is_file() or s not in rubrics:
            continue
        responses = []
        for row in read_jsonl(response_path):
            pid = str(row["prompt_occurrence_id"])
            if pid not in prompt_messages_by_id:
                raise ValueError(f"missing prompt messages for {pid}")
            responses.append(
                dict(row, prompt_messages=prompt_messages_by_id[pid], policy_step=s - 1)
            )
        # Fresh scores are already the canonical training reward and must be
        # reused; only stale evaluators incur new judge calls.
        saved = {
            str(x["response_id"]): x for x in read_jsonl(root / f"step-{s:06d}" / "rewards.jsonl")
        }
        fresh = []
        for row in responses:
            value = saved.get(str(row["response_id"]))
            if value is None:
                raise ValueError(f"missing canonical fresh reward for {row['response_id']}")
            fresh.append(
                dict(
                    value,
                    domain=config.domain,
                    method=config.method,
                    seed=config.seed,
                    global_step=s,
                    policy_step=s - 1,
                    evaluator_step=s,
                    checkpoint_id=str(s),
                    prompt_id=str(row["prompt_occurrence_id"]),
                    pool="train_batch",
                    policy_checkpoint=str(s - 1),
                    evaluator_checkpoint=str(s),
                    fresh_or_stale="fresh",
                )
            )
        previous_by_prompt = {}
        for p in committed:
            if p >= s or p not in rubrics:
                continue
            for pid in rubrics[p]:
                previous_by_prompt[prompt_key(pid)] = (p, pid)
        stale = []
        stale_groups = {}
        for row in responses:
            match = previous_by_prompt.get(prompt_key(str(row["prompt_occurrence_id"])))
            if match:
                p, old_pid = match
                stale_groups.setdefault((p, old_pid), []).append(row)
        for (previous, old_pid), group in stale_groups.items():
            stale.extend(
                score_pool(
                    group,
                    {str(row["prompt_occurrence_id"]): rubrics[previous][old_pid] for row in group},
                    evaluator_checkpoint=str(previous),
                    policy_checkpoint=str(s - 1),
                    pool="train_batch",
                    config=config,
                    grader=grader,
                    cache_dir=out / f"step-{s:06d}" / f"stale-{previous}-cache",
                )
            )
        write_score_receipts(out / f"step-{s:06d}" / "fresh.jsonl", fresh)
        if stale:
            write_score_receipts(out / f"step-{s:06d}" / f"stale-{previous}.jsonl", stale)
        results.append(
            {
                "policy_step": s - 1,
                "optimizer_step": s,
                "fresh_evaluator_step": s,
                "stale_evaluator_steps": sorted({int(x["evaluator_checkpoint"]) for x in stale}),
                "fresh_count": len(fresh),
                "stale_count": len(stale),
            }
        )
    manifest = {
        "schema_version": 1,
        "mode": "actual_train_shadow",
        "steps": results,
        "heldout_used": False,
        "ground_truth_used": False,
    }
    write_json_atomic(out / "manifest.json", manifest, immutable=True)
    return manifest
