from __future__ import annotations

import hashlib
from collections import defaultdict
from typing import Any

from .artifacts import read_jsonl, write_json_atomic, write_jsonl_atomic
from .hashing import canonical_json_bytes
from .pipeline import PipelineContext, StageError, _all_public_prompts
from .providers.base import GenerationRequest
from .providers.fake import FakeGenerator
from .seeds import SeedFamily, derive_seed, response_id


def _audit_prompts(context: PipelineContext) -> list[dict[str, Any]]:
    return [row for row in _all_public_prompts(context) if "audit" in str(row["split"])]


def run_generate_bon(context: PipelineContext) -> dict[str, Any]:
    checkpoints = context.run_root / "train-static" / "checkpoints.json"
    context.begin_stage(inputs=(checkpoints,), metadata={"shared_across_rubrics": True})
    if context.mode != "fake":
        raise StageError("live BoN generation requires the focal checkpoint endpoint registry")
    bon_cfg = context.raw.get("bon", {})
    focal_steps = [int(value) for value in bon_cfg.get("focal_steps", [3, 10, 30, 100])]
    pool_size = int(bon_cfg.get("pool_size", 64))
    generator = FakeGenerator("fake/qwen3-4b-v1")
    rows: list[dict[str, Any]] = []
    prompts = _audit_prompts(context)
    for policy_step in focal_steps:
        for prompt_index, prompt in enumerate(prompts):
            prompt_id = str(prompt["prompt_id"])
            for sample_index in range(pool_size):
                seed = derive_seed(
                    context.run_id, SeedFamily.AUDIT_BON, prompt_id, policy_step, sample_index
                )
                generated = generator.generate(
                    GenerationRequest(
                        prompt_id=prompt_id,
                        messages=tuple(prompt["messages"]),
                        family=SeedFamily.AUDIT_BON.value,
                        seed=seed,
                        temperature=1.0,
                        top_p=0.95,
                    )
                )
                rows.append(
                    {
                        "policy_id": f"pi_{policy_step}",
                        "policy_step": policy_step,
                        "prompt_id": prompt_id,
                        "sample_index": sample_index,
                        "global_candidate_id": policy_step * 10**9
                        + prompt_index * pool_size
                        + sample_index,
                        "response_id": response_id(
                            context.run_id,
                            SeedFamily.AUDIT_BON,
                            prompt_id,
                            policy_step,
                            sample_index,
                        ),
                        "seed": seed,
                        "response_text": generated.text,
                        "provider_call": {
                            "requested_model": generated.requested_model,
                            "returned_model": generated.returned_model,
                            "request_id": generated.request_id,
                            "created_at": generated.created_at,
                            "retry_count": generated.retry_count,
                        },
                    }
                )
    write_jsonl_atomic(context.stage_root() / "bon_pool.jsonl", rows)
    result = {
        "candidate_count": len(rows),
        "expected": len(focal_steps) * len(prompts) * pool_size,
        "focal_steps": focal_steps,
        "prompts": len(prompts),
        "pool_size": pool_size,
        "rubric_independent_generation": True,
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def _evaluation_rubrics(context: PipelineContext) -> list[dict[str, Any]]:
    static_rows = read_jsonl(context.run_root / "generate-static" / "static_rubrics.jsonl")
    replay_rows = read_jsonl(context.run_root / "replay-dynamic-final" / "replay_snapshots.jsonl")
    requested_steps = {int(value) for value in context.raw.get("bon", {}).get("rubric_steps", [])}
    result = [
        {
            "prompt_id": row["prompt_id"],
            "rubric_id": row["rubric_id"],
            "mode": "static",
            "rubric_step": 0,
            "criteria": row["criteria"],
        }
        for row in static_rows
    ]
    for row in replay_rows:
        if row["mode"] == "static":
            continue
        if requested_steps and int(row["policy_step"]) not in requested_steps:
            continue
        result.append(
            {
                "prompt_id": row["prompt_id"],
                "rubric_id": f"{row['prompt_id']}:{row['mode']}:R_{row['policy_step']}",
                "mode": row["mode"],
                "rubric_step": int(row["policy_step"]),
                "criteria": row["criteria"],
            }
        )
    return result


def _deterministic_probability(*parts: object) -> float:
    digest = hashlib.sha256(canonical_json_bytes(parts)).digest()
    return int.from_bytes(digest[:8], "big") / (2**64 - 1)


def run_score_proxy(context: PipelineContext) -> dict[str, Any]:
    bon_path = context.run_root / "generate-bon" / "bon_pool.jsonl"
    static_path = context.run_root / "generate-static" / "static_rubrics.jsonl"
    replay_path = context.run_root / "replay-dynamic-final" / "replay_snapshots.jsonl"
    context.begin_stage(
        inputs=(bon_path, static_path, replay_path), metadata={"criterion_cache": True}
    )
    if context.mode != "fake":
        raise StageError("live proxy scoring requires the pinned Qwen target-loglikelihood sidecar")
    pools = read_jsonl(bon_path)
    rubrics_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rubric in _evaluation_rubrics(context):
        rubrics_by_prompt[str(rubric["prompt_id"])].append(rubric)
    criterion_cache: dict[tuple[str, str], tuple[float, float]] = {}
    criterion_rows: list[dict[str, Any]] = []
    rubric_scores: list[dict[str, Any]] = []
    for candidate in pools:
        response_id_value = str(candidate["response_id"])
        for rubric in rubrics_by_prompt[str(candidate["prompt_id"])]:
            scores: list[float] = []
            repeat_scores: list[float] = []
            for criterion in rubric["criteria"]:
                key = response_id_value, str(criterion["criterion_id"])
                if key not in criterion_cache:
                    score = _deterministic_probability(
                        candidate["prompt_id"],
                        candidate["response_text"],
                        criterion["criterion_id"],
                        criterion["text"],
                        "fake/qwen3-32b-v1",
                    )
                    repeat_score = _deterministic_probability(
                        candidate["prompt_id"],
                        candidate["response_text"],
                        criterion["criterion_id"],
                        criterion["text"],
                        "fake/qwen3-32b-v1",
                        "judge-repeat-B",
                    )
                    criterion_cache[key] = score, repeat_score
                    criterion_rows.append(
                        {
                            "prompt_id": candidate["prompt_id"],
                            "response_id": response_id_value,
                            "criterion_id": criterion["criterion_id"],
                            "probability_yes": score,
                            "repeat_probability_yes": repeat_score,
                            "parse_success": True,
                        }
                    )
                score, repeat_score = criterion_cache[key]
                scores.append(score)
                repeat_scores.append(repeat_score)
            rubric_scores.append(
                {
                    "policy_id": candidate["policy_id"],
                    "policy_step": candidate["policy_step"],
                    "prompt_id": candidate["prompt_id"],
                    "global_candidate_id": candidate["global_candidate_id"],
                    "response_id": response_id_value,
                    "rubric_id": rubric["rubric_id"],
                    "mode": rubric["mode"],
                    "rubric_step": rubric["rubric_step"],
                    "score": sum(scores) / len(scores),
                    "judge_repeat_score": sum(repeat_scores) / len(repeat_scores),
                }
            )
    write_jsonl_atomic(context.stage_root() / "criterion_scores.jsonl", criterion_rows)
    write_jsonl_atomic(context.stage_root() / "rubric_scores.jsonl", rubric_scores)
    result = {
        "criterion_pairs_scored_once": len(criterion_rows),
        "rubric_scores_assembled": len(rubric_scores),
        "parse_success": 1.0,
        "scoring_mode": "normalized_yes_no_target_loglikelihood_fake",
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result


def run_select_bon(context: PipelineContext) -> dict[str, Any]:
    from .evaluation.bon import fixed_candidate_permutations, select_best_of_n, shared_pool_hash

    bon_path = context.run_root / "generate-bon" / "bon_pool.jsonl"
    scores_path = context.run_root / "score-proxy" / "rubric_scores.jsonl"
    context.begin_stage(inputs=(bon_path, scores_path), metadata={"shared_pool_required": True})
    pools = read_jsonl(bon_path)
    scores = read_jsonl(scores_path)
    candidate_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in pools:
        candidate_groups[(str(row["policy_id"]), str(row["prompt_id"]))].append(row)
    score_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in scores:
        score_groups[(str(row["policy_id"]), str(row["prompt_id"]), str(row["rubric_id"]))].append(
            row
        )
    sizes = [
        int(value) for value in context.raw.get("bon", {}).get("sizes", [1, 2, 4, 8, 16, 32, 64])
    ]
    permutation_count = int(context.raw.get("bon", {}).get("permutations", 5))
    selections: list[dict[str, Any]] = []
    pool_hashes: set[str] = set()
    for (policy_id, prompt_id, rubric_id), rubric_score_rows in sorted(score_groups.items()):
        candidates = candidate_groups[(policy_id, prompt_id)]
        candidate_by_id = {row["global_candidate_id"]: row for row in candidates}
        score_by_id = {row["global_candidate_id"]: float(row["score"]) for row in rubric_score_rows}
        pool_hash = shared_pool_hash(
            [
                {"candidate_id": row["global_candidate_id"], "response_text": row["response_text"]}
                for row in candidates
            ]
        )
        pool_hashes.add(pool_hash)
        sample = rubric_score_rows[0]
        permutations = fixed_candidate_permutations(
            list(candidate_by_id),
            seed=f"{context.run_id}:{policy_id}:{prompt_id}",
            count=permutation_count,
        )
        for permutation_index, permutation in enumerate(permutations):
            for n in sizes:
                if n > len(permutation):
                    continue
                selected_id = select_best_of_n(score_by_id, permutation, n)
                selected = candidate_by_id[selected_id]
                selections.append(
                    {
                        "policy_id": policy_id,
                        "policy_step": sample["policy_step"],
                        "prompt_id": prompt_id,
                        "rubric_id": rubric_id,
                        "mode": sample["mode"],
                        "rubric_step": sample["rubric_step"],
                        "n": n,
                        "permutation": permutation_index,
                        "pool_hash": pool_hash,
                        "global_candidate_id": selected_id,
                        "response_id": selected["response_id"],
                        "response_text": selected["response_text"],
                    }
                )
    write_jsonl_atomic(context.stage_root() / "selections.jsonl", selections)
    result = {
        "selections": len(selections),
        "pool_hashes": len(pool_hashes),
        "same_pool_hash_shared_by_rubrics": True,
        "tie_break": "lowest_global_candidate_id",
    }
    write_json_atomic(context.stage_root() / "result.json", result)
    return result
