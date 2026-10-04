"""Fail-closed Pool-B grading and prompt-level horizon metric artifacts."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from ..artifacts import read_json, read_jsonl, write_json_atomic, write_jsonl_atomic
from ..config import HORIZON_POLICY_TRAINING_REGIMES, STATIC_EXPERIMENT_ARM
from ..hashing import sha256_file
from ..providers.vllm import FullCriterionScore
from .advantage import grpo_scalar_advantages
from .contracts import (
    CriterionType,
    ImportanceClass,
    WeightedCriterion,
    WeightedRubric,
    criterion_content_hash,
)
from .grading import (
    NO_TARGET,
    TARGET_ENCODING_VERSION,
    YES_TARGET,
    HardGrade,
    assemble_variant_scores,
    hard_grade_from_target_logprobs,
)
from .live_rubrics import criterion_from_artifact
from .metrics import (
    advantage_degenerate,
    criterion_effectiveness_summary,
    exact_zero_advantage,
    low_reward_spread,
    pairwise_metrics,
    ranking_agreement,
    score_spread,
)


class HorizonGrader(Protocol):
    def score_many_full(
        self,
        items: Sequence[tuple[str, str, str, str, str]],
        *,
        prompt_text_by_id: Mapping[str, str] | None = None,
    ) -> tuple[FullCriterionScore, ...]: ...


HORIZON_GRADING_POOL_SIZES = {"pool_a": 8, "pool_a_combined": 16, "pool_b": 16}
RUBRIC_VARIANT_LABELS = {
    "r0": "R0",
    "control": "R_{t-delta}",
    "current": "R_t",
}


def r0_criterion_from_artifact(item: Mapping[str, Any]) -> WeightedCriterion:
    text = str(item["criterion"])
    expected_hash = criterion_content_hash(text)
    observed_hash = str(item.get("canonical_criterion_hash", expected_hash))
    if observed_hash != expected_hash:
        raise ValueError("R0 criterion hash does not match its text")
    return WeightedCriterion(
        criterion_instance_id=str(item["criterion_id"]),
        canonical_criterion_hash=observed_hash,
        text=text,
        importance_class=ImportanceClass(str(item["importance_class"])),
        criterion_type=CriterionType(str(item["criterion_type"])),
        weight_units=int(item["weight_units"]),
        source_checkpoint="0",
    )


def _index_unique(
    rows: Sequence[Mapping[str, Any]], key: str, *, label: str
) -> dict[str, Mapping[str, Any]]:
    indexed: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        value = str(row[key])
        if value in indexed:
            raise ValueError(f"duplicate {label}: {value}")
        indexed[value] = row
    return indexed


def _score_tuple(score: Any) -> tuple[int, int]:
    return int(score.numerator), int(score.denominator)


def _variant_summary(
    scores: Sequence[tuple[int, int]], *, epsilon: float, delta: float
) -> dict[str, Any]:
    values = [numerator / denominator for numerator, denominator in scores]
    advantages = grpo_scalar_advantages(values)
    return {
        "exact_zar": exact_zero_advantage(scores),
        "near_zero_zar": low_reward_spread(scores, epsilon=epsilon),
        "advantage_degenerate": advantage_degenerate(advantages, delta=delta),
        "advantages": list(advantages),
        "spread": score_spread(scores),
        "pairwise": pairwise_metrics(scores),
    }


def _hard_grade_from_artifact(row: Mapping[str, Any]) -> HardGrade:
    grade = hard_grade_from_target_logprobs(
        {
            YES_TARGET: (float(row["yes_logprob"]),),
            NO_TARGET: (float(row["no_logprob"]),),
        },
        retry_count=int(row.get("retry_count", 0)),
    )
    for field in ("parse_status", "public_label", "grade"):
        if row.get(field) != getattr(grade, field):
            raise ValueError(f"reused grade {field} disagrees with target logprobs")
    return grade


def grade_horizon_pool(
    grader: HorizonGrader,
    *,
    prompts: Sequence[Mapping[str, Any]],
    pool_rows: Sequence[Mapping[str, Any]],
    pool_family: str,
    rubric_rows: Sequence[Mapping[str, Any]],
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    source_paths: Mapping[str, Path] | None = None,
    reused_grade_rows: Sequence[Mapping[str, Any]] = (),
    include_control: bool = True,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
    target_encoding_version: str = TARGET_ENCODING_VERSION,
) -> dict[str, Any]:
    """Grade one seed/checkpoint shard and write grades, scores, summary, and seal."""

    if policy_training_regime not in HORIZON_POLICY_TRAINING_REGIMES:
        raise ValueError("unsupported policy training regime")
    if pool_family not in HORIZON_GRADING_POOL_SIZES:
        raise ValueError(f"unsupported grading pool family: {pool_family}")
    expected_response_count = HORIZON_GRADING_POOL_SIZES[pool_family]
    pool_label = pool_family.replace("_", " ").title()
    if not pool_rows:
        raise ValueError(f"{pool_label} rows must not be empty")
    prompt_by_id = _index_unique(prompts, "prompt_id", label="prompt")
    rubric_by_id = _index_unique(rubric_rows, "prompt_id", label="rubric prompt")
    pool_groups: dict[str, list[Mapping[str, Any]]] = {}
    seeds: set[str] = set()
    policy_steps: set[int] = set()
    for row in pool_rows:
        if str(row.get("pool_family")) != pool_family:
            raise ValueError(f"grading input must contain only {pool_label} rows")
        pool_groups.setdefault(str(row["prompt_id"]), []).append(row)
        seeds.add(str(row["training_seed"]))
        policy_steps.add(int(row["policy_step"]))
    if len(seeds) != 1 or len(policy_steps) != 1:
        raise ValueError("one grading shard must bind exactly one seed and policy step")
    if set(pool_groups) != set(prompt_by_id) or set(rubric_by_id) != set(prompt_by_id):
        raise ValueError(f"prompts, rubrics, and {pool_label} must have the same prompt grid")

    grade_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    reused_grades = {
        (
            str(row["prompt_id"]),
            str(row["response_id"]),
            str(row["canonical_criterion_hash"]),
        ): row
        for row in reused_grade_rows
    }
    if len(reused_grades) != len(reused_grade_rows):
        raise ValueError("duplicate reused criterion grade identity")
    used_reused_grades: set[tuple[str, str, str]] = set()
    new_grade_count = 0
    seed_id = next(iter(seeds))
    policy_step = next(iter(policy_steps))
    if expected_policy_step is not None and policy_step != expected_policy_step:
        raise ValueError(
            f"{pool_label} policy step differs from checkpoint schedule: "
            f"expected={expected_policy_step}, actual={policy_step}"
        )
    for prompt_id in sorted(prompt_by_id):
        prompt = prompt_by_id[prompt_id]
        rubric_row = rubric_by_id[prompt_id]
        responses = sorted(pool_groups[prompt_id], key=lambda row: int(row["sample_index"]))
        if len(responses) != expected_response_count or {
            int(row["sample_index"]) for row in responses
        } != set(range(expected_response_count)):
            raise ValueError(
                f"{pool_label} must contain exactly {expected_response_count} "
                f"response slots for {prompt_id}"
            )

        r0 = tuple(r0_criterion_from_artifact(item) for item in prompt["r0"]["criteria"])
        if checkpoint == 0.0:
            extension: tuple[WeightedCriterion, ...] = ()
        else:
            if expected_policy_step is not None and rubric_row.get("checkpoint_id") != (
                f"step{expected_policy_step}"
            ):
                raise ValueError(
                    f"rubric checkpoint lineage differs from {pool_label}: {prompt_id}"
                )
            extension = tuple(
                criterion_from_artifact(item) for item in rubric_row.get("extension", ())
            )
        na_reason = None
        if checkpoint == 0.0:
            control_extension: tuple[WeightedCriterion, ...] | None = ()
        elif not include_control:
            control_extension = None
        else:
            raw_control = rubric_row.get("control_extension")
            control_match = rubric_row.get("control_match")
            if (
                raw_control is None
                or not isinstance(control_match, Mapping)
                or not control_match.get("eligible")
            ):
                control_extension = None
                na_reason = "ineligible_count_matched_control"
            else:
                control_extension = tuple(criterion_from_artifact(item) for item in raw_control)
                if len(control_extension) != len(extension):
                    control_extension = None
                    na_reason = "control_count_mismatch"
        variants = {
            "r0": WeightedRubric(prompt_id, r0, rubric_id="r0"),
            "current": WeightedRubric(prompt_id, r0 + extension, rubric_id="current"),
        }
        if control_extension is not None:
            variants["control"] = WeightedRubric(
                prompt_id, r0 + control_extension, rubric_id="control"
            )
        criteria_by_hash: dict[str, WeightedCriterion] = {}
        for rubric in variants.values():
            for criterion in rubric.criteria:
                prior = criteria_by_hash.setdefault(criterion.canonical_criterion_hash, criterion)
                if prior.text != criterion.text:
                    raise ValueError("criterion hash collision")

        prompt_text = json.dumps(prompt["messages"], ensure_ascii=False, sort_keys=True)
        items = [
            (
                prompt_id,
                str(response["response_id"]),
                str(response["response_text"]),
                content_hash,
                criterion.text,
            )
            for response in responses
            for content_hash, criterion in sorted(criteria_by_hash.items())
        ]
        missing_items = [
            item
            for item in items
            if (item[0], item[1], item[3]) not in reused_grades
        ]
        raw_grades = grader.score_many_full(
            missing_items, prompt_text_by_id={prompt_id: prompt_text}
        )
        if len(raw_grades) != len(missing_items):
            raise ValueError("grader returned the wrong number of criterion scores")
        raw_by_identity = {
            (grade.prompt_id, grade.response_id, grade.criterion_id): grade
            for grade in raw_grades
        }
        if len(raw_by_identity) != len(raw_grades):
            raise ValueError("grader returned duplicate criterion score identities")

        grades_by_response: dict[str, dict[str, HardGrade]] = {
            str(response["response_id"]): {} for response in responses
        }
        for expected in items:
            expected_prompt, expected_response, _, expected_hash, _ = expected
            identity = (expected_prompt, expected_response, expected_hash)
            reused_row = reused_grades.get(identity)
            if reused_row is not None:
                if (
                    str(reused_row.get("seed_id")) != seed_id
                    or float(reused_row.get("checkpoint")) != checkpoint
                    or int(reused_row.get("policy_step")) != policy_step
                    or str(reused_row.get("pool_family")) != pool_family
                ):
                    raise ValueError("reused grade lineage differs from grading shard")
                grade = _hard_grade_from_artifact(reused_row)
                used_reused_grades.add(identity)
            else:
                raw_grade = raw_by_identity.get(identity)
                if raw_grade is None:
                    raise ValueError("grader output identity drifted")
                grade = hard_grade_from_target_logprobs(
                    {
                        YES_TARGET: (raw_grade.yes_logprob,),
                        NO_TARGET: (raw_grade.no_logprob,),
                    },
                    retry_count=raw_grade.retry_count,
                )
                if raw_grade.parse_status != grade.parse_status:
                    raise ValueError(
                        "grader parse status disagrees with exact target logprobs"
                    )
                new_grade_count += 1
            grades_by_response[expected_response][expected_hash] = grade
            grade_rows.append(
                {
                    "schema_version": 1,
                    "seed_id": seed_id,
                    "checkpoint": checkpoint,
                    "policy_step": policy_step,
                    "pool_family": pool_family,
                    "prompt_id": prompt_id,
                    "response_id": expected_response,
                    "canonical_criterion_hash": expected_hash,
                    **asdict(grade),
                }
            )

        variant_scores: dict[str, list[tuple[int, int]]] = {name: [] for name in variants}
        response_grades: list[dict[str, HardGrade]] = []
        for response in responses:
            response_id = str(response["response_id"])
            grades = grades_by_response[response_id]
            response_grades.append(grades)
            scores = assemble_variant_scores(variants, grades)
            for variant, score in scores.items():
                pair = _score_tuple(score)
                variant_scores[variant].append(pair)
                score_rows.append(
                    {
                        "schema_version": 1,
                        "seed_id": seed_id,
                        "checkpoint": checkpoint,
                        "policy_step": policy_step,
                        "pool_family": pool_family,
                        "prompt_id": prompt_id,
                        "response_id": response_id,
                        "policy_training_regime": policy_training_regime,
                        "variant": variant,
                        "paper_variant": RUBRIC_VARIANT_LABELS[variant],
                        "score_numerator": pair[0],
                        "score_denominator": pair[1],
                        "score": score.value,
                    }
                )

        variant_summary = {
            name: _variant_summary(scores, epsilon=epsilon_spread, delta=delta_advantage)
            for name, scores in variant_scores.items()
        }
        variant_summary["current"]["pairwise_vs_r0"] = pairwise_metrics(
            variant_scores["current"], baseline_scores=variant_scores["r0"]
        )
        variant_summary["current"]["ranking_vs_r0"] = ranking_agreement(
            variant_scores["r0"], variant_scores["current"]
        )
        criterion_groups = {
            "r0": r0,
            "extension": extension,
            "control_extension": control_extension or (),
        }
        effectiveness = {
            name: criterion_effectiveness_summary(
                {
                    criterion.canonical_criterion_hash: [
                        int(grades[criterion.canonical_criterion_hash].grade)
                        for grades in response_grades
                    ]
                    for criterion in criteria
                }
            )
            for name, criteria in criterion_groups.items()
        }
        summary_rows.append(
            {
                "schema_version": 1,
                "seed_id": seed_id,
                "checkpoint": checkpoint,
                "policy_step": policy_step,
                "pool_family": pool_family,
                "prompt_id": prompt_id,
                "response_count": expected_response_count,
                "policy_training_regime": policy_training_regime,
                "variant_aliases": dict(RUBRIC_VARIANT_LABELS),
                "analysis_status": "na" if na_reason else "valid",
                "na_reason": na_reason,
                "variants": variant_summary,
                "criterion_effectiveness": effectiveness,
                "online_criterion_count": len(extension),
                "control_criterion_count": (
                    len(control_extension) if control_extension is not None else None
                ),
            }
        )

    unused_reused_grades = set(reused_grades) - used_reused_grades
    if unused_reused_grades:
        raise ValueError("reused grade artifact contains identities outside the target rubric")

    output_dir.mkdir(parents=True, exist_ok=True)
    grades_path = output_dir / "criterion_grades.jsonl"
    scores_path = output_dir / "variant_scores.jsonl"
    summary_path = output_dir / "prompt_summary.jsonl"
    write_jsonl_atomic(grades_path, grade_rows)
    write_jsonl_atomic(scores_path, score_rows)
    write_jsonl_atomic(summary_path, summary_rows)
    sources = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in sorted((source_paths or {}).items())
    }
    seal = {
        "schema_version": 1,
        "artifact_type": "horizon_score_seal",
        "seed_id": seed_id,
        "checkpoint": checkpoint,
        "policy_step": policy_step,
        "pool_family": pool_family,
        "policy_training_regime": policy_training_regime,
        "variant_aliases": dict(RUBRIC_VARIANT_LABELS),
        "comparison_scope": "full" if include_control else "r0_current",
        "grader_model_revision": grader_model_revision,
        "tokenizer_revision": tokenizer_revision,
        "target_encoding_version": target_encoding_version,
        "config_hash": config_hash,
        "prompt_count": len(summary_rows),
        "response_count": len(pool_rows),
        "grade_count": len(grade_rows),
        "reused_grade_count": len(used_reused_grades),
        "new_grade_count": new_grade_count,
        "coverage": {
            "valid_prompts": sum(row["analysis_status"] == "valid" for row in summary_rows),
            "na_prompts": sum(row["analysis_status"] == "na" for row in summary_rows),
        },
        "sources": sources,
        "outputs": {
            "criterion_grades": sha256_file(grades_path),
            "variant_scores": sha256_file(scores_path),
            "prompt_summary": sha256_file(summary_path),
        },
    }
    grader_execution = getattr(grader, "artifact_metadata", None)
    if isinstance(grader_execution, Mapping):
        seal["grader_execution"] = dict(grader_execution)
    seal_path = output_dir / "score_seal.json"
    write_json_atomic(seal_path, seal)
    return {"output_dir": str(output_dir), "seal": str(seal_path), **seal}


def grade_horizon_pool_b(
    grader: HorizonGrader,
    *,
    prompts: Sequence[Mapping[str, Any]],
    pool_b_rows: Sequence[Mapping[str, Any]],
    rubric_rows: Sequence[Mapping[str, Any]],
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    source_paths: Mapping[str, Path] | None = None,
    reused_grade_rows: Sequence[Mapping[str, Any]] = (),
    include_control: bool = True,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
) -> dict[str, Any]:
    return grade_horizon_pool(
        grader,
        prompts=prompts,
        pool_rows=pool_b_rows,
        pool_family="pool_b",
        rubric_rows=rubric_rows,
        checkpoint=checkpoint,
        output_dir=output_dir,
        grader_model_revision=grader_model_revision,
        tokenizer_revision=tokenizer_revision,
        epsilon_spread=epsilon_spread,
        delta_advantage=delta_advantage,
        source_paths=source_paths,
        reused_grade_rows=reused_grade_rows,
        include_control=include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config_hash,
        policy_training_regime=policy_training_regime,
    )


def grade_horizon_pool_a(
    grader: HorizonGrader,
    *,
    prompts: Sequence[Mapping[str, Any]],
    pool_a_rows: Sequence[Mapping[str, Any]],
    rubric_rows: Sequence[Mapping[str, Any]],
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    source_paths: Mapping[str, Path] | None = None,
    reused_grade_rows: Sequence[Mapping[str, Any]] = (),
    include_control: bool = True,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
) -> dict[str, Any]:
    return grade_horizon_pool(
        grader,
        prompts=prompts,
        pool_rows=pool_a_rows,
        pool_family="pool_a",
        rubric_rows=rubric_rows,
        checkpoint=checkpoint,
        output_dir=output_dir,
        grader_model_revision=grader_model_revision,
        tokenizer_revision=tokenizer_revision,
        epsilon_spread=epsilon_spread,
        delta_advantage=delta_advantage,
        source_paths=source_paths,
        reused_grade_rows=reused_grade_rows,
        include_control=include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config_hash,
        policy_training_regime=policy_training_regime,
    )


def grade_horizon_pool_a_combined(
    grader: HorizonGrader,
    *,
    prompts: Sequence[Mapping[str, Any]],
    pool_a_combined_rows: Sequence[Mapping[str, Any]],
    rubric_rows: Sequence[Mapping[str, Any]],
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    source_paths: Mapping[str, Path] | None = None,
    reused_grade_rows: Sequence[Mapping[str, Any]] = (),
    include_control: bool = True,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
) -> dict[str, Any]:
    return grade_horizon_pool(
        grader,
        prompts=prompts,
        pool_rows=pool_a_combined_rows,
        pool_family="pool_a_combined",
        rubric_rows=rubric_rows,
        checkpoint=checkpoint,
        output_dir=output_dir,
        grader_model_revision=grader_model_revision,
        tokenizer_revision=tokenizer_revision,
        epsilon_spread=epsilon_spread,
        delta_advantage=delta_advantage,
        source_paths=source_paths,
        reused_grade_rows=reused_grade_rows,
        include_control=include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config_hash,
        policy_training_regime=policy_training_regime,
    )


def _load_reused_grade_rows(
    *,
    score_dir: Path,
    prompts_path: Path,
    pool_path: Path,
    pool_family: str,
    checkpoint: float,
    grader_model_revision: str,
    tokenizer_revision: str,
    expected_policy_step: int | None,
    config_hash: str | None,
) -> tuple[list[Mapping[str, Any]], Path, Path]:
    seal_path = score_dir / "score_seal.json"
    grades_path = score_dir / "criterion_grades.jsonl"
    seal = read_json(seal_path)
    expected_fields = {
        "artifact_type": "horizon_score_seal",
        "checkpoint": checkpoint,
        "pool_family": pool_family,
        "grader_model_revision": grader_model_revision,
        "tokenizer_revision": tokenizer_revision,
        "target_encoding_version": TARGET_ENCODING_VERSION,
        "config_hash": config_hash,
    }
    for field, expected in expected_fields.items():
        if seal.get(field) != expected:
            raise ValueError(f"reused score seal has incompatible {field}")
    if expected_policy_step is not None and int(seal.get("policy_step")) != expected_policy_step:
        raise ValueError("reused score seal has incompatible policy_step")
    sources = seal.get("sources", {})
    for name, path in (("prompts", prompts_path), (pool_family, pool_path)):
        source = sources.get(name)
        if not isinstance(source, Mapping) or source.get("sha256") != sha256_file(path):
            raise ValueError(f"reused score seal has incompatible {name} source")
    outputs = seal.get("outputs", {})
    if outputs.get("criterion_grades") != sha256_file(grades_path):
        raise ValueError("reused criterion grades fail score seal verification")
    return read_jsonl(grades_path), grades_path, seal_path


def grade_horizon_pool_b_from_files(
    grader: HorizonGrader,
    *,
    prompts_path: Path,
    pool_b_path: Path,
    rubrics_path: Path | None,
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
    reuse_score_dir: Path | None = None,
    include_control: bool = True,
) -> dict[str, Any]:
    prompts = read_jsonl(prompts_path)
    rubric_rows = (
        read_jsonl(rubrics_path)
        if rubrics_path is not None
        else [{"prompt_id": row["prompt_id"], "extension": []} for row in prompts]
    )
    source_paths = {"prompts": prompts_path, "pool_b": pool_b_path}
    if rubrics_path is not None:
        source_paths["rubrics"] = rubrics_path
    reused_grade_rows: Sequence[Mapping[str, Any]] = ()
    if reuse_score_dir is not None:
        reused_grade_rows, reused_grades_path, reused_seal_path = _load_reused_grade_rows(
            score_dir=reuse_score_dir,
            prompts_path=prompts_path,
            pool_path=pool_b_path,
            pool_family="pool_b",
            checkpoint=checkpoint,
            grader_model_revision=grader_model_revision,
            tokenizer_revision=tokenizer_revision,
            expected_policy_step=expected_policy_step,
            config_hash=config_hash,
        )
        source_paths["reused_criterion_grades"] = reused_grades_path
        source_paths["reused_score_seal"] = reused_seal_path
    return grade_horizon_pool_b(
        grader,
        prompts=prompts,
        pool_b_rows=read_jsonl(pool_b_path),
        rubric_rows=rubric_rows,
        checkpoint=checkpoint,
        output_dir=output_dir,
        grader_model_revision=grader_model_revision,
        tokenizer_revision=tokenizer_revision,
        epsilon_spread=epsilon_spread,
        delta_advantage=delta_advantage,
        source_paths=source_paths,
        reused_grade_rows=reused_grade_rows,
        include_control=include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config_hash,
        policy_training_regime=policy_training_regime,
    )


def grade_horizon_pool_a_combined_from_files(
    grader: HorizonGrader,
    *,
    prompts_path: Path,
    pool_a_combined_path: Path,
    rubrics_path: Path,
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
    reuse_score_dir: Path | None = None,
    include_control: bool = True,
) -> dict[str, Any]:
    prompts = read_jsonl(prompts_path)
    source_paths = {
        "prompts": prompts_path,
        "pool_a_combined": pool_a_combined_path,
        "rubrics": rubrics_path,
    }
    reused_grade_rows: Sequence[Mapping[str, Any]] = ()
    if reuse_score_dir is not None:
        reused_grade_rows, reused_grades_path, reused_seal_path = _load_reused_grade_rows(
            score_dir=reuse_score_dir,
            prompts_path=prompts_path,
            pool_path=pool_a_combined_path,
            pool_family="pool_a_combined",
            checkpoint=checkpoint,
            grader_model_revision=grader_model_revision,
            tokenizer_revision=tokenizer_revision,
            expected_policy_step=expected_policy_step,
            config_hash=config_hash,
        )
        source_paths["reused_criterion_grades"] = reused_grades_path
        source_paths["reused_score_seal"] = reused_seal_path
    return grade_horizon_pool_a_combined(
        grader,
        prompts=prompts,
        pool_a_combined_rows=read_jsonl(pool_a_combined_path),
        rubric_rows=read_jsonl(rubrics_path),
        checkpoint=checkpoint,
        output_dir=output_dir,
        grader_model_revision=grader_model_revision,
        tokenizer_revision=tokenizer_revision,
        epsilon_spread=epsilon_spread,
        delta_advantage=delta_advantage,
        source_paths=source_paths,
        reused_grade_rows=reused_grade_rows,
        include_control=include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config_hash,
        policy_training_regime=policy_training_regime,
    )


def grade_horizon_pool_a_from_files(
    grader: HorizonGrader,
    *,
    prompts_path: Path,
    pool_a_path: Path,
    rubrics_path: Path,
    checkpoint: float,
    output_dir: Path,
    grader_model_revision: str,
    tokenizer_revision: str,
    epsilon_spread: float,
    delta_advantage: float,
    expected_policy_step: int | None = None,
    config_hash: str | None = None,
    policy_training_regime: str = STATIC_EXPERIMENT_ARM,
    reuse_score_dir: Path | None = None,
    include_control: bool = True,
) -> dict[str, Any]:
    prompts = read_jsonl(prompts_path)
    source_paths = {
        "prompts": prompts_path,
        "pool_a": pool_a_path,
        "rubrics": rubrics_path,
    }
    reused_grade_rows: Sequence[Mapping[str, Any]] = ()
    if reuse_score_dir is not None:
        reused_grade_rows, reused_grades_path, reused_seal_path = _load_reused_grade_rows(
            score_dir=reuse_score_dir,
            prompts_path=prompts_path,
            pool_path=pool_a_path,
            pool_family="pool_a",
            checkpoint=checkpoint,
            grader_model_revision=grader_model_revision,
            tokenizer_revision=tokenizer_revision,
            expected_policy_step=expected_policy_step,
            config_hash=config_hash,
        )
        source_paths["reused_criterion_grades"] = reused_grades_path
        source_paths["reused_score_seal"] = reused_seal_path
    return grade_horizon_pool_a(
        grader,
        prompts=prompts,
        pool_a_rows=read_jsonl(pool_a_path),
        rubric_rows=read_jsonl(rubrics_path),
        checkpoint=checkpoint,
        output_dir=output_dir,
        grader_model_revision=grader_model_revision,
        tokenizer_revision=tokenizer_revision,
        epsilon_spread=epsilon_spread,
        delta_advantage=delta_advantage,
        source_paths=source_paths,
        reused_grade_rows=reused_grade_rows,
        include_control=include_control,
        expected_policy_step=expected_policy_step,
        config_hash=config_hash,
        policy_training_regime=policy_training_regime,
    )
