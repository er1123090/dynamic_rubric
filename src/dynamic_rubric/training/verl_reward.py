"""veRL custom reward hook for the immutable static ``R_0`` rubric.

The module is loaded by path inside Ray reward workers, so provider construction is
lazy and process-local. Only the public static-rubric artifact and the frozen
Qwen proxy endpoint are accepted as inputs.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
import json
from typing import Any, Mapping

from dynamic_rubric.artifacts import read_jsonl
from dynamic_rubric.providers.vllm import VLLMCriterionGrader, VLLMIdentity
from dynamic_rubric.rubrics.static import Criterion, StaticRubric
from dynamic_rubric.seeds import SeedFamily, derive_seed, response_id
from dynamic_rubric.training.static_reward import score_response


class VerlStaticRewardError(RuntimeError):
    """Raised when the live static reward contract is incomplete or corrupted."""


_STATE_LOCK = threading.Lock()
_STATE: tuple[VLLMCriterionGrader, dict[str, Mapping[str, Any]]] | None = None


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise VerlStaticRewardError(f"{name} is required by the veRL static reward hook")
    return value


def _grader_base_url() -> str:
    """Pin each long-lived reward worker to one configured judge endpoint."""

    raw_urls = os.environ.get("DYNAMIC_RUBRIC_VLLM_URLS", "")
    urls = [value.strip() for value in raw_urls.split(",") if value.strip()]
    if not urls:
        urls = [_required_environment("DYNAMIC_RUBRIC_VLLM_URL")]
    return urls[os.getpid() % len(urls)]


def _load_static_rubrics(path: Path) -> dict[str, Mapping[str, Any]]:
    if not path.is_file():
        raise VerlStaticRewardError(f"static rubric artifact is missing: {path}")
    rubrics: dict[str, Mapping[str, Any]] = {}
    for row in read_jsonl(path):
        prompt_id = str(row["prompt_id"])
        if isinstance(row.get("r0"), Mapping):
            criteria = row["r0"].get("criteria")
            if not isinstance(criteria, list) or not criteria:
                raise VerlStaticRewardError(f"RaR R0 rubric is empty: {prompt_id}")
            for item in criteria:
                if (
                    not isinstance(item, Mapping)
                    or not item.get("criterion_id")
                    or not item.get("criterion")
                    or int(item.get("weight_units", 0)) not in {3, 7, 9, 10}
                ):
                    raise VerlStaticRewardError(f"invalid RaR R0 criterion: {prompt_id}")
        else:
            rubric = StaticRubric(
                prompt_id=prompt_id,
                criteria=tuple(Criterion(**item) for item in row["criteria"]),
            )
            if row.get("content_hash") != rubric.content_hash:
                raise VerlStaticRewardError(f"static rubric content hash mismatch: {prompt_id}")
        if prompt_id in rubrics:
            raise VerlStaticRewardError(f"duplicate static rubric: {prompt_id}")
        rubrics[prompt_id] = row
    if not rubrics:
        raise VerlStaticRewardError("static rubric artifact is empty")
    return rubrics


def _state() -> tuple[VLLMCriterionGrader, dict[str, Mapping[str, Any]]]:
    global _STATE
    if _STATE is not None:
        return _STATE
    with _STATE_LOCK:
        if _STATE is None:
            grader = VLLMCriterionGrader(
                _grader_base_url(),
                VLLMIdentity(
                    served_model=_required_environment("DYNAMIC_RUBRIC_GRADER_MODEL"),
                    model_revision=_required_environment("DYNAMIC_RUBRIC_GRADER_REVISION"),
                    tokenizer_revision=_required_environment(
                        "DYNAMIC_RUBRIC_GRADER_TOKENIZER_REVISION"
                    ),
                    thinking=False,
                ),
                timeout_seconds=float(
                    os.environ.get("DYNAMIC_RUBRIC_GRADER_TIMEOUT_SECONDS", "900")
                ),
            )
            rubrics = _load_static_rubrics(
                Path(_required_environment("DYNAMIC_RUBRIC_STATIC_RUBRIC_PATH"))
            )
            _STATE = (grader, rubrics)
    return _STATE


def _score_rar_response(
    grader: VLLMCriterionGrader,
    prompt_id: str,
    response_id_value: str,
    response_text: str,
    rubric_row: Mapping[str, Any],
) -> tuple[float, list[dict[str, Any]], int, int]:
    r0 = rubric_row.get("r0")
    criteria = r0.get("criteria") if isinstance(r0, Mapping) else None
    if not isinstance(criteria, list) or not criteria:
        raise VerlStaticRewardError("RaR training rubric must contain R0 criteria")
    values = [
        (
            prompt_id,
            response_id_value,
            response_text,
            str(item["criterion_id"]),
            str(item["criterion"]),
        )
        for item in criteria
    ]
    prompt_context = json.dumps(rubric_row.get("messages", []), ensure_ascii=False)
    scores = grader.score_many_full(values, prompt_text_by_id={prompt_id: prompt_context})
    numerator = 0
    denominator = 0
    trace: list[dict[str, Any]] = []
    for criterion, score in zip(criteria, scores):
        if score.parse_status not in {"ok", "ambiguous_target_tie"}:
            raise VerlStaticRewardError("criterion grader parse failure")
        probability = float(score.probability_present)
        # bf16 can quantize distinct YES/NO logits to an exact tie. Preserve the
        # ambiguity in the trace and conservatively map training ties to 0.
        hard_grade = int(probability > 0.5)
        weight_units = int(criterion["weight_units"])
        numerator += weight_units * hard_grade
        denominator += weight_units
        trace.append(
            {
                "criterion_id": score.criterion_id,
                "probability_yes": probability,
                "probability_present": probability,
                "yes_logprob": score.yes_logprob,
                "no_logprob": score.no_logprob,
                "parse_status": score.parse_status,
                "retry_count": score.retry_count,
                "hard_grade": hard_grade,
                "weight_units": weight_units,
            }
        )
    return numerator / denominator, trace, numerator, denominator


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Mapping[str, Any] | None = None,
    **_: Any,
) -> dict[str, Any]:
    """Return the frozen static-R0 reward for the selected data-source contract."""

    data_source = str(data_source)
    if data_source not in {"dynamic_rubric_static_r0", "rar_static_r0"}:
        raise VerlStaticRewardError(f"unexpected reward data_source: {data_source}")
    expected_ground_truth = (
        "rar_static_r0_only" if data_source == "rar_static_r0" else "static_r0_only"
    )
    if ground_truth not in (None, expected_ground_truth):
        raise VerlStaticRewardError("training reward input must not contain hidden ground truth")
    if not isinstance(extra_info, Mapping) or not extra_info.get("prompt_id"):
        raise VerlStaticRewardError("reward input is missing extra_info.prompt_id")
    grader, rubrics = _state()
    prompt_id = str(extra_info["prompt_id"])
    if prompt_id not in rubrics:
        raise VerlStaticRewardError(f"no static rubric for prompt: {prompt_id}")
    required_seed_fields = ("run_id", "family", "policy_step", "sample_index", "logical_seed")
    missing = [field for field in required_seed_fields if field not in extra_info]
    if missing:
        raise VerlStaticRewardError(f"reward input is missing deterministic metadata: {missing}")
    run_id = str(extra_info["run_id"])
    family = SeedFamily(str(extra_info["family"]))
    policy_step = int(extra_info["policy_step"])
    sample_index = int(extra_info["sample_index"])
    logical_seed = int(extra_info["logical_seed"])
    expected_seed = derive_seed(run_id, family, prompt_id, policy_step, sample_index)
    if logical_seed != expected_seed:
        raise VerlStaticRewardError("reward input logical seed does not match canonical derivation")
    canonical_response_id = response_id(
        run_id, family, prompt_id, policy_step, sample_index
    )
    if data_source == "rar_static_r0":
        reward, criterion_scores, score_num, score_den = _score_rar_response(
            grader,
            prompt_id,
            canonical_response_id,
            solution_str,
            rubrics[prompt_id],
        )
    else:
        reward, criterion_scores = score_response(
            grader,
            prompt_id,
            canonical_response_id,
            solution_str,
            rubrics[prompt_id],
        )
        score_num = score_den = None
    result = {
        "score": reward,
        "static_reward": reward,
        "prompt_id": prompt_id,
        "response_id": canonical_response_id,
        "criterion_probabilities_json": json.dumps(
            [float(item["probability_yes"]) for item in criterion_scores],
            separators=(",", ":"),
        ),
        "criterion_count": len(criterion_scores),
        "family": family.value,
        "policy_step": policy_step,
        "sample_index": sample_index,
        "logical_seed": logical_seed,
    }
    if score_num is not None and score_den is not None:
        result.update(
            {
                "score_num": score_num,
                "score_den": score_den,
                "training_reward_mode": "hard_binary_weighted_rational_v1",
            }
        )
    return result


def _reset_state_for_tests() -> None:
    global _STATE
    with _STATE_LOCK:
        _STATE = None
