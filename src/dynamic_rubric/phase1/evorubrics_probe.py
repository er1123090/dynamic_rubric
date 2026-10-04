"""Offline fixed-probe generation and grading for real EvoRubrics checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

from ..artifacts import read_json, write_json_atomic
from ..hashing import canonical_json_bytes, sha256_file, sha256_json
from .config import load_phase1_config
from .evorubrics_audit import (
    CriterionJudgment,
    RubricGradeRecord,
    aggregate_evo_score_group,
    analyze_evo_grade_matrix,
)
from .evorubrics_observer import discover_checkpoint_pairs
from .provenance import load_probe_rows, response_id

RUBRIC_SEEDS = (11001, 11002, 11003, 11004)
POOL_B_COUNT = 16
JUDGE_ANSWERS_PER_CALL = 4


class EvoProbeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    prompt_id: str
    sample_index: int
    seed: int
    messages: tuple[Mapping[str, str], ...]


class GenerationBackend(Protocol):
    def generate(
        self,
        *,
        adapter_path: str,
        adapter_name: str,
        requests: Sequence[GenerationRequest],
        max_new_tokens: int,
        temperature: float,
    ) -> list[dict[str, Any]]: ...


class JudgeBackend(Protocol):
    def grade(
        self,
        *,
        question: str,
        answers: Sequence[str],
        criteria: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]: ...


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _upstream_path(repo_root: Path) -> Path:
    return repo_root / "environment/upstream/EvoRubrics/evorubric-main"


def _upstream_module(repo_root: Path, name: str):
    path = str(_upstream_path(repo_root))
    if path not in sys.path:
        sys.path.insert(0, path)
    return importlib.import_module(name)


def _seed(*parts: object) -> int:
    digest = hashlib.sha256("\0".join(map(str, parts)).encode()).digest()[:8]
    return int.from_bytes(digest, "big") % (2**31)


def _question(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list) or not messages:
        raise EvoProbeError(f"probe row {row.get('prompt_id')} has no messages")
    for message in reversed(messages):
        if message.get("role") == "user" and str(message.get("content", "")).strip():
            return str(message["content"])
    raise EvoProbeError(f"probe row {row.get('prompt_id')} has no user question")


def _policy_messages(row: Mapping[str, Any], repo_root: Path) -> tuple[Mapping[str, str], ...]:
    models = _upstream_module(repo_root, "models")
    messages = [{"role": "system", "content": models.POLICY_LLM_SYSTEM_PROMPT}]
    messages.extend(
        {"role": str(item["role"]), "content": str(item["content"])}
        for item in row["messages"]
        if item.get("role") != "system"
    )
    return tuple(messages)


def _rubric_messages(row: Mapping[str, Any], repo_root: Path) -> tuple[Mapping[str, str], ...]:
    models = _upstream_module(repo_root, "models")
    # RQ2 uses psi(q): probe answers are never supplied to the rubric generator.
    return (
        {"role": "system", "content": models.UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT},
        {"role": "user", "content": _question(row)},
    )


def parse_pairs(spec: str, available: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    allowed = {tuple(map(int, pair)) for pair in available}
    if spec.strip().lower() == "all":
        return tuple(sorted(allowed, key=lambda pair: (pair[1], pair[0])))
    requested = []
    for value in spec.split(","):
        try:
            evaluator, policy = (int(item) for item in value.strip().split(":"))
        except (TypeError, ValueError) as error:
            raise EvoProbeError(
                "pairs must be 'all' or comma-separated evaluator:policy"
            ) from error
        requested.append((evaluator, policy))
    if not requested or len(set(requested)) != len(requested):
        raise EvoProbeError("checkpoint pairs must be non-empty and unique")
    missing = set(requested) - allowed
    if missing:
        raise EvoProbeError(f"checkpoint pairs are not committed: {sorted(missing)}")
    selected_steps = {step for pair in requested for step in pair}
    expanded = set(requested)
    expanded.update((step, step) for step in selected_steps)
    return tuple(sorted(expanded, key=lambda pair: (pair[1], pair[0])))


class TransformersPeftBackend:
    """Local-files-only Transformers/PEFT generator; set CUDA_VISIBLE_DEVICES=1."""

    def __init__(self, model_path: str, *, device: str = "cuda:0") -> None:
        self.model_path = model_path
        self.device = device

    def generate(
        self,
        *,
        adapter_path: str,
        adapter_name: str,
        requests: Sequence[GenerationRequest],
        max_new_tokens: int,
        temperature: float,
    ) -> list[dict[str, Any]]:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True, local_files_only=True
        )
        base = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        model = PeftModel.from_pretrained(
            base, adapter_path, adapter_name=adapter_name, is_trainable=False
        )
        model.eval()
        output = []
        with torch.inference_mode():
            for request in requests:
                prompt = tokenizer.apply_chat_template(
                    list(request.messages),
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                torch.manual_seed(request.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(request.seed)
                generated = model.generate(
                    **encoded,
                    do_sample=True,
                    temperature=temperature,
                    max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )[0]
                prompt_length = int(encoded["input_ids"].shape[1])
                response_tokens = generated[prompt_length:]
                output.append(
                    {
                        "prompt_id": request.prompt_id,
                        "sample_index": request.sample_index,
                        "seed": request.seed,
                        "text": tokenizer.decode(response_tokens, skip_special_tokens=True),
                        "prompt_token_ids": encoded["input_ids"][0].tolist(),
                        "response_token_ids": response_tokens.tolist(),
                    }
                )
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output

    def score_logprobs(
        self,
        *,
        adapter_path: str,
        adapter_name: str,
        sequences: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """Teacher-force saved responses, excluding every prompt token from the sum."""
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM

        base = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
        ).to(self.device)
        model = PeftModel.from_pretrained(
            base, adapter_path, adapter_name=adapter_name, is_trainable=False
        )
        model.eval()
        output = []
        with torch.inference_mode():
            for sequence in sequences:
                prompt = [int(value) for value in sequence["prompt_token_ids"]]
                response = [int(value) for value in sequence["response_token_ids"]]
                if not prompt or not response:
                    raise EvoProbeError("KL sequences require non-empty prompt and response tokens")
                input_ids = torch.tensor([prompt + response], device=self.device)
                attention_mask = torch.ones_like(input_ids)
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                response_logits = logits[:, len(prompt) - 1 : -1, :].float()
                targets = input_ids[:, len(prompt) :]
                selected = torch.log_softmax(response_logits, dim=-1).gather(
                    -1, targets.unsqueeze(-1)
                )
                output.append(
                    {
                        "prompt_id": str(sequence["prompt_id"]),
                        "response_id": str(sequence["response_id"]),
                        "logprob_sum": float(selected.sum().item()),
                        "response_token_count": len(response),
                    }
                )
        del model, base
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return output


class UpstreamJudge:
    def __init__(self, repo_root: Path, *, base_url: str, model: str) -> None:
        evaluator = _upstream_module(repo_root, "llm_evaluator")
        self.client = evaluator.LLMEvaluator(
            evaluator_type="deepseek",
            api_key=os.getenv("PHASE1_GPT_OSS_API_KEY", "EMPTY"),
            base_url=base_url,
            model=model,
        )

    def grade(
        self,
        *,
        question: str,
        answers: Sequence[str],
        criteria: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        descriptions = [str(item["criterion"]) for item in criteria]
        points = [float(item["weight"]) for item in criteria]
        results = []
        for start in range(0, len(answers), JUDGE_ANSWERS_PER_CALL):
            answer_chunk = list(answers[start : start + JUDGE_ANSWERS_PER_CALL])
            graded = self.client.batch_evaluate_multiple_answers(
                answers=answer_chunk,
                rubric_descriptions=descriptions,
                rubric_points=points,
                question=question,
            )
            if len(graded) != len(answer_chunk):
                raise EvoProbeError("judge returned an incomplete four-answer chunk")
            results.extend(graded)
        return results


def _checkpoint_map(discovery: Mapping[str, Any]) -> dict[int, Mapping[str, Any]]:
    return {int(row["global_step"]): row for row in discovery["checkpoints"]}


def _load_cached(path: Path, *, expected: int) -> list[dict[str, Any]]:
    if not path.is_file():
        raise EvoProbeError(f"missing cached artifact: {path}")
    rows = read_json(path)
    if not isinstance(rows, list) or len(rows) != expected:
        raise EvoProbeError(f"invalid cached artifact cardinality: {path}")
    return rows


def generate_probe_artifacts(
    *,
    run_root: Path,
    repo_root: Path,
    checkpoint_rows: Mapping[int, Mapping[str, Any]],
    pairs: Sequence[tuple[int, int]],
    probe_rows: Sequence[Mapping[str, Any]],
    backend: GenerationBackend,
    experiment_seed: int,
    temperature: float = 0.7,
) -> dict[str, Any]:
    probe_root = run_root / "audit/fixed_probe"
    policies = sorted({policy for _, policy in pairs})
    evaluators = sorted({evaluator for evaluator, _ in pairs})
    for step in policies:
        destination = probe_root / "responses" / f"theta_{step:06d}.json"
        if destination.exists():
            _load_cached(destination, expected=len(probe_rows) * POOL_B_COUNT)
            continue
        requests = [
            GenerationRequest(
                prompt_id=str(row["prompt_id"]),
                sample_index=index,
                seed=_seed("evorubrics-probe-B", experiment_seed, row["prompt_id"], step, index),
                messages=_policy_messages(row, repo_root),
            )
            for row in probe_rows
            for index in range(POOL_B_COUNT)
        ]
        generated = backend.generate(
            adapter_path=str(checkpoint_rows[step]["policy"]["adapter_path"]),
            adapter_name="policy_llm",
            requests=requests,
            max_new_tokens=1024,
            temperature=temperature,
        )
        if len(generated) != len(requests):
            raise EvoProbeError("policy backend returned incomplete Pool-B generations")
        records = []
        for request, value in zip(requests, generated):
            if (value.get("prompt_id"), value.get("sample_index")) != (
                request.prompt_id,
                request.sample_index,
            ):
                raise EvoProbeError("policy backend changed request order or identity")
            records.append(
                {
                    **value,
                    "response_id": response_id(
                        domain="medicine",
                        method="evorubrics",
                        seed=experiment_seed,
                        prompt_id=request.prompt_id,
                        pool="probe_B",
                        policy_checkpoint=str(step),
                        sample_index=request.sample_index,
                    ),
                    "policy_step": step,
                    "pool": "probe_B",
                    "used_for_gradient": False,
                }
            )
        write_json_atomic(destination, records)

    for step in evaluators:
        destination = probe_root / "rubrics" / f"psi_{step:06d}.json"
        if destination.exists():
            _load_cached(destination, expected=len(probe_rows) * len(RUBRIC_SEEDS))
            continue
        requests = [
            GenerationRequest(
                prompt_id=str(row["prompt_id"]),
                sample_index=index,
                seed=seed,
                messages=_rubric_messages(row, repo_root),
            )
            for row in probe_rows
            for index, seed in enumerate(RUBRIC_SEEDS)
        ]
        generated = backend.generate(
            adapter_path=str(checkpoint_rows[step]["generator"]["adapter_path"]),
            adapter_name="rubrics_generator",
            requests=requests,
            max_new_tokens=1024,
            temperature=temperature,
        )
        parser = _upstream_module(repo_root, "reward_calculator").parse_rubrics
        records = []
        for request, value in zip(requests, generated):
            parsed = parser(str(value.get("text", "")))
            if not parsed:
                raise EvoProbeError(
                    f"rubric parse failed for psi={step}, prompt={request.prompt_id}, seed={request.seed}"
                )
            records.append(
                {
                    **value,
                    "rubric_id": f"{request.prompt_id}:psi-{step}:seed-{request.seed}",
                    "evaluator_step": step,
                    "criteria": [
                        {
                            "criterion_id": f"c{index}",
                            "criterion": item.description,
                            "weight": item.points,
                            "axis": item.axis,
                        }
                        for index, item in enumerate(parsed)
                    ],
                    "query_conditioned_only": True,
                    "probe_answers_supplied_to_generator": False,
                    "used_for_gradient": False,
                }
            )
        write_json_atomic(destination, records)
    return {"policies": policies, "evaluators": evaluators, "probe_prompt_count": len(probe_rows)}


def _judgments(
    result: Mapping[str, Any], criteria: Sequence[Mapping[str, Any]]
) -> tuple[CriterionJudgment, ...]:
    details = result.get("details", {})
    scores = details.get("rubric_scores")
    if scores is None:
        batch = details.get("batch_results")
        if isinstance(batch, list):
            scores = [
                {"rubric_index": item.get("index"), "criteria_met": item.get("criteria_met")}
                for item in batch
            ]
    if not isinstance(scores, list):
        raise EvoProbeError("judge receipt lacks per-criterion details")
    by_index: dict[int, Mapping[str, Any]] = {}
    for score in scores:
        index = score.get("rubric_index")
        if not isinstance(index, int) or index in by_index:
            raise EvoProbeError("judge receipt has invalid or duplicate rubric indices")
        if not isinstance(score.get("criteria_met"), bool):
            raise EvoProbeError("judge criterion result is not boolean")
        by_index[index] = score
    if set(by_index) != set(range(len(criteria))):
        raise EvoProbeError("judge receipt is missing criterion results")
    return tuple(
        CriterionJudgment(
            criterion_id=str(item["criterion_id"]),
            weight=float(item["weight"]),
            grade=bool(by_index[index]["criteria_met"]),
        )
        for index, item in enumerate(criteria)
    )


def score_probe_artifacts(
    *,
    run_root: Path,
    pairs: Sequence[tuple[int, int]],
    probe_rows: Sequence[Mapping[str, Any]],
    judge: JudgeBackend,
) -> list[RubricGradeRecord]:
    root = run_root / "audit/fixed_probe"
    by_prompt = {str(row["prompt_id"]): row for row in probe_rows}
    all_records: list[RubricGradeRecord] = []
    for evaluator_step, policy_step in pairs:
        destination = root / "grades" / f"psi_{evaluator_step:06d}_theta_{policy_step:06d}.json"
        if destination.exists():
            payload = read_json(destination)
            raw_records = payload.get("records", [])
            expected = len(probe_rows) * POOL_B_COUNT * len(RUBRIC_SEEDS)
            if len(raw_records) != expected:
                raise EvoProbeError(f"invalid cached grade cardinality: {destination}")
            all_records.extend(_record_from_json(row) for row in raw_records)
            continue
        responses = _load_cached(
            root / "responses" / f"theta_{policy_step:06d}.json",
            expected=len(probe_rows) * POOL_B_COUNT,
        )
        rubrics = _load_cached(
            root / "rubrics" / f"psi_{evaluator_step:06d}.json",
            expected=len(probe_rows) * len(RUBRIC_SEEDS),
        )
        responses_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        rubrics_by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in responses:
            responses_by_prompt[str(row["prompt_id"])].append(row)
        for row in rubrics:
            rubrics_by_prompt[str(row["prompt_id"])].append(row)
        records: list[RubricGradeRecord] = []
        raw_details = []
        for prompt_id, prompt_row in by_prompt.items():
            answers = sorted(
                responses_by_prompt[prompt_id], key=lambda row: int(row["sample_index"])
            )
            rubric_sets = sorted(
                rubrics_by_prompt[prompt_id], key=lambda row: int(row["sample_index"])
            )
            if len(answers) != POOL_B_COUNT or len(rubric_sets) != len(RUBRIC_SEEDS):
                raise EvoProbeError(f"incomplete response/rubric cache for prompt {prompt_id}")
            for rubric in rubric_sets:
                graded = judge.grade(
                    question=_question(prompt_row),
                    answers=[str(row["text"]) for row in answers],
                    criteria=rubric["criteria"],
                )
                if len(graded) != POOL_B_COUNT:
                    raise EvoProbeError("judge returned incomplete answer results")
                raw_details.append(
                    {
                        "prompt_id": prompt_id,
                        "rubric_id": rubric["rubric_id"],
                        "response_ids": [str(answer["response_id"]) for answer in answers],
                        "criteria": rubric["criteria"],
                        "results": graded,
                    }
                )
                for answer, result in zip(answers, graded):
                    records.append(
                        RubricGradeRecord(
                            policy_step=policy_step,
                            evaluator_step=evaluator_step,
                            policy_checkpoint=f"theta-{policy_step}",
                            evaluator_checkpoint=f"psi-{evaluator_step}",
                            prompt_id=prompt_id,
                            response_id=str(answer["response_id"]),
                            rubric_id=str(rubric["rubric_id"]),
                            judgments=_judgments(result, rubric["criteria"]),
                        )
                    )
        write_json_atomic(
            destination,
            {
                "schema_version": 1,
                "records": [asdict(row) for row in records],
                "raw_grading_details": raw_details,
            },
        )
        all_records.extend(records)
    return all_records


def _record_from_json(row: Mapping[str, Any]) -> RubricGradeRecord:
    raw_judgments = row.get("judgments")
    if not isinstance(raw_judgments, list) or not raw_judgments:
        raise EvoProbeError("grade record lacks criterion judgments")
    judgments = []
    for item in raw_judgments:
        if not isinstance(item, Mapping) or not isinstance(item.get("grade"), bool):
            raise EvoProbeError("grade record criterion result must be boolean")
        judgments.append(
            CriterionJudgment(
                criterion_id=str(item.get("criterion_id", "")),
                weight=float(item["weight"]),
                grade=item["grade"],
            )
        )
    parse_ok = row.get("parse_ok", True)
    if not isinstance(parse_ok, bool):
        raise EvoProbeError("grade record parse_ok must be boolean")
    return RubricGradeRecord(
        policy_step=int(row["policy_step"]),
        evaluator_step=int(row["evaluator_step"]),
        policy_checkpoint=str(row["policy_checkpoint"]),
        evaluator_checkpoint=str(row["evaluator_checkpoint"]),
        prompt_id=str(row["prompt_id"]),
        response_id=str(row["response_id"]),
        rubric_id=str(row["rubric_id"]),
        judgments=tuple(judgments),
        parse_ok=parse_ok,
    )


def analyze_cached_grades(
    *,
    run_root: Path,
    pairs: Sequence[tuple[int, int]],
    probe_rows: Sequence[Mapping[str, Any]],
    records: Sequence[RubricGradeRecord],
    epsilon_z: float,
    epsilon_t: float,
) -> dict[str, Any]:
    root = run_root / "audit/fixed_probe"
    response_ids = {}
    for policy_step in {policy for _, policy in pairs}:
        rows = _load_cached(
            root / "responses" / f"theta_{policy_step:06d}.json", expected=len(probe_rows) * 16
        )
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            grouped[str(row["prompt_id"])].append(row)
        for prompt_id, values in grouped.items():
            response_ids[(policy_step, prompt_id)] = tuple(
                str(row["response_id"])
                for row in sorted(values, key=lambda row: int(row["sample_index"]))
            )
    rubric_ids = {}
    for evaluator_step in {evaluator for evaluator, _ in pairs}:
        rows = _load_cached(
            root / "rubrics" / f"psi_{evaluator_step:06d}.json", expected=len(probe_rows) * 4
        )
        grouped = defaultdict(list)
        for row in rows:
            grouped[str(row["prompt_id"])].append(row)
        for prompt_id, values in grouped.items():
            rubric_ids[(evaluator_step, prompt_id)] = tuple(
                str(row["rubric_id"])
                for row in sorted(values, key=lambda row: int(row["sample_index"]))
            )
    return analyze_evo_grade_matrix(
        records,
        expected_pairs=pairs,
        expected_prompt_ids=[str(row["prompt_id"]) for row in probe_rows],
        response_ids_by_policy_prompt=response_ids,
        rubric_ids_by_evaluator_prompt=rubric_ids,
        epsilon_z=epsilon_z,
        epsilon_t=epsilon_t,
    )


def sampled_kl_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate same-response sampled KL receipts by token and then by prompt."""
    if not rows:
        raise EvoProbeError("sampled KL requires response receipts")
    by_prompt: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    total_delta = total_tokens = 0.0
    for row in rows:
        tokens = int(row["response_token_count"])
        current = float(row["current_logprob_sum"])
        stale = float(row["stale_logprob_sum"])
        if tokens <= 0 or not all(map(math.isfinite, (current, stale))):
            raise EvoProbeError("sampled KL receipt has invalid log probabilities or token count")
        by_prompt[str(row["prompt_id"])].append(row)
        total_delta += current - stale
        total_tokens += tokens
    prompt_values = []
    for values in by_prompt.values():
        delta = sum(
            float(row["current_logprob_sum"]) - float(row["stale_logprob_sum"]) for row in values
        )
        tokens = sum(int(row["response_token_count"]) for row in values)
        prompt_values.append(delta / tokens)
    return {
        "token_weighted_sampled_kl": total_delta / total_tokens,
        "prompt_balanced_sampled_kl_mean": sum(prompt_values) / len(prompt_values),
        "prompt_count": len(prompt_values),
        "response_count": len(rows),
        "response_token_count": int(total_tokens),
        "same_response_tokens": True,
        "prompt_tokens_excluded": True,
    }


def compute_policy_kl_artifacts(
    *,
    run_root: Path,
    pairs: Sequence[tuple[int, int]],
    checkpoint_rows: Mapping[int, Mapping[str, Any]],
    probe_prompt_count: int,
    backend: TransformersPeftBackend,
) -> list[dict[str, Any]]:
    """Score theta_t responses under theta_t and theta_tau on identical tokens."""
    root = run_root / "audit/fixed_probe"
    summaries = []
    for stale_step, current_step in pairs:
        destination = root / "policy_kl" / f"theta_{stale_step:06d}_to_{current_step:06d}.json"
        if destination.exists():
            summaries.append(read_json(destination))
            continue
        sequences = _load_cached(
            root / "responses" / f"theta_{current_step:06d}.json",
            expected=probe_prompt_count * POOL_B_COUNT,
        )
        if stale_step == current_step:
            receipts = [
                {
                    "prompt_id": row["prompt_id"],
                    "response_id": row["response_id"],
                    "current_logprob_sum": 0.0,
                    "stale_logprob_sum": 0.0,
                    "response_token_count": len(row["response_token_ids"]),
                }
                for row in sequences
            ]
            summary = sampled_kl_summary(receipts)
            summary["diagonal_shortcut"] = True
        else:
            current = backend.score_logprobs(
                adapter_path=str(checkpoint_rows[current_step]["policy"]["adapter_path"]),
                adapter_name="policy_llm",
                sequences=sequences,
            )
            stale = backend.score_logprobs(
                adapter_path=str(checkpoint_rows[stale_step]["policy"]["adapter_path"]),
                adapter_name="policy_llm",
                sequences=sequences,
            )
            current_by_id = {str(row["response_id"]): row for row in current}
            stale_by_id = {str(row["response_id"]): row for row in stale}
            expected_ids = {str(row["response_id"]) for row in sequences}
            if set(current_by_id) != expected_ids or set(stale_by_id) != expected_ids:
                raise EvoProbeError("KL backend changed or omitted response identities")
            receipts = []
            for sequence in sequences:
                response_key = str(sequence["response_id"])
                current_row = current_by_id[response_key]
                stale_row = stale_by_id[response_key]
                if current_row["response_token_count"] != stale_row["response_token_count"]:
                    raise EvoProbeError("KL backends did not score identical response tokens")
                receipts.append(
                    {
                        "prompt_id": sequence["prompt_id"],
                        "response_id": response_key,
                        "current_logprob_sum": current_row["logprob_sum"],
                        "stale_logprob_sum": stale_row["logprob_sum"],
                        "response_token_count": current_row["response_token_count"],
                    }
                )
            summary = sampled_kl_summary(receipts)
            summary["diagonal_shortcut"] = False
        payload = {
            "schema_version": 1,
            "stale_policy_step": stale_step,
            "current_policy_step": current_step,
            "response_policy_step": current_step,
            "summary": summary,
            "receipts": receipts,
        }
        write_json_atomic(destination, payload)
        summaries.append(payload)
    return summaries


def _tensor_digest(tensor: Any) -> str:
    raw = tensor.detach().cpu().contiguous().view(-1).view(__import__("torch").uint8)
    return hashlib.sha256(raw.numpy().tobytes()).hexdigest()


def _validate_advantage_proof(run_root: Path, proof: Mapping[str, Any], *, adapter: str) -> None:
    if (proof.get("global_step"), proof.get("input_step"), proof.get("adapter")) != (1, 0, adapter):
        raise EvoProbeError(f"advantage proof has wrong step or adapter identity for {adapter}")
    if not proof.get("actual_training"):
        raise EvoProbeError(f"advantage proof is not marked as actual training for {adapter}")
    uid = proof.get("uid")
    if not isinstance(uid, list) or not uid or proof.get("uid_count") != len(uid):
        raise EvoProbeError(f"advantage proof has invalid uid inventory for {adapter}")
    artifact = proof.get("artifact")
    if not isinstance(artifact, Mapping) or artifact.get("format") != "safetensors":
        raise EvoProbeError(f"advantage proof lacks safetensors artifact for {adapter}")
    relative = Path(str(artifact.get("path", "")))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix != ".safetensors":
        raise EvoProbeError(
            f"advantage artifact path must be safe and run-root-relative for {adapter}"
        )
    binary = (run_root / relative).resolve()
    try:
        binary.relative_to(run_root.resolve())
    except ValueError as error:
        raise EvoProbeError(f"advantage artifact escapes run root for {adapter}") from error
    if not binary.is_file():
        raise EvoProbeError(f"advantage safetensors artifact is missing for {adapter}")
    if artifact.get("bytes") != binary.stat().st_size or artifact.get("sha256") != sha256_file(
        binary
    ):
        raise EvoProbeError(f"advantage artifact digest or size mismatch for {adapter}")
    tensors = proof.get("tensors")
    tensor_keys = proof.get("tensor_keys")
    required = {"responses", "response_mask", "advantages", "old_log_probs", "ref_log_prob"}
    if (
        not isinstance(tensors, Mapping)
        or tensor_keys != sorted(tensors)
        or not required.issubset(tensors)
    ):
        raise EvoProbeError(f"advantage tensor manifest is incomplete for {adapter}")
    from safetensors import safe_open

    with safe_open(str(binary), framework="pt", device="cpu") as handle:
        if sorted(handle.keys()) != tensor_keys:
            raise EvoProbeError(f"advantage binary keys do not match manifest for {adapter}")
        for key in tensor_keys:
            tensor = handle.get_tensor(key)
            metadata = tensors[key]
            if (
                metadata.get("shape") != list(tensor.shape)
                or metadata.get("dtype") != str(tensor.dtype)
                or metadata.get("sha256") != _tensor_digest(tensor)
            ):
                raise EvoProbeError(f"advantage tensor metadata mismatch for {adapter}:{key}")


def _validate_grade_proof(grades: Mapping[str, Any], *, prompt_count: int) -> None:
    raw_records = grades.get("records")
    raw_details = grades.get("raw_grading_details")
    expected_records = prompt_count * POOL_B_COUNT * len(RUBRIC_SEEDS)
    if not isinstance(raw_records, list) or len(raw_records) != expected_records:
        raise EvoProbeError("stale 0:1 grade proof does not contain the complete 16x4 grid")
    if not isinstance(raw_details, list) or len(raw_details) != prompt_count * len(RUBRIC_SEEDS):
        raise EvoProbeError("stale 0:1 grade proof lacks raw judge receipts")
    by_prompt: dict[str, list[RubricGradeRecord]] = defaultdict(list)
    for raw in raw_records:
        record = _record_from_json(raw)
        if record.policy_step != 1 or record.evaluator_step != 0 or not record.parse_ok:
            raise EvoProbeError(
                "stale grade receipt has wrong checkpoint identity or parse failure"
            )
        by_prompt[record.prompt_id].append(record)
    if len(by_prompt) != prompt_count:
        raise EvoProbeError("stale grade receipt prompt inventory is incomplete")
    expected_raw_keys = set()
    record_by_key = {}
    for prompt_id, records in by_prompt.items():
        response_ids = tuple(sorted({record.response_id for record in records}))
        rubric_ids = tuple(sorted({record.rubric_id for record in records}))
        aggregate_evo_score_group(
            records,
            expected_response_ids=response_ids,
            expected_rubric_ids=rubric_ids,
        )
        expected_raw_keys.update((prompt_id, rubric_id) for rubric_id in rubric_ids)
        record_by_key.update(
            ((record.prompt_id, record.rubric_id, record.response_id), record) for record in records
        )
    raw_keys = set()
    for detail in raw_details:
        if not isinstance(detail, Mapping):
            raise EvoProbeError("raw judge receipt has invalid structure")
        results = detail.get("results")
        response_ids = detail.get("response_ids")
        criteria = detail.get("criteria")
        if (
            not isinstance(results, list)
            or not isinstance(response_ids, list)
            or not isinstance(criteria, list)
            or not criteria
            or len(results) != POOL_B_COUNT
            or len(response_ids) != POOL_B_COUNT
            or len(set(map(str, response_ids))) != POOL_B_COUNT
        ):
            raise EvoProbeError("raw judge receipt omits Pool-B or criterion evidence")
        prompt_id = str(detail.get("prompt_id", ""))
        rubric_id = str(detail.get("rubric_id", ""))
        raw_keys.add((prompt_id, rubric_id))
        for raw_response_id, result in zip(response_ids, results):
            if not isinstance(result, Mapping):
                raise EvoProbeError("raw judge result must be an object")
            recorded = record_by_key.get((prompt_id, rubric_id, str(raw_response_id)))
            if recorded is None or _judgments(result, criteria) != recorded.judgments:
                raise EvoProbeError("raw judge criteria do not match normalized grade records")
    if raw_keys != expected_raw_keys:
        raise EvoProbeError("raw judge receipt identity does not match criterion records")


def _validate_kl_proof(kl: Mapping[str, Any], *, prompt_count: int) -> float:
    if (
        kl.get("stale_policy_step"),
        kl.get("current_policy_step"),
        kl.get("response_policy_step"),
    ) != (0, 1, 1):
        raise EvoProbeError("policy KL proof has wrong checkpoint identity")
    receipts = kl.get("receipts")
    if not isinstance(receipts, list) or len(receipts) != prompt_count * POOL_B_COUNT:
        raise EvoProbeError("policy KL proof has incomplete same-response receipts")
    if len({str(row.get("response_id", "")) for row in receipts}) != len(receipts):
        raise EvoProbeError("policy KL response identities are empty or duplicated")
    recomputed = sampled_kl_summary(receipts)
    summary = kl.get("summary", {})
    if not summary.get("same_response_tokens") or not summary.get("prompt_tokens_excluded"):
        raise EvoProbeError(
            "policy KL proof does not certify same response tokens and prompt masking"
        )
    value = summary.get("prompt_balanced_sampled_kl_mean")
    if value is None or not math.isfinite(float(value)):
        raise EvoProbeError("stale 0:1 policy KL must be finite")
    if not math.isclose(float(value), recomputed["prompt_balanced_sampled_kl_mean"], abs_tol=1e-12):
        raise EvoProbeError("policy KL summary does not match its receipts")
    return float(value)


def validate_live_smoke(run_root: Path) -> dict[str, Any]:
    """Write smoke_complete.json only after every real one-step proof validates."""

    def require_json(relative: str) -> Mapping[str, Any]:
        path = run_root / relative
        if not path.is_file():
            raise EvoProbeError(f"live smoke proof is missing: {path}")
        value = read_json(path)
        if not isinstance(value, Mapping):
            raise EvoProbeError(f"live smoke proof is not an object: {path}")
        return value

    launch = require_json("launch_spec.json")
    config_path = run_root / "config.resolved.json"
    if not config_path.is_file() or launch.get("phase1_config_sha256") != sha256_json(
        read_json(config_path)
    ):
        raise EvoProbeError("launch_spec phase1_config_sha256 does not match config.resolved.json")
    provenance_path = run_root / "run_provenance.json"
    provenance = require_json("run_provenance.json")
    semantic_identity = provenance.get("semantic_identity")
    semantic_hash = provenance.get("semantic_identity_sha256")
    expected_semantic_hash = (
        hashlib.sha256(canonical_json_bytes(semantic_identity)).hexdigest()
        if isinstance(semantic_identity, Mapping)
        else None
    )
    if not semantic_hash or semantic_hash != expected_semantic_hash:
        raise EvoProbeError("run provenance semantic identity hash is invalid")
    if provenance.get("actual_training") is not True:
        raise EvoProbeError("run provenance does not prove actual training")
    scope = provenance.get("scope", {})
    if (
        scope.get("mode") != launch.get("mode")
        or scope.get("upstream_config_sha256") != launch.get("upstream_config_sha256")
        or scope.get("expected_steps") != launch.get("expected_steps")
        or scope.get("expected_prompt_exposures") != launch.get("expected_prompt_exposures")
    ):
        raise EvoProbeError("run provenance scope does not match launch_spec")
    training = require_json("training_complete.json")
    if training.get("status") != "training_passed" or not training.get("actual_training"):
        raise EvoProbeError("training_complete must prove actual training_passed")

    discovery = discover_checkpoint_pairs(run_root)
    if not {0, 1}.issubset(set(discovery["steps"])):
        raise EvoProbeError("live smoke requires committed theta/psi checkpoint pairs 0 and 1")
    for adapter in ("policy_llm", "rubrics_generator"):
        proof = require_json(f"audit/advantages/step_000001_{adapter}.json")
        _validate_advantage_proof(run_root, proof, adapter=adapter)

    judge = require_json("judge-preflight.json")
    if judge.get("status") != "passed" or not judge.get("actual_remote_call"):
        raise EvoProbeError("judge preflight must prove an actual passed remote call")
    analysis = require_json("audit/fixed_probe/analysis.json")
    complete_rows = {
        (int(row["evaluator_step"]), int(row["policy_step"])): row
        for row in analysis.get("cells", [])
        if row.get("status") == "complete"
    }
    if not {(0, 0), (0, 1), (1, 1)}.issubset(complete_rows):
        raise EvoProbeError("probe analysis requires complete diagonal and stale 0:1 cells")
    prompt_count = int(complete_rows[(0, 1)].get("prompt_count", 0))
    if prompt_count <= 0 or any(
        int(complete_rows[pair].get("prompt_count", 0)) != prompt_count
        for pair in ((0, 0), (0, 1), (1, 1))
    ):
        raise EvoProbeError("probe analysis cells have invalid or mismatched prompt counts")
    grades = require_json("audit/fixed_probe/grades/psi_000000_theta_000001.json")
    _validate_grade_proof(grades, prompt_count=prompt_count)
    kl = require_json("audit/fixed_probe/policy_kl/theta_000000_to_000001.json")
    kl_value = _validate_kl_proof(kl, prompt_count=prompt_count)
    resumed = require_json("resume_verified.json")
    if (
        resumed.get("status") != "passed"
        or resumed.get("actual_reload") is not True
        or resumed.get("completed_main_call") is not True
        or resumed.get("strict_optimizer_reload") is not True
        or resumed.get("checkpoint_step") != 1
        or resumed.get("final_checkpoint_step") != 1
        or resumed.get("semantic_identity_sha256") != semantic_hash
    ):
        raise EvoProbeError("resume_verified must prove a matching step-1 checkpoint reload")
    roles = resumed.get("roles")
    if not isinstance(roles, Mapping) or set(roles) != {"policy", "generator"}:
        raise EvoProbeError("resume_verified lacks both adapter reload roles")
    for role, evidence in roles.items():
        if not isinstance(evidence, Mapping):
            raise EvoProbeError(f"resume_verified role evidence is invalid: {role}")
        adapter_path = Path(str(evidence.get("adapter_path", ""))).resolve()
        optimizer_path = Path(str(evidence.get("optimizer_path", ""))).resolve()
        try:
            adapter_path.relative_to(run_root.resolve())
            optimizer_path.relative_to(run_root.resolve())
        except ValueError as error:
            raise EvoProbeError(f"resume_verified role path escapes run root: {role}") from error
        weights = adapter_path / "adapter_model.safetensors"
        adapter_config = adapter_path / "adapter_config.json"
        if (
            not weights.is_file()
            or not adapter_config.is_file()
            or not optimizer_path.is_file()
            or evidence.get("weights_sha256") != sha256_file(weights)
            or evidence.get("config_sha256") != sha256_file(adapter_config)
            or evidence.get("optimizer_sha256") != sha256_file(optimizer_path)
        ):
            raise EvoProbeError(f"resume_verified role artifacts do not match hashes: {role}")

    payload = {
        "schema_version": 1,
        "status": "passed",
        "actual_training": True,
        "actual_judge": True,
        "committed_checkpoint_steps": discovery["steps"],
        "validated_probe_pairs": [[0, 0], [0, 1], [1, 1]],
        "finite_policy_kl_0_to_1": float(kl_value),
        "resume_verified": True,
        "phase1_config_sha256": launch["phase1_config_sha256"],
        "semantic_identity_sha256": semantic_hash,
        "run_provenance_sha256": sha256_file(provenance_path),
    }
    write_json_atomic(run_root / "smoke_complete.json", payload, immutable=False)
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--pairs", default="all")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--model-path")
    parser.add_argument("--judge-base-url")
    parser.add_argument("--judge-model", default="openai/gpt-oss-120b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--cuda-visible-devices", default="1")
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--score", action="store_true")
    parser.add_argument("--kl", action="store_true", help="teacher-forced same-response policy KL")
    parser.add_argument("--validate-smoke", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    repo_root = _repo_root()
    config_path = args.config or repo_root / "configs/phase1/medicine_evorubrics.yaml"
    config = load_phase1_config(config_path)
    discovery = discover_checkpoint_pairs(args.run_root)
    pairs = parse_pairs(args.pairs, discovery["expected_cells"])
    checkpoint_rows = _checkpoint_map(discovery)
    probe_rows = load_probe_rows(config, repo_root=repo_root, limit=args.max_prompts)
    explicit_action = args.generate or args.score or args.kl or args.validate_smoke
    do_generate = args.generate or not explicit_action
    do_score = args.score or not explicit_action
    model_path = args.model_path or config.raw["models"]["policy"].get("local_snapshot")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", args.cuda_visible_devices)
    if do_generate:
        if not model_path or not Path(model_path).is_dir():
            raise EvoProbeError("a local --model-path is required; downloads are disabled")
        generate_probe_artifacts(
            run_root=args.run_root,
            repo_root=repo_root,
            checkpoint_rows=checkpoint_rows,
            pairs=pairs,
            probe_rows=probe_rows,
            backend=TransformersPeftBackend(model_path, device=args.device),
            experiment_seed=config.seed,
            temperature=float(config.raw["evorubrics"]["generation_temperature"]),
        )
    if do_score:
        base_url = args.judge_base_url or os.getenv("PHASE1_GPT_OSS_BASE_URL")
        if not base_url:
            raise EvoProbeError("--judge-base-url or PHASE1_GPT_OSS_BASE_URL is required")
        records = score_probe_artifacts(
            run_root=args.run_root,
            pairs=pairs,
            probe_rows=probe_rows,
            judge=UpstreamJudge(repo_root, base_url=base_url, model=args.judge_model),
        )
        report = analyze_cached_grades(
            run_root=args.run_root,
            pairs=pairs,
            probe_rows=probe_rows,
            records=records,
            epsilon_z=float(config.raw["analysis"]["epsilon_z"]),
            epsilon_t=float(config.raw["analysis"]["epsilon_t"]),
        )
        write_json_atomic(
            args.run_root / "audit/fixed_probe/analysis.json", report, immutable=False
        )
    if args.kl:
        if not model_path or not Path(model_path).is_dir():
            raise EvoProbeError("a local --model-path is required; downloads are disabled")
        compute_policy_kl_artifacts(
            run_root=args.run_root,
            pairs=pairs,
            checkpoint_rows=checkpoint_rows,
            probe_prompt_count=len(probe_rows),
            backend=TransformersPeftBackend(model_path, device=args.device),
        )
    if args.validate_smoke:
        validate_live_smoke(args.run_root)
    print(json.dumps({"status": "completed", "pairs": pairs, "prompt_count": len(probe_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
