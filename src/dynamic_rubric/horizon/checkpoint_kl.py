"""Post-training adjacent-checkpoint policy KL estimation on Pool-B responses."""

from __future__ import annotations

import base64
import hashlib
import json
import math
import statistics
import struct
import urllib.request
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..artifacts import (
    artifact_record,
    read_json,
    read_jsonl,
    validate_artifact_record,
    write_json_atomic,
    write_jsonl_atomic,
)


SCHEMA_VERSION = 1
K3_LOG_RATIO_CLIP = 20.0


class CheckpointKLError(RuntimeError):
    """Raised when checkpoint KL inputs or live log-prob responses are invalid."""


def _template_token_ids(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> list[int]:
    rendered = tokenizer.apply_chat_template(
        [dict(message) for message in messages],
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if isinstance(rendered, Mapping):
        rendered = rendered.get("input_ids")
    if not isinstance(rendered, Sequence) or isinstance(rendered, (str, bytes)):
        raise CheckpointKLError("chat template did not return token IDs")
    if rendered and isinstance(rendered[0], Sequence):
        if len(rendered) != 1:
            raise CheckpointKLError("chat template returned an unexpected batch")
        rendered = rendered[0]
    token_ids = [int(token_id) for token_id in rendered]
    if not token_ids:
        raise CheckpointKLError("chat template produced an empty prompt")
    return token_ids


def _encode_float32(values: Sequence[float]) -> str:
    if not values:
        raise CheckpointKLError("response token log-probs must not be empty")
    payload = struct.pack(f"<{len(values)}f", *(float(value) for value in values))
    return base64.b64encode(payload).decode("ascii")


def _decode_float32(value: str, expected_count: int) -> tuple[float, ...]:
    try:
        payload = base64.b64decode(value.encode("ascii"), validate=True)
    except Exception as error:
        raise CheckpointKLError("token log-prob payload is not valid base64") from error
    if len(payload) != expected_count * 4:
        raise CheckpointKLError("token log-prob payload length mismatch")
    values = struct.unpack(f"<{expected_count}f", payload)
    if not all(math.isfinite(item) for item in values):
        raise CheckpointKLError("token log-probs must be finite")
    return values


def _token_hash(token_ids: Sequence[int]) -> str:
    encoded = json.dumps(list(token_ids), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _selected_prompt_logprob(entry: Any, token_id: int) -> float:
    if not isinstance(entry, Mapping):
        raise CheckpointKLError("chosen response token has no prompt log-prob")
    selected = entry.get(str(token_id), entry.get(token_id))
    if isinstance(selected, Mapping):
        selected = selected.get("logprob")
    try:
        value = float(selected)
    except (TypeError, ValueError) as error:
        raise CheckpointKLError("chosen response token log-prob is absent") from error
    if not math.isfinite(value):
        raise CheckpointKLError("chosen response token log-prob is non-finite")
    return value


class VLLMPolicyLogprobClient:
    """Score existing response tokens through a checkpoint-bound policy proxy."""

    def __init__(
        self,
        base_url: str,
        *,
        served_model: str,
        model_revision: str,
        tokenizer_revision: str,
        checkpoint_hash: str,
        timeout_seconds: float = 900.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.served_model = served_model
        self.model_revision = model_revision
        self.tokenizer_revision = tokenizer_revision
        self.checkpoint_hash = checkpoint_hash
        self.timeout_seconds = timeout_seconds

    def preflight(self) -> Mapping[str, Any]:
        with urllib.request.urlopen(
            f"{self.base_url}/dynamic-rubric/identity", timeout=self.timeout_seconds
        ) as response:
            identity = json.loads(response.read())
        expected = {
            "served_model": self.served_model,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "checkpoint_hash": self.checkpoint_hash,
            "thinking": False,
        }
        if not isinstance(identity, Mapping) or any(
            identity.get(key) != value for key, value in expected.items()
        ):
            raise CheckpointKLError(
                f"policy identity mismatch: expected={expected}, got={identity}"
            )
        model_path = Path(str(identity.get("model_path", "")))
        if not model_path.is_dir():
            raise CheckpointKLError("policy identity model_path is absent")
        return identity

    def score(
        self,
        token_sequences: Sequence[Sequence[int]],
        response_starts: Sequence[int],
    ) -> list[tuple[float, ...]]:
        if not token_sequences or len(token_sequences) != len(response_starts):
            raise ValueError("token sequences and response starts must be aligned and non-empty")
        payload = {
            "model": self.served_model,
            "prompt": [list(sequence) for sequence in token_sequences],
            # vLLM requires max_tokens >= 1; the generated token is discarded.
            "max_tokens": 1,
            "echo": False,
            "logprobs": 0,
            "prompt_logprobs": 0,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{self.base_url}/v1/completions",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            value = json.loads(response.read())
        if value.get("model") != self.served_model:
            raise CheckpointKLError("policy model identity drifted during log-prob scoring")
        choices = sorted(value.get("choices", []), key=lambda item: int(item["index"]))
        if len(choices) != len(token_sequences):
            raise CheckpointKLError("policy log-prob response count mismatch")
        outputs: list[tuple[float, ...]] = []
        for choice, expected_ids, response_start in zip(
            choices, token_sequences, response_starts
        ):
            prompt_ids = [int(item) for item in choice.get("prompt_token_ids", [])]
            prompt_logprobs = choice.get("prompt_logprobs")
            if prompt_ids != list(expected_ids) or not isinstance(prompt_logprobs, list):
                raise CheckpointKLError("policy prompt token identity/log-probs mismatch")
            if not 0 < response_start < len(prompt_ids):
                raise CheckpointKLError("response token boundary is invalid")
            outputs.append(
                tuple(
                    _selected_prompt_logprob(prompt_logprobs[position], prompt_ids[position])
                    for position in range(response_start, len(prompt_ids))
                )
            )
        return outputs


def _validate_pool_grid(
    prompts: Sequence[Mapping[str, Any]], pool_rows: Sequence[Mapping[str, Any]]
) -> tuple[int, int]:
    prompt_ids = {str(row["prompt_id"]) for row in prompts}
    if not prompt_ids:
        raise CheckpointKLError("prompt inventory must not be empty")
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in pool_rows:
        if row.get("pool_family") != "pool_b":
            raise CheckpointKLError("checkpoint KL requires Pool-B responses")
        groups[str(row["prompt_id"])].append(row)
    if set(groups) != prompt_ids:
        raise CheckpointKLError("Pool-B prompt inventory mismatch")
    counts = {len(rows) for rows in groups.values()}
    if len(counts) != 1:
        raise CheckpointKLError("Pool-B response count differs across prompts")
    response_count = next(iter(counts))
    if response_count <= 0:
        raise CheckpointKLError("Pool-B groups must not be empty")
    for prompt_id, rows in groups.items():
        if {int(row["sample_index"]) for row in rows} != set(range(response_count)):
            raise CheckpointKLError(f"Pool-B sample slots mismatch for {prompt_id}")
    return len(prompt_ids), response_count


def _validate_existing_scores(
    path: Path,
    *,
    policy_step: int,
    pool_step: int,
    checkpoint_hash: str,
    expected_response_ids: set[str],
) -> list[Mapping[str, Any]]:
    rows = read_jsonl(path)
    if {str(row["response_id"]) for row in rows} != expected_response_ids:
        raise CheckpointKLError(f"existing checkpoint KL response inventory mismatch: {path}")
    if any(
        int(row.get("policy_step", -1)) != policy_step
        or int(row.get("pool_policy_step", -1)) != pool_step
        or row.get("scoring_checkpoint_hash") != checkpoint_hash
        for row in rows
    ):
        raise CheckpointKLError(f"existing checkpoint KL identity mismatch: {path}")
    return rows


def score_policy_logprobs_from_files(
    client: VLLMPolicyLogprobClient,
    *,
    prompts_path: Path,
    pool_paths: Sequence[Path],
    policy_step: int,
    output_dir: Path,
    batch_size: int = 32,
) -> dict[str, Any]:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    prompts = read_jsonl(prompts_path)
    prompts_by_id = {str(row["prompt_id"]): row for row in prompts}
    if len(prompts_by_id) != len(prompts):
        raise CheckpointKLError("prompt IDs must be unique")
    identity = client.preflight()

    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    tokenizer = AutoTokenizer.from_pretrained(
        str(identity["model_path"]), local_files_only=True
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    artifacts: list[dict[str, Any]] = []
    for pool_path in pool_paths:
        pool_rows = read_jsonl(pool_path)
        prompt_count, responses_per_prompt = _validate_pool_grid(prompts, pool_rows)
        pool_steps = {int(row["policy_step"]) for row in pool_rows}
        if len(pool_steps) != 1:
            raise CheckpointKLError("one Pool-B file must contain one policy step")
        pool_step = next(iter(pool_steps))
        output_path = output_dir / f"policy-step-{policy_step}_pool-step-{pool_step}.jsonl"
        expected_response_ids = {str(row["response_id"]) for row in pool_rows}
        if output_path.exists():
            existing = _validate_existing_scores(
                output_path,
                policy_step=policy_step,
                pool_step=pool_step,
                checkpoint_hash=client.checkpoint_hash,
                expected_response_ids=expected_response_ids,
            )
            artifacts.append(
                {
                    **artifact_record(output_path),
                    "rows": len(existing),
                    "reused": True,
                }
            )
            continue

        prepared: list[dict[str, Any]] = []
        for row in sorted(
            pool_rows,
            key=lambda item: (
                str(item["prompt_id"]),
                int(item["sample_index"]),
                str(item["response_id"]),
            ),
        ):
            prompt_id = str(row["prompt_id"])
            prefix_ids = _template_token_ids(tokenizer, prompts_by_id[prompt_id]["messages"])
            response_ids = [
                int(token_id)
                for token_id in tokenizer.encode(
                    str(row["response_text"]), add_special_tokens=False
                )
            ]
            if not response_ids:
                raise CheckpointKLError(f"empty response tokenization: {row['response_id']}")
            token_ids = [*prefix_ids, *response_ids]
            prepared.append(
                {
                    "pool": row,
                    "prefix_count": len(prefix_ids),
                    "response_ids": response_ids,
                    "token_ids": token_ids,
                }
            )

        scored_rows: list[dict[str, Any]] = []
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            batch_scores = client.score(
                [item["token_ids"] for item in batch],
                [int(item["prefix_count"]) for item in batch],
            )
            for item, token_logprobs in zip(batch, batch_scores):
                pool = item["pool"]
                response_ids = item["response_ids"]
                if len(token_logprobs) != len(response_ids):
                    raise CheckpointKLError("response token/log-prob count mismatch")
                scored_rows.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "policy_step": policy_step,
                        "pool_policy_step": pool_step,
                        "scoring_checkpoint_hash": client.checkpoint_hash,
                        "source_checkpoint_hash": str(pool["checkpoint_hash"]),
                        "model_revision": client.model_revision,
                        "tokenizer_revision": client.tokenizer_revision,
                        "prompt_id": str(pool["prompt_id"]),
                        "response_id": str(pool["response_id"]),
                        "sample_index": int(pool["sample_index"]),
                        "prompt_prefix_token_count": int(item["prefix_count"]),
                        "response_token_count": len(response_ids),
                        "response_token_hash": _token_hash(response_ids),
                        "response_logprob_sum": float(sum(token_logprobs)),
                        "response_logprob_mean": float(statistics.fmean(token_logprobs)),
                        "response_token_logprobs_f32le_b64": _encode_float32(
                            token_logprobs
                        ),
                    }
                )
        if {str(row["response_id"]) for row in scored_rows} != expected_response_ids:
            raise CheckpointKLError("scored response inventory mismatch")
        write_jsonl_atomic(output_path, scored_rows)
        artifacts.append(
            {
                **artifact_record(output_path),
                "rows": len(scored_rows),
                "prompts": prompt_count,
                "responses_per_prompt": responses_per_prompt,
                "reused": False,
            }
        )
    return {"policy_step": policy_step, "artifacts": artifacts}


def _score_index(path: Path) -> dict[str, Mapping[str, Any]]:
    rows = read_jsonl(path)
    index = {str(row["response_id"]): row for row in rows}
    if len(index) != len(rows) or not rows:
        raise CheckpointKLError(f"score artifact has duplicate or absent responses: {path}")
    return index


def _standard_error(values: Sequence[float]) -> float:
    return statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else 0.0


def analyze_adjacent_checkpoint_kl(
    *,
    score_dir: Path,
    checkpoint_steps: Sequence[int],
    prompts_path: Path,
    output_dir: Path,
    expected_responses_per_prompt: int,
) -> dict[str, Any]:
    if len(checkpoint_steps) < 2 or any(
        right <= left for left, right in zip(checkpoint_steps, checkpoint_steps[1:])
    ):
        raise ValueError("checkpoint_steps must be strictly increasing")
    prompt_ids = {str(row["prompt_id"]) for row in read_jsonl(prompts_path)}
    if not prompt_ids:
        raise CheckpointKLError("prompt inventory must not be empty")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "adjacent_kl_summary.json"
    prompt_path = output_dir / "adjacent_kl_prompt.jsonl"
    response_path = output_dir / "adjacent_kl_response.jsonl"
    seal_path = output_dir / "adjacent_kl_seal.json"
    if seal_path.exists():
        seal = read_json(seal_path)
        for record in seal.get("artifacts", []):
            validate_artifact_record(record)
        return {"reused": True, "seal": str(seal_path), "summary": str(summary_path)}

    pair_summaries: list[dict[str, Any]] = []
    prompt_rows: list[dict[str, Any]] = []
    response_rows: list[dict[str, Any]] = []
    for old_step, new_step in zip(checkpoint_steps, checkpoint_steps[1:]):
        old_path = score_dir / f"policy-step-{old_step}_pool-step-{old_step}.jsonl"
        new_path = score_dir / f"policy-step-{new_step}_pool-step-{old_step}.jsonl"
        if not old_path.is_file() or not new_path.is_file():
            raise CheckpointKLError(
                f"adjacent KL score artifacts are missing for {old_step}->{new_step}"
            )
        old_rows = _score_index(old_path)
        new_rows = _score_index(new_path)
        if set(old_rows) != set(new_rows):
            raise CheckpointKLError("old/new score response inventories differ")
        by_prompt: dict[str, list[dict[str, Any]]] = defaultdict(list)
        clipped_tokens = 0
        total_tokens = 0
        token_k1_sum = 0.0
        token_k3_sum = 0.0
        for response_id in sorted(old_rows):
            old = old_rows[response_id]
            new = new_rows[response_id]
            identity_fields = ("prompt_id", "sample_index", "response_token_count", "response_token_hash")
            if any(old.get(field) != new.get(field) for field in identity_fields):
                raise CheckpointKLError(f"old/new response identity mismatch: {response_id}")
            token_count = int(old["response_token_count"])
            old_values = _decode_float32(
                str(old["response_token_logprobs_f32le_b64"]), token_count
            )
            new_values = _decode_float32(
                str(new["response_token_logprobs_f32le_b64"]), token_count
            )
            k1_values: list[float] = []
            k3_values: list[float] = []
            response_clipped = 0
            for old_value, new_value in zip(old_values, new_values):
                log_ratio = new_value - old_value
                clipped = min(K3_LOG_RATIO_CLIP, max(-K3_LOG_RATIO_CLIP, log_ratio))
                response_clipped += int(clipped != log_ratio)
                k1_values.append(-log_ratio)
                k3_values.append(math.expm1(clipped) - clipped)
            response_row = {
                "schema_version": SCHEMA_VERSION,
                "direction": "old_to_new",
                "old_policy_step": old_step,
                "new_policy_step": new_step,
                "step_gap": new_step - old_step,
                "prompt_id": str(old["prompt_id"]),
                "response_id": response_id,
                "sample_index": int(old["sample_index"]),
                "response_token_count": token_count,
                "k1_mean": statistics.fmean(k1_values),
                "k3_clipped_mean": statistics.fmean(k3_values),
                "k3_clipped_token_count": response_clipped,
            }
            response_rows.append(response_row)
            by_prompt[response_row["prompt_id"]].append(response_row)
            total_tokens += token_count
            clipped_tokens += response_clipped
            token_k1_sum += sum(k1_values)
            token_k3_sum += sum(k3_values)

        if set(by_prompt) != prompt_ids:
            raise CheckpointKLError("adjacent KL prompt inventory mismatch")
        if any(len(rows) != expected_responses_per_prompt for rows in by_prompt.values()):
            raise CheckpointKLError("adjacent KL responses-per-prompt mismatch")
        current_prompt_rows: list[dict[str, Any]] = []
        for prompt_id in sorted(by_prompt):
            rows = by_prompt[prompt_id]
            current_prompt_rows.append(
                {
                    "schema_version": SCHEMA_VERSION,
                    "direction": "old_to_new",
                    "old_policy_step": old_step,
                    "new_policy_step": new_step,
                    "step_gap": new_step - old_step,
                    "prompt_id": prompt_id,
                    "response_count": len(rows),
                    "token_count": sum(int(row["response_token_count"]) for row in rows),
                    "k1_response_balanced_mean": statistics.fmean(
                        float(row["k1_mean"]) for row in rows
                    ),
                    "k3_clipped_response_balanced_mean": statistics.fmean(
                        float(row["k3_clipped_mean"]) for row in rows
                    ),
                }
            )
        prompt_rows.extend(current_prompt_rows)
        response_k1 = [float(row["k1_mean"]) for rows in by_prompt.values() for row in rows]
        response_k3 = [
            float(row["k3_clipped_mean"]) for rows in by_prompt.values() for row in rows
        ]
        prompt_k1 = [float(row["k1_response_balanced_mean"]) for row in current_prompt_rows]
        prompt_k3 = [
            float(row["k3_clipped_response_balanced_mean"]) for row in current_prompt_rows
        ]
        step_gap = new_step - old_step
        pair_summaries.append(
            {
                "old_policy_step": old_step,
                "new_policy_step": new_step,
                "step_gap": step_gap,
                "old_checkpoint_hash": next(iter(old_rows.values()))[
                    "scoring_checkpoint_hash"
                ],
                "new_checkpoint_hash": next(iter(new_rows.values()))[
                    "scoring_checkpoint_hash"
                ],
                "prompt_count": len(current_prompt_rows),
                "response_count": len(response_k1),
                "token_count": total_tokens,
                "k1_token_weighted_mean": token_k1_sum / total_tokens,
                "k1_response_balanced_mean": statistics.fmean(response_k1),
                "k1_prompt_balanced_mean": statistics.fmean(prompt_k1),
                "k1_prompt_balanced_se": _standard_error(prompt_k1),
                "k1_prompt_balanced_per_step_proxy": statistics.fmean(prompt_k1)
                / step_gap,
                "k3_clipped_token_weighted_mean": token_k3_sum / total_tokens,
                "k3_clipped_response_balanced_mean": statistics.fmean(response_k3),
                "k3_clipped_prompt_balanced_mean": statistics.fmean(prompt_k3),
                "k3_clipped_prompt_balanced_se": _standard_error(prompt_k3),
                "k3_clipped_prompt_balanced_per_step_proxy": statistics.fmean(prompt_k3)
                / step_gap,
                "k3_log_ratio_clip": K3_LOG_RATIO_CLIP,
                "k3_clipped_token_fraction": clipped_tokens / total_tokens,
            }
        )

    summary = {
        "schema_version": SCHEMA_VERSION,
        "estimator_direction": "KL(old_policy || new_policy)",
        "sampling_distribution": "old_checkpoint_pool_b",
        "primary_aggregation": "prompt_balanced",
        "k1_note": "Monte Carlo log-ratio estimator; finite samples may be negative.",
        "k3_note": "Non-negative k3 estimator with the recorded log-ratio clipping bound.",
        "checkpoint_steps": list(checkpoint_steps),
        "pairs": pair_summaries,
    }
    write_json_atomic(summary_path, summary)
    write_jsonl_atomic(prompt_path, prompt_rows)
    write_jsonl_atomic(response_path, response_rows)
    records = [artifact_record(path) for path in (summary_path, prompt_path, response_path)]
    write_json_atomic(
        seal_path,
        {
            "schema_version": SCHEMA_VERSION,
            "complete": True,
            "artifacts": records,
        },
    )
    return {
        "reused": False,
        "seal": str(seal_path),
        "summary": str(summary_path),
        "pairs": len(pair_summaries),
    }
