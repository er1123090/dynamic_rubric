"""Concrete DataProto runtime for paper-faithful OnlineRubrics training."""

from __future__ import annotations

import dataclasses
import hashlib
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamic_rubric.artifacts import read_json, write_json_atomic, write_jsonl_atomic
from dynamic_rubric.hashing import sha256_file, sha256_json
from dynamic_rubric.phase1.pi0_cache import ImmutablePi0Cache, Pi0CacheError
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.openai_responses import OpenAIResponsesAdapter
from dynamic_rubric.providers.vllm_chat import VLLMChatAdapter
from dynamic_rubric.providers.vllm_generation import VLLMPolicyGenerator
from dynamic_rubric.training.online_contracts import (
    OnlineStepInput,
    PolicySnapshot,
    PromptGroupInput,
    PromptOccurrence,
    ResponseRecord,
    StepState,
    WeightedCriterion,
)
from dynamic_rubric.training.online_step import (
    OnlineHookResult,
    OnlineStepCoordinator,
    OnlineStepError,
)


class OnlineRuntimeFactoryError(RuntimeError):
    pass


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise OnlineRuntimeFactoryError(f"{name} is required by the online veRL runtime")
    return value


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise OnlineStepError(f"{label} must be a mapping")
    return value


def _vllm_base_urls(name: str) -> tuple[str, ...]:
    singular = os.environ.get(name)
    plural = os.environ.get(f"{name}S")
    if singular and plural:
        raise OnlineRuntimeFactoryError(f"{name} and {name}S cannot both be set")
    raw = plural or singular
    if not raw:
        return ()
    urls = tuple(value.strip() for value in raw.split(",") if value.strip())
    if not 1 <= len(urls) <= 2:
        raise OnlineRuntimeFactoryError(f"{name}S must contain one or two comma-separated URLs")
    if len(set(urls)) != len(urls):
        raise OnlineRuntimeFactoryError(f"{name}S must contain unique URLs")
    return urls


def _online_generation_providers(
    *,
    extractor_model: str,
    grader_model: str,
    cache_root: Path,
) -> tuple[Any, Any]:
    extractor_urls = _vllm_base_urls("PHASE1_GPT_OSS_BASE_URL")
    grader_urls = _vllm_base_urls("PHASE1_QWEN32B_BASE_URL")
    if bool(extractor_urls) != bool(grader_urls):
        raise OnlineRuntimeFactoryError(
            "PHASE1_GPT_OSS_BASE_URL(S) and PHASE1_QWEN32B_BASE_URL(S) "
            "must be set together"
        )
    if extractor_urls and grader_urls:
        common_key = os.environ.get("PHASE1_VLLM_API_KEY")
        timeout = float(os.environ.get("PHASE1_VLLM_TIMEOUT_SECONDS", "120"))
        retries = int(os.environ.get("PHASE1_VLLM_MAX_RETRIES", "4"))
        extractor = VLLMChatAdapter(
            extractor_urls,
            extractor_model,
            cache_root / "extractor",
            api_key=os.environ.get("PHASE1_GPT_OSS_API_KEY", common_key),
            timeout_seconds=timeout,
            max_retries=retries,
        )
        grader = VLLMChatAdapter(
            grader_urls,
            grader_model,
            cache_root / "grader",
            api_key=os.environ.get("PHASE1_QWEN32B_API_KEY", common_key),
            timeout_seconds=timeout,
            max_retries=retries,
        )
        return extractor, grader

    api_key = _required_environment("OPENAI_API_KEY")
    return (
        OpenAIResponsesAdapter(api_key, extractor_model, cache_root / "extractor"),
        OpenAIResponsesAdapter(api_key, grader_model, cache_root / "grader"),
    )


def _directory_hash(root: Path, *, exclude_resume_state: bool = False) -> str:
    if not root.is_dir():
        raise OnlineStepError(f"checkpoint directory is missing: {root}")
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (
            exclude_resume_state
            and (
                path.name.startswith("optim_world_size_")
                or path.name.startswith("extra_state_world_size_")
                or path.name == "data.pt"
                or path.name.startswith("data_")
            )
        )
    )
    if not files:
        raise OnlineStepError(f"checkpoint directory is empty: {root}")
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(sha256_file(path)))
    return digest.hexdigest()


def _checkpoint_hash(root: Path) -> str:
    if not (root / "data.pt").is_file():
        raise OnlineStepError(f"per-step checkpoint is incomplete: {root}")
    return _directory_hash(root)


def _actor_parameter_hash(root: Path) -> str:
    """Hash only actor artifacts retained for evaluator audits."""

    return _directory_hash(root, exclude_resume_state=True)


class VerlOnlineRewardRuntime:
    def __init__(
        self,
        *,
        trainer: Any,
        coordinator: OnlineStepCoordinator,
        control_generator: VLLMPolicyGenerator | None,
        artifact_root: Path,
        run_id: str,
        actor_model: str,
        actor_revision: str,
        control_hash: str,
        control_concurrency: int,
        seed: int,
        control_cache: ImmutablePi0Cache | None = None,
    ) -> None:
        self.trainer = trainer
        self.coordinator = coordinator
        if (control_generator is None) == (control_cache is None):
            raise OnlineRuntimeFactoryError(
                "exactly one live pi0 generator or immutable pi0 cache is required"
            )
        self.control_generator = control_generator
        self.control_cache = control_cache
        self.control_model = (
            control_generator.model if control_generator is not None else control_cache.model
        )
        self.control_revision = (
            control_generator.revision if control_generator is not None else control_cache.revision
        )
        self.artifact_root = artifact_root
        self.run_id = run_id
        self.actor_model = actor_model
        self.actor_revision = actor_revision
        self.control_hash = control_hash
        self.control_concurrency = control_concurrency
        self.seed = seed
        self._prepared: dict[int, Any] = {}
        self._current_hashes: dict[int, str] = {}

    def _current_policy_hash(self, step: int) -> str:
        if step == 1:
            return self.control_hash
        latest_path = self.artifact_root.parent / "latest_commit.json"
        if not latest_path.is_file():
            raise OnlineStepError("step after bootstrap requires latest_commit.json")
        latest = read_json(latest_path)
        if int(latest.get("optimizer_update_index", -1)) != step - 1:
            raise OnlineStepError("latest commit is not the immediately previous optimizer step")
        prior_commit_path = self.artifact_root / f"step-{step - 1:06d}" / "commit.json"
        if not prior_commit_path.is_file():
            raise OnlineStepError("latest commit has no immutable prior commit manifest")
        prior_commit = read_json(prior_commit_path)

        artifacts = prior_commit.get("artifacts", {})
        logical_policy_token = str(latest.get("logical_policy_token", ""))
        if (
            str(prior_commit.get("state", "")) != StepState.COMMITTED.value
            or str(artifacts.get("logical_policy_token", "")) != logical_policy_token
            or str(latest.get("manifest_hash", "")) != sha256_json(prior_commit)
        ):
            raise OnlineStepError("latest commit disagrees with the immutable prior manifest")
        if len(logical_policy_token) != 64:
            raise OnlineStepError("latest commit has no valid logical policy-version token")

        if latest.get("checkpoint_saved") is True:
            checkpoint_path = Path(str(latest.get("checkpoint", "")))
            expected_root = Path(str(self.trainer.config.trainer.default_local_dir))
            if not expected_root.is_absolute():
                expected_root = Path.cwd() / expected_root
            expected_checkpoint = expected_root / f"global_step_{step - 1}"
            if checkpoint_path.resolve() != expected_checkpoint.resolve():
                raise OnlineStepError("latest commit points to the wrong prior checkpoint")
            resume_hash = str(latest.get("resume_checkpoint_hash", ""))
            parameter_hash = str(latest.get("actor_parameter_hash", ""))
            if (
                str(artifacts.get("resume_checkpoint_hash", "")) != resume_hash
                or str(artifacts.get("actor_parameter_hash", "")) != parameter_hash
            ):
                raise OnlineStepError("latest checkpoint hashes disagree with the prior manifest")
            if _checkpoint_hash(checkpoint_path) != resume_hash:
                raise OnlineStepError("latest resume checkpoint hash no longer matches disk")
            if _actor_parameter_hash(checkpoint_path / "actor") != parameter_hash:
                raise OnlineStepError("latest actor parameter hash no longer matches disk")

        return logical_policy_token

    def _control_responses(
        self,
        occurrences: Sequence[PromptOccurrence],
        *,
        step: int,
    ) -> tuple[dict[str, tuple[ResponseRecord, ...]], tuple[Mapping[str, Any], ...]]:
        if self.control_cache is not None:
            try:
                bound = tuple(
                    (
                        occurrence.prompt_occurrence_id,
                        *self.control_cache.bind(occurrence, step=step),
                    )
                    for occurrence in occurrences
                )
            except Pi0CacheError as error:
                raise OnlineStepError("immutable pi0 control cache validation failed") from error
            controls = {occurrence_id: records for occurrence_id, records, _ in bound}
            receipts = tuple(
                receipt
                for _, _, occurrence_receipts in bound
                for receipt in occurrence_receipts
            )
            return controls, receipts

        if self.control_generator is None:
            raise OnlineStepError("live pi0 control generator is unavailable")
        identity = self.control_generator.preflight()
        observed_hash = str(identity.get("checkpoint_hash") or identity.get("revision") or "")
        if observed_hash != self.control_hash:
            raise OnlineStepError("frozen pi_ref control checkpoint identity drifted")
        if str(identity.get("served_model", "")) != self.control_generator.model:
            raise OnlineStepError("frozen pi_ref control model identity drifted")
        def generate(
            occurrence: PromptOccurrence,
        ) -> tuple[str, tuple[ResponseRecord, ...], tuple[Mapping[str, Any], ...]]:
            request = GenerationRequest(
                prompt_id=occurrence.prompt_id,
                messages=occurrence.prompt,
                family="online_pi_ref_control",
                seed=self.seed + step,
                temperature=float(os.environ.get("ONLINE_CONTROL_TEMPERATURE", "1.0")),
                top_p=float(os.environ.get("ONLINE_CONTROL_TOP_P", "0.95")),
                max_output_tokens=int(os.environ.get("ONLINE_CONTROL_MAX_TOKENS", "3584")),
                metadata={
                    "optimizer_update_index": step,
                    "prompt_occurrence_id": occurrence.prompt_occurrence_id,
                    "control_policy": "pi_ref",
                    "control_checkpoint_hash": self.control_hash,
                },
            )
            generated = tuple(self.control_generator.generate_many(request, 8))
            snapshot = PolicySnapshot(
                policy_version=0,
                content_hash=self.control_hash,
                model=self.control_generator.model,
                revision=self.control_generator.revision,
            )
            records: list[ResponseRecord] = []
            receipts: list[Mapping[str, Any]] = []
            for index, result in enumerate(generated):
                response_id = sha256_json(
                    {
                        "family": "control",
                        "occurrence": occurrence.prompt_occurrence_id,
                        "index": index,
                        "text": result.text,
                        "policy_hash": self.control_hash,
                    }
                )
                records.append(
                    ResponseRecord(
                        prompt_occurrence_id=occurrence.prompt_occurrence_id,
                        response_id=response_id,
                        rollout_index=index,
                        text=result.text,
                        policy=snapshot,
                        family="control",
                    )
                )
                receipts.append(
                    {
                        "optimizer_update_index": step,
                        "prompt_occurrence_id": occurrence.prompt_occurrence_id,
                        "response_id": response_id,
                        "rollout_index": index,
                        "requested_model": result.requested_model,
                        "returned_model": result.returned_model,
                        "model_revision": self.control_generator.revision,
                        "checkpoint_hash": self.control_hash,
                        "request_id": result.request_id,
                        "raw_response_hash": result.raw_response_hash,
                        "preflight_identity": dict(identity),
                    }
                )
            return occurrence.prompt_occurrence_id, tuple(records), tuple(receipts)

        with ThreadPoolExecutor(
            max_workers=min(self.control_concurrency, len(occurrences))
        ) as pool:
            generated_rows = tuple(pool.map(generate, occurrences))
        controls = {occurrence_id: records for occurrence_id, records, _ in generated_rows}
        receipts = tuple(
            receipt
            for _, _, occurrence_receipts in generated_rows
            for receipt in occurrence_receipts
        )
        return controls, receipts

    def _step_input(self, batch: Any, *, step: int) -> OnlineStepInput:
        if step < 1:
            raise OnlineStepError("online optimizer step must be positive")
        responses = self.trainer.tokenizer.batch_decode(
            batch.batch["responses"], skip_special_tokens=True
        )
        extra_infos = batch.non_tensor_batch.get("extra_info")
        uids = batch.non_tensor_batch.get("uid")
        if extra_infos is None or uids is None or len(extra_infos) != len(responses):
            raise OnlineStepError("DataProto is missing extra_info/uid response bindings")
        if len(responses) % 16:
            raise OnlineStepError("online DataProto response inventory is not divisible by 16")

        group_identities = []
        for offset in range(0, len(responses), 16):
            first = _mapping(extra_infos[offset], "extra_info")
            group_identities.append(
                {
                    "prompt_occurrence_id": str(first.get("prompt_occurrence_id", "")),
                    "prompt_id": str(first.get("prompt_id", "")),
                    "source_row_id": str(first.get("source_row_id", "")),
                }
            )
        batch_uid = sha256_json(
            {
                "run_id": self.run_id,
                "step": step,
                "groups": group_identities,
            }
        )
        occurrences: list[PromptOccurrence] = []
        group_rows: list[tuple[PromptOccurrence, tuple[WeightedCriterion, ...], tuple[ResponseRecord, ...]]] = []
        current_policy_hash = self._current_policy_hash(step)
        self._current_hashes[step] = current_policy_hash
        current_snapshot = PolicySnapshot(
            policy_version=step - 1,
            content_hash=current_policy_hash,
            model=self.actor_model,
            revision=self.actor_revision,
        )
        for offset in range(0, len(responses), 16):
            group_extra = [_mapping(value, "extra_info") for value in extra_infos[offset : offset + 16]]
            first = group_extra[0]
            group_uids = {str(value) for value in uids[offset : offset + 16]}
            if len(group_uids) != 1:
                raise OnlineStepError("rollout group mixes veRL group uids")
            occurrence_id = str(first.get("prompt_occurrence_id", ""))
            prompt_id = str(first.get("prompt_id", ""))
            if not occurrence_id or not prompt_id:
                raise OnlineStepError("online row lacks prompt occurrence identity")
            if any(
                str(value.get("prompt_occurrence_id", "")) != occurrence_id
                for value in group_extra
            ):
                raise OnlineStepError("rollout group mixes prompt occurrences")
            prompt_messages = first.get("prompt_messages")
            offline_payload = first.get("offline_criteria")
            if not isinstance(prompt_messages, (list, tuple)) or not isinstance(
                offline_payload, (list, tuple)
            ):
                raise OnlineStepError("online row lacks prompt messages or offline criteria")
            occurrence = PromptOccurrence(
                run_id=self.run_id,
                optimizer_update_index=step,
                batch_uid=batch_uid,
                source_row_id=str(first.get("source_row_id", "")),
                prompt_id=prompt_id,
                prompt_occurrence_id=occurrence_id,
                prompt=tuple(_mapping(item, "prompt message") for item in prompt_messages),
            )
            criteria = tuple(
                WeightedCriterion(
                    criterion_id=str(_mapping(item, "offline criterion")["criterion_id"]),
                    text=str(_mapping(item, "offline criterion")["text"]),
                    weight=_mapping(item, "offline criterion")["weight"],
                    source="offline_r0",
                )
                for item in offline_payload
            )
            current = tuple(
                ResponseRecord(
                    prompt_occurrence_id=occurrence_id,
                    response_id=sha256_json(
                        {
                            "family": "current",
                            "occurrence": occurrence_id,
                            "step": step,
                            "index": index,
                            "text": responses[offset + index],
                        }
                    ),
                    rollout_index=index,
                    text=str(responses[offset + index]),
                    policy=current_snapshot,
                    family="current",
                )
                for index in range(16)
            )
            occurrences.append(occurrence)
            group_rows.append((occurrence, criteria, current))

        controls, control_receipts = self._control_responses(occurrences, step=step)
        groups = tuple(
            PromptGroupInput(
                occurrence=occurrence,
                offline_criteria=criteria,
                current_responses=current,
                control_responses=controls[occurrence.prompt_occurrence_id],
            )
            for occurrence, criteria, current in group_rows
        )
        artifact_dir = self.artifact_root / f"step-{step:06d}"
        write_json_atomic(
            artifact_dir / "batch.json",
            {
                "schema_version": 1,
                "run_id": self.run_id,
                "optimizer_update_index": step,
                "batch_uid": batch_uid,
                "prompt_occurrence_ids": [
                    occurrence.prompt_occurrence_id for occurrence in occurrences
                ],
                "current_policy": dataclasses.asdict(current_snapshot),
                "control_policy": {
                    "mode": "pi_ref",
                    "policy_version": 0,
                    "content_hash": self.control_hash,
                    "model": self.control_model,
                    "revision": self.control_revision,
                },
            },
        )
        write_jsonl_atomic(
            artifact_dir / "current_responses.jsonl",
            (
                dataclasses.asdict(response)
                for group in groups
                for response in group.current_responses
            ),
        )
        write_jsonl_atomic(
            artifact_dir / "control_responses.jsonl",
            (
                dataclasses.asdict(response)
                for group in groups
                for response in group.control_responses
            ),
        )
        write_jsonl_atomic(
            artifact_dir / "control_generation_receipts.jsonl",
            control_receipts,
        )
        return OnlineStepInput(
            run_id=self.run_id,
            optimizer_update_index=step,
            batch_uid=batch_uid,
            seed=self.seed,
            prompt_groups=groups,
        )

    def prepare_online_step(self, batch: Any, *, step: int) -> OnlineHookResult:
        if step in self._prepared:
            raise OnlineStepError(f"online step {step} was already prepared in this process")
        step_input = self._step_input(batch, step=step)
        artifact_dir = self.artifact_root / f"step-{step:06d}"
        result = self.coordinator.run(step_input, artifact_dir=artifact_dir)
        hook_result = OnlineHookResult(
            optimizer_update_index=step,
            rm_scores=result.reward_scalars,
            trace_refs=result.trace_refs,
            manifest_hash=result.manifest.content_hash,
            sealed=result.sealed,
        )
        self._prepared[step] = result
        return hook_result

    def apply_hook_result(self, batch: Any, result: OnlineHookResult) -> Any:
        import numpy as np
        import torch

        response_mask = batch.batch.get("response_mask")
        if response_mask is None:
            response_length = batch.batch["responses"].shape[-1]
            response_mask = batch.batch["attention_mask"][:, -response_length:]
            batch.batch["response_mask"] = response_mask
        if len(result.rm_scores) != response_mask.shape[0]:
            raise OnlineStepError("reward scalar count does not match DataProto responses")
        terminal = torch.where(
            response_mask.to(dtype=torch.bool),
            torch.arange(response_mask.shape[-1], device=response_mask.device).unsqueeze(0),
            -1,
        ).amax(dim=-1)
        if torch.any(terminal < 0):
            raise OnlineStepError("cannot place reward on an empty response")
        rm_scores = torch.zeros_like(response_mask, dtype=torch.float32)
        values = torch.tensor(result.rm_scores, dtype=torch.float32, device=rm_scores.device)
        rm_scores.scatter_(1, terminal.unsqueeze(-1), values.unsqueeze(-1))
        batch.batch["rm_scores"] = rm_scores
        batch.non_tensor_batch["online_trace_ref"] = np.asarray(
            result.trace_refs, dtype=object
        )
        batch.meta_info["online_step_sealed"] = result.sealed
        batch.meta_info["online_optimizer_update_index"] = result.optimizer_update_index
        batch.meta_info["online_manifest_hash"] = result.manifest_hash
        return batch

    def commit_step(
        self, *, global_step: int, checkpoint_dir: str | None, trainer: Any
    ) -> None:
        result = self._prepared.get(global_step)
        if result is None or not result.sealed:
            raise OnlineStepError("cannot commit an unprepared online step")
        latest_path = self.artifact_root.parent / "latest_commit.json"
        if global_step == 1:
            if latest_path.exists():
                raise OnlineStepError("first online commit refuses an existing latest commit")
        else:
            if not latest_path.is_file():
                raise OnlineStepError("noninitial online commit requires the previous commit")
            previous = read_json(latest_path)
            if int(previous.get("optimizer_update_index", -1)) != global_step - 1:
                raise OnlineStepError("online commits must be sequential")
            if self._current_hashes.get(global_step) != str(
                previous.get("logical_policy_token", "")
            ):
                raise OnlineStepError("prepared current policy is not the previous committed version")

        logical_policy_token = sha256_json(
            {
                "kind": "logical_policy_version",
                "run_id": self.run_id,
                "optimizer_update_index": global_step,
                "pre_update_policy_token": self._current_hashes.get(
                    global_step, self.control_hash
                ),
                "reward_manifest_hash": result.manifest.content_hash,
            }
        )
        checkpoint_root = Path(checkpoint_dir) if checkpoint_dir is not None else None
        physical_artifacts: dict[str, str] = {}
        latest_physical: dict[str, Any] = {"checkpoint_saved": False}
        if checkpoint_root is not None:
            resume_digest = _checkpoint_hash(checkpoint_root)
            parameter_digest = _actor_parameter_hash(checkpoint_root / "actor")
            physical_artifacts = {
                "resume_checkpoint_hash": resume_digest,
                "actor_parameter_hash": parameter_digest,
            }
            latest_physical = {
                "checkpoint_saved": True,
                "checkpoint": str(checkpoint_root),
                "resume_checkpoint_hash": resume_digest,
                "actor_parameter_hash": parameter_digest,
            }
        committed = dataclasses.replace(
            result.manifest,
            state=StepState.COMMITTED,
            artifacts={
                **dict(result.manifest.artifacts),
                "logical_policy_token": logical_policy_token,
                **physical_artifacts,
            },
        )
        step_root = self.artifact_root / f"step-{global_step:06d}"
        write_json_atomic(step_root / "commit.json", dataclasses.asdict(committed))
        write_json_atomic(
            self.artifact_root.parent / "latest_commit.json",
            {
                "schema_version": 1,
                "optimizer_update_index": global_step,
                "manifest_hash": committed.content_hash,
                "logical_policy_token": logical_policy_token,
                **latest_physical,
            },
            immutable=False,
        )
        del self._prepared[global_step]


def create_online_reward_runtime(*, config: Any, trainer: Any) -> VerlOnlineRewardRuntime:
    runtime_config = config.reward.get("online_step_runtime") or {}
    control_policy = str(runtime_config.get("control_policy", ""))
    if control_policy != "pi_ref":
        raise OnlineRuntimeFactoryError(
            "pi_old is disabled until a bounded one-batch lookahead sampler is implemented"
        )
    if runtime_config.get("frozen_control", False) is not True:
        raise OnlineRuntimeFactoryError("pi_ref requires frozen_control=true")
    artifact_root = Path(str(runtime_config.get("artifact_root", "")))
    if not artifact_root.is_absolute():
        raise OnlineRuntimeFactoryError("online_step_runtime.artifact_root must be absolute")

    extractor_model = _required_environment("ONLINE_EXTRACTOR_MODEL")
    grader_model = _required_environment("ONLINE_GRADER_MODEL")
    cache_root = artifact_root.parent / "provider_cache"
    extractor, grader = _online_generation_providers(
        extractor_model=extractor_model,
        grader_model=grader_model,
        cache_root=cache_root,
    )
    coordinator = OnlineStepCoordinator(
        extractor=extractor,
        deduper=extractor,
        grader=grader,
        extractor_model=extractor_model,
        grader_model=grader_model,
        extractor_concurrency=int(os.environ.get("ONLINE_EXTRACTOR_CONCURRENCY", "32")),
        grader_concurrency=int(os.environ.get("ONLINE_GRADER_CONCURRENCY", "32")),
        extractor_reasoning_effort=os.environ.get("ONLINE_EXTRACTOR_REASONING_EFFORT", "medium"),
        extractor_max_output_tokens=int(
            os.environ.get("ONLINE_EXTRACTOR_MAX_OUTPUT_TOKENS", "8192")
        ),
        dedup_max_output_tokens=int(
            os.environ.get("ONLINE_DEDUP_MAX_OUTPUT_TOKENS", "8192")
        ),
        grader_max_output_tokens=int(
            os.environ.get("ONLINE_GRADER_MAX_OUTPUT_TOKENS", "4096")
        ),
        extractor_returned_model=os.environ.get("ONLINE_EXTRACTOR_RETURNED_MODEL", extractor_model),
        grader_returned_model=os.environ.get("ONLINE_GRADER_RETURNED_MODEL", grader_model),
    )

    actor_model = _required_environment("ONLINE_ACTOR_MODEL")
    actor_revision = _required_environment("ONLINE_ACTOR_REVISION")
    expected_a0_hash = _required_environment("ONLINE_CONTROL_CHECKPOINT_HASH")
    if len(expected_a0_hash) != 64:
        raise OnlineRuntimeFactoryError("ONLINE_CONTROL_CHECKPOINT_HASH must be a SHA-256 digest")
    try:
        bytes.fromhex(expected_a0_hash)
    except ValueError as error:
        raise OnlineRuntimeFactoryError(
            "ONLINE_CONTROL_CHECKPOINT_HASH must be a SHA-256 digest"
        ) from error
    actor_snapshot = Path(_required_environment("MODEL_PATH"))
    try:
        observed_a0_hash = _directory_hash(actor_snapshot)
    except OnlineStepError as error:
        raise OnlineRuntimeFactoryError("the local A0 actor snapshot is unavailable") from error
    if observed_a0_hash != expected_a0_hash:
        raise OnlineRuntimeFactoryError(
            "the local A0 actor snapshot differs from ONLINE_CONTROL_CHECKPOINT_HASH"
        )
    control_model = os.environ.get("ONLINE_CONTROL_MODEL", actor_model)
    control_revision = os.environ.get("ONLINE_CONTROL_REVISION", actor_revision)
    control_tokenizer_revision = os.environ.get(
        "ONLINE_CONTROL_TOKENIZER_REVISION", control_revision
    )
    if control_model != actor_model or control_revision != actor_revision:
        raise OnlineRuntimeFactoryError("pi_ref control must be the exact initial actor model/revision")
    cache_path = os.environ.get("ONLINE_CONTROL_CACHE")
    control_cache: ImmutablePi0Cache | None = None
    control: VLLMPolicyGenerator | None = None
    if cache_path:
        try:
            control_cache = ImmutablePi0Cache(Path(cache_path))
        except (OSError, ValueError, Pi0CacheError) as error:
            raise OnlineRuntimeFactoryError("immutable pi0 control cache is invalid") from error
        if (
            control_cache.model != control_model
            or control_cache.revision != control_revision
            or control_cache.tokenizer_revision != control_tokenizer_revision
            or control_cache.checkpoint_hash != expected_a0_hash
        ):
            raise OnlineRuntimeFactoryError(
                "immutable pi0 control cache does not match the initial actor identity"
            )
        control_hash = control_cache.checkpoint_hash
    else:
        launch_spec = os.environ.get("ONLINE_CONTROL_LAUNCH_SPEC")
        control = VLLMPolicyGenerator(
            _required_environment("ONLINE_CONTROL_URL"),
            control_model,
            control_revision,
            control_tokenizer_revision,
            launch_spec_path=Path(launch_spec) if launch_spec else None,
            expected_checkpoint_hash=expected_a0_hash,
        )
        identity = control.preflight()
        control_hash = str(identity.get("checkpoint_hash", ""))
        if control_hash != expected_a0_hash:
            raise OnlineRuntimeFactoryError("frozen pi_ref control is not the local A0 snapshot")
    return VerlOnlineRewardRuntime(
        trainer=trainer,
        coordinator=coordinator,
        control_generator=control,
        artifact_root=artifact_root,
        run_id=_required_environment("ONLINE_RUN_ID"),
        actor_model=actor_model,
        actor_revision=actor_revision,
        control_hash=control_hash,
        control_concurrency=int(os.environ.get("ONLINE_CONTROL_CONCURRENCY", "32")),
        seed=int(os.environ.get("ONLINE_SEED", "1729")),
        control_cache=control_cache,
    )
