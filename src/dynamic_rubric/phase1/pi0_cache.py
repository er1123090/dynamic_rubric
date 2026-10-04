"""Immutable, content-addressed pi0 response cache for Phase-1 OnlineRubrics."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Mapping, Sequence

from dynamic_rubric.artifacts import (
    jsonl_bytes,
    read_json,
    read_jsonl,
    write_bytes_atomic,
    write_json_atomic,
)
from dynamic_rubric.hashing import canonical_json_bytes, sha256_bytes, sha256_file, sha256_json
from dynamic_rubric.providers.base import GenerationRequest
from dynamic_rubric.providers.vllm_generation import VLLMPolicyGenerator
from dynamic_rubric.training.online_contracts import (
    PolicySnapshot,
    PromptOccurrence,
    ResponseRecord,
)

RESPONSES_PER_PROMPT = 8
DEFAULT_PROMPT_COUNT = 1500
CACHE_SCHEMA_VERSION = 1


class Pi0CacheError(RuntimeError):
    pass


def _source_identity(
    *, prompt_id: str, source_row_id: str, messages: Sequence[Mapping[str, str]]
) -> str:
    return sha256_json(
        {
            "prompt_id": prompt_id,
            "source_row_id": source_row_id,
            "messages": [dict(message) for message in messages],
        }
    )


def _prompt_seed(base_seed: int, prompt_id: str, source_identity: str) -> int:
    digest = sha256_json(
        {
            "namespace": "phase1_pi0_control_v1",
            "base_seed": base_seed,
            "prompt_id": prompt_id,
            "source_identity": source_identity,
        }
    )
    return int(digest[:15], 16)


def build_pi0_cache(
    *,
    prompts: Sequence[Mapping[str, Any]],
    generator: VLLMPolicyGenerator,
    output_dir: Path,
    model_revision: str,
    checkpoint_hash: str,
    base_seed: int = 11,
    expected_prompt_count: int = DEFAULT_PROMPT_COUNT,
    max_concurrency: int = 8,
    source_path: Path | None = None,
    progress_every: int = 25,
) -> Path:
    """Generate exactly eight deterministic, non-thinking pi0 responses per prompt."""

    if len(prompts) != expected_prompt_count:
        raise Pi0CacheError(
            f"expected exactly {expected_prompt_count} prompts, got {len(prompts)}"
        )
    if max_concurrency < 1:
        raise Pi0CacheError("max_concurrency must be positive")
    if progress_every < 1:
        raise Pi0CacheError("progress_every must be positive")
    identity = dict(generator.preflight())
    if (
        identity.get("served_model") != generator.model
        or identity.get("model_revision") != model_revision
        or identity.get("checkpoint_hash") != checkpoint_hash
        or identity.get("thinking") is not False
    ):
        raise Pi0CacheError("pi0 generator identity does not match the pinned cache identity")

    prepared: list[dict[str, Any]] = []
    seen_prompt_ids: set[str] = set()
    seen_sources: set[tuple[str, str]] = set()
    for source_index, prompt in enumerate(prompts):
        prompt_id = str(prompt.get("prompt_id", ""))
        source_row_id = str(prompt.get("source_row_id", prompt_id))
        messages_value = prompt.get("messages")
        if not prompt_id or not source_row_id or not isinstance(messages_value, list):
            raise Pi0CacheError(f"invalid prompt identity at source index {source_index}")
        messages = tuple(dict(message) for message in messages_value)
        source_identity = _source_identity(
            prompt_id=prompt_id,
            source_row_id=source_row_id,
            messages=messages,
        )
        if prompt_id in seen_prompt_ids or (prompt_id, source_row_id) in seen_sources:
            raise Pi0CacheError(f"duplicate prompt/source identity: {prompt_id}/{source_row_id}")
        seen_prompt_ids.add(prompt_id)
        seen_sources.add((prompt_id, source_row_id))
        prepared.append(
            {
                "source_index": source_index,
                "prompt_id": prompt_id,
                "source_row_id": source_row_id,
                "messages": messages,
                "source_identity": source_identity,
            }
        )

    generation_config = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "model": generator.model,
        "model_revision": model_revision,
        "tokenizer_revision": generator.tokenizer_revision,
        "checkpoint_hash": checkpoint_hash,
        "thinking": False,
        "base_seed": base_seed,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_output_tokens": 3584,
        "responses_per_prompt": RESPONSES_PER_PROMPT,
        "generator_identity": identity,
    }
    config_identity = sha256_json(generation_config)
    shard_root = output_dir / ".staging" / config_identity / "prompt_shards"
    shard_root.mkdir(parents=True, exist_ok=True)

    def validate_shard(
        shard: Mapping[str, Any], item: Mapping[str, Any]
    ) -> tuple[Mapping[str, Any], ...]:
        if (
            shard.get("schema_version") != CACHE_SCHEMA_VERSION
            or shard.get("cache_kind") != "phase1_pi0_control_prompt_shard"
            or shard.get("config_identity") != config_identity
            or shard.get("prompt_id") != item["prompt_id"]
            or shard.get("source_row_id") != item["source_row_id"]
            or shard.get("source_identity") != item["source_identity"]
        ):
            raise Pi0CacheError("staged pi0 prompt shard identity mismatch")
        rows_value = shard.get("responses")
        if not isinstance(rows_value, list) or len(rows_value) != RESPONSES_PER_PROMPT:
            raise Pi0CacheError("staged pi0 prompt shard has an incomplete inventory")
        rows = tuple(rows_value)
        if [int(row.get("rollout_index", -1)) for row in rows] != list(
            range(RESPONSES_PER_PROMPT)
        ):
            raise Pi0CacheError("staged pi0 prompt shard has invalid rollout indexes")
        logical_seed = _prompt_seed(
            base_seed, str(item["prompt_id"]), str(item["source_identity"])
        )
        for rollout_index, row in enumerate(rows):
            text = str(row.get("text", ""))
            text_hash = sha256_json(text)
            provider_request = row.get("provider_request")
            expected_response_id = sha256_json(
                {
                    "family": "pi0_control_cache",
                    "prompt_id": item["prompt_id"],
                    "source_identity": item["source_identity"],
                    "rollout_index": rollout_index,
                    "text_hash": text_hash,
                    "checkpoint_hash": checkpoint_hash,
                }
            )
            if (
                row.get("schema_version") != CACHE_SCHEMA_VERSION
                or not text.strip()
                or row.get("prompt_id") != item["prompt_id"]
                or row.get("source_row_id") != item["source_row_id"]
                or row.get("source_identity") != item["source_identity"]
                or row.get("cache_response_id") != expected_response_id
                or row.get("text_hash") != text_hash
                or row.get("model") != generator.model
                or row.get("model_revision") != model_revision
                or row.get("tokenizer_revision") != generator.tokenizer_revision
                or row.get("checkpoint_hash") != checkpoint_hash
                or row.get("logical_seed") != logical_seed + rollout_index
                or not isinstance(provider_request, Mapping)
                or provider_request.get("requested_model") != generator.model
                or provider_request.get("returned_model") != generator.model
            ):
                raise Pi0CacheError("staged pi0 prompt response failed verification")
        return rows

    def generate_prompt(
        item: Mapping[str, Any],
    ) -> tuple[tuple[Mapping[str, Any], ...], bool]:
        shard_path = shard_root / f"{item['source_identity']}.json"
        if shard_path.is_file():
            try:
                cached = read_json(shard_path)
                if not isinstance(cached, Mapping):
                    raise Pi0CacheError("staged pi0 prompt shard must be an object")
                return validate_shard(cached, item), True
            except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError, Pi0CacheError):
                # Staging shards are disposable resume state. Remove only the
                # malformed prompt shard and regenerate that prompt.
                shard_path.unlink(missing_ok=True)

        logical_seed = _prompt_seed(
            base_seed, str(item["prompt_id"]), str(item["source_identity"])
        )
        request = GenerationRequest(
            prompt_id=str(item["prompt_id"]),
            messages=tuple(item["messages"]),
            family="phase1_pi0_control_precompute",
            seed=logical_seed,
            temperature=1.0,
            top_p=0.95,
            max_output_tokens=3584,
            metadata={
                "source_row_id": item["source_row_id"],
                "source_identity": item["source_identity"],
                "control_policy": "pi0_precomputed_immutable",
                "checkpoint_hash": checkpoint_hash,
                "config_identity": config_identity,
            },
        )
        generated = tuple(generator.generate_many(request, RESPONSES_PER_PROMPT))
        if len(generated) != RESPONSES_PER_PROMPT:
            raise Pi0CacheError("pi0 provider returned an incomplete response inventory")
        rows = []
        for rollout_index, result in enumerate(generated):
            text_hash = sha256_json(result.text)
            cache_response_id = sha256_json(
                {
                    "family": "pi0_control_cache",
                    "prompt_id": item["prompt_id"],
                    "source_identity": item["source_identity"],
                    "rollout_index": rollout_index,
                    "text_hash": text_hash,
                    "checkpoint_hash": checkpoint_hash,
                }
            )
            rows.append(
                {
                    "schema_version": CACHE_SCHEMA_VERSION,
                    "prompt_id": item["prompt_id"],
                    "source_row_id": item["source_row_id"],
                    "source_identity": item["source_identity"],
                    "rollout_index": rollout_index,
                    "cache_response_id": cache_response_id,
                    "text": result.text,
                    "text_hash": text_hash,
                    "model": generator.model,
                    "model_revision": model_revision,
                    "tokenizer_revision": generator.tokenizer_revision,
                    "checkpoint_hash": checkpoint_hash,
                    "logical_seed": logical_seed + rollout_index,
                    "provider_request": {
                        "request_id": result.request_id,
                        "requested_model": result.requested_model,
                        "returned_model": result.returned_model,
                        "created_at": result.created_at,
                        "retry_count": result.retry_count,
                        "usage": dict(result.usage),
                        "raw_response_hash": result.raw_response_hash,
                    },
                }
            )
        shard = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": "phase1_pi0_control_prompt_shard",
            "config_identity": config_identity,
            "prompt_id": item["prompt_id"],
            "source_row_id": item["source_row_id"],
            "source_identity": item["source_identity"],
            "responses": rows,
        }
        validated = validate_shard(shard, item)
        write_json_atomic(shard_path, shard, immutable=True)
        return validated, False

    groups: list[tuple[Mapping[str, Any], ...]] = []
    reused = 0
    generated_count = 0
    print(
        f"pi0-cache progress 0/{len(prepared)} (reused=0 generated=0)",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=min(max_concurrency, len(prepared))) as pool:
        for completed, (item, result) in enumerate(
            zip(prepared, pool.map(generate_prompt, prepared)), start=1
        ):
            rows, was_reused = result
            reused += int(was_reused)
            generated_count += int(not was_reused)
            groups.append(
                tuple({**dict(row), "source_index": item["source_index"]} for row in rows)
            )
            if completed % progress_every == 0 or completed == len(prepared):
                print(
                    f"pi0-cache progress {completed}/{len(prepared)} "
                    f"(reused={reused} generated={generated_count})",
                    flush=True,
                )
    records = tuple(row for group in groups for row in group)
    response_bytes = jsonl_bytes(records)
    response_hash = sha256_bytes(response_bytes)
    response_name = f"responses-{response_hash}.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_bytes_atomic(output_dir / response_name, response_bytes, immutable=True)

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_kind": "phase1_pi0_control_precomputed_immutable",
        "prompt_count": expected_prompt_count,
        "responses_per_prompt": RESPONSES_PER_PROMPT,
        "response_count": len(records),
        "model": generator.model,
        "model_revision": model_revision,
        "tokenizer_revision": generator.tokenizer_revision,
        "checkpoint_hash": checkpoint_hash,
        "thinking": False,
        "base_seed": base_seed,
        "seed_strategy": "sha256(source_identity,base_seed)+rollout_index",
        "source_path": str(source_path.resolve()) if source_path else None,
        "source_sha256": sha256_file(source_path) if source_path else None,
        "response_jsonl": response_name,
        "response_jsonl_sha256": response_hash,
        "generator_identity": identity,
        "staging_config_identity": config_identity,
    }
    manifest_bytes = canonical_json_bytes(manifest) + b"\n"
    manifest_hash = sha256_bytes(manifest_bytes)
    manifest_path = output_dir / f"manifest-{manifest_hash}.json"
    write_bytes_atomic(manifest_path, manifest_bytes, immutable=True)
    return manifest_path


class ImmutablePi0Cache:
    """Verified prompt-indexed view over a sealed pi0 cache."""

    def __init__(self, manifest_path: Path, *, expected_prompt_count: int = DEFAULT_PROMPT_COUNT):
        self.manifest_path = manifest_path.resolve()
        manifest = read_json(self.manifest_path)
        manifest_hash = sha256_file(self.manifest_path)
        expected_name = f"manifest-{manifest_hash}.json"
        if self.manifest_path.name != expected_name:
            raise Pi0CacheError("pi0 cache manifest is not content-addressed")
        if (
            manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or manifest.get("cache_kind") != "phase1_pi0_control_precomputed_immutable"
            or manifest.get("thinking") is not False
            or int(manifest.get("prompt_count", -1)) != expected_prompt_count
            or int(manifest.get("responses_per_prompt", -1)) != RESPONSES_PER_PROMPT
            or int(manifest.get("response_count", -1))
            != expected_prompt_count * RESPONSES_PER_PROMPT
        ):
            raise Pi0CacheError("pi0 cache manifest contract is invalid")
        response_name = str(manifest.get("response_jsonl", ""))
        response_path = self.manifest_path.parent / response_name
        if response_path.parent != self.manifest_path.parent:
            raise Pi0CacheError("pi0 response inventory must be beside the manifest")
        response_hash = str(manifest.get("response_jsonl_sha256", ""))
        if response_path.name != f"responses-{response_hash}.jsonl":
            raise Pi0CacheError("pi0 response inventory is not content-addressed")
        if not response_path.is_file() or sha256_file(response_path) != response_hash:
            raise Pi0CacheError("pi0 response inventory hash mismatch")

        rows = read_jsonl(response_path)
        groups: dict[str, list[Mapping[str, Any]]] = {}
        for row in rows:
            prompt_id = str(row.get("prompt_id", ""))
            groups.setdefault(prompt_id, []).append(row)
        if len(groups) != expected_prompt_count:
            raise Pi0CacheError("pi0 cache prompt inventory is incomplete")

        self.manifest = manifest
        self.model = str(manifest["model"])
        self.revision = str(manifest["model_revision"])
        self.tokenizer_revision = str(manifest["tokenizer_revision"])
        self.checkpoint_hash = str(manifest["checkpoint_hash"])
        self.manifest_hash = manifest_hash
        self._groups: dict[str, tuple[Mapping[str, Any], ...]] = {}
        for prompt_id, prompt_rows in groups.items():
            ordered = tuple(sorted(prompt_rows, key=lambda row: int(row["rollout_index"])))
            if [int(row["rollout_index"]) for row in ordered] != list(
                range(RESPONSES_PER_PROMPT)
            ):
                raise Pi0CacheError(f"pi0 response inventory is invalid for {prompt_id}")
            source_identities = {str(row.get("source_identity", "")) for row in ordered}
            source_row_ids = {str(row.get("source_row_id", "")) for row in ordered}
            if len(source_identities) != 1 or len(source_row_ids) != 1:
                raise Pi0CacheError(f"pi0 source identity drifted for {prompt_id}")
            for row in ordered:
                text = str(row.get("text", ""))
                rollout_index = int(row["rollout_index"])
                provider_request = row.get("provider_request")
                expected_cache_response_id = sha256_json(
                    {
                        "family": "pi0_control_cache",
                        "prompt_id": prompt_id,
                        "source_identity": row.get("source_identity"),
                        "rollout_index": rollout_index,
                        "text_hash": sha256_json(text),
                        "checkpoint_hash": self.checkpoint_hash,
                    }
                )
                if (
                    row.get("schema_version") != CACHE_SCHEMA_VERSION
                    or not text.strip()
                    or row.get("model") != self.model
                    or row.get("model_revision") != self.revision
                    or row.get("tokenizer_revision") != self.tokenizer_revision
                    or row.get("checkpoint_hash") != self.checkpoint_hash
                    or row.get("text_hash") != sha256_json(text)
                    or row.get("cache_response_id") != expected_cache_response_id
                    or not isinstance(provider_request, Mapping)
                    or provider_request.get("requested_model") != self.model
                    or provider_request.get("returned_model") != self.model
                ):
                    raise Pi0CacheError(f"pi0 cached response failed verification for {prompt_id}")
            self._groups[prompt_id] = ordered

    def bind(
        self, occurrence: PromptOccurrence, *, step: int
    ) -> tuple[tuple[ResponseRecord, ...], tuple[Mapping[str, Any], ...]]:
        rows = self._groups.get(occurrence.prompt_id)
        if rows is None:
            raise Pi0CacheError(f"pi0 cache has no prompt_id {occurrence.prompt_id}")
        source_identity = _source_identity(
            prompt_id=occurrence.prompt_id,
            source_row_id=occurrence.source_row_id,
            messages=occurrence.prompt,
        )
        if any(
            row.get("source_row_id") != occurrence.source_row_id
            or row.get("source_identity") != source_identity
            for row in rows
        ):
            raise Pi0CacheError(
                f"pi0 cache source identity mismatch for {occurrence.prompt_id}"
            )
        snapshot = PolicySnapshot(
            policy_version=0,
            content_hash=self.checkpoint_hash,
            model=self.model,
            revision=self.revision,
        )
        records = []
        receipts = []
        for row in rows:
            rollout_index = int(row["rollout_index"])
            response_id = sha256_json(
                {
                    "family": "control",
                    "occurrence": occurrence.prompt_occurrence_id,
                    "index": rollout_index,
                    "text": row["text"],
                    "policy_hash": self.checkpoint_hash,
                }
            )
            records.append(
                ResponseRecord(
                    prompt_occurrence_id=occurrence.prompt_occurrence_id,
                    response_id=response_id,
                    rollout_index=rollout_index,
                    text=str(row["text"]),
                    policy=snapshot,
                    family="control",
                )
            )
            receipts.append(
                {
                    "optimizer_update_index": step,
                    "prompt_occurrence_id": occurrence.prompt_occurrence_id,
                    "prompt_id": occurrence.prompt_id,
                    "source_row_id": occurrence.source_row_id,
                    "source_identity": source_identity,
                    "response_id": response_id,
                    "rollout_index": rollout_index,
                    "control_source": "precomputed_immutable",
                    "cache_manifest": str(self.manifest_path),
                    "cache_manifest_hash": self.manifest_hash,
                    "cache_response_id": row["cache_response_id"],
                    "requested_model": row["provider_request"]["requested_model"],
                    "returned_model": row["provider_request"]["returned_model"],
                    "model_revision": self.revision,
                    "checkpoint_hash": self.checkpoint_hash,
                    "request_id": row["provider_request"]["request_id"],
                    "raw_response_hash": row["provider_request"]["raw_response_hash"],
                    "preflight_identity": dict(self.manifest["generator_identity"]),
                }
            )
        return tuple(records), tuple(receipts)


def _load_prompts(path: Path) -> list[Mapping[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute immutable Phase-1 pi0 controls")
    parser.add_argument("--train-jsonl", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--tokenizer-revision")
    parser.add_argument("--checkpoint-hash", required=True)
    parser.add_argument("--launch-spec", type=Path)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--prompt-count", type=int, default=DEFAULT_PROMPT_COUNT)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()

    generator = VLLMPolicyGenerator(
        args.base_url,
        args.model,
        args.revision,
        args.tokenizer_revision or args.revision,
        launch_spec_path=args.launch_spec,
        expected_checkpoint_hash=args.checkpoint_hash,
    )
    manifest = build_pi0_cache(
        prompts=_load_prompts(args.train_jsonl),
        generator=generator,
        output_dir=args.output_dir,
        model_revision=args.revision,
        checkpoint_hash=args.checkpoint_hash,
        base_seed=args.seed,
        expected_prompt_count=args.prompt_count,
        max_concurrency=args.concurrency,
        source_path=args.train_jsonl,
        progress_every=args.progress_every,
    )
    print(manifest)


if __name__ == "__main__":
    main()
