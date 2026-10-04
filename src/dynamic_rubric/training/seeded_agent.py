"""Per-row seeded veRL single-turn agent used only for trajectory probes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dynamic_rubric.seeds import SeedFamily, derive_seed, response_id, vllm_seed
from dynamic_rubric.training.rollout_cache import ImmutableRolloutCache
from verl.experimental.agent_loop.agent_loop import (  # pyright: ignore[reportMissingImports]
    AgentLoopOutput,
)
from verl.experimental.agent_loop.single_turn_agent_loop import (  # pyright: ignore[reportMissingImports]
    SingleTurnAgentLoop,
)


class SeededSingleTurnAgentLoop(SingleTurnAgentLoop):
    """Inject the immutable logical seed carried by each probe parquet row."""

    async def run(
        self,
        sampling_params: dict[str, Any],
        run_id: str,
        seed_family: str,
        prompt_id: str,
        policy_step: int,
        rollout_index: int,
        seed_sample_index: int = -1,
        priority: int = 0,
        extra_info: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> AgentLoopOutput:
        family = SeedFamily(seed_family)
        sample_index = (
            int(seed_sample_index) if int(seed_sample_index) >= 0 else int(rollout_index)
        )
        logical_seed = derive_seed(
            str(run_id), family, str(prompt_id), int(policy_step), sample_index
        )
        canonical_response_id = response_id(
            str(run_id), family, str(prompt_id), int(policy_step), sample_index
        )
        seeded = dict(sampling_params)
        seeded["seed"] = vllm_seed(logical_seed)
        # Agent-loop kwargs may share object references with the trainer batch.
        # Keep response-specific seed metadata local so DataProto.union sees immutable inputs.
        metadata = dict(extra_info) if isinstance(extra_info, dict) else {}
        metadata.update(
            {
                "run_id": str(run_id),
                "family": family.value,
                "prompt_id": str(prompt_id),
                "policy_step": int(policy_step),
                "sample_index": sample_index,
                "logical_seed": str(logical_seed),
                "vllm_seed": seeded["seed"],
                "response_id": canonical_response_id,
            }
        )
        cache_root = os.environ.get("DYNAMIC_RUBRIC_ROLLOUT_CACHE_DIR")
        cache = ImmutableRolloutCache(Path(cache_root)) if cache_root else None
        identity = {
            "schema_version": 1,
            "response_id": canonical_response_id,
            "run_id": str(run_id),
            "family": family.value,
            "prompt_id": str(prompt_id),
            "policy_step": int(policy_step),
            "sample_index": sample_index,
            "logical_seed": str(logical_seed),
            "vllm_seed": seeded["seed"],
            "policy_model_path": str(self.config.actor_rollout_ref.model.path),
            "prompt_length": int(self.rollout_config.prompt_length),
            "response_length": int(self.rollout_config.response_length),
            "sampling_params": seeded,
            "raw_prompt": kwargs.get("raw_prompt"),
        }
        if cache is not None:
            cached = cache.read(identity)
            if cached is not None:
                result = AgentLoopOutput.model_validate(cached)
                result.extra_fields.update(metadata)
                return result

        result = await super().run(
            seeded,
            priority=int(logical_seed % (2**31)),
            extra_info=metadata,
            **kwargs,
        )
        result.extra_fields.update(metadata)
        if cache is None:
            return result
        published = cache.publish(identity, result.model_dump(mode="json"))
        return AgentLoopOutput.model_validate(published)
