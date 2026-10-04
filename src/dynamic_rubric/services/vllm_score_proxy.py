from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from threading import Condition, Lock
from typing import Any, Iterator, Mapping


@dataclass
class ProxyState:
    upstream: str | Sequence[str]
    served_model: str
    model_revision: str
    tokenizer_revision: str
    tokenizer: Any
    upstream_weights: Sequence[int] | None = None
    request_affinity: bool = False
    cache_dir: Path | None = None
    adaptive_upstream_index: int | None = None
    adaptive_gpu_index: int | None = None
    adaptive_low_utilization_weight: int | None = None
    adaptive_utilization_threshold: float = 70.0
    upstream_timeout_seconds: float = 900.0
    utilization_cache_seconds: float = 1.0
    managed_sleep_upstream_index: int | None = None
    managed_sleep_idle_seconds: float = 3.0
    managed_sleep_timeout_seconds: float = 300.0
    managed_sleep_level: int = 1
    managed_wake_gpu_index: int | None = None
    managed_wake_max_memory_used_mib: int | None = None
    managed_wake_poll_seconds: float = 1.0
    utilization_reader: Callable[[int], float] | None = field(default=None, repr=False)
    memory_used_reader: Callable[[int], float] | None = field(default=None, repr=False)
    _cache_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _utilization_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _metrics_lock: Lock = field(default_factory=Lock, init=False, repr=False)
    _managed_condition: Condition = field(default_factory=Condition, init=False, repr=False)
    _cached_utilization: float | None = field(default=None, init=False, repr=False)
    _cached_utilization_at: float = field(default=0.0, init=False, repr=False)
    _routing_upstreams: tuple[str, ...] = field(default=(), init=False, repr=False)
    _upstream_request_counts: list[int] = field(default_factory=list, init=False, repr=False)
    _managed_active_requests: int = field(default=0, init=False, repr=False)
    _managed_request_epoch: int = field(default=0, init=False, repr=False)
    _managed_transitioning: bool = field(default=False, init=False, repr=False)
    _managed_awake: bool = field(default=True, init=False, repr=False)
    _managed_wake_count: int = field(default=0, init=False, repr=False)
    _managed_sleep_count: int = field(default=0, init=False, repr=False)

    def __post_init__(self) -> None:
        values = (self.upstream,) if isinstance(self.upstream, str) else tuple(self.upstream)
        normalized = tuple(value.rstrip("/") for value in values if value.rstrip("/"))
        if not normalized:
            raise ValueError("at least one upstream is required")
        if len(set(normalized)) != len(normalized):
            raise ValueError("upstreams must be unique")
        if self.upstream_weights is None:
            weights = (1,) * len(normalized)
        else:
            weights = tuple(self.upstream_weights)
            if len(weights) != len(normalized):
                raise ValueError("upstream weights must align one-to-one with upstreams")
            if any(
                isinstance(weight, bool) or not isinstance(weight, int) or weight < 1
                for weight in weights
            ):
                raise ValueError("upstream weights must be positive integers")
        self.upstream = normalized
        self.upstream_weights = weights
        self._routing_upstreams = tuple(
            upstream for upstream, weight in zip(normalized, weights) for _ in range(weight)
        )
        self._upstream_request_counts = [0] * len(normalized)
        adaptive_values = (
            self.adaptive_upstream_index,
            self.adaptive_gpu_index,
            self.adaptive_low_utilization_weight,
        )
        if any(value is not None for value in adaptive_values) and not all(
            value is not None for value in adaptive_values
        ):
            raise ValueError(
                "adaptive routing requires upstream index, GPU index, and low-utilization weight"
            )
        if self.adaptive_upstream_index is not None and not (
            0 <= self.adaptive_upstream_index < len(normalized)
        ):
            raise ValueError("adaptive upstream index is out of range")
        if self.adaptive_gpu_index is not None and self.adaptive_gpu_index < 0:
            raise ValueError("adaptive GPU index must be non-negative")
        if (
            self.adaptive_low_utilization_weight is not None
            and self.adaptive_low_utilization_weight < 1
        ):
            raise ValueError("adaptive low-utilization weight must be positive")
        if not 0 <= self.adaptive_utilization_threshold <= 100:
            raise ValueError("adaptive utilization threshold must be between 0 and 100")
        if self.upstream_timeout_seconds <= 0:
            raise ValueError("upstream timeout seconds must be positive")
        if self.utilization_cache_seconds < 0:
            raise ValueError("utilization cache seconds must be non-negative")
        if self.managed_sleep_upstream_index is not None and not (
            0 <= self.managed_sleep_upstream_index < len(normalized)
        ):
            raise ValueError("managed sleep upstream index is out of range")
        if self.managed_sleep_idle_seconds < 0:
            raise ValueError("managed sleep idle seconds must be non-negative")
        if self.managed_sleep_timeout_seconds <= 0:
            raise ValueError("managed sleep timeout seconds must be positive")
        if self.managed_sleep_level not in {1, 2}:
            raise ValueError("managed sleep level must be 1 or 2")
        wake_gate_values = (
            self.managed_wake_gpu_index,
            self.managed_wake_max_memory_used_mib,
        )
        if any(value is not None for value in wake_gate_values) and not all(
            value is not None for value in wake_gate_values
        ):
            raise ValueError("managed wake memory gate requires GPU index and maximum memory used")
        if self.managed_wake_gpu_index is not None and self.managed_wake_gpu_index < 0:
            raise ValueError("managed wake GPU index must be non-negative")
        if (
            self.managed_wake_max_memory_used_mib is not None
            and self.managed_wake_max_memory_used_mib <= 0
        ):
            raise ValueError("managed wake maximum memory used must be positive")
        if self.managed_wake_poll_seconds <= 0:
            raise ValueError("managed wake poll seconds must be positive")

    @property
    def upstreams(self) -> tuple[str, ...]:
        if isinstance(self.upstream, str):  # pragma: no cover - normalized on init
            return (self.upstream,)
        return tuple(self.upstream)

    @property
    def identity(self) -> dict[str, Any]:
        return {
            "served_model": self.served_model,
            "model_revision": self.model_revision,
            "tokenizer_revision": self.tokenizer_revision,
            "thinking": False,
            "scoring_mode": "full_vocab_yes_no_prompt_logprob_v4",
        }

    @property
    def runtime_metrics(self) -> dict[str, Any]:
        with self._managed_condition:
            managed_state = (
                "transitioning"
                if self._managed_transitioning
                else "awake"
                if self._managed_awake
                else "asleep"
            )
            managed_active_requests = self._managed_active_requests
            managed_wake_count = self._managed_wake_count
            managed_sleep_count = self._managed_sleep_count
        with self._metrics_lock:
            upstream_request_counts = list(self._upstream_request_counts)
        return {
            "upstream_request_counts": upstream_request_counts,
            "managed_sleep_upstream_index": self.managed_sleep_upstream_index,
            "managed_sleep_state": managed_state,
            "managed_active_requests": managed_active_requests,
            "managed_wake_count": managed_wake_count,
            "managed_sleep_count": managed_sleep_count,
        }

    def score(
        self,
        rendered_prompts: list[str],
        targets: list[str],
        *,
        upstream_index: int | None = None,
    ) -> list[dict[str, float]]:
        valid_targets = ({"YES", "NO"}, {" YES", " NO"})
        if not rendered_prompts or set(targets) not in valid_targets or len(targets) != 2:
            raise ValueError("rendered_prompts must be non-empty and targets one exact YES/NO pair")
        if self.cache_dir is None:
            with self._managed_request():
                return self._score_uncached(
                    rendered_prompts, targets, upstream_index=upstream_index
                )

        keys = [self._cache_key(rendered, targets) for rendered in rendered_prompts]
        rows_by_key: dict[str, dict[str, float]] = {}
        missing_prompts: dict[str, str] = {}
        for key, rendered in zip(keys, rendered_prompts):
            cached = self._read_cache(key, rendered, targets)
            if cached is None:
                missing_prompts.setdefault(key, rendered)
            else:
                rows_by_key[key] = cached

        if missing_prompts:
            missing_keys = list(missing_prompts)
            with self._managed_request():
                uncached = self._score_uncached(
                    [missing_prompts[key] for key in missing_keys],
                    targets,
                    upstream_index=upstream_index,
                )
            if len(uncached) != len(missing_keys):
                raise ValueError("uncached score count mismatch")
            for key, row in zip(missing_keys, uncached):
                rows_by_key[key] = self._publish_or_read_cache(
                    key, missing_prompts[key], targets, row
                )
        return [rows_by_key[key] for key in keys]

    @contextmanager
    def _managed_request(self) -> Iterator[None]:
        if self.managed_sleep_upstream_index is None:
            yield
            return
        self._enter_managed_request()
        try:
            yield
        finally:
            self._exit_managed_request()

    def _enter_managed_request(self) -> None:
        with self._managed_condition:
            self._managed_request_epoch += 1
            self._managed_condition.notify_all()
            while self._managed_transitioning:
                self._managed_condition.wait()
            if self._managed_awake:
                self._managed_active_requests += 1
                return
            self._managed_transitioning = True

        try:
            self._wait_for_managed_wake_capacity()
            self._set_managed_upstream_awake(True)
        except Exception:
            with self._managed_condition:
                self._managed_transitioning = False
                self._managed_condition.notify_all()
            raise
        with self._managed_condition:
            self._managed_awake = True
            self._managed_transitioning = False
            self._managed_wake_count += 1
            self._managed_active_requests += 1
            self._managed_condition.notify_all()

    def _exit_managed_request(self) -> None:
        with self._managed_condition:
            self._managed_active_requests -= 1
            if self._managed_active_requests < 0:
                raise RuntimeError("managed request count became negative")
            if self._managed_active_requests:
                return
            observed_epoch = self._managed_request_epoch
            deadline = time.monotonic() + self.managed_sleep_idle_seconds
            while (
                self._managed_active_requests == 0
                and self._managed_request_epoch == observed_epoch
                and not self._managed_transitioning
            ):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._managed_condition.wait(timeout=remaining)
            if (
                self._managed_active_requests
                or self._managed_request_epoch != observed_epoch
                or self._managed_transitioning
            ):
                return
            self._managed_transitioning = True

        try:
            self._set_managed_upstream_awake(False)
        except Exception:
            with self._managed_condition:
                self._managed_transitioning = False
                self._managed_condition.notify_all()
            raise
        with self._managed_condition:
            self._managed_awake = False
            self._managed_transitioning = False
            self._managed_sleep_count += 1
            self._managed_condition.notify_all()

    def initialize_managed_upstream(self) -> None:
        if self.managed_sleep_upstream_index is None:
            return
        if self._managed_upstream_is_sleeping():
            with self._managed_condition:
                self._managed_awake = False
            return
        self.sleep_managed_upstream()

    def sleep_managed_upstream(self) -> None:
        if self.managed_sleep_upstream_index is None:
            return
        with self._managed_condition:
            if self._managed_active_requests:
                raise RuntimeError("cannot sleep managed upstream with active requests")
            if self._managed_transitioning:
                raise RuntimeError("managed upstream transition is already in progress")
            if not self._managed_awake:
                return
            self._managed_transitioning = True
        try:
            self._set_managed_upstream_awake(False)
        except Exception:
            with self._managed_condition:
                self._managed_transitioning = False
                self._managed_condition.notify_all()
            raise
        with self._managed_condition:
            self._managed_awake = False
            self._managed_transitioning = False
            self._managed_sleep_count += 1
            self._managed_condition.notify_all()

    def _managed_upstream_is_sleeping(self) -> bool:
        if self.managed_sleep_upstream_index is None:
            raise RuntimeError("managed sleep upstream is not configured")
        upstream = self.upstreams[self.managed_sleep_upstream_index]
        request = urllib.request.Request(f"{upstream}/is_sleeping", method="GET")
        with urllib.request.urlopen(request, timeout=30.0) as response:
            status = json.loads(response.read())
        if not isinstance(status.get("is_sleeping"), bool):
            raise ValueError("managed upstream returned an invalid sleep status")
        return status["is_sleeping"]

    def _wait_for_managed_wake_capacity(self) -> None:
        if self.managed_wake_gpu_index is None:
            return
        if self.managed_wake_max_memory_used_mib is None:  # pragma: no cover - validated
            raise RuntimeError("managed wake memory threshold is not configured")
        deadline = time.monotonic() + self.managed_sleep_timeout_seconds
        reader = self.memory_used_reader or self._read_nvidia_gpu_memory_used_mib
        last_error: Exception | None = None
        while True:
            try:
                memory_used_mib = float(reader(self.managed_wake_gpu_index))
                if memory_used_mib < 0:
                    raise ValueError("GPU memory used must be non-negative")
                if memory_used_mib <= self.managed_wake_max_memory_used_mib:
                    return
                last_error = None
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                last_error = error
            if time.monotonic() >= deadline:
                detail = f"; last read failed: {last_error}" if last_error is not None else ""
                raise TimeoutError(
                    f"managed upstream wake capacity was not available before timeout{detail}"
                )
            time.sleep(self.managed_wake_poll_seconds)

    @staticmethod
    def _read_nvidia_gpu_memory_used_mib(gpu_index: int) -> float:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(result.stdout.strip().splitlines()[0])

    def _set_managed_upstream_awake(self, awake: bool) -> None:
        if self.managed_sleep_upstream_index is None:
            raise RuntimeError("managed sleep upstream is not configured")
        upstream = self.upstreams[self.managed_sleep_upstream_index]
        control_path = (
            "/wake_up" if awake else f"/sleep?level={self.managed_sleep_level}&mode=abort"
        )
        request = urllib.request.Request(f"{upstream}{control_path}", method="POST")
        with urllib.request.urlopen(
            request, timeout=self.managed_sleep_timeout_seconds
        ) as response:
            response.read()

        deadline = time.monotonic() + self.managed_sleep_timeout_seconds
        expected_sleeping = not awake
        while True:
            if self._managed_upstream_is_sleeping() is expected_sleeping:
                return
            if time.monotonic() >= deadline:
                action = "wake" if awake else "sleep"
                raise TimeoutError(f"managed upstream did not {action} before timeout")
            time.sleep(0.1)

    def _score_uncached(
        self,
        rendered_prompts: list[str],
        targets: list[str],
        *,
        upstream_index: int | None = None,
    ) -> list[dict[str, float]]:
        if upstream_index is not None or len(self.upstreams) == 1:
            return self._score_on_upstream(
                rendered_prompts,
                targets,
                upstream_index=upstream_index,
            )

        if len(rendered_prompts) == 1:
            preferred_index = self._select_upstream_index(rendered_prompts, targets)
            return self._score_on_upstream_with_failover(
                rendered_prompts, targets, preferred_index=preferred_index
            )

        if self.request_affinity:
            preferred_index = self._select_upstream_index(rendered_prompts, targets)
            return self._score_on_upstream_with_failover(
                rendered_prompts, targets, preferred_index=preferred_index
            )

        assignments = self._balanced_assignments(rendered_prompts)
        shards: list[list[tuple[int, str]]] = [[] for _ in self.upstreams]
        for prompt_index, replica_index in enumerate(assignments):
            shards[replica_index].append((prompt_index, rendered_prompts[prompt_index]))
        rows: list[dict[str, float] | None] = [None] * len(rendered_prompts)
        active = [(index, shard) for index, shard in enumerate(shards) if shard]
        with ThreadPoolExecutor(max_workers=len(active)) as executor:
            futures = {
                executor.submit(
                    self._score_on_upstream_with_failover,
                    [prompt for _, prompt in shard],
                    targets,
                    preferred_index=replica_index,
                ): shard
                for replica_index, shard in active
            }
            for future, shard in futures.items():
                shard_rows = future.result()
                if len(shard_rows) != len(shard):
                    raise ValueError("upstream shard score count mismatch")
                for (prompt_index, _), row in zip(shard, shard_rows):
                    rows[prompt_index] = row
        if any(row is None for row in rows):
            raise ValueError("parallel upstream merge left an unscored prompt")
        return [row for row in rows if row is not None]

    def _balanced_assignments(self, rendered_prompts: list[str]) -> tuple[int, ...]:
        weights = self._effective_weights()
        loads = [0] * len(self.upstreams)
        assignments = [0] * len(rendered_prompts)
        order = sorted(
            range(len(rendered_prompts)),
            key=lambda index: (-len(rendered_prompts[index]), index),
        )
        for prompt_index in order:
            replica_index = min(
                range(len(self.upstreams)),
                key=lambda index: (loads[index] / weights[index], index),
            )
            assignments[prompt_index] = replica_index
            loads[replica_index] += max(1, len(rendered_prompts[prompt_index]))
        return tuple(assignments)

    def _effective_weights(self) -> tuple[int, ...]:
        weights = list(self.upstream_weights or (1,) * len(self.upstreams))
        if self.adaptive_upstream_index is None:
            return tuple(weights)
        utilization = self._gpu_utilization()
        if (
            utilization is not None
            and utilization < self.adaptive_utilization_threshold
            and self.adaptive_low_utilization_weight is not None
        ):
            weights[self.adaptive_upstream_index] = self.adaptive_low_utilization_weight
        return tuple(weights)

    def _gpu_utilization(self) -> float | None:
        if self.adaptive_gpu_index is None:
            return None
        with self._utilization_lock:
            now = time.monotonic()
            if (
                self._cached_utilization is not None
                and now - self._cached_utilization_at < self.utilization_cache_seconds
            ):
                return self._cached_utilization
            try:
                reader = self.utilization_reader or self._read_nvidia_gpu_utilization
                utilization = float(reader(self.adaptive_gpu_index))
                if not 0 <= utilization <= 100:
                    raise ValueError("GPU utilization must be between 0 and 100")
            except (OSError, ValueError, subprocess.SubprocessError):
                return None
            self._cached_utilization = utilization
            self._cached_utilization_at = now
            return utilization

    @staticmethod
    def _read_nvidia_gpu_utilization(gpu_index: int) -> float:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return float(result.stdout.strip().splitlines()[0])

    def _score_on_upstream_with_failover(
        self,
        rendered_prompts: list[str],
        targets: list[str],
        *,
        preferred_index: int,
    ) -> list[dict[str, float]]:
        errors: list[tuple[int, Exception]] = []
        attempt_order = (preferred_index,) + tuple(
            index for index in range(len(self.upstreams)) if index != preferred_index
        )
        for upstream_index in attempt_order:
            try:
                return self._score_on_upstream(
                    rendered_prompts, targets, upstream_index=upstream_index
                )
            except Exception as error:
                if not self._is_transient_upstream_error(error):
                    raise
                errors.append((upstream_index, error))
        details = "; ".join(
            f"upstream[{index}]={type(error).__name__}: {error}" for index, error in errors
        )
        raise RuntimeError(f"all scoring upstreams failed: {details}") from errors[-1][1]

    @staticmethod
    def _is_transient_upstream_error(error: Exception) -> bool:
        if isinstance(error, urllib.error.HTTPError):
            return error.code in {408, 429} or error.code >= 500
        return isinstance(
            error,
            (urllib.error.URLError, TimeoutError, ConnectionError, OSError),
        )

    def _score_on_upstream(
        self,
        rendered_prompts: list[str],
        targets: list[str],
        *,
        upstream_index: int | None = None,
    ) -> list[dict[str, float]]:
        target_token_ids: list[int] = []
        for target in targets:
            target_ids = self.tokenizer.encode(target, add_special_tokens=False)
            if len(target_ids) != 1:
                raise ValueError(f"target must tokenize to exactly one token: {target}")
            target_token_ids.append(int(target_ids[0]))
        token_sequences = [
            self.tokenizer.encode(rendered, add_special_tokens=True)
            for rendered in rendered_prompts
        ]
        selected_index = self._select_upstream_index(
            rendered_prompts, targets, upstream_index=upstream_index
        )
        upstream = self.upstreams[selected_index]
        with self._metrics_lock:
            self._upstream_request_counts[selected_index] += 1
        rows: list[dict[str, float]] = [{} for _ in rendered_prompts]
        scoring_items = [
            (prompt_index, target, [*tokens, target_token_id])
            for target, target_token_id in zip(targets, target_token_ids)
            for prompt_index, tokens in enumerate(token_sequences)
        ]
        payload = {
            "model": self.served_model,
            "prompt": [item[2] for item in scoring_items],
            "max_tokens": 0,
            "echo": True,
            "logprobs": 1,
            "return_token_ids": True,
            "return_tokens_as_token_ids": True,
            "temperature": 0,
        }
        request = urllib.request.Request(
            f"{upstream}/v1/completions",
            data=json.dumps(payload).encode(),
            method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.upstream_timeout_seconds) as response:
            value = json.loads(response.read())
        if value.get("model") != self.served_model:
            raise ValueError(
                f"upstream model drift: expected={self.served_model}, got={value.get('model')}"
            )
        choices = sorted(value.get("choices", []), key=lambda item: int(item["index"]))
        if len(choices) != len(scoring_items):
            raise ValueError("upstream completion count mismatch")
        for (prompt_index, target, expected_prompt_ids), choice in zip(scoring_items, choices):
            prompt_token_ids = [int(item) for item in choice.get("prompt_token_ids", [])]
            if prompt_token_ids != expected_prompt_ids:
                raise ValueError("upstream echoed prompt token identity mismatch")
            logprobs = choice.get("logprobs")
            if not isinstance(logprobs, Mapping):
                raise ValueError("upstream prompt-token logprobs are absent")
            token_logprobs = logprobs.get("token_logprobs")
            if not isinstance(token_logprobs, list) or len(token_logprobs) != len(
                expected_prompt_ids
            ):
                raise ValueError("upstream prompt-token logprobs mismatch")
            score = float(token_logprobs[-1])
            if not math.isfinite(score):
                raise ValueError(f"target token logprob is non-finite: {target}")
            rows[prompt_index][target] = score
        # Appending each candidate to the prompt and echoing it returns the
        # candidate's full-vocabulary log-probability. Restricting generation to
        # one allowed token would renormalize that token to probability 1.
        return rows

    def _select_upstream(
        self,
        rendered_prompts: list[str],
        targets: list[str],
        *,
        upstream_index: int | None = None,
    ) -> str:
        index = self._select_upstream_index(
            rendered_prompts, targets, upstream_index=upstream_index
        )
        return self.upstreams[index]

    def _select_upstream_index(
        self,
        rendered_prompts: list[str],
        targets: list[str],
        *,
        upstream_index: int | None = None,
    ) -> int:
        if upstream_index is not None:
            if (
                isinstance(upstream_index, bool)
                or not isinstance(upstream_index, int)
                or not 0 <= upstream_index < len(self.upstreams)
            ):
                raise ValueError("routing upstream_index is out of range")
            return upstream_index
        routing_payload = json.dumps(
            {"rendered_prompts": rendered_prompts, "targets": targets},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        routing_key = int.from_bytes(hashlib.sha256(routing_payload).digest()[:8], "big")
        routing_indices = tuple(
            index for index, weight in enumerate(self._effective_weights()) for _ in range(weight)
        )
        return routing_indices[routing_key % len(routing_indices)]

    def validate_upstreams(self, *, timeout: float = 30.0) -> None:
        for upstream in self.upstreams:
            request = urllib.request.Request(f"{upstream}/v1/models", method="GET")
            with urllib.request.urlopen(request, timeout=timeout) as response:
                value = json.loads(response.read())
            models = value.get("data", [])
            model_ids = {
                model_id
                for item in models
                if isinstance(item, Mapping) and isinstance(model_id := item.get("id"), str)
            }
            if self.served_model not in model_ids:
                raise ValueError(
                    "upstream model drift: "
                    f"upstream={upstream}, expected={self.served_model}, "
                    f"got={sorted(model_ids)}"
                )

    def _cache_key(self, rendered_prompt: str, targets: list[str]) -> str:
        payload = {
            "identity": self.identity,
            "rendered_prompt": rendered_prompt,
            "targets": targets,
        }
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(encoded).hexdigest()

    def _cache_path(self, key: str) -> Path:
        if self.cache_dir is None:
            raise RuntimeError("score cache is disabled")
        return self.cache_dir / key[:2] / f"{key}.json"

    def _cache_envelope(
        self,
        key: str,
        rendered_prompt: str,
        targets: list[str],
        row: Mapping[str, float],
    ) -> dict[str, Any]:
        normalized = {target: float(row[target]) for target in targets}
        if set(row) != set(targets) or not all(
            math.isfinite(value) for value in normalized.values()
        ):
            raise ValueError("invalid target logprobs")
        return {
            "schema_version": 1,
            "key": key,
            "identity": self.identity,
            "rendered_prompt_sha256": hashlib.sha256(rendered_prompt.encode()).hexdigest(),
            "targets": targets,
            "target_logprobs": normalized,
        }

    def _read_cache(
        self, key: str, rendered_prompt: str, targets: list[str]
    ) -> dict[str, float] | None:
        path = self._cache_path(key)
        try:
            envelope = json.loads(path.read_text())
        except FileNotFoundError:
            return None
        expected = self._cache_envelope(
            key,
            rendered_prompt,
            targets,
            envelope.get("target_logprobs", {}),
        )
        if envelope != expected:
            raise ValueError(f"score cache identity mismatch: {path}")
        return expected["target_logprobs"]

    def _publish_or_read_cache(
        self,
        key: str,
        rendered_prompt: str,
        targets: list[str],
        row: Mapping[str, float],
    ) -> dict[str, float]:
        envelope = self._cache_envelope(key, rendered_prompt, targets, row)
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        encoded = (
            json.dumps(envelope, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode()

        with self._cache_lock:
            cached = self._read_cache(key, rendered_prompt, targets)
            if cached is not None:
                return cached
            temporary: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                try:
                    os.link(temporary, path)
                except FileExistsError:
                    cached = self._read_cache(key, rendered_prompt, targets)
                    if cached is None:
                        raise RuntimeError("score cache publication race")
                    return cached
            finally:
                if temporary is not None:
                    temporary.unlink(missing_ok=True)
        return envelope["target_logprobs"]


def build_app(state: ProxyState) -> Any:
    from fastapi import FastAPI, HTTPException  # pyright: ignore[reportMissingImports]

    app = FastAPI()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/dynamic-rubric/identity")
    def identity() -> dict[str, Any]:
        return state.identity

    @app.get("/dynamic-rubric/routing")
    def routing() -> dict[str, Any]:
        return {
            "strategy": "adaptive-deterministic-length-balanced-with-failover",
            "upstream_count": len(state.upstreams),
            "upstream_weights": list(state.upstream_weights or ()),
            "effective_upstream_weights": list(state._effective_weights()),
            "request_affinity": state.request_affinity,
            "adaptive_upstream_index": state.adaptive_upstream_index,
            "adaptive_gpu_index": state.adaptive_gpu_index,
            "adaptive_low_utilization_weight": state.adaptive_low_utilization_weight,
            "adaptive_utilization_threshold": state.adaptive_utilization_threshold,
            "transient_failover": True,
            **state.runtime_metrics,
        }

    @app.post("/dynamic-rubric/score-targets")
    def score(request: dict[str, Any]) -> dict[str, Any]:
        try:
            rendered_prompts = request.get("rendered_prompts")
            targets = request.get("targets")
            upstream_index = request.get("routing_upstream_index")
            if (
                not isinstance(rendered_prompts, list)
                or not all(isinstance(item, str) for item in rendered_prompts)
                or not isinstance(targets, list)
                or not all(isinstance(item, str) for item in targets)
                or request.get("temperature", 0) != 0
                or request.get("thinking", False) is not False
                or (
                    upstream_index is not None
                    and (isinstance(upstream_index, bool) or not isinstance(upstream_index, int))
                )
            ):
                raise ValueError("invalid deterministic score request")
            rows = state.score(rendered_prompts, targets, upstream_index=upstream_index)
        except Exception as error:
            raise HTTPException(status_code=502, detail=str(error)) from error
        return {"target_logprobs": rows[0] if len(rows) == 1 else rows}

    return app


def main() -> None:
    from transformers import AutoTokenizer  # pyright: ignore[reportMissingImports]

    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", action="append", required=True)
    parser.add_argument("--upstream-weight", action="append", type=int)
    parser.add_argument("--request-affinity", action="store_true")
    parser.add_argument("--adaptive-upstream-index", type=int)
    parser.add_argument("--adaptive-gpu-index", type=int)
    parser.add_argument("--adaptive-low-utilization-weight", type=int)
    parser.add_argument("--adaptive-utilization-threshold", type=float, default=70.0)
    parser.add_argument("--upstream-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--managed-sleep-upstream-index", type=int)
    parser.add_argument("--managed-sleep-idle-seconds", type=float, default=3.0)
    parser.add_argument("--managed-sleep-timeout-seconds", type=float, default=300.0)
    parser.add_argument("--managed-sleep-level", type=int, choices=(1, 2), default=1)
    parser.add_argument("--managed-wake-gpu-index", type=int)
    parser.add_argument("--managed-wake-max-memory-used-mib", type=int)
    parser.add_argument("--managed-wake-poll-seconds", type=float, default=1.0)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--tokenizer-revision", required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8102)
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    state = ProxyState(
        upstream=args.upstream,
        served_model=args.served_model,
        model_revision=args.model_revision,
        tokenizer_revision=args.tokenizer_revision,
        tokenizer=tokenizer,
        upstream_weights=args.upstream_weight,
        request_affinity=args.request_affinity,
        cache_dir=args.cache_dir,
        adaptive_upstream_index=args.adaptive_upstream_index,
        adaptive_gpu_index=args.adaptive_gpu_index,
        adaptive_low_utilization_weight=args.adaptive_low_utilization_weight,
        managed_sleep_upstream_index=args.managed_sleep_upstream_index,
        managed_sleep_idle_seconds=args.managed_sleep_idle_seconds,
        managed_sleep_timeout_seconds=args.managed_sleep_timeout_seconds,
        managed_sleep_level=args.managed_sleep_level,
        managed_wake_gpu_index=args.managed_wake_gpu_index,
        managed_wake_max_memory_used_mib=args.managed_wake_max_memory_used_mib,
        managed_wake_poll_seconds=args.managed_wake_poll_seconds,
        adaptive_utilization_threshold=args.adaptive_utilization_threshold,
        upstream_timeout_seconds=args.upstream_timeout_seconds,
    )
    state.validate_upstreams()
    state.initialize_managed_upstream()
    import uvicorn  # pyright: ignore[reportMissingImports]

    uvicorn.run(build_app(state), host=args.host, port=args.port, workers=1)


if __name__ == "__main__":
    main()
