# ruff: noqa: E402 -- Direct runtime invocation needs the repository import paths.
from __future__ import annotations

import copy
from pathlib import Path
import sys
import threading
import time
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from tensordict import TensorDict
from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "environment" / "upstream" / "verl"))

from verl.protocol import DataProto
from verl.trainer.ppo.core_algos import (
    agg_loss, compute_grpo_outcome_advantage, compute_policy_loss_vanilla, kl_penalty,
)

from dynamic_rubric.training.logprob_prefetch import (
    clone_cpu_payload,
    prepare_inference_inputs,
    prepare_with_logprob_prefetch,
)


BALANCE_ORDER = torch.tensor([2, 0, 3, 1])


def _batch() -> DataProto:
    input_ids = torch.tensor(
        [
            [0, 0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8, 9],
            [10, 11, 12, 13, 14, 15],
            [0, 0, 0, 16, 17, 18],
        ],
        dtype=torch.long,
    )
    attention_mask = (input_ids != 0).to(torch.long)
    responses = input_ids[:, -3:].clone()
    response_mask = attention_mask[:, -3:].clone()
    position_ids = torch.arange(input_ids.shape[1]).repeat(input_ids.shape[0], 1)
    rollout_log_probs = -(responses.to(torch.float64) / 17.0)
    tensors = TensorDict(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "responses": responses,
            "response_mask": response_mask,
            "position_ids": position_ids,
            "rollout_log_probs": rollout_log_probs,
        },
        batch_size=[4],
    )
    return DataProto(
        batch=tensors,
        non_tensor_batch={
            "uid": np.array(["u0", "u1", "u2", "u3"], dtype=object),
            "prompt_occurrence_id": np.array(["p0", "p1", "p2", "p3"], dtype=object),
            "grpo_group_id": np.array(["g0", "g0", "g1", "g1"], dtype=object),
            "multi_modal_inputs": np.array([{}, {}, {}, {}], dtype=object),
        },
        meta_info={"temperature": 1.0},
    )


def _data_proto(**tensors: torch.Tensor) -> DataProto:
    first = next(iter(tensors.values()))
    return DataProto(batch=TensorDict(tensors, batch_size=[first.shape[0]]))


def _nested_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor):
        return isinstance(right, torch.Tensor) and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_nested_equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return type(left) is type(right) and len(left) == len(right) and all(
            _nested_equal(a, b) for a, b in zip(left, right)
        )
    return left == right


class FakeTrainer:
    def __init__(self, *, overlap_barrier: bool, reward_failure: bool = False,
                 old_failure: bool = False, ref_failure: bool = False):
        torch.manual_seed(41)
        self.actor = torch.nn.Linear(2, 1, bias=True, dtype=torch.float64)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.01)
        self.reference_weight = torch.tensor([0.031, -0.017], dtype=torch.float64)
        self.reference_bias = torch.tensor(0.23, dtype=torch.float64)
        self.config = SimpleNamespace(
            trainer=SimpleNamespace(balance_batch=True),
            actor_rollout_ref=SimpleNamespace(
                actor=SimpleNamespace(calculate_entropy=False, entropy_coeff=0.0)
            ),
            algorithm={"rollout_correction": {"bypass_mode": False}},
        )
        self.use_reference_policy = True
        self.overlap_barrier = overlap_barrier
        self.reward_failure = reward_failure
        self.old_failure = old_failure
        self.ref_failure = ref_failure
        self.started = threading.Barrier(2) if overlap_barrier else None
        self.events: list[tuple[str, float]] = []
        self._event_lock = threading.Lock()
        self.optimizer_steps = 0

    def _record(self, name: str) -> None:
        with self._event_lock:
            self.events.append((name, time.perf_counter()))

    def _balance_batch(self, batch: DataProto, metrics: dict) -> None:
        batch.reorder(BALANCE_ORDER)
        metrics["balance/permutation_applied"] = 1

    @staticmethod
    def _features(batch: DataProto) -> torch.Tensor:
        responses = batch.batch["responses"].to(torch.float64)
        response_positions = batch.batch["position_ids"][:, -responses.shape[1] :].to(torch.float64)
        return torch.stack((responses / 19.0, response_positions / 7.0), dim=-1)

    def policy_log_probs(self, batch: DataProto) -> torch.Tensor:
        return self.actor(self._features(batch)).squeeze(-1)

    def _compute_old_log_prob(self, batch: DataProto, *, calculate_entropy: bool):
        self._record("old_start")
        self.assertions_for_old(batch, calculate_entropy)
        if self.started is not None:
            self.started.wait(timeout=5)
            time.sleep(0.01)
        if self.old_failure:
            self._record("old_fail")
            raise RuntimeError("synthetic old-logprob failure")
        with torch.no_grad():
            old = self.policy_log_probs(batch).detach().clone()
        self._record("old_end")
        return _data_proto(old_log_probs=old), 0.125

    def assertions_for_old(self, batch: DataProto, calculate_entropy: bool) -> None:
        if calculate_entropy:
            raise AssertionError("prefetch must preserve entropy-disabled configuration")
        if batch.non_tensor_batch["uid"].tolist() != ["u2", "u0", "u3", "u1"]:
            raise AssertionError("old log-prob saw the wrong balanced row order")
        if "global_token_num" not in batch.meta_info or "images_seqlens" not in batch.meta_info:
            raise AssertionError("old log-prob saw incomplete trainer metadata")

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        self._record("ref_start")
        if self.ref_failure:
            raise RuntimeError("synthetic ref-logprob failure")
        if "old_log_probs" not in batch.batch:
            raise AssertionError("reference must see the serial old-logprob union")
        with torch.no_grad():
            ref = torch.einsum("bsf,f->bs", self._features(batch), self.reference_weight)
            ref = ref + self.reference_bias
        self._record("ref_end")
        return _data_proto(ref_log_prob=ref)

    def _prepare_online_rewards(self, batch: DataProto) -> DataProto:
        self._record("reward_start")
        if self.started is not None:
            self.started.wait(timeout=5)
            time.sleep(0.05)
        if self.reward_failure:
            self._record("reward_fail")
            raise RuntimeError("synthetic remote-reward failure")
        score_by_uid = {"u0": 0.2, "u1": 0.8, "u2": -0.1, "u3": 0.5}
        response_mask = batch.batch["response_mask"]
        scores = torch.zeros_like(response_mask, dtype=torch.float64)
        for row, uid in enumerate(batch.non_tensor_batch["uid"]):
            terminal = int(torch.where(response_mask[row].bool())[0][-1])
            scores[row, terminal] = score_by_uid[str(uid)]
        batch.batch["rm_scores"] = scores
        batch.non_tensor_batch["online_trace_ref"] = np.array(
            [f"trace:{uid}" for uid in batch.non_tensor_batch["uid"]], dtype=object
        )
        batch.meta_info.update(
            {
                "online_step_sealed": True,
                "online_optimizer_update_index": 32,
                "online_manifest_hash": "manifest-32",
            }
        )
        self._record("reward_end")
        return batch

    def update_once(self, batch: DataProto, old: DataProto, ref: DataProto):
        rewards = batch.batch["rm_scores"].sum(dim=-1)
        advantages, _ = compute_grpo_outcome_advantage(
            batch.batch["rm_scores"], batch.batch["response_mask"],
            batch.non_tensor_batch["grpo_group_id"], norm_adv_by_std_in_grpo=True,
        )
        current = self.policy_log_probs(batch)
        mask = batch.batch["response_mask"].to(torch.float64)
        old_values = old.batch["old_log_probs"]
        ref_values = ref.batch["ref_log_prob"]
        policy_term, _ = compute_policy_loss_vanilla(
            old_values.detach(), current, advantages, mask,
            config=OmegaConf.create({"clip_ratio": 0.2, "clip_ratio_low": None,
                                     "clip_ratio_high": None, "global_batch_info": {}}),
        )
        kl_term = 0.01 * agg_loss(
            kl_penalty(current, ref_values.detach(), "low_var_kl"), mask, "token-mean"
        )
        loss = policy_term + kl_term
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.optimizer_steps += 1
        return rewards.detach(), advantages.detach(), loss.detach()


def _serial_step(trainer: FakeTrainer, batch: DataProto):
    metrics: dict = {}
    batch = trainer._prepare_online_rewards(batch)
    prepare_inference_inputs(trainer, batch, metrics)
    old, old_mfu = trainer._compute_old_log_prob(batch, calculate_entropy=False)
    reference_input = clone_cpu_payload(batch).union(old)
    ref = trainer._compute_ref_log_prob(reference_input)
    return batch, old, old_mfu, ref, metrics


class LogProbPrefetchEquivalenceTest(unittest.TestCase):
    def test_nontrivial_balance_prefetch_matches_serial_training_step(self) -> None:
        serial = FakeTrainer(overlap_barrier=False)
        speculative = FakeTrainer(overlap_barrier=True)

        serial_batch, serial_old, serial_mfu, serial_ref, _ = _serial_step(serial, _batch())
        overlap_metrics: dict = {}
        prefetched_batch, prefetched = prepare_with_logprob_prefetch(
            speculative, _batch(), overlap_metrics
        )
        prepare_inference_inputs(speculative, prefetched_batch, {})
        prefetched.validate_inputs(prefetched_batch)

        self.assertEqual(prefetched_batch.non_tensor_batch["uid"].tolist(), ["u2", "u0", "u3", "u1"])
        self.assertTrue(torch.equal(serial_batch.batch["rm_scores"], prefetched_batch.batch["rm_scores"]))
        self.assertEqual(
            serial_batch.non_tensor_batch["online_trace_ref"].tolist(),
            prefetched_batch.non_tensor_batch["online_trace_ref"].tolist(),
        )
        self.assertTrue(prefetched.matches(serial_old, "old"))
        self.assertTrue(prefetched.matches(serial_ref, "ref"))
        self.assertEqual(serial_mfu, prefetched.old_mfu)

        serial_result = serial.update_once(serial_batch, serial_old, serial_ref)
        prefetch_result = speculative.update_once(prefetched_batch, prefetched.old, prefetched.ref)
        for expected, actual in zip(serial_result, prefetch_result):
            self.assertTrue(torch.equal(expected, actual))
        self.assertTrue(_nested_equal(serial.actor.state_dict(), speculative.actor.state_dict()))
        self.assertTrue(_nested_equal(serial.optimizer.state_dict(), speculative.optimizer.state_dict()))

        stamps = dict(speculative.events)
        self.assertLess(stamps["old_start"], stamps["old_end"])
        self.assertLess(stamps["old_end"], stamps["ref_start"])
        self.assertLess(stamps["ref_start"], stamps["ref_end"])
        self.assertLess(stamps["reward_start"], stamps["old_end"])
        self.assertLess(stamps["old_start"], stamps["reward_end"])
        self.assertIn("overlap/remote_reward_s", overlap_metrics)
        self.assertIn("overlap/logprob_s", overlap_metrics)

    def test_snapshot_isolated_and_input_output_drift_fails_closed(self) -> None:
        trainer = FakeTrainer(overlap_barrier=True)
        actual, prefetched = prepare_with_logprob_prefetch(trainer, _batch(), {})
        prepare_inference_inputs(trainer, actual, {})
        prefetched.validate_inputs(actual)

        actual_before = actual.batch["input_ids"].clone()
        prefetched.inputs.batch["input_ids"][0, -1] += 100
        self.assertTrue(torch.equal(actual.batch["input_ids"], actual_before))
        prefetched.inputs.batch["input_ids"][0, -1] -= 100

        numeric_drift = clone_cpu_payload(actual)
        numeric_drift.batch["input_ids"][0, -1] += 1
        with self.assertRaisesRegex(RuntimeError, "input/order mismatch: input_ids"):
            prefetched.validate_inputs(numeric_drift)

        order_drift = clone_cpu_payload(actual)
        order_drift.non_tensor_batch["uid"] = order_drift.non_tensor_batch["uid"][[1, 0, 2, 3]]
        with self.assertRaisesRegex(RuntimeError, "row identity mismatch: uid"):
            prefetched.validate_inputs(order_drift)

        metadata_drift = clone_cpu_payload(actual)
        metadata_drift.meta_info["global_token_num"][0] += 1
        with self.assertRaisesRegex(RuntimeError, "metadata mismatch: global_token_num"):
            prefetched.validate_inputs(metadata_drift)

        serial_old = clone_cpu_payload(prefetched.old)
        serial_old.batch["old_log_probs"][0, 0] += torch.finfo(torch.float64).eps
        self.assertFalse(prefetched.matches(serial_old, "old"))
        serial_ref = clone_cpu_payload(prefetched.ref)
        serial_ref.batch["ref_log_prob"][0, 0] += torch.finfo(torch.float64).eps
        self.assertFalse(prefetched.matches(serial_ref, "ref"))

    def test_remote_failure_drains_logprob_worker_before_optimizer(self) -> None:
        trainer = FakeTrainer(overlap_barrier=True, reward_failure=True)
        parameters_before = copy.deepcopy(trainer.actor.state_dict())
        with self.assertRaisesRegex(RuntimeError, "synthetic remote-reward failure"):
            prepare_with_logprob_prefetch(trainer, _batch(), {})

        self.assertIn("old_end", [name for name, _ in trainer.events])
        self.assertIn("ref_end", [name for name, _ in trainer.events])
        self.assertFalse(any(t.name.startswith("online-logprob") for t in threading.enumerate()))
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertEqual(len(trainer.optimizer.state), 0)
        self.assertTrue(_nested_equal(parameters_before, trainer.actor.state_dict()))

    def test_reference_failure_is_joined_and_blocks_optimizer(self) -> None:
        trainer = FakeTrainer(overlap_barrier=True, ref_failure=True)
        parameters_before = copy.deepcopy(trainer.actor.state_dict())
        with self.assertRaisesRegex(RuntimeError, "synthetic ref-logprob failure"):
            prepare_with_logprob_prefetch(trainer, _batch(), {})
        self.assertIn("old_end", [name for name, _ in trainer.events])
        self.assertFalse(any(t.name.startswith("online-logprob") for t in threading.enumerate()))
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertEqual(len(trainer.optimizer.state), 0)
        self.assertTrue(_nested_equal(parameters_before, trainer.actor.state_dict()))

    def test_local_failure_is_joined_and_blocks_optimizer(self) -> None:
        trainer = FakeTrainer(overlap_barrier=True, old_failure=True)
        parameters_before = copy.deepcopy(trainer.actor.state_dict())
        with self.assertRaisesRegex(RuntimeError, "synthetic old-logprob failure"):
            prepare_with_logprob_prefetch(trainer, _batch(), {})

        names = [name for name, _ in trainer.events]
        self.assertIn("reward_end", names)
        self.assertIn("old_fail", names)
        self.assertNotIn("ref_start", names)
        self.assertFalse(any(t.name.startswith("online-logprob") for t in threading.enumerate()))
        self.assertEqual(trainer.optimizer_steps, 0)
        self.assertEqual(len(trainer.optimizer.state), 0)
        self.assertTrue(_nested_equal(parameters_before, trainer.actor.state_dict()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
