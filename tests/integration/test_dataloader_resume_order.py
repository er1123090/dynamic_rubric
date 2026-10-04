"""CPU-only real torchdata regression; runnable with the veRL runtime Python."""
# ruff: noqa: E402 -- Direct runtime invocation needs the repository src path.
from copy import deepcopy
from pathlib import Path
import sys
import unittest

import torch
from torchdata.stateful_dataloader import StatefulDataLoader
from torchdata.stateful_dataloader.sampler import RandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from dynamic_rubric.training.dataloader_resume import restore_online_dataloader_state


def loader():
    dataset = list(range(1500))
    return StatefulDataLoader(
        dataset, sampler=RandomSampler(dataset, generator=torch.Generator().manual_seed(11)),
        batch_size=96, drop_last=False, num_workers=0,
    )


class DataloaderResumeTest(unittest.TestCase):
    def test_initial_checkpoint_preserves_first_epoch(self):
        initial = loader()
        state = deepcopy(initial.state_dict())
        expected = [batch.tolist() for batch in initial]
        resumed = loader()
        restore_online_dataloader_state(resumed, state, global_step=0)
        self.assertEqual([batch.tolist() for batch in resumed], expected)

    def test_epoch_boundary_preserves_all_next_epoch_batches(self):
        uninterrupted = loader()
        list(uninterrupted)  # epoch one
        iterator = iter(uninterrupted)
        for _ in range(16):
            next(iterator)
        boundary = deepcopy(uninterrupted.state_dict())
        with self.assertRaises(StopIteration):
            next(iterator)
        expected = [batch.tolist() for batch in uninterrupted]

        resumed = loader()
        restore_online_dataloader_state(resumed, boundary, global_step=32)
        self.assertEqual([batch.tolist() for batch in resumed], expected)
        self.assertNotEqual(next(iter(loader())).tolist(), expected[0])
        self.assertEqual([len(batch) for batch in expected], [96] * 15 + [60])
        self.assertEqual(len(set(index for batch in expected for index in batch)), 1500)

    def test_mid_epoch_restore_keeps_remaining_batches(self):
        uninterrupted = loader()
        iterator = iter(uninterrupted)
        for _ in range(6):
            next(iterator)
        state = deepcopy(uninterrupted.state_dict())
        expected = [batch.tolist() for batch in iterator]
        resumed = loader()
        restore_online_dataloader_state(resumed, state, global_step=6)
        self.assertEqual([batch.tolist() for batch in resumed], expected)

    def test_wrong_boundary_state_fails_before_loading(self):
        original = loader()
        next(iter(original))
        state = original.state_dict()
        with self.assertRaisesRegex(RuntimeError, "final-batch"):
            restore_online_dataloader_state(loader(), state, global_step=32)


if __name__ == "__main__":
    unittest.main(verbosity=2)
