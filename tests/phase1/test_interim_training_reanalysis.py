"""CPU analysis helpers: prompt-paired resampling and pooled denominators."""

import importlib.util
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts/phase1/analyze_interim_training.py"
SPEC = importlib.util.spec_from_file_location("interim_training_reanalysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
HAS_SCIENTIFIC_STACK = all(importlib.util.find_spec(name) for name in ("numpy", "matplotlib"))
if HAS_SCIENTIFIC_STACK:
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(HAS_SCIENTIFIC_STACK, "analysis requires numpy and matplotlib")
class ReanalysisTests(unittest.TestCase):
    def test_constant_paired_difference(self):
        self.assertEqual(MODULE.paired_ci([1, 1, 1], replicates=201), [1, 1, 1])

    def test_seed_is_reproducible(self):
        self.assertEqual(
            MODULE.paired_ci([-1, 0, 1], replicates=401),
            MODULE.paired_ci([-1, 0, 1], replicates=401),
        )

    def test_invalid_bootstrap(self):
        for values, count in (([], 20), ([1], 0)):
            with self.assertRaises(ValueError):
                MODULE.paired_ci(values, replicates=count)

    def test_pooled_criteria_not_mean_of_ratios(self):
        rows = [
            {
                "fresh_effective_criterion_count": 1,
                "fresh_saturated_criterion_count": 0,
                "fresh_dead_criterion_count": 0,
            },
            {
                "fresh_effective_criterion_count": 0,
                "fresh_saturated_criterion_count": 9,
                "fresh_dead_criterion_count": 0,
            },
        ]
        result = MODULE.pooled_criteria(rows, "fresh")
        self.assertEqual(result["count"], 10)
        self.assertEqual(result["effective_ratio"], 0.1)

    def test_no_criteria_is_undefined(self):
        self.assertIsNone(MODULE.pooled_criteria([], "fresh")["effective_ratio"])


if __name__ == "__main__":
    unittest.main()
