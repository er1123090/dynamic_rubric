import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
SPEC = importlib.util.spec_from_file_location(
    "probe_kl_reanalysis",
    Path(__file__).resolve().parents[2] / "scripts/phase1/reanalyze_probe_kl34.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
if HAS_NUMPY:
    SPEC.loader.exec_module(MODULE)


@unittest.skipUnless(HAS_NUMPY, "offline analysis requires numpy")
class ProbeKlTests(unittest.TestCase):
    def test_direction_is_current_to_stale(self):
        k1, k3, clipped = MODULE.token_estimators([math.log(0.4)], [math.log(0.2)])
        self.assertAlmostEqual(k1[0], math.log(2))
        self.assertAlmostEqual(k3[0], math.log(2) - 0.5)
        self.assertEqual(clipped, 0)

    def test_identical_policies(self):
        k1, k3, clipped = MODULE.token_estimators([-1, -2], [-1, -2])
        self.assertTrue(all(k1 == 0))
        self.assertTrue(all(k3 == 0))
        self.assertEqual(clipped, 0)

    def test_clipping_only_affects_k3(self):
        k1, k3, clipped = MODULE.token_estimators([-31], [-1])
        self.assertEqual(k1[0], -30)
        self.assertAlmostEqual(k3[0], math.expm1(20) - 20)
        self.assertEqual(clipped, 1)

    def test_jsonl_keeps_unicode_line_separator_inside_text(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "response.jsonl"
            expected = {"text": "left\u2028right"}
            path.write_text(json.dumps(expected, ensure_ascii=False) + "\n")
            self.assertEqual(MODULE.read_rows(path), [expected])


if __name__ == "__main__":
    unittest.main()
