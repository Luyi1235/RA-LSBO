"""Safety tests for the new, separate interrupted-SOC postmortem tool."""
import json
import math
import unittest

import numpy as np

from tools.diagnose_failed_soc_prefix import sanitize


class SocDiagnosticSafetyTests(unittest.TestCase):
    def test_nonfinite_numbers_do_not_crash_json_or_turn_into_zero(self):
        payload = {"nan": float("nan"), "pos": np.float64(float("inf")),
                   "neg": -float("inf"), "nested": np.asarray([1.2, -np.inf]),
                   "none": None}
        checked = sanitize(payload)
        blob = json.dumps(checked, allow_nan=False)
        result = json.loads(blob)
        self.assertEqual(result["nan"], {"nonfinite": "nan"})
        self.assertEqual(result["pos"], {"nonfinite": "+inf"})
        self.assertEqual(result["neg"], {"nonfinite": "-inf"})
        self.assertEqual(result["nested"][1], {"nonfinite": "-inf"})
        self.assertEqual(result["nested"][0], 1.2)
        self.assertIsNone(result["none"])

    def test_normal_finite_records_preserve_values(self):
        original = {"status": "ok", "value": np.float64(-1.5),
                    "counts": {"exact": np.int64(17)}, "chosen": [0.05, 1.5]}
        parsed = json.loads(json.dumps(sanitize(original), allow_nan=False))
        self.assertEqual(parsed["counts"]["exact"], 17)
        self.assertEqual(parsed["value"], -1.5)
        self.assertEqual(parsed["chosen"], [0.05, 1.5])

    def test_unrecognized_object_refused_instead_of_silently_repr(self):
        with self.assertRaises(TypeError):
            sanitize({"not_safe": object()})


if __name__ == "__main__":
    unittest.main()
