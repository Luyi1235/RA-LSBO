from copy import deepcopy
from dataclasses import asdict, replace
import unittest

from ra_lsbo.mechanism_system import METHODS, SystemSettings
from tools.run_system_validation_v2 import validate_plan


class FullPlanScopeTests(unittest.TestCase):
    def plan(self):
        probes = [{"index": i+1, "candidate": {"continuous": {"x": i/120}, "discrete": {}}}
                  for i in range(120)]
        return {
            "schema": "ra_system_validation_v2", "operator_approved": True,
            "complete_plan_mda_cap": 480, "probe_call_cap_including_carried": 120,
            "fresh_calibration_calls": 72, "methods": list(METHODS), "seeds": [31, 32, 33],
            "settings": asdict(replace(SystemSettings(), calibration_window=72)),
            "parallel_workers": 4, "carried_mda_calls": 4, "remaining_mda_cap": 476,
            "probes": probes, "carried_probes": deepcopy(probes[:4])}

    def test_complete_plan_counts_carried_calls_and_fresh_calibration(self):
        plan = self.plan()
        validate_plan(plan)
        self.assertEqual(120+72+len(plan["methods"])*len(plan["seeds"])*plan["settings"]["calls"], 480)
        self.assertEqual(plan["remaining_mda_cap"]+plan["carried_mda_calls"], 480)

    def test_expanded_scope_missing_calibration_and_changed_repetition_rejected(self):
        for field, value in (("complete_plan_mda_cap", 504), ("fresh_calibration_calls", 0),
                             ("remaining_mda_cap", 480), ("seeds", [31, 32, 34]),
                             ("operator_approved", False)):
            plan = self.plan()
            plan[field] = value
            with self.assertRaises(ValueError):
                validate_plan(plan)

    def test_copied_or_changed_carried_candidate_rejected(self):
        plan = self.plan()
        plan["probes"][90]["candidate"] = deepcopy(plan["probes"][0]["candidate"])
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            validate_plan(plan)
        plan = self.plan()
        plan["carried_probes"][0]["candidate"]["continuous"]["x"] = .55
        with self.assertRaisesRegex(ValueError, "Carried"):
            validate_plan(plan)


if __name__ == "__main__":
    unittest.main()
