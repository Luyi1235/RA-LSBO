from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from ra_lsbo.budget import digest
from ra_lsbo.mechanism_system import FiveModeProblem, METHODS, SystemSettings
from tools.run_system_validation_v3 import RADII, TARGET, freeze_neighborhoods, validate_plan


BASE = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"


class TargetedDataPlanTests(unittest.TestCase):
    def fixture(self):
        problem = FiveModeProblem(BASE)
        problem.problem = SimpleNamespace(precheck=lambda c: problem.precheck(c.to_dict()))
        carried = []
        for x, y in ((.2, .25), (.22, .28), (.7, .6), (.65, .65)):
            candidate = problem.space.default().to_dict()
            candidate["continuous"].update(x=x, y=y, spacing=.1)
            candidate["discrete"].update(layout="distributed", count=2)
            candidate = problem.canonical(candidate)
            carried.append({"index": len(carried)+1, "candidate": candidate, "risk": problem.evaluate(candidate)})
        old_probes = [{"index": r["index"], "candidate": r["candidate"], "source": "controlled_test"}
                      for r in carried]
        probes = freeze_neighborhoods(problem, [], carried, old_probes)
        plan = {
            "schema": "ra_system_validation_v3", "operator_approved": True, "complete_plan_mda_cap": 480,
            "probe_call_cap_including_carried": 120, "fresh_calibration_calls": 72,
            "methods": list(METHODS), "seeds": [31, 32, 33], "parallel_workers": 4,
            "settings": asdict(replace(SystemSettings(), calibration_window=72)),
            "target_mode": TARGET, "target_positive": 20, "target_negative": 20,
            "neighborhood_radii": list(RADII), "carried_mda_calls": 4, "remaining_mda_cap": 476,
            "carried_probes": carried, "probes": probes}
        return problem, plan

    def test_freezes_legal_unique_bounded_neighbors_of_verified_same_configuration(self):
        problem, plan = self.fixture()
        validate_plan(plan)
        self.assertEqual(len({digest(p["candidate"]) for p in plan["probes"]}), 120)
        anchors = {r["index"]: r for r in plan["carried_probes"]}
        for item in plan["probes"][4:]:
            anchor = anchors[item["anchor_index"]]
            self.assertEqual(anchor["risk"]["hidden_labels"][TARGET], 1)
            self.assertEqual(item["candidate"]["discrete"], anchor["candidate"]["discrete"])
            candidate = problem.candidate_class(**item["candidate"])
            center = problem.candidate_class(**anchor["candidate"])
            self.assertLessEqual(float(np.max(np.abs(
                problem.space.unit(candidate)-problem.space.unit(center)))), item["radius"]+1e-10)
            self.assertTrue(all(v <= 0 for v in problem.precheck(item["candidate"]).values()))

    def test_complete_scope_counts_prior_versions_and_cannot_expand_or_lower_training_target(self):
        _, plan = self.fixture()
        self.assertEqual(120+72+8*3*12, plan["complete_plan_mda_cap"])
        for key, value in (("complete_plan_mda_cap", 500), ("remaining_mda_cap", 480),
                           ("fresh_calibration_calls", 0), ("target_positive", 3),
                           ("seeds", [31, 32, 34]), ("operator_approved", False)):
            altered = deepcopy(plan)
            altered[key] = value
            with self.assertRaises(ValueError):
                validate_plan(altered)

    def test_unobserved_anchor_changed_discrete_and_duplicate_candidate_rejected(self):
        _, plan = self.fixture()
        altered = deepcopy(plan)
        altered["carried_probes"][0]["risk"]["hidden_labels"][TARGET] = None
        altered["carried_probes"][1]["risk"]["hidden_labels"][TARGET] = None
        with self.assertRaisesRegex(ValueError, "anchor"):
            validate_plan(altered)
        altered = deepcopy(plan)
        altered["probes"][4]["candidate"]["discrete"]["count"] = 4
        with self.assertRaisesRegex(ValueError, "configuration"):
            validate_plan(altered)
        altered = deepcopy(plan)
        altered["probes"][5]["candidate"] = deepcopy(altered["probes"][4]["candidate"])
        with self.assertRaisesRegex(ValueError, "Duplicated"):
            validate_plan(altered)


if __name__ == "__main__":
    unittest.main()
