"""Cumulative authorization and legal held-out sampling guards."""

from copy import deepcopy
from dataclasses import asdict, replace
import unittest

import numpy as np

from ra_lsbo.mechanism_system import METHODS, SystemSettings
from tools.run_system_validation_v5 import CAP, QUOTAS, TARGET, validate_plan


def fixture_plan():
    carried = [dict(index=i, candidate={"id": i},
                    risk={"hidden_labels": {TARGET: 1 if i in (5, 19) else 0}})
               for i in range(1, 129)]
    for row in carried:
        row["candidate"] = dict(continuous={"x": row["index"]/200},
                                discrete={"architecture": "old"})
    probes = [dict(index=r["index"], candidate=r["candidate"]) for r in carried]
    for i in range(129, 141):
        probes.append(dict(index=i, candidate=dict(continuous={"x": i/200},
            discrete={"architecture": "new"}), anchor_index=5, radius=.002))
    return dict(schema="ra_system_validation_v5", operator_approved=True,
        complete_plan_mda_cap=500, probe_call_cap_including_carried=140,
        fresh_calibration_calls=72, calibration_quotas=QUOTAS,
        methods=list(METHODS), seeds=[31, 32, 33],
        settings=asdict(replace(SystemSettings(), calibration_window=72)),
        parallel_workers=4, carried_mda_calls=128, remaining_mda_cap=372,
        prior_call_cost_by_version={"v1": 4, "v2": 48, "v3": 16, "v4": 60},
        carried_probes=carried, probes=probes)


class ValidationV5Tests(unittest.TestCase):
    def test_all_version_calls_cannot_be_reset_or_expand_per_method(self):
        plan = fixture_plan()
        validate_plan(plan)
        self.assertEqual(140+72+8*3*12, CAP)
        for key, value in (("complete_plan_mda_cap", 628), ("carried_mda_calls", 0),
                           ("remaining_mda_cap", 500), ("parallel_workers", 8)):
            changed = deepcopy(plan)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_plan(changed)
        changed = deepcopy(plan)
        changed["settings"]["calls"] = 13
        with self.assertRaises(ValueError):
            validate_plan(changed)

    def test_architecture_change_requires_observed_failure_and_single_factor(self):
        plan = fixture_plan()
        for change in ("anchor", "same", "multiple"):
            altered = deepcopy(plan)
            item = altered["probes"][-1]
            if change == "anchor":
                item["anchor_index"] = 6
            elif change == "same":
                item["candidate"]["discrete"] = {"architecture": "old"}
            else:
                item["candidate"]["discrete"] = {"architecture": "new", "other": "new"}
                altered["carried_probes"][4]["candidate"]["discrete"]["other"] = "old"
            with self.assertRaises((ValueError, KeyError)):
                validate_plan(altered)


if __name__ == "__main__":
    unittest.main()
