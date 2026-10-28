from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from frozen_lift_fixture import HistoricalLiftSource

from ra_lsbo.audit import load_json
from ra_lsbo.partial_observation import normalize_with_partial_lift
from tools.run_system_validation_v4 import validate_plan


class V4ContinuationTests(unittest.TestCase):
    def test_actual_audited_response_is_partial_not_feasible(self):
        root = Path(__file__).resolve().parents[1]
        source = root/"runs/system_validation_v3_20261020T173826Z_f5a31950"
        if not source.exists():
            self.skipTest("Local immutable campaign not distributed")
        rows = [load_json(line) for line in (source/"probe_results.jsonl").read_text(encoding="utf-8").splitlines()]
        original = rows[65]
        raw = load_json((Path(original["risk"]["evidence"]["worker_dir"])/"response.json").read_bytes())
        required = tuple(original["risk"]["constraints"])
        with patch("ra_lsbo.partial_observation.LIFT_SOURCE", HistoricalLiftSource()):
            value = normalize_with_partial_lift(raw, required)
        self.assertEqual(value.state, "incomplete")
        self.assertEqual(value.objective, 49431.89097596289)
        self.assertIsNone(value.feasible_label)
        self.assertEqual(value.evidence["partial_observation_proof"]["validated_mission_segments"], 11)
        self.assertEqual(original["risk"]["state"], "unknown")

    def test_old_plan_cannot_execute_under_new_observation_contract(self):
        root = Path(__file__).resolve().parents[1]
        source = root/"runs/system_validation_v3_20261020T173826Z_f5a31950/plan.json"
        if not source.exists():
            self.skipTest("Local immutable campaign not distributed")
        plan = load_json(source.read_bytes())
        with self.assertRaises(ValueError):
            validate_plan(plan)
        changed = deepcopy(plan)
        changed["schema"] = "ra_system_validation_v4"
        with self.assertRaises(ValueError):
            validate_plan(changed)
