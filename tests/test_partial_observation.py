from copy import deepcopy
import unittest
from unittest.mock import patch

from frozen_lift_fixture import HistoricalLiftSource

from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.partial_observation import (
    LIFT_SOURCE, LIFT_VERSION, MESSAGE, RULE_ID, normalize_with_partial_lift)


class PartialLiftObservationTests(unittest.TestCase):
    required = ("soc", "stall_speed", "structure")

    def raw(self):
        return {
            "status": "evaluation_error:ValueError", "objective": None, "solver_success": False,
            "constraints": {"solver": 1.}, "metrics": {
                "mda_evaluated": True,
                "failure_stage": "procedure.post_process", "failure_message": MESSAGE,
                "failure_origin": {"file": str(LIFT_SOURCE), "function": "corrected_limit"},
                "postprocess_version": "nasa_charge_soc_energy_audit_fail_closed_v1",
                "energy_audit_status": "passed", "mission_depletion_Wh": 49431.89097596289,
                "mission_solver_segments": [{"segment": "mission", "converged": True}],
                "simplified_powered_lift": {
                    "version": LIFT_VERSION, "status": "failed", "failure": MESSAGE,
                    "dry_clmax": 1.8, "dry_stall_speed_mps": 26., "delta_cl": -2.4}}}

    def test_audited_energy_retained_but_no_constraint_or_feasibility_invented(self):
        raw = self.raw()
        original = deepcopy(raw)
        with patch("ra_lsbo.partial_observation.LIFT_SOURCE", HistoricalLiftSource()):
            value = normalize_with_partial_lift(raw, self.required)
        self.assertEqual(raw, original)
        self.assertEqual(value.raw_status, "evaluation_error:ValueError")
        self.assertEqual(value.state, "incomplete")
        self.assertEqual(value.objective, 49431.89097596289)
        self.assertTrue(value.objective_observed)
        self.assertIsNone(value.feasible_label)
        self.assertTrue(all(v is None for v in value.constraints.values()))
        self.assertFalse(any(value.constraint_observed.values()))
        self.assertTrue(all(v == 0 for v in value.hidden_labels.values()))
        self.assertEqual(value.evidence["partial_observation_proof"]["rule"], RULE_ID)
        self.assertEqual(normalize_evaluation(raw, self.required).state, "unknown")

    def test_changed_physical_source_cannot_recover_historical_partial_output(self):
        raw = self.raw()
        with patch("ra_lsbo.partial_observation.LIFT_SOURCE", HistoricalLiftSource(b"changed")):
            value = normalize_with_partial_lift(raw, self.required)
        self.assertEqual(value.state, "unknown")
        self.assertFalse(value.objective_observed)

    def test_unrelated_error_missing_audit_unconverged_or_nonfinite_energy_not_promoted(self):
        variants = []
        for key, value in (("failure_stage", "procedure.missions.design_mission"),
                           ("failure_message", "ValueError: unrelated"),
                           ("energy_audit_status", "not_run"), ("mission_depletion_Wh", None),
                           ("mission_depletion_Wh", float("nan")),
                           ("mission_solver_segments", []),
                           ("mission_solver_segments", [{"converged": False}])):
            raw = self.raw()
            raw["metrics"][key] = value
            variants.append(raw)
        raw = self.raw()
        raw["metrics"]["failure_origin"]["function"] = "arbitrary_value_error"
        variants.append(raw)
        raw = self.raw()
        raw["metrics"]["simplified_powered_lift"]["delta_cl"] = 0.
        variants.append(raw)
        for raw in variants:
            self.assertEqual(
                normalize_with_partial_lift(raw, self.required).to_dict(),
                normalize_evaluation(raw, self.required).to_dict())

    def test_known_mission_failure_keeps_original_censoring_and_does_not_become_partial(self):
        raw = self.raw()
        raw["status"] = "evaluation_error:DomainSolveError"
        raw["metrics"]["failure_stage"] = "procedure.missions.design_mission"
        value = normalize_with_partial_lift(raw, self.required)
        self.assertEqual(value.to_dict(), normalize_evaluation(raw, self.required).to_dict())
        self.assertEqual(value.state, "hidden_failure")
        self.assertIsNone(value.objective)
