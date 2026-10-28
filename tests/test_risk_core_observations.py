from dataclasses import replace
import unittest

from ra_lsbo.risk_core import (
    OutputObservation, RunObservation, completion_rows, output_rows)


class RiskCoreObservationTests(unittest.TestCase):
    def run_observation(self):
        return RunObservation(
            design_id="design-a", contract_id="physics-a", evidence_id="raw-a",
            status="completed", objective=OutputObservation("exact", "energy-a", value=42.),
            constraints=(
                ("energy", OutputObservation("exact", "margin-a", value=-1.)),
                ("stall", OutputObservation("missing", "stall-a", reason="not_returned")),
            ), passed_stages=("mission",))

    def test_valid_objective_survives_known_later_failure(self):
        run = replace(self.run_observation(), status="hidden_failure",
                      failed_stage="postprocess", failure_cause="invalid_lift_output")
        self.assertEqual(output_rows([run], "energy", role="objective")[0]["value"], 42.)
        self.assertEqual(completion_rows([run])[0]["label"], 0)
        self.assertEqual(output_rows([run], "stall"), [])

    def test_inequality_evidence_is_not_exported_as_numeric_pseudo_target(self):
        bound = OutputObservation("interval", "proven-bound", lower=0., lower_inclusive=False)
        run = replace(self.run_observation(), constraints=(("stall", bound),))
        row = output_rows([run], "stall")[0]
        self.assertEqual(row["kind"], "interval")
        self.assertIsNone(row["value"])
        self.assertFalse(bound.constraint_satisfaction())

    def test_known_hidden_failure_does_not_create_missing_constraint_bound(self):
        run = replace(self.run_observation(), status="hidden_failure",
                      failed_stage="solver", failure_cause="nonconvergence")
        self.assertEqual(output_rows([run], "stall"), [])
        self.assertEqual(completion_rows([run])[0]["passed_stages"], ("mission",))

    def test_unknown_and_infrastructure_do_not_become_negative_labels(self):
        for status in ("unknown", "infrastructure_error"):
            run = replace(self.run_observation(), status=status)
            self.assertEqual(completion_rows([run]), [])
            self.assertEqual(output_rows([run], "energy"), [])
            self.assertEqual(output_rows([run], "energy", role="objective"), [])

    def test_interval_boundary_and_crossing_preserve_unknown(self):
        self.assertIsNone(OutputObservation("interval", "a", lower=0.).constraint_satisfaction())
        self.assertIsNone(
            OutputObservation("interval", "b", lower=-1., upper=1.).constraint_satisfaction())
        self.assertTrue(OutputObservation("interval", "c", upper=0.).constraint_satisfaction())

    def test_different_physics_cannot_share_a_training_batch(self):
        first = self.run_observation()
        second = replace(first, contract_id="different-physics")
        with self.assertRaises(ValueError):
            output_rows([first, second], "energy")
        with self.assertRaises(ValueError):
            completion_rows([first, second])

    def test_invalid_evidence_and_stage_contradictions_are_rejected(self):
        invalid = [
            {"kind": "exact", "evidence_id": "a", "value": float("nan")},
            {"kind": "exact", "evidence_id": "a", "value": True},
            {"kind": "interval", "evidence_id": "a"},
            {"kind": "interval", "evidence_id": "a", "lower": 1., "upper": 1.},
            {"kind": "missing", "evidence_id": "a", "reason": "failed", "value": 10.},
        ]
        for arguments in invalid:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                OutputObservation(**arguments)
        with self.assertRaises(ValueError):
            replace(self.run_observation(), status="hidden_failure",
                    failed_stage="mission", failure_cause="contradiction")
        with self.assertRaises(ValueError):
            replace(self.run_observation(), constraints=(
                ("same", OutputObservation("exact", "a", value=-1.)),
                ("same", OutputObservation("exact", "b", value=-2.))))


if __name__ == "__main__":
    unittest.main()
