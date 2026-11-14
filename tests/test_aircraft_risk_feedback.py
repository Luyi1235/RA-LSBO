from dataclasses import asdict
import unittest

from aircraft_risk_feedback import constraint_feedback, feedback_columns, feedback_receipt
from aircraft_risk_protocol import plan
from ra_lsbo.risk_core.observations import OutputObservation


class AircraftRiskFeedbackTests(unittest.TestCase):
    def test_only_signed_intervals_are_coarsened_to_zero(self):
        violated = OutputObservation("interval", "violation", lower=.04)
        satisfied = OutputObservation("interval", "satisfied", upper=-.03)
        ambiguous = OutputObservation("interval", "unknown_sign", lower=-.02)
        zero = OutputObservation("interval", "zero_inclusive", lower=0.)
        exact = OutputObservation("exact", "structure", value=.2)
        self.assertEqual(constraint_feedback(violated, "bounds"), violated)
        limit = constraint_feedback(violated, "sign")
        self.assertEqual(limit.lower, 0.)
        self.assertFalse(limit.lower_inclusive)
        self.assertIs(limit.constraint_satisfaction(), False)
        self.assertEqual(constraint_feedback(satisfied, "sign").upper, 0.)
        self.assertEqual(constraint_feedback(ambiguous, "sign").kind, "missing")
        self.assertEqual(constraint_feedback(zero, "sign").kind, "missing")
        self.assertEqual(constraint_feedback(violated, "exact").kind, "missing")
        for mode in ("bounds", "sign", "exact"):
            self.assertIs(constraint_feedback(exact, mode), exact)
            self.assertEqual(constraint_feedback(exact, mode, trusted=False).kind, "missing")

    def test_receipt_distinguishes_magnitude_from_same_sign(self):
        source = {"observation": {"status": "completed", "constraints": [
            ["soc", asdict(OutputObservation("interval", "abort:soc", lower=.02))],
        ]}, "diagnostics": {"mission_completed": False}}
        sign = feedback_receipt(feedback_columns([source], ["soc"], "sign"), "sign")
        bounds = feedback_receipt(feedback_columns([source], ["soc"], "bounds"), "bounds")
        self.assertEqual(sign["kinds"], bounds["kinds"])
        self.assertNotEqual(sign["identity"], bounds["identity"])
        self.assertEqual(source["observation"]["constraints"][0][1]["lower"], .02)

    def test_profiles_match_across_wind(self):
        nominal, headwind, smoke = (plan("feedback-"+name) for name in ("nominal", "headwind", "smoke"))
        self.assertEqual(nominal["seeds"], headwind["seeds"])
        self.assertEqual(nominal["budget"], {"initial": 64, "online": 1024, "heldout": 512})
        self.assertEqual(nominal["budget"], headwind["budget"])
        self.assertEqual(sum(smoke["budget"].values()), 32)
