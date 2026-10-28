import importlib.util
import unittest

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    from tools.report_boundary_validation import render_report


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class BoundaryReportTests(unittest.TestCase):
    def test_early_unknown_has_no_success_or_calibration_claim(self):
        result = {
            "status": "stopped_evaluator_error", "reason": "timeout", "plan_sha256": "plan",
            "contract_id": "contract", "ledger": {
                "charged_calls": 1, "settled_calls": 1, "unresolved_started_calls": 0},
            "results": [{"index": 1, "phase": "calibration", "risk": {
                "state": "infrastructure_error", "feasible_label": None,
                "constraints": {"g": None}, "constraint_observed": {"g": False}}}]}
        audit = {"raw_responses_matched": 0, "unavailable_error_responses": [1],
                 "frozen_sources_verified": 176}
        text = render_report(result, audit)
        self.assertIn("unknown labels: 1; feasible: 0; known nonfeasible: 0", text)
        self.assertIn("| calibration | 1 | 0 | 0 | 1 | 59 |", text)
        self.assertIn("| test | 0 | 0 | 0 | 0 | 60 |", text)
        self.assertIn("No claim of improved calibration", text)

    def test_precheck_rejection_is_not_a_returned_evaluation(self):
        result = {
            "status": "stopped_by_policy", "reason": "final_precheck_changed",
            "plan_sha256": "plan", "contract_id": "contract", "ledger": {
                "charged_calls": 0, "settled_calls": 0, "unresolved_started_calls": 0},
            "results": [{"index": 1, "phase": "calibration", "risk": None}]}
        audit = {"raw_responses_matched": 0, "unavailable_error_responses": [],
                 "frozen_sources_verified": 176}
        text = render_report(result, audit)
        self.assertIn("Returned evaluations: 0", text)
        self.assertIn("| calibration | 0 | 0 | 0 | 0 | 60 |", text)


if __name__ == "__main__":
    unittest.main()
