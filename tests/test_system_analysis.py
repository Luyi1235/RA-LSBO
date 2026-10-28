from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ra_lsbo.mechanism_system import METHODS
from tools.analyze_system_validation import compare_trials, trial_metrics


class ScientificComparisonTests(unittest.TestCase):
    def trials(self):
        full = {
            "method": "ra_full", "seed": 31, "evaluations": 2, "feasible_evaluations": 2,
            "initial_best": 10., "best_feasible_objective": 6.,
            "improvement_percent": 40., "first_improvement_call": 1,
            "status": "complete", "states": {"feasible": 2},
            "curve": [{"calls": 0, "best": 10.}, {"calls": 1, "best": 8.},
                      {"calls": 2, "best": 6.}]}
        return [dict(deepcopy(full), method=method) for method in METHODS]

    def test_unequal_cost_uses_actual_common_prefix_not_two_endpoints(self):
        trials = self.trials()
        trials[1].update(evaluations=1, feasible_evaluations=1,
                         best_feasible_objective=9., improvement_percent=10.,
                         curve=[{"calls": 0, "best": 10.}, {"calls": 1, "best": 9.}])
        comparison = compare_trials(trials)["paired_common_actual_calls"][0]
        self.assertEqual(comparison["common_actual_calls"], 1)
        self.assertEqual(comparison["full_best"], 8.)
        self.assertEqual(comparison["other_minus_full"], 1.)

    def test_zero_call_and_early_stop_not_dropped_or_extrapolated(self):
        trials = self.trials()
        trials[1].update(evaluations=0, feasible_evaluations=0, states={},
                         status="stopped_model_error", first_improvement_call=None,
                         best_feasible_objective=10., improvement_percent=0.,
                         curve=[{"calls": 0, "best": 10.}])
        result = compare_trials(trials)
        comparison = result["paired_common_actual_calls"][0]
        self.assertEqual(comparison["common_actual_calls"], 0)
        self.assertFalse(comparison["has_new_call_evidence"])
        self.assertEqual(comparison["other_minus_full"], 0.)
        aggregate = result["aggregate"][1]
        self.assertEqual(aggregate["runs"], 1)
        self.assertIsNone(aggregate["feasible_fraction"])
        self.assertEqual(aggregate["statuses"], {"stopped_model_error": 1})

    def test_historical_best_is_not_a_new_gain_and_unknown_constraints_not_violations(self):
        from ra_lsbo.records import HIDDEN_MODES
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            risk = {
                "feasible_label": True, "objective_observed": True, "objective": 10.,
                "state": "feasible", "constraints": {"observed": -1., "censored": 999.},
                "constraint_observed": {"observed": True, "censored": False},
                "hidden_labels": dict.fromkeys(HIDDEN_MODES, 0),
            }
            row = {"evaluation": 421, "risk": risk, "phase": "development",
                   "proposal": {"source": "global_coverage"}}
            (folder/"observations.jsonl").write_text(json.dumps(row)+"\n", encoding="utf-8")
            run = {
                "directory": str(folder), "method": "ra_full", "seed": 31, "status": "complete",
                "stop": None, "evaluations": 1, "initial_best": 10.,
                "best_feasible_objective": 10., "feasible_evaluations": 1,
                "ledger": {"charged_calls": 1}, "states": {"feasible": 1}, "wall_seconds": 1.}
            result = trial_metrics(run)
            self.assertEqual(result["improvement"], 0.)
            self.assertIsNone(result["first_improvement_call"])
            self.assertEqual(result["first_new_feasible_call"], 1)
            self.assertEqual(result["explicit_constraint_violations"], {})
            self.assertEqual(result["proposal_sources"], {"global_coverage": 1})


if __name__ == "__main__":
    unittest.main()
