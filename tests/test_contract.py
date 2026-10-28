from copy import deepcopy
from dataclasses import replace
import json
import unittest

from ra_lsbo import HIDDEN_MODES, normalize_evaluation, training_rows


REQUIRED = ("soc", "stall_speed", "span_min")
ANALYTICAL = ("span_min",)


def completed(**updates):
    raw = dict(objective=100.0, solver_success=True, status="ok",
               constraints=dict.fromkeys(REQUIRED, -0.1),
               metrics={"mda_evaluated": True}, elapsed_seconds=12.0)
    raw.update(updates)
    return raw


def normalize(raw):
    return normalize_evaluation(raw, REQUIRED, ANALYTICAL)


def row(index, raw):
    return {"evaluation": index, "candidate": {"continuous": {"x": 0.5},
                                              "discrete": {"layout": "a"}},
            "risk": normalize(raw).to_dict()}


class ContractTests(unittest.TestCase):
    def test_completed_feasible(self):
        result = normalize(completed())
        self.assertIs(result.feasible_label, True)
        self.assertEqual(result.hidden_success, 1)
        self.assertEqual(result.hidden_labels, dict.fromkeys(HIDDEN_MODES, 0))
        self.assertEqual(result.objective, 100)

    def test_multiple_explicit_violations_keep_objective(self):
        result = normalize(completed(status="constraint_violation",
                                     constraints={"soc": 0.1, "stall_speed": 2, "span_min": -1}))
        self.assertEqual(result.failure_modes, ["constraint:soc", "constraint:stall_speed"])
        self.assertIs(result.feasible_label, False)
        self.assertTrue(result.objective_observed)
        self.assertEqual(result.hidden_success, 1)

    def test_nonconvergence_censors_other_hidden_modes(self):
        result = normalize(completed(status="mission_not_converged", solver_success=False,
                                     objective=None))
        self.assertEqual(result.hidden_labels["mission_solver_nonconvergence"], 1)
        self.assertIsNone(result.hidden_labels["energy_audit_failure"])
        self.assertFalse(any(result.constraint_observed.values()))
        self.assertEqual(result.hidden_success, 0)

    def test_failed_objective_never_becomes_training_energy(self):
        result = normalize(completed(status="mission_not_converged", solver_success=False,
                                     objective=1e9))
        self.assertIsNone(result.objective)
        self.assertIn("discarded_untrusted_objective", result.issues)

    def test_partial_observations_need_explicit_validity(self):
        result = normalize(completed(status="mission_not_converged", solver_success=False,
                                     objective=None, constraint_observed={"soc": True}))
        self.assertTrue(result.constraint_observed["soc"])
        self.assertFalse(result.constraint_observed["stall_speed"])
        self.assertIsNone(result.constraints["stall_speed"])

    def test_intermediate_domain_error_does_not_label_success_as_failure(self):
        raw = completed(metrics={"mission_solver_segments": [{
            "segment": "climb", "converged": True, "native_failure": "Battery depleted",
            "last_domain_error": "Only nonnegative discharge supported",
        }]})
        self.assertEqual(normalize(raw).state, "feasible")

    def test_failed_segment_is_terminal_evidence(self):
        raw = completed(status="evaluation_error:ValueError", solver_success=False, objective=None,
                        metrics={"mission_solver_segments": [{"segment": "descent", "converged": False}]})
        self.assertEqual(normalize(raw).failure_modes, ["mission_solver_nonconvergence"])

    def test_domain_error_status_is_recognized(self):
        raw = completed(status="evaluation_error:DomainSolveError", objective=None, solver_success=False)
        self.assertEqual(normalize(raw).hidden_labels["mission_solver_nonconvergence"], 1)

    def test_energy_audit_failure_has_no_valid_objective(self):
        result = normalize(completed(metrics={"energy_audit_status": "failed"}))
        self.assertIsNone(result.objective)
        self.assertEqual(result.hidden_labels["energy_audit_failure"], 1)
        self.assertIsNone(result.hidden_labels["mission_solver_nonconvergence"])

    def test_explicit_hidden_mode_statuses(self):
        for mode in HIDDEN_MODES:
            with self.subTest(mode=mode):
                result = normalize(completed(status=mode, objective=None, solver_success=False))
                self.assertEqual(result.hidden_labels[mode], 1)

    def test_infrastructure_is_not_a_physical_failure_label(self):
        for status in ("timeout", "worker_no_result", "worker_nonzero_exit", "invalid_worker_response"):
            with self.subTest(status=status):
                result = normalize(completed(status=status, solver_success=False, objective=None))
                self.assertEqual(result.state, "infrastructure_error")
                self.assertIsNone(result.feasible_label)
                self.assertIsNone(result.hidden_success)
                self.assertTrue(result.budget_charged)
                self.assertFalse(any(result.constraint_observed.values()))

    def test_unknown_errors_are_not_classified_from_message_keywords(self):
        raw = completed(status="evaluation_error:ValueError", objective=None, solver_success=False,
                        metrics={"failure_message": "SOC network geometry invalid"})
        result = normalize(raw)
        self.assertEqual(result.state, "unknown")
        self.assertEqual(result.failure_modes, [])
        self.assertIsNone(result.feasible_label)

    def test_precheck_not_hidden_failure_and_not_expensive_budget(self):
        raw = completed(status="analytical_constraint_rejected", objective=None, solver_success=False,
                        constraints={"span_min": 1.0}, metrics={"mda_evaluated": False})
        result = normalize(raw)
        self.assertEqual(result.state, "precheck_rejected")
        self.assertTrue(result.constraint_observed["span_min"])
        self.assertIsNone(result.hidden_success)
        self.assertFalse(result.budget_charged)
        tables = training_rows([row(1, raw)])
        self.assertEqual(tables["final_feasibility"], [])
        self.assertNotIn("span_min", tables["constraints"])

    def test_missing_constraint_is_not_a_negative_failure_observation(self):
        result = normalize(completed(constraints={"soc": -1, "span_min": -1}))
        self.assertEqual(result.state, "incomplete")
        self.assertIsNone(result.feasible_label)
        self.assertTrue(result.objective_observed)

    def test_nonfinite_and_boolean_margins_stay_missing(self):
        for value in (float("nan"), float("inf"), -float("inf"), True, "-1"):
            with self.subTest(value=value):
                result = normalize(completed(constraints={"soc": value, "stall_speed": -1, "span_min": -1}))
                self.assertIsNone(result.constraints["soc"])
                self.assertIsNone(result.feasible_label)
                json.dumps(result.to_dict(), allow_nan=False)

    def test_positive_known_margin_proves_infeasibility_despite_missing_other(self):
        result = normalize(completed(status="constraint_violation", constraints={"soc": 1}))
        self.assertIs(result.feasible_label, False)

    def test_bad_contract_is_rejected(self):
        with self.assertRaises(ValueError):
            normalize_evaluation(completed(), REQUIRED, ("unknown",))
        with self.assertRaises(ValueError):
            normalize_evaluation(completed(), ("soc", "soc"))
        with self.assertRaises(ValueError):
            normalize(completed(constraint_observed={"soc": "true"}))
        with self.assertRaises(ValueError):
            normalize(completed(status="geometry_rejected"))

    def test_input_unchanged(self):
        raw = completed()
        before = deepcopy(raw)
        normalize(raw)
        self.assertEqual(raw, before)

    def test_independent_head_training_masks(self):
        failure = completed(status="mission_not_converged", objective=None, solver_success=False,
                            constraint_observed={"soc": True})
        rows = [row(1, completed()), row(2, failure),
                row(3, completed(status="timeout", solver_success=False, objective=None))]
        tables = training_rows(rows)
        self.assertEqual([r["evaluation"] for r in tables["objective"]], [1])
        self.assertEqual([r["evaluation"] for r in tables["constraints"]["soc"]], [1, 2])
        self.assertEqual([r["evaluation"] for r in tables["constraints"]["stall_speed"]], [1])
        self.assertNotIn("span_min", tables["constraints"])
        self.assertEqual([r["target"] for r in tables["hidden_success"]], [1, 0])
        self.assertEqual([r["target"] for r in tables["hidden_modes"]["energy_audit_failure"]], [0])
        self.assertEqual([r["target"] for r in tables["hidden_modes"]["mission_solver_nonconvergence"]], [0, 1])
        self.assertEqual([r["target"] for r in tables["final_feasibility"]], [1, 0])

    def test_negative_seed_root_is_diagnostic_only(self):
        raw = completed(status="mission_not_converged", objective=None, solver_success=False, metrics={
            "solver_discharge_margin_W": -22.0,
            "solver_discharge_margin_source": "inadmissible_seed_root_not_a_valid_mission"})
        result = normalize(raw)
        self.assertEqual(result.evidence["solver_discharge_margin_W"], -22)
        self.assertNotIn("solver_discharge_margin_W", result.constraints)
        self.assertIsNone(result.objective)

    def test_dataclass_invariants(self):
        result = normalize(completed())
        with self.assertRaises(ValueError):
            replace(result, objective=None)
        with self.assertRaises(ValueError):
            replace(result, hidden_labels={})
        with self.assertRaises(ValueError):
            replace(result, constraints={"soc": -1})
        failed = normalize(completed(status="mission_not_converged", objective=None, solver_success=False))
        with self.assertRaises(ValueError):
            replace(failed, objective=float("nan"))
        with self.assertRaises(ValueError):
            replace(result, hidden_labels=dict.fromkeys(HIDDEN_MODES, 1))
        with self.assertRaises(ValueError):
            replace(result, feasible_label="true")

    def test_training_rejects_invalid_serialized_contract(self):
        sample = row(1, completed())
        sample["risk"]["objective"] = None
        with self.assertRaises(ValueError):
            training_rows([sample])

    def test_terminal_state_cannot_disagree_with_labels(self):
        result = normalize(completed())
        with self.assertRaises(ValueError):
            replace(result, state="unknown")
        with self.assertRaises(ValueError):
            replace(result, state="infeasible", feasible_label=False)
        with self.assertRaises(ValueError):
            replace(result, state="incomplete", feasible_label=None)


if __name__ == "__main__":
    unittest.main()
