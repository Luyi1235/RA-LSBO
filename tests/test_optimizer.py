from copy import deepcopy
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HAS_MODELS = all(importlib.util.find_spec(name) for name in ("numpy", "torch", "sklearn", "botorch"))
if HAS_MODELS:
    import numpy as np
    import torch
    from ra_lsbo.budget import BudgetPolicy, RiskLedger, digest
    from ra_lsbo.optimizer import OptimizerSettings, dispatch, initial_candidates, run_optimizer
    from ra_lsbo.selection import candidate_pool, incumbent, select_candidate
    from ra_lsbo.synthetic import SyntheticProblem
    from test_models import BASE


@unittest.skipUnless(HAS_MODELS, "Use existing Rcaide_env")
class OptimizerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.problem = SyntheticProblem(BASE)
        self.settings = OptimizerSettings(budget=16, bootstrap=10, calibration_window=4,
                                          pool_size=8, epochs=1, gp_steps=2, update_every=2)

    def proposal(self, candidate):
        return {"candidate": candidate, "mode": "bootstrap", "source": "test",
                "evidence": {"basis": "unavailable"}}

    def test_initialization_is_paired_and_canonical(self):
        a = initial_candidates(self.problem, self.settings)
        b = initial_candidates(self.problem, replace(self.settings, algorithm="cbo"))
        self.assertEqual(a, b)
        self.assertEqual(len({digest(c) for c in a}), len(a))
        self.assertTrue(all(c == self.problem.canonical(c) for c in a))

    def test_sparse_simultaneous_constraints_and_serial_failures(self):
        raw = self.problem.space.default().to_dict()
        raw["continuous"].update(x=.99, y=.99)
        risk = self.problem.evaluate(raw)
        self.assertGreater(risk["constraints"]["envelope"], 0)
        self.assertGreater(risk["constraints"]["load"], 0)
        self.assertTrue(risk["objective_observed"])
        raw["continuous"]["x"] = .01
        failed = self.problem.evaluate(raw)
        self.assertEqual(failed["hidden_labels"]["geometry_generation_failure"], 1)
        self.assertIsNone(failed["hidden_labels"]["mission_solver_nonconvergence"])
        self.assertIsNone(failed["constraints"]["load"])
        self.assertIsNone(incumbent([{"risk": failed}]))

    def test_dispatch_once_precheck_no_charge_and_canonical_change_rejected(self):
        ledger = RiskLedger(self.root / "budget.jsonl", BudgetPolicy(3, 3, 3, 0, 0))
        self.addCleanup(ledger.close)
        raw = self.problem.space.default().to_dict()
        original = self.problem.evaluate
        calls = []
        def evaluate(candidate):
            calls.append(candidate)
            return original(candidate)
        self.problem.evaluate = evaluate
        dispatch(self.problem, ledger, self.proposal(raw))
        self.assertEqual(len(calls), 1)
        bad = deepcopy(raw)
        bad["discrete"].update(layout="distributed", count=4)
        bad["continuous"]["spacing"] = .4
        self.assertIsNone(dispatch(self.problem, ledger, self.proposal(bad)))
        self.assertEqual(ledger.summary()["charged_calls"], 1)
        changed = deepcopy(raw)
        changed["continuous"]["spacing"] = .3
        with self.assertRaisesRegex(ValueError, "rescore"):
            dispatch(self.problem, ledger, self.proposal(changed))

    def test_dispatch_exception_is_unknown_charged_and_recorded(self):
        ledger = RiskLedger(self.root / "budget.jsonl", BudgetPolicy(3, 3, 3, 0, 0))
        self.addCleanup(ledger.close)
        def fail(candidate):
            raise RuntimeError("controlled failure")
        self.problem.evaluate = fail
        risk = dispatch(self.problem, ledger, self.proposal(self.problem.space.default().to_dict()))
        self.assertEqual(risk["state"], "unknown")
        self.assertEqual(ledger.summary()["unknown_outcomes"], 1)
        self.assertEqual(ledger.summary()["charged_calls"], 1)

    def test_interrupted_dispatch_keeps_unresolved_charge(self):
        ledger = RiskLedger(self.root / "interrupted.jsonl", BudgetPolicy(3, 3, 3, 0, 0))
        self.addCleanup(ledger.close)
        def interrupt(candidate):
            raise KeyboardInterrupt()
        self.problem.evaluate = interrupt
        with self.assertRaises(KeyboardInterrupt):
            dispatch(self.problem, ledger, self.proposal(self.problem.space.default().to_dict()))
        self.assertEqual(ledger.summary()["charged_calls"], 1)
        self.assertEqual(ledger.summary()["unresolved_started_calls"], 1)
        self.assertEqual(RiskLedger(ledger.path).summary(), ledger.summary())

    def test_final_pool_has_no_duplicate_or_cheap_failure(self):
        pool, _ = candidate_pool(self.problem.space, np.random.default_rng(0), 50, set(), [], self.problem.precheck)
        self.assertEqual(len(pool), 50)
        self.assertEqual(len({digest(item["candidate"]) for item in pool}), 50)
        self.assertTrue(all(max(self.problem.precheck(item["candidate"]).values()) <= 0 for item in pool))

    def test_loop_budget_replay_and_no_future_training(self):
        rows, summary = run_optimizer(self.problem, BASE, self.settings, self.root / "run")
        self.assertEqual(summary["evaluations"], 16)
        self.assertEqual(summary["ledger"]["charged_calls"], 16)
        fits = [json.loads(line) for line in (self.root / "run" / "model_fits.jsonl").read_text().splitlines()]
        for fit in fits:
            self.assertLess(max(fit["training_indices"]), min(fit["calibration_indices"]))
            self.assertLess(max(fit["calibration_indices"]), fit["before_evaluation"])
        self.assertTrue(all(row["phase"] in ("bootstrap", "discovery") for row in rows))
        self.assertEqual(RiskLedger(self.root / "run" / "budget.jsonl").summary(), summary["ledger"])
        proposals = [json.loads(line) for line in (self.root / "run" / "proposals.jsonl").read_text().splitlines()]
        self.assertTrue(all("risk" not in row for row in proposals))

    def test_no_feasible_initialization_and_discovery_cap(self):
        problem = SyntheticProblem(BASE, "no_initial_feasible")
        settings = replace(self.settings, budget=24)
        rows, summary = run_optimizer(problem, BASE, settings, self.root / "run")
        self.assertTrue(all(row["risk"]["feasible_label"] is not True for row in rows[:10]))
        self.assertEqual(summary["status"], "stopped_by_policy")
        self.assertEqual(summary["evaluations"], 18)
        self.assertEqual(summary["ledger"]["mode_calls"]["discovery"], 8)

    def test_initialization_cannot_bypass_risk_limit(self):
        policy = BudgetPolicy(16, 2, 10, 8, 8)
        rows, summary = run_optimizer(self.problem, BASE, self.settings, self.root / "run", policy)
        self.assertEqual(len(rows), 2)
        self.assertEqual(summary["stop"]["reason"], "risk_budget")

    def test_cbo_and_random_use_explicit_baseline_policy(self):
        for algorithm in ("cbo", "random"):
            _, summary = run_optimizer(self.problem, BASE, replace(self.settings, algorithm=algorithm),
                                       self.root / algorithm)
            self.assertEqual(summary["evaluations"], 16)
            self.assertEqual(summary["ledger"]["risk_charge"], 16)

    def test_random_baseline_does_not_use_elite_neighborhood(self):
        rows, _ = run_optimizer(self.problem, BASE, replace(self.settings, algorithm="random"),
                                self.root / "random")
        self.assertTrue(all(row["proposal"]["source"] == "global_coverage"
                            for row in rows[self.settings.bootstrap:]))

    def test_boundary_schedule_and_development_threshold_use_final_candidates(self):
        pool = [{"candidate": {"continuous": {"x": x}, "discrete": {}}, "source": "repaired"}
                for x in (.1, .2, .3)]
        probability = [.9, .5, .95]
        risks = SimpleNamespace(
            predict=lambda candidates: {"final_feasibility": probability, "hidden_success": probability},
            heads={"final_feasibility": SimpleNamespace(model=None)}, model_id="test_model")
        responses = SimpleNamespace(predict=lambda candidates: {
            "objective": {"mean": [1., 1., 1.], "std": [1., 1., 1.]},
            "constraint:g": {"p_satisfied_raw": [0., .5, .1]}})
        calibrated = SimpleNamespace(
            predict=lambda candidates: {"risk_probabilities_calibrated": {"final_feasibility": probability}},
            calibration_id="calibration")
        rows = [{"risk": {"feasible_label": True, "objective": 2.}}]
        ledger = RiskLedger(self.root / "select.jsonl", BudgetPolicy(5, 5, 0, 2, 1))
        self.addCleanup(ledger.close)
        with patch("ra_lsbo.selection.log_expected_improvement", return_value=np.log([2., 100., 1.])):
            proposal, _ = select_candidate(pool, rows, risks, responses, calibrated, ledger, 0)
            self.assertEqual(proposal["mode"], "boundary")
            self.assertEqual(proposal["candidate"], pool[1]["candidate"])
            self.assertEqual(proposal["evidence"]["candidate_key"], digest(pool[1]["candidate"]))
            proposal, _ = select_candidate(pool, rows, risks, responses, calibrated, ledger, 1)
            self.assertEqual(proposal["mode"], "development")
            self.assertEqual(proposal["candidate"], pool[0]["candidate"])
            proposal, _ = select_candidate(pool, [], risks, responses, calibrated, ledger, 1)
            self.assertEqual(proposal["mode"], "discovery")
            proposal, _ = select_candidate(pool, rows, risks, responses, None, ledger, 1)
            self.assertEqual(proposal["mode"], "discovery")
        with patch("ra_lsbo.selection.log_expected_improvement", return_value=np.array([-1000., -2000., -900.])):
            proposal, info = select_candidate(pool, rows, risks, responses, calibrated, ledger, 1)
            self.assertEqual(proposal["candidate"], pool[2]["candidate"])
            self.assertEqual(info["ei"], 0.)
            self.assertEqual(info["log_ei"], -900.)


if __name__ == "__main__":
    unittest.main()
