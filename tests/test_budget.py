from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from ra_lsbo.budget import BudgetPolicy, BudgetRejected, RiskLedger, digest
from ra_lsbo.legacy import normalize_evaluation


def outcome(status="ok"):
    return normalize_evaluation({
        "status": status, "objective": 10. if status == "ok" else None,
        "solver_success": status == "ok", "constraints": {"capacity": -1.},
    }, ["capacity"]).to_dict()


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "budget.jsonl"
        self.ledger = RiskLedger(self.path, BudgetPolicy(5, 3., 2, 1, 1))
        self.addCleanup(self.ledger.close)
        self.candidate = {"continuous": {"x": .5}, "discrete": {}}

    def evidence(self, p=.9):
        return {"probability": p, "basis": "empirical_calibrated", "model_id": "model",
                "calibration_id": "calibration", "candidate_key": digest(self.candidate)}

    def test_invalid_policy_and_probability(self):
        with self.assertRaises(ValueError):
            BudgetPolicy(True, 3, 1, 1, 1)
        with self.assertRaises(ValueError):
            BudgetPolicy(3, float("nan"), 1, 1, 1)
        for probability in (True, -1, float("nan")):
            with self.assertRaises(ValueError):
                self.ledger.reserve(self.candidate, "development", self.evidence(probability), "local")

    def test_reserve_start_settle_and_replay(self):
        token = self.ledger.reserve(self.candidate, "development", self.evidence(), "local")
        self.assertEqual(self.ledger.summary()["charged_calls"], 0)
        self.assertEqual(self.ledger.summary()["reserved_calls"], 1)
        self.ledger.start(token)
        self.assertEqual(self.ledger.summary()["charged_calls"], 1)
        self.ledger.settle(token, outcome())
        self.assertAlmostEqual(self.ledger.summary()["risk_charge"], .1)
        self.assertEqual(RiskLedger(self.path).summary(), self.ledger.summary())

    def test_duplicate_dispatch_and_settlement(self):
        token = self.ledger.reserve(self.candidate, "development", self.evidence(), "decoded")
        self.ledger.start(token)
        with self.assertRaises(ValueError):
            self.ledger.start(token)
        self.ledger.settle(token, outcome())
        events = len(self.ledger.events)
        self.ledger.settle(token, outcome())
        self.assertEqual(len(self.ledger.events), events)
        with self.assertRaises(ValueError):
            self.ledger.settle(token, outcome("timeout"))
        with self.assertRaises(BudgetRejected):
            self.ledger.reserve(self.candidate, "development", self.evidence(), "recovery")

    def test_pending_calls_keep_risk_and_cannot_be_refunded(self):
        token = self.ledger.reserve(self.candidate, "discovery",
                                    {"basis": "raw", "probability": .99}, "random")
        self.ledger.start(token)
        with self.assertRaises(ValueError):
            self.ledger.cancel(token)
        recovered = RiskLedger(self.path)
        self.assertEqual(recovered.summary()["unresolved_started_calls"], 1)
        self.assertEqual(recovered.summary()["risk_charge"], 1.)
        with self.assertRaises(RuntimeError):
            recovered.settle(token, outcome())

    def test_cancel_before_dispatch_releases_only_reservation(self):
        token = self.ledger.reserve(self.candidate, "bootstrap", {"basis": "unavailable"}, "initial")
        self.ledger.cancel(token)
        self.assertEqual(self.ledger.summary()["allocated_calls"], 0)
        self.assertEqual(self.ledger.summary()["risk_charge"], 0.)

    def test_unknown_failure_is_charged_not_physical_label(self):
        token = self.ledger.reserve(self.candidate, "bootstrap", {"basis": "unavailable"}, "initial")
        self.ledger.start(token)
        self.ledger.settle(token, outcome("timeout"))
        summary = self.ledger.summary()
        self.assertEqual(summary["unknown_outcomes"], 1)
        self.assertEqual(summary["known_infeasible"], 0)
        self.assertEqual(summary["observed_nonfeasible"], 1)
        self.assertEqual(summary["risk_charge"], 1.)

    def test_authoritative_precheck_rejection_does_not_consume_budget(self):
        token = self.ledger.reserve(self.candidate, "bootstrap", {"basis": "unavailable"}, "initial")
        self.ledger.start(token)
        rejected = normalize_evaluation(
            {"status": "analytical_constraint_rejected", "constraints": {"span": 1.}},
            ["span"], ["span"]).to_dict()
        self.ledger.settle(token, rejected)
        self.assertEqual(self.ledger.summary()["charged_calls"], 0)
        self.assertEqual(self.ledger.summary()["risk_charge"], 0.)
        self.assertEqual(RiskLedger(self.path).summary(), self.ledger.summary())

    def test_development_rejects_raw_prior_and_low_probability(self):
        for evidence in ({"basis": "raw", "probability": .999}, self.evidence(.6)):
            with self.assertRaises(BudgetRejected):
                self.ledger.reserve(self.candidate, "development", evidence, "local")
        self.assertEqual(len(self.ledger.events), 1)

    def test_modified_candidate_invalidates_prediction(self):
        changed = deepcopy(self.candidate)
        changed["continuous"]["x"] = .7
        with self.assertRaisesRegex(ValueError, "final candidate"):
            self.ledger.reserve(changed, "development", self.evidence(), "decoded_repaired")

    def test_boundary_discovery_and_total_limits(self):
        token = self.ledger.reserve(self.candidate, "boundary", self.evidence(.4), "boundary")
        self.ledger.start(token)
        self.ledger.settle(token, outcome())
        self.candidate["continuous"]["x"] = .6
        with self.assertRaisesRegex(BudgetRejected, "boundary_budget"):
            self.ledger.reserve(self.candidate, "boundary", self.evidence(), "coverage")
        token = self.ledger.reserve(self.candidate, "discovery", {"basis": "unavailable"}, "recovery")
        self.ledger.start(token)
        self.ledger.settle(token, outcome())
        self.candidate["continuous"]["x"] = .7
        with self.assertRaisesRegex(BudgetRejected, "discovery_budget"):
            self.ledger.reserve(self.candidate, "discovery", {"basis": "unavailable"}, "random")

    def test_risk_budget_and_baseline_cannot_be_bypassed(self):
        limited = RiskLedger(Path(self.directory.name) / "limited.jsonl", BudgetPolicy(5, .05, 2, 1, 1))
        self.addCleanup(limited.close)
        with self.assertRaisesRegex(BudgetRejected, "risk_budget"):
            limited.reserve(self.candidate, "development", self.evidence(.9), "local")
        with self.assertRaisesRegex(BudgetRejected, "baseline_bypass"):
            self.ledger.reserve(self.candidate, "baseline", {"basis": "unavailable"}, "fallback")

    def test_call_and_observed_failure_limits(self):
        for name, policy, reason in (
                ("call", BudgetPolicy(1, 2, 2, 2, 2), "call_budget"),
                ("failure", BudgetPolicy(5, 5, 2, 2, 2, max_observed_nonfeasible=1), "observed_failure_stop")):
            ledger = RiskLedger(Path(self.directory.name) / (name+".jsonl"), policy)
            self.addCleanup(ledger.close)
            token = ledger.reserve(self.candidate, "bootstrap", {"basis": "unavailable"}, "initial")
            ledger.start(token)
            ledger.settle(token, outcome("timeout"))
            changed = deepcopy(self.candidate)
            changed["continuous"]["x"] = .8
            with self.assertRaisesRegex(BudgetRejected, reason):
                ledger.reserve(changed, "bootstrap", {"basis": "unavailable"}, "initial")

    def test_corruption_detected_and_existing_file_not_overwritten(self):
        with self.assertRaises(FileExistsError):
            RiskLedger(self.path, self.ledger.policy)
        self.ledger.close()
        event = json.loads(self.path.read_text())
        event["payload"]["policy"]["max_calls"] = 500
        self.path.write_text(json.dumps(event)+"\n")
        with self.assertRaisesRegex(ValueError, "hash chain"):
            RiskLedger(self.path)


if __name__ == "__main__":
    unittest.main()
