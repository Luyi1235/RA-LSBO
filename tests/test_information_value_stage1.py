"""Stdlib tests: finite decisions, no hidden labels, audit isolation and billing."""
from contextlib import nullcontext
from copy import deepcopy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from ra_lsbo.experiments.information_value_stage1.audit import fingerprint, snapshot, summarize
from ra_lsbo.experiments.information_value_stage1.io import pack, verify
from ra_lsbo.experiments.information_value_stage1.mechanism import Outcome, fixture, run_mechanisms
from ra_lsbo.experiments.information_value_stage1.runner import ChargedEvaluator, execute, plan


class MechanismTests(unittest.TestCase):
    def test_failed_objective_is_not_observed(self):
        with self.assertRaises(ValueError):
            Outcome(False, 999., ("region>0",))

    def test_short_horizon_selects_deliverable_design(self):
        decision = fixture().choose((.25,) * 4, 1)
        self.assertEqual(decision["action"], "safe")
        self.assertAlmostEqual(decision["value"], 4.)

    def test_two_step_pays_for_failed_but_informative_probe(self):
        decision = fixture().choose((.25,) * 4, 2)
        self.assertEqual(decision["action"], "probe")
        self.assertAlmostEqual(decision["value"], 8.)
        self.assertGreater(decision["scores"]["probe"], decision["scores"]["blind_probe"])

    def test_failure_evidence_ablation_removes_probe_preference(self):
        model = fixture()
        decision = model.choose((.25,) * 4, 2, feedback="without_failure_evidence")
        self.assertNotEqual(decision["action"], "probe")
        self.assertAlmostEqual(decision["value"], 4.5)

    def test_same_failed_event_can_have_different_information(self):
        model = fixture()
        post = model.update((.25,) * 4, "probe", model.worlds[2]["probe"])
        self.assertEqual(post, (0., 0., 1., 0.))
        self.assertEqual(model.choose(post, 1, used=("probe",))["action"], "target_2")
        coarse = model.update((.25,) * 4, "probe", model.worlds[2]["probe"],
                              "without_failure_evidence")
        self.assertEqual(coarse, (.25,) * 4)

    def test_uninformative_negative_control(self):
        self.assertNotEqual(fixture(False).choose((.25,) * 4, 2)["action"], "probe")

    def test_all_calls_are_charged_and_expensive_case_stops(self):
        self.assertIsNone(fixture(cost=7.).choose((.25,) * 4, 2)["action"])

    def test_zero_remaining_is_zero_call_stop(self):
        d = fixture().choose((.25,) * 4, 0, incumbent=5.)
        self.assertIsNone(d["action"])
        self.assertEqual(d["value"], 5.)
        self.assertEqual(d["scores"], {})

    def test_unknown_feedback_and_bad_prior_rejected(self):
        with self.assertRaises(ValueError):
            fixture().choose((.25,) * 4, 2, feedback="hidden_full_objective")
        with self.assertRaises(ValueError):
            fixture().choose((0.,) * 4, 2)

    def test_decision_is_identical_before_world_is_observed(self):
        rows = [r for r in run_mechanisms()["rows"] if r["scenario"] == "informative_abort"
                and r["horizon"] == 2 and r["arm"] == "two_step_full"]
        self.assertEqual(len(rows), 4)
        self.assertEqual(len({fingerprint(r["trace"][0]["decision"]) for r in rows}), 1)
        self.assertTrue(all(r["calls"] == 2 and r["net_utility"] == 8 for r in rows))
        self.assertTrue(all(r["trace"][0]["outcome"][1] is None for r in rows))


class FakeRecord:
    def __init__(self, candidate, known=True):
        self.candidate = candidate
        good = candidate["k"] % 3 == 0
        self.physical_feasible = good if known else None
        self.usable_feasible = good
        self.observation = SimpleNamespace(status="completed" if known else "unknown",
            objective=SimpleNamespace(kind="exact" if good else "missing",
                                      value=float(candidate["k"]) if good else None))
        self.diagnostics = {}

    def to_dict(self):
        return {"candidate": self.candidate, "status": self.observation.status,
                "physical_feasible": self.physical_feasible,
                "objective": self.observation.objective.value}


class FakeAdapter:
    """Integration fixture, explicitly not a physical or GP implementation."""
    def __init__(self, raise_unknown=False):
        self.events = []
        self.raise_unknown = raise_unknown

    def scope(self):
        return nullcontext()

    def problem(self, case):
        owner = self
        class Problem:
            def canonical(self, candidate):
                return candidate
            def evaluate(self, candidate):
                owner.events.append(("evaluate", candidate["k"]))
                return FakeRecord(candidate, known=not owner.raise_unknown)
        return Problem()

    def initial(self, problem, case, seed):
        return [{"k": i} for i in range(12)]

    def pool(self, problem, case, seed, iteration, history):
        return [{"k": 1000 * (iteration + 1) + i} for i in range(128)]

    def extend(self, problem, case, seed, iteration, history, original, size):
        return original + [{"k": 100000 + 1000 * iteration + i} for i in range(size - 128)]

    def engine(self, problem, seed, output):
        owner = self
        class Engine:
            def propose(self, history, pool, iteration):
                owner.events.append(("propose", iteration))
                meta = {"mode": "fixture_gp", "pool": pool, "eligible": [True] * len(pool),
                    "candidate_gp_original_risk": [.2] * len(pool),
                    "candidate_net_utility": [1. - i / len(pool) for i in range(len(pool))],
                    "credibility_decision": {"selected_predicted_failure": .2}}
                return pool[0], meta
            def accept(self, rec):
                owner.events.append(("accept", rec.candidate["k"]))
        return Engine()


class AuditTests(unittest.TestCase):
    def setUp(self):
        self.pool = [{"k": 30}, {"k": 31}, {"k": 32}]
        self.meta = {"mode": "model", "pool": self.pool, "eligible": [True, False, True],
                     "candidate_gp_original_risk": [.1, .2, .7],
                     "candidate_net_utility": [.9, .8, .1],
                     "credibility_decision": {"selected_predicted_failure": .1}}
        self.snap = snapshot("fixture", 1, 1, [FakeRecord({"k": 0})],
            self.pool, self.pool + [{"k": 33}], self.pool[0], self.meta, 50.)
        self.truth = [{"id": fingerprint(c), "feasible": c["k"] % 3 == 0,
            "feasible_objective": float(c["k"]) if c["k"] % 3 == 0 else None}
            for c in self.snap["audit_pool"]]

    def test_opportunity_and_probability_denominators(self):
        result = summarize(self.snap, self.truth)
        self.assertEqual(result["original"]["feasible"], 1)
        self.assertEqual(result["expanded"]["feasible"], 2)
        self.assertEqual(result["eligible"]["candidates"], 2)
        self.assertEqual(result["risk_all_model_candidates"]["n"], 3)
        self.assertEqual(result["risk_actual_selected_with_receipt"]["n"], 1)
        self.assertEqual(result["original"]["best_improvement"], 20.)

    def test_no_forecast_is_not_zero_risk(self):
        s = snapshot("fixture", 1, 1, [], self.pool, self.pool,
                     self.pool[1], {"mode": "scheduled_coverage"}, None)
        r = summarize(s, self.truth[:3])
        self.assertEqual(r["risk_all_model_candidates"]["n"], 0)
        self.assertIsNone(r["risk_actual_selected_with_receipt"]["brier"])
        self.assertIsNone(r["selected_feasible_gap_to_original_best"])
        self.assertIsNone(r["original"]["best_improvement"])

    def test_unknowns_incomplete_or_duplicate_truth_rejected(self):
        for bad in (self.truth[:-1], self.truth + self.truth[:1],
                    [dict(self.truth[0], feasible=None)] + self.truth[1:]):
            with self.assertRaises(ValueError):
                summarize(self.snap, bad)

    def test_failed_objective_cannot_be_imputed(self):
        rows = deepcopy(self.truth)
        rows[1]["feasible_objective"] = 1e6
        with self.assertRaises(ValueError):
            summarize(self.snap, rows)

    def test_prediction_binding_checked(self):
        meta = deepcopy(self.meta)
        meta["candidate_gp_original_risk"] = [.1]
        with self.assertRaises(ValueError):
            snapshot("x", 1, 1, [], self.pool, self.pool, self.pool[0], meta, None)

    def test_extension_cannot_change_original_candidate_prefix(self):
        with self.assertRaises(ValueError):
            snapshot("x", 1, 1, [], self.pool, list(reversed(self.pool)),
                     self.pool[0], self.meta, None)

    def test_no_model_receipt_attributed_to_coverage_override(self):
        meta = deepcopy(self.meta)
        del meta["credibility_decision"]
        s = snapshot("x", 1, 1, [], self.pool, self.pool, self.pool[1], meta, None)
        r = summarize(s, self.truth[:3])
        self.assertEqual(r["risk_all_model_candidates"]["n"], 3)
        self.assertEqual(r["risk_actual_selected_with_receipt"]["n"], 0)


class RunnerTests(unittest.TestCase):
    def test_registered_costs_and_disjoint_seed_blocks(self):
        s, f = plan("smoke"), plan("full")
        self.assertEqual(s["total_mathematical_call_ceiling"], 696)
        self.assertEqual(f["total_mathematical_call_ceiling"], 16672)
        self.assertEqual(f["online_calls"], 96)
        self.assertEqual(f["postrun_audit_calls"], 16384)
        self.assertFalse(set(s["seeds"]) & set(f["seeds"]))

    def test_trace_then_audit_integrity_and_exact_billing(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"evidence"
            api = FakeAdapter()
            report = execute(out, plan("smoke"), api, {"TEST_FIXTURE_NOT_REAL_MODELS": True})
            self.assertEqual(report["actual_total_mathematical_calls"], 696)
            self.assertEqual(report["actual_initial_and_online_calls"], 56)
            self.assertEqual(report["actual_postrun_audit_calls"], 640)
            self.assertEqual(sum(e[0] == "accept" for e in api.events), 8)
            last_accept = max(i for i, event in enumerate(api.events) if event[0] == "accept")
            self.assertTrue(all(e[0] == "evaluate" for e in api.events[last_accept+1:]))
            self.assertEqual(len(api.events[last_accept+1:]), 640)
            self.assertGreater(verify(out)["verified_files"], 10)
            self.assertTrue((out/"trace_seal.json").exists())
            pack(out, Path(directory)/"bundles")
            self.assertTrue((Path(directory)/"bundles"/"bundle_manifest.json").exists())
            with self.assertRaises(FileExistsError):
                execute(out, plan("smoke"), FakeAdapter(), {})
            (out/"report.json").write_text("tampered", encoding="utf-8")
            with self.assertRaises(ValueError):
                verify(out)

    def test_unknown_evaluation_aborts_without_fake_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)/"failed"
            with self.assertRaises(ValueError):
                execute(out, plan("smoke"), FakeAdapter(True), {})
            f = json.loads((out/"failure.json").read_text(encoding="utf-8"))
            self.assertEqual(f["trace_started"], 1)
            self.assertEqual(f["trace_settled"], 0)
            self.assertEqual(f["audit_started"], 0)
            self.assertFalse((out/"report.json").exists())

    def test_exception_is_still_a_started_call(self):
        with tempfile.TemporaryDirectory() as directory:
            evaluator = ChargedEvaluator(Path(directory)/"ledger.jsonl")
            class Broken:
                def evaluate(self, candidate):
                    raise RuntimeError("infrastructure failure")
            with self.assertRaises(RuntimeError):
                evaluator.call(Broken(), {"k": 1}, "initial")
            self.assertEqual((evaluator.started, evaluator.settled), (1, 0))


if __name__ == "__main__":
    unittest.main()
