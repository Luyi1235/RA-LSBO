from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from ra_lsbo.budget import RiskLedger
from ra_lsbo.mechanism_system import FiveModeProblem, FullSystemLearning, HIDDEN_MODES, SystemSettings
from ra_lsbo.records import RiskEvaluation
from ra_lsbo.system_experiment import policy_for, sample_rows, run_system_trial


BASE = Path(__file__).resolve().parents[2]/"LSBO"/"X57-LSBO-refactored"


class MechanismSystemTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def candidate(self, problem, x, y, distributed=True):
        value = problem.space.default().to_dict()
        value["continuous"].update(x=x, y=y, spacing=.1)
        value["discrete"].update(layout="distributed" if distributed else "single", count=4)
        return problem.canonical(value)

    def test_all_five_terminal_events_have_explicit_serial_censoring(self):
        problem = FiveModeProblem(BASE)
        cases = [(.03, .6), (.5, .03), (.95, .6), (.2, .25), (.5, .95)]
        for i, (x, y) in enumerate(cases):
            risk = problem.evaluate(self.candidate(problem, x, y))
            RiskEvaluation(**risk)
            self.assertEqual(risk["state"], "hidden_failure")
            self.assertIsNone(risk["objective"])
            for j, name in enumerate(HIDDEN_MODES):
                self.assertEqual(risk["hidden_labels"][name], 0 if j < i else 1 if j == i else None)
            self.assertEqual(sum(s["executed"] for s in risk["evidence"]["controlled_stage_trace"]), i+1)

    def test_completed_violation_is_not_a_hidden_failure(self):
        problem = FiveModeProblem(BASE)
        risk = problem.evaluate(self.candidate(problem, .8, .8))
        self.assertEqual(risk["state"], "infeasible")
        self.assertTrue(risk["objective_observed"])
        self.assertTrue(all(value == 0 for value in risk["hidden_labels"].values()))

    def test_mechanism_scores_are_used_and_ablation_has_same_training_prefix(self):
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 120, 72)
        config = SystemSettings(calls=8, failure_cap=4, calibration_window=40, encoder_epochs=3,
                                incremental_epochs=2, gp_steps=2, pool_size=8)
        variables = [asdict(v) for v in problem.space.variables]
        full, meta = FullSystemLearning("ra_full", config, 1).fit(
            rows, variables, BASE, ["envelope", "load"])
        no_mechanism, other = FullSystemLearning("ra_no_mechanisms", config, 1).fit(
            rows, variables, BASE, ["envelope", "load"])
        self.assertEqual(meta["representation_snapshot"], other["representation_snapshot"])
        self.assertEqual(meta["training_indices"], list(range(1, 81)))
        self.assertEqual(meta["calibration_indices"], list(range(81, 121)))
        candidates = [row["candidate"] for row in rows]
        p, q = full[0].predict(candidates), no_mechanism[0].predict(candidates)
        self.assertTrue(full[0].active_mechanisms)
        self.assertEqual(no_mechanism[0].active_mechanisms, ())
        self.assertTrue(np.all(np.asarray(p["final_feasibility"]) <= q["final_feasibility"]))
        self.assertNotEqual(full[0].model_id, no_mechanism[0].model_id)
        from ra_lsbo.constraint_risk import ConstraintRiskHeads
        parent = {"final_feasibility": [.9], "hidden_success": [.8],
                  **{name: [.7] for name in HIDDEN_MODES}}
        with patch.object(ConstraintRiskHeads, "predict", side_effect=lambda _: deepcopy(parent)):
            combined = full[0].predict([candidates[0]])
            ablated = no_mechanism[0].predict([candidates[0]])
        self.assertAlmostEqual(combined["final_feasibility"][0], .3)
        self.assertAlmostEqual(ablated["final_feasibility"][0], .9)
        self.assertGreater(full[0].mechanism_uncertainty(parent)[0], 0.)
        self.assertEqual(no_mechanism[0].mechanism_uncertainty(parent)[0], 0.)

    def test_no_calibration_does_not_train_on_the_withheld_window(self):
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 60, 7)
        config = SystemSettings(calibration_window=30, encoder_epochs=2, gp_steps=1)
        learner = FullSystemLearning("ra_no_calibration", config)
        models, metadata = learner.fit(rows, [asdict(v) for v in problem.space.variables], BASE,
                                       ["envelope", "load"])
        self.assertIsNone(models[2])
        self.assertEqual(metadata["training_indices"], list(range(1, 31)))
        self.assertEqual(metadata["unused_holdout_indices"], list(range(31, 61)))
        self.assertIsNone(metadata["calibration_id"])

    def test_raw_ablation_is_fully_charged_and_not_calibrated(self):
        problem = FiveModeProblem(BASE)
        settings = SystemSettings(calls=2, failure_cap=2)
        candidate = self.candidate(problem, .7, .65)
        with tempfile.TemporaryDirectory() as folder:
            with RiskLedger(Path(folder)/"ledger.jsonl", policy_for("ra_no_calibration", settings)) as ledger:
                result = ledger.assess(candidate, "baseline", {"basis": "raw", "probability": .99}, "test")
                self.assertTrue(result["allowed"])
                self.assertEqual(result["charge"], 1.)

    def test_new_model_refresh_invalidates_old_mechanism_and_calibration(self):
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 65, 90)
        settings = SystemSettings(calibration_window=30, encoder_epochs=2, incremental_epochs=1, gp_steps=1)
        learner = FullSystemLearning("ra_full", settings)
        models, first = learner.fit(rows[:60], [asdict(v) for v in problem.space.variables], BASE,
                                    ["envelope", "load"])
        _, second = learner.fit(rows, [asdict(v) for v in problem.space.variables], BASE,
                                ["envelope", "load"])
        self.assertEqual(second["encoder_update"], "append_only_incremental")
        self.assertNotEqual(first["representation_snapshot"], second["representation_snapshot"])
        with self.assertRaises(RuntimeError):
            models[0].predict([rows[0]["candidate"]])
        with self.assertRaises(RuntimeError):
            models[2].predict([rows[0]["candidate"]])

    def test_trial_preserves_history_and_only_counts_new_calls(self):
        from ra_lsbo.system_experiment import save_new
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 50, 5)
        before = deepcopy(rows)
        config = SystemSettings(calls=2, failure_cap=2, calibration_window=30, encoder_epochs=2,
                                incremental_epochs=1, gp_steps=1, pool_size=8)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_new(folder/"history.json", rows)
            spec = {"kind": "controlled", "base_project": str(BASE), "contract_id": problem.contract_id}
            result = run_system_trial(spec, "ra_full", 5, asdict(config),
                                      folder/"history.json", folder/"trial", folder/"STOP.json")
            self.assertEqual(result["historical_calls"], 50)
            self.assertEqual(result["evaluations"], result["ledger"]["charged_calls"])
            self.assertLessEqual(result["evaluations"], 2)
            self.assertEqual(rows, before)
            self.assertEqual(result["ledger"]["unresolved_started_calls"], 0)
            from tools.audit_system_validation import verify_trial
            audited = verify_trial(folder/"trial", {}, spec)
            self.assertEqual(audited["worker_or_control_outcomes"], result["evaluations"])
            self.assertGreater(audited["checkpoints"], 0)

    def test_auditor_rejects_future_training_indices(self):
        import json
        from ra_lsbo.system_experiment import save_new
        from tools.audit_system_validation import verify_trial
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 50, 5)
        config = SystemSettings(calls=1, failure_cap=1, calibration_window=30,
                                encoder_epochs=2, gp_steps=1, pool_size=8)
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_new(folder/"history.json", rows)
            spec = {"kind": "controlled", "base_project": str(BASE), "contract_id": problem.contract_id}
            run_system_trial(spec, "ra_full", 5, asdict(config), folder/"history.json",
                             folder/"trial", folder/"STOP.json")
            path = folder/"trial/model_fits.jsonl"
            record = json.loads(path.read_text())
            record["training_indices"].append(999)
            path.write_text(json.dumps(record)+"\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Future"):
                verify_trial(folder/"trial", {}, spec)

    def test_original_lsbo_warm_start_does_not_repeat_physical_evaluations(self):
        from ra_lsbo.system_experiment import save_new
        problem = FiveModeProblem(BASE)
        rows = sample_rows(problem, 45, 31)
        config = SystemSettings(calls=1, failure_cap=1, gp_steps=2)
        evaluate = FiveModeProblem.evaluate
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary)
            save_new(folder/"history.json", rows)
            spec = {"kind": "controlled", "base_project": str(BASE), "contract_id": problem.contract_id}
            with patch.object(FiveModeProblem, "evaluate", autospec=True, side_effect=evaluate) as calls:
                result = run_system_trial(spec, "lsbo", 31, asdict(config), folder/"history.json",
                                          folder/"trial", folder/"STOP.json")
            self.assertNotEqual(result["status"], "stopped_model_error", result["stop"])
            self.assertEqual(result["evaluations"], 1)
            self.assertEqual(calls.call_count, 1)
            settings = __import__("json").loads((folder/"trial/original_settings.json").read_text())
            self.assertEqual(settings["branches"], 12)
            self.assertEqual(settings["restarts"], 4)
            self.assertEqual(settings["raw_samples"], 64)


if __name__ == "__main__":
    unittest.main()
