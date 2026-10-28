from copy import deepcopy
import unittest

import torch

from ra_lsbo.calibration import CalibrationPolicy
from ra_lsbo.features import LatentFeatures
from ra_lsbo.legacy import normalize_evaluation
from ra_lsbo.transfer_validation import candidate_key, gate_diagnostics, transfer_calibration, unseen_rows
from test_calibration import synthetic_history
from test_models import BASE, VARIABLES


class TransferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def run_model(self, target, callback=None):
        return transfer_calibration(
            synthetic_history(120), {"other": target}, {"design_variables": VARIABLES}, BASE,
            train_count=60, calibration_count=30, epochs=1, trees=8,
            policy=CalibrationPolicy(min_samples=5, min_per_class=2),
            on_prediction=callback)

    def target(self):
        rows = deepcopy(synthetic_history(140))
        for row in rows:
            row["source_journal_sha256"] = "different-journal"
        return rows

    def test_inactive_coordinates_do_not_hide_overlap(self):
        features = LatentFeatures(BASE, VARIABLES)
        original = synthetic_history(1)[0]
        changed = deepcopy(original)
        changed["candidate"]["continuous"]["spacing"] = .97
        changed["candidate"]["discrete"]["count"] = 4
        key = candidate_key(features, original["candidate"])
        self.assertEqual(key, candidate_key(features, changed["candidate"]))
        selected, removed = unseen_rows([changed], features, {key})
        self.assertEqual(selected, [])
        self.assertEqual(removed[0]["reason"], "seen_in_fit")

    def test_overlap_removed_and_forecasts_written_without_labels(self):
        written = []
        predictions, metadata, _ = self.run_model(self.target(), written.append)
        self.assertEqual(len(predictions["other"]), 50)
        self.assertTrue(all(p["evaluation"] > 90 for p in predictions["other"]))
        self.assertTrue(all("observed" not in p for p in written))
        self.assertEqual(metadata["cohorts"]["other"]["overlap_with_previous_test_cohorts"], 30)
        self.assertEqual(metadata["unique_test_designs_across_cohorts"], 50)

    def test_target_labels_cannot_change_frozen_model_or_predictions(self):
        original = self.target()
        changed = deepcopy(original)
        for row in changed:
            row["risk"] = normalize_evaluation({
                "status": "constraint_violation", "solver_success": True,
                "objective": 10000., "constraints": {"capacity": 1., "span_min": -1.},
            }, ["capacity", "span_min"], ["span_min"]).to_dict()
        first, first_meta, _ = self.run_model(original)
        second, second_meta, _ = self.run_model(changed)
        self.assertEqual(first_meta["raw_model_id"], second_meta["raw_model_id"])
        self.assertEqual(first_meta["calibration_id"], second_meta["calibration_id"])
        self.assertEqual([p["risk_probabilities_raw"] for p in first["other"]],
                         [p["risk_probabilities_raw"] for p in second["other"]])

    def test_cross_contract_or_copy_rejected_before_training(self):
        rows = self.target()
        for row in rows:
            row["evaluator_contract_sha256"] = "changed"
        with self.assertRaisesRegex(ValueError, "contract"):
            self.run_model(rows)
        with self.assertRaisesRegex(ValueError, "copy"):
            self.run_model(synthetic_history(140))

    def test_within_block_duplicates_scored_once(self):
        features = LatentFeatures(BASE, VARIABLES)
        rows = synthetic_history(1)
        duplicate = deepcopy(rows[0])
        duplicate["evaluation"] = 2
        selected, removed = unseen_rows(rows+[duplicate], features, set())
        self.assertEqual(len(selected), 1)
        self.assertEqual(removed[0]["reason"], "repeated_in_block")

    def test_unfitted_calibration_cannot_pass_gate(self):
        result = gate_diagnostics([{
            "risk_probabilities_calibrated": {"final_feasibility": None},
            "risk_probabilities_effective": {"final_feasibility": .99},
        }])
        self.assertEqual(result["accepted"], 0)
        self.assertIsNone(result["observed_feasible_fraction"])


if __name__ == "__main__":
    unittest.main()
